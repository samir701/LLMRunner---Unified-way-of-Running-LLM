"""Host coordinator: peer discovery (outbound + inbound), handshake, assignment, and health checks."""

from __future__ import annotations

import hmac
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from src.common.config import HostConfig, PeerTarget
from src.common.messages import (
    PROTOCOL_VERSION,
    MessageType,
    assign_msg,
    error_msg,
    hello_msg,
    ping_msg,
    stop_msg,
)
from src.common.transport import (
    Listener,
    TCPConnection,
    TransportError,
    TransportTimeoutError,
    connect,
    get_local_hostname,
    listen,
)
from src.model.loader import (
    LoadedShard,
    get_free_memory_bytes,
    get_num_layers,
    load_model_shard,
    read_model_config,
    resolve_device,
)
from src.model.shard import LayerRange, NodeMemoryInfo, compute_shard_assignments

logger = logging.getLogger(__name__)


@dataclass
class ConnectedPeer:
    """Represents an active peer in the pipeline."""

    target: PeerTarget
    conn: TCPConnection
    device_type: str
    free_memory_bytes: int
    fixed_layers: tuple[int, int] | None
    layer_range: LayerRange | None = None
    is_last: bool = False


@dataclass
class PeerStatus:
    """Status snapshot returned by `llmrun status`."""

    name: str
    address: str
    port: int
    reachable: bool
    device_type: str = "-"
    free_memory_bytes: int = 0
    error: str = ""


class Coordinator:
    """Manages peer connections (LAN outbound + WAN inbound), shard assignment, and heartbeats."""

    def __init__(self, config: HostConfig) -> None:
        self.config = config
        self.host_name = get_local_hostname()
        self.peers: list[ConnectedPeer] = []
        self.host_range: LayerRange | None = None
        self.host_shard: LoadedShard | None = None
        self.total_layers: int = 0
        self.use_relay: bool = False

        self._pipeline_lock = threading.Lock()
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._heartbeat_error: Exception | None = None

    def _verify_auth(self, token: str | None) -> bool:
        if self.config.auth_token is None:
            return True
        if token is None:
            return False
        return hmac.compare_digest(self.config.auth_token, str(token))

    def probe_status(self) -> list[PeerStatus]:
        """Check outbound peers directly and report inbound peers' expected host endpoint."""
        results: list[PeerStatus] = []
        for target in self.config.peers:
            if target.mode == "inbound":
                results.append(
                    PeerStatus(
                        name=target.name,
                        address=f"inbound -> :{self.config.listen_port}",
                        port=self.config.listen_port,
                        reachable=True,
                        device_type="pending",
                        free_memory_bytes=0,
                        error="Waits for peer to connect at startup",
                    )
                )
                continue

            assert target.address is not None and target.port is not None
            try:
                conn = connect(
                    target.address,
                    target.port,
                    timeout=5.0,
                    label=target.name,
                )
                try:
                    conn.settimeout(5.0)
                    conn.send_message(
                        hello_msg(
                            self.host_name,
                            PROTOCOL_VERSION,
                            auth_token=self.config.auth_token,
                        )
                    )
                    reply = conn.recv_message()
                    if reply.msg_type == MessageType.ERROR:
                        raise TransportError(str(reply.header.get("detail", "Auth/handshake error")))
                    if reply.msg_type != MessageType.INFO:
                        raise TransportError(f"Expected INFO, got {reply.msg_type.value}")
                    results.append(
                        PeerStatus(
                            name=target.name,
                            address=target.address,
                            port=target.port,
                            reachable=True,
                            device_type=str(reply.header.get("device_type", "unknown")),
                            free_memory_bytes=int(reply.header.get("free_memory_bytes", 0)),
                        )
                    )
                finally:
                    conn.close()
            except Exception as exc:
                results.append(
                    PeerStatus(
                        name=target.name,
                        address=target.address,
                        port=target.port,
                        reachable=False,
                        error=str(exc),
                    )
                )
        return results

    def _connect_outbound_peer(self, target: PeerTarget) -> ConnectedPeer:
        assert target.address is not None and target.port is not None
        conn = connect(target.address, target.port, timeout=15.0, label=target.name)
        try:
            conn.settimeout(30.0)
            conn.send_message(
                hello_msg(
                    self.host_name,
                    PROTOCOL_VERSION,
                    auth_token=self.config.auth_token,
                )
            )
            reply = conn.recv_message()
            if reply.msg_type == MessageType.ERROR:
                raise TransportError(str(reply.header.get("detail", "Rejected by peer")))
            if reply.msg_type != MessageType.INFO:
                raise TransportError(
                    f"Peer '{target.name}' sent {reply.msg_type.value} instead of INFO."
                )
            if not reply.header.get("ready", True):
                raise TransportError(f"Peer '{target.name}' reported ready=False.")
            raw_fixed = reply.header.get("fixed_layers")
            fixed = (int(raw_fixed[0]), int(raw_fixed[1])) if raw_fixed else None
            return ConnectedPeer(
                target=target,
                conn=conn,
                device_type=str(reply.header.get("device_type", "cpu")),
                free_memory_bytes=int(reply.header.get("free_memory_bytes", 0)),
                fixed_layers=fixed,
            )
        except Exception:
            conn.close()
            raise

    def _accept_inbound_peers(
        self,
        inbound_targets: list[PeerTarget],
    ) -> dict[str, ConnectedPeer]:
        """Listen on `config.listen_port` until all configured inbound peers register."""
        expected = {t.name: t for t in inbound_targets}
        registered: dict[str, ConnectedPeer] = {}
        listener: Listener = listen("0.0.0.0", self.config.listen_port)
        deadline = time.monotonic() + self.config.peer_wait_timeout

        logger.info(
            "Listening on 0.0.0.0:%d for %d inbound peer(s): %s",
            self.config.listen_port,
            len(expected),
            list(expected.keys()),
        )

        try:
            while len(registered) < len(expected):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    missing = [n for n in expected if n not in registered]
                    raise RuntimeError(
                        f"Timed out waiting for inbound peer(s): {', '.join(missing)}"
                    )
                try:
                    conn, addr = listener.accept(timeout=min(1.0, remaining))
                except TransportTimeoutError:
                    continue

                try:
                    conn.settimeout(15.0)
                    reg = conn.recv_message()
                    if reg.msg_type != MessageType.REGISTER:
                        conn.send_message(
                            error_msg(self.host_name, f"Expected REGISTER, got {reg.msg_type.value}")
                        )
                        conn.close()
                        continue

                    if not self._verify_auth(reg.header.get("auth_token")):
                        conn.send_message(error_msg(self.host_name, "Invalid auth_token"))
                        conn.close()
                        logger.warning("Rejected inbound connection from %s:%d (bad auth_token)", addr[0], addr[1])
                        continue

                    peer_name = str(reg.header.get("peer_name", ""))
                    if peer_name not in expected:
                        conn.send_message(
                            error_msg(self.host_name, f"Unknown peer '{peer_name}' not in host.yaml")
                        )
                        conn.close()
                        continue

                    if peer_name in registered:
                        conn.send_message(
                            error_msg(self.host_name, f"Peer '{peer_name}' already connected")
                        )
                        conn.close()
                        continue

                    conn.peer_label = peer_name
                    conn.send_message(
                        hello_msg(
                            self.host_name,
                            PROTOCOL_VERSION,
                            auth_token=self.config.auth_token,
                        )
                    )
                    reply = conn.recv_message()
                    if reply.msg_type != MessageType.INFO:
                        raise TransportError(
                            f"Inbound peer '{peer_name}' sent {reply.msg_type.value} instead of INFO."
                        )
                    raw_fixed = reply.header.get("fixed_layers")
                    fixed = (int(raw_fixed[0]), int(raw_fixed[1])) if raw_fixed else None
                    registered[peer_name] = ConnectedPeer(
                        target=expected[peer_name],
                        conn=conn,
                        device_type=str(reply.header.get("device_type", "cpu")),
                        free_memory_bytes=int(reply.header.get("free_memory_bytes", 0)),
                        fixed_layers=fixed,
                    )
                    logger.info(
                        "Inbound peer '%s' registered from %s:%d (%s)",
                        peer_name,
                        addr[0],
                        addr[1],
                        registered[peer_name].device_type,
                    )
                except (TransportError, ValueError) as exc:
                    logger.warning("Ignoring invalid inbound connection from %s:%d: %s", addr[0], addr[1], exc)
                    conn.close()
                    continue
                except Exception:
                    conn.close()
                    raise
        except Exception:
            for p in registered.values():
                p.conn.close()
            raise
        finally:
            listener.close()

        return registered

    def startup(self) -> None:
        """Discover all outbound and inbound peers, assign layer ranges, and load shards."""
        model_cfg = read_model_config(self.config.model)
        self.total_layers = get_num_layers(model_cfg)

        host_device = resolve_device("auto")
        host_free_mem = get_free_memory_bytes(host_device)
        logger.info(
            "Host '%s' (%s) free memory: %.2f GiB, model layers: %d",
            self.host_name,
            host_device,
            host_free_mem / (1024**3),
            self.total_layers,
        )

        inbound_targets = [t for t in self.config.peers if t.mode == "inbound"]
        outbound_targets = [t for t in self.config.peers if t.mode == "outbound"]

        inbound_map: dict[str, ConnectedPeer] = {}
        outbound_map: dict[str, ConnectedPeer] = {}

        try:
            if inbound_targets:
                inbound_map = self._accept_inbound_peers(inbound_targets)

            for target in outbound_targets:
                try:
                    outbound_map[target.name] = self._connect_outbound_peer(target)
                except Exception as exc:
                    raise RuntimeError(
                        f"Peer '{target.name}' ({target.address}:{target.port}) is unreachable "
                        f"or failed handshake: {exc}"
                    ) from exc
        except Exception:
            for p in (*inbound_map.values(), *outbound_map.values()):
                p.conn.close()
            raise

        # Preserve the pipeline order declared in host.yaml.
        self.peers = [
            inbound_map[t.name] if t.mode == "inbound" else outbound_map[t.name]
            for t in self.config.peers
        ]

        if self.config.routing == "host_relay":
            self.use_relay = True
        elif self.config.routing == "direct":
            if inbound_targets:
                raise ValueError(
                    "routing='direct' cannot be used with mode='inbound' peers behind NAT; "
                    "use routing='auto' or routing='host_relay'."
                )
            self.use_relay = False
        else:
            self.use_relay = len(inbound_targets) > 0

        # Compute contiguous shard assignments across [host, peer_1, ..., peer_n].
        nodes = [NodeMemoryInfo(name=self.host_name, free_memory_bytes=host_free_mem)]
        for p in self.peers:
            nodes.append(
                NodeMemoryInfo(
                    name=p.target.name,
                    free_memory_bytes=p.free_memory_bytes,
                    fixed_layers=p.fixed_layers,
                )
            )
        ranges = compute_shard_assignments(self.total_layers, nodes)
        self.host_range = ranges[0]

        # Send ASSIGN to each peer.
        for idx, peer in enumerate(self.peers):
            peer_range = ranges[idx + 1]
            is_last = idx == len(self.peers) - 1
            peer.layer_range = peer_range
            peer.is_last = is_last
            next_addr = None if (is_last or self.use_relay) else self.peers[idx + 1].target.address
            next_port = None if (is_last or self.use_relay) else self.peers[idx + 1].target.port
            peer.conn.send_message(
                assign_msg(
                    model_name=self.config.model,
                    layer_start=peer_range.start,
                    layer_end=peer_range.end,
                    is_last=is_last,
                    next_address=next_addr,
                    next_port=next_port,
                    relay_via_host=self.use_relay,
                )
            )

        # Load the host's own first shard while peers load theirs.
        self.host_shard = load_model_shard(
            model_dir=self.config.model,
            layer_range=self.host_range,
            is_first=True,
            is_last=False,
            device_pref="auto",
        )

        # Wait for READY from every peer.
        for peer in self.peers:
            peer.conn.settimeout(300.0)
            reply = peer.conn.recv_message()
            if reply.msg_type == MessageType.ERROR:
                detail = reply.header.get("detail", "unknown error")
                raise RuntimeError(f"Peer '{peer.target.name}' failed loading shard: {detail}")
            if reply.msg_type != MessageType.READY:
                raise RuntimeError(
                    f"Peer '{peer.target.name}' sent {reply.msg_type.value} instead of READY."
                )
            peer.conn.settimeout(120.0)
            logger.info("Peer '%s' READY with layers %s", peer.target.name, peer.layer_range)

        self.start_heartbeat()

    @property
    def first_peer(self) -> ConnectedPeer:
        return self.peers[0]

    @property
    def last_peer(self) -> ConnectedPeer:
        return self.peers[-1]

    @contextmanager
    def pipeline_io(self) -> Iterator[None]:
        """Serialize active pipeline I/O with periodic heartbeat checks."""
        with self._pipeline_lock:
            self.raise_if_unhealthy()
            yield

    def check_health(self) -> None:
        """Send PING to every peer and verify PONG; raises RuntimeError on failure."""
        ts = time.time()
        for peer in self.peers:
            try:
                peer.conn.send_message(ping_msg(ts))
                reply = peer.conn.recv_message()
                if reply.msg_type != MessageType.PONG:
                    raise TransportError(
                        f"Expected PONG from '{peer.target.name}', got {reply.msg_type.value}"
                    )
            except Exception as exc:
                raise RuntimeError(
                    f"Health check failed for peer '{peer.target.name}': {exc}"
                ) from exc

    def start_heartbeat(self, interval: float = 5.0) -> None:
        """Start periodic background PING/PONG heartbeat checks."""
        self.stop_heartbeat()
        self._heartbeat_error = None
        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            args=(interval,),
            daemon=True,
        )
        self._heartbeat_thread.start()

    def stop_heartbeat(self) -> None:
        """Stop the periodic heartbeat thread."""
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=2.0)
            self._heartbeat_thread = None

    def raise_if_unhealthy(self) -> None:
        """Raise the recorded heartbeat error if a peer missed a periodic PING/PONG."""
        if self._heartbeat_error is not None:
            raise self._heartbeat_error

    def _heartbeat_loop(self, interval: float) -> None:
        while not self._heartbeat_stop.wait(interval):
            if not self._pipeline_lock.acquire(blocking=False):
                continue
            try:
                self.check_health()
            except Exception as exc:
                self._heartbeat_error = exc
                logger.error("Periodic heartbeat failed: %s", exc)
                self._heartbeat_stop.set()
            finally:
                self._pipeline_lock.release()

    def shutdown(self, reason: str = "session_end") -> None:
        """Stop heartbeat, send STOP to all peers, and close persistent TCP connections."""
        self.stop_heartbeat()
        for peer in self.peers:
            try:
                peer.conn.send_message(stop_msg(reason=reason))
            except Exception:
                pass
            finally:
                peer.conn.close()
        self.peers.clear()
