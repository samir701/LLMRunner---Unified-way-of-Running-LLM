"""Peer TCP server/client: supports listening on LAN or connecting outbound to a global host."""

from __future__ import annotations

import hmac
import logging
import threading
import time

from rich.console import Console

from src.common.config import PeerConfig
from src.common.messages import (
    PROTOCOL_VERSION,
    MessageType,
    error_msg,
    info_msg,
    pong_msg,
    ready_msg,
    register_msg,
)
from src.common.transport import (
    ConnectionClosedError,
    Listener,
    TCPConnection,
    TransportError,
    TransportTimeoutError,
    connect,
    listen,
)
from src.model.engine import ShardEngine
from src.model.loader import get_free_memory_bytes, load_model_shard, resolve_device
from src.model.shard import LayerRange
from src.peer.worker import ShardWorker

logger = logging.getLogger(__name__)


def _connect_with_retry(
    address: str,
    port: int,
    label: str,
    attempts: int = 10,
    delay: float = 0.5,
) -> TCPConnection:
    last_exc: Exception | None = None
    for _ in range(attempts):
        try:
            return connect(address, port, timeout=10.0, label=label)
        except TransportError as exc:
            last_exc = exc
            time.sleep(delay)
    raise TransportError(f"Could not connect to {label}: {last_exc}")


def _verify_auth(expected: str | None, received: str | None) -> bool:
    if expected is None:
        return True
    if received is None:
        return False
    return hmac.compare_digest(expected, str(received))


def _serve_upstream_activations(
    listener: Listener,
    worker: ShardWorker,
    stop_event: threading.Event,
) -> None:
    """Accept connection from previous peer (direct mode) and process ACTIVATE messages."""
    upstream_conn: TCPConnection | None = None
    try:
        while not stop_event.is_set():
            try:
                upstream_conn, addr = listener.accept(timeout=1.0)
                logger.info("Accepted upstream peer connection from %s:%d", addr[0], addr[1])
                break
            except TransportTimeoutError:
                continue

        if upstream_conn is None:
            return

        upstream_conn.settimeout(2.0)
        while not stop_event.is_set():
            try:
                msg = upstream_conn.recv_message()
            except TransportTimeoutError:
                continue
            if msg.msg_type == MessageType.ACTIVATE:
                worker.process_activate(msg)
            elif msg.msg_type == MessageType.STOP:
                break
    except ConnectionClosedError:
        logger.info("Upstream peer closed connection.")
    except Exception:
        if not stop_event.is_set():
            logger.exception("Error in upstream activation listener")
    finally:
        if upstream_conn is not None:
            upstream_conn.close()


def _run_session(
    config: PeerConfig,
    device,
    host_conn: TCPConnection,
    listener: Listener | None,
    console: Console,
) -> None:
    """Run one host session over an established `host_conn`."""
    next_conn: TCPConnection | None = None
    worker: ShardWorker | None = None
    stop_event = threading.Event()
    upstream_thread: threading.Thread | None = None

    try:
        host_conn.settimeout(30.0)

        # 1. HELLO -> INFO
        hello = host_conn.recv_message()
        if hello.msg_type == MessageType.ERROR:
            raise TransportError(str(hello.header.get("detail", "Rejected by host")))
        if hello.msg_type != MessageType.HELLO:
            raise TransportError(f"Expected HELLO, got {hello.msg_type.value}")

        if not _verify_auth(config.auth_token, hello.header.get("auth_token")):
            host_conn.send_message(error_msg(config.name, "Invalid auth_token"))
            raise TransportError("Host failed auth_token verification.")

        free_mem = get_free_memory_bytes(device)
        host_conn.send_message(
            info_msg(
                peer_name=config.name,
                free_memory_bytes=free_mem,
                device_type=device.type,
                ready=True,
                fixed_layers=config.layers,
            )
        )

        # If this was only `llmrun status`, host closes after INFO.
        try:
            assign = host_conn.recv_message()
        except ConnectionClosedError:
            logger.info("Host disconnected after INFO (status probe).")
            return

        if assign.msg_type != MessageType.ASSIGN:
            raise TransportError(f"Expected ASSIGN, got {assign.msg_type.value}")

        layer_start = int(assign.header["layer_start"])
        layer_end = int(assign.header["layer_end"])
        is_last = bool(assign.header["is_last"])
        relay_via_host = bool(assign.header.get("relay_via_host", False))
        next_address = assign.header.get("next_address")
        next_port = assign.header.get("next_port")
        layer_range = LayerRange(layer_start, layer_end)

        console.print(
            f"Assigned layers [green]{layer_range}[/green] "
            f"({'last' if is_last else 'middle'}, "
            f"{'host-relay' if relay_via_host else 'direct'}). Loading shard..."
        )

        # 2. Load only assigned shard.
        try:
            shard = load_model_shard(
                model_dir=config.model,
                layer_range=layer_range,
                is_first=False,
                is_last=is_last,
                device_pref=config.device,
            )
        except Exception as exc:
            host_conn.send_message(error_msg(config.name, str(exc)))
            raise

        if not is_last and not relay_via_host and next_address and next_port:
            next_conn = _connect_with_retry(
                str(next_address),
                int(next_port),
                label=f"{next_address}:{next_port}",
            )

        engine = ShardEngine(shard)
        worker = ShardWorker(
            peer_name=config.name,
            engine=engine,
            is_last=is_last,
            host_conn=host_conn,
            next_conn=next_conn,
            relay_via_host=relay_via_host,
        )

        # Start upstream listener thread only in direct mode when listening.
        if listener is not None and not relay_via_host:
            upstream_thread = threading.Thread(
                target=_serve_upstream_activations,
                args=(listener, worker, stop_event),
                daemon=True,
            )
            upstream_thread.start()

        host_conn.send_message(ready_msg(config.name, layer_start, layer_end))
        console.print(f"[bold green]READY[/bold green] (layers {layer_range}).")

        # 3. Serve control messages (PING/STOP) and ACTIVATE.
        host_conn.settimeout(None)
        while True:
            msg = host_conn.recv_message()
            if msg.msg_type == MessageType.ACTIVATE:
                worker.process_activate(msg)
            elif msg.msg_type == MessageType.PING:
                host_conn.send_message(
                    pong_msg(config.name, float(msg.header.get("ts", 0.0)))
                )
            elif msg.msg_type == MessageType.STOP:
                console.print("[dim]Received STOP. Clearing caches...[/dim]")
                break
            else:
                logger.warning("Ignoring unexpected message: %s", msg.msg_type.value)
    finally:
        stop_event.set()
        if upstream_thread is not None:
            upstream_thread.join(timeout=2.0)
        if worker is not None:
            worker.clear_caches()
        if next_conn is not None:
            next_conn.close()
        host_conn.close()


def run_peer_server(config: PeerConfig) -> None:
    """Run peer in either outbound-to-host mode (for NAT/Kaggle/WAN) or LAN listen mode."""
    console = Console()
    device = resolve_device(config.device)

    if config.host_address and config.host_port:
        console.print(
            f"[bold cyan]Peer '{config.name}'[/bold cyan] connecting to host "
            f"[green]{config.host_address}:{config.host_port}[/green] (device: {device})"
        )
        try:
            while True:
                try:
                    host_conn = connect(
                        config.host_address,
                        config.host_port,
                        timeout=15.0,
                        label=f"host({config.host_address}:{config.host_port})",
                        transport=config.host_transport,
                    )
                    host_conn.send_message(
                        register_msg(
                            peer_name=config.name,
                            protocol_version=PROTOCOL_VERSION,
                            auth_token=config.auth_token,
                        )
                    )
                    _run_session(config, device, host_conn, listener=None, console=console)
                except ConnectionClosedError:
                    console.print("[yellow]Host disconnected. Reconnecting in 3s...[/yellow]")
                except Exception as exc:
                    console.print(f"[yellow]Waiting for host ({exc}). Retrying in 3s...[/yellow]")
                time.sleep(3.0)
        except KeyboardInterrupt:
            console.print("\n[yellow]Peer stopped by user.[/yellow]")
        return

    # Classic LAN listen mode.
    listener = listen("0.0.0.0", config.listen_port)
    console.print(
        f"[bold cyan]Peer '{config.name}'[/bold cyan] listening on port "
        f"[green]{config.listen_port}[/green] (device: {device})"
    )

    try:
        while True:
            try:
                while True:
                    try:
                        host_conn, addr = listener.accept(timeout=1.0)
                        break
                    except TransportTimeoutError:
                        continue

                logger.info("Host connected from %s:%d", addr[0], addr[1])
                _run_session(config, device, host_conn, listener=listener, console=console)
            except ConnectionClosedError:
                console.print("[yellow]Host disconnected. Returning to listening state...[/yellow]")
            except Exception as exc:
                console.print(f"[red]Session error: {exc}[/red]")
                logger.exception("Peer session error")
    except KeyboardInterrupt:
        console.print("\n[yellow]Peer stopped by user.[/yellow]")
    finally:
        listener.close()
