"""TCP and WebSocket (WSS over port 443) transport layer for Local LLM Runner.

Only this module knows about sockets and TLS.
- Raw TCP: 4-byte big-endian length prefix + encoded Message frame.
- WebSocket (RFC 6455): binary frames carrying the exact same 4-byte length-prefixed
  Message frame, allowing Kaggle / NAT peers to connect over port 443 via
  Cloudflare Tunnel (`cloudflared tunnel --url http://localhost:5000`).
"""

from __future__ import annotations

import base64
import hashlib
import os
import socket
import ssl
import struct
import threading

from src.common.messages import Message

_FRAME_LEN_STRUCT = struct.Struct("!I")
_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


class TransportError(Exception):
    """Base error for network transport failures."""


class TransportTimeoutError(TransportError):
    """Raised when a socket accept or receive operation times out."""


class ConnectionClosedError(TransportError):
    """Raised when the remote peer closes the connection."""


def _ws_accept_key(client_key: str) -> str:
    digest = hashlib.sha1((client_key.strip() + _WS_GUID).encode("ascii")).digest()
    return base64.b64encode(digest).decode("ascii")


def _ws_encode_frame(payload: bytes, opcode: int = 0x2, mask: bool = False) -> bytes:
    """Encode one RFC 6455 WebSocket frame."""
    header = bytearray([0x80 | (opcode & 0x0F)])
    length = len(payload)
    mask_bit = 0x80 if mask else 0x00

    if length < 126:
        header.append(mask_bit | length)
    elif length < 65536:
        header.append(mask_bit | 126)
        header.extend(struct.pack("!H", length))
    else:
        header.append(mask_bit | 127)
        header.extend(struct.pack("!Q", length))

    if not mask:
        return bytes(header) + payload

    mask_key = os.urandom(4)
    header.extend(mask_key)
    masked = bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload))
    return bytes(header) + masked


class TCPConnection:
    """Thread-safe wrapper around a persistent TCP or WebSocket-upgraded socket."""

    def __init__(
        self,
        sock: socket.socket,
        peer_label: str = "",
        is_websocket: bool = False,
        ws_client: bool = False,
    ) -> None:
        self._sock = sock
        self.peer_label = peer_label
        self._is_ws = is_websocket
        self._ws_client = ws_client
        self._checked_upgrade = is_websocket
        self._send_lock = threading.Lock()
        self._recv_lock = threading.Lock()
        try:
            self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass

    def settimeout(self, timeout: float | None) -> None:
        self._sock.settimeout(timeout)

    def send_message(self, msg: Message) -> None:
        """Send a length-prefixed Message over TCP or WebSocket."""
        frame = msg.encode()
        packet = _FRAME_LEN_STRUCT.pack(len(frame)) + frame
        with self._send_lock:
            wire_bytes = (
                _ws_encode_frame(packet, opcode=0x2, mask=self._ws_client)
                if self._is_ws
                else packet
            )
            try:
                self._sock.sendall(wire_bytes)
            except (OSError, socket.timeout) as exc:
                raise TransportError(
                    f"Failed sending {msg.msg_type.value} to {self.peer_label or 'remote'}: {exc}"
                ) from exc

    def recv_message(self) -> Message:
        """Receive one length-prefixed Message from TCP or WebSocket."""
        with self._recv_lock:
            if not self._checked_upgrade:
                self._maybe_accept_websocket_upgrade()

            if self._is_ws:
                packet = self._recv_ws_message_bytes()
                if len(packet) < _FRAME_LEN_STRUCT.size:
                    raise TransportError("WebSocket frame too short for length prefix.")
                (frame_len,) = _FRAME_LEN_STRUCT.unpack_from(packet, 0)
                if frame_len == 0 or len(packet) < _FRAME_LEN_STRUCT.size + frame_len:
                    raise TransportError("Corrupt WebSocket message payload.")
                frame = packet[_FRAME_LEN_STRUCT.size : _FRAME_LEN_STRUCT.size + frame_len]
                return Message.decode(frame)

            len_bytes = self._recv_exact(_FRAME_LEN_STRUCT.size)
            (frame_len,) = _FRAME_LEN_STRUCT.unpack(len_bytes)
            if frame_len == 0:
                raise TransportError("Received empty 0-byte frame.")
            frame = self._recv_exact(frame_len)
            return Message.decode(frame)

    def _maybe_accept_websocket_upgrade(self) -> None:
        """Auto-detect an incoming HTTP WebSocket upgrade (`GET `) on the server side."""
        self._checked_upgrade = True
        first4 = self._recv_exact(4)
        if first4 != b"GET ":
            # Normal raw TCP stream: store the 4 bytes back in a tiny pre-buffer.
            self._prebuffer = first4
            return

        # Read remaining HTTP request headers up to \r\n\r\n.
        buf = bytearray(first4)
        while b"\r\n\r\n" not in buf:
            chunk = self._recv_some(4096)
            buf.extend(chunk)
            if len(buf) > 16384:
                raise TransportError("HTTP upgrade header too large.")

        header_text = bytes(buf).decode("latin-1")
        ws_key = None
        for line in header_text.split("\r\n"):
            if line.lower().startswith("sec-websocket-key:"):
                ws_key = line.split(":", 1)[1].strip()
                break
        if not ws_key:
            self._sock.sendall(
                b"HTTP/1.1 426 Upgrade Required\r\n"
                b"Content-Type: text/plain\r\n"
                b"Content-Length: 27\r\n"
                b"Connection: close\r\n\r\n"
                b"WebSocket upgrade required."
            )
            raise ConnectionClosedError("Received plain HTTP request without Sec-WebSocket-Key.")

        accept_val = _ws_accept_key(ws_key)
        response = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept_val}\r\n\r\n"
        ).encode("ascii")
        self._sock.sendall(response)
        self._is_ws = True

    def _recv_ws_message_bytes(self) -> bytes:
        """Read one complete WebSocket message (handling fragmentation and PING/CLOSE)."""
        fragments: list[bytes] = []
        while True:
            b1, b2 = self._recv_exact(2)
            fin = (b1 & 0x80) != 0
            opcode = b1 & 0x0F
            masked = (b2 & 0x80) != 0
            length = b2 & 0x7F

            if length == 126:
                (length,) = struct.unpack("!H", self._recv_exact(2))
            elif length == 127:
                (length,) = struct.unpack("!Q", self._recv_exact(8))

            mask_key = self._recv_exact(4) if masked else b""
            payload = self._recv_exact(length) if length > 0 else b""
            if masked and payload:
                payload = bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload))

            if opcode == 0x8:  # CLOSE
                raise ConnectionClosedError(
                    f"WebSocket closed by {self.peer_label or 'remote'}."
                )
            if opcode == 0x9:  # PING -> reply PONG
                with self._send_lock:
                    self._sock.sendall(_ws_encode_frame(payload, opcode=0xA, mask=self._ws_client))
                continue
            if opcode == 0xA:  # PONG
                continue

            fragments.append(payload)
            if fin:
                return b"".join(fragments)

    def _recv_some(self, max_bytes: int) -> bytes:
        try:
            chunk = self._sock.recv(max_bytes)
        except socket.timeout as exc:
            raise TransportTimeoutError(
                f"Timed out waiting for data from {self.peer_label or 'remote'}."
            ) from exc
        except OSError as exc:
            raise TransportError(
                f"Socket error from {self.peer_label or 'remote'}: {exc}"
            ) from exc
        if not chunk:
            raise ConnectionClosedError(
                f"Connection closed by {self.peer_label or 'remote'}."
            )
        return chunk

    def _recv_exact(self, num_bytes: int) -> bytes:
        chunks: list[bytes] = []
        remaining = num_bytes

        prebuf = getattr(self, "_prebuffer", b"")
        if prebuf:
            take = min(remaining, len(prebuf))
            chunks.append(prebuf[:take])
            self._prebuffer = prebuf[take:]
            remaining -= take

        while remaining > 0:
            try:
                chunk = self._sock.recv(remaining)
            except socket.timeout as exc:
                raise TransportTimeoutError(
                    f"Timed out waiting for data from {self.peer_label or 'remote'}."
                ) from exc
            except OSError as exc:
                raise TransportError(
                    f"Socket error from {self.peer_label or 'remote'}: {exc}"
                ) from exc
            if not chunk:
                raise ConnectionClosedError(
                    f"Connection closed by {self.peer_label or 'remote'}."
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def close(self) -> None:
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self._sock.close()
        except OSError:
            pass


class Listener:
    """Wrapper around a listening TCP server socket."""

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock

    def accept(self, timeout: float | None = None) -> tuple[TCPConnection, tuple[str, int]]:
        """Accept an incoming TCP (or WebSocket-upgraded) connection."""
        self._sock.settimeout(timeout)
        try:
            client_sock, addr = self._sock.accept()
        except socket.timeout as exc:
            raise TransportTimeoutError("Timed out waiting for incoming connection.") from exc
        except OSError as exc:
            raise TransportError(f"Listener accept failed: {exc}") from exc
        return TCPConnection(client_sock, peer_label=f"{addr[0]}:{addr[1]}"), addr

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass


def get_local_hostname() -> str:
    """Return the local machine's hostname."""
    return socket.gethostname()


def connect(
    address: str,
    port: int,
    timeout: float = 15.0,
    label: str = "",
    transport: str = "tcp",
) -> TCPConnection:
    """Open a persistent TCP or WebSocket-over-TLS (`wss`) connection."""
    target_label = label or f"{address}:{port}"
    try:
        sock = socket.create_connection((address, port), timeout=timeout)
    except OSError as exc:
        raise TransportError(f"Could not connect to {target_label}: {exc}") from exc

    if transport == "tcp":
        return TCPConnection(sock, peer_label=target_label)

    if transport not in ("ws", "wss"):
        sock.close()
        raise ValueError(f"Unsupported transport '{transport}'. Use 'tcp', 'ws', or 'wss'.")

    if transport == "wss":
        try:
            ctx = ssl.create_default_context()
            sock = ctx.wrap_socket(sock, server_hostname=address)
        except OSError as exc:
            sock.close()
            raise TransportError(f"TLS handshake failed with {target_label}: {exc}") from exc

    # Perform RFC 6455 WebSocket client handshake.
    ws_key = base64.b64encode(os.urandom(16)).decode("ascii")
    host_header = address if (transport == "wss" and port == 443) else f"{address}:{port}"
    req = (
        f"GET / HTTP/1.1\r\n"
        f"Host: {host_header}\r\n"
        f"Upgrade: websocket\r\n"
        f"Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {ws_key}\r\n"
        f"Sec-WebSocket-Version: 13\r\n\r\n"
    ).encode("ascii")
    try:
        sock.sendall(req)
        buf = bytearray()
        while b"\r\n\r\n" not in buf:
            chunk = sock.recv(4096)
            if not chunk:
                raise ConnectionClosedError(f"WebSocket handshake closed by {target_label}.")
            buf.extend(chunk)
        status_line = bytes(buf).split(b"\r\n", 1)[0].decode("latin-1")
        if "101" not in status_line:
            raise TransportError(f"WebSocket upgrade rejected by {target_label}: {status_line}")
    except Exception:
        sock.close()
        raise

    return TCPConnection(sock, peer_label=target_label, is_websocket=True, ws_client=True)


def listen(host: str, port: int, backlog: int = 16) -> Listener:
    """Bind and listen on `(host, port)` with SO_REUSEADDR."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(backlog)
    return Listener(srv)
