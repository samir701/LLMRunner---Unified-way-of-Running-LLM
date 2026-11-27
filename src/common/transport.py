"""TCP, WebSocket (WSS over port 443), and optional SOCKS5 transport layer for Local LLM Runner.

Only this module knows about sockets and TLS.
- Raw TCP: 4-byte big-endian length prefix + encoded Message frame.
- WebSocket (RFC 6455): binary frames carrying the exact same 4-byte length-prefixed
  Message frame, allowing Kaggle / NAT peers to connect over port 443 via
  Tailscale Funnel or Cloudflare Tunnel.
- Optional SOCKS5 proxy: allows peers running userspace Tailscale (`--socks5-server`)
  to connect over direct WireGuard TCP without TLS/WebSocket overhead.
"""

from __future__ import annotations

import base64
import hashlib
import os
import socket
import ssl
import struct
import threading

import numpy as np

from src.common.messages import Message

_FRAME_LEN_STRUCT = struct.Struct("!I")
_TWO_U32_STRUCT = struct.Struct("!II")
_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
_TCP_QUICKACK = getattr(socket, "TCP_QUICKACK", None)


class TransportError(Exception):
    """Base error for network transport failures."""


class TransportTimeoutError(TransportError):
    """Raised when a socket accept or receive operation times out."""


class ConnectionClosedError(TransportError):
    """Raised when the remote peer closes the connection."""


def _tune_socket(sock: socket.socket) -> None:
    """Apply low-latency TCP socket options (TCP_NODELAY, SO_KEEPALIVE, TCP_QUICKACK)."""
    for level, opt, val in (
        (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1),
        (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
    ):
        try:
            sock.setsockopt(level, opt, val)
        except OSError:
            pass
    if _TCP_QUICKACK is not None:
        try:
            sock.setsockopt(socket.IPPROTO_TCP, _TCP_QUICKACK, 1)
        except OSError:
            pass


def _ws_accept_key(client_key: str) -> str:
    digest = hashlib.sha1((client_key.strip() + _WS_GUID).encode("ascii")).digest()
    return base64.b64encode(digest).decode("ascii")


def _xor_mask_inplace(buf: bytearray, mask_key: bytes, offset: int = 0) -> None:
    """Vectorized RFC 6455 WebSocket XOR masking/unmasking in-place."""
    n = len(buf) - offset
    if n <= 0:
        return
    arr = np.frombuffer(buf, dtype=np.uint8, offset=offset)
    rem = n & 3
    n4 = n - rem
    if n4:
        key32 = np.frombuffer(mask_key, dtype=np.uint32)[0]
        arr[:n4].view(np.uint32).__ixor__(key32)
    for i in range(rem):
        arr[n4 + i] ^= mask_key[i]


def _ws_header_size(length: int, mask: bool) -> int:
    base = 2
    if length >= 65536:
        base += 8
    elif length >= 126:
        base += 2
    return base + (4 if mask else 0)


def _write_ws_header(
    buf: bytearray,
    length: int,
    opcode: int = 0x2,
    mask_key: bytes | None = None,
) -> int:
    """Write an RFC 6455 WebSocket header into `buf` starting at offset 0; return header length."""
    buf[0] = 0x80 | (opcode & 0x0F)
    mask_bit = 0x80 if mask_key is not None else 0x00
    pos = 2
    if length < 126:
        buf[1] = mask_bit | length
    elif length < 65536:
        buf[1] = mask_bit | 126
        struct.pack_into("!H", buf, 2, length)
        pos = 4
    else:
        buf[1] = mask_bit | 127
        struct.pack_into("!Q", buf, 2, length)
        pos = 10

    if mask_key is not None:
        buf[pos : pos + 4] = mask_key
        pos += 4
    return pos


def _ws_encode_frame(
    payload: bytes | bytearray | memoryview,
    opcode: int = 0x2,
    mask: bool = False,
) -> bytes:
    """Encode one RFC 6455 WebSocket frame using vectorized masking."""
    length = len(payload)
    mask_key = os.urandom(4) if mask else None
    hdr_len = _ws_header_size(length, mask=mask)
    out = bytearray(hdr_len + length)
    _write_ws_header(out, length, opcode=opcode, mask_key=mask_key)
    if length:
        out[hdr_len:] = payload
        if mask_key is not None:
            _xor_mask_inplace(out, mask_key, offset=hdr_len)
    return bytes(out)


class TCPConnection:
    """Thread-safe wrapper around a persistent TCP or WebSocket-upgraded socket."""

    def __init__(
        self,
        sock: socket.socket,
        peer_label: str = "",
        is_websocket: bool = False,
        ws_client: bool = False,
        prebuffer: bytes = b"",
    ) -> None:
        self._sock = sock
        self.peer_label = peer_label
        self._is_ws = is_websocket
        self._ws_client = ws_client
        self._checked_upgrade = is_websocket
        self._prebuffer = prebuffer
        self._send_lock = threading.Lock()
        self._recv_lock = threading.Lock()
        _tune_socket(self._sock)

    def settimeout(self, timeout: float | None) -> None:
        self._sock.settimeout(timeout)

    def send_message(self, msg: Message) -> None:
        """Send a length-prefixed Message over TCP or WebSocket in a single buffer/syscall."""
        header_bytes, payload = msg.encode_parts()
        header_len = len(header_bytes)
        payload_len = len(payload)
        frame_len = 4 + header_len + payload_len
        packet_len = 4 + frame_len

        with self._send_lock:
            if self._is_ws:
                mask_key = os.urandom(4) if self._ws_client else None
                ws_hdr_len = _ws_header_size(packet_len, mask=self._ws_client)
                wire = bytearray(ws_hdr_len + packet_len)
                _write_ws_header(wire, packet_len, opcode=0x2, mask_key=mask_key)
                _TWO_U32_STRUCT.pack_into(wire, ws_hdr_len, frame_len, header_len)
                hdr_off = ws_hdr_len + 8
                wire[hdr_off : hdr_off + header_len] = header_bytes
                if payload_len:
                    wire[hdr_off + header_len :] = payload
                if mask_key is not None:
                    _xor_mask_inplace(wire, mask_key, offset=ws_hdr_len)
            else:
                wire = bytearray(packet_len)
                _TWO_U32_STRUCT.pack_into(wire, 0, frame_len, header_len)
                wire[8 : 8 + header_len] = header_bytes
                if payload_len:
                    wire[8 + header_len :] = payload

            try:
                self._sock.sendall(wire)
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
                frame = memoryview(packet)[
                    _FRAME_LEN_STRUCT.size : _FRAME_LEN_STRUCT.size + frame_len
                ]
                msg = Message.decode(frame)
            else:
                len_buf = self._recv_exact_bytearray(_FRAME_LEN_STRUCT.size)
                (frame_len,) = _FRAME_LEN_STRUCT.unpack_from(len_buf, 0)
                if frame_len == 0:
                    raise TransportError("Received empty 0-byte frame.")
                frame_buf = self._recv_exact_bytearray(frame_len)
                msg = Message.decode(frame_buf)

            if _TCP_QUICKACK is not None:
                try:
                    self._sock.setsockopt(socket.IPPROTO_TCP, _TCP_QUICKACK, 1)
                except OSError:
                    pass
            return msg

    def _maybe_accept_websocket_upgrade(self) -> None:
        """Auto-detect an incoming HTTP WebSocket upgrade (`GET `) on the server side."""
        self._checked_upgrade = True
        first4 = bytes(self._recv_exact_bytearray(4))
        if first4 in (b"HEAD", b"POST", b"PUT ", b"OPTI", b"PRI "):
            self._sock.sendall(
                b"HTTP/1.1 426 Upgrade Required\r\n"
                b"Content-Type: text/plain\r\n"
                b"Content-Length: 27\r\n"
                b"Connection: close\r\n\r\n"
                b"WebSocket upgrade required."
            )
            raise ConnectionClosedError(f"Received non-WebSocket HTTP request ({first4!r}).")
        if first4 != b"GET ":
            # Normal raw TCP stream: store the 4 bytes back in a tiny pre-buffer.
            self._prebuffer = first4 + self._prebuffer
            return

        # Read remaining HTTP request headers up to \r\n\r\n.
        buf = bytearray(first4)
        while b"\r\n\r\n" not in buf:
            chunk = self._recv_some(4096)
            buf.extend(chunk)
            if len(buf) > 16384:
                raise TransportError("HTTP upgrade header too large.")

        header_raw, trailing = bytes(buf).split(b"\r\n\r\n", 1)
        if trailing:
            self._prebuffer = trailing + self._prebuffer

        header_text = header_raw.decode("latin-1")
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

    def _recv_ws_message_bytes(self) -> bytearray:
        """Read one complete WebSocket message (handling fragmentation and PING/CLOSE)."""
        fragments: list[bytearray] = []
        while True:
            hdr = self._recv_exact_bytearray(2)
            b1, b2 = hdr[0], hdr[1]
            fin = (b1 & 0x80) != 0
            opcode = b1 & 0x0F
            masked = (b2 & 0x80) != 0
            length = b2 & 0x7F

            if length == 126:
                (length,) = struct.unpack_from("!H", self._recv_exact_bytearray(2), 0)
            elif length == 127:
                (length,) = struct.unpack_from("!Q", self._recv_exact_bytearray(8), 0)

            mask_key = bytes(self._recv_exact_bytearray(4)) if masked else b""
            payload_buf = self._recv_exact_bytearray(length) if length > 0 else bytearray()
            if masked and length > 0:
                _xor_mask_inplace(payload_buf, mask_key)

            if opcode == 0x8:  # CLOSE
                raise ConnectionClosedError(
                    f"WebSocket closed by {self.peer_label or 'remote'}."
                )
            if opcode == 0x9:  # PING -> reply PONG
                with self._send_lock:
                    self._sock.sendall(
                        _ws_encode_frame(payload_buf, opcode=0xA, mask=self._ws_client)
                    )
                continue
            if opcode == 0xA:  # PONG
                continue

            if fin and not fragments:
                return payload_buf
            fragments.append(payload_buf)
            if fin:
                out = bytearray()
                for frag in fragments:
                    out.extend(frag)
                return out

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

    def _recv_exact_bytearray(self, num_bytes: int) -> bytearray:
        """Read exactly `num_bytes` directly into a writable bytearray via `recv_into`."""
        buf = bytearray(num_bytes)
        offset = 0

        prebuf = self._prebuffer
        if prebuf:
            take = min(num_bytes, len(prebuf))
            buf[:take] = prebuf[:take]
            self._prebuffer = prebuf[take:]
            offset = take

        if offset < num_bytes:
            mv = memoryview(buf)
            while offset < num_bytes:
                try:
                    n = self._sock.recv_into(mv[offset:])
                except socket.timeout as exc:
                    raise TransportTimeoutError(
                        f"Timed out waiting for data from {self.peer_label or 'remote'}."
                    ) from exc
                except OSError as exc:
                    raise TransportError(
                        f"Socket error from {self.peer_label or 'remote'}: {exc}"
                    ) from exc
                if n == 0:
                    raise ConnectionClosedError(
                        f"Connection closed by {self.peer_label or 'remote'}."
                    )
                offset += n
        return buf

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


def _parse_proxy(proxy: str) -> tuple[str, int]:
    text = proxy.strip()
    if text.startswith("socks5://"):
        text = text[len("socks5://") :]
    if text.startswith("["):
        end = text.find("]")
        if end == -1 or end + 1 >= len(text) or text[end + 1] != ":":
            raise ValueError(f"Invalid IPv6 socks5_proxy '{proxy}'.")
        return text[1:end], int(text[end + 2 :])
    host, port_s = text.rsplit(":", 1)
    return host, int(port_s)


def _recv_n_sock(sock: socket.socket, n: int, label: str) -> bytes:
    buf = bytearray(n)
    mv = memoryview(buf)
    off = 0
    while off < n:
        r = sock.recv_into(mv[off:])
        if r == 0:
            raise ConnectionClosedError(f"SOCKS5 proxy closed connection to {label}.")
        off += r
    return bytes(buf)


def _connect_via_socks5(
    proxy: str,
    address: str,
    port: int,
    timeout: float,
    label: str,
) -> socket.socket:
    """Establish a TCP connection to `(address, port)` through a SOCKS5 proxy (e.g. Tailscale)."""
    proxy_host, proxy_port = _parse_proxy(proxy)
    sock = socket.create_connection((proxy_host, proxy_port), timeout=timeout)
    _tune_socket(sock)
    try:
        sock.sendall(b"\x05\x01\x00")
        greeting = _recv_n_sock(sock, 2, label)
        if greeting != b"\x05\x00":
            raise TransportError(f"SOCKS5 proxy rejected no-auth greeting for {label}: {greeting!r}")

        try:
            addr_bytes = socket.inet_aton(address)
            req_addr = b"\x01" + addr_bytes
        except OSError:
            try:
                addr6_bytes = socket.inet_pton(socket.AF_INET6, address)
                req_addr = b"\x04" + addr6_bytes
            except OSError:
                host_b = address.encode("utf-8")
                if len(host_b) > 255:
                    raise ValueError(f"Target hostname too long for SOCKS5: {address}")
                req_addr = b"\x03" + bytes([len(host_b)]) + host_b

        sock.sendall(b"\x05\x01\x00" + req_addr + struct.pack("!H", port))
        rep = _recv_n_sock(sock, 4, label)
        if rep[0] != 5 or rep[1] != 0:
            raise TransportError(f"SOCKS5 CONNECT to {label} failed with status {rep[1]}.")

        atyp = rep[3]
        if atyp == 1:
            _recv_n_sock(sock, 4 + 2, label)
        elif atyp == 3:
            (dlen,) = _recv_n_sock(sock, 1, label)
            _recv_n_sock(sock, dlen + 2, label)
        elif atyp == 4:
            _recv_n_sock(sock, 16 + 2, label)
        else:
            raise TransportError(f"SOCKS5 unknown ATYP {atyp} for {label}.")
        return sock
    except Exception:
        sock.close()
        raise


def connect(
    address: str,
    port: int,
    timeout: float = 15.0,
    label: str = "",
    transport: str = "tcp",
    socks5_proxy: str | None = None,
) -> TCPConnection:
    """Open a persistent TCP or WebSocket-over-TLS (`wss`) connection."""
    address = address.strip()
    target_label = label or f"{address}:{port}"
    try:
        if socks5_proxy:
            sock = _connect_via_socks5(socks5_proxy, address, port, timeout, target_label)
        else:
            sock = socket.create_connection((address, port), timeout=timeout)
            _tune_socket(sock)
    except TransportError:
        raise
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
        header_raw, trailing = bytes(buf).split(b"\r\n\r\n", 1)
        status_line = header_raw.split(b"\r\n", 1)[0].decode("latin-1")
        if "101" not in status_line:
            raise TransportError(f"WebSocket upgrade rejected by {target_label}: {status_line}")
    except Exception:
        sock.close()
        raise

    return TCPConnection(
        sock,
        peer_label=target_label,
        is_websocket=True,
        ws_client=True,
        prebuffer=trailing,
    )


def listen(host: str, port: int, backlog: int = 16) -> Listener:
    """Bind and listen on `(host, port)` with SO_REUSEADDR."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(backlog)
    return Listener(srv)
