"""TCP socket transport layer for Local LLM Runner V1.

Only this module knows about sockets. Every message is sent as a 4-byte
big-endian length prefix followed by the encoded Message frame.
"""

from __future__ import annotations

import socket
import struct
import threading

from src.common.messages import Message

_FRAME_LEN_STRUCT = struct.Struct("!I")


class TransportError(Exception):
    """Base error for network transport failures."""


class TransportTimeoutError(TransportError):
    """Raised when a socket accept or receive operation times out."""


class ConnectionClosedError(TransportError):
    """Raised when the remote peer closes the TCP connection."""


class TCPConnection:
    """Thread-safe wrapper around a persistent TCP socket."""

    def __init__(self, sock: socket.socket, peer_label: str = "") -> None:
        self._sock = sock
        self.peer_label = peer_label
        self._send_lock = threading.Lock()
        self._recv_lock = threading.Lock()
        try:
            self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass

    def settimeout(self, timeout: float | None) -> None:
        self._sock.settimeout(timeout)

    def send_message(self, msg: Message) -> None:
        """Send a length-prefixed Message over the TCP connection."""
        frame = msg.encode()
        packet = _FRAME_LEN_STRUCT.pack(len(frame)) + frame
        with self._send_lock:
            try:
                self._sock.sendall(packet)
            except (OSError, socket.timeout) as exc:
                raise TransportError(
                    f"Failed sending {msg.msg_type.value} to {self.peer_label or 'remote'}: {exc}"
                ) from exc

    def recv_message(self) -> Message:
        """Receive one length-prefixed Message from the TCP connection."""
        with self._recv_lock:
            len_bytes = self._recv_exact(_FRAME_LEN_STRUCT.size)
            (frame_len,) = _FRAME_LEN_STRUCT.unpack(len_bytes)
            if frame_len == 0:
                raise TransportError("Received empty 0-byte frame.")
            frame = self._recv_exact(frame_len)
        return Message.decode(frame)

    def _recv_exact(self, num_bytes: int) -> bytes:
        chunks: list[bytes] = []
        remaining = num_bytes
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
        """Accept an incoming TCP connection, raising TransportTimeoutError on timeout."""
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


def connect(address: str, port: int, timeout: float = 15.0, label: str = "") -> TCPConnection:
    """Open a persistent TCP connection to `(address, port)`."""
    target_label = label or f"{address}:{port}"
    try:
        sock = socket.create_connection((address, port), timeout=timeout)
    except OSError as exc:
        raise TransportError(f"Could not connect to {target_label}: {exc}") from exc
    return TCPConnection(sock, peer_label=target_label)


def listen(host: str, port: int, backlog: int = 16) -> Listener:
    """Bind and listen on `(host, port)` with SO_REUSEADDR."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(backlog)
    return Listener(srv)
