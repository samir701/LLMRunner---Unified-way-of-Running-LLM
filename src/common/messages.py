"""Protocol message definitions and wire encoding/decoding.

Wire format (V1/V2-WAN):
- Outer framing (handled by transport.py): 4-byte big-endian total length prefix.
- Frame body: 4-byte big-endian JSON header length + UTF-8 JSON header + raw payload bytes.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

PROTOCOL_VERSION = "1.1"
_HEADER_LEN_STRUCT = struct.Struct("!I")


class MessageType(str, Enum):
    REGISTER = "REGISTER"
    HELLO = "HELLO"
    INFO = "INFO"
    ASSIGN = "ASSIGN"
    READY = "READY"
    ACTIVATE = "ACTIVATE"
    LOGITS = "LOGITS"
    PING = "PING"
    PONG = "PONG"
    STOP = "STOP"
    ERROR = "ERROR"


@dataclass
class Message:
    """A protocol message with a JSON header and optional binary payload."""

    msg_type: MessageType
    header: dict[str, Any] = field(default_factory=dict)
    payload: bytes = b""

    def encode(self) -> bytes:
        """Serialize message to `header_len (4B) + json_header + payload`."""
        full_header = {"type": self.msg_type.value, **self.header}
        header_bytes = json.dumps(full_header, separators=(",", ":")).encode("utf-8")
        return _HEADER_LEN_STRUCT.pack(len(header_bytes)) + header_bytes + self.payload

    @classmethod
    def decode(cls, frame: bytes) -> "Message":
        """Deserialize a frame (`header_len + json_header + payload`) into a Message."""
        if len(frame) < _HEADER_LEN_STRUCT.size:
            raise ValueError("Frame too short to contain header length.")
        (header_len,) = _HEADER_LEN_STRUCT.unpack_from(frame, 0)
        header_start = _HEADER_LEN_STRUCT.size
        header_end = header_start + header_len
        if len(frame) < header_end:
            raise ValueError(
                f"Truncated frame: expected {header_end} header bytes, got {len(frame)}."
            )
        header_dict = json.loads(frame[header_start:header_end].decode("utf-8"))
        msg_type_raw = header_dict.pop("type", None)
        if msg_type_raw is None:
            raise ValueError("Message JSON header missing required 'type' field.")
        msg_type = MessageType(msg_type_raw)
        payload = frame[header_end:]
        return cls(msg_type=msg_type, header=header_dict, payload=payload)


# --- Convenience constructors ---


def register_msg(
    peer_name: str,
    protocol_version: str = PROTOCOL_VERSION,
    auth_token: str | None = None,
) -> Message:
    header: dict[str, Any] = {
        "peer_name": peer_name,
        "protocol_version": protocol_version,
    }
    if auth_token is not None:
        header["auth_token"] = auth_token
    return Message(MessageType.REGISTER, header)


def hello_msg(
    host_name: str,
    protocol_version: str = PROTOCOL_VERSION,
    auth_token: str | None = None,
) -> Message:
    header: dict[str, Any] = {
        "host_name": host_name,
        "protocol_version": protocol_version,
    }
    if auth_token is not None:
        header["auth_token"] = auth_token
    return Message(MessageType.HELLO, header)


def info_msg(
    peer_name: str,
    free_memory_bytes: int,
    device_type: str,
    ready: bool = True,
    fixed_layers: tuple[int, int] | None = None,
    device_name: str = "",
) -> Message:
    header: dict[str, Any] = {
        "peer_name": peer_name,
        "free_memory_bytes": free_memory_bytes,
        "device_type": device_type,
        "device_name": device_name,
        "ready": ready,
    }
    if fixed_layers is not None:
        header["fixed_layers"] = list(fixed_layers)
    return Message(MessageType.INFO, header)


def assign_msg(
    model_name: str,
    layer_start: int,
    layer_end: int,
    is_last: bool,
    next_address: str | None = None,
    next_port: int | None = None,
    relay_via_host: bool = False,
) -> Message:
    return Message(
        MessageType.ASSIGN,
        {
            "model_name": model_name,
            "layer_start": layer_start,
            "layer_end": layer_end,
            "is_last": is_last,
            "next_address": next_address,
            "next_port": next_port,
            "relay_via_host": relay_via_host,
        },
    )


def ready_msg(peer_name: str, layer_start: int, layer_end: int) -> Message:
    return Message(
        MessageType.READY,
        {
            "peer_name": peer_name,
            "layer_start": layer_start,
            "layer_end": layer_end,
        },
    )


def activate_msg(
    session_id: str,
    token_pos: int,
    shape: list[int],
    dtype: str,
    payload: bytes,
) -> Message:
    return Message(
        MessageType.ACTIVATE,
        {
            "session_id": session_id,
            "token_pos": token_pos,
            "shape": shape,
            "dtype": dtype,
        },
        payload=payload,
    )


def logits_msg(
    session_id: str,
    token_pos: int,
    shape: list[int],
    dtype: str,
    payload: bytes,
) -> Message:
    return Message(
        MessageType.LOGITS,
        {
            "session_id": session_id,
            "token_pos": token_pos,
            "shape": shape,
            "dtype": dtype,
        },
        payload=payload,
    )


def ping_msg(timestamp: float) -> Message:
    return Message(MessageType.PING, {"ts": timestamp})


def pong_msg(peer_name: str, timestamp: float) -> Message:
    return Message(MessageType.PONG, {"peer_name": peer_name, "ts": timestamp})


def stop_msg(reason: str = "session_end") -> Message:
    return Message(MessageType.STOP, {"reason": reason})


def error_msg(source: str, detail: str) -> Message:
    return Message(MessageType.ERROR, {"source": source, "detail": detail})
