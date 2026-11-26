"""Tests for WebSocket auto-upgrade and message round-trip in transport.py."""

from __future__ import annotations

import threading

from src.common.messages import MessageType, activate_msg, register_msg
from src.common.transport import connect, listen


def test_websocket_auto_upgrade_and_binary_round_trip() -> None:
    listener = listen("127.0.0.1", 0)
    port = listener._sock.getsockname()[1]
    received = []

    def server_task() -> None:
        conn, _ = listener.accept(timeout=5.0)
        try:
            msg1 = conn.recv_message()
            received.append(msg1)
            conn.send_message(
                activate_msg("s1", 0, [1, 4], "float16", b"\x01\x02\x03\x04\x05\x06\x07\x08")
            )
        finally:
            conn.close()

    t = threading.Thread(target=server_task, daemon=True)
    t.start()

    try:
        client = connect("127.0.0.1", port, timeout=5.0, transport="ws")
        try:
            client.send_message(register_msg("kaggle-1", auth_token="tok"))
            reply = client.recv_message()
            assert reply.msg_type == MessageType.ACTIVATE
            assert reply.payload == b"\x01\x02\x03\x04\x05\x06\x07\x08"
        finally:
            client.close()
        t.join(timeout=5.0)
        assert len(received) == 1
        assert received[0].msg_type == MessageType.REGISTER
        assert received[0].header["peer_name"] == "kaggle-1"
    finally:
        listener.close()
