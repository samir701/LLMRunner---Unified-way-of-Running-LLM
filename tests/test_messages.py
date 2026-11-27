"""Tests for protocol message encoding and decoding round-trips."""

from __future__ import annotations

from src.common.messages import (
    Message,
    MessageType,
    activate_msg,
    assign_msg,
    hello_msg,
    info_msg,
    logits_msg,
    ping_msg,
    pong_msg,
    ready_msg,
    register_msg,
    stop_msg,
)


def test_control_messages_round_trip() -> None:
    messages = [
        register_msg("kaggle-1", auth_token="secret-123"),
        hello_msg("my-host", auth_token="secret-123"),
        info_msg("peer-1", free_memory_bytes=8_000_000_000, device_type="cuda"),
        assign_msg(
            "models/qwen",
            10,
            20,
            is_last=False,
            next_address="192.168.1.22",
            next_port=5001,
            relay_via_host=True,
        ),
        ready_msg("peer-1", 10, 20),
        ping_msg(1234.5),
        pong_msg("peer-1", 1234.5),
        stop_msg("user_exit"),
    ]
    for msg in messages:
        decoded = Message.decode(msg.encode())
        assert decoded.msg_type == msg.msg_type
        assert decoded.header == msg.header
        assert decoded.payload == msg.payload


def test_tensor_payload_messages_round_trip() -> None:
    payload = b"\x01\x02\x03\x04\x05\x06\x07\x08"
    act = activate_msg("sess-1", token_pos=7, shape=[1, 1, 4], dtype="float16", payload=payload)
    decoded_act = Message.decode(act.encode())
    assert decoded_act.msg_type == MessageType.ACTIVATE
    assert decoded_act.header["session_id"] == "sess-1"
    assert decoded_act.header["token_pos"] == 7
    assert decoded_act.payload == payload

    log = logits_msg("sess-1", token_pos=7, shape=[1, 4], dtype="float16", payload=payload)
    decoded_log = Message.decode(log.encode())
    assert decoded_log.msg_type == MessageType.LOGITS
    assert decoded_log.payload == payload

    from src.common.messages import token_msg

    act_s = activate_msg(
        "sess-1",
        token_pos=0,
        shape=[1, 3, 4],
        dtype="float16",
        payload=payload,
        sample={"t": 0.5, "p": 0.9, "k": 40, "r": 1.1},
    )
    decoded_act_s = Message.decode(act_s.encode())
    assert decoded_act_s.header["sample"] == {"t": 0.5, "p": 0.9, "k": 40, "r": 1.1}

    tok = token_msg("sess-1", token_pos=7, token_id=42)
    decoded_tok = Message.decode(tok.encode())
    assert decoded_tok.msg_type == MessageType.TOKEN
    assert decoded_tok.header["token_id"] == 42
