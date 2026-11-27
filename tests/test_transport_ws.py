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


def test_websocket_large_and_odd_payload_masking() -> None:
    import os

    listener = listen("127.0.0.1", 0)
    port = listener._sock.getsockname()[1]
    big_payload = os.urandom(303875)  # > 65536 and not a multiple of 4
    received = []

    def server_task() -> None:
        conn, _ = listener.accept(timeout=5.0)
        try:
            msg = conn.recv_message()
            received.append(bytes(msg.payload))
            conn.send_message(activate_msg("s1", 1, [1, 1], "float16", b"\xaa\xbb\xcc"))
        finally:
            conn.close()

    t = threading.Thread(target=server_task, daemon=True)
    t.start()

    try:
        client = connect("127.0.0.1", port, timeout=5.0, transport="ws")
        try:
            client.send_message(activate_msg("s1", 0, [1, len(big_payload)], "float16", big_payload))
            reply = client.recv_message()
            assert bytes(reply.payload) == b"\xaa\xbb\xcc"
        finally:
            client.close()
        t.join(timeout=5.0)
        assert received == [big_payload]
    finally:
        listener.close()


def test_socks5_proxy_tunnel_connection() -> None:
    import socket
    import struct

    # Target echo-protocol server
    target_listener = listen("127.0.0.1", 0)
    target_port = target_listener._sock.getsockname()[1]

    # Minimal SOCKS5 proxy listening on localhost
    proxy_srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    proxy_srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    proxy_srv.bind(("127.0.0.1", 0))
    proxy_srv.listen(4)
    proxy_port = proxy_srv.getsockname()[1]

    def target_task() -> None:
        conn, _ = target_listener.accept(timeout=5.0)
        try:
            msg = conn.recv_message()
            conn.send_message(activate_msg("ok", msg.header["token_pos"], [1, 1], "float16", b"\x11\x22"))
        finally:
            conn.close()

    def proxy_task() -> None:
        cli, _ = proxy_srv.accept()
        try:
            greeting = cli.recv(256)
            assert greeting == b"\x05\x01\x00"
            cli.sendall(b"\x05\x00")
            req = cli.recv(256)
            assert req[:4] == b"\x05\x01\x00\x01"
            dst_ip = socket.inet_ntoa(req[4:8])
            (dst_port,) = struct.unpack("!H", req[8:10])
            upstream = socket.create_connection((dst_ip, dst_port), timeout=5.0)
            cli.sendall(b"\x05\x00\x00\x01\x7f\x00\x00\x01" + struct.pack("!H", dst_port))

            def pipe(src: socket.socket, dst: socket.socket) -> None:
                try:
                    while True:
                        data = src.recv(4096)
                        if not data:
                            break
                        dst.sendall(data)
                except OSError:
                    pass

            t1 = threading.Thread(target=pipe, args=(cli, upstream), daemon=True)
            t2 = threading.Thread(target=pipe, args=(upstream, cli), daemon=True)
            t1.start()
            t2.start()
            t1.join(timeout=5.0)
            t2.join(timeout=5.0)
            upstream.close()
        finally:
            cli.close()

    tt = threading.Thread(target=target_task, daemon=True)
    pt = threading.Thread(target=proxy_task, daemon=True)
    tt.start()
    pt.start()

    try:
        conn = connect(
            "127.0.0.1",
            target_port,
            timeout=5.0,
            transport="tcp",
            socks5_proxy=f"127.0.0.1:{proxy_port}",
        )
        try:
            conn.send_message(activate_msg("s", 9, [1, 1], "float16", b"\x01\x02"))
            reply = conn.recv_message()
            assert reply.header["token_pos"] == 9
            assert bytes(reply.payload) == b"\x11\x22"
        finally:
            conn.close()
        tt.join(timeout=5.0)
    finally:
        proxy_srv.close()
        target_listener.close()
