"""Tests for the V2 Desktop Application API bridge and WebView window."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from src.gui.api import DesktopAPI, build_host_config_from_dict, build_peer_config_from_dict


def test_desktop_api_initial_state_and_yaml_round_trip(tmp_path: Path) -> None:
    events: list[tuple[str, object]] = []
    api = DesktopAPI(on_event=lambda ev, data: events.append((ev, data)))

    init = api.get_initial_state()
    assert "host_config" in init
    assert "peer_config" in init
    assert init["hardware"]["ok"] is True

    host_yaml = tmp_path / "host.yaml"
    host_data = init["host_config"]
    host_data["peers"] = [{"name": "kaggle-1", "mode": "inbound", "address": "", "port": 5001}]
    api.save_host_yaml(str(host_yaml), host_data)
    reloaded_host = api.load_host_yaml(str(host_yaml))
    assert reloaded_host["config"]["peers"][0]["name"] == "kaggle-1"

    peer_yaml = tmp_path / "peer.yaml"
    peer_data = {
        "name": "peer-test",
        "listen_port": 5001,
        "model": "models/Qwen2.5-0.5B-Instruct",
        "device": "cpu",
        "layer_start": "",
        "layer_end": "",
        "conn_mode": "wan",
        "host_address": "example.ts.net",
        "host_port": 443,
        "host_transport": "wss",
        "socks5_proxy": "",
        "auth_token": "tok",
    }
    api.save_peer_yaml(str(peer_yaml), peer_data)
    reloaded_peer = api.load_peer_yaml(str(peer_yaml))
    assert reloaded_peer["config"]["host_address"] == "example.ts.net"
    assert reloaded_peer["config"]["host_transport"] == "wss"

    built_h = build_host_config_from_dict(host_data)
    built_p = build_peer_config_from_dict(peer_data)
    assert built_h.peers[0].mode == "inbound"
    assert built_p.host_port == 443


def test_webview_window_loads_frontend_and_js_bridge() -> None:
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        pytest.skip("Headless Linux without DISPLAY")

    import webview

    api = DesktopAPI()
    html_path = Path(__file__).resolve().parent.parent / "src" / "gui" / "web" / "index.html"
    assert html_path.exists()

    win = webview.create_window(
        "Smoke Test",
        url=html_path.as_uri(),
        js_api=api,
        hidden=True,
    )
    api.attach_window(win)
    result: dict[str, object] = {}

    def verify_and_close() -> None:
        result["title"] = win.evaluate_js("document.title")
        result["has_host"] = win.evaluate_js("Boolean(document.getElementById('hostSection'))")
        result["has_peer"] = win.evaluate_js("Boolean(document.getElementById('peerSection'))")
        win.destroy()

    webview.start(verify_and_close)
    assert result["title"] == "Local LLM Runner V2"
    assert result["has_host"] is True
    assert result["has_peer"] is True
