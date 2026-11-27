"""Tests for first-run setup wizard environment checks and Tailscale parsing."""

from __future__ import annotations

import json
from pathlib import Path

from src.gui.api import DesktopAPI
from src.gui.setup_wizard import (
    check_dependencies_status,
    check_python_status,
    parse_tailscale_status_json,
)


def test_python_and_dependencies_checks_pass_in_venv() -> None:
    py = check_python_status()
    assert py["ok"] is True
    assert py["bundled_version"]

    deps = check_dependencies_status()
    assert deps["ok"] is True
    assert deps["missing"] == []


def test_parse_tailscale_status_json_extracts_dns_and_state() -> None:
    sample = json.dumps(
        {
            "BackendState": "Running",
            "Self": {
                "DNSName": "my-pc.tail1234.ts.net.",
                "TailscaleIPs": ["100.64.0.10", "fd7a::1"],
            },
        }
    )
    parsed = parse_tailscale_status_json(sample)
    assert parsed["logged_in"] is True
    assert parsed["dns_name"] == "my-pc.tail1234.ts.net"
    assert parsed["ip"] == "100.64.0.10"


def test_desktop_api_setup_status_and_complete(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "host.example.yaml").write_text(
        'model: "models/dummy"\nlisten_port: 5000\npeers:\n  - name: "k1"\n    mode: "inbound"\n',
        encoding="utf-8",
    )
    (tmp_path / "config" / "peer.example.yaml").write_text(
        'name: "p1"\nlisten_port: 5001\nmodel: "models/dummy"\n',
        encoding="utf-8",
    )

    api = DesktopAPI()
    st = api.get_setup_status()
    assert st["setup_complete"] is False

    res = api.complete_setup(
        {
            "role": "host",
            "model": "models/Qwen2.5-0.5B-Instruct",
            "auth_token": "tok-xyz",
            "host_address": "my-pc.tail1234.ts.net",
        }
    )
    assert res["ok"] is True
    assert (tmp_path / "config" / ".setup_done").exists()
    assert (tmp_path / "config" / "host.yaml").exists()
    assert (tmp_path / "config" / "peer.yaml").exists()
