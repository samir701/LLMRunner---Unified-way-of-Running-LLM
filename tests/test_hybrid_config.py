"""Tests for hybrid LAN (outbound) + global/NAT (inbound) configuration parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.common.config import load_host_config, load_peer_config


def test_hybrid_host_config_allows_mixed_inbound_and_outbound_peers(tmp_path: Path) -> None:
    cfg_file = tmp_path / "host.yaml"
    cfg_file.write_text(
        """
model: "models/qwen"
listen_port: 5000
auth_token: "secret"
routing: "auto"
peers:
  - name: "local-laptop"
    mode: "outbound"
    address: "192.168.1.21"
    port: 5001
  - name: "kaggle-1"
    mode: "inbound"
  - name: "friend-pc"
    mode: "inbound"
""",
        encoding="utf-8",
    )

    cfg = load_host_config(cfg_file)
    assert cfg.auth_token == "secret"
    assert cfg.routing == "auto"
    assert len(cfg.peers) == 3
    assert cfg.peers[0].mode == "outbound"
    assert cfg.peers[0].address == "192.168.1.21"
    assert cfg.peers[1].mode == "inbound"
    assert cfg.peers[1].address is None
    assert cfg.peers[2].mode == "inbound"


def test_outbound_peer_config_requires_both_host_address_and_port(tmp_path: Path) -> None:
    good_peer = tmp_path / "peer_good.yaml"
    good_peer.write_text(
        """
name: "kaggle-1"
model: "models/qwen"
host_address: "0.tcp.ngrok.io"
host_port: 12345
auth_token: "secret"
""",
        encoding="utf-8",
    )
    cfg = load_peer_config(good_peer)
    assert cfg.host_address == "0.tcp.ngrok.io"
    assert cfg.host_port == 12345
    assert cfg.auth_token == "secret"

    bad_peer = tmp_path / "peer_bad.yaml"
    bad_peer.write_text(
        """
name: "kaggle-1"
model: "models/qwen"
host_address: "0.tcp.ngrok.io"
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="host_address"):
        load_peer_config(bad_peer)
