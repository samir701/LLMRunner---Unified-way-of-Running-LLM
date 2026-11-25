"""Load and validate YAML configuration (`host.yaml` and `peer.yaml`).

Config is read once at startup by this module and passed around as an
immutable-style dataclass object; no other module reads YAML directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PeerTarget:
    """A peer entry in the host's configuration.

    `mode="outbound"` (default): host connects to `address:port` (LAN or tunneled peer).
    `mode="inbound"`: peer connects outbound to the host's `listen_port` (for NAT/Kaggle/WAN peers).
    """

    name: str
    address: str | None = None
    port: int | None = None
    mode: str = "outbound"


@dataclass(frozen=True)
class GenerationConfig:
    """Sampling and generation limits for the host."""

    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass(frozen=True)
class HostConfig:
    """Validated configuration for the host machine."""

    model: str
    listen_port: int
    peers: list[PeerTarget]
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    auth_token: str | None = None
    routing: str = "auto"  # "auto", "direct", or "host_relay"
    peer_wait_timeout: float = 300.0


@dataclass(frozen=True)
class PeerConfig:
    """Validated configuration for a peer machine."""

    name: str
    listen_port: int
    model: str
    device: str = "auto"
    layers: tuple[int, int] | None = None
    host_address: str | None = None
    host_port: int | None = None
    auth_token: str | None = None


def _read_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration file must contain a YAML mapping: {config_path}")
    return data


def load_host_config(path: str | Path) -> HostConfig:
    """Load and validate a host YAML configuration file."""
    data = _read_yaml(path)

    model = data.get("model")
    if not model or not isinstance(model, str):
        raise ValueError("Host config must specify a non-empty 'model' path.")

    listen_port = int(data.get("listen_port", 5000))
    auth_token = data.get("auth_token")
    if auth_token is not None:
        auth_token = str(auth_token)

    routing = str(data.get("routing", "auto"))
    if routing not in ("auto", "direct", "host_relay"):
        raise ValueError(f"Invalid routing '{routing}'. Use 'auto', 'direct', or 'host_relay'.")

    peer_wait_timeout = float(data.get("peer_wait_timeout", 300.0))

    raw_peers = data.get("peers")
    if not isinstance(raw_peers, list) or len(raw_peers) == 0:
        raise ValueError("Host config must specify at least one peer under 'peers'.")

    peers: list[PeerTarget] = []
    seen_names: set[str] = set()
    for idx, entry in enumerate(raw_peers):
        if not isinstance(entry, dict):
            raise ValueError(f"Peer entry #{idx + 1} must be a mapping.")
        name = entry.get("name")
        if not name or not isinstance(name, str):
            raise ValueError(f"Peer entry #{idx + 1} is missing a valid 'name'.")
        if name in seen_names:
            raise ValueError(f"Duplicate peer name in host config: '{name}'.")
        seen_names.add(name)

        mode = str(entry.get("mode", "outbound"))
        if mode not in ("outbound", "inbound"):
            raise ValueError(f"Peer '{name}' has invalid mode '{mode}'. Use 'outbound' or 'inbound'.")

        address = entry.get("address")
        port = entry.get("port")
        if mode == "outbound":
            if not address or not isinstance(address, str):
                raise ValueError(f"Outbound peer '{name}' is missing a valid 'address'.")
            if port is None:
                raise ValueError(f"Outbound peer '{name}' is missing 'port'.")
            peers.append(PeerTarget(name=name, address=address, port=int(port), mode=mode))
        else:
            peers.append(
                PeerTarget(
                    name=name,
                    address=str(address) if address else None,
                    port=int(port) if port is not None else None,
                    mode=mode,
                )
            )

    raw_gen = data.get("generation") or {}
    if not isinstance(raw_gen, dict):
        raise ValueError("'generation' in host config must be a mapping.")
    generation = GenerationConfig(
        max_tokens=int(raw_gen.get("max_tokens", 512)),
        temperature=float(raw_gen.get("temperature", 0.7)),
        top_p=float(raw_gen.get("top_p", 0.9)),
    )
    if generation.max_tokens <= 0:
        raise ValueError("generation.max_tokens must be positive.")
    if generation.temperature < 0:
        raise ValueError("generation.temperature must be >= 0.")
    if not (0.0 < generation.top_p <= 1.0):
        raise ValueError("generation.top_p must be in (0, 1].")

    return HostConfig(
        model=model,
        listen_port=listen_port,
        peers=peers,
        generation=generation,
        auth_token=auth_token,
        routing=routing,
        peer_wait_timeout=peer_wait_timeout,
    )


def load_peer_config(path: str | Path) -> PeerConfig:
    """Load and validate a peer YAML configuration file."""
    data = _read_yaml(path)

    name = data.get("name")
    if not name or not isinstance(name, str):
        raise ValueError("Peer config must specify a non-empty 'name'.")

    listen_port = int(data.get("listen_port", 5001))
    model = data.get("model")
    if not model or not isinstance(model, str):
        raise ValueError("Peer config must specify a non-empty 'model' path.")

    device = str(data.get("device", "auto"))
    if device not in ("auto", "cpu", "cuda") and not device.startswith("cuda:"):
        raise ValueError(f"Unsupported device '{device}'. Use 'auto', 'cpu', or 'cuda[:idx]'.")

    raw_layers = data.get("layers")
    layers: tuple[int, int] | None = None
    if raw_layers is not None:
        if (
            not isinstance(raw_layers, (list, tuple))
            or len(raw_layers) != 2
            or not all(isinstance(x, int) for x in raw_layers)
        ):
            raise ValueError("Peer config 'layers' must be a two-integer range [start, end].")
        start, end = raw_layers[0], raw_layers[1]
        if start < 0 or end < start:
            raise ValueError(f"Invalid layer range [{start}, {end}].")
        layers = (start, end)

    host_address = data.get("host_address")
    host_port = data.get("host_port")
    if (host_address is None) != (host_port is None):
        raise ValueError("Specify both 'host_address' and 'host_port' for outbound peer mode.")

    auth_token = data.get("auth_token")
    if auth_token is not None:
        auth_token = str(auth_token)

    return PeerConfig(
        name=name,
        listen_port=listen_port,
        model=model,
        device=device,
        layers=layers,
        host_address=str(host_address) if host_address else None,
        host_port=int(host_port) if host_port is not None else None,
        auth_token=auth_token,
    )
