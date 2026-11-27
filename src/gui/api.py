"""Python JS-API bridge for the Local LLM Runner V2 Desktop Application."""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import yaml

from src.common.config import (
    DEFAULT_SYSTEM_PROMPT,
    GenerationConfig,
    HostConfig,
    PeerConfig,
    PeerTarget,
    load_host_config,
    load_peer_config,
)
from src.gui.setup_wizard import (
    check_environment,
    install_python,
    install_requirements,
    install_tailscale,
    mark_setup_complete,
    tailscale_enable_funnel,
    tailscale_login,
)
from src.host.coordinator import Coordinator
from src.host.session import ChatSession
from src.model.loader import get_device_name, get_free_memory_bytes, resolve_device
from src.peer.server import run_peer_server


def _fmt_mem(num_bytes: int) -> str:
    return f"{num_bytes / (1024**3):.2f} GiB"


def _host_config_to_dict(cfg: HostConfig) -> dict[str, Any]:
    return {
        "model": cfg.model,
        "listen_port": cfg.listen_port,
        "auth_token": cfg.auth_token or "",
        "routing": cfg.routing,
        "peer_wait_timeout": int(cfg.peer_wait_timeout),
        "peers": [
            {
                "name": p.name,
                "mode": p.mode,
                "address": p.address or "",
                "port": p.port or 5001,
            }
            for p in cfg.peers
        ],
        "generation": {
            "max_tokens": cfg.generation.max_tokens,
            "temperature": cfg.generation.temperature,
            "top_p": cfg.generation.top_p,
            "top_k": cfg.generation.top_k,
            "repetition_penalty": cfg.generation.repetition_penalty,
            "history_turns": cfg.generation.history_turns,
            "system_prompt": cfg.generation.system_prompt,
        },
    }


def _peer_config_to_dict(cfg: PeerConfig) -> dict[str, Any]:
    return {
        "name": cfg.name,
        "listen_port": cfg.listen_port,
        "model": cfg.model,
        "device": cfg.device,
        "layer_start": "" if cfg.layers is None else str(cfg.layers[0]),
        "layer_end": "" if cfg.layers is None else str(cfg.layers[1]),
        "conn_mode": "wan" if (cfg.host_address and cfg.host_port) else "lan",
        "host_address": cfg.host_address or "",
        "host_port": cfg.host_port or 443,
        "host_transport": cfg.host_transport,
        "socks5_proxy": cfg.socks5_proxy or "",
        "auth_token": cfg.auth_token or "",
    }


def build_host_config_from_dict(data: dict[str, Any]) -> HostConfig:
    """Validate and build a HostConfig from UI form data."""
    raw_peers = data.get("peers") or []
    if not raw_peers:
        raise ValueError("Add at least one peer before starting or probing.")

    peers: list[PeerTarget] = []
    seen: set[str] = set()
    for entry in raw_peers:
        name = str(entry.get("name", "")).strip()
        mode = str(entry.get("mode", "outbound")).strip()
        if not name:
            raise ValueError("Every peer must have a name.")
        if name in seen:
            raise ValueError(f"Duplicate peer name: '{name}'.")
        seen.add(name)
        if mode == "outbound":
            addr = str(entry.get("address", "")).strip()
            if not addr:
                raise ValueError(f"Outbound peer '{name}' requires an IP or hostname.")
            peers.append(
                PeerTarget(
                    name=name,
                    mode="outbound",
                    address=addr,
                    port=int(entry.get("port") or 5001),
                )
            )
        else:
            peers.append(PeerTarget(name=name, mode="inbound"))

    g = data.get("generation") or {}
    gen = GenerationConfig(
        max_tokens=int(g.get("max_tokens", 512)),
        temperature=float(g.get("temperature", 0.5)),
        top_p=float(g.get("top_p", 0.9)),
        top_k=int(g.get("top_k", 40)),
        repetition_penalty=float(g.get("repetition_penalty", 1.1)),
        history_turns=int(g.get("history_turns", 6)),
        system_prompt=str(g.get("system_prompt", DEFAULT_SYSTEM_PROMPT)).strip(),
    )
    if gen.max_tokens <= 0:
        raise ValueError("Max tokens must be positive.")
    if gen.temperature < 0:
        raise ValueError("Temperature must be >= 0.")
    if not (0.0 < gen.top_p <= 1.0):
        raise ValueError("Top-P must be in (0, 1].")
    if gen.top_k < 0:
        raise ValueError("Top-K must be >= 0.")
    if gen.repetition_penalty < 1.0:
        raise ValueError("Repetition penalty must be >= 1.0.")

    model = str(data.get("model", "")).strip()
    if not model:
        raise ValueError("Model folder cannot be empty.")

    return HostConfig(
        model=model,
        listen_port=int(data.get("listen_port", 5000)),
        peers=peers,
        generation=gen,
        auth_token=str(data.get("auth_token", "")).strip() or None,
        routing=str(data.get("routing", "auto")).strip(),
        peer_wait_timeout=float(data.get("peer_wait_timeout", 300)),
    )


def build_peer_config_from_dict(data: dict[str, Any]) -> PeerConfig:
    """Validate and build a PeerConfig from UI form data."""
    name = str(data.get("name", "")).strip()
    model = str(data.get("model", "")).strip()
    if not name:
        raise ValueError("Peer name cannot be empty.")
    if not model:
        raise ValueError("Model folder cannot be empty.")

    ls = str(data.get("layer_start", "")).strip()
    le = str(data.get("layer_end", "")).strip()
    layers: tuple[int, int] | None = None
    if ls or le:
        if not (ls and le):
            raise ValueError("Specify both start and end for fixed layers, or leave both blank.")
        layers = (int(ls), int(le))
        if layers[0] < 0 or layers[1] < layers[0]:
            raise ValueError("Invalid fixed layer range.")

    is_wan = str(data.get("conn_mode", "wan")) == "wan"
    host_addr = str(data.get("host_address", "")).strip() if is_wan else None
    host_port = int(data.get("host_port", 443)) if is_wan else None
    if is_wan and not host_addr:
        raise ValueError("Host address is required in Global / WAN Connect mode.")

    return PeerConfig(
        name=name,
        listen_port=int(data.get("listen_port", 5001)),
        model=model,
        device=str(data.get("device", "auto")).strip(),
        layers=layers,
        host_address=host_addr,
        host_port=host_port,
        host_transport=str(data.get("host_transport", "wss")).strip(),
        auth_token=str(data.get("auth_token", "")).strip() or None,
        socks5_proxy=str(data.get("socks5_proxy", "")).strip() or None,
    )


class UILogHandler(logging.Handler):
    """Forwards log lines to the DesktopAPI event stream."""

    def __init__(self, emit_cb: Callable[[str], None]) -> None:
        super().__init__()
        self._emit_cb = emit_cb
        self.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._emit_cb(self.format(record))
        except Exception:
            pass


class DesktopAPI:
    """Thread-safe backend API exposed to the pywebview frontend."""

    def __init__(self, on_event: Callable[[str, Any], None] | None = None) -> None:
        self._window: Any = None
        self._on_event = on_event

        self._coordinator: Coordinator | None = None
        self._session: ChatSession | None = None
        self._host_running = False
        self._generating = False
        self._startup_cancel = threading.Event()
        self._gen_cancel = threading.Event()
        self._health_stop = threading.Event()

        self._peer_running = False
        self._peer_stop = threading.Event()

    def attach_window(self, window: Any) -> None:
        self._window = window

    def _emit(self, event: str, data: Any) -> None:
        if self._on_event is not None:
            self._on_event(event, data)
        if self._window is not None:
            payload = json.dumps({"event": event, "data": data})
            try:
                self._window.evaluate_js(
                    f"window.__onBackendEvent && window.__onBackendEvent({payload});"
                )
            except Exception:
                pass

    def emit_log(self, line: str) -> None:
        self._emit("log_line", {"line": line})

    # ------------------------------------------------------------------
    # Initial State & Config Management
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_default_yaml(primary: str, fallback: str) -> Path:
        p = Path(primary)
        if p.exists():
            return p
        fb = Path(fallback)
        if fb.exists():
            return fb
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            bundled = Path(meipass) / fallback
            if bundled.exists():
                return bundled
        return fb

    def get_initial_state(self) -> dict[str, Any]:
        host_path = self._resolve_default_yaml("config/host.yaml", "config/host.example.yaml")
        try:
            host_cfg = _host_config_to_dict(load_host_config(host_path))
        except Exception:
            host_cfg = {
                "model": "models/Qwen2.5-0.5B-Instruct",
                "listen_port": 5000,
                "auth_token": "",
                "routing": "auto",
                "peer_wait_timeout": 300,
                "peers": [{"name": "kaggle-1", "mode": "inbound", "address": "", "port": 5001}],
                "generation": {
                    "max_tokens": 512,
                    "temperature": 0.5,
                    "top_p": 0.9,
                    "top_k": 40,
                    "repetition_penalty": 1.1,
                    "history_turns": 6,
                    "system_prompt": DEFAULT_SYSTEM_PROMPT,
                },
            }

        peer_path = self._resolve_default_yaml("config/peer.yaml", "config/peer.example.yaml")
        try:
            peer_cfg = _peer_config_to_dict(load_peer_config(peer_path))
        except Exception:
            peer_cfg = {
                "name": "peer-1",
                "listen_port": 5001,
                "model": "models/Qwen2.5-0.5B-Instruct",
                "device": "auto",
                "layer_start": "",
                "layer_end": "",
                "conn_mode": "wan",
                "host_address": "",
                "host_port": 443,
                "host_transport": "wss",
                "socks5_proxy": "",
                "auth_token": "",
            }

        return {
            "host_path": "config/host.yaml",
            "host_config": host_cfg,
            "peer_path": "config/peer.yaml",
            "peer_config": peer_cfg,
            "hardware": self.detect_hardware("auto"),
            "setup": check_environment(),
        }

    # ------------------------------------------------------------------
    # First-Run Installation & Setup Wizard
    # ------------------------------------------------------------------

    def get_setup_status(self) -> dict[str, Any]:
        return check_environment()

    def run_setup_action(self, action: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        opts = payload or {}

        def emit_line(line: str) -> None:
            self._emit("setup_log", {"line": line})

        def worker() -> None:
            try:
                if action == "install_python":
                    install_python(emit_line)
                elif action == "install_requirements":
                    install_requirements(emit_line)
                elif action == "install_tailscale":
                    install_tailscale(emit_line)
                elif action == "tailscale_login":
                    tailscale_login(emit_line)
                elif action == "tailscale_funnel":
                    port = int(opts.get("port", 5000))
                    tailscale_enable_funnel(port, emit_line)
                else:
                    raise ValueError(f"Unknown setup action: {action}")
                self._emit("setup_action_done", {"action": action, "ok": True, "setup": check_environment()})
            except Exception as exc:
                emit_line(f"ERROR: {exc}")
                self._emit(
                    "setup_action_done",
                    {"action": action, "ok": False, "error": str(exc), "setup": check_environment()},
                )

        threading.Thread(target=worker, daemon=True).start()
        return {"ok": True}

    def complete_setup(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        opts = options or {}
        mark_setup_complete()
        role = str(opts.get("role", "host"))
        model = str(opts.get("model", "")).strip()
        auth_token = str(opts.get("auth_token", "")).strip()
        host_address = str(opts.get("host_address", "")).strip()

        # Apply starter model / token / Tailscale address into host.yaml or peer.yaml if requested.
        if model or auth_token:
            host_path = self._resolve_default_yaml("config/host.yaml", "config/host.example.yaml")
            try:
                h_data = _host_config_to_dict(load_host_config(host_path))
                if model:
                    h_data["model"] = model
                if auth_token:
                    h_data["auth_token"] = auth_token
                self.save_host_yaml("config/host.yaml", h_data)
            except Exception:
                pass

            peer_path = self._resolve_default_yaml("config/peer.yaml", "config/peer.example.yaml")
            try:
                p_data = _peer_config_to_dict(load_peer_config(peer_path))
                if model:
                    p_data["model"] = model
                if auth_token:
                    p_data["auth_token"] = auth_token
                if host_address:
                    p_data["conn_mode"] = "wan"
                    p_data["host_address"] = host_address
                    p_data["host_port"] = 443
                    p_data["host_transport"] = "wss"
                self.save_peer_yaml("config/peer.yaml", p_data)
            except Exception:
                pass

        return {"ok": True, "role": role, "state": self.get_initial_state()}

    def detect_hardware(self, device_pref: str = "auto") -> dict[str, Any]:
        try:
            dev = resolve_device(device_pref)
            name = get_device_name(dev)
            free_bytes = get_free_memory_bytes(dev)
            return {
                "ok": True,
                "type": dev.type,
                "name": name,
                "free_mem": _fmt_mem(free_bytes),
                "free_bytes": free_bytes,
            }
        except Exception as exc:
            return {"ok": False, "type": "unknown", "name": str(exc), "free_mem": "-"}

    def browse_yaml(self) -> str | None:
        if self._window is None:
            return None
        import webview

        res = self._window.create_file_dialog(
            webview.FileDialog.OPEN,
            file_types=("YAML files (*.yaml;*.yml)", "All files (*.*)"),
        )
        return str(res[0]) if res else None

    def browse_folder(self) -> str | None:
        if self._window is None:
            return None
        import webview

        res = self._window.create_file_dialog(webview.FileDialog.FOLDER)
        return str(res[0]) if res else None

    def load_host_yaml(self, path: str) -> dict[str, Any]:
        cfg = load_host_config(Path(path.strip()))
        return {"ok": True, "config": _host_config_to_dict(cfg)}

    def save_host_yaml(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        cfg = build_host_config_from_dict(data)
        out_path = Path(path.strip())
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "model": cfg.model,
            "listen_port": cfg.listen_port,
            "routing": cfg.routing,
            "peer_wait_timeout": int(cfg.peer_wait_timeout),
            "peers": [
                (
                    {"name": p.name, "mode": "outbound", "address": p.address, "port": p.port}
                    if p.mode == "outbound"
                    else {"name": p.name, "mode": "inbound"}
                )
                for p in cfg.peers
            ],
            "generation": {
                "max_tokens": cfg.generation.max_tokens,
                "temperature": cfg.generation.temperature,
                "top_p": cfg.generation.top_p,
                "top_k": cfg.generation.top_k,
                "repetition_penalty": cfg.generation.repetition_penalty,
                "history_turns": cfg.generation.history_turns,
                "system_prompt": cfg.generation.system_prompt,
            },
        }
        if cfg.auth_token:
            payload["auth_token"] = cfg.auth_token
        out_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        self.emit_log(f"Saved host config to {out_path}")
        return {"ok": True, "path": str(out_path)}

    def load_peer_yaml(self, path: str) -> dict[str, Any]:
        cfg = load_peer_config(Path(path.strip()))
        return {"ok": True, "config": _peer_config_to_dict(cfg)}

    def save_peer_yaml(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        cfg = build_peer_config_from_dict(data)
        out_path = Path(path.strip())
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "name": cfg.name,
            "listen_port": cfg.listen_port,
            "model": cfg.model,
            "device": cfg.device,
        }
        if cfg.layers is not None:
            payload["layers"] = list(cfg.layers)
        if cfg.host_address and cfg.host_port:
            payload["host_address"] = cfg.host_address
            payload["host_port"] = cfg.host_port
            payload["host_transport"] = cfg.host_transport
        if cfg.socks5_proxy:
            payload["socks5_proxy"] = cfg.socks5_proxy
        if cfg.auth_token:
            payload["auth_token"] = cfg.auth_token
        out_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        self.emit_log(f"Saved peer config to {out_path}")
        return {"ok": True, "path": str(out_path)}

    def download_model(self, repo_id: str, target_role: str = "host") -> dict[str, Any]:
        clean = repo_id.strip()
        if not clean:
            raise ValueError("Enter a Hugging Face repository ID, e.g. Qwen/Qwen2.5-0.5B-Instruct")

        def worker() -> None:
            self._emit("download_state", {"status": "downloading", "repo_id": clean})
            try:
                from huggingface_hub import snapshot_download

                target = Path("models") / clean.split("/")[-1]
                target.mkdir(parents=True, exist_ok=True)
                snapshot_download(
                    repo_id=clean,
                    local_dir=str(target),
                    allow_patterns=["*.json", "*.safetensors", "tokenizer.*", "*.model", "*.txt"],
                )
                self.emit_log(f"Downloaded model '{clean}' to {target}")
                self._emit(
                    "download_state",
                    {
                        "status": "done",
                        "repo_id": clean,
                        "path": target.as_posix(),
                        "target_role": target_role,
                    },
                )
            except Exception as exc:
                self._emit(
                    "download_state",
                    {"status": "error", "repo_id": clean, "error": str(exc)},
                )

        threading.Thread(target=worker, daemon=True).start()
        return {"ok": True}

    # ------------------------------------------------------------------
    # Host Operations
    # ------------------------------------------------------------------

    def probe_peers(self, data: dict[str, Any]) -> dict[str, Any]:
        cfg = build_host_config_from_dict(data)

        def worker() -> None:
            coord = Coordinator(cfg)
            statuses = coord.probe_status()
            rows = [
                {
                    "name": s.name,
                    "address": f"{s.address}:{s.port}" if ":" not in s.address else s.address,
                    "reachable": s.reachable,
                    "device": s.device_type,
                    "free_mem": _fmt_mem(s.free_memory_bytes) if s.free_memory_bytes > 0 else "-",
                    "detail": s.error or "Ready for connection",
                }
                for s in statuses
            ]
            self._emit("probe_results", {"rows": rows})

        threading.Thread(target=worker, daemon=True).start()
        return {"ok": True}

    def start_host(self, data: dict[str, Any]) -> dict[str, Any]:
        if self._host_running:
            return {"ok": False, "error": "Host is already running."}
        cfg = build_host_config_from_dict(data)
        if not Path(cfg.model).exists():
            raise FileNotFoundError(
                f"Model folder '{cfg.model}' not found. Click '+ Download HF' or '...' to choose a downloaded model."
            )

        self._host_running = True
        self._startup_cancel.clear()
        self._health_stop.clear()
        self._emit("host_status", {"state": "STARTING", "detail": "Initializing coordinator..."})

        coord = Coordinator(cfg)
        self._coordinator = coord

        def worker() -> None:
            try:
                coord.startup(
                    on_status=lambda msg: self._emit("host_status", {"state": "STARTING", "detail": msg}),
                    cancel_event=self._startup_cancel,
                )
                session = ChatSession(coord, cfg.generation)
                self._session = session
                pipeline = self._serialize_pipeline(coord)
                self._emit("pipeline_update", pipeline)
                self._emit(
                    "host_status",
                    {
                        "state": "READY",
                        "detail": (
                            f"{coord.total_layers} layers across {1 + len(coord.peers)} nodes "
                            f"({'host-relayed' if coord.use_relay else 'direct'})"
                        ),
                    },
                )
                self.emit_log("Host pipeline is READY.")
                self._run_health_watch(coord)
            except Exception as exc:
                coord.shutdown(reason="startup_failed")
                self._host_running = False
                self._coordinator = None
                self._session = None
                self._emit("host_status", {"state": "ERROR", "detail": str(exc)})
                self.emit_log(f"Host startup error: {exc}")

        threading.Thread(target=worker, daemon=True).start()
        return {"ok": True}

    def _serialize_pipeline(self, coord: Coordinator) -> dict[str, Any]:
        nodes = [
            {
                "name": coord.host_name,
                "role": "Host (embed + first shard)",
                "device": coord.host_device_name or coord.host_device_type,
                "device_type": coord.host_device_type,
                "free_mem": _fmt_mem(coord.host_free_memory_bytes),
                "connection": "local",
                "rtt": "0 ms",
                "layers": str(coord.host_range),
                "layer_count": coord.host_range.count if coord.host_range else 0,
                "is_host": True,
            }
        ]
        for p in coord.peers:
            nodes.append(
                {
                    "name": p.target.name,
                    "role": "Last Peer (+ output head)" if p.is_last else "Worker Peer",
                    "device": p.device_name or p.device_type,
                    "device_type": p.device_type,
                    "free_mem": _fmt_mem(p.free_memory_bytes),
                    "connection": (
                        f"inbound (:{coord.config.listen_port})"
                        if p.target.mode == "inbound"
                        else f"{p.target.address}:{p.target.port}"
                    ),
                    "rtt": f"{p.rtt_ms:.0f} ms" if p.rtt_ms > 0 else "-",
                    "layers": str(p.layer_range),
                    "layer_count": p.layer_range.count if p.layer_range else 0,
                    "is_host": False,
                }
            )
        return {
            "total_layers": coord.total_layers,
            "routing": "host-relayed" if coord.use_relay else "direct",
            "nodes": nodes,
        }

    def _run_health_watch(self, coord: Coordinator) -> None:
        while not self._health_stop.wait(2.0):
            if not self._host_running:
                break
            try:
                coord.raise_if_unhealthy()
            except Exception as exc:
                self.emit_log(f"Heartbeat error: {exc}")
                self._emit("host_status", {"state": "ERROR", "detail": str(exc)})
                self.stop_host()
                break

    def stop_host(self, silent: bool = False) -> dict[str, Any]:
        self._startup_cancel.set()
        self._gen_cancel.set()
        self._health_stop.set()
        coord = self._coordinator
        sess = self._session
        self._host_running = False
        self._generating = False
        self._coordinator = None
        self._session = None

        def cleanup() -> None:
            if sess is not None:
                sess.clear_caches()
            if coord is not None:
                coord.shutdown()

        if coord is not None or sess is not None:
            threading.Thread(target=cleanup, daemon=True).start()

        if not silent:
            self._emit("host_status", {"state": "STOPPED", "detail": "Host session stopped."})
        return {"ok": True}

    def send_prompt(self, prompt: str) -> dict[str, Any]:
        text = prompt.strip()
        if not text:
            return {"ok": False, "error": "Prompt is empty."}
        if not self._host_running or self._session is None:
            return {"ok": False, "error": "Start the Host pipeline first."}
        if self._generating:
            return {"ok": False, "error": "Generation already in progress."}

        self._generating = True
        self._gen_cancel.clear()
        session = self._session

        def worker() -> None:
            t0 = time.perf_counter()
            t_first: float | None = None
            chunks = 0
            try:
                for chunk in session.generate(text, cancel_event=self._gen_cancel):
                    if t_first is None:
                        t_first = time.perf_counter()
                    chunks += 1
                    self._emit("chat_chunk", {"text": chunk})

                elapsed = max(1e-6, time.perf_counter() - t0)
                stats = session.last_stats
                n = max(1, stats.steps)
                self._emit(
                    "chat_done",
                    {
                        "steps": chunks,
                        "elapsed_s": round(elapsed, 2),
                        "tok_s": round(chunks / elapsed, 2),
                        "ttft_s": round(t_first - t0, 2) if t_first is not None else 0.0,
                        "host_ms": round(stats.host_ms / n, 1),
                        "peer_ms": round(stats.peer_ms / n, 1) if stats.peer_ms is not None else None,
                        "tunnel_ms": round(stats.tunnel_ms / n, 1),
                        "rx_mode": stats.rx_mode,
                        "rx_bytes": stats.rx_payload_bytes // n,
                    },
                )
            except Exception as exc:
                self._emit("chat_error", {"error": str(exc)})
                self.emit_log(f"Generation error: {exc}")
            finally:
                self._generating = False

        threading.Thread(target=worker, daemon=True).start()
        return {"ok": True}

    def stop_generation(self) -> dict[str, Any]:
        self._gen_cancel.set()
        return {"ok": True}

    def clear_chat(self) -> dict[str, Any]:
        if self._session is not None:
            self._session.clear_history()
            self._session.clear_caches()
        return {"ok": True}

    # ------------------------------------------------------------------
    # Peer Operations
    # ------------------------------------------------------------------

    def start_peer(self, data: dict[str, Any]) -> dict[str, Any]:
        if self._peer_running:
            return {"ok": False, "error": "Peer is already running."}
        cfg = build_peer_config_from_dict(data)
        if not Path(cfg.model).exists():
            raise FileNotFoundError(
                f"Model folder '{cfg.model}' not found. Click '+ Download HF' or '...' to choose a downloaded model."
            )

        self._peer_running = True
        self._peer_stop.clear()
        self._emit("peer_status", {"state": "CONNECTING", "detail": "Starting peer..."})

        def on_status(state: str, detail: str) -> None:
            self._emit("peer_status", {"state": state, "detail": detail})

        def worker() -> None:
            try:
                run_peer_server(cfg, on_status=on_status, external_stop=self._peer_stop)
            except Exception as exc:
                self._emit("peer_status", {"state": "ERROR", "detail": str(exc)})
            finally:
                self._peer_running = False
                self._emit("peer_status", {"state": "STOPPED", "detail": "Peer stopped."})

        threading.Thread(target=worker, daemon=True).start()
        return {"ok": True}

    def stop_peer(self) -> dict[str, Any]:
        self._peer_stop.set()
        self._peer_running = False
        return {"ok": True}

    def shutdown(self, silent: bool = True) -> None:
        if silent:
            # Detach window/callbacks first so closing never deadlocks on evaluate_js.
            self._window = None
            self._on_event = None
        self.stop_host(silent=silent)
        self.stop_peer()
