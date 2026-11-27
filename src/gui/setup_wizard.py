"""First-run installation & environment setup wizard backend for Local LLM Runner V2."""

from __future__ import annotations

import importlib
import json
import os
import secrets
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable


def _find_tailscale_exe() -> str | None:
    found = shutil.which("tailscale")
    if found:
        return found
    if os.name == "nt":
        candidates = [
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Tailscale" / "tailscale.exe",
            Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Tailscale" / "tailscale.exe",
        ]
        for c in candidates:
            if c.exists():
                return str(c)
    return None


def _run_cmd_stream(
    cmd: list[str],
    on_output: Callable[[str], None] | None = None,
    timeout: int = 600,
) -> int:
    """Run a command and stream combined stdout/stderr lines via `on_output`."""
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creationflags,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        clean = line.rstrip()
        if clean and on_output:
            on_output(clean)
    return proc.wait(timeout=timeout)


def check_python_status() -> dict[str, Any]:
    """Check Python runtime, frozen EXE bundle status, and local .venv presence."""
    frozen = bool(getattr(sys, "frozen", False))
    sys_python = shutil.which("python") or shutil.which("python3")
    venv_python = (
        Path(".venv/Scripts/python.exe") if os.name == "nt" else Path(".venv/bin/python")
    )
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    sys_ver_str = ""
    if sys_python:
        try:
            out = subprocess.check_output(
                [sys_python, "--version"],
                text=True,
                stderr=subprocess.STDOUT,
                timeout=5,
            ).strip()
            sys_ver_str = out
        except Exception:
            sys_ver_str = "detected"

    return {
        "frozen": frozen,
        "bundled_version": py_ver,
        "system_python": sys_python or "",
        "system_python_version": sys_ver_str,
        "venv_exists": venv_python.exists(),
        "venv_python": str(venv_python) if venv_python.exists() else "",
        "ok": frozen or bool(sys_python),
    }


def check_dependencies_status() -> dict[str, Any]:
    """Check whether core runtime packages from requirements.txt are importable."""
    required = ["torch", "transformers", "safetensors", "typer", "yaml", "rich", "numpy", "webview"]
    missing: list[str] = []
    versions: dict[str, str] = {}

    for pkg in required:
        try:
            mod = importlib.import_module(pkg)
            versions[pkg] = str(getattr(mod, "__version__", "ok"))
        except Exception:
            missing.append(pkg)

    return {
        "ok": len(missing) == 0,
        "missing": missing,
        "versions": versions,
        "requirements_exists": Path("requirements.txt").exists(),
    }


def parse_tailscale_status_json(raw_json: str) -> dict[str, Any]:
    """Parse `tailscale status --json` output into UI-friendly status fields."""
    data = json.loads(raw_json)
    backend_state = str(data.get("BackendState", "Unknown"))
    self_node = data.get("Self") or {}
    dns_name = str(self_node.get("DNSName", "")).rstrip(".")
    ips = self_node.get("TailscaleIPs") or []
    return {
        "backend_state": backend_state,
        "logged_in": backend_state == "Running",
        "dns_name": dns_name,
        "ip": str(ips[0]) if ips else "",
    }


def check_tailscale_status() -> dict[str, Any]:
    """Inspect Tailscale installation, login state, MagicDNS hostname, and Funnel."""
    exe = _find_tailscale_exe()
    if not exe:
        return {
            "installed": False,
            "logged_in": False,
            "backend_state": "NotInstalled",
            "dns_name": "",
            "ip": "",
            "funnel_active": False,
            "funnel_detail": "",
        }

    info: dict[str, Any] = {
        "installed": True,
        "exe": exe,
        "logged_in": False,
        "backend_state": "Unknown",
        "dns_name": "",
        "ip": "",
        "funnel_active": False,
        "funnel_detail": "",
    }

    try:
        out = subprocess.check_output(
            [exe, "status", "--json"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=6,
        )
        info.update(parse_tailscale_status_json(out))
    except Exception as exc:
        info["backend_state"] = f"Error: {exc}"

    if info["logged_in"]:
        try:
            fout = subprocess.check_output(
                [exe, "funnel", "status"],
                text=True,
                stderr=subprocess.STDOUT,
                timeout=6,
            ).strip()
            info["funnel_detail"] = fout
            info["funnel_active"] = "https://" in fout or ":5000" in fout or "proxy" in fout.lower()
        except Exception:
            info["funnel_active"] = False

    return info


def list_local_models() -> list[str]:
    """Return paths to downloaded model folders under `models/`."""
    models_dir = Path("models")
    if not models_dir.exists():
        return []
    found: list[str] = []
    for child in sorted(models_dir.iterdir()):
        if child.is_dir() and (child / "config.json").exists():
            found.append(f"models/{child.name}")
    return found


def is_setup_complete() -> bool:
    return Path("config/.setup_done").exists()


def mark_setup_complete() -> None:
    cfg_dir = Path("config")
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / ".setup_done").write_text("completed\n", encoding="utf-8")


def check_environment() -> dict[str, Any]:
    """Collect full first-run setup status across Python, deps, Tailscale, and models."""
    return {
        "setup_complete": is_setup_complete(),
        "os": os.name,
        "python": check_python_status(),
        "dependencies": check_dependencies_status(),
        "tailscale": check_tailscale_status(),
        "models": list_local_models(),
        "suggested_token": secrets.token_urlsafe(12),
    }


def install_python(on_output: Callable[[str], None]) -> None:
    """Install Python 3.12 using winget on Windows."""
    if os.name != "nt":
        raise RuntimeError(
            "On Linux/macOS, install Python 3.11+ via your package manager: "
            "sudo apt install python3 python3-venv python3-pip"
        )
    winget = shutil.which("winget")
    if not winget:
        raise RuntimeError("winget not found. Install Python 3.12 from https://www.python.org/downloads/")
    on_output("Installing Python 3.12 via winget...")
    rc = _run_cmd_stream(
        [
            winget,
            "install",
            "--id",
            "Python.Python.3.12",
            "-e",
            "--accept-package-agreements",
            "--accept-source-agreements",
        ],
        on_output=on_output,
    )
    if rc != 0:
        raise RuntimeError(f"winget Python install exited with code {rc}")


def install_requirements(on_output: Callable[[str], None]) -> None:
    """Create `.venv` if needed and install `requirements.txt`."""
    req_file = Path("requirements.txt")
    if not req_file.exists():
        raise FileNotFoundError("requirements.txt not found in current workspace.")

    py_cmd = shutil.which("python") or shutil.which("python3")
    if not py_cmd:
        if not getattr(sys, "frozen", False):
            py_cmd = sys.executable
        else:
            raise RuntimeError("No system Python found. Install Python in Step 1 first.")

    venv_dir = Path(".venv")
    venv_py = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not venv_py.exists():
        on_output("Creating virtual environment (.venv)...")
        rc = _run_cmd_stream([py_cmd, "-m", "venv", ".venv"], on_output=on_output)
        if rc != 0:
            raise RuntimeError(f"Failed creating .venv (exit code {rc})")

    on_output("Upgrading pip...")
    _run_cmd_stream([str(venv_py), "-m", "pip", "install", "--upgrade", "pip"], on_output=on_output)

    on_output("Installing packages from requirements.txt...")
    rc = _run_cmd_stream(
        [str(venv_py), "-m", "pip", "install", "-r", str(req_file)],
        on_output=on_output,
        timeout=1800,
    )
    if rc != 0:
        raise RuntimeError(f"pip install -r requirements.txt failed (exit code {rc})")

    if Path("pyproject.toml").exists():
        on_output("Registering llmrun CLI & GUI entry points...")
        _run_cmd_stream([str(venv_py), "-m", "pip", "install", "-e", ".", "--no-deps"], on_output=on_output)


def install_tailscale(on_output: Callable[[str], None]) -> None:
    """Install Tailscale via winget (Windows) or official install script (Linux)."""
    if os.name == "nt":
        winget = shutil.which("winget")
        if not winget:
            raise RuntimeError("winget not found. Download Tailscale from https://tailscale.com/download/windows")
        on_output("Installing Tailscale via winget...")
        rc = _run_cmd_stream(
            [
                winget,
                "install",
                "--id",
                "Tailscale.Tailscale",
                "-e",
                "--accept-package-agreements",
                "--accept-source-agreements",
            ],
            on_output=on_output,
        )
        if rc != 0:
            raise RuntimeError(f"winget Tailscale install exited with code {rc}")
        return

    on_output("Installing Tailscale on Linux...")
    rc = _run_cmd_stream(
        ["sh", "-c", "curl -fsSL https://tailscale.com/install.sh | sh"],
        on_output=on_output,
    )
    if rc != 0:
        raise RuntimeError(f"Tailscale install script failed (exit code {rc})")


def tailscale_login(on_output: Callable[[str], None]) -> None:
    """Run `tailscale up` to sign in and connect this device."""
    exe = _find_tailscale_exe()
    if not exe:
        raise RuntimeError("Tailscale is not installed yet.")
    on_output("Launching Tailscale sign-in (`tailscale up`)...")
    rc = _run_cmd_stream([exe, "up"], on_output=on_output, timeout=180)
    if rc != 0:
        raise RuntimeError(f"`tailscale up` exited with code {rc}")


def tailscale_enable_funnel(port: int, on_output: Callable[[str], None]) -> None:
    """Expose `localhost:<port>` via `tailscale funnel --bg <port>` on port 443."""
    exe = _find_tailscale_exe()
    if not exe:
        raise RuntimeError("Tailscale is not installed yet.")
    on_output(f"Enabling Tailscale Funnel for localhost:{port}...")
    rc = _run_cmd_stream([exe, "funnel", "--bg", str(port)], on_output=on_output, timeout=60)
    if rc != 0:
        raise RuntimeError(
            f"`tailscale funnel --bg {port}` exited with code {rc}. "
            "If prompted with a tailnet URL, open it in your browser to enable Funnel once, then click Enable Funnel again."
        )
