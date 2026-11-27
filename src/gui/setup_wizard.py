"""First-run installation & environment setup wizard backend for Local LLM Runner V2."""

from __future__ import annotations

import importlib
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Any, Callable

_URL_RE = re.compile(r"https://[^\s\"'<>]+")
_REQUIRED_PKGS = [
    "torch",
    "transformers",
    "safetensors",
    "typer",
    "yaml",
    "rich",
    "numpy",
    "webview",
    "huggingface_hub",
]


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
    on_url: Callable[[str], None] | None = None,
    timeout: int = 600,
) -> int:
    """Run a command with DEVNULL stdin (safe in console=False EXEs) and stream output."""
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
        creationflags=creationflags,
    )
    assert proc.stdout is not None
    opened_urls: set[str] = set()
    for line in proc.stdout:
        clean = line.rstrip()
        if clean:
            if on_output:
                on_output(clean)
            for match in _URL_RE.findall(clean):
                if ("tailscale.com" in match or "login." in match) and match not in opened_urls:
                    opened_urls.add(match)
                    try:
                        webbrowser.open(match)
                    except Exception:
                        pass
                    if on_url:
                        on_url(match)
    return proc.wait(timeout=timeout)


def check_python_status() -> dict[str, Any]:
    """Check Python runtime, frozen EXE bundle status, and local .venv presence."""
    frozen = bool(getattr(sys, "frozen", False))
    venv_python = (
        Path(".venv/Scripts/python.exe") if os.name == "nt" else Path(".venv/bin/python")
    )
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    sys_python = ""
    sys_ver_str = ""
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    for candidate in ("python", "python3"):
        exe = shutil.which(candidate)
        if not exe:
            continue
        # Ignore Microsoft Store app execution alias stub if Python isn't actually installed.
        if os.name == "nt" and "WindowsApps" in exe:
            continue
        try:
            out = subprocess.check_output(
                [exe, "--version"],
                stdin=subprocess.DEVNULL,
                text=True,
                stderr=subprocess.STDOUT,
                timeout=5,
                creationflags=creationflags,
            ).strip()
            if out.startswith("Python 3."):
                sys_python = exe
                sys_ver_str = out
                break
        except Exception:
            continue

    return {
        "frozen": frozen,
        "bundled_version": py_ver,
        "system_python": sys_python,
        "system_python_version": sys_ver_str,
        "venv_exists": venv_python.exists(),
        "venv_python": str(venv_python) if venv_python.exists() else "",
        "ok": frozen or bool(sys_python),
    }


def _check_venv_packages(venv_python: Path) -> tuple[bool, list[str]]:
    if not venv_python.exists():
        return False, list(_REQUIRED_PKGS)
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    probe = (
        "import importlib, json; "
        f"pkgs = {_REQUIRED_PKGS!r}; "
        "missing = []; "
        "\nfor p in pkgs:\n"
        "  try:\n    importlib.import_module(p)\n"
        "  except Exception:\n    missing.append(p)\n"
        "print(json.dumps(missing))"
    )
    try:
        out = subprocess.check_output(
            [str(venv_python), "-c", probe],
            stdin=subprocess.DEVNULL,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=15,
            creationflags=creationflags,
        ).strip()
        missing = json.loads(out)
        return len(missing) == 0, list(missing)
    except Exception:
        return False, list(_REQUIRED_PKGS)


def check_dependencies_status() -> dict[str, Any]:
    """Check runtime packages in current process (and `.venv` if present)."""
    frozen = bool(getattr(sys, "frozen", False))
    missing: list[str] = []
    versions: dict[str, str] = {}

    for pkg in _REQUIRED_PKGS:
        try:
            mod = importlib.import_module(pkg)
            versions[pkg] = str(getattr(mod, "__version__", "ok"))
        except Exception:
            missing.append(pkg)

    venv_python = (
        Path(".venv/Scripts/python.exe") if os.name == "nt" else Path(".venv/bin/python")
    )
    venv_ok, venv_missing = _check_venv_packages(venv_python)

    req_exists = Path("requirements.txt").exists()
    if not req_exists and getattr(sys, "_MEIPASS", None):
        req_exists = (Path(sys._MEIPASS) / "requirements.txt").exists()  # type: ignore[attr-defined]

    return {
        "ok": len(missing) == 0,
        "frozen": frozen,
        "missing": missing,
        "versions": versions,
        "venv_ok": venv_ok,
        "venv_missing": venv_missing,
        "requirements_exists": req_exists,
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
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

    try:
        out = subprocess.check_output(
            [exe, "status", "--json"],
            stdin=subprocess.DEVNULL,
            text=True,
            stderr=subprocess.STDOUT,
            timeout=6,
            creationflags=creationflags,
        )
        info.update(parse_tailscale_status_json(out))
    except Exception as exc:
        info["backend_state"] = f"Error: {exc}"

    if info["logged_in"]:
        try:
            fout = subprocess.check_output(
                [exe, "funnel", "status"],
                stdin=subprocess.DEVNULL,
                text=True,
                stderr=subprocess.STDOUT,
                timeout=6,
                creationflags=creationflags,
            ).strip()
            info["funnel_detail"] = fout
            info["funnel_active"] = (
                "Funnel on" in fout
                or "https://" in fout
                or "proxy" in fout.lower()
            )
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
        webbrowser.open("https://www.python.org/downloads/")
        raise RuntimeError("winget not found. Opened python.org download page in your browser.")
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
    if not req_file.exists() and getattr(sys, "_MEIPASS", None):
        bundled_req = Path(sys._MEIPASS) / "requirements.txt"  # type: ignore[attr-defined]
        if bundled_req.exists():
            req_file = bundled_req
    if not req_file.exists():
        raise FileNotFoundError("requirements.txt not found in current workspace.")

    py_status = check_python_status()
    py_cmd = py_status["system_python"]
    if not py_cmd:
        if not getattr(sys, "frozen", False):
            py_cmd = sys.executable
        else:
            raise RuntimeError("No system Python 3.11+ found. Install Python in Step 1 first.")

    venv_dir = Path(".venv")
    venv_py = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not venv_py.exists():
        on_output(f"Creating virtual environment (.venv) using {py_cmd}...")
        rc = _run_cmd_stream([py_cmd, "-m", "venv", ".venv"], on_output=on_output)
        if rc != 0:
            raise RuntimeError(f"Failed creating .venv (exit code {rc})")

    on_output("Upgrading pip in .venv...")
    _run_cmd_stream([str(venv_py), "-m", "pip", "install", "--upgrade", "pip"], on_output=on_output)

    on_output(f"Installing packages from {req_file}...")
    rc = _run_cmd_stream(
        [str(venv_py), "-m", "pip", "install", "-r", str(req_file)],
        on_output=on_output,
        timeout=1800,
    )
    if rc != 0:
        raise RuntimeError(f"pip install -r requirements.txt failed (exit code {rc})")

    if Path("pyproject.toml").exists():
        on_output("Registering llmrun & llmrun-gui entry points...")
        _run_cmd_stream([str(venv_py), "-m", "pip", "install", "-e", ".", "--no-deps"], on_output=on_output)


def install_tailscale(on_output: Callable[[str], None]) -> None:
    """Install Tailscale via winget (Windows) or official install script (Linux)."""
    if os.name == "nt":
        winget = shutil.which("winget")
        if not winget:
            webbrowser.open("https://tailscale.com/download/windows")
            raise RuntimeError("winget not found. Opened Tailscale Windows download page.")
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


def tailscale_login(
    on_output: Callable[[str], None],
    on_url: Callable[[str], None] | None = None,
) -> None:
    """Run `tailscale up` (and open Windows Tailscale GUI if needed)."""
    exe = _find_tailscale_exe()
    if not exe:
        raise RuntimeError("Tailscale is not installed yet.")

    if os.name == "nt":
        gui_exe = Path(exe).parent / "tailscale-ipn.exe"
        if gui_exe.exists():
            try:
                subprocess.Popen(
                    [str(gui_exe)],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass

    on_output("Running `tailscale up` (if a login link appears, your browser will open)...")
    rc = _run_cmd_stream([exe, "up"], on_output=on_output, on_url=on_url, timeout=180)
    if rc != 0:
        raise RuntimeError(f"`tailscale up` exited with code {rc}")


def tailscale_enable_funnel(
    port: int,
    on_output: Callable[[str], None],
    on_url: Callable[[str], None] | None = None,
) -> None:
    """Expose `localhost:<port>` via `tailscale funnel --bg <port>` on port 443."""
    exe = _find_tailscale_exe()
    if not exe:
        raise RuntimeError("Tailscale is not installed yet.")
    on_output(f"Enabling Tailscale Funnel for http://127.0.0.1:{port} on port 443...")
    rc = _run_cmd_stream(
        [exe, "funnel", "--bg", str(port)],
        on_output=on_output,
        on_url=on_url,
        timeout=300,
    )
    if rc != 0:
        raise RuntimeError(
            f"`tailscale funnel --bg {port}` exited with code {rc}. "
            "Check that MagicDNS & HTTPS are enabled at https://login.tailscale.com/admin/dns"
        )


def tailscale_disable_funnel(on_output: Callable[[str], None]) -> None:
    """Turn off Tailscale Funnel on port 443."""
    exe = _find_tailscale_exe()
    if not exe:
        raise RuntimeError("Tailscale is not installed.")
    on_output("Disabling Tailscale Funnel (--https=443 off)...")
    rc = _run_cmd_stream([exe, "funnel", "--https=443", "off"], on_output=on_output, timeout=30)
    if rc != 0:
        raise RuntimeError(f"`tailscale funnel --https=443 off` exited with code {rc}")
