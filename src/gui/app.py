"""Main Desktop Application launcher for Local LLM Runner V2 (pywebview)."""

from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path

import webview

from src.common.logging_setup import setup_logging
from src.gui.api import DesktopAPI, UILogHandler


def _prepare_runtime_workdir() -> None:
    """Ensure CWD has `config/`, `models/`, and `logs/` when launched as a frozen .exe."""
    if not getattr(sys, "frozen", False):
        return

    exe_dir = Path(sys.executable).resolve().parent
    # If running from `dist/LocalLLMRunner.exe` inside the repo, use the repo root.
    if (exe_dir.parent / "src").exists() and (exe_dir.parent / "config").exists():
        os.chdir(exe_dir.parent)
        return

    os.chdir(exe_dir)
    (exe_dir / "config").mkdir(exist_ok=True)
    (exe_dir / "models").mkdir(exist_ok=True)
    (exe_dir / "logs").mkdir(exist_ok=True)

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        bundled_cfg = Path(meipass) / "config"
        for name in ("host.example.yaml", "peer.example.yaml"):
            src_file = bundled_cfg / name
            dst_file = exe_dir / "config" / name
            if src_file.exists() and not dst_file.exists():
                shutil.copy2(src_file, dst_file)

        # Auto-seed starter host.yaml / peer.yaml on first standalone launch.
        if not (exe_dir / "config" / "host.yaml").exists() and (exe_dir / "config" / "host.example.yaml").exists():
            shutil.copy2(exe_dir / "config" / "host.example.yaml", exe_dir / "config" / "host.yaml")
        if not (exe_dir / "config" / "peer.yaml").exists() and (exe_dir / "config" / "peer.example.yaml").exists():
            shutil.copy2(exe_dir / "config" / "peer.example.yaml", exe_dir / "config" / "peer.yaml")


def _resolve_web_index() -> Path:
    """Locate `src/gui/web/index.html` in both source and PyInstaller frozen builds."""
    candidates: list[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / "src" / "gui" / "web" / "index.html")
    candidates.append(Path(__file__).resolve().parent / "web" / "index.html")
    candidates.append(Path(__file__).resolve().parent / "src" / "gui" / "web" / "index.html")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"GUI frontend not found in {candidates}")


def main() -> None:
    """Launch the V2 Desktop GUI window (Host + Peer in one app)."""
    _prepare_runtime_workdir()
    setup_logging("desktop")
    api = DesktopAPI()

    handler = UILogHandler(api.emit_log)
    handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)

    html_path = _resolve_web_index()
    window = webview.create_window(
        title="Local LLM Runner V2",
        url=html_path.as_uri(),
        js_api=api,
        width=1340,
        height=840,
        min_size=(1020, 680),
        background_color="#FFFDF7",
    )
    api.attach_window(window)

    def _on_closing() -> None:
        # Clear window reference BEFORE shutting down so no evaluate_js deadlocks WinForms.
        api.shutdown(silent=True)

    window.events.closing += _on_closing
    try:
        webview.start(debug=False)
    finally:
        api.shutdown(silent=True)
        logging.shutdown()
        if getattr(sys, "frozen", False):
            os._exit(0)


if __name__ == "__main__":
    main()
