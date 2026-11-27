"""Main Desktop Application launcher for Local LLM Runner V2 (pywebview)."""

from __future__ import annotations

import logging
from pathlib import Path

import webview

from src.common.logging_setup import setup_logging
from src.gui.api import DesktopAPI, UILogHandler


def main() -> None:
    """Launch the V2 Desktop GUI window (Host + Peer in one app)."""
    setup_logging("desktop")
    api = DesktopAPI()

    handler = UILogHandler(api.emit_log)
    handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)

    html_path = Path(__file__).resolve().parent / "web" / "index.html"
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
        api.shutdown()

    window.events.closing += _on_closing
    webview.start(debug=False)


if __name__ == "__main__":
    main()
