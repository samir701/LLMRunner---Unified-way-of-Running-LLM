"""Build a standard Debian `.deb` package (`dist/local-llm-runner_<version>_all.deb`).

Works cross-platform (Windows, macOS, and Linux) without requiring `dpkg-deb`
locally by writing a standard GNU `ar` archive containing `debian-binary`,
`control.tar.gz`, and `data.tar.gz`.
"""

from __future__ import annotations

import gzip
import io
import os
import tarfile
import time
from pathlib import Path

PACKAGE_NAME = "local-llm-runner"
VERSION = "0.2.0"
ARCH = "all"
INSTALL_ROOT = "opt/local-llm-runner"

CONTROL_TEXT = f"""Package: {PACKAGE_NAME}
Version: {VERSION}
Section: utils
Priority: optional
Architecture: {ARCH}
Maintainer: Samir <samir701@users.noreply.github.com>
Depends: python3 (>= 3.11), python3-venv, python3-pip
Recommends: python3-gi, gir1.2-webkit2-4.1 | gir1.2-webkit2-4.0
Description: Local LLM Runner V2 (LAN + Global WAN Distributed LLM Pipeline)
 Run a large language model across multiple LAN or global machines by
 splitting transformer layers across peers, with both a modern desktop UI
 (Host + Peer modes) and headless CLI commands.
"""

POSTINST_TEXT = """#!/bin/sh
set -e
APP_DIR="/opt/local-llm-runner"
if [ ! -d "$APP_DIR/.venv" ]; then
    python3 -m venv "$APP_DIR/.venv"
fi
"$APP_DIR/.venv/bin/pip" install --upgrade pip >/dev/null 2>&1 || true
"$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.txt"
"$APP_DIR/.venv/bin/pip" install -e "$APP_DIR" --no-deps
echo "Local LLM Runner installed! Run 'llmrun-gui' for Desktop UI or 'llmrun --help' for CLI."
"""

PRERM_TEXT = """#!/bin/sh
set -e
if [ "$1" = "remove" ] || [ "$1" = "purge" ]; then
    rm -rf /opt/local-llm-runner/.venv /opt/local-llm-runner/__pycache__
fi
"""

LLMRUN_BIN = """#!/bin/sh
cd /opt/local-llm-runner
exec /opt/local-llm-runner/.venv/bin/python -m src.cli "$@"
"""

LLMRUN_GUI_BIN = """#!/bin/sh
cd /opt/local-llm-runner
exec /opt/local-llm-runner/.venv/bin/python -m src.gui.app "$@"
"""

DESKTOP_ENTRY = """[Desktop Entry]
Name=Local LLM Runner V2
Comment=Distributed LLM Runner (Host & Peer Desktop App)
Exec=/usr/local/bin/llmrun-gui
Terminal=false
Type=Application
Categories=Development;Utility;
"""


def _add_bytes_to_tar(
    tar: tarfile.TarFile,
    arcname: str,
    data: bytes,
    mode: int = 0o644,
    mtime: int = 0,
) -> None:
    info = tarfile.TarInfo(name=arcname)
    info.size = len(data)
    info.mode = mode
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    info.mtime = mtime
    tar.addfile(info, io.BytesIO(data))


def _ar_entry(name: str, data: bytes, mtime: int) -> bytes:
    header = (
        f"{name:<16}"
        f"{mtime:<12}"
        f"{0:<6}"
        f"{0:<6}"
        f"{'100644':<8}"
        f"{len(data):<10}"
        f"`\n"
    ).encode("ascii")
    assert len(header) == 60
    pad = b"\n" if len(data) % 2 == 1 else b""
    return header + data + pad


def _should_include(rel_path: Path) -> bool:
    parts = rel_path.parts
    if not parts:
        return False
    if parts[0] in (".git", ".venv", "venv", "dist", "build", ".pytest_cache", "__pycache__", "logs", "models"):
        return False
    if "__pycache__" in parts or rel_path.suffix in (".pyc", ".pyo"):
        return False
    if rel_path.as_posix() in ("config/host.yaml", "config/peer.yaml"):
        return False
    return True


def build_deb(repo_root: Path | None = None) -> Path:
    root = (repo_root or Path(__file__).resolve().parent.parent).resolve()
    dist_dir = root / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    out_path = dist_dir / f"{PACKAGE_NAME}_{VERSION}_{ARCH}.deb"
    now = int(time.time())

    # 1. Build control.tar.gz
    control_buf = io.BytesIO()
    with gzip.GzipFile(fileobj=control_buf, mode="wb", mtime=now) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            _add_bytes_to_tar(tar, "./control", CONTROL_TEXT.encode("utf-8"), 0o644, now)
            _add_bytes_to_tar(tar, "./postinst", POSTINST_TEXT.encode("utf-8"), 0o755, now)
            _add_bytes_to_tar(tar, "./prerm", PRERM_TEXT.encode("utf-8"), 0o755, now)

    # 2. Build data.tar.gz
    data_buf = io.BytesIO()
    with gzip.GzipFile(fileobj=data_buf, mode="wb", mtime=now) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            include_roots = [
                "README.md",
                "requirements.txt",
                "pyproject.toml",
                "config",
                "src",
                "scripts",
            ]
            for item in include_roots:
                target = root / item
                if not target.exists():
                    continue
                if target.is_file():
                    raw = target.read_bytes()
                    _add_bytes_to_tar(tar, f"./{INSTALL_ROOT}/{item}", raw, 0o644, now)
                else:
                    for dirpath, _, filenames in os.walk(target):
                        for fn in filenames:
                            fpath = Path(dirpath) / fn
                            rel = fpath.relative_to(root)
                            if not _should_include(rel):
                                continue
                            mode = 0o755 if fpath.suffix == ".sh" else 0o644
                            _add_bytes_to_tar(
                                tar,
                                f"./{INSTALL_ROOT}/{rel.as_posix()}",
                                fpath.read_bytes(),
                                mode,
                                now,
                            )

            _add_bytes_to_tar(tar, "./usr/local/bin/llmrun", LLMRUN_BIN.encode("utf-8"), 0o755, now)
            _add_bytes_to_tar(
                tar, "./usr/local/bin/llmrun-gui", LLMRUN_GUI_BIN.encode("utf-8"), 0o755, now
            )
            _add_bytes_to_tar(
                tar,
                "./usr/share/applications/local-llm-runner.desktop",
                DESKTOP_ENTRY.encode("utf-8"),
                0o644,
                now,
            )

    # 3. Assemble standard Debian `ar` package
    deb_bytes = (
        b"!<arch>\n"
        + _ar_entry("debian-binary", b"2.0\n", now)
        + _ar_entry("control.tar.gz", control_buf.getvalue(), now)
        + _ar_entry("data.tar.gz", data_buf.getvalue(), now)
    )
    out_path.write_bytes(deb_bytes)
    return out_path


if __name__ == "__main__":
    deb = build_deb()
    print(f"Built Debian package: {deb} ({deb.stat().st_size} bytes)")
