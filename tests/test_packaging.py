"""Tests for Debian .deb package builder."""

from __future__ import annotations

import gzip
import importlib.util
import io
import tarfile
from pathlib import Path


def _load_build_deb():
    script = Path(__file__).resolve().parent.parent / "packaging" / "build_deb.py"
    spec = importlib.util.spec_from_file_location("local_build_deb", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_deb


def test_build_deb_produces_valid_ar_and_tar_members() -> None:
    build_deb = _load_build_deb()
    deb_path = build_deb()
    assert deb_path.exists()
    raw = deb_path.read_bytes()
    assert raw.startswith(b"!<arch>\n")

    pos = 8
    members: dict[str, bytes] = {}
    while pos < len(raw):
        hdr = raw[pos : pos + 60].decode("ascii")
        name = hdr[:16].strip()
        size = int(hdr[48:58].strip())
        pos += 60
        data = raw[pos : pos + size]
        members[name] = data
        pos += size + (size % 2)

    assert members["debian-binary"] == b"2.0\n"
    assert "control.tar.gz" in members
    assert "data.tar.gz" in members

    with gzip.GzipFile(fileobj=io.BytesIO(members["control.tar.gz"])) as gz:
        with tarfile.open(fileobj=gz) as tar:
            ctrl_names = tar.getnames()
            assert "./control" in ctrl_names
            assert "./postinst" in ctrl_names

    with gzip.GzipFile(fileobj=io.BytesIO(members["data.tar.gz"])) as gz:
        with tarfile.open(fileobj=gz) as tar:
            data_names = tar.getnames()
            assert "./usr/local/bin/llmrun-gui" in data_names
            assert "./usr/local/bin/llmrun" in data_names
            assert "./opt/local-llm-runner/src/gui/web/index.html" in data_names
