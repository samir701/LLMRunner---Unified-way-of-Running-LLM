# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for building LocalLLMRunner.exe (Windows V2 Desktop App)."""

import os
from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))

datas = [
    (os.path.join(ROOT, "src", "gui", "web"), "src/gui/web"),
    (os.path.join(ROOT, "config"), "config"),
]
binaries = []
hiddenimports = [
    "src",
    "src.cli",
    "src.gui",
    "src.gui.app",
    "src.gui.api",
    "src.common",
    "src.common.config",
    "src.common.messages",
    "src.common.tensor_codec",
    "src.common.transport",
    "src.common.logging_setup",
    "src.common.sampler",
    "src.model",
    "src.model.loader",
    "src.model.shard",
    "src.model.engine",
    "src.host",
    "src.host.coordinator",
    "src.host.session",
    "src.host.sampler",
    "src.host.repl",
    "src.peer",
    "src.peer.server",
    "src.peer.worker",
    "yaml",
    "safetensors",
    "safetensors.torch",
    "huggingface_hub",
]

for pkg in ("webview", "clr_loader", "pythonnet", "transformers", "tokenizers"):
    p_datas, p_bins, p_hidden = collect_all(pkg)
    datas += p_datas
    binaries += p_bins
    hiddenimports += p_hidden

hiddenimports += collect_submodules("transformers.models.qwen2")
hiddenimports += collect_submodules("transformers.models.llama")

a = Analysis(
    [os.path.join(ROOT, "src", "gui", "app.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "customtkinter", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="LocalLLMRunner",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
