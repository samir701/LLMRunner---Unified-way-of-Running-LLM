"""Model loader: reads config/manifest and loads ONLY this machine's assigned layer shard.

Only this module knows where model files live on disk. Everything else asks
it for a LoadedShard or tokenizer.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from src.model.shard import LayerRange


@dataclass
class LoadedShard:
    """Holds only the model components belonging to this machine's layer range."""

    model_dir: Path
    config: Any
    layer_range: LayerRange
    is_first: bool
    is_last: bool
    device: torch.device
    dtype: torch.dtype
    embed_tokens: nn.Module | None
    rotary_emb: nn.Module | None
    layers: nn.ModuleDict
    norm: nn.Module | None
    lm_head: nn.Module | None


def resolve_device(preference: str = "auto") -> torch.device:
    """Select CUDA GPU if available and requested, otherwise CPU."""
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preference.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device '{preference}', but CUDA is not available.")
    return torch.device(preference)


def get_free_memory_bytes(device: torch.device) -> int:
    """Return available VRAM (for CUDA) or system RAM (for CPU) in bytes."""
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        free_bytes, _ = torch.cuda.mem_get_info(idx)
        return int(free_bytes)

    # Cross-platform system RAM check without requiring extra dependencies.
    if os.name == "nt":
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return int(stat.ullAvailPhys)

    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                kb = int(line.split()[1])
                return kb * 1024

    # Conservative fallback (4 GiB) if OS memory query is unavailable.
    return 4 * 1024 * 1024 * 1024


def read_model_config(model_dir: str | Path) -> Any:
    """Read `config.json` from the local model directory without loading weights."""
    path = Path(model_dir)
    if not path.exists():
        raise FileNotFoundError(f"Model directory not found: {path}")
    return AutoConfig.from_pretrained(str(path), local_files_only=True)


def get_num_layers(model_config: Any) -> int:
    """Extract total transformer layer count from a Hugging Face config."""
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        val = getattr(model_config, attr, None)
        if isinstance(val, int) and val > 0:
            return val
    raise ValueError("Could not determine layer count from model config.json.")


def load_tokenizer(model_dir: str | Path) -> Any:
    """Load the model's tokenizer (used by the host only)."""
    return AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)


def _find_safetensors_files(model_dir: Path) -> dict[str, Path]:
    """Return mapping of tensor_name -> safetensors file path."""
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index_data = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index_data.get("weight_map", {})
        return {k: model_dir / v for k, v in weight_map.items()}

    single_path = model_dir / "model.safetensors"
    if single_path.exists():
        with safe_open(str(single_path), framework="pt", device="cpu") as f:
            return {k: single_path for k in f.keys()}

    raise FileNotFoundError(
        f"No model.safetensors or model.safetensors.index.json found in {model_dir}."
    )


def _needed_prefixes(layer_range: LayerRange, is_first: bool, is_last: bool) -> tuple[str, ...]:
    prefixes: list[str] = [
        f"model.layers.{i}." for i in range(layer_range.start, layer_range.end + 1)
    ]
    if is_first:
        prefixes.append("model.embed_tokens.")
    if is_last:
        prefixes.extend(["model.norm.", "lm_head."])
    return tuple(prefixes)


def _load_filtered_weights(
    weight_map: dict[str, Path],
    prefixes: tuple[str, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Load only tensors matching `prefixes` from safetensors files."""
    by_file: dict[Path, list[str]] = {}
    for key, file_path in weight_map.items():
        if key.startswith(prefixes):
            by_file.setdefault(file_path, []).append(key)

    loaded: dict[str, torch.Tensor] = {}
    target_device = str(device)
    for file_path, keys in by_file.items():
        with safe_open(str(file_path), framework="pt", device="cpu") as f:
            for key in keys:
                loaded[key] = f.get_tensor(key).to(dtype=dtype, device=target_device)
    return loaded


def load_model_shard(
    model_dir: str | Path,
    layer_range: LayerRange,
    is_first: bool,
    is_last: bool,
    device_pref: str = "auto",
) -> LoadedShard:
    """Load only the assigned layer shard into GPU/RAM and discard the rest."""
    path = Path(model_dir)
    device = resolve_device(device_pref)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    config = read_model_config(path)
    weight_map = _find_safetensors_files(path)
    prefixes = _needed_prefixes(layer_range, is_first=is_first, is_last=is_last)
    weights = _load_filtered_weights(weight_map, prefixes, dtype=dtype, device=device)

    # Build module skeleton on meta device so unused layers consume zero RAM/VRAM.
    with torch.device("meta"):
        skeleton = AutoModelForCausalLM.from_config(config)

    inner_model = getattr(skeleton, "model", skeleton)
    all_layers = inner_model.layers

    shard_layers = nn.ModuleDict()
    for idx in range(layer_range.start, layer_range.end + 1):
        layer_mod = all_layers[idx].to_empty(device=device)
        prefix = f"model.layers.{idx}."
        layer_state = {
            k[len(prefix) :]: v for k, v in weights.items() if k.startswith(prefix)
        }
        layer_mod.load_state_dict(layer_state, strict=False, assign=True)
        layer_mod.eval()
        shard_layers[str(idx)] = layer_mod

    embed_tokens: nn.Module | None = None
    if is_first:
        embed_tokens = inner_model.embed_tokens.to_empty(device=device)
        embed_state = {
            k[len("model.embed_tokens.") :]: v
            for k, v in weights.items()
            if k.startswith("model.embed_tokens.")
        }
        embed_tokens.load_state_dict(embed_state, strict=False, assign=True)
        embed_tokens.eval()

    rotary_emb: nn.Module | None = getattr(inner_model, "rotary_emb", None)
    if rotary_emb is not None:
        rotary_emb = rotary_emb.to_empty(device=device)
        # Re-initialize buffers (e.g. inv_freq) on target device.
        if hasattr(rotary_emb, "inv_freq"):
            rotary_emb = type(rotary_emb)(config=config, device=device)
        rotary_emb.eval()

    norm: nn.Module | None = None
    lm_head: nn.Module | None = None
    if is_last:
        norm = inner_model.norm.to_empty(device=device)
        norm_state = {
            k[len("model.norm.") :]: v
            for k, v in weights.items()
            if k.startswith("model.norm.")
        }
        norm.load_state_dict(norm_state, strict=False, assign=True)
        norm.eval()

        lm_head = skeleton.lm_head.to_empty(device=device)
        lm_state = {
            k[len("lm_head.") :]: v
            for k, v in weights.items()
            if k.startswith("lm_head.")
        }
        # Handle tied embeddings if lm_head.weight was not stored separately.
        if not lm_state and "model.embed_tokens.weight" in weight_map:
            tied = _load_filtered_weights(
                weight_map, ("model.embed_tokens.",), dtype=dtype, device=device
            )
            if "model.embed_tokens.weight" in tied:
                lm_state["weight"] = tied["model.embed_tokens.weight"]
        lm_head.load_state_dict(lm_state, strict=False, assign=True)
        lm_head.eval()

    return LoadedShard(
        model_dir=path,
        config=config,
        layer_range=layer_range,
        is_first=is_first,
        is_last=is_last,
        device=device,
        dtype=dtype,
        embed_tokens=embed_tokens,
        rotary_emb=rotary_emb,
        layers=shard_layers,
        norm=norm,
        lm_head=lm_head,
    )
