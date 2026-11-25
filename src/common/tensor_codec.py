"""Tensor serialization and deserialization using float16 on the wire.

Half precision (float16) halves LAN traffic versus float32 with negligible
quality change for activations and logits.
"""

from __future__ import annotations

import numpy as np
import torch


def encode_tensor(tensor: torch.Tensor) -> tuple[bytes, list[int], str]:
    """Convert a PyTorch tensor to contiguous float16 raw bytes.

    Returns:
        (raw_bytes, shape_list, dtype_str)
    """
    t_f16 = tensor.detach().to(device="cpu", dtype=torch.float16).contiguous()
    arr = t_f16.numpy()
    return arr.tobytes(), list(t_f16.shape), "float16"


def decode_tensor(
    data: bytes,
    shape: list[int] | tuple[int, ...],
    dtype: str = "float16",
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Reconstruct a PyTorch tensor from raw wire bytes and shape metadata."""
    if dtype != "float16":
        raise ValueError(f"Unsupported wire dtype '{dtype}'; V1 expects 'float16'.")
    np_arr = np.frombuffer(data, dtype=np.float16).copy()
    tensor = torch.from_numpy(np_arr).reshape(tuple(shape))
    if str(device) != "cpu":
        tensor = tensor.to(device=device)
    return tensor
