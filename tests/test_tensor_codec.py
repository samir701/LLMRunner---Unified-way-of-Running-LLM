"""Tests for float16 tensor serialization and deserialization round-trips."""

from __future__ import annotations

import torch

from src.common.tensor_codec import decode_tensor, encode_tensor


def test_tensor_round_trip_preserves_shape_and_values() -> None:
    original = torch.randn(1, 5, 64, dtype=torch.float16)
    raw_bytes, shape, dtype = encode_tensor(original)

    assert dtype == "float16"
    assert shape == [1, 5, 64]

    restored = decode_tensor(raw_bytes, shape=shape, dtype=dtype, device="cpu")
    assert restored.shape == original.shape
    assert restored.dtype == torch.float16
    assert torch.equal(original, restored)
