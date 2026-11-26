"""Tests for layer-range sharding: full coverage without gaps or overlaps."""

from __future__ import annotations

import pytest

from src.model.shard import (
    LayerRange,
    NodeMemoryInfo,
    compute_shard_assignments,
    validate_layer_ranges,
)


def test_automatic_sharding_covers_all_layers_proportionally() -> None:
    nodes = [
        NodeMemoryInfo("host", free_memory_bytes=8 * 1024**3),
        NodeMemoryInfo("peer-1", free_memory_bytes=16 * 1024**3),
        NodeMemoryInfo("peer-2", free_memory_bytes=8 * 1024**3),
    ]
    ranges = compute_shard_assignments(total_layers=32, nodes=nodes)

    assert len(ranges) == 3
    validate_layer_ranges(ranges, total_layers=32)
    # Peer-1 has 2x the memory of host and peer-2, so it should get the largest shard.
    assert ranges[1].count > ranges[0].count
    assert ranges[1].count > ranges[2].count


def test_fixed_layer_ranges_for_peers() -> None:
    nodes = [
        NodeMemoryInfo("host", free_memory_bytes=8 * 1024**3),
        NodeMemoryInfo("peer-1", free_memory_bytes=8 * 1024**3, fixed_layers=(10, 20)),
        NodeMemoryInfo("peer-2", free_memory_bytes=8 * 1024**3, fixed_layers=(21, 31)),
    ]
    ranges = compute_shard_assignments(total_layers=32, nodes=nodes)

    assert ranges == [LayerRange(0, 9), LayerRange(10, 20), LayerRange(21, 31)]
    validate_layer_ranges(ranges, total_layers=32)


def test_validate_detects_gaps_and_overlaps() -> None:
    with pytest.raises(ValueError, match="gap or overlap"):
        validate_layer_ranges([LayerRange(0, 9), LayerRange(11, 31)], total_layers=32)

    with pytest.raises(ValueError, match="gap or overlap"):
        validate_layer_ranges([LayerRange(0, 10), LayerRange(10, 31)], total_layers=32)

    with pytest.raises(ValueError, match="end at"):
        validate_layer_ranges([LayerRange(0, 9), LayerRange(10, 29)], total_layers=32)


def test_gpu_peers_receive_bulk_of_layers_over_cpu_host() -> None:
    nodes = [
        NodeMemoryInfo("cpu-host", free_memory_bytes=16 * 1024**3, device_type="cpu", device_name="cpu"),
        NodeMemoryInfo("kaggle-t4", free_memory_bytes=15 * 1024**3, device_type="cuda", device_name="Tesla T4"),
        NodeMemoryInfo("home-4090", free_memory_bytes=24 * 1024**3, device_type="cuda", device_name="NVIDIA GeForce RTX 4090"),
    ]
    ranges = compute_shard_assignments(total_layers=24, nodes=nodes)
    validate_layer_ranges(ranges, total_layers=24)

    # CPU host only gets 1 layer so it doesn't bottleneck the GPUs.
    assert ranges[0].count == 1
    # Faster/larger RTX 4090 gets more layers than Tesla T4.
    assert ranges[2].count > ranges[1].count

