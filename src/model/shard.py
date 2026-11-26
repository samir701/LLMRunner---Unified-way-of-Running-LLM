"""Layer-range calculation: assign contiguous shards across host and peers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerRange:
    """Inclusive 0-based layer range `[start, end]` assigned to one machine."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"Invalid LayerRange({self.start}, {self.end}).")

    @property
    def count(self) -> int:
        return self.end - self.start + 1

    def __str__(self) -> str:
        return f"{self.start}-{self.end}"


@dataclass(frozen=True)
class NodeMemoryInfo:
    """Memory capacity, compute device tier, and optional fixed layers for a node."""

    name: str
    free_memory_bytes: int
    device_type: str = "cpu"
    device_name: str = ""
    fixed_layers: tuple[int, int] | None = None


def gpu_tier_multiplier(device_type: str, device_name: str = "") -> float:
    """Compute a relative throughput multiplier based on device type and GPU model."""
    if device_type != "cuda":
        return 1.0

    upper = device_name.upper()
    if any(tag in upper for tag in ("H100", "H200", "B100", "B200")):
        return 3.0
    if any(tag in upper for tag in ("A100", "RTX 5090", "RTX 4090", "L40")):
        return 2.5
    if any(tag in upper for tag in ("L4", "A10", "RTX 40", "RTX 3090", "RTX 3080")):
        return 2.0
    if any(tag in upper for tag in ("V100", "RTX 30", "RTX 20", "A4000", "T4", "P100")):
        return 1.3
    return 1.2


def validate_layer_ranges(ranges: list[LayerRange], total_layers: int) -> None:
    """Verify that `ranges` cover `0..total_layers-1` contiguously with no gaps or overlaps."""
    if total_layers <= 0:
        raise ValueError(f"total_layers must be positive, got {total_layers}.")
    if not ranges:
        raise ValueError("At least one layer range is required.")

    expected_start = 0
    for idx, r in enumerate(ranges):
        if r.start != expected_start:
            raise ValueError(
                f"Layer range #{idx} starts at {r.start}, expected {expected_start} "
                "(gap or overlap detected)."
            )
        expected_start = r.end + 1

    if expected_start != total_layers:
        raise ValueError(
            f"Layer ranges end at {expected_start - 1}, expected {total_layers - 1}."
        )


def compute_shard_assignments(
    total_layers: int,
    nodes: list[NodeMemoryInfo],
) -> list[LayerRange]:
    """Compute contiguous layer ranges for `[host, peer_1, ..., peer_n]`.

    - If peers specify `fixed_layers`, validates full contiguous coverage.
    - If GPU (`cuda`) and CPU nodes are mixed, CPU nodes receive 1 layer minimum so
      they don't bottleneck inference, and remaining layers are split across GPU nodes
      proportionally to `(free_VRAM * gpu_tier_multiplier)`.
    - If all nodes are the same class (all CPU or all GPU), splits proportionally to
      `free_memory_bytes * gpu_tier_multiplier`.
    """
    if total_layers <= 0:
        raise ValueError(f"total_layers must be positive, got {total_layers}.")
    if len(nodes) < 2:
        raise ValueError("Pipeline requires at least 2 nodes (1 host + 1 peer).")
    if total_layers < len(nodes):
        raise ValueError(
            f"Model has only {total_layers} layers, cannot split across {len(nodes)} nodes."
        )

    # Check if peers specified fixed layer ranges.
    peer_nodes = nodes[1:]
    fixed_flags = [n.fixed_layers is not None for n in peer_nodes]
    if any(fixed_flags):
        if not all(fixed_flags):
            raise ValueError(
                "Either all peers must specify fixed 'layers' in config, or none should."
            )
        peer_ranges = [LayerRange(n.fixed_layers[0], n.fixed_layers[1]) for n in peer_nodes]  # type: ignore[index]
        host_end = peer_ranges[0].start - 1
        if host_end < 0:
            raise ValueError(
                f"First peer starts at layer {peer_ranges[0].start}; host needs at least layer 0."
            )
        all_ranges = [LayerRange(0, host_end), *peer_ranges]
        validate_layer_ranges(all_ranges, total_layers)
        return all_ranges

    has_gpu = any(n.device_type == "cuda" for n in nodes)
    has_cpu = any(n.device_type != "cuda" for n in nodes)

    # Give every node at least 1 layer first.
    remaining = total_layers - len(nodes)
    counts = [1] * len(nodes)

    if remaining > 0:
        if has_gpu and has_cpu:
            # Keep CPU-only nodes at 1 layer so GPU peers handle the heavy compute.
            weights = [
                max(1.0, n.free_memory_bytes * gpu_tier_multiplier(n.device_type, n.device_name))
                if n.device_type == "cuda"
                else 0.0
                for n in nodes
            ]
        else:
            weights = [
                max(1.0, n.free_memory_bytes * gpu_tier_multiplier(n.device_type, n.device_name))
                for n in nodes
            ]

        total_weight = sum(weights)
        exact_shares = [remaining * w / total_weight for w in weights]
        floors = [int(s) for s in exact_shares]
        remainders = [(exact_shares[i] - floors[i], i) for i in range(len(nodes)) if weights[i] > 0]
        for i in range(len(nodes)):
            counts[i] += floors[i]
        leftover = remaining - sum(floors)
        remainders.sort(key=lambda x: (-x[0], x[1]))
        for _, idx in remainders[:leftover]:
            counts[idx] += 1

    ranges: list[LayerRange] = []
    cursor = 0
    for count in counts:
        ranges.append(LayerRange(cursor, cursor + count - 1))
        cursor += count

    validate_layer_ranges(ranges, total_layers)
    return ranges
