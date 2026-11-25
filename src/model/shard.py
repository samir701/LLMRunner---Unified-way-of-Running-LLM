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
    """Memory capacity and optional fixed layer range for a pipeline node."""

    name: str
    free_memory_bytes: int
    fixed_layers: tuple[int, int] | None = None


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

    If all peers specify `fixed_layers`, the host receives the remaining prefix
    `[0, first_peer_start - 1]` and the full pipeline is validated for contiguous
    coverage. Otherwise, layers are distributed proportionally to each node's
    `free_memory_bytes` (guaranteeing at least 1 layer per node).
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

    # Automatic memory-proportional assignment (Hamilton / largest-remainder method).
    weights = [max(1, n.free_memory_bytes) for n in nodes]
    total_weight = sum(weights)

    # Give every node at least 1 layer first, then distribute the remaining layers.
    remaining = total_layers - len(nodes)
    counts = [1] * len(nodes)
    if remaining > 0:
        exact_shares = [remaining * w / total_weight for w in weights]
        floors = [int(s) for s in exact_shares]
        remainders = [
            (exact_shares[i] - floors[i], i) for i in range(len(nodes))
        ]
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
