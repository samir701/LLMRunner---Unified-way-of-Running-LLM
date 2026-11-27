"""Interactive terminal chat loop and streaming output for the host."""

from __future__ import annotations

import time

from rich.console import Console
from rich.table import Table

from src.host.coordinator import Coordinator
from src.host.session import ChatSession


def _fmt_mem(num_bytes: int) -> str:
    return f"{num_bytes / (1024**3):.2f} GiB"


def print_pipeline_table(console: Console, coordinator: Coordinator) -> None:
    """Display the assigned layer shards, hardware devices, and memory across nodes."""
    routing_label = "host-relayed" if coordinator.use_relay else "direct"
    table = Table(
        title=f"Local LLM Runner Pipeline ({coordinator.total_layers} layers | routing: {routing_label})"
    )
    table.add_column("Node", style="bold cyan")
    table.add_column("Role")
    table.add_column("Device / Free Mem")
    table.add_column("Connection")
    table.add_column("RTT")
    table.add_column("Layers", style="green")

    host_dev = coordinator.host_device_name or coordinator.host_device_type
    table.add_row(
        coordinator.host_name,
        "Host (embed + first shard)",
        f"{host_dev} ({_fmt_mem(coordinator.host_free_memory_bytes)})",
        "local",
        "0ms",
        str(coordinator.host_range),
    )
    for peer in coordinator.peers:
        role = "Last Peer (+ output head)" if peer.is_last else "Peer"
        dev_label = peer.device_name or peer.device_type
        conn_desc = (
            f"inbound (:{coordinator.config.listen_port})"
            if peer.target.mode == "inbound"
            else f"{peer.target.address}:{peer.target.port}"
        )
        rtt_str = f"{peer.rtt_ms:.0f}ms" if peer.rtt_ms > 0 else "-"
        table.add_row(
            peer.target.name,
            role,
            f"{dev_label} ({_fmt_mem(peer.free_memory_bytes)})",
            conn_desc,
            rtt_str,
            str(peer.layer_range),
        )
    console.print(table)


def run_repl(coordinator: Coordinator) -> None:
    """Run the V1 terminal chat loop until `/exit` or Ctrl+C."""
    console = Console()
    print_pipeline_table(console, coordinator)
    console.print("[dim]Type a prompt and press Enter. Commands: /clear (reset chat), /exit (quit).[/dim]\n")

    session = ChatSession(coordinator, coordinator.config.generation)

    try:
        while True:
            try:
                prompt = console.input("[bold green]You > [/bold green]").strip()
            except EOFError:
                break

            if not prompt:
                coordinator.raise_if_unhealthy()
                continue
            if prompt == "/exit":
                break
            if prompt == "/clear":
                session.clear_history()
                session.clear_caches()
                console.print("[dim]Conversation history cleared.[/dim]\n")
                continue

            coordinator.raise_if_unhealthy()

            console.print("[bold blue]Assistant > [/bold blue]", end="")
            t0 = time.perf_counter()
            t_first: float | None = None
            chunks = 0
            for chunk in session.generate(prompt):
                if t_first is None:
                    t_first = time.perf_counter()
                chunks += 1
                console.print(chunk, end="", markup=False, highlight=False)
            elapsed = max(1e-6, time.perf_counter() - t0)
            ttft_part = f" | TTFT {t_first - t0:.2f}s" if t_first is not None else ""
            stats = session.last_stats
            n = max(1, stats.steps)
            peer_part = f", peer {stats.peer_ms / n:.0f}ms" if stats.peer_ms is not None else ""
            warn = " [yellow](update peer src/ for 67B TOKEN mode!)[/yellow]" if stats.rx_mode == "LOGITS" else ""
            console.print(
                f"\n[dim]({chunks} steps in {elapsed:.1f}s | {chunks / elapsed:.2f} tok/s{ttft_part} | "
                f"avg/step: host {stats.host_ms / n:.0f}ms{peer_part}, tunnel {stats.tunnel_ms / n:.0f}ms | "
                f"rx {stats.rx_mode} {stats.rx_payload_bytes // n}B)[/dim]{warn}\n"
            )
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted. Shutting down pipeline...[/yellow]")
    finally:
        session.clear_caches()
        coordinator.shutdown()
        console.print("[dim]Session ended cleanly.[/dim]")
