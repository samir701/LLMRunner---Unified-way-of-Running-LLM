"""Interactive terminal chat loop and streaming output for the host."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from src.host.coordinator import Coordinator
from src.host.session import ChatSession


def print_pipeline_table(console: Console, coordinator: Coordinator) -> None:
    """Display the assigned layer shards across host and peers."""
    routing_label = "host-relayed" if coordinator.use_relay else "direct"
    table = Table(
        title=f"Local LLM Runner Pipeline ({coordinator.total_layers} layers | routing: {routing_label})"
    )
    table.add_column("Node", style="bold cyan")
    table.add_column("Role")
    table.add_column("Connection")
    table.add_column("Layers", style="green")

    table.add_row(
        coordinator.host_name,
        "Host (embed + first shard)",
        "local",
        str(coordinator.host_range),
    )
    for peer in coordinator.peers:
        role = "Last Peer (+ output head)" if peer.is_last else "Peer"
        conn_desc = (
            f"inbound (:{coordinator.config.listen_port})"
            if peer.target.mode == "inbound"
            else f"{peer.target.address}:{peer.target.port}"
        )
        table.add_row(
            peer.target.name,
            role,
            conn_desc,
            str(peer.layer_range),
        )
    console.print(table)


def run_repl(coordinator: Coordinator) -> None:
    """Run the V1 terminal chat loop until `/exit` or Ctrl+C."""
    console = Console()
    print_pipeline_table(console, coordinator)
    console.print("[dim]Type a prompt and press Enter. Type /exit to quit.[/dim]\n")

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

            coordinator.raise_if_unhealthy()

            console.print("[bold blue]Assistant > [/bold blue]", end="")
            for chunk in session.generate(prompt):
                console.print(chunk, end="", markup=False, highlight=False)
            console.print("\n")
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted. Shutting down pipeline...[/yellow]")
    finally:
        session.clear_caches()
        coordinator.shutdown()
        console.print("[dim]Session ended cleanly.[/dim]")
