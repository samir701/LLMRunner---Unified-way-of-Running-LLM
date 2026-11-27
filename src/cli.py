"""CLI entry point for Local LLM Runner (`llmrun host`, `llmrun peer`, `llmrun status`)."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from src.common.config import load_host_config, load_peer_config
from src.common.logging_setup import setup_logging

app = typer.Typer(
    name="llmrun",
    help="Local LLM Runner V2 - split an LLM by layers across LAN and global machines.",
    no_args_is_help=True,
)


@app.command()
def ui() -> None:
    """Launch the V2 Desktop Application (Host + Peer in one window)."""
    from src.gui.app import main as launch_gui

    launch_gui()


@app.command()
def peer(
    config: Path = typer.Option(
        Path("config/peer.yaml"),
        "--config",
        "-c",
        help="Path to peer YAML config file.",
    ),
) -> None:
    """Start this machine in peer (worker) mode and listen for the host."""
    setup_logging("peer")
    peer_cfg = load_peer_config(config)

    from src.peer.server import run_peer_server

    run_peer_server(peer_cfg)


@app.command()
def host(
    config: Path = typer.Option(
        Path("config/host.yaml"),
        "--config",
        "-c",
        help="Path to host YAML config file.",
    ),
) -> None:
    """Start this machine in host mode: connect peers, assign shards, and open chat."""
    setup_logging("host")
    console = Console()
    host_cfg = load_host_config(config)

    from src.host.coordinator import Coordinator
    from src.host.repl import run_repl

    coordinator = Coordinator(host_cfg)
    try:
        console.print("[bold]Connecting to peers and loading shards...[/bold]")
        coordinator.startup()
    except Exception as exc:
        console.print(f"[bold red]Startup failed:[/bold red] {exc}")
        coordinator.shutdown(reason="startup_failed")
        raise typer.Exit(code=1)

    run_repl(coordinator)


@app.command()
def status(
    config: Path = typer.Option(
        Path("config/host.yaml"),
        "--config",
        "-c",
        help="Path to host YAML config file.",
    ),
) -> None:
    """Check which configured peers are reachable and report their free memory."""
    setup_logging("status")
    console = Console()
    host_cfg = load_host_config(config)

    from src.host.coordinator import Coordinator

    coordinator = Coordinator(host_cfg)
    statuses = coordinator.probe_status()

    table = Table(title="Peer Status")
    table.add_column("Peer", style="bold cyan")
    table.add_column("Address")
    table.add_column("Reachable")
    table.add_column("Device")
    table.add_column("Free Memory")
    table.add_column("Detail")

    all_ok = True
    for s in statuses:
        if s.reachable:
            mem_gib = f"{s.free_memory_bytes / (1024**3):.2f} GiB"
            table.add_row(
                s.name,
                f"{s.address}:{s.port}",
                "[green]YES[/green]",
                s.device_type,
                mem_gib,
                "-",
            )
        else:
            all_ok = False
            table.add_row(
                s.name,
                f"{s.address}:{s.port}",
                "[red]NO[/red]",
                "-",
                "-",
                s.error,
            )

    console.print(table)
    if not all_ok:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
