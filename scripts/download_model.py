"""Fetch Hugging Face model files into `models/<model-name>/` for offline/LAN use."""

from __future__ import annotations

from pathlib import Path

import typer
from huggingface_hub import snapshot_download

app = typer.Typer(help="Download a Hugging Face model into the local models/ directory.")


@app.command()
def download(
    repo_id: str = typer.Argument(..., help="Hugging Face repo ID, e.g. Qwen/Qwen2.5-0.5B-Instruct"),
    output_dir: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Target directory under models/. Defaults to models/<repo-name>.",
    ),
) -> None:
    """Download model config, tokenizer, and safetensors weights."""
    target = output_dir or (Path("models") / repo_id.split("/")[-1])
    target.mkdir(parents=True, exist_ok=True)
    typer.echo(f"Downloading '{repo_id}' to '{target}'...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        allow_patterns=["*.json", "*.safetensors", "tokenizer.*", "*.model", "*.txt"],
    )
    typer.echo(f"Done. Model saved to {target}")


if __name__ == "__main__":
    app()
