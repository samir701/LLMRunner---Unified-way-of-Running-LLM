# Local LLM Runner (V2)

Run a large language model that is too big for one machine by splitting its layers across several computers on your local network (LAN) and/or globally over the internet (WAN/Kaggle/Tailscale).

## What's New in V2
- **Unified Desktop Application (`llmrun-gui` / `llmrun ui`)**: One desktop window with dedicated **Host Mode** and **Peer Mode** sections, visual YAML config editor, Hugging Face model downloader, peer status checker, pipeline & RTT table, streaming chat with per-step latency breakdown, and live peer activity logs.
- **Hybrid LAN + Global WAN**: Mix local LAN peers (`outbound`), NAT/Kaggle peers (`inbound` over `wss` port `443`), and SOCKS5 proxies in the same pipeline with `auth_token` verification.
- **GPU-Tier + VRAM-Aware Sharding & Universal Tuning**: Automatically keeps CPU-only hosts at 1 layer while splitting remaining layers across CUDA GPUs by GPU tier and free VRAM, with universal chat-template fallback, `top_k`, `repetition_penalty`, and multi-turn history.

## Installation

```bash
python -m venv .venv
# Windows:
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
pip install -e . --no-deps
```

## Launch the Desktop Application (Host + Peer)

```bash
llmrun-gui
# or:
llmrun ui
# or without activating .venv on Windows:
.\.venv\Scripts\python.exe -m src.gui.app
```

## Headless CLI (for Kaggle / Remote Servers)

1. Copy `config/peer.example.yaml` to `config/peer.yaml` on each peer machine (or `config/host.example.yaml` to `config/host.yaml` on the host).
2. Download the same model on every machine:
   ```bash
   python scripts/download_model.py Qwen/Qwen2.5-0.5B-Instruct
   ```
3. Run in terminal:
   ```bash
   python -m src.cli peer --config config/peer.yaml
   python -m src.cli host --config config/host.yaml
   python -m src.cli status --config config/host.yaml
   ```
