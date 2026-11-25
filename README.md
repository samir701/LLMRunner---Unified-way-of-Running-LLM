# Local LLM Runner (V1)

Run a large language model that is too big for one machine by splitting its layers across several computers on your own local network (LAN).

## How it works
- **Host** (`llmrun host`): Coordinates the session, holds the first layer shard (plus tokenizer and sampler), and streams output to the terminal.
- **Peers** (`llmrun peer`): Listen on the LAN, load their assigned layer shards, run forward passes on incoming activations, and forward to the next machine (or return logits to the host).

## Installation

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

## Quick Start

1. Copy `config/peer.example.yaml` to `config/peer.yaml` on each peer machine and edit `name`, `listen_port`, and `model`.
2. Copy `config/host.example.yaml` to `config/host.yaml` on the host machine and list your peers' LAN IPs and ports.
3. Start peers first:
   ```bash
   python -m src.cli peer --config config/peer.yaml
   ```
4. Start the host:
   ```bash
   python -m src.cli host --config config/host.yaml
   ```
