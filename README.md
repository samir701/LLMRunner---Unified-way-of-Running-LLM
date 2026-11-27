# Local LLM Runner (V2)

Run a large language model that is too big for one machine by splitting its transformer layers across multiple computers—combining **local LAN machines** (Windows / Linux) and **global WAN peers** (Kaggle GPUs, remote friends' PCs) in a single pipeline.

> **Cross-Version Compatible:** The V2 Desktop App (`llmrun-gui` / `LocalLLMRunner.exe`) and the headless CLI (`llmrun` / `python -m src.cli`) share the exact same protocol, so a Desktop Host works seamlessly with Kaggle CLI peers and Linux/Windows CLI peers.

---

## Table of Contents
1. [How the Pipeline Works](#1-how-the-pipeline-works)
2. [Desktop App Setup (Windows `.exe` & Linux `.deb`)](#2-desktop-app-setup-windows-exe--linux-deb)
   - [Installing & Running on Windows (`.exe`)](#21-installing--running-on-windows-exe)
   - [Installing & Running on Linux (`.deb`)](#22-installing--running-on-linux-deb)
   - [First-Run Setup Wizard](#23-first-run-setup-wizard)
   - [Setting Up the Desktop App as Host (Windows or Linux)](#24-setting-up-the-desktop-app-as-host-windows-or-linux)
   - [Setting Up the Desktop App as Peer (Windows or Linux)](#25-setting-up-the-desktop-app-as-peer-windows-or-linux)
3. [Kaggle Notebook Peer Setup (Free Cloud GPU)](#3-kaggle-notebook-peer-setup-free-cloud-gpu)
4. [Full CLI Setup from Source (Windows & Linux)](#4-full-cli-setup-from-source-windows--linux)
   - [4.1 Clone & Install (Windows & Linux)](#41-clone--install-windows--linux)
   - [4.2 CLI as Host (Windows & Linux)](#42-cli-as-host-windows--linux)
   - [4.3 CLI as Peer (Windows & Linux)](#43-cli-as-peer-windows--linux)
5. [Global Networking (Tailscale Funnel) & Troubleshooting](#5-global-networking-tailscale-funnel--troubleshooting)
6. [Building Installers from Source](#6-building-installers-from-source)

---

## 1. How the Pipeline Works

- **Host** (`Host Mode`): Coordinates startup, holds embeddings + the first shard (only 1 layer if CPU-only while GPU peers are present), samples tokens, and streams responses.
- **Peers** (`Peer Mode`): Each peer holds a contiguous slice of layers based on GPU tier and free VRAM/RAM, runs forward passes, and returns activations or next-token IDs.
- **Every machine must have the same model folder** under `models/<model-name>/` on disk (each machine only loads its assigned layers into RAM/VRAM).

---

## 2. Desktop App Setup (Windows `.exe` & Linux `.deb`)

### 2.1 Installing & Running on Windows (`.exe`)
1. Download `LocalLLMRunner.exe` (from GitHub Actions artifacts or `dist\LocalLLMRunner.exe` if built locally).
2. Double-click **`LocalLLMRunner.exe`** (allow 20–30 seconds on first launch while the bundled PyTorch runtime unpacks).
3. On first launch, `config/`, `models/`, and `logs/` are created automatically alongside the executable (or in the repo root if launched from `dist\`).

### 2.2 Installing & Running on Linux (`.deb`)
1. Download `local-llm-runner_0.2.0_all.deb` (from GitHub Actions artifacts or `dist/`).
2. Install on Ubuntu / Debian:
   ```bash
   sudo apt update
   sudo apt install ./local-llm-runner_0.2.0_all.deb
   ```
   *(This installs the app to `/opt/local-llm-runner`, sets up `.venv`, installs dependencies, and registers `llmrun-gui` and `llmrun`.)*
3. Launch the Desktop App from your Applications menu (**Local LLM Runner V2**) or terminal:
   ```bash
   llmrun-gui
   ```

### 2.3 First-Run Setup Wizard
The Setup Wizard opens automatically before the main workspace (and can be reopened anytime via the **Setup Wizard** button in the top bar):
1. **Step 1 — Python Runtime:** Verifies the bundled `.exe` runtime or system Python 3.11+ (with a 1-click **Install Python 3.12** button on Windows via `winget`).
2. **Step 2 — Packages (`requirements.txt`):** Shows bundled runtime packages and `.venv` status, with a 1-click **Create `.venv` & Install `requirements.txt`** button for CLI/dev usage.
3. **Step 3 — Tailscale & Funnel (for Global / Kaggle Peers):**
   - Click **1. Install Tailscale** (uses `winget` on Windows or Tailscale's install script on Linux).
   - Click **2. Sign In (`tailscale up`)** to connect your account.
   - Click **3. Enable Funnel (`:5000`)** to expose your Host on HTTPS/WSS port `443` and copy your permanent `*.ts.net` hostname.
4. **Step 4 — Model & Role Config:** Download a starter Hugging Face model (e.g. `Qwen/Qwen2.5-0.5B-Instruct`), choose **Host Mode** or **Peer Mode**, generate a shared `auth_token`, and click **Save & Launch App →**.

### 2.4 Setting Up the Desktop App as Host (Windows or Linux)
1. Switch the top toggle to **Host Mode**.
2. In the left sidebar:
   - **Model Folder:** Pick an existing folder under `models/` or click **+ Download HF** (e.g. `Qwen/Qwen2.5-0.5B-Instruct`).
   - **Listen Port:** Keep `5000`.
   - **Routing:** Keep `auto (smart)` (automatically uses `host_relay` when any peer is `inbound`, or `direct` for pure LAN).
   - **Auth Token:** Enter a shared secret (required if accepting remote peers over Tailscale Funnel).
   - **Pipeline Peers:** Click **+ Add Peer** for each peer joining:
     - For **Kaggle / remote friend behind NAT**: choose `inbound` (e.g. `kaggle-1`, `kaggle-2`, `friend-pc`).
     - For a **local LAN machine**: choose `outbound` and enter its LAN IP and port (`5001`).
   - Click **Save** to write `config/host.yaml`.
3. If accepting remote/Kaggle peers, ensure **Tailscale Funnel** is active on port `5000` (via Step 3 of the Setup Wizard or `tailscale funnel --bg 5000`).
4. Click **Start Host**, wait for peers to connect and load their shards, then chat in the **Chat** tab (or view node splits & RTT in **Pipeline Topology**).

### 2.5 Setting Up the Desktop App as Peer (Windows or Linux)
1. Switch the top toggle to **Peer Mode**.
2. Set **Peer Name** (must exactly match its name in the Host's `peers:` list, e.g. `friend-pc` or `local-laptop`).
3. Select the **same Model Folder** (or click **+ Download HF** to download the identical model).
4. Set **Compute Device** (`auto` uses CUDA GPU if available, otherwise CPU) and leave **Fixed Start / End** blank for automatic sharding.
5. Choose **Connection Mode**:
   - **Global / WAN Connect** *(recommended for remote friends behind NAT)*:
     - **Host Address:** Host's `*.ts.net` hostname (e.g. `desktop-abc123.tailnet.ts.net`)
     - **Host Port:** `443`
     - **Transport:** `wss (Port 443 TLS)`
     - **Auth Token:** Same secret as the Host's `host.yaml`
   - **LAN Listen** *(when Host connects directly to this machine on local Wi-Fi/Ethernet)*:
     - **Listen Port:** `5001` (allow TCP port `5001` in your firewall)
     - **Auth Token:** Same secret as the Host's `host.yaml`
6. Click **Save**, then click **Start Peer**.

---

## 3. Kaggle Notebook Peer Setup (Free Cloud GPU)

Kaggle notebooks have no display, so they run the headless CLI peer (`python -m src.cli peer`) and connect outbound to your Host over port `443` (`wss`).

1. In Kaggle **Notebook → Session options**:
   - **Accelerator:** Select **GPU T4 x2** or **GPU P100**
   - **Internet:** Toggle **ON**
2. Run these cells inside Kaggle:

```python
# Cell 1: Clone repository
!git clone https://github.com/samir701/llmRunner-.git /kaggle/working/local-llm-runner
%cd /kaggle/working/local-llm-runner
```

```python
# Cell 2: Install Python dependencies (keep Kaggle's preinstalled CUDA PyTorch)
!pip install -q "transformers>=4.45.0,<5.0.0" "safetensors>=0.4.2,<1.0.0" "typer>=0.12.0,<1.0.0" "pyyaml>=6.0.1,<7.0.0" "rich>=13.7.0,<14.0.0"
```

```python
# Cell 3: Download the same model as your Host
!python scripts/download_model.py Qwen/Qwen2.5-0.5B-Instruct
```

```python
# Cell 4: Write config/peer.yaml and start the peer (start Host on your PC first!)
peer_yaml = """
name: "kaggle-1"                              # use "kaggle-2" on a second notebook
model: "models/Qwen2.5-0.5B-Instruct"
device: "auto"
host_address: "your-pc.your-tailnet.ts.net"   # your Host's Tailscale Funnel hostname
host_port: 443
host_transport: "wss"
auth_token: "choose-a-strong-secret"          # must match host.yaml
"""

with open("config/peer.yaml", "w", encoding="utf-8") as f:
    f.write(peer_yaml.strip() + "\n")

!python -m src.cli peer --config config/peer.yaml
```

---

## 4. Full CLI Setup from Source (Windows & Linux)

### 4.1 Clone & Install (Windows & Linux)
Requires **Python 3.11+**.

**Windows (PowerShell):**
```powershell
git clone https://github.com/samir701/llmRunner-.git
cd llmRunner-
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps
python scripts/download_model.py Qwen/Qwen2.5-0.5B-Instruct
```
*(If PowerShell blocks `Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or invoke `.\.venv\Scripts\python.exe -m src.cli ...` directly.)*

**Linux / macOS (Bash):**
```bash
git clone https://github.com/samir701/llmRunner-.git
cd llmRunner-
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps
python scripts/download_model.py Qwen/Qwen2.5-0.5B-Instruct
```

### 4.2 CLI as Host (Windows & Linux)
1. Copy `config/host.example.yaml` to `config/host.yaml` and edit it:
   ```yaml
   model: "models/Qwen2.5-0.5B-Instruct"
   listen_port: 5000
   auth_token: "choose-a-strong-secret"
   routing: "auto"
   peer_wait_timeout: 300

   peers:
     - name: "local-laptop"
       mode: "outbound"
       address: "192.168.1.21"
       port: 5001

     - name: "kaggle-1"
       mode: "inbound"

   generation:
     max_tokens: 512
     temperature: 0.5
     top_p: 0.9
     top_k: 40
     repetition_penalty: 1.1
     history_turns: 6
   ```
2. If you have `inbound` (remote/Kaggle) peers, enable Tailscale Funnel on port `5000`:
   ```bash
   tailscale funnel --bg 5000
   ```
3. Check outbound LAN peers (optional) and start the Host:
   ```bash
   llmrun status --config config/host.yaml
   llmrun host --config config/host.yaml
   ```
   *(Or `python -m src.cli host --config config/host.yaml`. Inside the chat prompt, use `/clear` to reset conversation history and `/exit` to stop cleanly.)*

### 4.3 CLI as Peer (Windows & Linux)
1. Copy `config/peer.example.yaml` to `config/peer.yaml`.
2. **For a LAN Peer** (Host connects to this peer):
   ```yaml
   name: "local-laptop"
   listen_port: 5001
   model: "models/Qwen2.5-0.5B-Instruct"
   device: "auto"
   auth_token: "choose-a-strong-secret"
   # layers: [10, 20]   # optional 0-based inclusive [start, end]; omit for auto-split
   ```
3. **For a Global / WAN Peer** (Peer connects outbound to Host via Tailscale Funnel):
   ```yaml
   name: "friend-pc"
   model: "models/Qwen2.5-0.5B-Instruct"
   device: "auto"
   host_address: "your-pc.your-tailnet.ts.net"
   host_port: 443
   host_transport: "wss"
   auth_token: "choose-a-strong-secret"
   ```
4. Start the Peer:
   ```bash
   llmrun peer --config config/peer.yaml
   # or: python -m src.cli peer --config config/peer.yaml
   ```

*(To launch the Desktop UI from CLI anytime: `llmrun ui` or `llmrun-gui`.)*

---

## 5. Global Networking (Tailscale Funnel) & Troubleshooting

- **One-time Tailscale setup on the Host PC:**
  1. Enable **MagicDNS** and **HTTPS Certificates** in `https://login.tailscale.com/admin/dns`.
  2. Run `tailscale funnel --bg 5000` (if prompted with an approval URL, open it once in your browser).
  3. Check status with `tailscale funnel status`.
- **`[SSL: UNEXPECTED_EOF_WHILE_READING]` in Kaggle:** Tailscale Funnel is not serving port `443` on the Host PC. Run `tailscale funnel status`—if it says `No serve config`, enable it with `tailscale funnel --bg 5000`. If it says `foreground listener already exists for port 443`, close the foreground `tailscale` window or run `tailscale funnel reset` first, then `tailscale funnel --bg 5000`.
- **`Host failed auth_token verification`:** Ensure `auth_token` in `config/host.yaml` is uncommented and matches `config/peer.yaml` character-for-character.
- **PowerShell `llmrun` not recognized:** Activate `.venv` first (`.\.venv\Scripts\Activate.ps1`) or run `.\.venv\Scripts\python.exe -m src.cli ...`.
- **Fixed layer ranges:** Always use **0-based inclusive** indices (e.g., for a 24-layer model: Host `0-0`, Peer 1 `1-12`, Peer 2 `13-23`).

---

## 6. Building Installers from Source

- **Windows `.exe`** (`dist/LocalLLMRunner.exe`):
  ```powershell
  pip install "pyinstaller>=6.0"
  .\packaging\build_windows.ps1
  ```
- **Linux `.deb`** (`dist/local-llm-runner_0.2.0_all.deb`):
  ```bash
  python packaging/build_deb.py
  ```
- **Run test suite:**
  ```bash
  pytest -v
  ```
