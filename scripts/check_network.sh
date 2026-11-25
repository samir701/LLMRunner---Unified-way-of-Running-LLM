#!/usr/bin/env bash
# Ping peers and test TCP ports from config/host.yaml before starting a session.
set -euo pipefail

CONFIG_FILE="${1:-config/host.yaml}"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Config file not found: $CONFIG_FILE" >&2
  exit 1
fi

echo "Checking peers listed in $CONFIG_FILE..."
python - "$CONFIG_FILE" <<'PYEOF'
import sys
from src.common.config import load_host_config
from src.common.transport import TransportError, connect

cfg = load_host_config(sys.argv[1])
all_ok = True
for peer in cfg.peers:
    try:
        conn = connect(peer.address, peer.port, timeout=3.0, label=peer.name)
        conn.close()
        print(f"[OK]   {peer.name} ({peer.address}:{peer.port}) is reachable")
    except TransportError as exc:
        print(f"[FAIL] {peer.name} ({peer.address}:{peer.port}) -> {exc}")
        all_ok = False

sys.exit(0 if all_ok else 1)
PYEOF
