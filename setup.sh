#!/usr/bin/env bash
# setup.sh — Ubuntu environment setup for Mono_QMIX
# Installs SUMO (via official PPA) and Python dependencies.
# Run this before cloning the repo: bash setup.sh

set -euo pipefail

# ── helpers ───────────────────────────────────────────────────────────────────
info()  { echo "[INFO]  $*"; }
error() { echo "[ERROR] $*" >&2; exit 1; }

# ── 1. require Ubuntu / apt ───────────────────────────────────────────────────
if ! command -v apt-get &>/dev/null; then
    error "apt-get not found. This script targets Ubuntu/Debian only."
fi

# ── 2. system packages ────────────────────────────────────────────────────────
info "Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y software-properties-common python3 python3-pip python3-venv git

# ── 3. install SUMO via the official Eclipse PPA ─────────────────────────────
info "Adding SUMO PPA and installing SUMO..."
sudo add-apt-repository -y ppa:sumo/stable
sudo apt-get update -qq
sudo apt-get install -y sumo sumo-tools sumo-doc

if ! command -v sumo &>/dev/null; then
    error "SUMO binary not found after install. Check the PPA output above."
fi
info "SUMO installed: $(sumo --version 2>&1 | head -1)"

# ── 4. set SUMO_HOME ──────────────────────────────────────────────────────────
# The PPA places SUMO under /usr/share/sumo on Ubuntu
SUMO_HOME_PATH="/usr/share/sumo"

if [ ! -d "$SUMO_HOME_PATH" ]; then
    error "Expected SUMO_HOME at $SUMO_HOME_PATH but directory not found."
fi

if ! grep -q "SUMO_HOME" ~/.bashrc; then
    info "Adding SUMO_HOME to ~/.bashrc..."
    {
        echo ""
        echo "# SUMO"
        echo "export SUMO_HOME=\"$SUMO_HOME_PATH\""
        echo "export PYTHONPATH=\"\$SUMO_HOME/tools:\${PYTHONPATH:-}\""
    } >> ~/.bashrc
fi

export SUMO_HOME="$SUMO_HOME_PATH"
export PYTHONPATH="$SUMO_HOME/tools:${PYTHONPATH:-}"
info "SUMO_HOME set to $SUMO_HOME"

# ── 5. install Python dependencies ────────────────────────────────────────────
info "Upgrading pip..."
# --break-system-packages bypasses the PEP 668 externally-managed-environment
# guard on Ubuntu 22.04+. Safe here since this is a single-purpose training VM.
python3 -m pip install --upgrade pip --break-system-packages -q

info "Installing Python packages (traci, sumolib, numpy, torch)..."
python3 -m pip install --break-system-packages \
    "traci>=1.25" \
    "sumolib>=1.25" \
    numpy \
    pyyaml \
    torch

# ── 6. smoke check ────────────────────────────────────────────────────────────
info "Verifying installs..."

python3 - <<'EOF'
import sys

results = []

try:
    import traci
    results.append(f"  traci     OK ({traci.__version__})")
except Exception as e:
    results.append(f"  traci     FAILED: {e}")

try:
    import yaml
    results.append(f"  pyyaml    OK ({yaml.__version__})")
except Exception as e:
    results.append(f"  pyyaml    FAILED: {e}")

try:
    import sumolib
    results.append(f"  sumolib   OK")
except Exception as e:
    results.append(f"  sumolib   FAILED: {e}")

try:
    import numpy as np
    results.append(f"  numpy     OK ({np.__version__})")
except Exception as e:
    results.append(f"  numpy     FAILED: {e}")

try:
    import torch
    results.append(f"  torch     OK ({torch.__version__})")
except Exception as e:
    results.append(f"  torch     FAILED: {e}")

for r in results:
    print(r)

if any("FAILED" in r for r in results):
    sys.exit(1)
EOF

echo ""
echo "──────────────────────────────────────────────────────"
echo "Setup complete. SUMO and Python dependencies are ready."
echo "Next: clone the repo, then run:  source ~/.bashrc"
echo "──────────────────────────────────────────────────────"
