#!/usr/bin/env bash
# smoke_test.sh — Quick 10k-step training smoke test for Mono_QMIX
# Run from the repo root after setup.sh has been executed.
# Usage: bash smoke_test.sh

set -euo pipefail

# ── helpers ───────────────────────────────────────────────────────────────────
info()  { echo "[INFO]  $*"; }
error() { echo "[ERROR] $*" >&2; exit 1; }

# ── resolve repo root ─────────────────────────────────────────────────────────
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_DIR/.venv"

# ── ensure SUMO_HOME is set ───────────────────────────────────────────────────
if [ -z "${SUMO_HOME:-}" ]; then
    if grep -q "SUMO_HOME" ~/.bashrc 2>/dev/null; then
        # shellcheck disable=SC1090
        source ~/.bashrc
    fi
fi

[ -n "${SUMO_HOME:-}" ] || error "SUMO_HOME is not set. Run setup.sh first."
export PYTHONPATH="$SUMO_HOME/tools:${PYTHONPATH:-}"

# ── create venv and install deps if not already done ─────────────────────────
if [ ! -f "$VENV_DIR/bin/python" ]; then
    info "Creating virtual environment at .venv..."
    python3 -m venv "$VENV_DIR"
fi

PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

if ! "$PYTHON" -c "import torch" &>/dev/null; then
    info "Installing Python dependencies into .venv..."
    "$PIP" install --upgrade pip -q
    "$PIP" install -r "$REPO_DIR/requirements.txt"
else
    info "Dependencies already installed in .venv."
fi

info "Using Python: $PYTHON"

# ── run the smoke test ────────────────────────────────────────────────────────
info "Starting 10k smoke test (seed=0, t_max=10000, eval_episodes=5)..."
echo ""

"$PYTHON" "$REPO_DIR/run_experiments.py" \
    --seeds 0 \
    --t_max 10000 \
    --eval_episodes 5

echo ""
echo "──────────────────────────────────────────────────────"
echo "Smoke test complete. Results saved under results/eval/"
echo "──────────────────────────────────────────────────────"
