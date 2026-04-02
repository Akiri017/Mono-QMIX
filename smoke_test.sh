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

# ── pick Python: prefer .venv if present, else system python3 ─────────────────
if [ -f "$REPO_DIR/.venv/bin/python" ]; then
    PYTHON="$REPO_DIR/.venv/bin/python"
    info "Using venv Python: $PYTHON"
else
    PYTHON="$(command -v python3 || true)"
    [ -n "$PYTHON" ] || error "python3 not found. Run setup.sh first."
    info "Using system Python: $PYTHON"
fi

# ── ensure SUMO_HOME is set ───────────────────────────────────────────────────
if [ -z "${SUMO_HOME:-}" ]; then
    # setup.sh writes this to ~/.bashrc; source it if not already in env
    if grep -q "SUMO_HOME" ~/.bashrc 2>/dev/null; then
        # shellcheck disable=SC1090
        source ~/.bashrc
    fi
fi

[ -n "${SUMO_HOME:-}" ] || error "SUMO_HOME is not set. Run setup.sh first."
export PYTHONPATH="$SUMO_HOME/tools:${PYTHONPATH:-}"

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
