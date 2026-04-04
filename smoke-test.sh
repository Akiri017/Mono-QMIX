#!/usr/bin/env bash
# smoke_test.sh — 500k-step training smoke test for Mono_QMIX
# Run from the repo root after setup.sh has been executed.
# Usage: bash smoke_test.sh

set -euo pipefail

# ── helpers ───────────────────────────────────────────────────────────────────
info()  { echo "[INFO]  $*"; }
error() { echo "[ERROR] $*" >&2; exit 1; }

# ── resolve repo root ─────────────────────────────────────────────────────────
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── ensure SUMO_HOME is set ───────────────────────────────────────────────────
if [ -z "${SUMO_HOME:-}" ]; then
    if grep -q "SUMO_HOME" ~/.bashrc 2>/dev/null; then
        # shellcheck disable=SC1090
        source ~/.bashrc
    fi
fi

[ -n "${SUMO_HOME:-}" ] || error "SUMO_HOME is not set. Run setup.sh first."
export PYTHONPATH="$SUMO_HOME/tools:${PYTHONPATH:-}"

# ── verify deps are in place (setup.sh should have handled this) ──────────────
for pkg in torch traci libsumo tensorboard; do
    python3 -c "import $pkg" 2>/dev/null \
        || error "Missing Python dependency: $pkg. Run setup.sh first."
done

info "Using Python: $(which python3)"

# ── run the smoke test ────────────────────────────────────────────────────────
info "Starting 500k smoke test (seed=0, t_max=500000, eval_episodes=5)..."
echo ""

python3 "$REPO_DIR/run_experiments.py" \
    --seeds 0 \
    --t_max 500000 \
    --eval_episodes 5

echo ""
echo "──────────────────────────────────────────────────────"
echo "Smoke test complete. Results saved under results/eval/"
echo "──────────────────────────────────────────────────────"
