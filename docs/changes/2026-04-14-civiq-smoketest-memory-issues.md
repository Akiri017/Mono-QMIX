# Debug Session: Civiq Smoketest Memory Issues and Crashes
**Date:** 2026-04-14
**Status:** Partial — two fixes applied, one pending approval
**Affected areas:** `pymarl/src/learners/hierarchical_q_learner.py`, `pymarl/src/evaluate.py`, `pymarl/src/config/algs/civiq_sumo.yaml`

## Context
First smoketest run of Civiq on the 4x4 synthetic map surfaced three sequential crashes: a missing `psutil` module, a Windows ACCESS_VIOLATION (0xC0000005) mid-training, and a 46GB CPU memory allocation failure at buffer initialization. Each was investigated and root-caused separately.

## Discussion and decisions

### Issue 1: Missing psutil module
`sumo_grid_reroute.py` imports `psutil` for CPU monitoring metrics (`cpu_percent_mean`, `cpu_percent_peak`, `process_cpu_s`). The module was not installed in the `.venv`. Fixed by installing `psutil`.

### Issue 2: Windows ACCESS_VIOLATION (exit code 3221225477) after validation
Training crashed immediately after the first validation checkpoint at t=100 with a native C++ ACCESS_VIOLATION inside libsumo. Mono-QMIX never hit this crash.

**Root cause:** `HierarchicalQLearner.train()` was missing `torch.no_grad()` around the target MAC forward pass (lines 156–162). This was present in `QLearner` (line 195, commit `f37bd2f3`) but was dropped when `HierarchicalQLearner` was forked. Without the guard, PyTorch builds a full gradient graph through the target network on every `train()` call. Civiq's tensors (`local_states`, `rsu_agent_qs`, `agent_masks_per_rsu`) are significantly larger than Mono-QMIX's, so the unnecessary graph caused heap pressure that corrupted libsumo's in-process C++ heap, manifesting as an ACCESS_VIOLATION on the next libsumo call after training.

**Decision:** Add `with torch.no_grad():` to match `QLearner` exactly. Switching to TraCI backend was considered and rejected — the crash is a code bug, not a libsumo instability.

### Issue 3: 46GB buffer allocation (OOM at init)
`buffer_size` in `civiq_sumo.yaml` was set to 400 (full training value, not reverted for smoketests). Civiq's `local_states` field has shape `(max_rsus=17, max_agents_per_rsu=28, obs_dim=65)` = 30,940 floats per timestep — roughly 124KB/timestep vs Mono-QMIX's ~8KB/timestep state field. At buffer_size=400 and episode_limit=1000:

`400 × 1001 × 17 × 1820 × 4 bytes = 49.5 GB`

This exceeds any reasonable machine (including Kaggle's 15GB VRAM).

**Root cause identified:** `local_states` is pre-allocated into the replay buffer at initialization despite only being consumed in `HierarchicalQLearner.train()`. It can be computed on-the-fly from `batch["obs"]` and `batch["zone_assignments"]`, which are already in the buffer and are tiny by comparison. Removing it from the scheme reduces the buffer footprint by ~99.5%.

**Immediate mitigation:** `buffer_size` reduced from 400 → 32 (= `batch_size`, minimum viable for training). Full fix (compute on-the-fly) is pending.

## What changed
- `pymarl/src/learners/hierarchical_q_learner.py` lines 156–162 — wrapped target MAC forward pass in `torch.no_grad()` to match `QLearner`; fixes ACCESS_VIOLATION crash caused by unnecessary gradient graph memory pressure
- `pymarl/src/evaluate.py` lines 110–154 — added `reset_mask` to base scheme (was missing, causing hidden state resets to be silently skipped for all algorithms during eval); added Civiq field block guarded by `args.get("mixer") == "civiq"` for batch integrity
- `pymarl/src/config/algs/civiq_sumo.yaml` line 29 — `buffer_size` reduced from 400 → 32 with updated comment documenting memory implications per episode

## Implementation notes
- The `torch.no_grad()` fix is the same change made to `QLearner` in commit `f37bd2f3` ("q-learner wrapped for gpu optimization"). The HierarchicalQLearner fork simply missed carrying it over.
- `reset_mask` was missing from `evaluate.py`'s hardcoded scheme (unlike `main.py` which had it after commit `ff91678b`). This affected both Mono-QMIX and Civiq evaluations — hidden states were never reset on slot turnovers during eval. The Civiq field block in `evaluate.py` is harmless for action selection (mixers are not called during eval) but correct for batch integrity.
- Even at `buffer_size=32`, `local_states` still allocates ~4GB on CPU. The pending fix (compute on-the-fly) is the correct long-term solution and reduces this to ~20MB per batch at training time.
- The 46GB allocation is Civiq-specific and does not affect Mono-QMIX because Mono-QMIX has no `local_states` field in its scheme.

## Status and follow-up
- **Status:** Partial — two of three issues fixed; buffer OOM mitigation in place but not fully resolved
- **Follow-up items:**
  - **[PENDING APPROVAL]** Remove `local_states` from the replay buffer scheme (`main.py`, `evaluate.py`, `episode_runner.py`) and compute it on-the-fly in `HierarchicalQLearner._build_local_states()` from `batch["obs"]` + `batch["zone_assignments"]`. Estimated memory reduction: 99.5% (~4GB → ~20MB). No breaking changes — only `hierarchical_q_learner.py` reads `batch["local_states"]`.
  - Verify smoketest completes successfully with `buffer_size=32` before raising it for full training runs.
  - For full 200k training, `buffer_size=200–400` will require 25–50GB RAM with current scheme, or ~1GB with the on-the-fly fix applied.
