# Civiq — Full Implementation Summary

End-to-end record of everything built from Phase 0 (RSU Zone Manager) through BGC Core data collection and RSU placement. This document is the single-source overview; each phase section below links to its detailed doc in `docs/rsu/`.

---

## Architecture Overview

Civiq extends Mono QMIX with a **3-level hierarchical value decomposition**:

```
Vehicles (32 agents)
     │  individual Q-values  (MAC forward)
     ▼
LocalQMixer  ×  max_rsus         Level 2 — per RSU zone
     │  local_Q_tot per zone
     ▼
GlobalQMixer  ×  1               Level 3 — across all RSU zones
     │  global_Q_tot (scalar)
     ▼
TD Loss  →  Optimizer
```

Vehicles are assigned to RSU zones spatially (nearest centroid). Each RSU runs one LocalQMixer (weights shared across zones). All RSU-level outputs feed one GlobalQMixer. A single Adam optimizer spans MAC + LocalQMixer + GlobalQMixer.

---

## Parameter Summary (Synthetic 4×4 Map)

| Component | Parameters |
|-----------|------------|
| MAC (BasicMAC, RNN agent) | 29,314 |
| LocalQMixer | 528,641 |
| GlobalQMixer | 561,921 |
| **Total trainable** | **~1.12M** |

---

## Config Constants

| Key | Value | Notes |
|-----|-------|-------|
| `n_agents` | 32 | Controlled vehicles |
| `n_actions` | 4 | Routing actions |
| `obs_dim` | 65 | Per-agent observation dim |
| `max_agents_per_rsu` | 28 | Synthetic 4×4 (LOS E peak 23 + 5 buffer) |
| `max_rsus` | 12 | TEMPORARY — matches synthetic 4×4; update to BGC Full count |
| `global_state_dim` | 2080 | `n_agents × obs_dim = 32 × 65` |
| `local_state_dim` | 1820 | `max_agents_per_rsu × obs_dim = 28 × 65` |
| `radius` | 300.0 m | RSU capture radius |
| `lr` | 0.0005 | Adam |
| `gamma` | 0.99 | Discount |
| `td_lambda` | 0.8 | |
| `grad_norm_clip` | 10000 | Civiq; higher than standard QMIX due to reward scale |
| `buffer_size` | 100 | Reduced from 400 due to RAM; local_states dominates at ~83 MB/episode |

---

## Phase 0 — RSU Zone Manager

> **Doc:** [docs/rsu/phase0_overview.md](rsu/phase0_overview.md)

**What was built:**
- `pymarl/src/components/rsu_zone_manager.py` — `RSUZoneManager` class
- `config/envs/rsu/synthetic_4x4.yaml` — zone config for the 12-node synthetic map
- `scripts/rsu/extract_rsu_coords.py` — extracts junction x/y from `final_map.net.xml`
- `scripts/rsu/validate_zone_manager.py` — Phase 0 gate

**Key design:**
- Nearest-RSU assignment via vectorised numpy distance matrix — O(n_vehicles × n_rsus)
- Every vehicle assigned to exactly one zone; no vehicle ever left unassigned
- `radius` field stored but not enforced for assignment (used for visualization/filtering)
- `max_agents_per_rsu: 28` — empirical from LOS E 3600-step run (peak 23, +5 buffer)

**Gate result:** PASSED — 100 steps, `train_med.sumocfg`, all 3 assertions held for every step.

---

## Phase 1 — LocalQMixer

> **Doc:** [docs/rsu/phase1_local_qmixer.md](rsu/phase1_local_qmixer.md)

**What was built:**
- `pymarl/src/modules/mixers/local_qmixer.py` — LocalQMixer class (verified pre-existing, confirmed correct)
- `pymarl/src/config/algs/civiq_sumo.yaml` — created; Civiq-specific hyperparameters
- `tests/test_local_qmixer.py` — Phase 1 gate

**Key design:**
- Mirrors QMixer exactly; `max_agents_per_rsu` replaces `n_agents`; state input is `local_states` (per-RSU concatenated agent obs)
- Agent mask applied **before** first bmm — padded slots contribute zero to the mix
- Empty-zone bias: all-zero mask still produces finite output (bias terms fire); no NaN risk
- Weights shared across all RSU zones — one object, called once per zone per timestep

**Gate result:** PASSED — 4 tests (full batch, partial, empty zone, gradient flow).

---

## Phase 2 — GlobalQMixer

> **Doc:** [docs/rsu/phase2_global_qmixer.md](rsu/phase2_global_qmixer.md)

**What was built:**
- `pymarl/src/modules/mixers/global_qmixer.py` — GlobalQMixer class (verified pre-existing)
- `civiq_sumo.yaml` updated with `global_state_dim`
- `tests/test_global_qmixer.py` — Phase 2 gate

**Key design:**
- Mirrors LocalQMixer; `max_rsus` replaces `max_agents_per_rsu`; state input is `batch["state"]` (global)
- RSU mask applied before bmm — inactive RSU slots contribute zero
- `global_state_dim = 2080 = 32 × 65` (NOT 4485; only the 32 controlled agents have obs vectors)

**Gate result:** PASSED — 4 tests (full, partial, empty network, gradient flow).

---

## Phase 3 — HierarchicalQLearner Scaffold

> **Doc:** [docs/rsu/phase3_hierarchical_q_learner.md](rsu/phase3_hierarchical_q_learner.md)

**What was built:**
- `pymarl/src/learners/hierarchical_q_learner.py` — HierarchicalQLearner class (scaffold)
- `pymarl/src/learners/__init__.py` updated
- `tests/test_hierarchical_learner.py` — Phase 3 gate

**Key design:**
- Fork of QLearner — MAC, double-Q, TD loss, grad clip, target update, save/load all identical
- Mixer section stubs to `NotImplementedError` pending Phase 4 batch fields
- Single Adam optimizer across MAC + LocalQMixer + GlobalQMixer
- `target_local_mixer` and `target_global_mixer` deepcopied on init
- `RSUZoneManager` instantiated in learner (for future use); rsu_config path resolved to repo root

**Gate result:** PASSED — 4 tests (construction, parameter grouping, target independence, save/load round-trip).

---

## Phase 4 — Data Collection (Episode Runner + Replay Buffer)

> **Doc:** [docs/rsu/phase4_data_collection.md](rsu/phase4_data_collection.md)

**What was built / modified:**
- `pymarl/src/envs/sumo_grid_reroute.py` — 4 new Civiq observation methods + zone_manager init
- `pymarl/src/runners/episode_runner.py` — zone data collection at every timestep
- `pymarl/src/main.py` — `get_scheme()` adds Civiq fields when `mixer == "civiq"`
- `civiq_sumo.yaml` — `global_state_dim` corrected from 4485 → 2080
- `tests/test_phase4_data_collection.py` — Phase 4 gate

**New batch fields:**

| Field | Shape | Notes |
|-------|-------|-------|
| `zone_assignments` | `(B, T+1, n_agents)` | RSU id per agent slot; −1 = unassigned |
| `rsu_agent_qs` | `(B, T+1, max_rsus, max_agents_per_rsu)` | Zero in buffer; scatter-filled in `train()` |
| `agent_masks_per_rsu` | `(B, T+1, max_rsus, max_agents_per_rsu)` | 1.0 = real, 0.0 = padding |
| `local_states` | `(B, T+1, max_rsus, max_agents_per_rsu × obs_dim)` | Per-RSU concatenated agent observations |

**Key fix:** `_make_env` in runner forwarded Civiq keys (`rsu_config`, `max_rsus`, `max_agents_per_rsu`, `obs_dim`) to the env — previously they lived only in the alg config and were not passed through.

**RAM note:** `local_states` dominates at ~83 MB/episode × 100 episodes = ~8.3 GB. `buffer_size` reduced from 400 → 100.

**Gate result:** PASSED — 50 steps, `train_med.sumocfg`, libsumo.

---

## Phase 5 — Hierarchical Training (`HierarchicalQLearner.train()`)

> **Doc:** [docs/rsu/phase5_hierarchical_train.md](rsu/phase5_hierarchical_train.md)

**What was built / modified:**
- `pymarl/src/learners/hierarchical_q_learner.py` — NotImplementedError replaced with full mixing
- `pymarl/src/main.py` — HierarchicalQLearner import, learner selection, `--alg_config` CLI arg
- `pymarl/src/config/algs/civiq_sumo.yaml` — added `learner: "hierarchical_q_learner"`
- `pymarl/src/config/algs/civiq_single_rsu.yaml` — single-RSU equivalence check config
- `config/envs/rsu/synthetic_single_rsu.yaml` — single RSU at map centroid, radius 2000m
- `tests/test_phase5_gradient_flow.py` — Phase 5 gate

**`rsu_agent_qs` construction — vectorised scatter:**

```python
# Pre-compute membership and slot indices for all RSUs at once (1 kernel launch)
in_rsu_all   = (zone_assignments.unsqueeze(-1) == torch.arange(max_rsus))  # (B,T,A,R)
slot_idx_all = (in_rsu_all.long().cumsum(dim=2) - 1).clamp(0, A-1)        # (B,T,A,R)

out = torch.zeros(B, T, max_rsus, max_agents_per_rsu)
for r in range(max_rsus):
    src = qvals * in_rsu_all[..., r].float()   # non-RSU agents → 0 (safe for scatter_add_)
    out[:, :, r, :].scatter_add_(dim=-1, index=slot_idx_all[..., r], src=src)
```

Agents within each RSU are ordered by ascending agent index, matching `_build_agents_per_rsu` in the env. `scatter_add_` is safe: non-RSU agents contribute `src=0`.

**Mixing forward pass:**

```
Online path:
  local_qtots  = local_mixer(rsu_agent_qs, local_states, agent_masks)   → (BT, R)
  global_qtot  = global_mixer(local_qtots, global_states, rsu_mask)     → (B, T, 1)

Target path:  same, using [:, 1:] batch fields and target mixers
```

**TD loss:** `loss = ((global_qtot - (r + γ(1−d)·target_global_qtot).detach()) * mask)² / mask.sum()`

**`--alg_config` CLI:** selects from `config/algs/`. Default = `qmix_sumo.yaml` (backward compatible).

**Gate results:** PASSED

| Test | Result |
|------|--------|
| Batch field shapes | PASSED |
| train() completes; loss=3.46M, qtot=−2.35, no NaN/Inf | PASSED |
| All MAC/LocalQMixer/GlobalQMixer params have non-None, non-NaN grads | PASSED |

**10k smoke test:**

| Metric | Value |
|--------|-------|
| Episodes completed | 100 |
| Loss at t=5000 | 4,051,339 |
| Loss at t=10000 | 3,541,044 (−12.6%) |
| NaN/Inf | None |
| Checkpoint saved | Yes |

**Single-RSU equivalence check:** Both 12-RSU and 1-RSU configs produced losses in the same order of magnitude (4M vs 8M). No 100× divergence. Passed.

---

## BGC Core — Block A: Data Collection

**Script:** `scripts/bgc/block_a_bgc_core.py`

Standalone libsumo script — no RL, no training. Collects raw vehicle telemetry from the BGC Core map at LOS E for RSU placement analysis.

| Parameter | Value |
|-----------|-------|
| Map | `bgc_core/final_map.net.xml` |
| Demand file | `bgc_core/trips_high.xml` (LOS E, period=0.7) |
| Backend | libsumo |
| Steps | 3600 |
| Outputs | `vehicle_positions_bgc_core_los_e.csv`, `vehicle_density_bgc_core_los_e.png` |

**Results:**
- Peak simultaneous vehicles: **197**
- Total CSV rows: **179,470**
- Vehicle activity concentrated in steps 1–1500 (trips_high.xml spawns until t=1000; steps 1501–3600 drain to empty)

**CSV schema:** `timestep, vehicle_id, x, y`

---

## BGC Core — RSU Placement

**Tool:** netedit (manual intersection selection by user)

8 RSUs placed at traffic-light intersections on the BGC Core map, confirmed visually in sumo-gui with 300m orange circles and cyan PoI markers.

| RSU | x | y | LOS E Peak |
|-----|---|---|------------|
| RSU00 | 37.6 | 221.4 | 41 |
| RSU01 | 228.0 | 159.4 | **54** |
| RSU02 | 417.3 | 95.0 | 37 |
| RSU03 | 606.9 | 30.2 | 28 |
| RSU04 | 676.0 | 220.0 | 24 |
| RSU05 | 481.0 | 284.8 | 41 |
| RSU06 | 386.5 | 317.4 | 45 |
| RSU07 | 426.4 | 417.6 | 15 |

**Config:** `config/envs/rsu/bgc_core.yaml`

**Overlap analysis:** All 8 RSUs mutually overlap (300m radius on a ~700×450m map). RSU05/RSU06/RSU07 are extremely tight (100–145m apart); RSU07 starves as a result (peak 15 vs RSU06's 45).

---

## BGC Core — Block B: Zone Peak Count

**Script:** `scripts/bgc/block_b_zone_peak_count.py`

Reads `vehicle_positions_bgc_core_los_e.csv` + `bgc_core.yaml`, assigns each vehicle at each timestep to its nearest RSU within 300m, and reports peak simultaneous vehicle count per zone.

**Result:** Overall peak = **54** (RSU01). `max_agents_per_rsu` set to **50** (hard ceiling; RSU01 loses 4 vehicles at peak — masked).

**Pending placement revision:**
- RSU07 is a candidate for removal (peak 15, only 108m from RSU06)
- RSU01 area needs a relief RSU between (37.6, 221.4) and (228.0, 159.4) to split the 54-vehicle load below the 50-vehicle ceiling

---

## Open Items

| Item | Status |
|------|--------|
| Revise BGC Core RSU positions (remove RSU07, add RSU near RSU01) | Pending user netedit selection |
| Fill in `pymarl/src/config/envs/sumo_bgc_core.yaml` | Pending |
| Update `max_rsus` globally once BGC Full node count is known | Pending |
| Reward normalization for GlobalQMixer (grad norms ~2–3M at current reward scale) | Pending |
| Full 200k-step training run on BGC Core vs Mono QMIX baseline | Pending |

---

## File Map

```
pymarl/src/
  components/
    rsu_zone_manager.py          Phase 0 — zone assignment
  modules/mixers/
    local_qmixer.py              Phase 1 — per-RSU mixing
    global_qmixer.py             Phase 2 — cross-RSU mixing
  learners/
    hierarchical_q_learner.py    Phases 3+5 — end-to-end training
  runners/
    episode_runner.py            Phase 4 — batch field collection
  envs/
    sumo_grid_reroute.py         Phase 4 — zone observation methods
  main.py                        Phase 4+5 — scheme, learner selection, CLI
  config/algs/
    civiq_sumo.yaml              Civiq hyperparameters
    civiq_single_rsu.yaml        Equivalence check config
  config/envs/
    sumo_bgc_core.yaml           BGC Core env config (pending)

config/envs/rsu/
  synthetic_4x4.yaml             12-RSU config (synthetic map)
  synthetic_single_rsu.yaml      1-RSU equivalence check
  bgc_core.yaml                  8-RSU config (BGC Core)

scripts/
  rsu/
    extract_rsu_coords.py        Extract junction coords from net.xml
    validate_zone_manager.py     Phase 0 gate
    los_e_zone_validation.py     LOS E peak measurement
  bgc/
    block_a_bgc_core.py          BGC Core data collection (3600 steps)
    block_b_zone_peak_count.py   Per-zone peak count from CSV

tests/
  test_local_qmixer.py           Phase 1 gate
  test_global_qmixer.py          Phase 2 gate
  test_hierarchical_learner.py   Phase 3 gate
  test_phase4_data_collection.py Phase 4 gate
  test_phase5_gradient_flow.py   Phase 5 gate

docs/rsu/
  phase0_overview.md
  phase1_local_qmixer.md
  phase2_global_qmixer.md
  phase3_hierarchical_q_learner.md
  phase4_data_collection.md
  phase5_hierarchical_train.md
  rsu_zone_manager_api.md
  los_e_validation_results.md
  civiq_implementation_summary.md  ← this file
```
