# Phase 7 — BGC Full Wiring: Complete Record

**Date:** 2026-04-11 → 2026-04-13
**Branch:** `bgc-full-mono-qmix` (merged from `rsu-placement`)
**Status:** Wiring complete, BGC Core 10k smoke test passed, BGC Full 1k smoke test in progress

---

## Background: Why Phase 7 Existed

BGC Full was the final scenario needed to complete the thesis comparison:
4x4 (synthetic) → BGC Core (real, small) → BGC Full (real, full city).

The scenario existed on disk but was completely unrunnable. Three separate problems
stacked on each other:

1. **The network was too large** — 2,723 edges made the Civiq observation infeasible
2. **RSU placements were invalid** — several RSUs sat on roads that didn't exist or were overcrowded
3. **All route files were stale** — generated against old network versions with missing edges

The decision was made to fix the map first before any code changes, since the
map edit would invalidate everything else anyway.

---

## Part 1 — The Map Problem (Pre-condition for everything else)

### Why 2,723 edges was impossible

Civiq encodes each agent's current road segment as a one-hot vector over all
network edges. At 2,723 edges:

```
obs_dim         = 5 (ego) + 2,723 (one-hot) + 12 (traffic stats) = 2,740
local_state_dim = max_agents_per_rsu × obs_dim
               = 384 × 2,740 = 1,052,160   → ~170 GB replay buffer
```

No hardware fix would make this tractable. The only options were:
- Implement learned edge embeddings (weeks of architecture work)
- Reduce the network itself

### Map edit decision

The user edited `bgc_full/final_map.net.xml` in netedit:
- **Removed residential areas** — low-traffic side streets with no routing decisions
- **Removed C5 road segments** — C5 is a controlled-access arterial; agents make
  no routing decisions on it; OD access points at BGC boundary were preserved

Result: **734 non-internal edges**. Full one-hot now tractable:
```
obs_dim         = 5 + 734 + 12 = 751
local_state_dim = 30 × 751     = 22,530   → ~9.6 GB at buffer_size=100 ✓
```

The area still constitutes BGC proper — residential removals were peripheral
blocks, not the core commercial/financial district. C5 removal keeps the BGC
boundary connections intact.

**File:** `bgc_full/final_map.net.xml`

---

## Part 2 — RSU Config Cleanup

### Problem after map edit

With the network shrunk, four RSUs were now positioned on roads that no longer
existed. The peak count from the original Block B run (379 vehicles at RSU17)
was also inflated — it was measured on the old dense network.

### RSUs removed (invalid positions):
| RSU | Reason |
|-----|--------|
| RSU02 | On removed residential road |
| RSU13 (original) | On removed segment, index conflict |
| RSU13 (duplicate) | On removed segment |
| RSU18 | On removed segment |

RSU09 and RSU12 were also dropped to resolve ISO 250m spacing violations.

### RSUs added (targeted placement):
New RSUs were placed at TLS cluster junctions in the revised network to cover
zones left empty by the removals.

> **Note:** RSU02 and RSU03 remain ~201m apart (below the 250m ISO minimum).
> This was a deliberate decision to preserve strategic coverage at that junction
> pair. Accepted as a documented deviation.

### Block A + B re-run

- **Block A** (`block_a_bgc_full.py`): duarouter pre-routes `trips_highEnough.xml`
  → `bgc_full_los_e_routed.rou.xml`, then libsumo collects XY positions over a
  3,600-step LOS E run
- **Block B** (`block_b_bgc_full.py`): assigns each vehicle to nearest RSU within
  300m, reports max simultaneous per zone

Result: **peak = 25** (RSU13 post-cleanup) → `max_agents_per_rsu = 30` (25 + 5 buffer)

**Final RSU config:** 17 RSUs, all at valid TLS junctions

**File:** `config/envs/rsu/bgc_full.yaml`

---

## Part 3 — Environment Config Wiring

### sumo_bgc_full.yaml: Dimensions updated

The yaml had stale dimensions from the original 2,723-edge network with a lossy
256-cap workaround. The cap was the original "fix" for the memory problem — it
meant ~91% of the network was invisible to agents (all-zeros in the edge slot).
With 734 edges, the cap is gone entirely.

| Key | Old (2,723-edge era) | New |
|-----|----------------------|-----|
| `obs_edge_dim` | 256 (lossy cap) | 734 (full one-hot) |
| `obs_shape` | 273 | 751 |
| `state_shape` | 8,736 | 24,032 |

Civiq overrides section was also missing entirely — added:
```yaml
obs_dim: 751
global_state_dim: 24032
max_agents_per_rsu: 30
max_rsus: 17
local_state_dim: 22530
rsu_config: "config/envs/rsu/bgc_full.yaml"
```

**File:** `pymarl/src/config/envs/sumo_bgc_full.yaml`

### Training sumocfgs created

`bgc_full/train_low/med/high.sumocfg` did not exist — the env's `los_cfg`
resolver had nothing to point to.

Wardrop route files (pre-computed paths) were stale and couldn't be regenerated:
netedit produces duplicate internal edges that duarouter can't handle even with
`--ignore-errors`. Workaround: sumocfgs use raw trips files with
`--ignore-route-errors true`. SUMO routes OD trips internally at simulation time.

**Files created:** `bgc_full/train_low.sumocfg`, `train_med.sumocfg`, `train_high.sumocfg`

### max_rsus: 17 propagated globally

`max_rsus` is baked into the GlobalQMixer tensor shape — all configs must match.
The value was a stale placeholder (12) in three files and wrong (20) in the old
bgc_full yaml.

| File | Old | New |
|------|-----|-----|
| `pymarl/src/config/algs/civiq_sumo.yaml` | 12 | 17 |
| `pymarl/src/config/envs/sumo_bgc_core.yaml` | 12 | 17 |
| `config/envs/rsu/bgc_core.yaml` | 12 | 17 |
| `config/envs/rsu/synthetic_4x4.yaml` | 12 | 17 |
| `config/envs/rsu/bgc_full.yaml` | 20 | 17 |

---

## Part 4 — Branch Merge (rsu-placement → bgc-full-mono-qmix)

Civiq scaffolding lived on `rsu-placement`; BGC Full wiring lived on
`bgc-full-mono-qmix`. Two merge conflicts:

**`main.py`:** `rsu-placement` added `--alg_config` pre-parser; `bgc-full-mono-qmix`
added `--los_level`. Resolution: combined both into one pre-parser.

**`sumo_grid_reroute.py`:** Both branches modified `_LOS_CFG` handling. Resolution:
took `rsu-placement`'s cleaner one-liner `_LOS_CFG = env_args.get("los_cfg", {...})`.

**Git issue:** Push rejected because `vehicle_positions_bgc_full_los_e.csv` (107 MB)
and `bgc_full_los_e_routed.rou.xml` were committed. Fixed via soft-reset 2 commits,
added to `.gitignore`, recommitted.

---

## Part 5 — Smoke Test Fixes (2026-04-13)

Running the actual smoke tests exposed several more issues:

### main.py: `--alg_config civiq_sumo` caused "config not found"

The pre-parser used user-supplied values as-is. The default path had `.yaml`
hardcoded; user-supplied paths did not. Added `_ensure_yaml()` helper that
appends `.yaml` when missing.

### evaluate.py: hardcoded 4x4 configs

`evaluate.py` always loaded `qmix_sumo.yaml` + `sumo_grid4x4.yaml`. Loading a
BGC Full checkpoint caused `load_state_dict` to fail with shape mismatch
(`fc1.weight: [64, 65]` vs `[64, 751]`). Added `--alg_config` and `--env_config`
arguments mirroring `main.py`.

### main.py: hardcoded "SUMO Grid 4x4" banner

Training always printed `Environment: SUMO Grid 4x4`. Fixed by storing
`env_config_name` in args at config-load time; banner reads from it.

### BGC Full trips: merge conflicts + invalid edges

`trips_low.xml` and `trips_med.xml` had unresolved git merge conflict markers
making them unparseable XML. All three trips files also referenced edges removed
during the map edit. libsumo raised a `TraCIException` at `simulation.start()`
despite `--ignore-route-errors` in the sumocfg — libsumo is stricter than
SUMO's standalone mode on departure edge validity.

Fix: one-pass Python script stripped conflict markers (keeping HEAD side) and
removed trips whose `from=` or `to=` edge was absent from the 734-edge network.

| File | Removed | Remaining |
|------|---------|-----------|
| `bgc_full/trips_low.xml` | 737 | 164 |
| `bgc_full/trips_med.xml` | 1,468 | 333 |
| `bgc_full/trips_highEnough.xml` | 5,854 | 1,347 |

### controlled_init.rou.xml: stale edge `743811528`

The fleet file was generated against a pre-edit network version. Edge `743811528`
(start of `ctrl_11`'s route) no longer existed. The env's `_compute_shortest_route`
called `sumolib.net.getEdge('743811528')` → `KeyError`, silently dropping that
agent slot on every episode reset.

Root cause: the file had been regenerated at some point but against the wrong
network version. Regenerated again against the verified 734-edge network; all
32 OD pairs checked — 0 bad edges.

### hierarchical_q_learner.py: `.view()` on non-contiguous tensors

`learner.train()` crashed:
```
RuntimeError: view size is not compatible with input tensor's size and stride.
Use .reshape(...) instead.
```

Cause: `local_states`, `global_states`, `agent_masks_per_rsu` (and target
counterparts) are temporal slices (`batch[...][:, :-1]`, `[:, 1:]`). Slicing
along a non-last dimension creates non-contiguous memory views. `.view()` requires
contiguous memory. `.reshape()` handles this by calling `.contiguous()` internally.

Six tensors fixed across online and target paths:

| Tensor | Fix |
|--------|-----|
| `local_states.view(BT*R, -1)` | → `.reshape()` |
| `global_states.view(BT, -1)` | → `.reshape()` |
| `rsu_mask.view(BT, R)` | → `.reshape()` |
| `target_local_states.view(BT*R, -1)` | → `.reshape()` |
| `target_global_states.view(BT, -1)` | → `.reshape()` |
| `target_rsu_mask.view(BT, R)` | → `.reshape()` |

**File:** `pymarl/src/learners/hierarchical_q_learner.py`

---

## Part 6 — Diverse Vehicle Types

### Decision

All scenarios previously simulated only passenger cars (`thesis_car` /
`DEFAULT_VEHTYPE`). BGC traffic in reality includes motorcycles and trucks in
significant proportions (Philippines urban context).

**Scope:** Background traffic only. Controlled agents (the 32 MARL vehicles)
remain `thesis_car` for training stability. No obs_dim or state_shape changes.

### Vehicle types

| ID | vClass | accel | decel | length | maxSpeed | Share |
|----|--------|-------|-------|--------|----------|-------|
| `thesis_car` | passenger | 2.6 | 4.5 | 4.5m | 50 km/h | 65% |
| `thesis_moto` | motorcycle | 3.5 | 5.0 | 2.2m | 60 km/h | 25% |
| `thesis_truck` | truck | 1.0 | 3.0 | 8.0m | 35 km/h | 10% |

`thesis_bus` was considered and removed — jeepneys (the Philippine bus equivalent)
behave more like `thesis_car` in terms of road usage and don't warrant a
separate vehicle class for this simulation scope.

### What changes in the simulation

- Motorcycles accelerate faster, fit tighter gaps, filter through congestion
- Trucks are slower, wider, block lanes longer at junctions
- Emissions metric increases (trucks emit significantly more per km)
- Controlled agents observe the *effects* (speed, occupancy, queue) but not the
  vehicle type directly — vType is not in the observation vector

### Files changed

**`vtypes.add.xml`** — updated in `bgc_core/`, `bgc_full/`; created new in `4by4_map/`

**4x4 sumocfgs** — added `<additional-files value="vtypes.add.xml"/>` to all three
`Configuration_*.sumocfg` files (4x4 previously loaded no vtype file at all)

**Trips files** (8 total across all three scenarios) — `type=` attribute assigned
to every `<trip>` using the distribution above, seeded at 42 for reproducibility

**`scripts/bgc/assign_vehicle_types.py`** — new one-time assignment script;
re-run if trips files are ever replaced

**`sumo_grid_reroute.py`** — `typeID="DEFAULT_VEHTYPE"` → `typeID="thesis_car"`
for dynamically spawned controlled vehicles

---

## Smoke Test Results

| Scenario | Config | t_max | batch | Result |
|----------|--------|-------|-------|--------|
| BGC Core | `civiq_sumo` | 10,000 | 4 | Pass — training started, no crash, no NaN |
| BGC Full | `civiq_sumo` | 1,000 | 1 | In progress (diverse types just wired) |

---

## Final State

| Item | Value |
|------|-------|
| Network edges (BGC Full) | 734 |
| RSU count (BGC Full) | 17 |
| `max_agents_per_rsu` | 30 (peak 25 + 5 buffer) |
| `obs_dim` | 751 (5 + 734 + 12) |
| `local_state_dim` | 22,530 (30 × 751) |
| `global_state_dim` | 24,032 (32 × 751) |
| Replay buffer footprint | ~9.6 GB at buffer_size=100 |
| Observation encoding | Full one-hot — no lossy cap |
| Vehicle types | 3 (car 65%, moto 25%, truck 10%) |

---

## Remaining Open Items

| Item | Notes |
|------|-------|
| BGC Full 1k smoke test | Confirm pass with diverse types |
| Phase 7 gate doc | Write only after smoke test passes |
| Reward normalization | Grad norms ~2–3M; needed before full 200k training |
| 200k training — BGC Core | Civiq vs Mono-QMIX baseline |
| 200k training — BGC Full | After Phase 7 gate |
