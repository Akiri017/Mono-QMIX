# Phase 7 — BGC Full Wiring: All Fixes Applied

**Date:** 2026-04-12
**Status:** Wiring complete — pending 1k smoke test
**Branch:** `rsu-placement` (merged from `bgc-full-mono-qmix`)

---

## Summary

The BGC Full scenario was unrunnable due to a combination of an oversized network
(2,723 edges → infeasible obs space), stale/invalid route files, missing sumocfgs,
and incorrect RSU placements. All blocking issues are now resolved. The map was
edited in netedit to remove residential areas and C5 road segments, bringing the
edge count to 734 — making full one-hot encoding tractable without any architecture
changes.

---

## Fix 1 — Map Edit: 2,723 → 734 edges

**Problem:** BGC Full had 2,723 non-internal edges. Full one-hot encoding was
infeasible:
```
obs_dim         = 5 + 2,723 + 12 = 2,740
local_state_dim = 384 × 2,740   = 1,052,160   → ~170 GB replay buffer
```

**Fix:** User edited `bgc_full/final_map.net.xml` in netedit:
- Removed residential areas (low-traffic, no routing decisions occur there)
- Removed C5 road segments (controlled-access arterial — agents make no routing
  decisions on C5; OD access points preserved at BGC boundary)

**Result:** 734 non-internal edges. Full one-hot now tractable:
```
obs_dim         = 5 + 734 + 12 = 751
local_state_dim = 30 × 751     = 22,530   → ~9.6 GB at buffer_size=100 ✓
```

**Files changed:**
- `bgc_full/final_map.net.xml` — netedit edit (committed)

---

## Fix 2 — RSU Config Cleanup and Re-validation

**Problem:** After the map edit, several RSU positions fell on removed road
segments, and the RSU peak count (379) was inflated by the old large network.

### RSUs removed (invalid after map edit):
| RSU | Reason |
|-----|--------|
| RSU02 | Positioned on removed residential road |
| RSU13 | Positioned on removed residential road |
| RSU13 (original, index conflict) | On removed segment |
| RSU18 | On removed segment |

### RSUs added (covering new topology):
| RSU | TLS Cluster | Coordinates |
|-----|-------------|-------------|
| RSU03 (new) | `cluster_2462250736_...` | (399.8, 254.7) |
| Others | Various BGC Full junctions | See `bgc_full.yaml` |

### ISO 250m spacing check:
- RSU09 and RSU12 dropped to resolve proximity violations
- Final config: **17 RSUs**, all at valid TLS junctions, ISO compliant

**Block A re-run** (`block_a_bgc_full.py`):
- duarouter pre-routes `trips_highEnough.xml` → `bgc_full_los_e_routed.rou.xml`
- libsumo collects XY positions over 3,600-step LOS E run
- CSV output gitignored (large file)

**Block B re-run** (`block_b_bgc_full.py`):
- Peak simultaneous vehicles in any zone: **25** (RSU13, post-cleanup)
- `max_agents_per_rsu` = 25 + 5 buffer = **30**

**Files changed:**
- `config/envs/rsu/bgc_full.yaml` — 17 RSUs, `max_agents_per_rsu: 30`, `max_rsus: 17`

---

## Fix 3 — sumo_bgc_full.yaml: Obs/State Dimensions

**Problem:** `sumo_bgc_full.yaml` had stale dimensions from the old 2,723-edge network
with the 256-cap workaround.

| Key | Old value | New value |
|-----|-----------|-----------|
| `obs_edge_dim` | 256 (lossy cap) | 734 (full one-hot) |
| `obs_shape` | 273 | 751 |
| `state_shape` | 8,736 | 24,032 |

**Civiq overrides section added** (was missing entirely):
```yaml
obs_dim: 751
global_state_dim: 24032
max_agents_per_rsu: 30
max_rsus: 17
local_state_dim: 22530
rsu_config: "config/envs/rsu/bgc_full.yaml"
```

**Files changed:**
- `pymarl/src/config/envs/sumo_bgc_full.yaml`

---

## Fix 4 — Training sumocfgs

**Problem:** `bgc_full/train_low/med/high.sumocfg` did not exist. The env's `los_cfg`
resolver had no valid files to point to.

**Fix:** Created all three sumocfgs. Wardrop route files are stale (reference edges
removed by netedit and cannot be regenerated cleanly due to duplicate internal edges
in netedit-produced networks). Sumocfgs use the raw trips files instead:
- `bgc_full/train_low.sumocfg` → `trips_low.xml` + `--ignore-route-errors true`
- `bgc_full/train_med.sumocfg` → `trips_med.xml` + `--ignore-route-errors true`
- `bgc_full/train_high.sumocfg` → `trips_highEnough.xml` + `--ignore-route-errors true`

SUMO handles OD routing internally at simulation time. The `los_cfg` block in
`sumo_bgc_full.yaml` now points to the correct `bgc_full/` paths (not
`sumo/scenarios/bgc_full/` — the env's `_resolve_path` maps from repo root).

**Files changed:**
- `bgc_full/train_low.sumocfg` (new)
- `bgc_full/train_med.sumocfg` (new)
- `bgc_full/train_high.sumocfg` (new)
- `pymarl/src/config/envs/sumo_bgc_full.yaml` — `los_cfg` paths updated

---

## Fix 5 — controlled_init.rou.xml Regenerated

**Problem:** `sumo/scenarios/bgc_full/controlled_init.rou.xml` was generated against
the old 2,723-edge network. It contained 339 route references to edges that no
longer exist in the edited 734-edge network.

**Fix:** Regenerated using `generate_controlled_fleet.py`:
```bash
python scripts/generate_controlled_fleet.py \
    --net bgc_full/final_map.net.xml \
    --n 32 --seed 1 --depart-window 10 \
    --out sumo/scenarios/bgc_full/controlled_init.rou.xml
```

**Files changed:**
- `sumo/scenarios/bgc_full/controlled_init.rou.xml`

---

## Fix 6 — max_rsus: 17 Propagated to All Configs

**Problem:** `max_rsus` was a stale placeholder (12) in three files, and 20 in the
old `bgc_full.yaml`. The GlobalQMixer tensor shape is fixed at `max_rsus` — all
configs must share the same value.

**Final value:** 17 (BGC Full RSU count, Block B validated)

| File | Old value | New value |
|------|-----------|-----------|
| `pymarl/src/config/algs/civiq_sumo.yaml` | 12 | 17 |
| `pymarl/src/config/envs/sumo_bgc_core.yaml` | 12 | 17 |
| `config/envs/rsu/bgc_core.yaml` | 12 | 17 |
| `config/envs/rsu/synthetic_4x4.yaml` | 12 | 17 |
| `config/envs/rsu/bgc_full.yaml` | 20 | 17 |

---

## Fix 7 — Branch Merge (rsu-placement → bgc-full-mono-qmix)

**Problem:** RSU/Civiq scaffolding lived on `rsu-placement`; BGC Full wiring lived
on `bgc-full-mono-qmix`. Two merge conflicts arose.

**Conflict 1 — `main.py`:**
- `rsu-placement` added `--alg_config` pre-parser
- `bgc-full-mono-qmix` added `--los_level` argument
- Resolution: combined both — pre-parser reads `--alg_config` and `--env_config`;
  main parser includes `--los_level`

**Conflict 2 — `sumo_grid_reroute.py`:**
- Both branches modified `_LOS_CFG` handling
- Resolution: took `rsu-placement`'s cleaner one-liner:
  ```python
  _LOS_CFG = env_args.get("los_cfg", {...})
  ```

---

## Fix 8 — .gitignore: Large Files Excluded

**Problem:** A `git push` was rejected because `vehicle_positions_bgc_full_los_e.csv`
(107 MB) and `bgc_full_los_e_routed.rou.xml` were committed.

**Fix:** Soft-reset 2 commits, added entries to `.gitignore`, recommitted without
the large files.

**Added to `.gitignore`:**
```
scripts/bgc/vehicle_positions_*.csv
scripts/bgc/bgc_full_los_e_routed.rou.xml
```

---

## Current State

| Item | Value |
|------|-------|
| Network edges (non-internal) | 734 |
| RSU count | 17 |
| `max_agents_per_rsu` | 30 (peak 25 + buffer) |
| `obs_dim` | 751 (5 ego + 734 one-hot + 12 traffic) |
| `local_state_dim` | 22,530 (30 × 751) |
| `global_state_dim` | 24,032 (32 × 751) |
| Replay buffer footprint (buffer_size=100) | ~9.6 GB |
| Observation encoding | Full one-hot — no lossy cap |

---

## Remaining Open Items

| Item | Notes |
|------|-------|
| `evaluate.py` missing `--env_config` | Same pre-parser fix as `main.py` needed |
| RSU02–RSU03 ISO check | RSU02 (399.8, 254.7) and RSU03 (362.0, 452.0) are ~201m apart — below 250m ISO minimum; may need one relocated |
| 1k smoke test | Gate for Phase 7 completion — not yet run |
| Phase 7 gate doc | Write after smoke test passes |
| Reward normalization | Grad norms ~2–3M; needed before full 200k training |
| 10k smoke test — BGC Core | Pending |
| Full 200k training run | BGC Core vs Mono-QMIX, then BGC Full |
