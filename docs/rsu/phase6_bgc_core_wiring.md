# Phase 6 — BGC Core Environment Wiring

## Purpose

Phase 6 wires the BGC Core real-world map into the Civiq training pipeline. This covers RSU placement finalisation, per-zone peak validation, environment config, sumocfg files, and CLI extensions — everything needed to run `HierarchicalQLearner` on BGC Core end-to-end.

---

## Files Created / Modified

| File | Action | Description |
|------|--------|-------------|
| `config/envs/rsu/bgc_core.yaml` | Created | 8-RSU zone config for BGC Core |
| `scripts/bgc/block_b_zone_peak_count.py` | Created | Per-zone peak count from collected CSV |
| `scripts/bgc/rsu_placed.add.xml` | Updated | Confirmed placement visualization (all orange) |
| `pymarl/src/config/envs/sumo_bgc_core.yaml` | Created | Full BGC Core env config |
| `bgc_core/train_low.sumocfg` | Created | LOS low sumocfg (trips_low.xml + final_map.net.xml) |
| `bgc_core/train_med.sumocfg` | Created | LOS med sumocfg |
| `bgc_core/train_high.sumocfg` | Created | LOS high sumocfg (LOS E — used for RSU validation) |
| `pymarl/src/envs/sumo_grid_reroute.py` | Updated | `_LOS_CFG` made configurable via `env_args["los_cfg"]` |
| `pymarl/src/main.py` | Updated | Added `--env_config` CLI argument |
| `docs/civiq_implementation_summary.md` | Created | Full end-to-end implementation summary |

---

## RSU Placement — Final Configuration

8 RSUs at BGC Core traffic-light intersections. Placement went through two iterations:

**Iteration 1 (8 RSUs, user-selected via netedit):**

| RSU | x | y | Peak |
|-----|---|---|------|
| RSU00 | 37.6 | 221.4 | 41 |
| RSU01 | 228.0 | 159.4 | **54** ← over ceiling |
| RSU02 | 417.3 | 95.0 | 37 |
| RSU03 | 606.9 | 30.2 | 28 |
| RSU04 | 676.0 | 220.0 | 24 |
| RSU05 | 481.0 | 284.8 | 41 |
| RSU06 | 386.5 | 317.4 | 45 |
| RSU07 | 426.4 | 417.6 | 15 ← redundant (108m from RSU06) |

Issues: RSU01 exceeded 50-vehicle ceiling; RSU07 starved due to proximity to RSU06.

**Iteration 2 (revised, confirmed):**
- Removed old RSU07 (426.4, 417.6)
- Added new RSU07 at (133.0, 190.0) — midpoint between RSU00 and RSU01, splitting RSU01's load

| RSU | x | y | Peak |
|-----|---|---|------|
| RSU00 | 37.6 | 221.4 | 26 |
| RSU01 | 228.0 | 159.4 | 49 |
| RSU02 | 417.3 | 95.0 | 37 |
| RSU03 | 606.9 | 30.2 | 28 |
| RSU04 | 676.0 | 220.0 | 24 |
| RSU05 | 481.0 | 284.8 | 42 |
| RSU06 | 386.5 | 317.4 | **50** |
| RSU07 | 133.0 | 190.0 | 29 |

Overall peak: **50** — exactly at ceiling, zero vehicles masked. `max_agents_per_rsu: 50` validated.

---

## Block B — Zone Peak Count (`block_b_zone_peak_count.py`)

Reads `vehicle_positions_bgc_core_los_e.csv` (179,470 rows from Block A), assigns each vehicle at each timestep to nearest RSU within 300m, reports peak simultaneous count per zone.

```
python scripts/bgc/block_b_zone_peak_count.py
```

Output (iteration 2): overall peak = 50, recommended `max_agents_per_rsu = 50`.

---

## BGC Core Network Facts

Extracted from `bgc_core/final_map.net.xml` via sumolib:

| Property | Value |
|----------|-------|
| Edges (no internal) | 184 |
| Total junctions | 129 |
| Traffic-light junctions | 8 |
| Map extent (approx.) | 700 × 450 m |

Derived dimensions:
- `obs_dim = 5 + 184 + (4 × 3) = 201`
- `state_dim = 32 × 201 = 6432`
- `local_state_dim = 50 × 201 = 10,050`

---

## Environment Config (`sumo_bgc_core.yaml`)

Key differences vs `sumo_grid4x4.yaml`:

| Key | 4×4 | BGC Core |
|-----|-----|----------|
| `network_file` | `4by4_map/final_map.net.xml` | `bgc_core/final_map.net.xml` |
| `controlled_routes` | `sumo/scenarios/4by4_map/controlled_init.rou.xml` | `bgc_core/trips_low.xml` |
| `n_actions` | 2 | 4 |
| `obs_edge_dim` | 48 | 184 |
| `obs_dim` | 65 | 201 |
| `state_shape` | 2080 | 6432 |
| `global_state_dim` | 2080 | 6432 |
| `max_agents_per_rsu` | 28 | 50 |
| `rsu_config` | `synthetic_4x4.yaml` | `bgc_core.yaml` |

The env config overrides Civiq alg defaults at the top level:
```yaml
obs_dim: 201
global_state_dim: 6432
max_agents_per_rsu: 50
rsu_config: "config/envs/rsu/bgc_core.yaml"
```

---

## Fixes Applied During Smoke Test

### 1 — `--env_config` CLI argument (`main.py`)

Previously the env config was hardcoded to `sumo_grid4x4.yaml`. Added `--env_config` to both the pre-parser and full parser:

```python
parser.add_argument("--env_config", type=str, default=None,
                    help="Environment config filename under config/envs/")
env_config_name = pre_args.env_config if pre_args.env_config else "sumo_grid4x4.yaml"
```

### 2 — `_LOS_CFG` configurable (`sumo_grid_reroute.py`)

`_LOS_CFG` was hardcoded to 4×4 sumocfg paths. Made it read from `env_args`:

```python
_LOS_CFG = env_args.get("los_cfg", {
    "low":  "sumo/scenarios/4by4_map/train_low.sumocfg",
    "med":  "sumo/scenarios/4by4_map/train_med.sumocfg",
    "high": "sumo/scenarios/4by4_map/train_high.sumocfg",
})
```

BGC Core provides its own map via `los_cfg` in `sumo_bgc_core.yaml`:
```yaml
los_cfg:
  low:  "bgc_core/train_low.sumocfg"
  med:  "bgc_core/train_med.sumocfg"
  high: "bgc_core/train_high.sumocfg"
```

### 3 — `controlled_routes` → `trips_low.xml`

`bgc_core/routes.rou.xml` was routed against an older OSM network version — its edge IDs were not present in `final_map.net.xml`. Switched to `trips_low.xml` (generated directly from `final_map.net.xml`), which the env uses only for OD pair extraction.

### 4 — BGC Core sumocfg files

Created `bgc_core/train_{low,med,high}.sumocfg` pointing to `final_map.net.xml` + the corresponding trips file + `vtypes.add.xml`.

---

## Smoke Test Result

```
python main.py --alg_config civiq_sumo.yaml --env_config sumo_bgc_core.yaml \
               --t_max 1000 --no_validation --batch_size 1 --buffer_size 2
```

| Metric | Value |
|--------|-------|
| Episodes completed | 10 |
| Timesteps | 1000 |
| avg_return (test at t=100) | −114,848 |
| mean_travel_time | 7.4 s |
| arrival_rate | 68.1% |
| NaN / crash | None |

**PHASE 6 GATE PASSED**

---

## Run Command

```bash
# From repo root
cd pymarl/src
python main.py --alg_config civiq_sumo.yaml --env_config sumo_bgc_core.yaml --los_level low
```

---

## Open Items

| Item | Status |
|------|--------|
| 10k smoke test on BGC Core | Pending |
| Reward normalization (grad norms ~2–3M) | Pending |
| `max_rsus` global lock to BGC Full node count | Pending |
| Full 200k training run + comparison vs Mono QMIX baseline | Pending |
