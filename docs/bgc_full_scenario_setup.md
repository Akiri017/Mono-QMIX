# Feature + Architecture: BGC Full Scenario Setup for Mono-QMIX

**Date:** 2026-04-11
**Status:** Partial — pipeline runs, observation representation known to be lossy
**Affected areas:**
- `pymarl/src/config/envs/sumo_bgc_full.yaml` (new)
- `pymarl/src/envs/sumo_grid_reroute.py`
- `sumo/scenarios/bgc_full/` (new directory)
- `pymarl/src/main.py`
- `run_experiments.py`

---

## Context

The thesis requires running Mono-QMIX on the full Bonifacio Global City (BGC) road network to evaluate scalability beyond the 4x4 synthetic grid and bgc_core. The bgc_full scenario (2723 non-internal edges) had no PyMARL environment config, no training sumocfgs, and no controlled fleet init file — only raw network and demand files in `bgc_full/`.

---

## Discussion and decisions

**`los_cfg` override mechanism instead of hardcoding bgc_full into env**
The existing `_LOS_CFG` dict in `sumo_grid_reroute.py` is hardcoded to 4x4 paths. Rather than adding bgc_full paths to the env class, a `los_cfg` key was added to `env_args` — if present, it replaces `_LOS_CFG` entirely. This keeps the env class scenario-agnostic; any new scenario adds its own LOS map in its yaml.

**`controlled_init.rou.xml` generated from bgc_full network**
The 4x4 scenario uses `generate_controlled_fleet.py` to produce a fixed-N controlled fleet init file. bgc_full had no equivalent. The same script was run against `bgc_full/final_map.net.xml` with the same parameters (`--n 32 --seed 1 --depart-window 10`) to produce `sumo/scenarios/bgc_full/controlled_init.rou.xml`.

**`wardrop_routes.rou.xml` for background traffic, not as `controlled_routes`**
Initially `wardrop_routes.rou.xml` was used as the `controlled_routes` source (OD pairs). This was corrected after reviewing the 4x4 README — the controlled fleet should come from its own dedicated init file, not the background traffic routes.

**`obs_edge_dim: 256` instead of full 2723**
The full one-hot encoding over 2723 edges requires ~130 GB replay buffer at `buffer_size: 400`, and ~28 GB even at `buffer_size: 400` with obs + state tensors. With 16 GB RAM, the only feasible option short of implementing learned embeddings is to cap `obs_edge_dim`. Set to 256 (~9% of network). This is a known representational limitation — see follow-up.

**`buffer_size: 100` override in env yaml**
Even with `obs_edge_dim: 256`, the default `buffer_size: 400` requires ~28 GB. Reduced to 100 in `sumo_bgc_full.yaml` (env yaml keys override alg yaml on merge), giving ~7 GB buffer footprint, which fits in 16 GB alongside SUMO and OS overhead. The 4x4 config is unaffected.

**Training sumocfg structure mirrors 4x4 pattern**
The `bgc_full/osm.sumocfg` is a baseline/selfish run config with output files — not suitable for repeated RL episodes. Clean training sumocfgs (`train_low/med/high.sumocfg`) were created in `sumo/scenarios/bgc_full/` following the exact 4x4 structure: minimal processing (only `time-to-teleport` and `ignore-junction-blocker`), no output files, explicit `step-length`. One difference from 4x4: `vtypes.add.xml` is included as additional-files because bgc_full wardrop routes reference `thesis_car` vehicle type; 4x4 uses the default vtype.

---

## What changed

### New files
- `pymarl/src/config/envs/sumo_bgc_full.yaml` — full PyMARL env config for bgc_full; includes `los_cfg` block, `obs_edge_dim: 256`, `buffer_size: 100` override
- `sumo/scenarios/bgc_full/train_low.sumocfg` — low demand (~900 veh/hr, wardrop_routes_low.rou.xml)
- `sumo/scenarios/bgc_full/train_med.sumocfg` — medium demand (~1800 veh/hr, wardrop_routes_med.rou.xml)
- `sumo/scenarios/bgc_full/train_high.sumocfg` — high demand (~4800 veh/hr, wardrop_routes.rou.xml)
- `sumo/scenarios/bgc_full/controlled_init.rou.xml` — 32 controlled vehicles, seed 1, generated via `generate_controlled_fleet.py`
- `sumo/scenarios/bgc_full/controlled_init.trips.xml` — source trips for the above
- `sumo/scenarios/bgc_full/README.md` — mirrors 4x4 README structure with bgc_full specifics

### Modified: `pymarl/src/envs/sumo_grid_reroute.py`
Added `los_cfg` override block in `__init__` (after `_LOS_CFG` definition):
```python
_los_cfg_override = env_args.get("los_cfg", None)
if _los_cfg_override:
    _LOS_CFG = _los_cfg_override
```
If `env_args["los_cfg"]` is present, it replaces the hardcoded 4x4 `_LOS_CFG`. Fully backward compatible — 4x4 yaml has no `los_cfg` key so it falls through to the existing map.

### Modified: `pymarl/src/main.py`
Two changes to support runtime env config and LOS level selection:

1. **Pre-parse `--env_config` before `load_config()`** — uses a secondary `ArgumentParser(add_help=False)` to read `--env_config` early, since the env yaml path must be known before the config is loaded:
```python
_pre = _ap.ArgumentParser(add_help=False)
_pre.add_argument("--env_config", type=str, default="sumo_grid4x4")
_pre_args, _ = _pre.parse_known_args()
env_config_path = script_dir / "config" / "envs" / f"{_pre_args.env_config}.yaml"
```

2. **`--env_config` and `--los_level` in main argparse** — both added to the main `ArgumentParser` for `--help` visibility. `--los_level` overrides `args["env_args"]["los_level"]` after config is loaded:
```python
if cmd_args.los_level is not None:
    args["env_args"]["los_level"] = cmd_args.los_level
```

### Modified: `run_experiments.py`
- Added `--env_config` argument (default: `"sumo_grid4x4"`) — always forwarded to `main.py` via `train_extra`
- Added `--los_level` argument (choices: `low`, `med`, `high`) — forwarded to `main.py` via `train_extra` when set

---

## Implementation notes

- `sumolib.net.getEdges()` returns only non-internal edges for bgc_full — confirmed 2723 edges. Internal junction edges (prefixed `:`) are excluded by sumolib by default.
- The `_resolve_path()` method in the env resolves all paths relative to repo root (3 levels up from `pymarl/src/envs/`). All yaml paths use this convention.
- The pre-parser pattern in `main.py` is necessary because Python's `argparse` does not support reading args before calling `parse_args()` cleanly. `parse_known_args()` on a minimal pre-parser avoids conflicts with the full parser defined later.
- `buffer_size` placed in the env yaml works because `load_config()` merges alg and env configs as `{**alg_config, **env_config}` — env keys win. This is architecturally impure (buffer size is an alg concern) but avoids creating a bgc_full-specific alg yaml.

---

## Runtime commands

```bash
# Full pipeline — bgc_full, high demand (yaml default), seed 1, 500k steps
python run_experiments.py --env_config sumo_bgc_full --seeds 1 --t_max 500000

# Override LOS level
python run_experiments.py --env_config sumo_bgc_full --seeds 1 --t_max 500000 --los_level med

# Resume from checkpoint directly
python pymarl/src/main.py --env_config sumo_bgc_full --resume_from results/models/seed1/best

# Resume with LOS override
python pymarl/src/main.py --env_config sumo_bgc_full --los_level low --resume_from results/models/seed1/best
```

---

## Status and follow-up

- **Status:** Partial — bgc_full pipeline is runnable end-to-end, but the observation representation is lossy: only the 256 lowest-indexed edges (by sumolib ordering) receive a one-hot signal. The remaining ~91% of the network (2467 edges) produce all-zeros in the edge slot, meaning agents on those edges are positionally indistinguishable from each other.

- **Follow-up items:**
  - **Implement learned edge embedding** — replace the one-hot edge slot with `nn.Embedding(n_edges, embed_dim)` (e.g., embed_dim=64). Requires: (1) env outputs integer edge index instead of one-hot vector, (2) agent network adds embedding lookup before RNN input, (3) buffer scheme uses `dtype=torch.long` for the edge index field. This fixes both the memory problem and the representational gap, making bgc_full results meaningful.
  - **Evaluate whether bgc_full results are thesis-valid at `obs_edge_dim=256`** — if bgc_full is used as a primary comparison scenario, document the limitation explicitly and consider whether results are attributable to the policy or the broken observation.
  - **Add `--env_config` support to `evaluate.py`** — currently `evaluate.py` also hardcodes the 4x4 env config; bgc_full evaluation runs will need the same treatment applied to `main.py`.
