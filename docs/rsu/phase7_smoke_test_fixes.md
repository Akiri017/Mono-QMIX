# Phase 7 — Smoke Test Fixes

**Date:** 2026-04-12
**Branch:** `rsu-placement`
**Scope:** Fixes applied during BGC Full and BGC Core smoke tests

---

## Fix 1 — main.py: `--alg_config` / `--env_config` missing `.yaml` extension

**Problem:** Passing `--alg_config civiq_sumo` caused a "config not found" error because
the pre-parser used the value as-is without appending `.yaml`, while the default
path did have it hardcoded.

**Fix:** Added `_ensure_yaml()` helper in `main.py`:
```python
def _ensure_yaml(name, default):
    if not name:
        return default
    return name if name.endswith(".yaml") else name + ".yaml"
```
Applied to both `alg_config_name` and `env_config_name`.

**File:** `pymarl/src/main.py`

---

## Fix 2 — evaluate.py: hardcoded 4x4 config paths

**Problem:** `evaluate.py` always loaded `qmix_sumo.yaml` + `sumo_grid4x4.yaml`
regardless of which scenario was being evaluated.

**Fix:** Added `--alg_config` and `--env_config` arguments (same pattern as `main.py`).
Paths are now derived from the argument values:
```python
alg_config_path = script_dir / "config" / "algs" / f"{args_cmd.alg_config}.yaml"
env_config_path = script_dir / "config" / "envs" / f"{args_cmd.env_config}.yaml"
```
Defaults preserved (`qmix_sumo`, `sumo_grid4x4`) so existing usage is unchanged.

**File:** `pymarl/src/evaluate.py`

---

## Fix 3 — trips files: merge conflict markers + invalid edges

**Problem:** `trips_low.xml` and `trips_med.xml` had unresolved git merge conflict
markers (`<<<<<<<`, `=======`, `>>>>>>>`), making them unparseable XML. All three
trips files (`low`, `med`, `highEnough`) also contained trips referencing edges
deleted during the netedit map edit, causing libsumo to raise a `TraCIException`
at `simulation.start()` even with `--ignore-route-errors` in the sumocfg.

**Counts after filtering:**

| File | Removed | Remaining |
|------|---------|-----------|
| `trips_low.xml` | 737 | 164 |
| `trips_med.xml` | 1,468 | 333 |
| `trips_highEnough.xml` | 5,854 | 1,347 |

**Fix:** Python script that:
1. Strips conflict markers keeping the HEAD side
2. Removes any `<trip>` whose `from=` or `to=` edge is absent from the current
   734-edge network (validated via `sumolib.net.readNet`)

**Files:** `bgc_full/trips_low.xml`, `bgc_full/trips_med.xml`, `bgc_full/trips_highEnough.xml`

---

## Fix 4 — controlled_init.rou.xml: stale edges from old network

**Problem:** `sumo/scenarios/bgc_full/controlled_init.rou.xml` was generated against
a pre-edit version of the network. Edge `743811528` (start of `ctrl_11`'s route)
no longer exists in the 734-edge network. The env's `_compute_shortest_route` calls
`sumolib.net.getEdge('743811528')` which raised a `KeyError`, causing that agent
slot to silently fail to spawn on every episode reset.

**Fix:** Regenerated via `generate_controlled_fleet.py`:
```bash
python scripts/generate_controlled_fleet.py \
    --net bgc_full/final_map.net.xml \
    --n 32 --seed 1 --depart-window 10 \
    --out-dir sumo/scenarios/bgc_full
```
All 32 OD pairs verified against current network — 0 bad edges.

**File:** `sumo/scenarios/bgc_full/controlled_init.rou.xml`

---

## Fix 5 — main.py: hardcoded "SUMO Grid 4x4" environment label

**Problem:** The training banner always printed `Environment: SUMO Grid 4x4`
regardless of `--env_config`.

**Fix:** Label now reads from `args["env_config_name"]`, set at config-load time:
```python
args["env_config_name"] = env_config_name.replace(".yaml", "")
# prints e.g. "Environment: sumo_bgc_core"
```

**File:** `pymarl/src/main.py`

---

## Fix 6 — hierarchical_q_learner.py: `.view()` on non-contiguous tensors

**Problem:** `learner.train()` crashed with:
```
RuntimeError: view size is not compatible with input tensor's size and stride
(at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
```
Tensors `local_states`, `global_states`, `agent_masks_per_rsu`, and their target
counterparts are produced by temporal slicing (`batch["local_states"][:, :-1]`,
`[:, 1:]`). Slicing along a non-last dimension creates non-contiguous views;
`.view()` requires contiguous memory and fails on these.

**Fix:** Replaced `.view()` with `.reshape()` for all pre-mixer reshapes in both
the online and target paths. `.reshape()` handles non-contiguous tensors by calling
`.contiguous()` internally when needed. The final `.view()` after the mixer output
(which is always contiguous) was left unchanged.

**Lines changed (online path):**
```python
# Before
local_states.view(BT * R, -1)
global_states.view(BT, -1)
rsu_mask.view(BT, R)

# After
local_states.reshape(BT * R, -1)
global_states.reshape(BT, -1)
rsu_mask.reshape(BT, R)
```
Same pattern applied to `target_local_states`, `target_global_states`, `target_rsu_mask`.

**File:** `pymarl/src/learners/hierarchical_q_learner.py`
