# Levels of Service (LOS) Implementation

## Overview

Implements configurable traffic demand levels (LOS) for the SUMO 4x4 grid environment. Each LOS corresponds to a different background vehicle spawn rate, enabling curriculum training across congestion levels.

---

## LOS Levels

| Level | Spawn Period | Approx. veh/hr | Background Route File              |
|-------|-------------|----------------|------------------------------------|
| low   | 6.4s        | ~562           | `wardrop_routes_low.rou.xml`       |
| med   | 3.2s        | ~1125          | `wardrop_routes_med.rou.xml`       |
| high  | 2.56s       | ~1406          | `wardrop_routes_high.rou.xml`      |

Spawn periods are sourced from the original `randomTrips.py` generation parameters embedded in each trip file header.

---

## Files Changed

### New Files

- `sumo/scenarios/4by4_map/train_low.sumocfg`
  - Points to `wardrop_routes_low.rou.xml` for low-demand background traffic
- `sumo/scenarios/4by4_map/train_high.sumocfg`
  - Points to `wardrop_routes_high.rou.xml` for high-demand background traffic
- `sumo/scenarios/4by4_map/train_med.sumocfg` *(pre-existing, updated)*
  - Points to `wardrop_routes_med.rou.xml` — updated for consistency with low/high

### Modified: `pymarl/src/config/envs/sumo_grid4x4.yaml`

Added `los_level` parameter and set `sumo_cfg` to `null`:

```yaml
los_level: "med"   # "low", "med", or "high"
sumo_cfg: null     # if null, derived automatically from los_level
```

### Modified: `pymarl/src/envs/sumo_grid_reroute.py`

Added resolution logic in `__init__` (after the file path block):

- Defines a `_LOS_CFG` mapping from level strings to sumocfg paths
- If `sumo_cfg` is `null`/empty, resolves it from `los_level`
- Raises `ValueError` on unrecognized `los_level` values
- Logs `los_level` and resolved `sumo_cfg` path when `verbose=True`

Backward compatible: if `sumo_cfg` is set explicitly, it takes precedence and `los_level` is ignored for path resolution.

---

## Usage

### Default (medium demand)
No changes needed — `los_level: "med"` is the default in the yaml.

### Per-run override for curriculum training
Pass `los_level` as a config override when launching a training run:

```bash
python pymarl/src/main.py env_args.los_level=low
python pymarl/src/main.py env_args.los_level=med
python pymarl/src/main.py env_args.los_level=high
```

### Explicit sumocfg override (bypass LOS)
Set `sumo_cfg` directly to point at any custom scenario:

```yaml
sumo_cfg: "sumo/scenarios/4by4_map/train_med.sumocfg"
```

---

## Curriculum Training Notes

- Each stage is a separate training run — LOS does not change mid-episode
- Suggested curriculum order: `low` → `med` → `high`
- Transfer learning: initialize weights from the previous stage's checkpoint before starting the next
- The `los_level` value is logged at environment init (verbose mode) for traceability in experiment logs
