# Phase 0 — RSU Zone Manager

## Purpose

Phase 0 implements the **RSU Zone Manager**, the foundational component for the hierarchical Civiq architecture. It partitions the SUMO network into spatial zones (one per intersection/RSU) and assigns every simulated vehicle to its nearest zone at each timestep.

This component is **standalone** — it runs independently of the RL environment wrapper and has no dependencies on `sumo_grid_reroute.py`. Integration into the environment wrapper happens in Phase 1 (LocalQMixer).

---

## Files Created

| File | Description |
|------|-------------|
| `scripts/rsu/extract_rsu_coords.py` | One-off script: reads `4by4_map/final_map.net.xml` via sumolib and dumps node id/x/y to `scripts/rsu/rsu_coords_synthetic_4x4.txt` |
| `scripts/rsu/rsu_coords_synthetic_4x4.txt` | Raw coordinate output (12 nodes) |
| `config/envs/rsu/synthetic_4x4.yaml` | RSU zone config for the synthetic 4×4 map |
| `pymarl/src/components/rsu_zone_manager.py` | `RSUZoneManager` class |
| `pymarl/src/components/__init__.py` | Updated to export `RSUZoneManager` |
| `scripts/rsu/los_e_zone_validation.py` | LOS E (train_high) 3600-step peak measurement |
| `scripts/rsu/validate_zone_manager.py` | Phase 0 gate — 100-step assertion suite |

---

## Network Topology

The synthetic map (`4by4_map/final_map.net.xml`) contains **12 intersection nodes**, arranged in a 3×4 grid (named "4×4" by project convention).

Coordinate ranges:
- X: −600 to +900 (SUMO net units, metres)
- Y: −500 to +1000

---

## RSU Zone Config (`config/envs/rsu/synthetic_4x4.yaml`)

```yaml
radius: 300.0
max_agents_per_rsu: 28     # peak 23 + 5 buffer — LOS E validated
max_rsus: 12
rsu_positions:
  - {id: J0, x: 900.0, y: 500.0}
  ...  # 12 entries total
```

**`radius`** is stored for future use (e.g., filtering observations) but the zone manager currently assigns every vehicle to its nearest RSU regardless of distance — no vehicle is ever left unassigned.

**`max_agents_per_rsu: 28`** was determined empirically:
- Ran `train_high.sumocfg` (LOS E, ~1406 veh/hr) for 3600 seconds
- Observed peak per zone = 23 vehicles (RSUs J2 and J4)
- Added 5-vehicle buffer → 28

---

## RSUZoneManager API

```python
from components.rsu_zone_manager import RSUZoneManager

zm = RSUZoneManager(rsu_config_dict)

# Assign vehicles to nearest RSU
zones = zm.assign_vehicles_to_zones({"v0": (x, y), "v1": (x, y), ...})
# → {0: ["v0", ...], 1: [...], ..., 11: [...]}

# Active-RSU mask (for LocalQMixer padding)
mask = zm.get_rsu_mask()
# → np.ndarray shape (max_rsus,), 1.0 for active slots
```

### Guarantees

- Every vehicle appears in **exactly one** zone list.
- All `n_rsus` zone keys are always present (empty list if no vehicles nearby).
- Tie-breaking: lower RSU index wins (numpy `argmin` default).
- Vectorised via numpy — O(n_vehicles × n_rsus) distance matrix.

---

## Phase 0 Gate Result

```
PHASE 0 GATE PASSED
```

Ran `scripts/rsu/validate_zone_manager.py` against `train_med.sumocfg`, 100 steps. All three assertions passed for every step:
1. No vehicle assigned to more than one zone
2. Every traci vehicle covered by exactly one zone
3. No zone exceeded `max_agents_per_rsu = 28`

---

## Next Step

Phase 1 — **LocalQMixer**: integrate `RSUZoneManager` into `sumo_grid_reroute.py` and implement per-zone local Q-value mixing.
