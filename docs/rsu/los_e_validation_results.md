# LOS E Validation Results — Synthetic 4×4 Map

**Script:** `scripts/los_e_zone_validation.py`  
**Scenario:** `sumo/scenarios/4by4_map/train_high.sumocfg` (~1406 veh/hr, LOS E)  
**Duration:** 3600 simulation steps (1 hour simulated time)  
**Date run:** 2026-04-06

---

## Peak Vehicles per RSU Zone

| RSU Index | Node ID | Peak Count |
|-----------|---------|-----------|
| 0  | J0  | 5  |
| 1  | J1  | 21 |
| 2  | J2  | 23 |
| 3  | J3  | 6  |
| 4  | J4  | 23 |
| 5  | J5  | 5  |
| 6  | J6  | 21 |
| 7  | J7  | 5  |
| 8  | J8  | 6  |
| 9  | J9  | 6  |
| 10 | J10 | 5  |
| 11 | J11 | 5  |

**Overall peak:** 23 (RSUs J2 and J4 — interior mid-column nodes)

---

## Derived Configuration Value

```
max_agents_per_rsu = peak + 5 buffer = 23 + 5 = 28
```

This value was written to `config/envs/rsu/synthetic_4x4.yaml`.

---

## Observations

- The two hotspot RSUs (J2 and J4) are interior nodes in the middle column of the grid, receiving vehicles from multiple incoming edges.
- Corner and edge nodes (J0, J3, J5, J7, J8–J11) carry at most 6 vehicles simultaneously at LOS E.
- The total live vehicle count during the run ranged from ~51 to 63 at the sampled checkpoints — the zones partition this cleanly with no overlap.
- The 5-vehicle buffer provides headroom for stochastic route variation in training runs that may temporarily exceed the LOS E baseline.
