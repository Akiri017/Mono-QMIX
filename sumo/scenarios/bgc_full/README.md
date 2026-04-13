# SUMO scenario wrapper (bgc_full)

This folder provides a **canonical training config** that reuses the already-copied SUMO network and demand files in the repo root under `bgc_full/`.

## Key idea
- **Background traffic** is loaded from existing wardrop `.rou.xml` files (realistic, equilibrium-routed departures over time).
- **Controlled fleet** is initially inserted from `controlled_init.rou.xml` (fixed N departing early).
- **Replacement/lifelong agents** are implemented at runtime by the TraCI environment by removing/adding controlled vehicles as they arrive. SUMO configs do not encode the replacement logic.

## Files
- `train_low.sumocfg`: low demand background (~900 veh/hr, period=4.0s)
- `train_med.sumocfg`: medium demand background (~1800 veh/hr, period=2.0s)
- `train_high.sumocfg`: high demand background (~4800 veh/hr, period=0.75s)
- `controlled_init.rou.xml`: generated fixed-N controlled fleet initial vehicles
- `controlled_init.trips.xml`: source trips used to generate the above

## Network
- **Source:** `bgc_full/final_map.net.xml` (Bonifacio Global City, full extent)
- **Edges:** 2723 non-internal edges (sumolib count)
- **Vehicle type:** `thesis_car` (defined in `bgc_full/vtypes.add.xml`, included via additional-files)

## LOS levels

| Level | Period  | Approx. veh/hr | Background route file                        |
|-------|---------|----------------|----------------------------------------------|
| low   | 4.0s    | ~900           | `bgc_full/wardrop_routes_low.rou.xml`        |
| med   | 2.0s    | ~1800          | `bgc_full/wardrop_routes_med.rou.xml`        |
| high  | 0.75s   | ~4800          | `bgc_full/wardrop_routes.rou.xml`            |

Spawn periods are sourced from the original `randomTrips.py` generation parameters embedded in each trip file header (`bgc_full/trips_*.xml`).

## Regenerating controlled_init
Run from repo root:
```
python scripts/generate_controlled_fleet.py --net bgc_full/final_map.net.xml --n 32 --depart-window 10 --seed 1 --out-dir sumo/scenarios/bgc_full
```

## Environment config
See `pymarl/src/config/envs/sumo_bgc_full.yaml` for the full PyMARL environment configuration.
