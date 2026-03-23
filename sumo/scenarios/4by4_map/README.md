# SUMO scenario wrapper (4by4_map)

This folder provides a **canonical training config** that reuses the already-copied SUMO network and demand files in the repo root under `4by4_map/`.

## Key idea
- **Background traffic** is loaded from existing `.rou.xml` (realistic departures over time).
- **Controlled fleet** is initially inserted from `controlled_init.rou.xml` (fixed N departing early).
- **Replacement/lifelong agents** are implemented at runtime by the TraCI environment by removing/adding controlled vehicles as they arrive. SUMO configs do not encode the replacement logic.

## Files
- `train_med.sumocfg`: canonical config (explicit `step-length=1.0`)
- `controlled_init.rou.xml`: generated fixed-N controlled fleet initial vehicles

## Regenerating controlled_init
Run from repo root:
- `python scripts/generate_controlled_fleet.py --net 4by4_map/final_map.net.xml --n 32 --depart-window 10 --seed 1 --out-dir sumo/scenarios/4by4_map`
