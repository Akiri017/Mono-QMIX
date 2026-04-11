"""
LOS E Zone Validation — Phase 0, Civiq Scaffolding

Runs a 3600-step headless SUMO simulation using the highest-demand
scenario (train_high.sumocfg, ~1406 veh/hr) and records the peak
vehicle count per RSU zone.

Prints:
  - Peak count per zone
  - Recommended max_agents_per_rsu = peak + 5

Run from repo root:
    python scripts/rsu/los_e_zone_validation.py
"""

import os
import sys
import yaml

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(repo_root, "pymarl", "src"))

from components.rsu_zone_manager import RSUZoneManager

# Use traci (socket-based) for standalone scripts — no libsumo dependency
import traci

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SUMOCFG = os.path.join(repo_root, "sumo", "scenarios", "4by4_map", "train_high.sumocfg")
RSU_YAML = os.path.join(repo_root, "config", "envs", "rsu", "synthetic_4x4.yaml")
SIM_STEPS = 3600


def main():
    # Load RSU config
    with open(RSU_YAML) as f:
        rsu_cfg = yaml.safe_load(f)
    zone_manager = RSUZoneManager(rsu_cfg)
    n_rsus = zone_manager.n_rsus
    rsu_ids = zone_manager.rsu_ids
    print(f"RSU zone manager loaded: {n_rsus} RSUs")

    # Build SUMO command (mirror sumo_grid_reroute._start_sumo pattern)
    sumo_cmd = [
        "sumo",
        "-c", SUMOCFG,
        "--step-length", "1.0",
        "--no-step-log", "true",
        "--time-to-teleport", "-1",
        "--no-warnings",
    ]

    print(f"Starting SUMO: {' '.join(sumo_cmd)}")
    traci.start(sumo_cmd)

    peak_per_zone = [0] * n_rsus
    peak_total = 0

    try:
        for step in range(SIM_STEPS):
            traci.simulationStep()

            veh_ids = traci.vehicle.getIDList()
            positions = {v: traci.vehicle.getPosition(v) for v in veh_ids}
            zones = zone_manager.assign_vehicles_to_zones(positions)

            for rsu_idx in range(n_rsus):
                count = len(zones[rsu_idx])
                if count > peak_per_zone[rsu_idx]:
                    peak_per_zone[rsu_idx] = count

            peak_total = max(peak_total, sum(len(v) for v in zones.values()))

            if step % 300 == 0:
                total = len(veh_ids)
                print(f"  step {step:4d}/{SIM_STEPS} | vehicles in sim: {total}")

    finally:
        traci.close()
        print("SUMO closed.")

    # Report results
    overall_peak = max(peak_per_zone)
    recommended = overall_peak + 5

    print("\n--- Peak vehicles per RSU zone ---")
    for i, rsu_id in enumerate(rsu_ids):
        print(f"  RSU {i:2d} ({rsu_id:>4s}): peak = {peak_per_zone[i]}")

    print(f"\nOverall peak across all zones : {overall_peak}")
    print(f"Recommended max_agents_per_rsu: {recommended}  (peak + 5 buffer)")
    print(f"\npeak_total_agents: {peak_total}")
    print(f"global_state_dim should be:    {peak_total * 65}")


if __name__ == "__main__":
    main()
