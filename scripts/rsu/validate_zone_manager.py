"""
Phase 0 Gate — Zone Manager Sanity Check

Runs a 100-step SUMO episode and asserts per-step:
  1. No vehicle assigned to more than one zone (no duplicates)
  2. Every vehicle in traci.vehicle.getIDList() appears in exactly one zone
  3. No zone exceeds max_agents_per_rsu

Prints "PHASE 0 GATE PASSED" on success, or a detailed failure message.

Run from repo root:
    python scripts/rsu/validate_zone_manager.py
"""

import os
import sys
import yaml

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(repo_root, "pymarl", "src"))

from components.rsu_zone_manager import RSUZoneManager

import traci

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Use med-demand for the sanity check (stable, not too sparse)
SUMOCFG = os.path.join(repo_root, "sumo", "scenarios", "4by4_map", "train_med.sumocfg")
RSU_YAML = os.path.join(repo_root, "config", "envs", "rsu", "synthetic_4x4.yaml")
SIM_STEPS = 100


def main():
    with open(RSU_YAML) as f:
        rsu_cfg = yaml.safe_load(f)
    zone_manager = RSUZoneManager(rsu_cfg)
    max_cap = zone_manager.max_agents_per_rsu
    n_rsus = zone_manager.n_rsus
    print(f"RSU zone manager: {n_rsus} RSUs, max_agents_per_rsu={max_cap}")

    sumo_cmd = [
        "sumo",
        "-c", SUMOCFG,
        "--step-length", "1.0",
        "--no-step-log", "true",
        "--time-to-teleport", "-1",
        "--no-warnings",
    ]

    print(f"Starting SUMO for {SIM_STEPS}-step validation run...")
    traci.start(sumo_cmd)

    passed = True

    try:
        for step in range(SIM_STEPS):
            traci.simulationStep()

            veh_ids_set = set(traci.vehicle.getIDList())
            positions = {v: traci.vehicle.getPosition(v) for v in veh_ids_set}
            zones = zone_manager.assign_vehicles_to_zones(positions)

            # Collect all assigned vehicles across zones
            all_assigned = []
            for rsu_idx in range(n_rsus):
                all_assigned.extend(zones[rsu_idx])

            all_assigned_set = set(all_assigned)

            # Assertion 1: no duplicates (each vehicle in at most one zone)
            if len(all_assigned) != len(all_assigned_set):
                duplicates = [v for v in all_assigned if all_assigned.count(v) > 1]
                print(f"[FAIL] Step {step} — Assertion 1 FAILED: duplicates found")
                print(f"  Duplicate vehicles: {list(set(duplicates))}")
                passed = False
                break

            # Assertion 2: every traci vehicle appears in exactly one zone
            if all_assigned_set != veh_ids_set:
                missing = veh_ids_set - all_assigned_set
                extra = all_assigned_set - veh_ids_set
                print(f"[FAIL] Step {step} — Assertion 2 FAILED: vehicle coverage mismatch")
                if missing:
                    print(f"  Missing from zones: {missing}")
                if extra:
                    print(f"  Extra in zones (not in traci): {extra}")
                passed = False
                break

            # Assertion 3: no zone exceeds max_agents_per_rsu
            for rsu_idx in range(n_rsus):
                count = len(zones[rsu_idx])
                if count > max_cap:
                    print(f"[FAIL] Step {step} — Assertion 3 FAILED: RSU {rsu_idx} "
                          f"has {count} vehicles > max_agents_per_rsu ({max_cap})")
                    print(f"  Zone {rsu_idx} vehicles: {zones[rsu_idx]}")
                    passed = False
                    break
            if not passed:
                break

    finally:
        traci.close()
        print("SUMO closed.")

    if passed:
        print("\nPHASE 0 GATE PASSED")
    else:
        print("\n[GATE FAILED] Fix the issues above before proceeding to Phase 1.")
        sys.exit(1)


if __name__ == "__main__":
    main()
