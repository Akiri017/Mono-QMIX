#!/usr/bin/env python3
"""
Block A — BGC Full LOS E Data Collection

Runs a headless 3600-step SUMO simulation on BGC Full at LOS E
(wardrop_routes_highEnough.rou.xml) using libsumo. Logs all vehicle positions every
timestep and outputs:

  vehicle_positions_bgc_full_los_e.csv   — timestep, vehicle_id, x, y
  vehicle_density_bgc_full_los_e.png     — 2D spatial heatmap (all timesteps)

Peak simultaneous vehicle count is printed to stdout.

Usage (from repo root):
    python scripts/bgc/block_a_bgc_full.py

Output files are written to the same directory as this script.
"""

import os
import sys
import csv
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless — no display required
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parents[1]          # scripts/bgc/ -> scripts/ -> repo

BGC_DIR     = REPO_ROOT / "bgc_full"
NET_FILE    = BGC_DIR / "final_map.net.xml"
TRIPS_FILE  = BGC_DIR / "trips_highEnough.xml"   # OD trip demands (LOS E)
VTYPES_FILE = BGC_DIR / "vtypes.add.xml"

OUT_DIR      = SCRIPT_DIR                         # outputs land beside this script
ROUTES_FILE  = OUT_DIR / "bgc_full_los_e_routed.rou.xml"   # duarouter output
CSV_PATH     = OUT_DIR / "vehicle_positions_bgc_full_los_e.csv"
PNG_PATH     = OUT_DIR / "vehicle_density_bgc_full_los_e.png"

EPISODE_STEPS = 3600

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    for p in (NET_FILE, TRIPS_FILE, VTYPES_FILE):
        if not p.exists():
            sys.exit(f"[ERROR] File not found: {p}")

    print("BGC Full LOS E -- data collection run")
    print(f"  net:    {NET_FILE}")
    print(f"  trips:  {TRIPS_FILE}")
    print(f"  vtypes: {VTYPES_FILE}")
    print(f"  steps:  {EPISODE_STEPS}")
    print(f"  output: {OUT_DIR}")
    print()

    # ------------------------------------------------------------------
    # Step 0 — route trips against current network with duarouter
    # ------------------------------------------------------------------
    if not ROUTES_FILE.exists():
        print("Running duarouter to build valid routes from trips_highEnough.xml...")
        duarouter_cmd = [
            "duarouter",
            "--net-file",      str(NET_FILE),
            "--route-files",   str(TRIPS_FILE),
            "--additional-files", str(VTYPES_FILE),
            "--output-file",   str(ROUTES_FILE),
            "--ignore-errors", "true",
            "--no-warnings",   "true",
            "--no-step-log",   "true",
        ]
        result = subprocess.run(duarouter_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stderr[-2000:])
            sys.exit("[ERROR] duarouter failed — check that SUMO is on PATH.")
        print(f"Routes written to: {ROUTES_FILE}")
        print()
    else:
        print(f"Using cached routes: {ROUTES_FILE}")
        print()

    try:
        import libsumo as traci
    except ImportError:
        sys.exit("[ERROR] libsumo not found. Ensure SUMO is installed and "
                 "SUMO_HOME/tools is on PYTHONPATH.")

    sumo_cmd = [
        "sumo",
        "-n", str(NET_FILE),
        "-r", str(ROUTES_FILE),
        "-a", str(VTYPES_FILE),
        "--no-step-log",         "true",
        "--time-to-teleport",    "300",
        "--ignore-route-errors", "true",
        "--no-warnings",         "true",
    ]

    print("Starting SUMO...")
    traci.start(sumo_cmd)

    all_x      = []
    all_y      = []
    peak_count = 0
    row_count  = 0

    with open(CSV_PATH, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["timestep", "vehicle_id", "x", "y"])

        for step in range(EPISODE_STEPS):
            traci.simulationStep()

            veh_ids = traci.vehicle.getIDList()
            count   = len(veh_ids)
            if count > peak_count:
                peak_count = count

            for vid in veh_ids:
                x, y = traci.vehicle.getPosition(vid)
                writer.writerow([step + 1, vid, f"{x:.2f}", f"{y:.2f}"])
                all_x.append(x)
                all_y.append(y)
                row_count += 1

            if (step + 1) % 500 == 0 or step == 0:
                print(f"  step {step + 1:4d}/{EPISODE_STEPS}"
                      f"  vehicles={count:4d}"
                      f"  peak_so_far={peak_count:4d}")

    traci.close()

    print(f"\nPeak simultaneous vehicle count: {peak_count}")
    print(f"Total CSV rows written:          {row_count:,}")
    print(f"CSV saved to: {CSV_PATH}")

    # ------------------------------------------------------------------
    # Heatmap
    # ------------------------------------------------------------------
    print("\nGenerating density heatmap...")
    all_x = np.array(all_x, dtype=np.float32)
    all_y = np.array(all_y, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(18, 14))
    h = ax.hist2d(all_x, all_y, bins=150, cmap="hot_r")
    plt.colorbar(h[3], ax=ax, label="Vehicle-timestep count")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(
        f"BGC Full Vehicle Density -- LOS E ({EPISODE_STEPS} steps)\n"
        f"Peak simultaneous vehicles: {peak_count}"
    )
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=150)
    plt.close()
    print(f"Heatmap saved to: {PNG_PATH}")

    print("\nDone.")


if __name__ == "__main__":
    main()
