#!/usr/bin/env python3
"""
Block B — Per-Zone Peak Vehicle Count

Reads vehicle_positions_bgc_core_los_e.csv and the 8-RSU bgc_core.yaml,
assigns each vehicle at each timestep to its nearest RSU within 300 m,
and reports the peak simultaneous vehicle count per zone.

The maximum over all RSUs is the value to use for max_agents_per_rsu
in config/envs/rsu/bgc_core.yaml (with a +5 buffer recommended).

Usage (from repo root):
    python scripts/bgc/block_b_zone_peak_count.py

Outputs a summary table to stdout; no files written.
"""

import csv
import math
from collections import defaultdict
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parents[1]

CSV_PATH    = SCRIPT_DIR / "vehicle_positions_bgc_core_los_e.csv"
RSU_CONFIG  = REPO_ROOT / "config" / "envs" / "rsu" / "bgc_core.yaml"

# ---------------------------------------------------------------------------
# Load RSU positions
# ---------------------------------------------------------------------------
def load_rsus(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    radius = cfg["radius"]
    rsus = [(r["id"], r["x"], r["y"]) for r in cfg["rsu_positions"]]
    return rsus, radius


def nearest_rsu(vx, vy, rsus, radius):
    """Return RSU id of closest RSU within radius, or None."""
    best_id, best_dist = None, float("inf")
    for rsu_id, rx, ry in rsus:
        d = math.hypot(vx - rx, vy - ry)
        if d < best_dist and d <= radius:
            best_id, best_dist = rsu_id, d
    return best_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rsus, radius = load_rsus(RSU_CONFIG)
    print(f"Loaded {len(rsus)} RSUs from {RSU_CONFIG.name}  (radius={radius} m)")
    print(f"Reading CSV: {CSV_PATH}")
    print()

    # peak_count[rsu_id] = max simultaneous vehicles ever in that zone
    peak_count = defaultdict(int)
    # current_count[timestep][rsu_id] = vehicle count at that step
    # We only need per-timestep snapshots, not the full mapping.

    prev_step = None
    step_counts = defaultdict(int)   # rsu_id -> count for current timestep

    def flush_step():
        for rsu_id, cnt in step_counts.items():
            if cnt > peak_count[rsu_id]:
                peak_count[rsu_id] = cnt

    with open(CSV_PATH, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            step = int(row["timestep"])
            vx   = float(row["x"])
            vy   = float(row["y"])

            if step != prev_step:
                if prev_step is not None:
                    flush_step()
                step_counts.clear()
                prev_step = step

            rsu_id = nearest_rsu(vx, vy, rsus, radius)
            if rsu_id is not None:
                step_counts[rsu_id] += 1

    flush_step()  # last timestep

    # ---------------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------------
    overall_peak = max(peak_count.values()) if peak_count else 0

    print(f"{'RSU':<10}  {'Peak vehicles':>14}")
    print("-" * 27)
    for rsu_id, _, _ in rsus:
        pk = peak_count.get(rsu_id, 0)
        print(f"{rsu_id:<10}  {pk:>14}")
    print("-" * 27)
    print(f"{'OVERALL PEAK':<10}  {overall_peak:>14}")
    print()
    print(f"Recommended max_agents_per_rsu: {overall_peak + 5}  (peak + 5 buffer)")
    print()
    print("Update config/envs/rsu/bgc_core.yaml:")
    print(f"  max_agents_per_rsu: {overall_peak + 5}")


if __name__ == "__main__":
    main()
