#!/usr/bin/env python3
"""
BGC Full Network Capacity Calibration
======================================
Measures the throughput ceiling of the BGC Full network by running SUMO
at escalating demand levels. The saturation throughput (where adding more
demand no longer increases completions) is taken as the network capacity.

v/c ratios are then computed for each LOS scenario to validate HCM compliance.

Usage:
    python scripts/calibrate_bgc_capacity.py [--quick]

    --quick  Run only the three LOS demand levels (skips full sweep).
             Use this to verify LOS labels without waiting for overload runs.

Output:
    - Console table: demand vs throughput vs v/c vs HCM LOS
    - scripts/bgc_capacity_results.txt  (saved automatically)
"""

import argparse
import os
import shutil
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
NET_FILE   = REPO_ROOT / "bgc_full" / "final_map.net.xml"
VTYPES_FILE = REPO_ROOT / "bgc_full" / "vtypes.add.xml"
RESULTS_OUT = Path(__file__).resolve().parent / "bgc_capacity_results.txt"

# Wardrop route files — used to extract realistic OD edge sequences
ROUTE_SOURCES = [
    REPO_ROOT / "bgc_full" / "wardrop_routes_low.rou.xml",
    REPO_ROOT / "bgc_full" / "wardrop_routes_med.rou.xml",
    REPO_ROOT / "bgc_full" / "wardrop_routes_highEnough.rou.xml",
]

# Actual LOS demand rates derived from route file vehicle counts and periods
# LOS A: 152 vehicles, period ~25.6s => 3600/25.6 = 141 veh/hr
# LOS C: 311 vehicles, period ~12.8s => 3600/12.8 = 281 veh/hr
# LOS E: 1269 vehicles, period ~3.2s => 3600/3.2  = 1125 veh/hr
LOS_DEMANDS = {
    "LOS A": 141,
    "LOS C": 281,
    "LOS E": 1125,
}

# Full sweep: ramps from well below LOS A up to 5x LOS E (deep overload)
DEMAND_SWEEP = [100, 141, 200, 281, 400, 600, 900, 1125, 1500, 2000, 3000, 4500, 6000]

SIM_TIME_S = 3600

# HCM 7th edition v/c thresholds (Table 18-3, urban arterials)
HCM_THRESHOLDS = [
    ("A", 0.00, 0.35),
    ("B", 0.35, 0.54),
    ("C", 0.54, 0.77),
    ("D", 0.77, 0.93),
    ("E", 0.93, 1.00),
    ("F", 1.00, float("inf")),
]


def hcm_los(vc: float) -> str:
    for label, lo, hi in HCM_THRESHOLDS:
        if lo <= vc < hi:
            return label
    return "F"


# ── SUMO setup ────────────────────────────────────────────────────────────────

def find_sumo_bin() -> str:
    sumo_home = os.environ.get("SUMO_HOME", "")
    for name in ("sumo", "sumo.exe"):
        via_path = shutil.which(name)
        if via_path:
            return via_path
        if sumo_home:
            candidate = os.path.join(sumo_home, "bin", name)
            if os.path.isfile(candidate):
                return candidate
    sys.exit(
        "ERROR: SUMO binary not found.\n"
        "  • Add SUMO to PATH, or\n"
        "  • Set SUMO_HOME to your SUMO installation directory."
    )


def ensure_traci():
    """Import traci, adding SUMO tools to sys.path if needed."""
    try:
        import traci  # noqa: F401
        return
    except ImportError:
        pass
    sumo_home = os.environ.get("SUMO_HOME", "")
    if sumo_home:
        tools = os.path.join(sumo_home, "tools")
        if tools not in sys.path:
            sys.path.append(tools)
    try:
        import traci  # noqa: F401
    except ImportError:
        sys.exit(
            "ERROR: traci not found.\n"
            "  Install via: pip install traci\n"
            "  Or set SUMO_HOME so the tools/ directory can be found."
        )


# ── Route loading ─────────────────────────────────────────────────────────────

def load_routes(sources: List[Path]) -> List[List[str]]:
    """
    Return a list of edge sequences from wardrop route files.
    Handles both <route edges="..."/> and <routeDistribution> children.
    """
    routes = []
    for f in sources:
        if not f.exists():
            print(f"  WARNING: {f.name} not found — skipping")
            continue
        tree = ET.parse(f)
        root = tree.getroot()
        for veh in root.findall("vehicle"):
            # Try direct <route> first, then <routeDistribution><route>
            route_elem = veh.find("route")
            if route_elem is None:
                rd = veh.find("routeDistribution")
                if rd is not None:
                    route_elem = rd.find("route")
            if route_elem is not None:
                edges = route_elem.get("edges", "").split()
                if len(edges) >= 2:
                    routes.append(edges)
    return routes


# ── File generation ───────────────────────────────────────────────────────────

def write_route_file(routes: List[List[str]], demand_veh_hr: int, out_path: str) -> int:
    """
    Write a SUMO route XML file at the given demand rate.
    Vehicles are evenly spaced over SIM_TIME_S; routes are cycled from the
    pool so every vehicle has a realistic OD pair.
    Returns number of vehicles written.
    """
    n = max(1, round(demand_veh_hr * SIM_TIME_S / 3600))
    period = SIM_TIME_S / n

    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<routes>"]
    for i in range(n):
        depart = i * period
        edges = " ".join(routes[i % len(routes)])
        lines.append(
            f'  <vehicle id="cal_{i}" type="thesis_car" depart="{depart:.2f}" '
            f'departLane="best" departSpeed="max">'
        )
        lines.append(f'    <route edges="{edges}"/>')
        lines.append("  </vehicle>")
    lines.append("</routes>")

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return n


def write_sumocfg(net: Path, rou: str, vtypes: Path, out_path: str):
    content = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
  <input>
    <net-file value="{net}"/>
    <route-files value="{rou}"/>
    <additional-files value="{vtypes}"/>
  </input>
  <processing>
    <time-to-teleport value="300"/>
    <ignore-junction-blocker value="1"/>
  </processing>
  <time>
    <begin value="0"/>
    <end value="{SIM_TIME_S}"/>
    <step-length value="1.0"/>
  </time>
</configuration>
"""
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(content)


# ── Simulation ────────────────────────────────────────────────────────────────

def run_simulation(sumo_bin: str, cfg_path: str) -> Dict:
    """Run SUMO headlessly via traci and return throughput metrics."""
    import traci

    traci.start([
        sumo_bin, "-c", cfg_path,
        "--no-step-log",
        "--no-warnings",
        "--xml-validation", "never",
    ])

    arrivals = 0
    spawned = 0

    try:
        for _ in range(SIM_TIME_S):
            traci.simulationStep()
            spawned   += traci.simulation.getDepartedNumber()
            arrivals  += traci.simulation.getArrivedNumber()
    finally:
        traci.close()

    throughput = arrivals * 3600.0 / SIM_TIME_S
    return {
        "spawned":          spawned,
        "arrivals":         arrivals,
        "throughput_veh_hr": throughput,
        "arrival_rate":     arrivals / spawned if spawned > 0 else 0.0,
    }


# ── Capacity estimation ────────────────────────────────────────────────────────

def estimate_capacity(results: List[Dict]) -> float:
    """
    Network capacity = peak measured throughput across all demand levels.
    Once demand exceeds capacity, throughput plateaus or drops (due to gridlock);
    the highest observed throughput is the best estimate of saturation flow.
    """
    return max(r["throughput_veh_hr"] for r in results)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true",
        help="Run only LOS A/C/E demand levels (3 runs instead of full sweep)."
    )
    args = parser.parse_args()

    demand_levels = sorted(LOS_DEMANDS.values()) if args.quick else DEMAND_SWEEP

    sumo_bin = find_sumo_bin()
    ensure_traci()

    header = [
        "=" * 66,
        "BGC Full Network Capacity Calibration",
        f"Mode   : {'Quick (LOS levels only)' if args.quick else 'Full sweep'}",
        f"SUMO   : {sumo_bin}",
        f"Network: {NET_FILE}",
        "=" * 66,
    ]
    output_lines = header[:]
    for line in header:
        print(line)

    print("\nLoading OD routes from wardrop route files...")
    routes = load_routes(ROUTE_SOURCES)
    if not routes:
        sys.exit("ERROR: No routes loaded. Check ROUTE_SOURCES paths.")
    msg = f"  Loaded {len(routes)} routes from {len(ROUTE_SOURCES)} files"
    print(msg)
    output_lines.append(msg)

    results = []
    tmpdir = tempfile.mkdtemp(prefix="bgc_calib_")
    col_hdr = f"\n{'Demand':>10}  {'Spawned':>8}  {'Arrivals':>9}  {'Throughput':>13}  {'ArrRate':>8}  {'Wall':>6}"
    sep = "-" * 64
    print(col_hdr)
    print(sep)
    output_lines += [col_hdr, sep]

    try:
        for demand in demand_levels:
            rou_path = os.path.join(tmpdir, f"rou_{demand}.rou.xml")
            cfg_path = os.path.join(tmpdir, f"sim_{demand}.sumocfg")

            n_written = write_route_file(routes, demand, rou_path)
            write_sumocfg(NET_FILE, rou_path, VTYPES_FILE, cfg_path)

            t0 = time.monotonic()
            res = run_simulation(sumo_bin, cfg_path)
            elapsed = time.monotonic() - t0

            res["demand_veh_hr"] = demand
            res["n_written"] = n_written
            results.append(res)

            row = (
                f"{demand:>8} vhr"
                f"  {res['spawned']:>8}"
                f"  {res['arrivals']:>9}"
                f"  {res['throughput_veh_hr']:>11.1f} vhr"
                f"  {res['arrival_rate']:>8.3f}"
                f"  {elapsed:>4.0f}s"
            )
            print(row)
            output_lines.append(row)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # ── Capacity & HCM report ─────────────────────────────────────────────────
    capacity = estimate_capacity(results)

    section = [
        "",
        "=" * 66,
        f"Estimated Network Capacity: {capacity:.1f} veh/hr",
        "(Peak throughput across all demand levels — saturation flow rate)",
        "=" * 66,
        "",
        "HCM Level of Service Validation — BGC Full Map",
        f"  Capacity baseline: {capacity:.1f} veh/hr",
        "",
        f"  {'Scenario':<10}  {'Demand':>10}  {'v/c ratio':>10}  {'HCM LOS':>8}  {'Target':>7}  {'Match':>6}",
        "  " + "-" * 56,
    ]

    all_match = True
    for label, demand in LOS_DEMANDS.items():
        vc = demand / capacity
        actual_los = hcm_los(vc)
        target_letter = label.split()[-1]
        match_str = "YES" if actual_los == target_letter else "NO ← MISMATCH"
        if actual_los != target_letter:
            all_match = False
        row = (
            f"  {label:<10}  {demand:>8} vhr  {vc:>10.3f}  {actual_los:>8}  {target_letter:>7}  {match_str:>6}"
        )
        section.append(row)

    section += [
        "",
        "Full demand sweep (demand → throughput mapping):",
        f"  {'Demand':>10}  {'Throughput':>13}  {'v/c':>8}  {'HCM LOS':>8}",
        "  " + "-" * 46,
    ]
    for r in results:
        d  = r["demand_veh_hr"]
        tp = r["throughput_veh_hr"]
        vc = d / capacity
        los = hcm_los(vc)
        section.append(f"  {d:>10}  {tp:>11.1f} vhr  {vc:>8.3f}  {los:>8}")

    verdict = (
        "\nVERDICT: All LOS labels match HCM v/c thresholds."
        if all_match else
        "\nVERDICT: One or more LOS labels do NOT match HCM v/c thresholds.\n"
        "         See MISMATCH rows above — demand levels need adjustment."
    )
    section.append(verdict)

    for line in section:
        print(line)
    output_lines += section

    # ── Save results ──────────────────────────────────────────────────────────
    with open(RESULTS_OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(output_lines))
    print(f"\nResults saved to: {RESULTS_OUT}")


if __name__ == "__main__":
    main()
