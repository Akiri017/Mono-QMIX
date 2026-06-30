"""
run_selfish_demo.py — Launch a selfish-routing SUMO-GUI simulation for BGC Full.

Usage:
    python scripts/run_selfish_demo.py low
    python scripts/run_selfish_demo.py med
    python scripts/run_selfish_demo.py high
    python scripts/run_selfish_demo.py all      # open all three in sequence

Each vehicle independently minimises its own travel time (online Dijkstra
rerouting every 60 s), producing the Nash / Wardrop selfish equilibrium.

Visual encoding (selfish_demo.view.xml):
  - Green  → vehicles / edges moving at free-flow speed
  - Yellow → moderate congestion
  - Red    → heavy congestion / stopped vehicles
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BGC_DIR   = REPO_ROOT / "bgc_full"

CONFIGS = {
    "low":  BGC_DIR / "selfish_low.sumocfg",
    "med":  BGC_DIR / "selfish_med.sumocfg",
    "high": BGC_DIR / "selfish_high.sumocfg",
}

DEMAND_INFO = {
    "low":  "~900 veh/hr  (period=4.0 s) — free-flow, green network",
    "med":  "~1800 veh/hr (period=2.0 s) — moderate congestion emerges",
    "high": "~4800 veh/hr (period=0.75 s) — heavy congestion, red network",
}


def find_sumo_gui() -> str:
    sumo_home = os.environ.get("SUMO_HOME", "")
    candidates = []

    if sumo_home:
        candidates += [
            Path(sumo_home) / "bin" / "sumo-gui.exe",
            Path(sumo_home) / "bin" / "sumo-gui",
        ]

    candidates += [
        Path(r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe"),
        Path(r"C:\Program Files\Eclipse\Sumo\bin\sumo-gui.exe"),
        Path("/usr/bin/sumo-gui"),
        Path("/usr/local/bin/sumo-gui"),
    ]

    for p in candidates:
        if p.exists():
            return str(p)

    # Fall back to PATH
    return "sumo-gui"


def launch(level: str) -> None:
    cfg = CONFIGS[level]
    if not cfg.exists():
        print(f"[ERROR] Config not found: {cfg}")
        sys.exit(1)

    sumo_gui = find_sumo_gui()
    print(f"\n{'='*60}")
    print(f"  Demand level : {level.upper()}")
    print(f"  Traffic load : {DEMAND_INFO[level]}")
    print(f"  Config       : {cfg.name}")
    print(f"  Binary       : {sumo_gui}")
    print(f"{'='*60}")
    print("  TIP: Press the green ▶ play button in SUMO-GUI to start.")
    print("  TIP: Use View > Settings to adjust speed slider / zoom.\n")

    subprocess.run([sumo_gui, "-c", str(cfg)], cwd=str(BGC_DIR))


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in (*CONFIGS, "all"):
        print(__doc__)
        print(f"Valid levels: {', '.join(CONFIGS)}  or  all")
        sys.exit(1)

    arg = sys.argv[1]
    levels = list(CONFIGS) if arg == "all" else [arg]

    for level in levels:
        launch(level)


if __name__ == "__main__":
    main()
