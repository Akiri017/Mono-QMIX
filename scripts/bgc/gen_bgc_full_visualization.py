#!/usr/bin/env python3
"""
Generate bgc_full/rsu_placed.add.xml from config/envs/rsu/bgc_full.yaml.

Usage (from repo root):
    python scripts/bgc/gen_bgc_full_visualization.py

Open the result in netedit:
    netedit --net-file bgc_full/final_map.net.xml \
            --additional-files bgc_full/rsu_placed.add.xml
"""

import math
import yaml
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RSU_CFG   = REPO_ROOT / "config" / "envs" / "rsu" / "bgc_full.yaml"
OUT_XML   = REPO_ROOT / "bgc_full" / "rsu_placed.add.xml"

N_POINTS  = 48          # polygon resolution (same as BGC Core)
FILL_COLOR  = "255,140,0,18"    # translucent orange fill
RING_COLOR  = "255,140,0,200"   # solid orange ring
POI_COLOR   = "0,220,255,255"   # cyan dot at centre
POI_SIZE    = 20                # POI width/height in metres (smaller for large map)


def circle_polygon(cx, cy, r, n=N_POINTS):
    pts = []
    for i in range(n):
        angle = 2 * math.pi * i / n
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    pts.append(pts[0])          # close the ring
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)


def main():
    with open(RSU_CFG) as f:
        cfg = yaml.safe_load(f)

    radius = cfg["radius"]
    rsus   = cfg["rsu_positions"]

    lines = ["<additional>"]

    for rsu in rsus:
        rid = rsu["id"]
        cx, cy = rsu["x"], rsu["y"]
        shape = circle_polygon(cx, cy, radius)

        lines.append(
            f'    <poly id="fill_{rid}" color="{FILL_COLOR}" fill="true"'
            f' layer="0" shape="{shape}"/>'
        )
        lines.append(
            f'    <poly id="ring_{rid}" color="{RING_COLOR}" fill="false"'
            f' lineWidth="3" layer="1" shape="{shape}"/>'
        )
        lines.append(
            f'    <poi id="{rid}" x="{cx}" y="{cy}"'
            f' color="{POI_COLOR}" width="{POI_SIZE}" height="{POI_SIZE}"'
            f' layer="10" name="{rid}"/>'
        )

    lines.append("</additional>")

    OUT_XML.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(rsus)} RSUs -> {OUT_XML}")
    print()
    print("Open in netedit:")
    print(f'  netedit --net-file bgc_full/final_map.net.xml \\')
    print(f'          --additional-files bgc_full/rsu_placed.add.xml')
    print()
    print("Or in sumo-gui:")
    print(f'  sumo-gui -n bgc_full/final_map.net.xml \\')
    print(f'           -a bgc_full/rsu_placed.add.xml')


if __name__ == "__main__":
    main()
