"""
Extract intersection coordinates from the Synthetic 4x4 SUMO network.

Loads the net.xml using sumolib, iterates all nodes (intersections),
and saves their id, x, y to scripts/rsu/rsu_coords_synthetic_4x4.txt.

Run from repo root:
    python scripts/rsu/extract_rsu_coords.py
"""

import os
import sys

# Ensure sumolib is importable
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(repo_root, "pymarl", "src"))

import sumolib

NET_XML = os.path.join(repo_root, "4by4_map", "final_map.net.xml")
OUTPUT_TXT = os.path.join(repo_root, "scripts", "rsu", "rsu_coords_synthetic_4x4.txt")


def main():
    if not os.path.exists(NET_XML):
        print(f"[ERROR] net.xml not found at: {NET_XML}")
        sys.exit(1)

    print(f"Loading network: {NET_XML}")
    net = sumolib.net.readNet(NET_XML)

    nodes = net.getNodes()
    print(f"Total nodes found: {len(nodes)}")

    xs = [n.getCoord()[0] for n in nodes]
    ys = [n.getCoord()[1] for n in nodes]
    print(f"X range: {min(xs):.2f} to {max(xs):.2f}")
    print(f"Y range: {min(ys):.2f} to {max(ys):.2f}")

    lines = []
    for node in nodes:
        nid = node.getID()
        x, y = node.getCoord()
        line = f"{nid}\t{x:.6f}\t{y:.6f}"
        print(line)
        lines.append(line)

    with open(OUTPUT_TXT, "w") as f:
        f.write("id\tx\ty\n")
        f.write("\n".join(lines))

    print(f"\nSaved {len(nodes)} nodes to: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
