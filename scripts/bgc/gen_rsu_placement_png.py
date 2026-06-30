#!/usr/bin/env python3
"""
Generate RSU placement PNG for BGC Core and BGC Full.

BGC Full uses a SUMO-style render:
  - roads coloured by OSM type with dark casing (outline)
  - lane widths scaled to real-world metres
  - junction polygons filled
  - RSU dots with glow + dashed coverage rings

BGC Core keeps the lighter overview style.

Usage (from repo root):
    python scripts/bgc/gen_rsu_placement_png.py

Outputs:
    scripts/bgc/rsu_placement_bgc_core.png
    scripts/bgc/rsu_placement_bgc_full.png
"""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sumolib

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR   = Path(__file__).resolve().parent

MAPS = [
    {
        "label":   "BGC Core",
        "net":     REPO_ROOT / "bgc_core" / "final_map.net.xml",
        "add":     REPO_ROOT / "scripts" / "bgc" / "rsu_placed.add.xml",
        "out":     OUT_DIR / "rsu_placement_bgc_core.png",
        "style":   "overview",
    },
    {
        "label":   "BGC Full",
        "net":     REPO_ROOT / "bgc_full" / "final_map.net.xml",
        "add":     REPO_ROOT / "bgc_full" / "rsu_placed.add.xml",
        "out":     OUT_DIR / "rsu_placement_bgc_full.png",
        "style":   "sumo",
    },
]

# ── Overview style (BGC Core) ────────────────────────────────────────────────
OV_BG         = "#1a1a2e"
OV_ROAD_COLOR = "#555555"
OV_ROAD_LW    = 0.6

# ── SUMO real-world style (BGC Full) ─────────────────────────────────────────
SUMO_BG = "#1c1c1c"

# (keyword_in_type, casing_hex, fill_hex, lane_width_multiplier)
ROAD_STYLE_TABLE = [
    ("motorway",    "#2a2800", "#FFD700", 1.4),
    ("trunk",       "#221600", "#FFA500", 1.3),
    ("primary",     "#1a0d00", "#FF8C00", 1.2),
    ("secondary",   "#1e1e1e", "#C8C8C8", 1.0),
    ("tertiary",    "#1a1a1a", "#AAAAAA", 0.9),
    ("residential", "#161616", "#777777", 0.8),
    ("service",     "#141414", "#555555", 0.65),
    ("living",      "#141414", "#4a4a4a", 0.65),
    ("path",        "#141414", "#3a3a3a", 0.5),
    ("footway",     "#141414", "#3a3a3a", 0.5),
]
ROAD_STYLE_DEFAULT = ("#181818", "#606060", 0.75)

# Draw priority: minor roads first so major roads render on top
PRIORITY_ORDER = [
    "footway", "path", "living", "service",
    "residential", "tertiary", "secondary",
    "primary", "trunk", "motorway",
]

# RSU visuals (shared)
RSU_DOT  = "#00DCFF"
RSU_RING = "#FF8C00"
LABEL_C  = "#FFFFFF"


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_rsus(add_xml: Path):
    tree = ET.parse(add_xml)
    rsus, radii = {}, {}
    for poi in tree.findall(".//poi"):
        rsus[poi.get("id")] = (float(poi.get("x")), float(poi.get("y")))
    for poly in tree.findall(".//poly"):
        pid = poly.get("id", "")
        if not pid.startswith("ring_"):
            continue
        rid = pid[5:]
        pts = [tuple(map(float, p.split(","))) for p in poly.get("shape").split()]
        cx, cy = rsus.get(rid, (0, 0))
        radii[rid] = math.hypot(pts[0][0] - cx, pts[0][1] - cy)
    return [(rid, x, y, radii.get(rid, 300)) for rid, (x, y) in rsus.items()]


def road_style(edge_type: str):
    t = (edge_type or "").lower()
    for kw, casing, fill, wm in ROAD_STYLE_TABLE:
        if kw in t:
            return casing, fill, wm
    return ROAD_STYLE_DEFAULT


def pts_per_meter(fig, ax):
    """Points per data-unit (metre) after axis limits are set."""
    fig.canvas.draw()          # force renderer so transData is valid
    x1, x2 = ax.get_xlim()
    p1 = ax.transData.transform((x1, 0))
    p2 = ax.transData.transform((x2, 0))
    pixels_per_m = (p2[0] - p1[0]) / (x2 - x1)
    return pixels_per_m * 72.0 / fig.dpi


def draw_rsu_layer(ax, rsus):
    for rid, cx, cy, radius in rsus:
        # semi-transparent fill
        ax.add_patch(mpatches.Circle(
            (cx, cy), radius, color=RSU_RING, alpha=0.05, linewidth=0, zorder=8))
        # dashed coverage ring
        ax.add_patch(mpatches.Circle(
            (cx, cy), radius, color=RSU_RING, alpha=0.80,
            fill=False, linewidth=1.2, linestyle="--", zorder=9))
        # glow halo
        ax.plot(cx, cy, "o", color=RSU_RING, markersize=14, alpha=0.25, zorder=10)
        # centre dot
        ax.plot(cx, cy, "o", color=RSU_DOT,  markersize=6,  zorder=11)
        # label with dark box
        ax.text(cx, cy + radius * 0.09, rid,
                color=LABEL_C, fontsize=5.5, ha="center", va="bottom",
                fontweight="bold", zorder=12,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#000000",
                          alpha=0.55, edgecolor="none"))


# ── renderers ────────────────────────────────────────────────────────────────

def render_overview(cfg, net, rsus):
    """Simple monochrome render used for BGC Core."""
    fig, ax = plt.subplots(figsize=(12, 10), facecolor=OV_BG)
    ax.set_facecolor(OV_BG)
    ax.set_aspect("equal")
    ax.axis("off")

    for edge in net.getEdges():
        for lane in edge.getLanes():
            shape = lane.getShape()
            if len(shape) < 2:
                continue
            xs, ys = zip(*shape)
            ax.plot(xs, ys, color=OV_ROAD_COLOR, linewidth=OV_ROAD_LW,
                    solid_capstyle="round")

    xmin, ymin, xmax, ymax = net.getBoundary()
    pad_x = (xmax - xmin) * 0.05
    pad_y = (ymax - ymin) * 0.05
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)

    draw_rsu_layer(ax, rsus)

    ax.set_title(f"{cfg['label']} — RSU Placement ({len(rsus)} RSUs)",
                 color=LABEL_C, fontsize=13, pad=10)
    fig.savefig(cfg["out"], dpi=150, bbox_inches="tight", facecolor=OV_BG)
    plt.close(fig)


def render_sumo(cfg, net, rsus):
    """SUMO real-world style render for BGC Full."""
    xmin, ymin, xmax, ymax = net.getBoundary()
    pad_x = (xmax - xmin) * 0.04
    pad_y = (ymax - ymin) * 0.04

    fig, ax = plt.subplots(figsize=(16, 13), facecolor=SUMO_BG)
    ax.set_facecolor(SUMO_BG)
    ax.set_aspect("equal")
    ax.axis("off")

    # Set limits before computing scale so transData is correct
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)

    ppm = pts_per_meter(fig, ax)
    LANE_W = 3.2 * ppm   # standard 3.2 m lane in display points

    print(f"    scale {ppm:.4f} pts/m  |  1 lane ~{LANE_W:.2f} pts")

    # ── bucket edges by draw priority ────────────────────────────────────
    buckets = {k: [] for k in PRIORITY_ORDER}
    buckets["other"] = []
    for edge in net.getEdges():
        t = (edge.getType() or "").lower()
        placed = False
        for kw in PRIORITY_ORDER:
            if kw in t:
                buckets[kw].append(edge)
                placed = True
                break
        if not placed:
            buckets["other"].append(edge)

    # draw from lowest to highest priority (minor roads first)
    for bk in PRIORITY_ORDER + ["other"]:
        for edge in buckets[bk]:
            casing, fill, wm = road_style(edge.getType())
            road_w = edge.getLaneNumber() * LANE_W * wm
            for lane in edge.getLanes():
                shape = lane.getShape()
                if len(shape) < 2:
                    continue
                xs, ys = zip(*shape)
                ax.plot(xs, ys, color=casing, linewidth=road_w + 1.4,
                        solid_capstyle="round", solid_joinstyle="round", zorder=2)
                ax.plot(xs, ys, color=fill,   linewidth=road_w,
                        solid_capstyle="round", solid_joinstyle="round", zorder=3)

    # ── junction fills ────────────────────────────────────────────────────
    for node in net.getNodes():
        shape = node.getShape()
        if len(shape) >= 3:
            xs, ys = zip(*shape)
            ax.fill(xs, ys, color="#3a3a3a", zorder=4)

    # ── RSUs ──────────────────────────────────────────────────────────────
    draw_rsu_layer(ax, rsus)

    # ── legend ────────────────────────────────────────────────────────────
    legend = [
        mpatches.Patch(color="#FFD700", label="Motorway / trunk"),
        mpatches.Patch(color="#FF8C00", label="Primary"),
        mpatches.Patch(color="#C8C8C8", label="Secondary"),
        mpatches.Patch(color="#AAAAAA", label="Tertiary"),
        mpatches.Patch(color="#777777", label="Residential / service"),
        mpatches.Patch(color=RSU_DOT,  label=f"RSU node ({len(rsus)} total)"),
        mpatches.Patch(color=RSU_RING, alpha=0.7, label="Coverage radius"),
    ]
    ax.legend(handles=legend, loc="lower right", framealpha=0.45,
              facecolor="#0d0d0d", edgecolor="#444444", labelcolor=LABEL_C,
              fontsize=7.5, title="Legend", title_fontsize=8,
              labelspacing=0.4)

    ax.set_title(f"{cfg['label']} — RSU Placement ({len(rsus)} RSUs)",
                 color=LABEL_C, fontsize=14, pad=12, fontweight="bold")

    fig.savefig(cfg["out"], dpi=150, bbox_inches="tight", facecolor=SUMO_BG)
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def render(cfg):
    print(f"Rendering {cfg['label']} ({cfg['style']}) ...")
    net  = sumolib.net.readNet(str(cfg["net"]), withInternal=False)
    rsus = parse_rsus(cfg["add"])
    if cfg["style"] == "sumo":
        render_sumo(cfg, net, rsus)
    else:
        render_overview(cfg, net, rsus)
    print(f"  -> {cfg['out']}")


def main():
    for cfg in MAPS:
        render(cfg)
    print("Done.")


if __name__ == "__main__":
    main()
