"""
build_selfish_frontend_data.py

Reads the evaluate.py output JSONs for the selfish routing baseline across all
LOS levels and maps, then writes frontend-ready metrics JSON files to:

  frontend/data/selfish_routing/metrics_4x4.json
  frontend/data/selfish_routing/metrics_bgc_full.json

Run from pymarl/src/:
    python build_selfish_frontend_data.py

LOS mapping:
  low  -> free_flow   (LOS A)
  med  -> stable_flow (LOS C)
  high -> forced_flow (LOS E)
"""

import json
import os
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).parent
EVAL_DIR     = SCRIPT_DIR / "results" / "eval"
REPO_ROOT    = SCRIPT_DIR.parent.parent
FRONTEND_DIR = REPO_ROOT / "frontend" / "data" / "selfish_routing"

# Eval JSON filenames
EVAL_FILES = {
    "4x4": {
        "free_flow":   "selfish_routing_low_seed5.json.json",   # double .json naming quirk
        "stable_flow": "selfish_routing_med_seed5.json",
        "forced_flow": "selfish_routing_high_seed5.json",
    },
    "bgc_full": {
        "free_flow":   "selfish_routing_bgc_full_low_seed5.json",
        "stable_flow": "selfish_routing_bgc_full_med_seed5.json",
        "forced_flow": "selfish_routing_bgc_full_high_seed5.json",
    },
}

# Output filenames
OUTPUT_FILES = {
    "4x4":     "metrics_4x4.json",
    "bgc_full": "metrics_bgc_full.json",
}

# Max episode duration (seconds) — used to derive completed vehicle count
MAX_EPISODE_STEPS = 1000

# Known data quality issue: bgc_full low/med co2_g_per_km and fuel_l_per_100km
# are inflated because the emissions denominator (completed distance) is small
# relative to total episode emissions when few vehicles complete their routes.
CO2_RELIABILITY_NOTES = {
    "bgc_full": {
        "free_flow":   "co2_g_per_km and fuel_l_per_100km may be inflated: few vehicles complete routes in low demand, making the per-km denominator small relative to total episode emissions.",
        "stable_flow": "co2_g_per_km and fuel_l_per_100km may be inflated for the same reason as free_flow.",
        "forced_flow": None,
    },
    "4x4": {
        "free_flow":   None,
        "stable_flow": None,
        "forced_flow": None,
    },
}

LOS_LABELS = {
    "free_flow":   "LOS A — Free Flow",
    "stable_flow": "LOS C — Stable Flow",
    "forced_flow": "LOS E — Forced Flow",
}

MAP_LABELS = {
    "4x4":     "4×4 Grid Map",
    "bgc_full": "BGC Full (2 km)",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_eval(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def mean(d: dict, key: str):
    """Return mean value for a metric key, or None if missing."""
    return d["metrics"][key]["mean"] if key in d["metrics"] else None


def std(d: dict, key: str):
    return d["metrics"][key]["std"] if key in d["metrics"] else None


def build_level(eval_data: dict, map_key: str, los_key: str) -> dict:
    """Convert one eval JSON into a single traffic-level entry."""
    n = eval_data["n_episodes"]
    sim_hours = MAX_EPISODE_STEPS / 3600.0

    # Core metrics
    travel_time_s    = mean(eval_data, "mean_travel_time")
    waiting_time_s   = mean(eval_data, "mean_waiting_time")
    throughput_veh_h = mean(eval_data, "network_throughput")
    co2_g_per_km     = mean(eval_data, "co2_g_per_km")
    fuel_l_per_100km = mean(eval_data, "fuel_l_per_100km")
    avg_route_m      = mean(eval_data, "avg_route_length_m")
    arrival_rate     = mean(eval_data, "arrival_rate")
    rtf              = mean(eval_data, "real_time_factor")
    ctrl_travel_s    = mean(eval_data, "controlled_mean_travel_time")
    co2_g_episode    = mean(eval_data, "co2_emissions")
    fuel_ml_episode  = mean(eval_data, "fuel_consumption")
    total_stops      = mean(eval_data, "total_stops")

    # Derived
    travel_time_min = round(travel_time_s / 60, 3) if travel_time_s else None
    avg_speed_kmh   = round((avg_route_m / travel_time_s) * 3.6, 2) if (avg_route_m and travel_time_s) else None
    # Approximate completed vehicle count from throughput × sim_hours
    completed_approx = round(throughput_veh_h * sim_hours) if throughput_veh_h else None

    note = CO2_RELIABILITY_NOTES.get(map_key, {}).get(los_key)

    entry = {
        "source": "pymarl_evaluate",
        "n_episodes": n,
        "los_label": LOS_LABELS[los_key],
        "vehicles": {
            "completed_approx": completed_approx,
            "arrival_rate_mean": round(arrival_rate, 4) if arrival_rate else None,
            "arrival_rate_std":  round(std(eval_data, "arrival_rate"), 4) if std(eval_data, "arrival_rate") else None,
        },
        "kpis": {
            "avg_travel_time_s":            round(travel_time_s, 3)    if travel_time_s    else None,
            "avg_travel_time_s_std":        round(std(eval_data, "mean_travel_time"), 3) if std(eval_data, "mean_travel_time") else None,
            "avg_travel_time_min":          travel_time_min,
            "avg_waiting_time_s":           round(waiting_time_s, 3)   if waiting_time_s   else None,
            "avg_waiting_time_s_std":       round(std(eval_data, "mean_waiting_time"), 3) if std(eval_data, "mean_waiting_time") else None,
            "network_throughput_veh_h":     round(throughput_veh_h, 2) if throughput_veh_h else None,
            "throughput":                   completed_approx,
            "avg_speed_kmh":                avg_speed_kmh,
            "avg_route_length_m":           round(avg_route_m, 2)      if avg_route_m      else None,
            "avg_route_length_m_std":       round(std(eval_data, "avg_route_length_m"), 3) if std(eval_data, "avg_route_length_m") else None,
            "real_time_factor":             round(rtf, 2)              if rtf              else None,
            "co2_emissions_g_episode":      round(co2_g_episode, 2)    if co2_g_episode    else None,
            "fuel_consumption_ml_episode":  round(fuel_ml_episode, 2)  if fuel_ml_episode  else None,
            "avg_co2_g_per_km":             round(co2_g_per_km, 2)     if co2_g_per_km     else None,
            "avg_co2_g_per_km_std":         round(std(eval_data, "co2_g_per_km"), 3) if std(eval_data, "co2_g_per_km") else None,
            "avg_fuel_l_per_100km":         round(fuel_l_per_100km, 3) if fuel_l_per_100km else None,
            "avg_fuel_l_per_100km_std":     round(std(eval_data, "fuel_l_per_100km"), 3) if std(eval_data, "fuel_l_per_100km") else None,
            "controlled_mean_travel_time_s": round(ctrl_travel_s, 3)   if ctrl_travel_s    else None,
            "total_stops_mean":             round(total_stops, 1)       if total_stops      else None,
            "time_loss_s":                  None,  # not available from PyMARL pipeline
        },
        "timeseries_file": None,  # PyMARL runs do not produce summary XML timeseries
    }

    if note:
        entry["data_quality_note"] = note

    return entry


def build_map(map_key: str) -> dict:
    traffic_levels = {}
    for los_key, fname in EVAL_FILES[map_key].items():
        path = EVAL_DIR / fname
        if not path.exists():
            print(f"  [SKIP] {fname} not found")
            continue
        eval_data = load_eval(path)
        traffic_levels[los_key] = build_level(eval_data, map_key, los_key)
        n = eval_data["n_episodes"]
        print(f"  [OK]   {los_key:<12} — {n} episodes  ({fname})")

    return {
        "algorithm": "selfish_routing",
        "map": map_key,
        "map_label": MAP_LABELS[map_key],
        "scenario": "Wardrop User Equilibrium / Selfish Nash Routing (noop controlled agents)",
        "simulation_duration_s": MAX_EPISODE_STEPS,
        "seed": 5,
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000000"),
        "traffic_levels": traffic_levels,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FRONTEND_DIR.mkdir(parents=True, exist_ok=True)

    for map_key, out_fname in OUTPUT_FILES.items():
        print(f"\nBuilding {out_fname} ({MAP_LABELS[map_key]}) ...")
        data = build_map(map_key)

        out_path = FRONTEND_DIR / out_fname
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  -> Written to {out_path}")


if __name__ == "__main__":
    main()
