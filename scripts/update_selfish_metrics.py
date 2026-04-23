"""
Transform new Selfish Routing BGC eval files into the frontend metrics schema.
- Replaces KPIs in frontend/data/selfish_routing/metrics_bgc_full.json
- Preserves existing structure, baselines, timeseries_file references
- Prints SELFISH_LOS_REF constants for copy-paste into simulation page
"""
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).parent.parent

EVAL_FILES = {
    "free_flow":   ROOT / "pymarl/src/results/eval/selfish_routing_bgc_full_low_seed50.json",
    "stable_flow": ROOT / "pymarl/src/results/eval/selfish_routing_bgc_full_med_seed51.json",
    "forced_flow": ROOT / "pymarl/src/results/eval/selfish_routing_bgc_full_high_seed52.json",
}

METRICS_PATH = ROOT / "frontend/data/selfish_routing/metrics_bgc_full.json"

LOS_LABELS = {
    "free_flow":   "LOS A — Free Flow",
    "stable_flow": "LOS C — Stable Flow",
    "forced_flow": "LOS E — Forced Flow",
}

SIMULATION_DURATION_S = 1000  # 100 steps x 10s per step

def r4(v):
    return round(v, 4) if v is not None else None

def r2(v):
    return round(v, 2) if v is not None else None

def update():
    with open(METRICS_PATH, encoding="utf-8") as f:
        existing = json.load(f)

    ref_values = {}

    for level, eval_path in EVAL_FILES.items():
        with open(eval_path, encoding="utf-8") as f:
            ev = json.load(f)

        m = ev["metrics"]
        r = ev["returns"]
        existing_level = existing["traffic_levels"][level]

        travel_time_s = m["mean_travel_time"]["mean"]
        waiting_time_s = m["mean_waiting_time"]["mean"]
        throughput_veh_h = m["network_throughput"]["mean"]
        route_len_m = m["avg_route_length_m"]["mean"]

        # Derived
        travel_time_min = travel_time_s / 60.0
        throughput_count = round(throughput_veh_h * SIMULATION_DURATION_S / 3600)
        avg_speed_kmh = (route_len_m / travel_time_s) * 3.6 if travel_time_s > 0 else 0

        new_level = {
            "source": "pymarl_evaluate",
            "n_episodes": ev["n_episodes"],
            "los_label": LOS_LABELS[level],
            "vehicles": {
                "completed_approx": throughput_count,
                "arrival_rate_mean": r4(m["arrival_rate"]["mean"]),
                "arrival_rate_std":  r4(m["arrival_rate"]["std"]),
            },
            "kpis": {
                "avg_travel_time_s":             r4(travel_time_s),
                "avg_travel_time_s_std":          r4(m["mean_travel_time"]["std"]),
                "avg_travel_time_min":            r4(travel_time_min),
                "avg_waiting_time_s":             r4(waiting_time_s),
                "avg_waiting_time_s_std":         r4(m["mean_waiting_time"]["std"]),
                "network_throughput_veh_h":       r4(throughput_veh_h),
                "throughput":                     throughput_count,
                "avg_speed_kmh":                  r4(avg_speed_kmh),
                "avg_route_length_m":             r4(route_len_m),
                "avg_route_length_m_std":         r4(m["avg_route_length_m"]["std"]),
                "real_time_factor":               r4(m["real_time_factor"]["mean"]),
                "co2_emissions_g_episode":        r2(m["co2_emissions"]["mean"]),
                "fuel_consumption_ml_episode":    r2(m["fuel_consumption"]["mean"]),
                "avg_co2_g_per_km":               r4(m["co2_g_per_km"]["mean"]),
                "avg_co2_g_per_km_std":           r4(m["co2_g_per_km"]["std"]),
                "avg_fuel_l_per_100km":           r4(m["fuel_l_per_100km"]["mean"]),
                "avg_fuel_l_per_100km_std":       r4(m["fuel_l_per_100km"]["std"]),
                "controlled_mean_travel_time_s":  r4(m["controlled_mean_travel_time"]["mean"]),
                "total_stops_mean":               r4(m["total_stops"]["mean"]),
                "time_loss_s":                    None,
                "returnMean":                     r4(r["mean"]),
            },
            # Per-episode arrays (50 episodes) from eval run
            "evalReturns":      [r4(x) for x in r["raw"]],
            "evalTravelTimes":  [r4(x) for x in m["mean_travel_time"]["raw"]],
            "evalWaitingTimes": [r4(x) for x in m["mean_waiting_time"]["raw"]],
            "evalThroughputs":  [r4(x) for x in m["network_throughput"]["raw"]],

            # Preserve existing timeseries file references
            "timeseries_file": existing_level.get("timeseries_file", None),
        }

        # Preserve data_quality_note if it exists
        if "data_quality_note" in existing_level:
            new_level["data_quality_note"] = existing_level["data_quality_note"]

        # Preserve baselines if they exist
        if "baselines" in existing_level:
            new_level["baselines"] = existing_level["baselines"]

        existing["traffic_levels"][level] = new_level

        ref_values[level] = {
            "travelTime_s": r4(travel_time_s),
            "waitTime_s":   r4(waiting_time_s),
            "throughput":   r4(throughput_veh_h),
            "returnMean":   r4(r["mean"]),
        }

        print(f"[{level}]")
        print(f"  travelTime_s : {r4(travel_time_s)}")
        print(f"  waitTime_s   : {r4(waiting_time_s)}")
        print(f"  throughput   : {r4(throughput_veh_h)} veh/hr")
        print(f"  returnMean   : {r4(r['mean'])}")
        print(f"  co2_g_per_km : {r4(m['co2_g_per_km']['mean'])}")
        print(f"  fuel_l/100km : {r4(m['fuel_l_per_100km']['mean'])}")
        print(f"  episodes     : {ev['n_episodes']}")
        print()

    existing["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000000")

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    print(f"Updated -> {METRICS_PATH.name}")
    print()

    # Print the SELFISH_LOS_REF constant for copy-paste
    ff = ref_values["free_flow"]
    sf = ref_values["stable_flow"]
    ef = ref_values["forced_flow"]
    print("--- SELFISH_LOS_REF (paste into simulation/page.tsx) ---")
    print(f"const SELFISH_LOS_REF = {{")
    print(f"  free_flow:   {{ travelTime_s: {ff['travelTime_s']}, waitTime_s: {ff['waitTime_s']},  throughput: {ff['throughput']}, returnMean: {ff['returnMean']} }},")
    print(f"  stable_flow: {{ travelTime_s: {sf['travelTime_s']}, waitTime_s: {sf['waitTime_s']}, throughput: {sf['throughput']}, returnMean: {sf['returnMean']} }},")
    print(f"  forced_flow: {{ travelTime_s: {ef['travelTime_s']}, waitTime_s: {ef['waitTime_s']}, throughput: {ef['throughput']}, returnMean: {ef['returnMean']} }},")
    print(f"}} as const")

    # Also print forced_flow co2/fuel for SELFISH fallback constants
    ff_level = existing["traffic_levels"]["forced_flow"]
    print()
    print("--- SELFISH fallback episode series (forced_flow) ---")
    print(f"co2_g_per_km  = {ff_level['kpis']['avg_co2_g_per_km']}")
    print(f"fuel_l/100km  = {ff_level['kpis']['avg_fuel_l_per_100km']}")

if __name__ == "__main__":
    update()
