"""
Transform new QMIX BGC eval files into the frontend metrics schema.
- Replaces KPIs and eval arrays in frontend/data/mono_qmix/metrics_los_*.json
- Preserves existing baselines and training curves
"""
import json
import os
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent

EVAL_FILES = {
    "los_a": ROOT / "pymarl/src/results/eval/qmix_bgc_full_low_seed50.json",
    "los_c": ROOT / "pymarl/src/results/eval/qmix_bgc_full_med_seed51.json",
    "los_e": ROOT / "pymarl/src/results/eval/qmix_bgc_full_high_seed52.json",
}

METRICS_DIR = ROOT / "frontend/data/mono_qmix"

SCENARIO_META = {
    "los_a": {"scenario": "LOS A (Low traffic)",      "map": "BGC Full (2km²)", "seed": 1801},
    "los_c": {"scenario": "LOS C (Stable flow)",       "map": "BGC Full (2km²)", "seed": 1802},
    "los_e": {"scenario": "LOS E (Forced flow / near-capacity)", "map": "BGC Full (2km²)", "seed": 1803},
}

def round4(v):
    return round(v, 4) if v is not None else None

def transform(los: str):
    eval_path = EVAL_FILES[los]
    metrics_path = METRICS_DIR / f"metrics_{los}.json"

    with open(eval_path, encoding="utf-8") as f:
        ev = json.load(f)
    with open(metrics_path, encoding="utf-8") as f:
        existing = json.load(f)

    m = ev["metrics"]
    r = ev["returns"]

    new_data = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "scenario": SCENARIO_META[los]["scenario"],
        "map":      SCENARIO_META[los]["map"],
        "seed":     SCENARIO_META[los]["seed"],
        "t_max":    existing.get("t_max", 1000000),
        "n_eval_episodes": ev["n_episodes"],

        "kpis": {
            "travelTime_s":       round4(m["mean_travel_time"]["mean"]),
            "travelTime_std":     round4(m["mean_travel_time"]["std"]),
            "waitTime_s":         round4(m["mean_waiting_time"]["mean"]),
            "waitTime_std":       round4(m["mean_waiting_time"]["std"]),
            "throughput":         round4(m["network_throughput"]["mean"]),
            "throughput_std":     round4(m["network_throughput"]["std"]),
            "co2_g_per_km":       round4(m["co2_g_per_km"]["mean"]),
            "fuel_l_per_100km":   round4(m["fuel_l_per_100km"]["mean"]),
            "cpuMean":            round4(m["cpu_percent_mean"]["mean"]),
            "cpuPeak":            round4(m["cpu_percent_peak"]["mean"]),
            "arrivalRate":        round4(m["arrival_rate"]["mean"]),
            "avgRouteLength_m":   round4(m["avg_route_length_m"]["mean"]),
            "totalStops":         round4(m["total_stops"]["mean"]),
            "returnMean":         round4(r["mean"]),
            "realTimeFactor":     round4(m["real_time_factor"]["mean"]),
        },

        # Preserve existing baselines (noop / greedy_shortest)
        "baselines": existing.get("baselines", {}),

        # Updated per-episode arrays (50 episodes)
        "evalReturns":      [round4(x) for x in r["raw"]],
        "evalTravelTimes":  [round4(x) for x in m["mean_travel_time"]["raw"]],
        "evalWaitingTimes": [round4(x) for x in m["mean_waiting_time"]["raw"]],
        "evalThroughputs":  [round4(x) for x in m["network_throughput"]["raw"]],
        "evalCpuMeans":     [round4(x) for x in m["cpu_percent_mean"]["raw"]],

        # Preserve training curve
        "training": existing.get("training", {}),
    }

    # Preserve optional fields that may exist
    for opt in ("trainMetrics", "testCurve"):
        if opt in existing:
            new_data[opt] = existing[opt]

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)

    print(f"[{los}] updated -> {metrics_path.name}")
    print(f"  returnMean:   {new_data['kpis']['returnMean']}")
    print(f"  travelTime_s: {new_data['kpis']['travelTime_s']} ± {new_data['kpis']['travelTime_std']}")
    print(f"  throughput:   {new_data['kpis']['throughput']} ± {new_data['kpis']['throughput_std']}")
    print(f"  episodes:     {new_data['n_eval_episodes']}")

if __name__ == "__main__":
    for los in ("los_a", "los_c", "los_e"):
        transform(los)
    print("\nDone.")
