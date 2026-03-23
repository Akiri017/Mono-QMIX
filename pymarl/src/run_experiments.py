"""
Multi-Seed Experiment Runner (Step 6)

Trains QMIX across multiple random seeds, evaluates each run alongside
baselines, and aggregates results with confidence intervals.

Usage:
    # Full experiment: train 5 seeds, then evaluate vs baselines
    python run_experiments.py --seeds 5 --t_max 1000000 --eval_episodes 50

    # Quick smoke test
    python run_experiments.py --seeds 2 --t_max 10000 --eval_episodes 5

    # Evaluation only (skip training, load existing models)
    python run_experiments.py --eval_only --checkpoint_root results/experiments
"""

import os
import sys
import json
import argparse
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent))


# ----------------------------------------------
# Seed management
# ----------------------------------------------

DEFAULT_SEEDS = [42, 43, 44, 45, 46]


# ----------------------------------------------
# Training helpers
# ----------------------------------------------

def train_seed(seed: int, t_max: int, checkpoint_root: str,
               extra_args: Optional[List[str]] = None) -> str:
    """
    Train a QMIX model for one seed via subprocess.

    Args:
        seed: Random seed
        t_max: Total training steps
        checkpoint_root: Root directory; model saved to <root>/seed_<seed>/
        extra_args: Additional CLI args forwarded to main.py

    Returns:
        Path to the trained model directory (best or final).
    """
    model_dir = os.path.join(checkpoint_root, f"seed_{seed}")
    os.makedirs(model_dir, exist_ok=True)

    script = str(Path(__file__).parent / "main.py")
    cmd = [
        sys.executable, script,
        "--seed", str(seed),
        "--t_max", str(t_max),
        "--checkpoint_path", model_dir,
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"Training seed {seed} | t_max={t_max} | dir={model_dir}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, check=True)

    # Prefer best model (saved by validation), fall back to final
    best_path = os.path.join(model_dir, "best")
    final_path = os.path.join(model_dir, "final")
    if os.path.isdir(best_path):
        return best_path
    return final_path


# ----------------------------------------------
# Evaluation helpers
# ----------------------------------------------

def evaluate_policy_subprocess(policy_type: str, seed: int,
                                eval_episodes: int, output_path: str,
                                model_path: Optional[str] = None,
                                baseline_name: Optional[str] = None) -> Dict:
    """
    Run evaluate.py for one policy via subprocess and load results JSON.

    Returns the loaded results dict, or an empty dict on failure.
    """
    script = str(Path(__file__).parent / "evaluate.py")
    cmd = [sys.executable, script, "--seed", str(seed), "--episodes", str(eval_episodes)]

    if policy_type == "qmix":
        cmd += ["--model", model_path, "--output", output_path]
    else:
        cmd += ["--baseline", baseline_name, "--output", output_path]

    print(f"  Evaluating {'QMIX' if policy_type == 'qmix' else baseline_name} "
          f"(seed={seed}, episodes={eval_episodes}) …")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"  [WARNING] Evaluation failed: {e}")
        return {}

    # Locate result file
    result_file = f"results/eval/{output_path}_seed{seed}.json"
    if os.path.exists(result_file):
        with open(result_file, "r") as f:
            return json.load(f)
    return {}


# ----------------------------------------------
# Statistics
# ----------------------------------------------

def aggregate_results(results_list: List[Dict]) -> Dict:
    """
    Aggregate per-seed evaluation results.

    Returns a dict with mean/std/ci95 for returns and each traffic metric.
    """
    valid = [r for r in results_list if r]
    if not valid:
        return {}

    METRIC_KEYS = [
        "mean_travel_time", "mean_waiting_time",
        "total_stops", "total_emissions",
        "arrival_rate", "controlled_mean_travel_time",
    ]

    def _agg(values):
        arr = np.array([v for v in values if v is not None], dtype=float)
        if len(arr) == 0:
            return {"mean": None, "std": None, "ci95": None, "n": 0}
        ci95 = 1.96 * np.std(arr, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1) if len(arr) > 1 else 0.0),
            "ci95": float(ci95),
            "n": len(arr),
            "raw": arr.tolist(),
        }

    # Returns
    return_means = [r["returns"]["mean"] for r in valid if "returns" in r and r["returns"]["mean"] is not None]
    agg = {"returns": _agg(return_means)}

    # Traffic metrics
    agg["metrics"] = {}
    for key in METRIC_KEYS:
        vals = []
        for r in valid:
            m = r.get("metrics", {}).get(key, {})
            if m.get("mean") is not None:
                vals.append(m["mean"])
        agg["metrics"][key] = _agg(vals)

    return agg


def print_aggregate_table(label: str, agg: Dict) -> None:
    """Print a formatted summary for one policy."""
    print(f"\n  {label}")
    print(f"  {'-'*55}")
    if not agg:
        print("    No data.")
        return

    ret = agg.get("returns", {})
    if ret.get("mean") is not None:
        print(f"    Return:              {ret['mean']:8.2f} ± {ret['ci95']:.2f} (95% CI), "
              f"std={ret['std']:.2f}, n={ret['n']}")

    LABELS = {
        "mean_travel_time":           "Mean Travel Time (s)",
        "mean_waiting_time":          "Mean Waiting Time (s)",
        "total_stops":                "Total Stops",
        "total_emissions":            "Total Emissions (g)",
        "arrival_rate":               "Arrival Rate",
        "controlled_mean_travel_time":"Controlled Travel Time (s)",
    }
    for key, lbl in LABELS.items():
        m = agg.get("metrics", {}).get(key, {})
        if m.get("mean") is not None:
            print(f"    {lbl:<30}: {m['mean']:8.3f} ± {m['ci95']:.3f}  (n={m['n']})")
        else:
            print(f"    {lbl:<30}: N/A")


def compare_aggregates(policy_aggs: Dict[str, Dict]) -> None:
    """Pairwise t-test comparison between policies."""
    try:
        from scipy import stats as scipy_stats
    except ImportError:
        print("\n[scipy not installed — skipping significance tests]")
        return

    names = list(policy_aggs.keys())
    if len(names) < 2:
        return

    print("\n" + "=" * 60)
    print("Pairwise Significance Tests (t-test on per-seed means)")
    print("=" * 60)

    METRICS = [
        ("Return",      "returns",      None),
        ("Travel Time", "metrics",      "mean_travel_time"),
        ("Arrival Rate","metrics",      "arrival_rate"),
    ]

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1, n2 = names[i], names[j]
            print(f"\n  {n1} vs {n2}:")
            for label, top_key, sub_key in METRICS:
                agg1 = policy_aggs[n1]
                agg2 = policy_aggs[n2]

                if top_key == "returns":
                    d1 = (agg1.get("returns") or {}).get("raw", [])
                    d2 = (agg2.get("returns") or {}).get("raw", [])
                    higher_is_better = True
                else:
                    d1 = (agg1.get("metrics", {}).get(sub_key) or {}).get("raw", [])
                    d2 = (agg2.get("metrics", {}).get(sub_key) or {}).get("raw", [])
                    higher_is_better = (sub_key == "arrival_rate")

                if len(d1) >= 2 and len(d2) >= 2:
                    t_stat, p = scipy_stats.ttest_ind(d1, d2)
                    sig = ("***" if p < 0.001 else
                           "**"  if p < 0.01  else
                           "*"   if p < 0.05  else "ns")
                    m1, m2 = np.mean(d1), np.mean(d2)
                    if higher_is_better:
                        better = n1 if m1 > m2 else n2
                    else:
                        better = n1 if m1 < m2 else n2
                    print(f"    [{label}] t={t_stat:.3f}, p={p:.4f} {sig:3s} "
                          f"-> {'better: ' + better if sig != 'ns' else 'no sig. difference'}")
                else:
                    print(f"    [{label}] insufficient data")


# ----------------------------------------------
# Main experiment loop
# ----------------------------------------------

def run_experiments(args) -> None:
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_root = args.checkpoint_root or f"results/experiments/{timestamp}"
    os.makedirs(checkpoint_root, exist_ok=True)
    os.makedirs("results/eval", exist_ok=True)

    baselines = ["noop", "greedy_shortest"]
    if args.include_random:
        baselines.append("random")

    print("\n" + "=" * 60)
    print("Multi-Seed Experiment Runner")
    print("=" * 60)
    print(f"  Seeds       : {seeds}")
    print(f"  t_max       : {args.t_max}")
    print(f"  Eval eps    : {args.eval_episodes}")
    print(f"  Baselines   : {baselines}")
    print(f"  Output dir  : {checkpoint_root}")
    print("=" * 60)

    # -- Phase 1: Training --------------------------------------
    qmix_model_paths: Dict[int, str] = {}

    if not args.eval_only:
        print("\n" + "-" * 60)
        print("Phase 1: Training")
        print("-" * 60)

        extra = []
        if args.no_validation:
            extra.append("--no_validation")

        for seed in seeds:
            try:
                model_path = train_seed(seed, args.t_max, checkpoint_root, extra)
                qmix_model_paths[seed] = model_path
                print(f"  [OK] Seed {seed} -> {model_path}")
            except subprocess.CalledProcessError as e:
                print(f"  [FAIL] Seed {seed} training failed: {e}")
    else:
        # Load pre-existing models
        for seed in seeds:
            best = os.path.join(checkpoint_root, f"seed_{seed}", "best")
            final = os.path.join(checkpoint_root, f"seed_{seed}", "final")
            if os.path.isdir(best):
                qmix_model_paths[seed] = best
            elif os.path.isdir(final):
                qmix_model_paths[seed] = final
            else:
                print(f"  [WARNING] No model found for seed {seed} in {checkpoint_root}")

    # -- Phase 2: Evaluation ------------------------------------
    print("\n" + "-" * 60)
    print("Phase 2: Evaluation")
    print("-" * 60)

    # Results keyed by policy name -> list of per-seed result dicts
    all_results: Dict[str, List[Dict]] = {"qmix": []}
    for bl in baselines:
        all_results[bl] = []

    for seed in seeds:
        print(f"\n  Seed {seed}:")

        # QMIX
        if seed in qmix_model_paths:
            r = evaluate_policy_subprocess(
                "qmix", seed, args.eval_episodes,
                output_path=f"qmix_exp_{seed}",
                model_path=qmix_model_paths[seed],
            )
            all_results["qmix"].append(r)
        else:
            all_results["qmix"].append({})

        # Baselines (only need one seed's data if non-random, but we run per seed for consistency)
        for bl in baselines:
            r = evaluate_policy_subprocess(
                "baseline", seed, args.eval_episodes,
                output_path=f"{bl}_exp_{seed}",
                baseline_name=bl,
            )
            all_results[bl].append(r)

    # -- Phase 3: Aggregation & Reporting ----------------------
    print("\n" + "=" * 60)
    print("Phase 3: Results Summary")
    print("=" * 60)

    aggregated: Dict[str, Dict] = {}

    for policy, results_list in all_results.items():
        agg = aggregate_results(results_list)
        aggregated[policy] = agg
        print_aggregate_table(policy.upper(), agg)

    # Statistical comparisons
    compare_aggregates(aggregated)

    # -- Save summary --------------------------------------------
    summary = {
        "timestamp": timestamp,
        "seeds": seeds,
        "t_max": args.t_max,
        "eval_episodes": args.eval_episodes,
        "baselines": baselines,
        "aggregated": {
            policy: {
                "returns": agg.get("returns"),
                "metrics": agg.get("metrics"),
            }
            for policy, agg in aggregated.items()
        },
    }

    summary_path = os.path.join(checkpoint_root, "experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


# ----------------------------------------------
# CLI
# ----------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Multi-seed QMIX experiment runner")

    # Seed control
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of seeds to run (default: 5)")
    parser.add_argument("--first_seed", type=int, default=42,
                        help="First seed value; subsequent seeds are first_seed+1, +2, …")

    # Training
    parser.add_argument("--t_max", type=int, default=1_000_000,
                        help="Total training steps per seed (default: 1M)")
    parser.add_argument("--no_validation", action="store_true",
                        help="Disable validation during training")

    # Evaluation
    parser.add_argument("--eval_episodes", type=int, default=50,
                        help="Evaluation episodes per policy per seed (default: 50)")
    parser.add_argument("--include_random", action="store_true",
                        help="Also evaluate a random-action baseline")

    # Paths
    parser.add_argument("--checkpoint_root", type=str, default=None,
                        help="Root dir for model checkpoints (default: results/experiments/<timestamp>)")

    # Modes
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training; load existing models from checkpoint_root")

    args = parser.parse_args()
    run_experiments(args)


if __name__ == "__main__":
    main()
