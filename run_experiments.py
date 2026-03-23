"""
Multi-Seed Experiment Runner (Step 6)

Trains QMIX across multiple random seeds, then evaluates the best checkpoint
from each seed against all baselines.  Produces per-seed JSON result files
and a final aggregate comparison table.

Usage:
    # Full pipeline (train + eval) with default seeds
    python run_experiments.py

    # Evaluation only (using already-trained models)
    python run_experiments.py --eval_only

    # Custom seeds / episode budget
    python run_experiments.py --seeds 1 2 3 --t_max 500000 --eval_episodes 30

    # Quick smoke-test (very short runs)
    python run_experiments.py --seeds 0 --t_max 10000 --eval_episodes 5
"""

import os
import sys
import json
import argparse
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime


# ── Default configuration ─────────────────────────────────────────────────────
DEFAULT_SEEDS      = [42, 123, 456]
DEFAULT_T_MAX      = 2_000_000
DEFAULT_EVAL_EPS   = 50          # episodes per seed / policy combo

BASELINES          = ["noop", "greedy_shortest", "random"]
SRC_DIR            = Path(__file__).parent / "pymarl" / "src"
RESULTS_DIR        = Path(__file__).parent / "results"

# ─────────────────────────────────────────────────────────────────────────────

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_training(seed: int, t_max: int, extra_args: list) -> Path:
    """Launch training for one seed and return the expected 'best' model dir."""
    model_dir = RESULTS_DIR / "models" / f"seed{seed}"
    cmd = [
        sys.executable, str(SRC_DIR / "main.py"),
        "--seed", str(seed),
        "--t_max", str(t_max),
        "--checkpoint_path", str(model_dir),
    ] + extra_args

    print(f"\n{'='*60}")
    print(f"  TRAINING  seed={seed}  t_max={t_max:,}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=str(SRC_DIR))
    if result.returncode != 0:
        print(f"[WARNING] Training seed={seed} exited with code {result.returncode}")

    best_dir = model_dir / "best"
    if not best_dir.exists():
        # Fall back to 'final' if best was never written
        final_dir = model_dir / "final"
        if final_dir.exists():
            print(f"  [INFO] 'best' not found, using 'final' for seed {seed}")
            return final_dir
        print(f"  [WARNING] No model found for seed {seed}")
        return None

    return best_dir


def run_evaluation(policy_type: str, seed: int, eval_episodes: int,
                   model_path: Path = None, baseline: str = None,
                   extra_args: list = None) -> Path:
    """
    Launch evaluate.py for one (policy, seed) combination.

    Returns path to the saved JSON results file.
    """
    extra_args = extra_args or []
    eval_dir = RESULTS_DIR / "eval"

    if policy_type == "qmix":
        stem = f"qmix_seed{seed}"
        cmd = [
            sys.executable, str(SRC_DIR / "evaluate.py"),
            "--model", str(model_path),
            "--episodes", str(eval_episodes),
            "--seed", str(seed),
            "--output", stem,
        ] + extra_args
    else:
        stem = f"{baseline}_seed{seed}"
        cmd = [
            sys.executable, str(SRC_DIR / "evaluate.py"),
            "--baseline", baseline,
            "--episodes", str(eval_episodes),
            "--seed", str(seed),
            "--output", stem,
        ] + extra_args

    print(f"\n  Evaluating [{policy_type if policy_type=='qmix' else baseline}]  seed={seed}  episodes={eval_episodes}")

    result = subprocess.run(cmd, cwd=str(SRC_DIR))
    if result.returncode != 0:
        print(f"  [WARNING] Evaluation exited with code {result.returncode}")

    out_file = eval_dir / f"{stem}.json"
    return out_file if out_file.exists() else None


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def aggregate_results(result_files: list) -> dict:
    """Aggregate a list of per-seed result dicts (same policy)."""
    if not result_files:
        return {}

    all_returns = []
    per_metric = {}

    for r in result_files:
        all_returns.extend(r["returns"]["raw"])
        for key, stats in r.get("metrics", {}).items():
            per_metric.setdefault(key, []).extend(stats.get("raw", []))

    def _agg(vals):
        if not vals:
            return {}
        return {
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals)),
            "median": float(np.median(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "n":   len(vals),
        }

    return {
        "returns": _agg(all_returns),
        "metrics": {k: _agg(v) for k, v in per_metric.items()},
    }


def print_comparison(policy_aggs: dict):
    """Print a formatted comparison table."""
    print("\n" + "=" * 78)
    print("FINAL COMPARISON (aggregated across seeds)")
    print("=" * 78)

    col_w = 20
    header = (f"{'Policy':<{col_w}} {'Return':>14} {'Travel Time':>14} "
              f"{'Arrival Rate':>14} {'N':>6}")
    print(header)
    print("-" * 78)

    for policy_name, agg in sorted(policy_aggs.items()):
        ret = agg.get("returns", {})
        tt  = agg.get("metrics", {}).get("mean_travel_time", {})
        arr = agg.get("metrics", {}).get("arrival_rate", {})

        ret_str = f"{ret['mean']:.2f}±{ret['std']:.2f}" if ret.get("mean") is not None else "N/A"
        tt_str  = f"{tt['mean']:.1f}±{tt['std']:.1f}"  if tt.get("mean")  is not None else "N/A"
        arr_str = f"{arr['mean']:.3f}±{arr['std']:.3f}" if arr.get("mean") is not None else "N/A"
        n_str   = str(ret.get("n", "?"))

        print(f"{policy_name:<{col_w}} {ret_str:>14} {tt_str:>14} {arr_str:>14} {n_str:>6}")

    print()


def save_summary(policy_aggs: dict, seeds: list, t_max: int):
    """Save aggregate summary JSON."""
    summary = {
        "seeds": seeds,
        "t_max": t_max,
        "timestamp": timestamp(),
        "policies": policy_aggs,
    }
    out = RESULTS_DIR / "eval" / f"summary_{timestamp()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Aggregate summary saved to: {out}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Multi-seed QMIX experiment runner")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help="Random seeds to use")
    parser.add_argument("--t_max", type=int, default=DEFAULT_T_MAX,
                        help="Training timesteps per seed")
    parser.add_argument("--eval_episodes", type=int, default=DEFAULT_EVAL_EPS,
                        help="Evaluation episodes per (policy, seed) pair")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training; evaluate pre-existing models")
    parser.add_argument("--baselines", type=str, nargs="+", default=BASELINES,
                        choices=BASELINES,
                        help="Baseline policies to evaluate")
    parser.add_argument("--use_cuda", action="store_true",
                        help="Enable CUDA for training")
    parser.add_argument("--use_gui", action="store_true",
                        help="Enable SUMO GUI (not recommended for batch runs)")
    args = parser.parse_args()

    # Build extra args forwarded to sub-processes
    train_extra = []
    eval_extra  = []
    if args.use_cuda:
        train_extra.append("--use_cuda")
    if args.use_gui:
        train_extra.append("--use_gui")
        eval_extra.append("--use_gui")

    print(f"\nExperiment plan:")
    print(f"  Seeds         : {args.seeds}")
    print(f"  t_max         : {args.t_max:,}")
    print(f"  Eval episodes : {args.eval_episodes}")
    print(f"  Baselines     : {args.baselines}")
    print(f"  Training      : {'SKIPPED (eval_only)' if args.eval_only else 'YES'}")

    # ── Step 1: Train one QMIX model per seed ────────────────────────────────
    qmix_model_dirs = {}
    if not args.eval_only:
        for seed in args.seeds:
            model_dir = run_training(seed, args.t_max, train_extra)
            if model_dir:
                qmix_model_dirs[seed] = model_dir
    else:
        # Locate pre-existing models
        for seed in args.seeds:
            best  = RESULTS_DIR / "models" / f"seed{seed}" / "best"
            final = RESULTS_DIR / "models" / f"seed{seed}" / "final"
            if best.exists():
                qmix_model_dirs[seed] = best
            elif final.exists():
                qmix_model_dirs[seed] = final
            else:
                print(f"[WARNING] No pre-trained model found for seed={seed}, skipping QMIX eval")

    # ── Step 2: Evaluate QMIX (best checkpoint) per seed ─────────────────────
    print(f"\n{'='*60}")
    print("  EVALUATING QMIX")
    print(f"{'='*60}")

    qmix_result_files = []
    for seed in args.seeds:
        if seed not in qmix_model_dirs:
            continue
        out = run_evaluation("qmix", seed, args.eval_episodes,
                             model_path=qmix_model_dirs[seed],
                             extra_args=eval_extra)
        if out:
            qmix_result_files.append(load_json(out))

    # ── Step 3: Evaluate baselines per seed ───────────────────────────────────
    print(f"\n{'='*60}")
    print("  EVALUATING BASELINES")
    print(f"{'='*60}")

    baseline_results = {bl: [] for bl in args.baselines}
    for bl in args.baselines:
        for seed in args.seeds:
            out = run_evaluation("baseline", seed, args.eval_episodes,
                                 baseline=bl, extra_args=eval_extra)
            if out:
                baseline_results[bl].append(load_json(out))

    # ── Step 4: Aggregate and print ───────────────────────────────────────────
    policy_aggs = {}
    if qmix_result_files:
        policy_aggs["qmix"] = aggregate_results(qmix_result_files)
    for bl in args.baselines:
        if baseline_results[bl]:
            policy_aggs[bl] = aggregate_results(baseline_results[bl])

    print_comparison(policy_aggs)

    # ── Step 5: Run compare script on all per-seed JSONs ─────────────────────
    eval_dir = RESULTS_DIR / "eval"
    all_jsons = []
    for seed in args.seeds:
        for policy_stem in (["qmix"] + args.baselines):
            f = eval_dir / f"{policy_stem}_seed{seed}.json"
            if f.exists():
                all_jsons.append(str(f))

    if len(all_jsons) >= 2:
        print(f"\n{'='*60}")
        print("  STATISTICAL COMPARISON (t-tests)")
        print(f"{'='*60}")
        compare_cmd = [
            sys.executable, str(SRC_DIR / "evaluate.py"),
            "--compare"
        ] + all_jsons
        subprocess.run(compare_cmd, cwd=str(SRC_DIR))

    # Save aggregate summary
    save_summary(policy_aggs, args.seeds, args.t_max)

    print("\nAll experiments complete!")


if __name__ == "__main__":
    main()
