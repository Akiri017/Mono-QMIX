"""
Evaluation Script for QMIX and Baseline Policies (Step 6)

Evaluates trained QMIX models or baseline policies on the SUMO environment,
collecting comprehensive metrics for comparison.

Usage:
    # Evaluate trained QMIX model
    python evaluate.py --model results/models/final --episodes 100 --seed 42

    # Evaluate baseline policies
    python evaluate.py --baseline noop --episodes 100 --seed 42
    python evaluate.py --baseline greedy_shortest --episodes 50 --seed 42
    python evaluate.py --baseline random --episodes 50 --seed 42

    # Compare multiple result files
    python evaluate.py --compare results/eval/qmix.json results/eval/noop.json
"""

import os
import sys
import json
import yaml
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from controllers.basic_controller import BasicMAC
from controllers.baseline_controller import BaselineMAC
from runners.episode_runner import EpisodeRunner
from utils.logging import Logger


# Metrics reported in evaluation output
METRIC_KEYS = [
    "mean_travel_time",
    "mean_waiting_time",
    "total_stops",
    "total_emissions",
    "arrival_rate",
    "controlled_mean_travel_time",
]


def load_config(alg_config_path, env_config_path):
    """Load algorithm and environment configs."""
    with open(alg_config_path, 'r') as f:
        alg_config = yaml.safe_load(f)
    with open(env_config_path, 'r') as f:
        env_config = yaml.safe_load(f)
    return {**alg_config, **env_config}


def evaluate_policy(args, policy_type="qmix", model_path=None, baseline_type=None):
    """
    Evaluate a policy (QMIX or baseline) on the environment.

    Args:
        args: Configuration dictionary
        policy_type: "qmix" or "baseline"
        model_path: Path to trained model (for QMIX)
        baseline_type: Baseline policy type (for baseline)

    Returns:
        Dictionary with evaluation results including detailed episode metrics
    """
    print("\n" + "=" * 60)
    if policy_type == "qmix":
        print(f"Evaluating QMIX model from: {model_path}")
    else:
        print(f"Evaluating Baseline policy: {baseline_type}")
    print("=" * 60)

    # Set seeds
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])

    # Create logger (disable TensorBoard for evaluation)
    logger = Logger(use_tensorboard=False, log_dir=None)

    # Create runner
    runner = EpisodeRunner(args, logger)
    env_info = runner.get_env_info()
    args.update(env_info)

    print(f"\nEnvironment Info:")
    print(f"  n_agents: {env_info['n_agents']}")
    print(f"  n_actions: {env_info['n_actions']}")
    print(f"  obs_shape: {env_info['obs_shape']}")
    print(f"  state_shape: {env_info['state_shape']}")

    # Create data scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": torch.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": torch.uint8},
        "filled": {"vshape": (1,), "dtype": torch.uint8}
    }
    groups = {"agents": args["n_agents"]}
    preprocess = {}

    # Create MAC based on policy type
    if policy_type == "qmix":
        mac = BasicMAC(scheme, groups, args)

        # Load trained model
        if model_path and os.path.exists(model_path):
            print(f"\nLoading model from {model_path}...")
            agent_path = os.path.join(model_path, "agent.pth")
            mixer_path = os.path.join(model_path, "mixer.pth")

            if os.path.exists(agent_path):
                mac.agent.load_state_dict(torch.load(agent_path, map_location="cpu"))
                print("  [OK] Agent network loaded")
            else:
                print(f"  [WARNING] Agent model not found at {agent_path}")

            if hasattr(mac, 'mixer') and mac.mixer is not None and os.path.exists(mixer_path):
                mac.mixer.load_state_dict(torch.load(mixer_path, map_location="cpu"))
                print("  [OK] Mixer network loaded")

        # Greedy action selection is handled by test_mode=True in the runner

    else:  # baseline
        args["baseline_policy"] = baseline_type
        mac = BaselineMAC(scheme, groups, args)

        # Set environment reference for greedy_shortest
        if baseline_type == "greedy_shortest":
            mac.set_env(runner.env)

    # Setup runner
    runner.setup(scheme, groups, preprocess, mac)

    # Run evaluation episodes
    print(f"\nRunning {args['eval_episodes']} evaluation episodes...")
    print("-" * 60)

    episode_returns = []
    episode_lengths = []
    # Per-metric lists for aggregation
    metrics_collected: Dict[str, List[float]] = {k: [] for k in METRIC_KEYS}

    for ep in range(args['eval_episodes']):
        # run() now returns (batch, episode_metrics)
        episode_batch, ep_metrics = runner.run(test_mode=True)

        ep_return = episode_batch["reward"][:, :-1].sum().item()
        ep_length = int(episode_batch["filled"].float().sum().item())
        episode_returns.append(ep_return)
        episode_lengths.append(ep_length)

        # Collect detailed metrics
        for key in METRIC_KEYS:
            if key in ep_metrics:
                metrics_collected[key].append(float(ep_metrics[key]))

        # Progress line
        metric_str = ""
        if "mean_travel_time" in ep_metrics:
            metric_str = (f", tt={ep_metrics['mean_travel_time']:.0f}s"
                          f", arr={ep_metrics.get('arrival_rate', 0):.2f}")
        print(f"  Episode {ep+1}/{args['eval_episodes']}: "
              f"return={ep_return:.2f}, length={ep_length}{metric_str}")

    # Compute summary statistics
    def _stats(values):
        if not values:
            return {"mean": None, "std": None, "median": None, "min": None, "max": None, "raw": []}
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "raw": [float(v) for v in values],
        }

    results = {
        "policy_type": policy_type,
        "policy_name": model_path if policy_type == "qmix" else baseline_type,
        "n_episodes": args['eval_episodes'],
        "seed": args["seed"],
        "returns": _stats(episode_returns),
        "episode_lengths": _stats(episode_lengths),
        "metrics": {key: _stats(metrics_collected[key]) for key in METRIC_KEYS},
    }

    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Policy       : {results['policy_name']}")
    print(f"Episodes     : {results['n_episodes']}")
    print(f"Mean Return  : {results['returns']['mean']:.2f} ± {results['returns']['std']:.2f}")

    print("\n--- Traffic Metrics ---")
    labels = {
        "mean_travel_time":          "Mean Travel Time (s)",
        "mean_waiting_time":         "Mean Waiting Time (s)",
        "total_stops":               "Total Stops",
        "total_emissions":           "Total Emissions",
        "arrival_rate":              "Arrival Rate",
        "controlled_mean_travel_time": "Controlled Mean Travel Time (s)",
    }
    for key in METRIC_KEYS:
        stats = results["metrics"][key]
        if stats["mean"] is not None:
            print(f"  {labels[key]:<35}: {stats['mean']:.3f} ± {stats['std']:.3f}")
        else:
            print(f"  {labels[key]:<35}: N/A (no data)")

    # Cleanup
    runner.close_env()
    logger.close()

    return results


def save_results(results, output_path):
    """Save evaluation results to JSON file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


def compare_results(result_files):
    """
    Compare results from multiple evaluation JSON files.

    Prints a comparison table and performs pairwise t-tests on travel time
    and episode return.
    """
    from scipy import stats as scipy_stats

    print("\n" + "=" * 70)
    print("Policy Comparison")
    print("=" * 70)

    all_results = []
    for result_file in result_files:
        with open(result_file, 'r') as f:
            all_results.append(json.load(f))

    # Header
    col_w = 26
    print(f"\n{'Policy':<{col_w}} {'Return':>12} {'Travel Time':>14} {'Arrival Rate':>14} {'N':>6}")
    print("-" * (col_w + 50))

    for r in all_results:
        name = r['policy_name']
        if len(name) > col_w - 1:
            name = "..." + name[-(col_w - 4):]

        ret = r['returns']
        tt  = r['metrics']['mean_travel_time']
        arr = r['metrics']['arrival_rate']

        ret_str = f"{ret['mean']:.2f}±{ret['std']:.2f}" if ret['mean'] is not None else "N/A"
        tt_str  = f"{tt['mean']:.1f}±{tt['std']:.1f}"  if tt['mean']  is not None else "N/A"
        arr_str = f"{arr['mean']:.3f}±{arr['std']:.3f}" if arr['mean'] is not None else "N/A"

        print(f"{name:<{col_w}} {ret_str:>12} {tt_str:>14} {arr_str:>14} {r['n_episodes']:>6}")

    # Pairwise significance tests
    if len(all_results) >= 2:
        print("\n" + "=" * 70)
        print("Pairwise t-tests (* p<0.05, ** p<0.01, *** p<0.001)")
        print("=" * 70)

        for i in range(len(all_results)):
            for j in range(i + 1, len(all_results)):
                r1, r2 = all_results[i], all_results[j]
                n1 = r1['policy_name'][-20:]
                n2 = r2['policy_name'][-20:]

                for metric_label, key, raw_key in [
                    ("Return",      None,              "returns"),
                    ("Travel Time", "mean_travel_time", "metrics"),
                ]:
                    if raw_key == "returns":
                        d1 = r1['returns']['raw']
                        d2 = r2['returns']['raw']
                    else:
                        d1 = r1['metrics'][key]['raw']
                        d2 = r2['metrics'][key]['raw']

                    if len(d1) >= 2 and len(d2) >= 2 and d1 and d2:
                        t_stat, p_val = scipy_stats.ttest_ind(d1, d2)
                        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
                        better = n1 if np.mean(d1) > np.mean(d2) else n2
                        # For travel time, lower is better
                        if key == "mean_travel_time":
                            better = n1 if np.mean(d1) < np.mean(d2) else n2
                        print(f"  [{metric_label}] {n1} vs {n2}: "
                              f"t={t_stat:.3f}, p={p_val:.4f} {sig}  => better: {better}")
                print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate QMIX or baseline policies")

    # Mode selection
    parser.add_argument("--model", type=str, default=None,
                       help="Path to trained QMIX model directory")
    parser.add_argument("--baseline", type=str, default=None,
                       choices=["noop", "greedy_shortest", "random"],
                       help="Baseline policy type")

    # Evaluation settings
    parser.add_argument("--episodes", type=int, default=100,
                       help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--output", type=str, default=None,
                       help="Output filename stem (without extension)")

    # Comparison mode
    parser.add_argument("--compare", nargs='+', default=None,
                       help="Compare multiple result JSON files")

    # SUMO settings
    parser.add_argument("--use_gui", action="store_true",
                       help="Use SUMO GUI for visualization")

    args_cmd = parser.parse_args()

    # Comparison mode
    if args_cmd.compare is not None:
        compare_results(args_cmd.compare)
        return

    # Check that either model or baseline is specified
    if args_cmd.model is None and args_cmd.baseline is None:
        print("Error: Must specify either --model or --baseline")
        parser.print_help()
        sys.exit(1)

    # Load configs
    script_dir = Path(__file__).parent
    alg_config_path = script_dir / "config" / "algs" / "qmix_sumo.yaml"
    env_config_path = script_dir / "config" / "envs" / "sumo_grid4x4.yaml"

    if not alg_config_path.exists() or not env_config_path.exists():
        print(f"Error: Config files not found")
        print(f"  Algorithm config: {alg_config_path}")
        print(f"  Environment config: {env_config_path}")
        sys.exit(1)

    args = load_config(str(alg_config_path), str(env_config_path))

    # Override with command line args
    args["seed"] = args_cmd.seed
    args["use_gui"] = args_cmd.use_gui
    args["eval_episodes"] = args_cmd.episodes
    args["use_cuda"] = False  # Evaluation on CPU
    args["use_tensorboard"] = False

    # Determine policy type and run evaluation
    if args_cmd.model is not None:
        results = evaluate_policy(args, policy_type="qmix", model_path=args_cmd.model)
        policy_stem = args_cmd.output or "qmix"
    else:
        results = evaluate_policy(args, policy_type="baseline", baseline_type=args_cmd.baseline)
        policy_stem = args_cmd.output or args_cmd.baseline

    # Save results. If an explicit --output stem was provided by the caller
    # (e.g. "qmix_seed0"), use it directly — the seed is already embedded.
    # Otherwise fall back to appending _seed{N} for standalone invocations.
    if args_cmd.output:
        output_path = f"results/eval/{policy_stem}.json"
    else:
        output_path = f"results/eval/{policy_stem}_seed{args['seed']}.json"
    save_results(results, output_path)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
