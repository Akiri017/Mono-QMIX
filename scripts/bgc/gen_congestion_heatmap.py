"""
Generate congestion heatmap data from a Civiq evaluation episode.

Run from repo root:
    python scripts/bgc/gen_congestion_heatmap.py --los low
    python scripts/bgc/gen_congestion_heatmap.py --los low --compare

Outputs JSON to frontend/public/heatmap_output/bgc_full/
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "pymarl" / "src"))

import traci

# ── Config ────────────────────────────────────────────────────────────────────
CIVIQ_MODELS = {
    "low":  str(repo_root / "results/civiq/civiq-los-a/best"),
    "med":  str(repo_root / "results/civiq/civiq-los-c/best"),
    "high": str(repo_root / "results/civiq/civiq-los-e/best"),
}

MONO_MODELS = {
    "low":  str(repo_root / "results/mono-qmix/mono-qmix-los-a/seed_1801/best"),
    "med":  str(repo_root / "results/mono-qmix/mono-qmix-los-c/seed_1802/best"),
    "high": str(repo_root / "results/mono-qmix/mono-qmix-los-e/seed_1803/best"),
}

SUMOCFG = {
    "low":  str(repo_root / "sumo/scenarios/bgc_full/train_low.sumocfg"),
    "med":  str(repo_root / "sumo/scenarios/bgc_full/train_med.sumocfg"),
    "high": str(repo_root / "sumo/scenarios/bgc_full/train_high.sumocfg"),
}

OUTPUT_DIR = repo_root / "frontend" / "public" / "heatmap_output" / "bgc_full"
COLLECT_EVERY = 5   # collect lane metrics every N steps (reduce noise + file size)
EPISODE_STEPS = 100  # match your episode_limit


# ── Lane metric collection ────────────────────────────────────────────────────
def get_all_lanes():
    """Get all lane IDs that have at least one vehicle at any point."""
    return traci.lane.getIDList()


def collect_lane_snapshot(lane_ids):
    """Collect per-lane metrics at the current simulation step."""
    snapshot = []
    for lane_id in lane_ids:
        try:
            shape = traci.lane.getShape(lane_id)
            if len(shape) < 2:
                continue
            mid_x = np.mean([p[0] for p in shape])
            mid_y = np.mean([p[1] for p in shape])
            snapshot.append({
                "lane_id": lane_id,
                "x": round(mid_x, 2),
                "y": round(mid_y, 2),
                "density":      traci.lane.getLastStepVehicleNumber(lane_id),
                "speed":        round(traci.lane.getLastStepMeanSpeed(lane_id), 3),
                "occupancy":    round(traci.lane.getLastStepOccupancy(lane_id), 4),
                "waiting_time": round(traci.lane.getWaitingTime(lane_id), 3),
                "halting":      traci.lane.getLastStepHaltingNumber(lane_id),
            })
        except traci.TraCIException:
            continue
    return snapshot


# ── Run a single policy and collect spatial data ──────────────────────────────
def run_episode_with_collection(sumocfg, model_path, policy_label,
                                 alg_config, env_config, seed, los_level):
    """
    Runs evaluate.py's infrastructure for action selection,
    while separately collecting lane metrics via TraCI.

    Returns: dict with per-step snapshots and episode aggregate
    """
    import yaml
    import torch
    from controllers.basic_controller import BasicMAC
    from runners.episode_runner import EpisodeRunner
    from utils.logging import Logger

    # Load configs
    alg_cfg_path = repo_root / "pymarl/src/config/algs" / f"{alg_config}.yaml"
    env_cfg_path = repo_root / "pymarl/src/config/envs" / f"{env_config}.yaml"

    with open(alg_cfg_path) as f:
        alg_cfg = yaml.safe_load(f)
    with open(env_cfg_path) as f:
        env_cfg = yaml.safe_load(f)

    args = {**alg_cfg, **env_cfg}
    args["seed"] = seed
    args["eval_episodes"] = 1
    args["use_cuda"] = False
    args["use_tensorboard"] = False
    args["use_gui"] = False           # headless — we collect data, not watch
    args["env_args"]["los_level"] = los_level
    args["env_args"]["sumo_gui"] = False
    args["enable_cpu_monitoring"] = False

    logger = Logger(use_tensorboard=False, log_dir=None)
    runner = EpisodeRunner(args, logger)
    env_info = runner.get_env_info()
    args.update(env_info)

    scheme = {
        "state":        {"vshape": env_info["state_shape"]},
        "obs":          {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions":      {"vshape": (1,), "group": "agents", "dtype": torch.long},
        "avail_actions":{"vshape": (env_info["n_actions"],), "group": "agents", "dtype": torch.int},
        "reset_mask":   {"vshape": (1,), "group": "agents", "dtype": torch.uint8},
        "reward":       {"vshape": (1,)},
        "terminated":   {"vshape": (1,), "dtype": torch.uint8},
        "filled":       {"vshape": (1,), "dtype": torch.uint8},
    }

    if args.get("mixer") == "civiq":
        scheme.update({
            "zone_assignments":   {"vshape": (env_info["n_agents"],), "dtype": torch.int32},
            "rsu_agent_qs":       {"vshape": (args["max_rsus"], args["max_agents_per_rsu"]), "dtype": torch.float32},
            "agent_masks_per_rsu":{"vshape": (args["max_rsus"], args["max_agents_per_rsu"]), "dtype": torch.float32},
        })

    groups = {"agents": args["n_agents"]}
    mac = BasicMAC(scheme, groups, args)

    # Load weights
    agent_path = os.path.join(model_path, "agent.pth")
    mac.agent.load_state_dict(torch.load(agent_path, map_location="cpu"))
    print(f"  [{policy_label}] Agent loaded from {agent_path}")

    # Load mixer weights if Civiq
    if args.get("mixer") == "civiq":
        lm_path = os.path.join(model_path, "local_mixer.th")
        gm_path = os.path.join(model_path, "global_mixer.th")
        if hasattr(mac, 'local_mixer') and os.path.exists(lm_path):
            mac.local_mixer.load_state_dict(torch.load(lm_path, map_location="cpu"))
            print(f"  [{policy_label}] Local mixer loaded")
        if hasattr(mac, 'global_mixer') and os.path.exists(gm_path):
            mac.global_mixer.load_state_dict(torch.load(gm_path, map_location="cpu"))
            print(f"  [{policy_label}] Global mixer loaded")

    runner.setup(scheme, groups, {}, mac)

    # ── Monkey-patch the runner's step to also collect lane data ──────────────
    # We hook into the environment's TraCI connection after each step.
    # This works because EpisodeRunner calls env.step() which advances SUMO,
    # so TraCI state is current when we read it.

    step_snapshots = []
    original_step = runner.env.step

    def patched_step(actions):
        result = original_step(actions)
        current_step = runner.t  # runner.t increments after step
        if current_step % COLLECT_EVERY == 0:
            lane_ids = get_all_lanes()
            snapshot = collect_lane_snapshot(lane_ids)
            step_snapshots.append({
                "step": current_step,
                "lanes": snapshot
            })
        return result

    runner.env.step = patched_step

    # Run episode
    episode_batch, ep_metrics = runner.run(test_mode=True)
    runner.close_env()
    logger.close()

    return {
        "policy": policy_label,
        "los_level": los_level,
        "ep_metrics": ep_metrics,
        "step_snapshots": step_snapshots,
    }


# ── Aggregate snapshots to per-lane means ─────────────────────────────────────
def aggregate_to_lanes(step_snapshots):
    """Collapse per-step snapshots into per-lane mean metrics."""
    from collections import defaultdict
    accum = defaultdict(lambda: {
        "x": 0, "y": 0,
        "density": [], "speed": [],
        "occupancy": [], "waiting_time": [], "halting": []
    })

    for snap in step_snapshots:
        for lane in snap["lanes"]:
            lid = lane["lane_id"]
            accum[lid]["x"] = lane["x"]
            accum[lid]["y"] = lane["y"]
            for metric in ["density", "speed", "occupancy", "waiting_time", "halting"]:
                accum[lid][metric].append(lane[metric])

    result = []
    for lid, data in accum.items():
        result.append({
            "lane_id": lid,
            "x": data["x"],
            "y": data["y"],
            "density":      round(np.mean(data["density"]), 4),
            "speed":        round(np.mean(data["speed"]), 4),
            "occupancy":    round(np.mean(data["occupancy"]), 4),
            "waiting_time": round(np.mean(data["waiting_time"]), 4),
            "halting":      round(np.mean(data["halting"]), 4),
        })
    return result


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--los", choices=["low", "med", "high"], default="low")
    parser.add_argument("--seed", type=int, default=1804)
    parser.add_argument("--compare", action="store_true",
                        help="Also run Mono-QMIX baseline for comparison")
    parser.add_argument("--alg_config", default="civiq_sumo")
    parser.add_argument("--env_config", default="sumo_bgc_full")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Run Civiq ─────────────────────────────────────────────────────────────
    print(f"\nRunning Civiq (LOS {args.los.upper()})...")
    civiq_data = run_episode_with_collection(
        sumocfg=SUMOCFG[args.los],
        model_path=CIVIQ_MODELS[args.los],
        policy_label="civiq",
        alg_config=args.alg_config,
        env_config=args.env_config,
        seed=args.seed,
        los_level=args.los,
    )
    civiq_lanes = aggregate_to_lanes(civiq_data["step_snapshots"])

    out = {
        "policy":    "civiq",
        "los_level": args.los,
        "seed":      args.seed,
        "ep_metrics": civiq_data["ep_metrics"],
        "lanes":     civiq_lanes,
        "n_snapshots": len(civiq_data["step_snapshots"]),
    }
    out_path = OUTPUT_DIR / f"civiq_los_{args.los}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"  Saved -> {out_path}")

    # ── Optionally run Mono-QMIX ──────────────────────────────────────────────
    if args.compare:
        print(f"\nRunning Mono-QMIX baseline (LOS {args.los.upper()})...")
        mono_data = run_episode_with_collection(
            sumocfg=SUMOCFG[args.los],
            model_path=MONO_MODELS[args.los],
            policy_label="mono_qmix",
            alg_config="qmix_sumo",     # baseline uses vanilla QMIX config
            env_config=args.env_config,
            seed=args.seed,
            los_level=args.los,
        )
        mono_lanes = aggregate_to_lanes(mono_data["step_snapshots"])

        out_mono = {
            "policy":    "mono_qmix",
            "los_level": args.los,
            "seed":      args.seed,
            "ep_metrics": mono_data["ep_metrics"],
            "lanes":     mono_lanes,
            "n_snapshots": len(mono_data["step_snapshots"]),
        }
        out_path_mono = OUTPUT_DIR / f"mono_qmix_los_{args.los}.json"
        out_path_mono.write_text(json.dumps(out_mono, indent=2))
        print(f"  Saved -> {out_path_mono}")

    print("\nDone. Files written to:")
    for f in OUTPUT_DIR.iterdir():
        print(f"  {f}")


if __name__ == "__main__":
    main()