"""
Policy Comparison Heatmaps — BGC Full

Runs N eval episodes for each policy (CiViQ, Mono-QMIX, Noop), collects
per-edge mean speed at every simulation step via libsumo, then renders
side-by-side network heatmaps coloured by congestion level.

Usage (from repo root):
    python scripts/generate_policy_heatmaps.py --los high --episodes 3
    python scripts/generate_policy_heatmaps.py --los low  --episodes 3

Output:
    frontend/public/heatmap_output/policy_comparison/heatmap_<los>.png
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import yaml
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH  = REPO_ROOT / "pymarl" / "src"
sys.path.insert(0, str(SRC_PATH))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import sumolib

from envs.sumo_backend import set_backend, backend as traci
from envs.sumo_grid_reroute import SUMOGridRerouteEnv as SumoGridRerouteEnv
from controllers.basic_controller import BasicMAC
from utils.logging import Logger

# ── config ─────────────────────────────────────────────────────────────────────

NETWORK_FILE = REPO_ROOT / "bgc_full" / "final_map.net.xml"

POLICIES = {
    "noop":      "Noop (No Routing)",
    "mono_qmix": "Mono-QMIX",
    "civiq":     "CiViQ",
}

LOS_LABEL = {"low": "LOS A (Low Traffic)", "med": "LOS C (Med Traffic)", "high": "LOS E (High Traffic)"}

MODEL_DIRS = {
    "civiq": {
        "low":  REPO_ROOT / "results/civiq/civiq-los-a/best",
        "med":  REPO_ROOT / "results/civiq/civiq-los-c/best",
        "high": REPO_ROOT / "results/civiq/civiq-los-e/best",
    },
    "mono_qmix": {
        "low":  REPO_ROOT / "results/mono-qmix/mono-qmix-los-a/seed_1801/best",
        "med":  REPO_ROOT / "results/mono-qmix/mono-qmix-los-c/seed_1802/best",
        "high": REPO_ROOT / "results/mono-qmix/mono-qmix-los-e/seed_1803/best",
    },
}


# ── helpers ────────────────────────────────────────────────────────────────────

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_env_args(los: str) -> dict:
    alg_cfg = load_yaml(SRC_PATH / "config/algs/civiq_sumo.yaml")
    env_cfg = load_yaml(SRC_PATH / "config/envs/sumo_bgc_full.yaml")
    args = {**alg_cfg, **env_cfg}
    args["env_args"]["los_level"] = los
    args["env_args"]["enable_cpu_monitoring"] = False
    args["seed"] = 42
    return args


def load_agent(model_dir: Path, obs_shape: int, n_actions: int, n_agents: int, args: dict):
    """Load agent network weights from checkpoint directory."""
    scheme = {
        "obs":          {"vshape": obs_shape, "group": "agents"},
        "actions":      {"vshape": (1,), "group": "agents", "dtype": torch.long},
        "avail_actions":{"vshape": (n_actions,), "group": "agents", "dtype": torch.int},
        "state":        {"vshape": args.get("state_shape", obs_shape * n_agents)},
        "reward":       {"vshape": (1,)},
        "terminated":   {"vshape": (1,), "dtype": torch.uint8},
        "filled":       {"vshape": (1,), "dtype": torch.uint8},
        "reset_mask":   {"vshape": (1,), "group": "agents", "dtype": torch.uint8},
    }
    mac = BasicMAC(scheme, {"agents": n_agents}, args)
    agent_path = model_dir / "agent.pth"
    mac.agent.load_state_dict(torch.load(str(agent_path), map_location="cpu"))
    mac.agent.eval()
    return mac


def select_actions(mac, obs_list, avail_list, hidden):
    """Greedy action selection given a loaded mac and current obs."""
    obs_t   = torch.tensor(np.array(obs_list), dtype=torch.float32)   # [n_agents, obs_dim]
    avail_t = torch.tensor(np.array(avail_list), dtype=torch.float32)  # [n_agents, n_actions]

    # agent forward: inputs [n_agents, obs_dim], hidden [n_agents, hidden_dim]
    with torch.no_grad():
        agent_outs, hidden_out = mac.agent(obs_t, hidden)  # [n_agents, n_actions]

    agent_outs[avail_t == 0] = -1e9
    actions = agent_outs.argmax(dim=-1).numpy()
    return actions, hidden_out


def collect_edge_speeds(env: SumoGridRerouteEnv, mac, n_episodes: int, policy: str) -> dict:
    """
    Run n_episodes and collect per-edge mean speed via libsumo.
    Returns {edge_id: mean_speed_m_s}.
    """
    edge_speed_acc = {}
    edge_speed_cnt = {}
    edge_ids = None

    for ep in range(n_episodes):
        env.reset()

        # Collect edge IDs once after first reset (simulation is live)
        if edge_ids is None:
            raw_ids = list(traci.edge.getIDList())
            edge_ids = [e for e in raw_ids if not e.startswith(":")]
            for eid in edge_ids:
                edge_speed_acc[eid] = 0.0
                edge_speed_cnt[eid] = 0

        terminated = False
        hidden = mac.agent.init_hidden().expand(env.n_agents, -1).clone() if mac else None

        while not terminated:
            # Collect edge speeds
            for eid in edge_ids:
                spd = traci.edge.getLastStepMeanSpeed(eid)
                if spd >= 0:
                    edge_speed_acc[eid] += spd
                    edge_speed_cnt[eid] += 1

            # Choose actions
            if policy == "noop":
                actions = np.zeros(env.n_agents, dtype=np.int32)
            else:
                obs   = env.get_obs()
                avail = env.get_avail_actions()
                actions, hidden = select_actions(mac, obs, avail, hidden)

            _, terminated, _ = env.step(actions)

        print(f"  Episode {ep+1}/{n_episodes} complete")

    # Average
    result = {}
    for eid in edge_ids:
        result[eid] = edge_speed_acc[eid] / max(edge_speed_cnt[eid], 1)
    return result


# ── plotting ───────────────────────────────────────────────────────────────────

def plot_comparison(edge_data: dict, net, los: str, out_path: Path):
    policy_order = ["noop", "mono_qmix", "civiq"]
    fig, axes = plt.subplots(1, 3, figsize=(24, 9))
    fig.patch.set_facecolor("#0d0d0d")
    fig.suptitle(
        f"BGC Full — Per-Edge Mean Speed\n{LOS_LABEL[los]}",
        color="white", fontsize=15, y=1.02
    )

    all_speeds = [s for p in policy_order for s in edge_data[p].values() if s > 0]
    vmin = 0.0
    vmax = np.percentile(all_speeds, 95) if all_speeds else 14.0

    cmap = matplotlib.colormaps.get_cmap("RdYlGn")

    for ax, policy in zip(axes, policy_order):
        ax.set_facecolor("#111111")
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(POLICIES[policy], color="white", fontsize=12, pad=6)

        speeds = edge_data[policy]
        segments, colors_rgba = [], []

        for edge in net.getEdges():
            eid = edge.getID()
            if eid.startswith(":"):
                continue
            shape = edge.getShape()
            if len(shape) < 2:
                continue
            spd = speeds.get(eid, 0.0)
            norm = np.clip((spd - vmin) / (vmax - vmin + 1e-9), 0.0, 1.0)
            segments.append(shape)
            colors_rgba.append(cmap(norm))

        lc = LineCollection(segments, colors=colors_rgba, linewidths=1.4, alpha=0.9)
        ax.add_collection(lc)
        ax.autoscale_view()

    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.03, pad=0.06)
    cbar.set_label("Mean Speed (m/s)  ·  Red = congested  ·  Green = free-flow",
                   color="white", fontsize=10)
    cbar.ax.xaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.xaxis.get_ticklabels(), color="white")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print(f"Saved: {out_path}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--los",      default="high", choices=["low", "med", "high"])
    parser.add_argument("--episodes", type=int, default=3)
    cli = parser.parse_args()

    set_backend("libsumo")
    net = sumolib.net.readNet(str(NETWORK_FILE))

    edge_data = {}
    policy_order = ["noop", "mono_qmix", "civiq"]

    for policy in policy_order:
        print(f"\n{'='*50}")
        print(f" Policy: {POLICIES[policy]}  |  LOS: {cli.los}")
        print(f"{'='*50}")

        args = build_env_args(cli.los)
        env  = SumoGridRerouteEnv(args["env_args"])
        info = env.get_env_info()
        args.update(info)

        mac = None
        if policy != "noop":
            mac = load_agent(
                MODEL_DIRS[policy][cli.los],
                info["obs_shape"],
                info["n_actions"],
                info["n_agents"],
                args,
            )

        edge_data[policy] = collect_edge_speeds(env, mac, cli.episodes, policy)
        env.close()

    out_path = (REPO_ROOT / "frontend/public/heatmap_output/policy_comparison"
                / f"heatmap_{cli.los}.png")
    plot_comparison(edge_data, net, cli.los, out_path)

    json_out = out_path.with_suffix(".json")
    with open(json_out, "w") as f:
        json.dump(edge_data, f)
    print(f"Edge data: {json_out}")


if __name__ == "__main__":
    main()
