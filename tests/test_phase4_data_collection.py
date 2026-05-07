"""
Phase 4 Gate — Data Collection Unit Tests

Runs a short SUMO episode (50 steps, train_med.sumocfg) through the modified
EpisodeRunner and checks that Civiq batch fields have correct shapes,
dtypes, and value constraints.

Run from repo root:
    python tests/test_phase4_data_collection.py
"""

import os
import sys
import torch
import numpy as np

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(repo_root, "pymarl", "src"))

import yaml
from runners.episode_runner import EpisodeRunner
from components.episode_buffer import EpisodeBatch

# ---------------------------------------------------------------------------
# Config — load real alg + env configs then override for a short test run
# ---------------------------------------------------------------------------
def load_args():
    alg_path = os.path.join(repo_root, "pymarl", "src", "config", "algs", "civiq_sumo.yaml")
    env_path = os.path.join(repo_root, "pymarl", "src", "config", "envs", "sumo_grid4x4.yaml")
    with open(alg_path) as f:
        alg_cfg = yaml.safe_load(f)
    with open(env_path) as f:
        env_cfg = yaml.safe_load(f)
    args = {**alg_cfg, **env_cfg}

    # Override for fast test run
    args["episode_limit"] = 50
    args["use_cuda"] = False
    args["sumo_backend"] = "libsumo"
    args["los_level"] = "med"
    args["sumo_cfg"] = None  # use los_level lookup
    args["mixer"] = "civiq"
    args["batch_size_run"] = 1

    # rsu_config is a repo-relative path (e.g. "config/envs/rsu/synthetic_4x4.yaml").
    # The env's _resolve_path() handles resolution relative to repo root — leave as-is.
    # (The runner forwards this key from args into env_args via _make_env)
    if "rsu_config" not in args:
        args["rsu_config"] = "config/envs/rsu/synthetic_4x4.yaml"

    return args


# ---------------------------------------------------------------------------
# Minimal stub MAC — returns action 0 (no-op) for all agents every step
# ---------------------------------------------------------------------------
class StubMAC:
    def __init__(self, n_agents, n_actions):
        self.n_agents = n_agents
        self.n_actions = n_actions

    def init_hidden(self, batch_size):
        pass

    def select_actions(self, ep_batch, t_ep, t_env, test_mode=False):
        # Always take action 0 (no-op / keep-route), which is always available
        return torch.zeros(1, self.n_agents, 1, dtype=torch.long)

    def forward(self, batch, t):
        return torch.zeros(1, self.n_agents, self.n_actions)

    def parameters(self):
        return iter([])

    def save_models(self, path):
        pass

    def load_models(self, path):
        pass

    def load_state(self, other):
        pass

    def cuda(self):
        pass

    def cpu(self):
        pass


# ---------------------------------------------------------------------------
# Build civiq scheme (mirrors main.py get_scheme with civiq fields)
# ---------------------------------------------------------------------------
def build_civiq_scheme(env_info, args):
    max_rsus = args["max_rsus"]
    max_agents_per_rsu = args["max_agents_per_rsu"]
    obs_dim = args["obs_dim"]
    n_agents = env_info["n_agents"]

    return {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": torch.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": torch.uint8},
        "filled": {"vshape": (1,), "dtype": torch.uint8},
        "zone_assignments": {
            "vshape": (n_agents,),
            "dtype": torch.int32,
        },
        "rsu_agent_qs": {
            "vshape": (max_rsus, max_agents_per_rsu),
            "dtype": torch.float32,
        },
        "agent_masks_per_rsu": {
            "vshape": (max_rsus, max_agents_per_rsu),
            "dtype": torch.float32,
        },
        "local_states": {
            "vshape": (max_rsus, max_agents_per_rsu * obs_dim),
            "dtype": torch.float32,
        },
    }


# ---------------------------------------------------------------------------
# Run episode and return batch
# ---------------------------------------------------------------------------
def collect_episode(args):
    logger_stub = type("L", (), {
        "log_stat": lambda self, *a, **kw: None,
        "print_recent_stats": lambda self: None,
    })()

    runner = EpisodeRunner(args, logger_stub)
    env_info = runner.get_env_info()
    args.update(env_info)

    mac = StubMAC(args["n_agents"], args["n_actions"])
    scheme = build_civiq_scheme(env_info, args)
    groups = {"agents": args["n_agents"]}

    runner.setup(scheme, groups, preprocess={}, mac=mac)

    batch, _ = runner.run(test_mode=True)
    runner.close_env()
    return batch, args


# ---------------------------------------------------------------------------
# Test 1 — zone_assignments shape and value range
# ---------------------------------------------------------------------------
def test_zone_assignments(batch, args):
    n_agents = args["n_agents"]
    T = batch["zone_assignments"].shape[1]

    assert batch["zone_assignments"].shape == (1, T, n_agents), \
        f"zone_assignments shape wrong: {batch['zone_assignments'].shape}"
    assert batch["zone_assignments"].dtype == torch.int32, \
        f"zone_assignments dtype wrong: {batch['zone_assignments'].dtype}"

    vals = batch["zone_assignments"]
    max_rsus = args["max_rsus"]
    assert ((vals >= -1) & (vals < max_rsus)).all(), \
        f"zone_assignments out of range: min={vals.min()}, max={vals.max()}"

    print(f"  Test 1 PASSED — zone_assignments shape={tuple(batch['zone_assignments'].shape)}, "
          f"range=[{vals.min()}, {vals.max()}]")


# ---------------------------------------------------------------------------
# Test 2 — local_states shape and no NaN
# ---------------------------------------------------------------------------
def test_local_states(batch, args):
    max_rsus = args["max_rsus"]
    max_agents_per_rsu = args["max_agents_per_rsu"]
    obs_dim = args["obs_dim"]
    T = batch["local_states"].shape[1]

    expected_last_dims = (max_rsus, max_agents_per_rsu * obs_dim)
    assert batch["local_states"].shape == (1, T, *expected_last_dims), \
        f"local_states shape wrong: {batch['local_states'].shape}"
    assert not torch.isnan(batch["local_states"]).any(), \
        "NaN in local_states"

    print(f"  Test 2 PASSED — local_states shape={tuple(batch['local_states'].shape)}, no NaN")


# ---------------------------------------------------------------------------
# Test 3 — agent_masks_per_rsu shape and binary values
# ---------------------------------------------------------------------------
def test_agent_masks(batch, args):
    max_rsus = args["max_rsus"]
    max_agents_per_rsu = args["max_agents_per_rsu"]
    T = batch["agent_masks_per_rsu"].shape[1]

    assert batch["agent_masks_per_rsu"].shape == (1, T, max_rsus, max_agents_per_rsu), \
        f"agent_masks_per_rsu shape wrong: {batch['agent_masks_per_rsu'].shape}"

    masks = batch["agent_masks_per_rsu"]
    assert ((masks == 0.0) | (masks == 1.0)).all(), \
        "agent_masks_per_rsu contains values other than 0.0 and 1.0"

    # No RSU should exceed max_agents_per_rsu real agents
    per_rsu_counts = masks.sum(dim=-1)  # (1, T, max_rsus)
    assert (per_rsu_counts <= max_agents_per_rsu).all(), \
        f"An RSU has more agents than max_agents_per_rsu={max_agents_per_rsu}"

    print(f"  Test 3 PASSED — agent_masks_per_rsu shape={tuple(masks.shape)}, "
          f"binary, max_per_rsu={per_rsu_counts.max().item():.0f}")


# ---------------------------------------------------------------------------
# Test 4 — Conservation: total mask 1s == number of active agent slots
# ---------------------------------------------------------------------------
def test_conservation(batch, args):
    n_agents = args["n_agents"]
    T = batch["agent_masks_per_rsu"].shape[1]

    # Total 1s across all RSU masks at each timestep
    total_masked = batch["agent_masks_per_rsu"].sum(dim=(-1, -2))  # (1, T)

    # Number of agents assigned to any RSU (zone_assignments != -1)
    assigned = (batch["zone_assignments"] != -1).sum(dim=-1).float()  # (1, T)

    # They must match: every agent assigned to an RSU gets a mask slot
    assert torch.allclose(total_masked, assigned), \
        (f"Conservation violated: mask_sum={total_masked[0, :5].tolist()} "
         f"!= assigned={assigned[0, :5].tolist()} (first 5 timesteps)")

    print(f"  Test 4 PASSED — conservation holds: total mask 1s == assigned agents "
          f"(mean={assigned.mean().item():.1f} agents/step)")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"\nPhase 4 data collection tests")
    print(f"  Episode limit: 50 steps")
    print(f"  Scenario:      train_med.sumocfg (LOS B)")
    print()

    print("  Collecting episode (starting SUMO)...")
    args = load_args()
    batch, args = collect_episode(args)
    T = batch["zone_assignments"].shape[1]
    print(f"  Episode collected: {T} timesteps\n")

    test_zone_assignments(batch, args)
    test_local_states(batch, args)
    test_agent_masks(batch, args)
    test_conservation(batch, args)

    print("\nPHASE 4 GATE PASSED")
