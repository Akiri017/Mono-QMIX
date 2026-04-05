"""
Phase 5 Gate — Gradient Flow and Shape Test

Runs 1 real SUMO episode (50 steps, train_med.sumocfg), collects 1 batch,
runs 1 HierarchicalQLearner.train() step, and verifies:
  - Shapes of key intermediates (computed from batch fields)
  - All MAC / LocalQMixer / GlobalQMixer parameters have non-None, non-NaN grads
  - global_qtot is finite and loss is in a sane range

Run from repo root:
    python tests/test_phase5_gradient_flow.py
"""

import os
import sys
import torch
import numpy as np

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(repo_root, "pymarl", "src"))

import yaml
from runners.episode_runner import EpisodeRunner
from components.episode_buffer import EpisodeBatch, ReplayBuffer
from learners.hierarchical_q_learner import HierarchicalQLearner
from controllers.basic_controller import BasicMAC


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_args():
    alg_path = os.path.join(repo_root, "pymarl", "src", "config", "algs", "civiq_sumo.yaml")
    env_path = os.path.join(repo_root, "pymarl", "src", "config", "envs", "sumo_grid4x4.yaml")
    with open(alg_path) as f:
        alg_cfg = yaml.safe_load(f)
    with open(env_path) as f:
        env_cfg = yaml.safe_load(f)
    args = {**alg_cfg, **env_cfg}

    args["episode_limit"] = 50
    args["use_cuda"] = False
    args["sumo_backend"] = "libsumo"
    args["los_level"] = "med"
    args["sumo_cfg"] = None
    args["mixer"] = "civiq"
    args["batch_size_run"] = 1
    args["batch_size"] = 1      # train on 1 episode (buffer has exactly 1)

    if "rsu_config" not in args:
        args["rsu_config"] = "config/envs/rsu/synthetic_4x4.yaml"

    return args


def build_civiq_scheme(env_info, args):
    max_rsus = args["max_rsus"]
    max_agents_per_rsu = args["max_agents_per_rsu"]
    obs_dim = args["obs_dim"]
    n_agents = env_info["n_agents"]

    return {
        "state":    {"vshape": env_info["state_shape"]},
        "obs":      {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions":  {"vshape": (1,), "group": "agents", "dtype": torch.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": torch.int},
        "reward":   {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": torch.uint8},
        "filled":   {"vshape": (1,), "dtype": torch.uint8},
        "zone_assignments": {"vshape": (n_agents,), "dtype": torch.int32},
        "rsu_agent_qs": {"vshape": (max_rsus, max_agents_per_rsu), "dtype": torch.float32},
        "agent_masks_per_rsu": {"vshape": (max_rsus, max_agents_per_rsu), "dtype": torch.float32},
        "local_states": {"vshape": (max_rsus, max_agents_per_rsu * obs_dim), "dtype": torch.float32},
    }


# ---------------------------------------------------------------------------
# Collect one episode and return a 1-episode batch + learner
# Learner is created BEFORE the episode runs (matching main.py ordering)
# so that deepcopy of MAC happens while it's still a graph leaf.
# ---------------------------------------------------------------------------
def collect_episode(args):
    logger_stub = type("L", (), {
        "log_stat": lambda self, *a, **kw: None,
        "print_recent_stats": lambda self: None,
    })()

    runner = EpisodeRunner(args, logger_stub)
    env_info = runner.get_env_info()
    args.update(env_info)

    scheme = build_civiq_scheme(env_info, args)
    groups = {"agents": args["n_agents"]}

    mac = BasicMAC(scheme, groups, args)

    # Create learner BEFORE running episode (deepcopy while MAC is fresh)
    learner = HierarchicalQLearner(mac, scheme, logger_stub, args)
    learner.log_stats_t = -999999  # force logging on first step

    runner.setup(scheme, groups, preprocess={}, mac=mac)

    # Use a tiny replay buffer (1 episode) so we can sample immediately
    buffer = ReplayBuffer(
        scheme, groups,
        buffer_size=1,
        max_seq_length=args["episode_limit"] + 1,
        preprocess={},
        device="cpu",
    )

    episode_batch, _ = runner.run(test_mode=False)
    buffer.insert_episode_batch(episode_batch)
    runner.close_env()

    train_batch = buffer.sample(1)
    return train_batch, learner, scheme, groups, args


# ---------------------------------------------------------------------------
# Test 1 — shape assertions on batch fields (pre-train sanity)
# ---------------------------------------------------------------------------
def test_batch_shapes(batch, args):
    n_agents = args["n_agents"]
    max_rsus = args["max_rsus"]
    max_agents_per_rsu = args["max_agents_per_rsu"]
    obs_dim = args["obs_dim"]
    B, T_plus1 = batch["zone_assignments"].shape[0], batch["zone_assignments"].shape[1]
    T = T_plus1 - 1  # filled timesteps = episode_limit + 1, real transitions = T

    # chosen_action_qvals shape: (B, T, n_agents) — verified via batch fields
    assert batch["actions"].shape == (B, T_plus1, n_agents, 1), \
        f"actions shape wrong: {batch['actions'].shape}"
    assert batch["zone_assignments"].shape == (B, T_plus1, n_agents), \
        f"zone_assignments shape wrong: {batch['zone_assignments'].shape}"
    assert batch["agent_masks_per_rsu"].shape == (B, T_plus1, max_rsus, max_agents_per_rsu), \
        f"agent_masks_per_rsu shape wrong: {batch['agent_masks_per_rsu'].shape}"
    assert batch["local_states"].shape == (B, T_plus1, max_rsus, max_agents_per_rsu * obs_dim), \
        f"local_states shape wrong: {batch['local_states'].shape}"
    assert batch["state"].shape[2] == args["global_state_dim"], \
        f"state last dim wrong: {batch['state'].shape[2]} != {args['global_state_dim']}"

    print(f"  Test 1 PASSED — batch shapes correct, B={B}, T+1={T_plus1}")


# ---------------------------------------------------------------------------
# Test 2 — train() runs without error; global_qtot and loss are sane
# ---------------------------------------------------------------------------
def test_train_step(batch, learner, args):
    stats = learner.train(batch, t_env=1, episode_num=1)

    loss = stats["loss"]
    global_qtot_mean = stats["global_qtot_mean"]

    assert not (loss != loss), f"loss is NaN: {loss}"           # NaN check
    assert not (loss == float('inf')), f"loss is Inf: {loss}"
    assert loss > 0, f"loss is zero or negative: {loss}"
    assert loss < 1e8, f"loss is implausibly large: {loss}"
    assert not (global_qtot_mean != global_qtot_mean), "global_qtot_mean is NaN"
    assert not (global_qtot_mean == float('inf')), "global_qtot_mean is Inf"

    print(f"  Test 2 PASSED — train() completed:")
    print(f"    loss={loss:.6f}")
    print(f"    global_qtot_mean={global_qtot_mean:.6f}")
    print(f"    agent_grad_norm={stats['agent_grad_norm']:.6f}")
    print(f"    local_mixer_grad_norm={stats['local_mixer_grad_norm']:.6f}")
    print(f"    global_mixer_grad_norm={stats['global_mixer_grad_norm']:.6f}")


# ---------------------------------------------------------------------------
# Test 3 — gradient assertions (after backward; grads still live after step)
# ---------------------------------------------------------------------------
def test_gradient_flow(learner):
    # After train(), grads are live (zero_grad not called again yet)
    failed = []

    # BasicMAC exposes parameters() but not named_parameters()
    for i, param in enumerate(learner.mac.parameters()):
        if param.requires_grad:
            if param.grad is None:
                failed.append(f"mac.param[{i}]: grad is None")
            elif torch.isnan(param.grad).any():
                failed.append(f"mac.param[{i}]: grad has NaN")

    # LocalQMixer and GlobalQMixer are nn.Modules — named_parameters() is available
    for name, param in learner.local_mixer.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                failed.append(f"local_mixer.{name}: grad is None")
            elif torch.isnan(param.grad).any():
                failed.append(f"local_mixer.{name}: grad has NaN")

    for name, param in learner.global_mixer.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                failed.append(f"global_mixer.{name}: grad is None")
            elif torch.isnan(param.grad).any():
                failed.append(f"global_mixer.{name}: grad has NaN")

    if failed:
        raise AssertionError("Gradient failures:\n  " + "\n  ".join(failed))

    mac_params     = sum(p.numel() for p in learner.mac.parameters() if p.requires_grad)
    local_params   = sum(p.numel() for p in learner.local_mixer.parameters())
    global_params  = sum(p.numel() for p in learner.global_mixer.parameters())
    print(f"  Test 3 PASSED — gradients non-None, non-NaN for all parameters:")
    print(f"    MAC params with grad: {mac_params}")
    print(f"    LocalQMixer params:   {local_params}")
    print(f"    GlobalQMixer params:  {global_params}")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\nPhase 5 gradient flow and shape tests")
    print("  Episode limit: 50 steps")
    print("  Scenario:      train_med.sumocfg (LOS B)")
    print()

    print("  Collecting episode (starting SUMO)...")
    args = load_args()
    batch, learner, scheme, groups, args = collect_episode(args)
    T = batch["zone_assignments"].shape[1]
    print(f"  Episode collected: {T} timesteps (T+1)\n")

    test_batch_shapes(batch, args)
    test_train_step(batch, learner, args)
    test_gradient_flow(learner)

    print("\nSTEP 5.1 + 5.2 PASSED")
