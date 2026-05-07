"""
test_blockage.py — Verify random road blockage injection is working.

Runs a short episode on the 4x4 map with high blockage probability
and prints a summary of all blockage events that fired and cleared.
"""

import os
import sys
import numpy as np # pyright: ignore[reportMissingImports]
from pathlib import Path

# Add pymarl/src to path
sys.path.insert(0, str(Path(__file__).parent / "pymarl" / "src"))

from envs.sumo_grid_reroute import SUMOGridRerouteEnv # pyright: ignore[reportMissingImports]

def test_blockages():
    print("=" * 60)
    print("Road Blockage Injection Test")
    print("=" * 60)

    env_args = {
        # Scenario — using 4x4 map for speed
        "sumo_cfg": "sumo/scenarios/4by4_map/train_med.sumocfg",
        "network_file": "4by4_map/final_map.net.xml",
        "controlled_routes": "sumo/scenarios/4by4_map/controlled_init.rou.xml",

        # Agents
        "n_agents": 4,
        "n_actions": 2,
        "decision_period": 10,
        "sumo_step_length": 1.0,
        "max_episode_steps": 300,  # 300 seconds = enough to see multiple blockages

        # Replacement
        "replacement_enabled": True,
        "replacement_delay": 3,

        # Observations
        "obs_ego_dim": 5,
        "obs_edge_encoding": "onehot",
        "obs_edge_dim": 48,
        "obs_max_outgoing_edges": 4,
        "obs_traffic_features": 3,
        "obs_local_traffic": True,

        # State
        "state_concat_obs": True,
        "state_include_global_stats": False,
        "state_global_stats_dim": 0,

        # Actions
        "route_generation_method": "k_shortest",
        "route_cost_metric": "length",
        "route_refresh_each_step": True,
        "action_mask_enabled": True,
        "action_noop_as_keep_route": True,

        # Reward
        "reward_global": True,
        "reward_time_weight": 1.0,
        "reward_stops_weight": 0.05,
        "reward_emissions_weight": 0.001,
        "reward_stop_speed_threshold": 0.1,
        "reward_reroute_penalty": 0.0,

        # Emissions off for speed
        "emissions_enabled": False,
        "emissions_device": "CO2",

        # Masking
        "use_active_mask": True,
        "use_reset_mask": True,

        # SUMO settings
        "sumo_backend": "libsumo",
        "sumo_gui": False,
        "sumo_seed": 42,
        "sumo_warnings": False,
        "verbose": False,  # keep quiet — we print our own summary
        "render_mode": None,
        "enable_cpu_monitoring": False,

        # Blockage config — high probability for testing
        "blockage_enabled": True,
        "blockage_probability": 0.1,    # 10% per second — fires often
        "blockage_min_duration": 30,    # short so we see clears too
        "blockage_max_duration": 60,
        "blockage_max_concurrent": 3,
    }

    print("\nCreating environment...")
    try:
        env = SUMOGridRerouteEnv(env_args)
        print("[OK] Environment created")
    except Exception as e:
        print(f"[FAIL] Environment creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("Resetting environment...")
    try:
        env.reset()
        print("[OK] Environment reset\n")
    except Exception as e:
        print(f"[FAIL] Reset failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Run episode and track blockage events from logs
    print("Running episode (300 sim seconds)...")
    print("-" * 60)

    step = 0
    terminated = False
    total_reward = 0.0

    while not terminated:
        actions = np.zeros(env_args["n_agents"], dtype=int)  # always noop
        reward, terminated, info = env.step(actions)
        total_reward += reward
        step += 1

        # Print blockage state every 5 decision steps
        if step % 5 == 0:
            blocked = env.blockage_manager.get_blocked_edges()
            print(
                f"  Step {step:3d} | "
                f"sim_time={info['sim_time']:.0f}s | "
                f"reward={reward:.1f} | "
                f"blocked_edges={len(blocked)} {blocked}"
            )

    print("-" * 60)

    # Final blockage summary
    blocked = env.blockage_manager.get_blocked_edges()
    print(f"\nEpisode finished after {step} decision steps")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Blockages still active at end: {len(blocked)} {blocked}")

    env.close()
    print("\n[SUCCESS] Blockage test complete!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_blockages()
    sys.exit(0 if success else 1)