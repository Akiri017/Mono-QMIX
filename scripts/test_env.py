"""
Test script for SUMO Grid Reroute Environment

This script performs basic sanity checks on the environment:
1. Environment initialization
2. Reset functionality
3. Step execution with random actions
4. Observation/state/mask shapes
5. Episode completion

Usage:
    python scripts/test_env.py
"""

import os
import sys
import numpy as np
import yaml

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add pymarl to path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(repo_root, "pymarl", "src"))

from envs import SUMOGridRerouteEnv


def load_config(config_path: str) -> dict:
    """Load environment configuration from YAML."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config["env_args"]


def test_env_basic():
    """Test basic environment functionality."""
    print("=" * 60)
    print("TEST: Basic Environment Functionality")
    print("=" * 60)

    # Load config
    config_path = os.path.join(repo_root, "pymarl", "src", "config", "envs", "sumo_grid4x4.yaml")
    env_args = load_config(config_path)

    # Set to headless and verbose
    env_args["sumo_gui"] = False
    env_args["verbose"] = True
    env_args["max_episode_steps"] = 100  # Short episode for testing

    print(f"\n1. Initializing environment...")
    env = SUMOGridRerouteEnv(env_args)
    print(f"   ✓ Environment created")
    print(f"   - n_agents: {env.n_agents}")
    print(f"   - n_actions: {env.n_actions}")
    print(f"   - obs_dim: {env.obs_dim}")
    print(f"   - state_dim: {env.state_dim}")

    try:
        print(f"\n2. Resetting environment...")
        env.reset()
        print(f"   ✓ Reset successful")

        print(f"\n3. Checking observations...")
        obs = env.get_obs()
        print(f"   ✓ Observations shape: {obs.shape}")
        assert obs.shape == (env.n_agents, env.obs_dim), f"Expected ({env.n_agents}, {env.obs_dim}), got {obs.shape}"

        print(f"\n4. Checking state...")
        state = env.get_state()
        print(f"   ✓ State shape: {state.shape}")
        assert state.shape == (env.state_dim,), f"Expected ({env.state_dim},), got {state.shape}"

        print(f"\n5. Checking available actions...")
        avail_actions = env.get_avail_actions()
        print(f"   ✓ Available actions shape: {avail_actions.shape}")
        assert avail_actions.shape == (env.n_agents, env.n_actions)

        print(f"\n6. Checking active mask...")
        active_mask = env.get_active_mask()
        print(f"   ✓ Active mask shape: {active_mask.shape}")
        print(f"   - Active agents: {np.sum(active_mask)}/{env.n_agents}")

        print(f"\n7. Checking reset mask...")
        reset_mask = env.get_reset_mask()
        print(f"   ✓ Reset mask shape: {reset_mask.shape}")
        print(f"   - Reset agents: {np.sum(reset_mask)}/{env.n_agents}")

        print(f"\n8. Taking random actions...")
        for step in range(5):
            # Generate random actions (respecting available actions)
            actions = []
            for i in range(env.n_agents):
                valid_actions = np.where(avail_actions[i] > 0)[0]
                if len(valid_actions) > 0:
                    actions.append(np.random.choice(valid_actions))
                else:
                    actions.append(0)
            actions = np.array(actions)

            # Take step
            reward, terminated, info = env.step(actions)

            print(f"   Step {step + 1}:")
            print(f"     - Reward: {reward:.2f}")
            print(f"     - Terminated: {terminated}")
            print(f"     - Sim time: {info['sim_time']:.1f}s")
            print(f"     - Active agents: {info['active_agents']}/{env.n_agents}")

            # Update available actions for next step
            avail_actions = env.get_avail_actions()

            if terminated:
                print(f"   ✓ Episode terminated after {step + 1} steps")
                break

        print(f"\n9. Closing environment...")
        env.close()
        print(f"   ✓ Environment closed successfully")

        print(f"\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        sys.exit(1)


def test_env_info():
    """Test environment info dictionary."""
    print("\n" + "=" * 60)
    print("TEST: Environment Info")
    print("=" * 60)

    config_path = os.path.join(repo_root, "pymarl", "src", "config", "envs", "sumo_grid4x4.yaml")
    env_args = load_config(config_path)
    env_args["sumo_gui"] = False
    env_args["verbose"] = False

    env = SUMOGridRerouteEnv(env_args)
    env.reset()

    info = env.get_env_info()
    print(f"\nEnvironment Info:")
    for key, value in info.items():
        print(f"  - {key}: {value}")

    env.close()
    print("\n✓ Environment info test passed")


if __name__ == "__main__":
    # Run tests
    test_env_basic()
    test_env_info()

    print("\n" + "=" * 60)
    print("🎉 All environment tests completed successfully!")
    print("=" * 60)
