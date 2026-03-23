"""
Quick test to verify SUMO connection works
"""
import os
import sys
from pathlib import Path

# Add pymarl/src to path
sys.path.insert(0, str(Path(__file__).parent / "pymarl" / "src"))

def test_sumo_connection():
    print("=" * 60)
    print("SUMO Connection Test")
    print("=" * 60)

    # Check SUMO_HOME
    sumo_home = os.environ.get("SUMO_HOME")
    print(f"\nSUMO_HOME: {sumo_home}")

    if not sumo_home:
        print("[FAIL] SUMO_HOME not set!")
        return False

    # Check if SUMO binary exists
    sumo_binary = os.path.join(sumo_home, "bin", "sumo.exe")
    if not os.path.exists(sumo_binary):
        sumo_binary = os.path.join(sumo_home, "bin", "sumo")

    print(f"SUMO binary: {sumo_binary}")
    print(f"SUMO binary exists: {os.path.exists(sumo_binary)}")

    # Check scenario files
    print("\nChecking scenario files...")
    sumocfg = Path("sumo/scenarios/4by4_map/train_med.sumocfg")
    print(f"SUMO config: {sumocfg}")
    print(f"Config exists: {sumocfg.exists()}")

    network_file = Path("4by4_map/final_map.net.xml")
    print(f"Network file: {network_file}")
    print(f"Network exists: {network_file.exists()}")

    routes_file = Path("sumo/scenarios/4by4_map/controlled_init.rou.xml")
    print(f"Routes file: {routes_file}")
    print(f"Routes exists: {routes_file.exists()}")

    # Try to import traci
    print("\nTesting TraCI import...")
    try:
        import traci
        print("[OK] TraCI imported successfully")
    except Exception as e:
        print(f"[FAIL] Failed to import TraCI: {e}")
        return False

    # Try to import environment
    print("\nTesting environment import...")
    try:
        from envs.sumo_grid_reroute import SUMOGridRerouteEnv
        print("[OK] SUMOGridRerouteEnv imported successfully")
    except Exception as e:
        print(f"[FAIL] Failed to import environment: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Try to create environment
    print("\nTesting environment creation...")
    try:
        env_args = {
            "sumo_cfg": "sumo/scenarios/4by4_map/train_med.sumocfg",
            "network_file": "4by4_map/final_map.net.xml",
            "controlled_routes": "sumo/scenarios/4by4_map/controlled_init.rou.xml",
            "n_agents": 32,
            "n_actions": 4,
            "decision_period": 10,
            "sumo_step_length": 1.0,
            "max_episode_steps": 50,  # Short for test
            "episode_limit": 50,
            "replacement_enabled": True,
            "replacement_delay": 3,
            "obs_ego_dim": 5,
            "obs_edge_encoding": "onehot",
            "obs_edge_dim": 48,
            "obs_max_outgoing_edges": 4,
            "obs_traffic_features": 3,
            "obs_local_traffic": True,
            "state_concat_obs": True,
            "state_include_global_stats": False,
            "state_global_stats_dim": 0,
            "route_generation_method": "k_shortest",
            "route_cost_metric": "length",
            "route_refresh_each_step": True,
            "action_mask_enabled": True,
            "action_noop_as_keep_route": True,
            "reward_global": True,
            "reward_time_weight": 1.0,
            "reward_stops_weight": 0.05,
            "reward_emissions_weight": 0.001,
            "reward_stop_speed_threshold": 0.1,
            "reward_reroute_penalty": 0.0,
            "emissions_enabled": False,  # Disable for test
            "emissions_device": "CO2",
            "use_active_mask": True,
            "use_reset_mask": True,
            "sumo_gui": False,
            "sumo_seed": 42,
            "sumo_warnings": False,
            "verbose": True,
            "render_mode": None
        }

        print("Creating environment...")
        env = SUMOGridRerouteEnv(env_args)
        print("[OK] Environment created successfully")

        # Try to reset
        print("\nTesting environment reset...")
        env.reset()
        print("[OK] Environment reset successfully")

        # Get env info
        env_info = env.get_env_info()
        print("\nEnvironment info:")
        print(f"  n_agents: {env_info['n_agents']}")
        print(f"  n_actions: {env_info['n_actions']}")
        print(f"  obs_shape: {env_info['obs_shape']}")
        print(f"  state_shape: {env_info['state_shape']}")
        print(f"  episode_limit: {env_info.get('episode_limit', 'N/A')}")

        # Try one step
        print("\nTesting environment step with random actions...")
        import numpy as np
        actions = np.random.randint(0, env_info['n_actions'], size=env_info['n_agents'])
        reward, terminated, info = env.step(actions)
        print(f"[OK] Step completed: reward={reward:.3f}, terminated={terminated}")

        # Close environment
        print("\nClosing environment...")
        env.close()
        print("[OK] Environment closed successfully")

        print("\n" + "=" * 60)
        print("[SUCCESS] All SUMO connection tests passed!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n[FAIL] Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_sumo_connection()
    sys.exit(0 if success else 1)
