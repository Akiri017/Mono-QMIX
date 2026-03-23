"""
Test script for PyMARL QMIX training loop - Step 5 verification
Tests imports and runs a short smoke test of the training pipeline.
"""

import os
import sys
from pathlib import Path

# Add pymarl/src to path
sys.path.insert(0, str(Path(__file__).parent / "pymarl" / "src"))

def test_imports():
    """Test that all QMIX components can be imported."""
    print("=" * 60)
    print("TEST 1: Module Imports")
    print("=" * 60)

    try:
        print("Testing modules/agents/rnn_agent.py...")
        from modules.agents.rnn_agent import RNNAgent
        print("  [OK] RNNAgent imported successfully")
    except Exception as e:
        print(f"  [FAIL] Failed to import RNNAgent: {e}")
        return False

    try:
        print("Testing modules/mixers/qmix.py...")
        from modules.mixers.qmix import QMixer
        print("  [OK] QMixer imported successfully")
    except Exception as e:
        print(f"  [FAIL] Failed to import QMixer: {e}")
        return False

    try:
        print("Testing controllers/basic_controller.py...")
        from controllers.basic_controller import BasicMAC
        print("  [OK] BasicMAC imported successfully")
    except Exception as e:
        print(f"  [FAIL] Failed to import BasicMAC: {e}")
        return False

    try:
        print("Testing learners/q_learner.py...")
        from learners.q_learner import QLearner
        print("  [OK] QLearner imported successfully")
    except Exception as e:
        print(f"  [FAIL] Failed to import QLearner: {e}")
        return False

    try:
        print("Testing runners/episode_runner.py...")
        from runners.episode_runner import EpisodeRunner
        print("  [OK] EpisodeRunner imported successfully")
    except Exception as e:
        print(f"  [FAIL] Failed to import EpisodeRunner: {e}")
        return False

    try:
        print("Testing components/episode_buffer.py...")
        from components.episode_buffer import ReplayBuffer, EpisodeBatch
        print("  [OK] ReplayBuffer and EpisodeBatch imported successfully")
    except Exception as e:
        print(f"  [FAIL] Failed to import episode buffer: {e}")
        return False

    try:
        print("Testing utils/logging.py...")
        from utils.logging import Logger
        print("  [OK] Logger imported successfully")
    except Exception as e:
        print(f"  [FAIL] Failed to import Logger: {e}")
        return False

    try:
        print("Testing envs/sumo_grid_reroute.py...")
        from envs.sumo_grid_reroute import SUMOGridRerouteEnv
        print("  [OK] SUMOGridRerouteEnv imported successfully")
    except Exception as e:
        print(f"  [FAIL] Failed to import SUMOGridRerouteEnv: {e}")
        return False

    print("\n[OK] All imports successful!\n")
    return True


def test_smoke_training():
    """Run a minimal smoke test of the training loop."""
    print("=" * 60)
    print("TEST 2: Training Loop Smoke Test")
    print("=" * 60)

    try:
        import torch
        import yaml
        import numpy as np
        from controllers.basic_controller import BasicMAC
        from learners.q_learner import QLearner
        from runners.episode_runner import EpisodeRunner
        from components.episode_buffer import ReplayBuffer
        from utils.logging import Logger

        # Load configs
        script_dir = Path(__file__).parent / "pymarl" / "src"
        alg_config_path = script_dir / "config" / "algs" / "qmix_sumo.yaml"
        env_config_path = script_dir / "config" / "envs" / "sumo_grid4x4.yaml"

        print(f"\nLoading configs...")
        print(f"  Algorithm config: {alg_config_path}")
        print(f"  Environment config: {env_config_path}")

        with open(alg_config_path, 'r') as f:
            alg_config = yaml.safe_load(f)
        with open(env_config_path, 'r') as f:
            env_config = yaml.safe_load(f)

        # Merge configs
        args = {**alg_config, **env_config}

        # Override for quick test
        args["t_max"] = 500  # Very short test
        args["use_cuda"] = False  # Force CPU for testing
        args["use_gui"] = False
        args["test_interval"] = 200  # Test after 200 steps
        args["test_nepisode"] = 1  # Just 1 test episode
        args["log_interval"] = 100
        args["use_tensorboard"] = False  # Disable tensorboard for smoke test
        args["seed"] = 42
        args["buffer_size"] = 10  # Small buffer for smoke test
        args["episode_limit"] = 50  # Short episodes for smoke test
        args["max_episode_steps"] = 50  # Match episode limit

        print("\nTest configuration:")
        print(f"  Total timesteps: {args['t_max']}")
        print(f"  Batch size: {args['batch_size']}")
        print(f"  Buffer size: {args['buffer_size']}")
        print(f"  Episode limit: {args['episode_limit']}")
        print(f"  Using CUDA: {args['use_cuda']}")
        print(f"  Using GUI: {args['use_gui']}")

        # Set seeds
        np.random.seed(args["seed"])
        torch.manual_seed(args["seed"])

        print("\nInitializing components...")

        # Create logger
        logger = Logger(
            use_tensorboard=args["use_tensorboard"],
            log_dir="test_results/logs"
        )
        print("  [OK] Logger created")

        # Create runner
        print("  Creating runner (this will start SUMO)...")
        runner = EpisodeRunner(args, logger)
        env_info = runner.get_env_info()
        print(f"  [OK] Runner created")
        print(f"    - n_agents: {env_info['n_agents']}")
        print(f"    - n_actions: {env_info['n_actions']}")
        print(f"    - obs_shape: {env_info['obs_shape']}")
        print(f"    - state_shape: {env_info['state_shape']}")
        print(f"    - episode_limit: {env_info['episode_limit']}")

        # Update args with env info
        args.update(env_info)

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

        # Create MAC
        print("  Creating MAC...")
        mac = BasicMAC(scheme, groups, args)
        print("  [OK] MAC created")

        # Setup runner with MAC
        runner.setup(scheme, groups, preprocess, mac)
        print("  [OK] Runner setup complete")

        # Create replay buffer
        print("  Creating replay buffer...")
        buffer = ReplayBuffer(
            scheme, groups,
            buffer_size=args["buffer_size"],
            max_seq_length=args["episode_limit"] + 1,
            preprocess=preprocess,
            device="cpu"
        )
        print("  [OK] Buffer created")

        # Create learner
        print("  Creating learner...")
        learner = QLearner(mac, scheme, logger, args)
        print("  [OK] Learner created")

        print("\n" + "=" * 60)
        print("Starting smoke test training loop...")
        print("=" * 60)

        batch_size = args["batch_size"]
        episode_num = 0
        max_episodes = 3  # Just run a few episodes for smoke test

        print(f"\nRunning {max_episodes} episodes...")

        for ep in range(max_episodes):
            print(f"\n--- Episode {episode_num + 1} ---")

            # Run episode
            print("  Collecting episode...")
            episode_batch = runner.run(test_mode=False)

            # Get episode info
            ep_length = episode_batch["filled"].float().sum().item()
            ep_return = episode_batch["reward"][:, :-1].sum().item()

            print(f"  Episode length: {ep_length:.0f} steps")
            print(f"  Episode return: {ep_return:.2f}")
            print(f"  Total env steps: {runner.t_env}")

            # Store episode in buffer
            buffer.insert_episode_batch(episode_batch)
            print(f"  Episode stored in buffer (buffer size: {buffer.episodes_in_buffer})")

            episode_num += 1

            # Train if buffer has enough episodes
            if buffer.can_sample(batch_size):
                print(f"  Training (sampling {batch_size} episodes from buffer)...")

                # Sample batch from buffer
                train_batch = buffer.sample(batch_size)

                # Train on batch
                learner.train(train_batch, runner.t_env, episode_num)
                print(f"  Training step complete")
            else:
                print(f"  Buffer not ready for training yet ({buffer.episodes_in_buffer}/{batch_size})")

            # Stop if we reached t_max
            if runner.t_env >= args["t_max"]:
                print(f"\n  Reached t_max ({args['t_max']}), stopping...")
                break

        print("\n" + "=" * 60)
        print("Running test episode...")
        print("=" * 60)

        # Run one test episode
        print("\nCollecting test episode...")
        test_batch = runner.run(test_mode=True)
        test_return = test_batch["reward"][:, :-1].sum().item()
        test_length = test_batch["filled"].float().sum().item()

        print(f"Test episode return: {test_return:.2f}")
        print(f"Test episode length: {test_length:.0f} steps")

        # Cleanup
        print("\nCleaning up...")
        runner.close_env()
        logger.close()

        print("\n" + "=" * 60)
        print("[SUCCESS] Smoke test completed successfully!")
        print("=" * 60)
        print("\nSummary:")
        print(f"  - Ran {episode_num} training episodes")
        print(f"  - Total timesteps: {runner.t_env}")
        print(f"  - Buffer filled: {buffer.episodes_in_buffer}/{buffer.buffer_size}")
        print(f"  - Last test return: {test_return:.2f}")
        print("\nAll components working correctly!")

        return True

    except Exception as e:
        print(f"\n[FAIL] Smoke test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("PyMARL QMIX Training Loop Test - Step 5 Verification")
    print("=" * 60)
    print()

    # Test imports
    imports_ok = test_imports()

    if not imports_ok:
        print("\n✗ Import tests failed. Cannot proceed with training test.")
        return False

    # Test training loop
    print()
    training_ok = test_smoke_training()

    if training_ok:
        print("\n" + "=" * 60)
        print("*** SUCCESS *** ALL TESTS PASSED *** SUCCESS ***")
        print("=" * 60)
        print("\nStep 5 PyMARL QMIX implementation is verified and working!")
        print("\nYou can now:")
        print("  1. Run full training: cd pymarl/src && python main.py")
        print("  2. Monitor with TensorBoard: tensorboard --logdir results/logs")
        print("  3. Use custom settings: python main.py --t_max 1000000 --use_cuda")
        return True
    else:
        print("\n" + "=" * 60)
        print("*** FAILED *** TESTS FAILED *** FAILED ***")
        print("=" * 60)
        print("\nPlease check the error messages above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
