"""
Main Training Script for QMIX on SUMO Grid Environment

Trains QMIX agents to control traffic lights in a SUMO grid network.
"""

import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from controllers.basic_controller import BasicMAC
from learners.q_learner import QLearner
from runners.episode_runner import EpisodeRunner
from components.episode_buffer import ReplayBuffer
from utils.logging import Logger


def load_config(alg_config_path, env_config_path):
    """Load algorithm and environment configs."""
    with open(alg_config_path, 'r') as f:
        alg_config = yaml.safe_load(f)
    with open(env_config_path, 'r') as f:
        env_config = yaml.safe_load(f)

    # Merge configs
    config = {**alg_config, **env_config}
    return config


def get_scheme(env_info):
    """
    Create data scheme for episode buffer.

    Defines what data to store for each timestep.
    """
    return {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": torch.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": torch.uint8},
        "filled": {"vshape": (1,), "dtype": torch.uint8}  # Mask for valid timesteps
    }


def run_training(args):
    """
    Main training loop.

    Args:
        args: Configuration dictionary
    """
    # Set random seeds
    np.random.seed(args.get("seed", 42))
    torch.manual_seed(args.get("seed", 42))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.get("seed", 42))

    # Create logger
    logger = Logger(
        use_tensorboard=args.get("use_tensorboard", True),
        log_dir=args.get("log_dir", "results/logs")
    )

    # Create runner
    runner = EpisodeRunner(args, logger)
    env_info = runner.get_env_info()

    # Update args with env info
    args.update(env_info)

    # Create data scheme
    scheme = get_scheme(env_info)
    groups = {"agents": args["n_agents"]}
    preprocess = {}

    # Create MAC (Multi-Agent Controller)
    mac = BasicMAC(scheme, groups, args)

    # Setup runner with MAC
    runner.setup(scheme, groups, preprocess, mac)

    # Create replay buffer
    buffer = ReplayBuffer(
        scheme, groups,
        buffer_size=args.get("buffer_size", 5000),
        max_seq_length=args["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu"  # Keep buffer on CPU to save GPU memory
    )

    # Create learner
    learner = QLearner(mac, scheme, logger, args)

    # Move to GPU if available
    if args.get("use_cuda", False) and torch.cuda.is_available():
        learner.cuda()
        print("Using CUDA")
    else:
        print("Using CPU")

    # Training parameters
    t_max = args.get("t_max", 2000000)
    batch_size = args.get("batch_size", 32)
    test_interval = args.get("test_interval", 10000)
    test_nepisode = args.get("test_nepisode", 5)
    save_interval = args.get("save_model_interval", 100000)
    log_interval = args.get("log_interval", 5000)

    # Validation parameters (Step 6)
    use_validation = args.get("use_validation", True)
    validation_interval = args.get("validation_interval", 50000)
    validation_nepisode = args.get("validation_nepisode", 10)

    # Create results directory
    save_path = args.get("checkpoint_path", "results/models")
    os.makedirs(save_path, exist_ok=True)

    episode_num = 0
    last_test_t = -test_interval
    last_log_t = 0
    last_save_t = 0
    last_validation_t = -validation_interval

    # Best model tracking (Step 6)
    best_validation_return = float('-inf')  # Track best validation performance
    best_model_t = 0

    print("\n=== Starting Training ===")
    print(f"Environment: SUMO Grid {args.get('grid_size', '4x4')}")
    print(f"Agents: {args['n_agents']}")
    print(f"Actions per agent: {args['n_actions']}")
    print(f"State shape: {args['state_shape']}")
    print(f"Obs shape: {args['obs_shape']}")
    print(f"Episode limit: {args['episode_limit']}")
    print(f"Total timesteps: {t_max}")
    print(f"Batch size: {batch_size}")
    print(f"Buffer size: {args.get('buffer_size', 5000)}")
    print("========================\n")

    recent_train_returns = []  # accumulates between log intervals for smoothed mean

    try:
        while runner.t_env < t_max:
            # Run episode
            episode_batch, ep_metrics = runner.run(test_mode=False)

            # Store episode in buffer
            buffer.insert_episode_batch(episode_batch)

            episode_num += 1
            recent_train_returns.append(episode_batch["reward"][:, :-1].sum().item())

            # Train if buffer has enough episodes
            if buffer.can_sample(batch_size):
                # Sample batch from buffer
                train_batch = buffer.sample(batch_size)

                # Move batch to GPU if needed
                if args.get("use_cuda", False) and torch.cuda.is_available():
                    train_batch.to("cuda")

                # Train on batch
                learner.train(train_batch, runner.t_env, episode_num)

            # Testing
            if runner.t_env - last_test_t >= test_interval:
                print(f"\n[Testing at t={runner.t_env}]")
                test_returns = []
                test_metrics_list = []
                for _ in range(test_nepisode):
                    test_batch, test_ep_metrics = runner.run(test_mode=True)
                    test_return = test_batch["reward"][:, :-1].sum().item()
                    test_returns.append(test_return)
                    if test_ep_metrics:
                        test_metrics_list.append(test_ep_metrics)

                avg_test_return = np.mean(test_returns)
                print(f"Test results: avg_return={avg_test_return:.2f}, "
                      f"std={np.std(test_returns):.2f}")
                logger.log_stat("test_return_mean", avg_test_return, runner.t_env)

                # Log aggregate test metrics (Step 6)
                if test_metrics_list:
                    for key in ("mean_travel_time", "mean_waiting_time", "total_stops",
                                "total_emissions", "arrival_rate", "controlled_mean_travel_time"):
                        vals = [m[key] for m in test_metrics_list if key in m]
                        if vals:
                            logger.log_stat(f"test_{key}_avg", float(np.mean(vals)), runner.t_env)
                    print(f"  mean_travel_time={np.mean([m.get('mean_travel_time',0) for m in test_metrics_list]):.1f}s, "
                          f"arrival_rate={np.mean([m.get('arrival_rate',0) for m in test_metrics_list]):.3f}")

                last_test_t = runner.t_env

            # Validation for best model selection (Step 6)
            if use_validation and runner.t_env - last_validation_t >= validation_interval:
                print(f"\n[Validation at t={runner.t_env}]")
                validation_returns = []
                validation_travel_times = []
                for _ in range(validation_nepisode):
                    val_batch, val_ep_metrics = runner.run(test_mode=True)
                    val_return = val_batch["reward"][:, :-1].sum().item()
                    validation_returns.append(val_return)
                    if "mean_travel_time" in val_ep_metrics:
                        validation_travel_times.append(val_ep_metrics["mean_travel_time"])

                avg_validation_return = np.mean(validation_returns)
                print(f"Validation results: avg_return={avg_validation_return:.2f}, "
                      f"std={np.std(validation_returns):.2f}")
                logger.log_stat("validation_return_mean", avg_validation_return, runner.t_env)

                # Use travel time as primary validation metric (lower is better -> negate)
                if validation_travel_times:
                    avg_travel_time = np.mean(validation_travel_times)
                    val_metric = -avg_travel_time  # negate so higher is better
                    logger.log_stat("validation_mean_travel_time", avg_travel_time, runner.t_env)
                    print(f"  mean_travel_time={avg_travel_time:.1f}s")
                else:
                    val_metric = avg_validation_return

                # Save best model
                if val_metric > best_validation_return:
                    best_validation_return = val_metric
                    best_model_t = runner.t_env
                    best_save_dir = os.path.join(save_path, "best")
                    os.makedirs(best_save_dir, exist_ok=True)
                    learner.save_models(best_save_dir)
                    label = f"travel_time={-val_metric:.1f}s" if validation_travel_times else f"return={avg_validation_return:.2f}"
                    print(f"  [NEW BEST] Model saved ({label})")
                    logger.log_stat("best_validation_metric", best_validation_return, runner.t_env)
                else:
                    print(f"  Current best metric: {best_validation_return:.4f} at t={best_model_t}")

                last_validation_t = runner.t_env

            # Logging
            if runner.t_env - last_log_t >= log_interval:
                logger.print_recent_stats()

                # Epsilon (replicated from basic_controller._get_epsilon)
                eps_start = args.get("epsilon_start", 1.0)
                eps_finish = args.get("epsilon_finish", 0.05)
                eps_anneal = args.get("epsilon_anneal_time", 50000)
                epsilon = eps_start - (eps_start - eps_finish) * min(1.0, runner.t_env / eps_anneal)
                logger.log_stat("epsilon", epsilon, runner.t_env)

                # Buffer fill ratio (0→1 as replay warms up)
                logger.log_stat("buffer_fill", buffer.episodes_in_buffer / buffer.buffer_size, runner.t_env)

                # Smoothed training return over the last log_interval window
                if recent_train_returns:
                    logger.log_stat("train_return_mean", float(np.mean(recent_train_returns)), runner.t_env)
                    recent_train_returns.clear()

                last_log_t = runner.t_env

            # Save model
            if runner.t_env - last_save_t >= save_interval:
                save_dir = os.path.join(save_path, f"step_{runner.t_env}")
                os.makedirs(save_dir, exist_ok=True)
                learner.save_models(save_dir)
                print(f"Model saved at t={runner.t_env}")
                last_save_t = runner.t_env

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")

    finally:
        # Save final model
        save_dir = os.path.join(save_path, "final")
        os.makedirs(save_dir, exist_ok=True)
        learner.save_models(save_dir)
        print(f"\nFinal model saved to {save_dir}")

        # Print best model info (Step 6)
        if use_validation and best_model_t > 0:
            print(f"Best model: validation_return={best_validation_return:.2f} at t={best_model_t}")
            print(f"Best model saved to {os.path.join(save_path, 'best')}")

        # Close environment
        runner.close_env()
        logger.close()

        print(f"\nTraining complete! Total episodes: {episode_num}, Total timesteps: {runner.t_env}")


def main():
    """Main entry point."""
    # Get config paths
    script_dir = Path(__file__).parent
    alg_config_path = script_dir / "config" / "algs" / "qmix_sumo.yaml"
    env_config_path = script_dir / "config" / "envs" / "sumo_grid4x4.yaml"

    # Load config
    if not alg_config_path.exists():
        print(f"Error: Algorithm config not found at {alg_config_path}")
        return

    if not env_config_path.exists():
        print(f"Error: Environment config not found at {env_config_path}")
        return

    args = load_config(str(alg_config_path), str(env_config_path))

    # Override with command line args if needed
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--use_gui", action="store_true")
    parser.add_argument("--t_max", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--buffer_size", type=int, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Directory to save model checkpoints")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Directory for TensorBoard logs")
    # Step 6 validation flags
    parser.add_argument("--no_validation", action="store_true",
                        help="Disable validation runs during training")
    parser.add_argument("--validation_interval", type=int, default=None,
                        help="Steps between validation runs")
    parser.add_argument("--validation_nepisode", type=int, default=None,
                        help="Episodes per validation run")
    cmd_args = parser.parse_args()

    # Update args with command line overrides
    if cmd_args.seed is not None:
        args["seed"] = cmd_args.seed
    if cmd_args.use_cuda:
        args["use_cuda"] = True
    if cmd_args.use_gui:
        args["use_gui"] = True
    if cmd_args.t_max is not None:
        args["t_max"] = cmd_args.t_max
    if cmd_args.batch_size is not None:
        args["batch_size"] = cmd_args.batch_size
    if cmd_args.buffer_size is not None:
        args["buffer_size"] = cmd_args.buffer_size
    if cmd_args.checkpoint_path is not None:
        args["checkpoint_path"] = cmd_args.checkpoint_path
    if cmd_args.log_dir is not None:
        args["log_dir"] = cmd_args.log_dir
    if cmd_args.no_validation:
        args["use_validation"] = False
    if cmd_args.validation_interval is not None:
        args["validation_interval"] = cmd_args.validation_interval
    if cmd_args.validation_nepisode is not None:
        args["validation_nepisode"] = cmd_args.validation_nepisode

    # Run training
    run_training(args)


if __name__ == "__main__":
    main()
