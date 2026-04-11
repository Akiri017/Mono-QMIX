"""
Episode Runner

Collects episodes from the environment for training.
Handles interaction between MAC and environment.
"""

import torch
import numpy as np
from components.episode_buffer import EpisodeBatch


class EpisodeRunner:
    """
    Runs episodes in the environment for training/testing.

    Args:
        args: Configuration dict
        logger: Logger instance
    """

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = args.get("batch_size_run", 1)  # Parallel environments (1 for sequential)
        self.episode_limit = args.get("episode_limit", 1000)

        # Create environment
        self.env = self._make_env()

        # Get environment info
        env_info = self.env.get_env_info()
        self.n_agents = env_info["n_agents"]
        self.n_actions = env_info["n_actions"]
        self.state_shape = env_info["state_shape"]
        self.obs_shape = env_info["obs_shape"]

        # Update args with env info
        args.update(env_info)

        # Episode tracking
        self.t = 0
        self.t_env = 0

        # Device
        self.device = torch.device("cuda" if args.get("use_cuda", False) and torch.cuda.is_available() else "cpu")

    def _make_env(self):
        """Create the environment."""
        from envs import ENV_REGISTRY

        # Use env_args from config if available, otherwise use defaults
        if "env_args" in self.args:
            # Copy so we don't mutate the original config
            env_args = dict(self.args["env_args"])
            # Forward Civiq-specific keys that live in the alg config but are
            # needed by the environment (e.g. rsu_config for zone_manager)
            for key in ("rsu_config", "max_rsus", "max_agents_per_rsu", "obs_dim"):
                if key in self.args and key not in env_args:
                    env_args[key] = self.args[key]
        else:
            # Fallback: pass entire args dict to environment
            env_args = self.args

        return ENV_REGISTRY["sumo_grid_reroute"](env_args)

    def setup(self, scheme, groups, preprocess, mac):
        """
        Setup runner with data scheme and MAC.

        Args:
            scheme: Data scheme for episode buffer
            groups: Agent groups
            preprocess: Preprocessing functions
            mac: Multi-Agent Controller
        """
        self.new_batch = self._get_batch_factory(scheme, groups, preprocess)
        self.mac = mac

    def _get_batch_factory(self, scheme, groups, preprocess):
        """Get batch factory for creating episode batches."""
        def batch_factory(batch_size):
            return EpisodeBatch(scheme, groups, batch_size, self.episode_limit + 1,
                              preprocess=preprocess, device=self.device)
        return batch_factory

    def get_env_info(self):
        """Get environment configuration."""
        return {
            "n_agents": self.n_agents,
            "n_actions": self.n_actions,
            "state_shape": self.state_shape,
            "obs_shape": self.obs_shape,
            "episode_limit": self.episode_limit
        }

    def reset(self):
        """Reset the environment."""
        self.batch = self.new_batch(self.batch_size)
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        """
        Run one episode.

        Args:
            test_mode: If True, use greedy action selection (no exploration)

        Returns:
            batch: EpisodeBatch with collected transitions
            episode_metrics: Dict of episode-level metrics from the environment (or {})
        """
        self.reset()

        # Initialize hidden states
        self.mac.init_hidden(batch_size=self.batch_size)

        terminated = False
        episode_return = 0
        episode_metrics = {}

        # Get initial observations and state
        obs = self.env.get_obs()
        state = self.env.get_state()
        avail_actions = self.env.get_avail_actions()

        # Store initial data
        pre_transition_data = {
            "state": [state],
            "avail_actions": [avail_actions],
            "obs": [obs]
        }

        # Civiq: collect zone data at ts=0 (zone_manager present when using civiq config)
        _civiq = hasattr(self.env, "zone_manager") and self.env.zone_manager is not None
        if _civiq:
            _zone = self.env.get_zone_assignments()
            pre_transition_data["local_states"] = [self.env.get_local_obs_padded(_zone)]
            pre_transition_data["agent_masks_per_rsu"] = [self.env.get_agent_masks_padded(_zone)]
            pre_transition_data["zone_assignments"] = [self.env.get_zone_assignments_flat(_zone)]
            # rsu_agent_qs is NOT populated here — computed in HierarchicalQLearner.train()
            # from chosen_action_qvals sliced per zone_assignments

        self.batch.update(pre_transition_data, ts=0)

        # Run episode
        while not terminated:
            # Create episode batch slice for current timestep
            # Select actions
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            actions_list = actions.cpu().numpy()[0]  # (n_agents, 1)

            # Execute actions in environment
            reward, terminated, env_info = self.env.step(actions_list.squeeze(1))
            episode_return += reward

            # Capture episode metrics when episode terminates
            if terminated and "episode_metrics" in env_info:
                episode_metrics = env_info["episode_metrics"]

            # Get next observations and state
            obs = self.env.get_obs()
            state = self.env.get_state()
            avail_actions = self.env.get_avail_actions()

            # Store transition data
            post_transition_data = {
                "actions": actions.cpu(),
                "reward": [(reward,)],
                "terminated": [(terminated,)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            # Store next timestep data
            pre_transition_data = {
                "state": [state],
                "avail_actions": [avail_actions],
                "obs": [obs]
            }

            # Civiq: collect zone data at ts=t+1
            if _civiq:
                _zone = self.env.get_zone_assignments()
                pre_transition_data["local_states"] = [self.env.get_local_obs_padded(_zone)]
                pre_transition_data["agent_masks_per_rsu"] = [self.env.get_agent_masks_padded(_zone)]
                pre_transition_data["zone_assignments"] = [self.env.get_zone_assignments_flat(_zone)]

            self.batch.update(pre_transition_data, ts=self.t + 1)

            self.t += 1

            # Check episode limit
            if self.t >= self.episode_limit:
                terminated = True

        # Mark filled timesteps
        self.batch.data.transition_data["filled"][:, :self.t] = 1

        # Last timestep (terminal)
        last_data = {
            "state": [state],
            "avail_actions": [avail_actions],
            "obs": [obs]
        }
        self.batch.update(last_data, ts=self.t)

        # Update environment timesteps
        if not test_mode:
            self.t_env += self.t

        # Log episode return and metrics
        if test_mode:
            self.logger.log_stat("test_return", episode_return, self.t_env)
        else:
            self.logger.log_stat("episode_return", episode_return, self.t_env)

        # Log per-episode metrics (Step 6)
        prefix = "test_" if test_mode else "train_"
        for key in ("mean_travel_time", "mean_waiting_time", "total_stops",
                    "total_emissions", "arrival_rate", "controlled_mean_travel_time"):
            if key in episode_metrics:
                self.logger.log_stat(prefix + key, episode_metrics[key], self.t_env)

        return self.batch, episode_metrics

    def close_env(self):
        """Close the environment."""
        self.env.close()
