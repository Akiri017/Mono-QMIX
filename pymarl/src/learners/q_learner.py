"""
QMIX Learner

Implements Q-learning with QMIX value function factorization.
Handles training loop, target networks, gradient computation, and optimization.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from modules.mixers.qmix import QMixer
from copy import deepcopy


class QLearner:
    """
    QMIX learner with target networks and TD(lambda) updates.

    Args:
        mac: Multi-Agent Controller
        scheme: Data scheme
        logger: Logger instance
        args: Configuration dict
    """

    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.n_agents = args["n_agents"]
        self.n_actions = args["n_actions"]

        # Get device
        self.device = torch.device("cuda" if args.get("use_cuda", False) and torch.cuda.is_available() else "cpu")

        # Get state shape
        self.state_shape = scheme["state"]["vshape"]

        # Learning parameters
        self.gamma = args.get("gamma", 0.99)
        self.td_lambda = args.get("td_lambda", 0.8)
        self.double_q = args.get("double_q", True)
        self.grad_norm_clip = args.get("grad_norm_clip", 10)

        # Running reward normalisation — EMA over per-batch mean and variance.
        # Keeps TD targets in a stable range regardless of how vehicle density
        # grows as the policy improves, replacing the fixed reward_scale scalar.
        self.ema_decay = args.get("reward_ema_decay", 0.99)
        self.reward_running_mean = 0.0
        self.reward_running_var = 1.0  # initialised to 1 to avoid div-by-zero on first batch

        # PopArt stats for Q_tot-level target normalisation.
        # Reward-level EMA normalisation cannot bound the bootstrap term
        # (γ * Q_tot(s')), which grows unboundedly through the TD loop even
        # when rewards are stable. PopArt tracks the scale of the full TD
        # target (r_norm + γ * Q_tot') and normalises at that level, keeping
        # the mixer output in a bounded range throughout training.
        self.popart_mean = 0.0
        self.popart_std = 1.0  # initialised to 1 to start as identity transform
        self.popart_ema_decay = args.get("popart_ema_decay", 0.99)

        # Target network update
        self.target_update_interval = args.get("target_update_interval", 200)
        self.target_update_mode = args.get("target_update_mode", "hard")
        self.tau = args.get("tau", 0.001)  # For soft updates

        # Create mixer network
        mixer_args = {
            "n_agents": self.n_agents,
            "state_shape": self.state_shape,
            "mixing_embed_dim": args.get("mixing_embed_dim", 32),
            "hypernet_layers": args.get("hypernet_layers", 2),
            "hypernet_embed": args.get("hypernet_embed", 64)
        }
        self.mixer = QMixer(mixer_args)
        self.mixer.to(self.device)

        # Create target networks
        self.target_mac = deepcopy(mac)
        self.target_mixer = deepcopy(self.mixer)

        # Optimizer
        params = list(self.mac.parameters()) + list(self.mixer.parameters())
        self.optimizer = optim.Adam(params, lr=args.get("lr", 0.0005))

        # Training stats
        self.last_target_update_episode = 0

        self.log_stats_t = -1

    def _update_popart(self, targets_raw, mask):
        """Update PopArt running mean/std from valid (unmasked) TD targets.

        Args:
            targets_raw: (batch, T, 1) — unnormalised TD targets
            mask:        (batch, T, 1) — 1.0 for valid timesteps
        """
        valid_targets = targets_raw[mask.bool()]
        if valid_targets.numel() < 2:
            return
        batch_mean = valid_targets.mean().item()
        batch_std = valid_targets.std().item()
        self.popart_mean = (
            self.popart_ema_decay * self.popart_mean
            + (1 - self.popart_ema_decay) * batch_mean
        )
        # Floor std at 1e-8 so the normalisation denominator never collapses
        self.popart_std = (
            self.popart_ema_decay * self.popart_std
            + (1 - self.popart_ema_decay) * max(batch_std, 1e-8)
        )

    def _rescale_mixer_output(self, mu_old, sigma_old, mu_new, sigma_new):
        """Rescale mixer output_scale and output_shift after PopArt stat update.

        Applies the standard PopArt weight-preservation formula to both the
        online and target mixers so their outputs stay consistent at the new
        normalisation scale without a disruptive jump:

            scale_new = scale_old * (sigma_old / sigma_new)
            shift_new = (shift_old * sigma_old + mu_old - mu_new) / sigma_new

        Args:
            mu_old:    PopArt mean before the update (float)
            sigma_old: PopArt std before the update (float)
            mu_new:    PopArt mean after the update (float)
            sigma_new: PopArt std after the update (float)
        """
        if sigma_new < 1e-8:
            return
        ratio = sigma_old / sigma_new
        for mixer in (self.mixer, self.target_mixer):
            old_scale = mixer.output_scale.data.item()
            old_shift = mixer.output_shift.data.item()
            mixer.output_scale.data.fill_(old_scale * ratio)
            mixer.output_shift.data.fill_(
                (old_shift * sigma_old + mu_old - mu_new) / sigma_new
            )

    def train(self, batch, t_env, episode_num):
        """
        Train on a batch of episodes.

        Args:
            batch: EpisodeBatch with transitions
            t_env: Total environment steps
            episode_num: Current episode number

        Returns:
            stats: Dictionary of training statistics
        """
        # Get batch data
        rewards = batch["reward"][:, :-1]  # (batch, T, 1)
        actions = batch["actions"][:, :-1]  # (batch, T, n_agents, 1)
        terminated = batch["terminated"][:, :-1].float()  # (batch, T, 1)
        mask = batch["filled"][:, :-1].float()  # (batch, T, 1) - valid timesteps
        avail_actions = batch["avail_actions"]  # (batch, T+1, n_agents, n_actions)
        states = batch["state"]  # (batch, T+1, state_dim)

        batch_size = batch.batch_size
        max_t = rewards.shape[1]

        # Move to device — mask first so it's available for EMA indexing below
        rewards = rewards.to(self.device)
        mask = mask.to(self.device)
        actions = actions.to(self.device)
        terminated = terminated.to(self.device)
        states = states.to(self.device)
        avail_actions = avail_actions.to(self.device)

        # Update running mean/var from valid timesteps only, then normalize.
        # Using EMA so recent experience is weighted more heavily than early
        # noisy episodes, and the scale adapts as vehicle density grows.
        valid_rewards = rewards[mask.bool()]
        batch_mean = valid_rewards.mean().item()
        batch_var = valid_rewards.var().item() if valid_rewards.numel() > 1 else 1.0
        self.reward_running_mean = self.ema_decay * self.reward_running_mean + (1 - self.ema_decay) * batch_mean
        self.reward_running_var = self.ema_decay * self.reward_running_var + (1 - self.ema_decay) * batch_var
        rewards = (rewards - self.reward_running_mean) / (self.reward_running_var ** 0.5 + 1e-8)

        # Calculate Q-values
        mac_out = []
        self.mac.init_hidden(batch_size)
        for t in range(max_t + 1):
            agent_qs = self.mac.forward(batch, t)  # (batch, n_agents, n_actions)
            mac_out.append(agent_qs)
        mac_out = torch.stack(mac_out, dim=1)  # (batch, T+1, n_agents, n_actions)

        # Pick Q-values for chosen actions
        chosen_action_qvals = torch.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # (batch, T, n_agents)

        # Calculate target Q-values
        target_mac_out = []
        self.target_mac.init_hidden(batch_size)
        for t in range(max_t + 1):
            target_agent_qs = self.target_mac.forward(batch, t)
            target_mac_out.append(target_agent_qs)
        target_mac_out = torch.stack(target_mac_out, dim=1)  # (batch, T+1, n_agents, n_actions)

        # Mask out unavailable actions for target network
        target_mac_out[avail_actions == 0] = -1e10

        # Calculate target Q-values (with double Q-learning)
        if self.double_q:
            # Double Q-learning: use current network to select actions, target network to evaluate
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -1e10
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = torch.gather(target_mac_out[:, 1:], dim=3, index=cur_max_actions).squeeze(3)
        else:
            # Standard Q-learning: use target network for both selection and evaluation
            target_max_qvals = target_mac_out[:, 1:].max(dim=3)[0]

        # Mix agent Q-values
        chosen_action_qvals = chosen_action_qvals.view(batch_size * max_t, self.n_agents)
        states_reshaped = states[:, :-1].reshape(batch_size * max_t, -1)
        q_tot = self.mixer(chosen_action_qvals, states_reshaped)
        q_tot = q_tot.view(batch_size, max_t, 1)

        # Mix target agent Q-values
        target_max_qvals = target_max_qvals.view(batch_size * max_t, self.n_agents)
        target_states_reshaped = states[:, 1:].reshape(batch_size * max_t, -1)
        target_q_tot = self.target_mixer(target_max_qvals, target_states_reshaped)
        target_q_tot = target_q_tot.view(batch_size, max_t, 1)

        # De-normalise target Q_tot from PopArt space back to raw scale before
        # constructing the TD target. Without this, the bootstrap term
        # (γ * Q_tot') accumulates unboundedly — reward normalisation only
        # bounds the r term and has no effect on the dominant bootstrap term.
        target_q_tot_denorm = target_q_tot * self.popart_std + self.popart_mean

        # Raw TD target — rewards are already EMA-normalised; Q_tot is denormed
        targets_raw = rewards + self.gamma * (1 - terminated) * target_q_tot_denorm

        # Snapshot stats before update so the rescaling delta is computed correctly
        popart_mean_old = self.popart_mean
        popart_std_old = self.popart_std

        # Update PopArt stats from the raw target distribution
        # mask is still (batch, T, 1) here — same shape as targets_raw
        self._update_popart(targets_raw, mask)

        # Rescale mixer output_scale and output_shift to match the new stats.
        # This is the "Preserving Outputs Precisely" step — without it the
        # de-normalise cycle amplifies the bootstrap term exponentially.
        self._rescale_mixer_output(
            mu_old=popart_mean_old,
            sigma_old=popart_std_old,
            mu_new=self.popart_mean,
            sigma_new=self.popart_std,
        )

        targets = (targets_raw - self.popart_mean) / (self.popart_std + 1e-8)

        # TD error in PopArt normalised space
        td_error = q_tot - targets.detach()

        # Mask out invalid timesteps
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask

        # Loss (MSE)
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.mac.parameters()) + list(self.mixer.parameters()),
            self.grad_norm_clip
        )
        self.optimizer.step()

        # Update target networks every N episodes
        if episode_num - self.last_target_update_episode >= self.target_update_interval:
            self._update_targets()
            self.last_target_update_episode = episode_num

        # Logging
        if t_env - self.log_stats_t >= self.args.get("log_interval", 5000):
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            # chosen_action_qvals was reshaped to (batch*T, n_agents) for the
            # mixer; reshape back before multiplying with mask (batch, T, 1).
            q_log = chosen_action_qvals.view(batch_size, max_t, self.n_agents)
            valid_q = q_log[mask.expand_as(q_log).bool()]
            self.logger.log_stat("q_taken_mean", valid_q.mean().item(), t_env)
            self.logger.log_stat("q_taken_std", valid_q.std().item(), t_env)
            # target_mean logged in normalised space — should stay near 0 if PopArt is working
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("reward_running_mean", self.reward_running_mean, t_env)
            self.logger.log_stat("reward_running_std", self.reward_running_var ** 0.5, t_env)
            # PopArt stats — popart_std growing is expected and healthy;
            # it means the mixer is operating at a larger scale but Q_tot
            # stays bounded in normalised space
            self.logger.log_stat("popart_mean", self.popart_mean, t_env)
            self.logger.log_stat("popart_std", self.popart_std, t_env)
            self.log_stats_t = t_env

        return {
            "loss": loss.item(),
            "grad_norm": grad_norm.item(),
            "q_mean": q_tot.mean().item(),
            "target_mean": targets.mean().item()
        }

    def _update_targets(self):
        """Update target networks."""
        if self.target_update_mode == "hard":
            # Hard update: copy parameters
            self.target_mac.load_state(self.mac)
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        elif self.target_update_mode == "soft":
            # Soft update: exponential moving average
            for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def cuda(self):
        """Move networks to GPU."""
        self.mac.cuda()
        self.target_mac.cuda()
        self.mixer.cuda()
        self.target_mixer.cuda()
        self.device = torch.device("cuda")

    def cpu(self):
        """Move networks to CPU."""
        self.mac.cpu()
        self.target_mac.cpu()
        self.mixer.cpu()
        self.target_mixer.cpu()
        self.device = torch.device("cpu")

    def save_models(self, path):
        """Save model parameters and optimizer state."""
        self.mac.save_models(path)
        torch.save(self.mixer.state_dict(), f"{path}/mixer.pth")
        # Optimizer state preserves Adam momentum/variance across resume
        torch.save(self.optimizer.state_dict(), f"{path}/optimizer.pth")
        # Reward normalisation state — must be restored on resume so the scale
        # doesn't reset mid-training and destabilise the TD targets
        torch.save(
            {
                "running_mean": self.reward_running_mean,
                "running_var": self.reward_running_var,
                "popart_mean": self.popart_mean,
                "popart_std": self.popart_std,
            },
            f"{path}/reward_stats.pth"
        )

    def load_models(self, path):
        """Load model parameters and optimizer state."""
        self.mac.load_models(path)
        self.mixer.load_state_dict(torch.load(f"{path}/mixer.pth", map_location=self.device))
        self._update_targets()
        opt_path = f"{path}/optimizer.pth"
        if os.path.exists(opt_path):
            self.optimizer.load_state_dict(torch.load(opt_path, map_location=self.device))
        stats_path = f"{path}/reward_stats.pth"
        if os.path.exists(stats_path):
            stats = torch.load(stats_path, map_location="cpu")
            self.reward_running_mean = stats["running_mean"]
            self.reward_running_var = stats["running_var"]
            # .get with defaults so checkpoints saved before PopArt load cleanly
            self.popart_mean = stats.get("popart_mean", 0.0)
            self.popart_std = stats.get("popart_std", 1.0)
