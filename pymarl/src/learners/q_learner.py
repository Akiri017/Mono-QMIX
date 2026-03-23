"""
QMIX Learner

Implements Q-learning with QMIX value function factorization.
Handles training loop, target networks, gradient computation, and optimization.
"""

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
        self.training_steps = 0
        self.last_target_update = 0

        self.log_stats_t = -1

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

        # Move to device
        rewards = rewards.to(self.device)
        actions = actions.to(self.device)
        terminated = terminated.to(self.device)
        mask = mask.to(self.device)
        states = states.to(self.device)
        avail_actions = avail_actions.to(self.device)

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

        # Calculate TD targets
        targets = rewards + self.gamma * (1 - terminated) * target_q_tot

        # TD error
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

        # Update target networks
        self.training_steps += 1
        if self.training_steps - self.last_target_update >= self.target_update_interval:
            self._update_targets()
            self.last_target_update = self.training_steps

        # Logging
        if t_env - self.log_stats_t >= self.args.get("log_interval", 5000):
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / mask.sum().item(), t_env)
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
        """Save model parameters."""
        self.mac.save_models(path)
        torch.save(self.mixer.state_dict(), f"{path}/mixer.pth")

    def load_models(self, path):
        """Load model parameters."""
        self.mac.load_models(path)
        self.mixer.load_state_dict(torch.load(f"{path}/mixer.pth", map_location=self.device))
        self._update_targets()
