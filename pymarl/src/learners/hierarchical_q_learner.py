"""
Hierarchical Q-Learner (Civiq)

Wires LocalQMixer (per-RSU, Level 2) and GlobalQMixer (across RSUs, Level 3)
together with a single end-to-end TD loss. Fork of QLearner — all non-mixer
logic is identical.

train() mixing section raises NotImplementedError until Phase 4 batch fields
(rsu_agent_qs, agent_masks_per_rsu, local_states, zone_assignments) are
populated by the episode runner.
"""

import os
import copy
import yaml
import torch
import torch.optim as optim

from modules.mixers.local_qmixer import LocalQMixer
from modules.mixers.global_qmixer import GlobalQMixer
from components.rsu_zone_manager import RSUZoneManager


class HierarchicalQLearner:
    """
    Civiq hierarchical learner with two mixing levels.

    Level 2: LocalQMixer — per-RSU vehicle Q-values → local_Q_tot
    Level 3: GlobalQMixer — RSU local_Q_tots → global_Q_tot

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
        self.device = torch.device(
            "cuda" if args.get("use_cuda", False) and torch.cuda.is_available() else "cpu"
        )

        # Learning parameters — identical to QLearner
        self.gamma = args.get("gamma", 0.99)
        self.td_lambda = args.get("td_lambda", 0.8)
        self.double_q = args.get("double_q", True)
        self.grad_norm_clip = args.get("grad_norm_clip", 10)

        # Target network update — identical to QLearner
        self.target_update_interval = args.get("target_update_interval", 200)
        self.target_update_mode = args.get("target_update_mode", "hard")
        self.tau = args.get("tau", 0.001)

        # Level 2: LocalQMixer (per-RSU vehicle mixing)
        self.local_mixer = LocalQMixer(args)
        self.local_mixer.to(self.device)
        self.target_local_mixer = copy.deepcopy(self.local_mixer)

        # Level 3: GlobalQMixer (cross-RSU mixing)
        self.global_mixer = GlobalQMixer(args)
        self.global_mixer.to(self.device)
        self.target_global_mixer = copy.deepcopy(self.global_mixer)

        # Target MAC — identical to QLearner
        self.target_mac = copy.deepcopy(mac)

        # RSU zone manager — loaded from rsu_config yaml path
        rsu_config_path = args["rsu_config"]
        with open(rsu_config_path) as f:
            rsu_config_dict = yaml.safe_load(f)
        self.zone_manager = RSUZoneManager(rsu_config_dict)

        # Single optimizer over MAC + both mixers — identical Adam settings to QLearner
        self.params = list(self.mac.parameters())
        self.params += list(self.local_mixer.parameters())
        self.params += list(self.global_mixer.parameters())
        self.optimizer = optim.Adam(self.params, lr=args.get("lr", 0.0005),
                                    eps=args.get("optim_eps", 1e-5))

        # Training stats
        self.last_target_update_episode = 0
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
        # Get batch data — identical to QLearner
        rewards = batch["reward"][:, :-1]          # (batch, T, 1)
        actions = batch["actions"][:, :-1]          # (batch, T, n_agents, 1)
        terminated = batch["terminated"][:, :-1].float()  # (batch, T, 1)
        mask = batch["filled"][:, :-1].float()      # (batch, T, 1)
        avail_actions = batch["avail_actions"]       # (batch, T+1, n_agents, n_actions)

        batch_size = batch.batch_size
        max_t = rewards.shape[1]

        # Move to device
        rewards = rewards.to(self.device)
        actions = actions.to(self.device)
        terminated = terminated.to(self.device)
        mask = mask.to(self.device)
        avail_actions = avail_actions.to(self.device)

        # MAC forward pass — identical to QLearner
        mac_out = []
        self.mac.init_hidden(batch_size)
        for t in range(max_t + 1):
            agent_qs = self.mac.forward(batch, t)   # (batch, n_agents, n_actions)
            mac_out.append(agent_qs)
        mac_out = torch.stack(mac_out, dim=1)        # (batch, T+1, n_agents, n_actions)

        # Chosen action Q-values — identical to QLearner
        chosen_action_qvals = torch.gather(
            mac_out[:, :-1], dim=3, index=actions
        ).squeeze(3)                                 # (batch, T, n_agents)

        # Target MAC forward pass — identical to QLearner
        target_mac_out = []
        self.target_mac.init_hidden(batch_size)
        for t in range(max_t + 1):
            target_agent_qs = self.target_mac.forward(batch, t)
            target_mac_out.append(target_agent_qs)
        target_mac_out = torch.stack(target_mac_out, dim=1)  # (batch, T+1, n_agents, n_actions)

        target_mac_out[avail_actions == 0] = -1e10

        # Double Q-learning — identical to QLearner
        if self.double_q:
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -1e10
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = torch.gather(
                target_mac_out[:, 1:], dim=3, index=cur_max_actions
            ).squeeze(3)                             # (batch, T, n_agents)
        else:
            target_max_qvals = target_mac_out[:, 1:].max(dim=3)[0]

        # -----------------------------------------------------------------------
        # TODO (Phase 4): Hierarchical mixing forward pass
        #
        # Replace this block once the episode runner populates these batch fields:
        #   batch["rsu_agent_qs"]         (batch, T, max_rsus, max_agents_per_rsu)
        #   batch["agent_masks_per_rsu"]  (batch, T, max_rsus, max_agents_per_rsu)
        #   batch["local_states"]         (batch, T, max_rsus, max_agents_per_rsu * obs_dim)
        #   batch["zone_assignments"]     (batch, T, n_agents)
        #   batch["global_states"]        (batch, T+1, global_state_dim)
        #
        # Implementation will be:
        #
        #   # --- online path ---
        #   rsu_agent_qs   = batch["rsu_agent_qs"][:, :-1]           # (B, T, R, A)
        #   agent_masks    = batch["agent_masks_per_rsu"][:, :-1]     # (B, T, R, A)
        #   local_states   = batch["local_states"][:, :-1]            # (B, T, R, A*obs)
        #   global_states  = batch["global_states"][:, :-1]           # (B, T, G)
        #   rsu_mask       = (agent_masks.sum(-1) > 0).float()        # (B, T, R)
        #
        #   BT = batch_size * max_t
        #   R  = max_rsus
        #   local_qtots = self.local_mixer(
        #       rsu_agent_qs.view(BT * R, max_agents_per_rsu),
        #       local_states.view(BT * R, -1),
        #       agent_masks.view(BT * R, max_agents_per_rsu)
        #   ).view(BT, R)                                             # (BT, R)
        #   q_tot = self.global_mixer(
        #       local_qtots,
        #       global_states.view(BT, -1),
        #       rsu_mask.view(BT, R)
        #   ).view(batch_size, max_t, 1)
        #
        #   # --- target path (same reshape, use target mixers) ---
        #   ...
        #
        # -----------------------------------------------------------------------
        raise NotImplementedError(
            "HierarchicalQLearner.train() mixing section requires Phase 4 batch fields. "
            "See TODO block above."
        )

    def _update_targets(self):
        """Update target networks — mirrors QLearner pattern."""
        if self.target_update_mode == "hard":
            self.target_mac.load_state(self.mac)
            self.target_local_mixer.load_state_dict(self.local_mixer.state_dict())
            self.target_global_mixer.load_state_dict(self.global_mixer.state_dict())
        elif self.target_update_mode == "soft":
            for tp, p in zip(self.target_mac.parameters(), self.mac.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for tp, p in zip(self.target_local_mixer.parameters(), self.local_mixer.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for tp, p in zip(self.target_global_mixer.parameters(), self.global_mixer.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def cuda(self):
        """Move networks to GPU."""
        self.mac.cuda()
        self.target_mac.cuda()
        self.local_mixer.cuda()
        self.target_local_mixer.cuda()
        self.global_mixer.cuda()
        self.target_global_mixer.cuda()
        self.device = torch.device("cuda")

    def cpu(self):
        """Move networks to CPU."""
        self.mac.cpu()
        self.target_mac.cpu()
        self.local_mixer.cpu()
        self.target_local_mixer.cpu()
        self.global_mixer.cpu()
        self.target_global_mixer.cpu()
        self.device = torch.device("cpu")

    def save_models(self, path):
        """Save model parameters and optimizer state."""
        os.makedirs(path, exist_ok=True)
        self.mac.save_models(path)
        torch.save(self.local_mixer.state_dict(), f"{path}/local_mixer.th")
        torch.save(self.global_mixer.state_dict(), f"{path}/global_mixer.th")
        torch.save(self.optimizer.state_dict(), f"{path}/optimizer.pth")

    def load_models(self, path):
        """Load model parameters and optimizer state."""
        self.mac.load_models(path)
        self.local_mixer.load_state_dict(
            torch.load(f"{path}/local_mixer.th", map_location=self.device)
        )
        self.global_mixer.load_state_dict(
            torch.load(f"{path}/global_mixer.th", map_location=self.device)
        )
        self._update_targets()
        opt_path = f"{path}/optimizer.pth"
        if os.path.exists(opt_path):
            self.optimizer.load_state_dict(
                torch.load(opt_path, map_location=self.device)
            )
