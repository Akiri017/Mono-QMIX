"""
Basic Multi-Agent Controller (MAC)

Manages multiple agents, handles action selection (training and evaluation),
and maintains RNN hidden states with support for mid-episode resets.
"""

import torch
import torch.nn as nn
import numpy as np
from modules.agents import RNNAgent


class BasicMAC:
    """
    Multi-Agent Controller for QMIX.

    Manages n_agents, each with an RNN-based Q-network.
    Handles action selection (epsilon-greedy) and RNN state management.

    Args:
        scheme: Data scheme (not used here, kept for compatibility)
        groups: Agent groups (not used, kept for compatibility)
        args: Configuration dict
    """

    def __init__(self, scheme, groups, args):
        self.args = args
        self.n_agents = args["n_agents"]
        self.n_actions = args["n_actions"]

        # Input shape for each agent
        input_shape = self._get_input_shape(scheme)

        # Create agent network (shared parameters across all agents)
        self.agent = RNNAgent(input_shape, args)

        # Hidden states for all agents (batch_size, n_agents, hidden_dim)
        self.hidden_states = None

        # Device
        self.device = torch.device("cuda" if args.get("use_cuda", False) and torch.cuda.is_available() else "cpu")
        self.agent.to(self.device)

    def _get_input_shape(self, scheme):
        """Get input shape for agent network."""
        # Input is just the observation
        input_shape = scheme["obs"]["vshape"]
        return input_shape

    def init_hidden(self, batch_size):
        """Initialize hidden states for all agents."""
        # Shape: (batch_size, n_agents, hidden_dim)
        hidden = self.agent.init_hidden()

        if isinstance(hidden, tuple):  # LSTM
            h, c = hidden
            self.hidden_states = (
                h.expand(batch_size, self.n_agents, -1),
                c.expand(batch_size, self.n_agents, -1)
            )
        else:  # GRU
            self.hidden_states = hidden.expand(batch_size, self.n_agents, -1)

    def select_actions(self, ep_batch, t_ep, t_env, test_mode=False):
        """
        Select actions for all agents.

        Args:
            ep_batch: Episode batch data
            t_ep: Timestep within episode
            t_env: Global environment timestep (for epsilon annealing)
            test_mode: If True, use greedy action selection (epsilon=0)

        Returns:
            actions: Selected actions, shape (batch_size, n_agents, 1)
        """
        # Get available actions
        avail_actions = ep_batch["avail_actions"][:, t_ep]

        # Get agent Q-values
        agent_qs = self.forward(ep_batch, t_ep)

        # Mask unavailable actions with very large negative value
        agent_qs[avail_actions == 0] = -1e10

        # Epsilon-greedy action selection
        if test_mode:
            # Greedy
            actions = agent_qs.max(dim=2, keepdim=True)[1]
        else:
            # Epsilon-greedy
            epsilon = self._get_epsilon(t_env)

            # Random mask: which agents take random actions
            random_mask = torch.rand(agent_qs.shape[0], self.n_agents) < epsilon
            random_mask = random_mask.to(self.device)

            # Random actions (from available actions)
            random_actions = []
            avail_actions_cpu = avail_actions.cpu().numpy()
            for i in range(agent_qs.shape[0]):
                agent_random_actions = []
                for j in range(self.n_agents):
                    avail = avail_actions_cpu[i, j]
                    avail_indices = np.where(avail == 1)[0]
                    if len(avail_indices) > 0:
                        agent_random_actions.append(np.random.choice(avail_indices))
                    else:
                        agent_random_actions.append(0)  # Fallback
                random_actions.append(agent_random_actions)
            random_actions = torch.tensor(random_actions, dtype=torch.long, device=self.device).unsqueeze(2)

            # Greedy actions
            greedy_actions = agent_qs.max(dim=2, keepdim=True)[1]

            # Combine: use random where mask is True, greedy otherwise
            actions = torch.where(random_mask.unsqueeze(2), random_actions, greedy_actions)

        return actions

    def forward(self, ep_batch, t):
        """
        Forward pass through all agents.

        Args:
            ep_batch: Episode batch data
            t: Timestep within episode

        Returns:
            agent_qs: Q-values for all agents, shape (batch_size, n_agents, n_actions)
        """
        batch_size = ep_batch.batch_size
        obs = ep_batch["obs"][:, t]  # (batch_size, n_agents, obs_dim)

        # Check for reset mask (for mid-episode slot resets)
        if hasattr(ep_batch.data, 'transition_data') and "reset_mask" in ep_batch.data.transition_data and t > 0:
            reset_mask = ep_batch["reset_mask"][:, t]  # (batch_size, n_agents)
            self._reset_hidden_states(reset_mask)

        # Reshape observations: (batch_size * n_agents, obs_dim)
        obs = obs.reshape(-1, obs.shape[-1])

        # Reshape hidden states: (batch_size * n_agents, hidden_dim)
        if isinstance(self.hidden_states, tuple):  # LSTM
            h = self.hidden_states[0].reshape(-1, self.hidden_states[0].shape[-1])
            c = self.hidden_states[1].reshape(-1, self.hidden_states[1].shape[-1])
            hidden = (h, c)
        else:  # GRU
            hidden = self.hidden_states.reshape(-1, self.hidden_states.shape[-1])

        # Forward pass through agent network
        q, hidden = self.agent(obs, hidden)

        # Update hidden states
        if isinstance(hidden, tuple):  # LSTM
            self.hidden_states = (
                hidden[0].reshape(batch_size, self.n_agents, -1),
                hidden[1].reshape(batch_size, self.n_agents, -1)
            )
        else:  # GRU
            self.hidden_states = hidden.reshape(batch_size, self.n_agents, -1)

        # Reshape Q-values: (batch_size, n_agents, n_actions)
        q = q.reshape(batch_size, self.n_agents, -1)

        return q

    def _reset_hidden_states(self, reset_mask):
        """
        Reset hidden states for agents with reset_mask == 1.

        Args:
            reset_mask: Binary mask, shape (batch_size, n_agents)
        """
        reset_mask = reset_mask.to(self.device).unsqueeze(2)  # (batch_size, n_agents, 1)

        if isinstance(self.hidden_states, tuple):  # LSTM
            h, c = self.hidden_states
            # Set to zero where reset_mask == 1
            h = h * (1 - reset_mask)
            c = c * (1 - reset_mask)
            self.hidden_states = (h, c)
        else:  # GRU
            # Set to zero where reset_mask == 1
            self.hidden_states = self.hidden_states * (1 - reset_mask)

    def _get_epsilon(self, t_env):
        """Get current epsilon for epsilon-greedy exploration."""
        epsilon_start = self.args.get("epsilon_start", 1.0)
        epsilon_finish = self.args.get("epsilon_finish", 0.05)
        epsilon_anneal_time = self.args.get("epsilon_anneal_time", 50000)

        # Linear annealing
        epsilon = epsilon_start - (epsilon_start - epsilon_finish) * min(1.0, t_env / epsilon_anneal_time)
        return epsilon

    def parameters(self):
        """Return agent network parameters."""
        return self.agent.parameters()

    def load_state(self, other_mac):
        """Load parameters from another MAC."""
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        """Move agent network to GPU."""
        self.agent.cuda()
        self.device = torch.device("cuda")

    def cpu(self):
        """Move agent network to CPU."""
        self.agent.cpu()
        self.device = torch.device("cpu")

    def save_models(self, path):
        """Save agent network parameters."""
        torch.save(self.agent.state_dict(), f"{path}/agent.pth")

    def load_models(self, path):
        """Load agent network parameters."""
        self.agent.load_state_dict(torch.load(f"{path}/agent.pth", map_location=self.device))
