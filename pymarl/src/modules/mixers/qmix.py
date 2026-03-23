"""
QMIX Mixer Network

Monotonic value function factorization for multi-agent Q-learning.
The mixer takes individual agent Q-values and combines them into a global Q_tot,
ensuring that argmax Q_tot = argmax Q_i for each agent (monotonicity).

Reference: QMIX: Monotonic Value Function Factorisation for Decentralised Multi-Agent
           Reinforcement Learning (Rashid et al., 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QMixer(nn.Module):
    """
    QMIX mixing network that combines agent Q-values into global Q_tot.

    Uses hypernetworks to produce mixing weights that are conditioned on global state,
    ensuring monotonicity through absolute value activations.

    Args:
        args: Configuration dict with:
            - n_agents: Number of agents
            - state_shape: Dimension of global state
            - mixing_embed_dim: Embedding dimension for mixing network (default: 32)
            - hypernet_layers: Number of hypernetwork layers (default: 2)
            - hypernet_embed: Hypernetwork embedding dimension (default: 64)
    """

    def __init__(self, args):
        super(QMixer, self).__init__()
        self.args = args

        self.n_agents = args["n_agents"]
        self.state_dim = args["state_shape"]
        self.embed_dim = args.get("mixing_embed_dim", 32)
        self.hypernet_layers = args.get("hypernet_layers", 2)
        self.hypernet_embed = args.get("hypernet_embed", 64)

        # Hypernetwork for first layer weights
        # Output: (n_agents, embed_dim) mixing weights
        if self.hypernet_layers == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.n_agents * self.embed_dim)
        elif self.hypernet_layers == 2:
            self.hyper_w_1 = nn.Sequential(
                nn.Linear(self.state_dim, self.hypernet_embed),
                nn.ReLU(),
                nn.Linear(self.hypernet_embed, self.n_agents * self.embed_dim)
            )
        else:
            raise ValueError(f"hypernet_layers must be 1 or 2, got {self.hypernet_layers}")

        # Hypernetwork for first layer bias
        # Output: (embed_dim,) bias
        if self.hypernet_layers == 1:
            self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)
        else:
            self.hyper_b_1 = nn.Sequential(
                nn.Linear(self.state_dim, self.hypernet_embed),
                nn.ReLU(),
                nn.Linear(self.hypernet_embed, self.embed_dim)
            )

        # Hypernetwork for second layer weights
        # Output: (embed_dim, 1) mixing weights
        if self.hypernet_layers == 1:
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        else:
            self.hyper_w_final = nn.Sequential(
                nn.Linear(self.state_dim, self.hypernet_embed),
                nn.ReLU(),
                nn.Linear(self.hypernet_embed, self.embed_dim)
            )

        # V(s) - state-dependent bias for final layer
        if self.hypernet_layers == 1:
            self.V = nn.Linear(self.state_dim, 1)
        else:
            self.V = nn.Sequential(
                nn.Linear(self.state_dim, self.hypernet_embed),
                nn.ReLU(),
                nn.Linear(self.hypernet_embed, 1)
            )

    def forward(self, agent_qs, states):
        """
        Mix agent Q-values into global Q_tot.

        Args:
            agent_qs: Individual agent Q-values, shape (batch, n_agents)
            states: Global state, shape (batch, state_dim)

        Returns:
            q_tot: Mixed global Q-value, shape (batch, 1)
        """
        batch_size = agent_qs.size(0)

        # Reshape agent_qs to (batch, 1, n_agents)
        agent_qs = agent_qs.view(batch_size, 1, self.n_agents)

        # First layer
        # Generate weights: (batch, n_agents * embed_dim)
        w1 = torch.abs(self.hyper_w_1(states))  # Ensure non-negative (monotonicity)
        w1 = w1.view(batch_size, self.n_agents, self.embed_dim)  # (batch, n_agents, embed_dim)

        # Generate bias: (batch, embed_dim)
        b1 = self.hyper_b_1(states)
        b1 = b1.view(batch_size, 1, self.embed_dim)

        # First mixing layer: Q_agents @ W1 + b1
        # (batch, 1, n_agents) @ (batch, n_agents, embed_dim) = (batch, 1, embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)

        # Second layer
        # Generate weights: (batch, embed_dim)
        w_final = torch.abs(self.hyper_w_final(states))  # Ensure non-negative
        w_final = w_final.view(batch_size, self.embed_dim, 1)  # (batch, embed_dim, 1)

        # Generate state-dependent bias V(s)
        v = self.V(states).view(batch_size, 1, 1)

        # Final mixing layer: hidden @ W_final + V(s)
        # (batch, 1, embed_dim) @ (batch, embed_dim, 1) = (batch, 1, 1)
        q_tot = torch.bmm(hidden, w_final) + v

        # Reshape to (batch, 1)
        q_tot = q_tot.view(batch_size, 1)

        return q_tot
