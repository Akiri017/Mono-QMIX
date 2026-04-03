"""
Local QMIX Mixer Network (Level 2 — RSU-level mixing)

Part of the Civiq hierarchical architecture. Each RSU has a Local Mixer that
aggregates Q-values from vehicles in its zone into a local Q_tot. The mixer
uses the same hypernetwork structure as QMIX but is conditioned on a local state
(concatenated observations of vehicles in the zone, padded to fixed size).

Weights are shared across all RSU zones — one LocalQMixer instance is applied
to every zone with different inputs.

Reference architecture: docs/civiq-scaffold/civiq_sprints.md — Sprint 2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalQMixer(nn.Module):
    """
    Local mixing network that combines per-RSU agent Q-values into a local Q_tot.

    Uses the same hypernetwork-based monotonic mixing as QMixer, but operates on
    a per-zone subset of agents. Padded agent slots are masked out before mixing
    so they do not influence the output.

    Args:
        args: Configuration dict with:
            - max_agents_per_rsu: Max number of agents per RSU zone (padded dimension)
            - obs_dim: Per-agent observation dimension (65)
            - local_mixing_embed_dim: Embedding dimension for mixing network (default: 32)
            - hypernet_layers: Number of hypernetwork layers (default: 2)
            - hypernet_embed: Hypernetwork embedding dimension (default: 64)
    """

    def __init__(self, args):
        super(LocalQMixer, self).__init__()
        self.args = args

        self.max_agents = args["max_agents_per_rsu"]
        self.local_state_dim = args["max_agents_per_rsu"] * args["obs_dim"]
        self.embed_dim = args.get("local_mixing_embed_dim", 32)
        self.hypernet_layers = args.get("hypernet_layers", 2)
        self.hypernet_embed = args.get("hypernet_embed", 64)

        # Hypernetwork for first layer weights
        # Output: (max_agents, embed_dim) mixing weights
        if self.hypernet_layers == 1:
            self.hyper_w_1 = nn.Linear(self.local_state_dim, self.max_agents * self.embed_dim)
        elif self.hypernet_layers == 2:
            self.hyper_w_1 = nn.Sequential(
                nn.Linear(self.local_state_dim, self.hypernet_embed),
                nn.ReLU(),
                nn.Linear(self.hypernet_embed, self.max_agents * self.embed_dim)
            )
        else:
            raise ValueError(f"hypernet_layers must be 1 or 2, got {self.hypernet_layers}")

        # Hypernetwork for first layer bias
        # Output: (embed_dim,) bias
        if self.hypernet_layers == 1:
            self.hyper_b_1 = nn.Linear(self.local_state_dim, self.embed_dim)
        else:
            self.hyper_b_1 = nn.Sequential(
                nn.Linear(self.local_state_dim, self.hypernet_embed),
                nn.ReLU(),
                nn.Linear(self.hypernet_embed, self.embed_dim)
            )

        # Hypernetwork for second layer weights
        # Output: (embed_dim, 1) mixing weights
        if self.hypernet_layers == 1:
            self.hyper_w_final = nn.Linear(self.local_state_dim, self.embed_dim)
        else:
            self.hyper_w_final = nn.Sequential(
                nn.Linear(self.local_state_dim, self.hypernet_embed),
                nn.ReLU(),
                nn.Linear(self.hypernet_embed, self.embed_dim)
            )

        # V(s) - state-dependent bias for final layer
        if self.hypernet_layers == 1:
            self.V = nn.Linear(self.local_state_dim, 1)
        else:
            self.V = nn.Sequential(
                nn.Linear(self.local_state_dim, self.hypernet_embed),
                nn.ReLU(),
                nn.Linear(self.hypernet_embed, 1)
            )

    def forward(self, agent_qs, local_states, agent_mask):
        """
        Mix per-RSU agent Q-values into a local Q_tot.

        Padded agent Q-values are zeroed out via the mask before mixing,
        ensuring they do not influence the monotonic combination.

        Args:
            agent_qs:     (batch_size, max_agents_per_rsu) — zero-padded Q-values
            local_states: (batch_size, local_state_dim) — zero-padded local state
            agent_mask:   (batch_size, max_agents_per_rsu) — 1.0=real, 0.0=padded

        Returns:
            local_qtot:   (batch_size, 1)
        """
        batch_size = agent_qs.size(0)

        # Zero out padded agent Q-values so they don't contribute to mixing
        agent_qs = agent_qs * agent_mask

        # Reshape agent_qs to (batch, 1, max_agents)
        agent_qs = agent_qs.view(batch_size, 1, self.max_agents)

        # First layer
        # Generate weights: (batch, max_agents * embed_dim)
        w1 = torch.abs(self.hyper_w_1(local_states))  # Non-negative for monotonicity
        w1 = w1.view(batch_size, self.max_agents, self.embed_dim)

        # Generate bias: (batch, embed_dim)
        b1 = self.hyper_b_1(local_states)
        b1 = b1.view(batch_size, 1, self.embed_dim)

        # First mixing layer: Q_agents @ W1 + b1
        # (batch, 1, max_agents) @ (batch, max_agents, embed_dim) = (batch, 1, embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)

        # Second layer
        # Generate weights: (batch, embed_dim)
        w_final = torch.abs(self.hyper_w_final(local_states))  # Non-negative
        w_final = w_final.view(batch_size, self.embed_dim, 1)

        # Generate state-dependent bias V(s)
        v = self.V(local_states).view(batch_size, 1, 1)

        # Final mixing layer: hidden @ W_final + V(s)
        # (batch, 1, embed_dim) @ (batch, embed_dim, 1) = (batch, 1, 1)
        local_qtot = torch.bmm(hidden, w_final) + v

        # Reshape to (batch, 1)
        local_qtot = local_qtot.view(batch_size, 1)

        return local_qtot
