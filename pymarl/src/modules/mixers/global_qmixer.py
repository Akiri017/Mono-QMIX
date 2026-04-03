"""
Global QMIX Mixer Network (Level 3 — Global mixing)

Part of the Civiq hierarchical architecture. The Global Mixer aggregates
local Q_tot values from all RSU zones into a single global Q_tot. The mixer
uses the same hypernetwork structure as QMIX but is conditioned on the global
state (concatenated observations of all vehicles, padded to fixed size).

Reference architecture: docs/civiq-scaffold/civiq_sprints.md — Sprint 3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalQMixer(nn.Module):
    """
    Global mixing network that combines per-RSU local Q_tots into a global Q_tot.

    Uses the same hypernetwork-based monotonic mixing as QMixer, but operates on
    RSU-level Q_tot values instead of individual agent Q-values. Padded RSU slots
    are masked out before mixing so they do not influence the output.

    Args:
        args: Configuration dict with:
            - max_rsus: Max number of RSUs across all maps (padded dimension)
            - global_state_dim: Dimension of global state (max_total_agents * obs_dim)
            - global_mixing_embed_dim: Embedding dimension for mixing network (default: 32)
            - hypernet_layers: Number of hypernetwork layers (default: 2)
            - hypernet_embed: Hypernetwork embedding dimension (default: 64)
    """

    def __init__(self, args):
        super(GlobalQMixer, self).__init__()
        self.args = args

        self.max_rsus = args["max_rsus"]
        self.global_state_dim = args["global_state_dim"]
        self.embed_dim = args.get("global_mixing_embed_dim", 32)
        self.hypernet_layers = args.get("hypernet_layers", 2)
        self.hypernet_embed = args.get("hypernet_embed", 64)

        # Hypernetwork for first layer weights
        # Output: (max_rsus, embed_dim) mixing weights
        if self.hypernet_layers == 1:
            self.hyper_w_1 = nn.Linear(self.global_state_dim, self.max_rsus * self.embed_dim)
        elif self.hypernet_layers == 2:
            self.hyper_w_1 = nn.Sequential(
                nn.Linear(self.global_state_dim, self.hypernet_embed),
                nn.ReLU(),
                nn.Linear(self.hypernet_embed, self.max_rsus * self.embed_dim)
            )
        else:
            raise ValueError(f"hypernet_layers must be 1 or 2, got {self.hypernet_layers}")

        # Hypernetwork for first layer bias
        # Output: (embed_dim,) bias
        if self.hypernet_layers == 1:
            self.hyper_b_1 = nn.Linear(self.global_state_dim, self.embed_dim)
        else:
            self.hyper_b_1 = nn.Sequential(
                nn.Linear(self.global_state_dim, self.hypernet_embed),
                nn.ReLU(),
                nn.Linear(self.hypernet_embed, self.embed_dim)
            )

        # Hypernetwork for second layer weights
        # Output: (embed_dim, 1) mixing weights
        if self.hypernet_layers == 1:
            self.hyper_w_final = nn.Linear(self.global_state_dim, self.embed_dim)
        else:
            self.hyper_w_final = nn.Sequential(
                nn.Linear(self.global_state_dim, self.hypernet_embed),
                nn.ReLU(),
                nn.Linear(self.hypernet_embed, self.embed_dim)
            )

        # V(s) - state-dependent bias for final layer
        if self.hypernet_layers == 1:
            self.V = nn.Linear(self.global_state_dim, 1)
        else:
            self.V = nn.Sequential(
                nn.Linear(self.global_state_dim, self.hypernet_embed),
                nn.ReLU(),
                nn.Linear(self.hypernet_embed, 1)
            )

    def forward(self, rsu_qtots, global_states, rsu_mask):
        """
        Mix per-RSU local Q_tots into a global Q_tot.

        Padded RSU Q_tots are zeroed out via the mask before mixing,
        ensuring they do not influence the monotonic combination.

        Args:
            rsu_qtots:     (batch_size, max_rsus) — zero-padded local Q_tots
            global_states: (batch_size, global_state_dim) — zero-padded global state
            rsu_mask:      (batch_size, max_rsus) — 1.0=active RSU, 0.0=padded

        Returns:
            global_qtot:   (batch_size, 1)
        """
        batch_size = rsu_qtots.size(0)

        # Zero out padded RSU Q_tots so they don't contribute to mixing
        rsu_qtots = rsu_qtots * rsu_mask

        # Reshape rsu_qtots to (batch, 1, max_rsus)
        rsu_qtots = rsu_qtots.view(batch_size, 1, self.max_rsus)

        # First layer
        # Generate weights: (batch, max_rsus * embed_dim)
        w1 = torch.abs(self.hyper_w_1(global_states))  # Non-negative for monotonicity
        w1 = w1.view(batch_size, self.max_rsus, self.embed_dim)

        # Generate bias: (batch, embed_dim)
        b1 = self.hyper_b_1(global_states)
        b1 = b1.view(batch_size, 1, self.embed_dim)

        # First mixing layer: Q_rsus @ W1 + b1
        # (batch, 1, max_rsus) @ (batch, max_rsus, embed_dim) = (batch, 1, embed_dim)
        hidden = F.elu(torch.bmm(rsu_qtots, w1) + b1)

        # Second layer
        # Generate weights: (batch, embed_dim)
        w_final = torch.abs(self.hyper_w_final(global_states))  # Non-negative
        w_final = w_final.view(batch_size, self.embed_dim, 1)

        # Generate state-dependent bias V(s)
        v = self.V(global_states).view(batch_size, 1, 1)

        # Final mixing layer: hidden @ W_final + V(s)
        # (batch, 1, embed_dim) @ (batch, embed_dim, 1) = (batch, 1, 1)
        global_qtot = torch.bmm(hidden, w_final) + v

        # Reshape to (batch, 1)
        global_qtot = global_qtot.view(batch_size, 1)

        return global_qtot
