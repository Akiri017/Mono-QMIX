"""
Baseline Controllers for Policy Comparison (Step 6)

Implements non-learning baseline policies for evaluation:
- NoOp: Always keep current route (action 0)
- Greedy Shortest: Deterministically select shortest available route
- Random: Uniformly random action selection
"""

import torch
import numpy as np


class BaselineMAC:
    """
    Multi-Agent Controller for baseline policies.

    Unlike BasicMAC (learning), this controller implements simple deterministic
    or random policies for baseline comparison.
    """

    def __init__(self, scheme, groups, args):
        """
        Initialize baseline controller.

        Args:
            scheme: Data scheme (not used, but kept for compatibility)
            groups: Agent groups (not used, but kept for compatibility)
            args: Configuration dict with:
                - baseline_policy: "noop", "greedy_shortest", or "random"
                - n_agents: Number of agents
                - n_actions: Number of actions per agent
        """
        self.args = args
        self.n_agents = args["n_agents"]
        self.n_actions = args["n_actions"]
        self.policy_type = args.get("baseline_policy", "noop")

        # Environment reference (needed for greedy_shortest to access route info)
        self.env = None

        # Device
        self.device = torch.device("cpu")

        print(f"BaselineMAC initialized with policy: {self.policy_type}")

    def set_env(self, env):
        """
        Set environment reference.

        Required for greedy_shortest policy to access route candidates.
        """
        self.env = env

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        """
        Select actions according to baseline policy.

        Args:
            ep_batch: Episode batch with observations and masks
            t_ep: Current episode step
            t_env: Current environment step
            bs: Batch slice (default: all)
            test_mode: Whether in test mode (ignored for baselines)

        Returns:
            Tensor of actions (shape: batch_size x n_agents x 1)
        """
        batch_size = ep_batch.batch_size
        avail_actions = ep_batch["avail_actions"][:, t_ep]

        if self.policy_type == "noop":
            return self._select_noop(batch_size, avail_actions)
        elif self.policy_type == "greedy_shortest":
            return self._select_greedy_shortest(batch_size, avail_actions)
        elif self.policy_type == "random":
            return self._select_random(batch_size, avail_actions)
        else:
            raise ValueError(f"Unknown baseline policy: {self.policy_type}")

    def _select_noop(self, batch_size, avail_actions):
        """Select action 0 (no reroute) for all agents."""
        actions = torch.zeros(batch_size, self.n_agents, 1, dtype=torch.long, device=self.device)
        return actions

    def _select_random(self, batch_size, avail_actions):
        """Randomly select from available actions."""
        actions = torch.zeros(batch_size, self.n_agents, 1, dtype=torch.long, device=self.device)

        for b in range(batch_size):
            for agent_id in range(self.n_agents):
                # Get available actions for this agent
                avail = avail_actions[b, agent_id].cpu().numpy()
                available_actions = np.where(avail > 0)[0]

                if len(available_actions) > 0:
                    # Randomly select from available actions
                    action = np.random.choice(available_actions)
                    actions[b, agent_id, 0] = action
                else:
                    # No available actions, default to 0
                    actions[b, agent_id, 0] = 0

        return actions

    def _select_greedy_shortest(self, batch_size, avail_actions):
        """
        Greedily select shortest available route.

        Requires environment reference to access route candidates.
        """
        if self.env is None:
            raise ValueError("Environment reference not set. Call set_env() first.")

        actions = torch.zeros(batch_size, self.n_agents, 1, dtype=torch.long, device=self.device)

        for b in range(batch_size):
            for agent_id in range(self.n_agents):
                # Get available actions
                avail = avail_actions[b, agent_id].cpu().numpy()
                available_actions = np.where(avail > 0)[0]

                if len(available_actions) == 0:
                    actions[b, agent_id, 0] = 0
                    continue

                # Get route candidates from environment
                if agent_id not in self.env.route_candidates:
                    actions[b, agent_id, 0] = 0
                    continue

                candidates = self.env.route_candidates[agent_id]
                mask = self.env.route_masks[agent_id]

                # Compute cost for each available route
                min_cost = float('inf')
                best_action = 0

                for action_idx in available_actions:
                    if action_idx < len(candidates) and mask[action_idx]:
                        route = candidates[action_idx]
                        if route is not None:
                            cost = self._compute_route_cost(route)
                            if cost < min_cost:
                                min_cost = cost
                                best_action = action_idx

                actions[b, agent_id, 0] = best_action

        return actions

    def _compute_route_cost(self, route):
        """
        Compute cost of a route based on metric.

        Args:
            route: List of edge IDs

        Returns:
            Cost (float) - lower is better
        """
        if self.env is None or self.env.net is None:
            return 0.0

        total_cost = 0.0

        for edge_id in route:
            # Skip internal edges (junctions)
            if edge_id.startswith(':'):
                continue

            try:
                edge = self.env.net.getEdge(edge_id)

                # Use length as cost metric (can extend to travel time)
                if self.env.route_cost_metric == "length":
                    total_cost += edge.getLength()
                elif self.env.route_cost_metric == "traveltime":
                    # Use edge travel time if available
                    try:
                        travel_time = edge.getMeanSpeed()
                        if travel_time > 0:
                            total_cost += edge.getLength() / travel_time
                        else:
                            total_cost += edge.getLength()  # Fallback to length
                    except:
                        total_cost += edge.getLength()
                else:
                    total_cost += edge.getLength()

            except Exception as e:
                # Edge not found, skip
                pass

        return total_cost

    def init_hidden(self, batch_size):
        """Initialize hidden states (none needed for baselines)."""
        return None

    def parameters(self):
        """Return parameters (none for baselines)."""
        return []

    def load_state(self, other_mac):
        """Load state from another MAC (not applicable for baselines)."""
        pass

    def cuda(self):
        """Move to GPU (baselines stay on CPU)."""
        pass

    def save_models(self, path):
        """Save models (nothing to save for baselines)."""
        pass

    def load_models(self, path):
        """Load models (nothing to load for baselines)."""
        pass
