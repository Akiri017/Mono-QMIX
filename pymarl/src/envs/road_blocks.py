"""
RoadBlockageManager — random temporary road blockages for SUMO via libsumo/TraCI.

Injects stochastic road closure events into the simulation to stress-test
the Civiq framework's adaptive routing under dynamic network disruptions,
as required by the thesis panel.

Usage:
    manager = RoadBlockageManager(edge_list, seed=42)
    manager.step(current_sim_time)          # call every simulation sub-step
    blocked = manager.get_blocked_edges()   # for observation/logging
"""

import random
import logging

logger = logging.getLogger(__name__)


class RoadBlockageManager:
    """
    Manages random temporary road blockages in a SUMO simulation.

    Blockages are applied by disallowing all vehicle classes on all lanes
    of a randomly selected edge for a randomly sampled duration, then
    restored automatically when the duration expires.

    Args:
        edge_list:        List of SUMO edge IDs eligible for blockage.
                          Internal edges (starting with ':') are filtered out.
        block_probability: Probability per simulation second that a new
                          blockage event fires. Default 0.003 (~1 event per
                          episode on a 3600s episode).
        min_duration:     Minimum blockage duration in simulation seconds.
        max_duration:     Maximum blockage duration in simulation seconds.
        max_concurrent:   Maximum number of simultaneously blocked edges.
                          Prevents the network from becoming unsolvable.
        seed:             Random seed for reproducibility. Set per-episode
                          to ensure baselines face identical disruptions.
    """

    def __init__(
        self,
        edge_list: list,
        block_probability: float = 0.003,
        min_duration: float = 120.0,
        max_duration: float = 600.0,
        max_concurrent: int = 3,
        seed: int = None,
    ):
        # Filter out internal SUMO edges (junctions start with ':')
        self.edge_list = [e for e in edge_list if not e.startswith(':')]
        self.block_probability = block_probability
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.max_concurrent = max_concurrent

        # active_blockages: {edge_id: unblock_at_sim_time}
        self.active_blockages: dict = {}

        self._rng = random.Random(seed)

    def reset(self, seed: int = None):
        """
        Reset blockage state for a new episode.

        Call this at the start of each episode (after SUMO has started)
        so blockages don't carry over between episodes.

        Args:
            seed: New random seed for this episode. Pass the same seed
                  across all baselines to ensure fair comparison.
        """
        self.active_blockages.clear()
        if seed is not None:
            self._rng.seed(seed)

    def step(self, current_sim_time: float, step_length: float = 1.0):
        """
        Advance blockage state by one simulation step.

        Call this once per sub-step inside _advance_simulation(), after
        traci.simulationStep() has been called.

        Args:
            current_sim_time: Current simulation time in seconds (traci.simulation.getTime()).
            step_length:      Simulation step length in seconds (default 1.0).
        """
        from envs.sumo_backend import backend as traci

        # 1. Lift expired blockages
        expired = [
            edge_id for edge_id, unblock_at in self.active_blockages.items()
            if current_sim_time >= unblock_at
        ]
        for edge_id in expired:
            self._unblock_edge(edge_id, traci)
            del self.active_blockages[edge_id]
            logger.info(f"[Blockage] CLEARED on '{edge_id}' at t={current_sim_time:.0f}s")

        # 2. Possibly trigger a new blockage this step
        #    block_probability is per second, so scale by step_length
        if self._rng.random() < self.block_probability * step_length:
            if len(self.active_blockages) < self.max_concurrent:
                available = [
                    e for e in self.edge_list
                    if e not in self.active_blockages
                ]
                if available:
                    edge_id = self._rng.choice(available)
                    duration = self._rng.uniform(self.min_duration, self.max_duration)
                    self._block_edge(edge_id, traci)
                    self.active_blockages[edge_id] = current_sim_time + duration
                    logger.info(
                        f"[Blockage] CREATED on '{edge_id}' "
                        f"for {duration:.0f}s at t={current_sim_time:.0f}s"
                    )

    def get_blocked_edges(self) -> list:
        """Return list of currently blocked edge IDs."""
        return list(self.active_blockages.keys())

    def _block_edge(self, edge_id: str, traci) -> None:
        """Disallow all vehicle classes on every lane of an edge."""
        try:
            num_lanes = traci.edge.getLaneNumber(edge_id)
            for lane_idx in range(num_lanes):
                lane_id = f"{edge_id}_{lane_idx}"
                traci.lane.setDisallowed(lane_id, ["passenger", "truck", "motorcycle"])

            # Reroute any vehicles currently on this edge
            for veh_id in traci.edge.getLastStepVehicleIDs(edge_id):
                try:
                    traci.vehicle.rerouteTraveltime(veh_id)
                except Exception:
                    pass

        except Exception as e:
            logger.warning(f"[Blockage] Failed to block edge '{edge_id}': {e}")

    def _unblock_edge(self, edge_id: str, traci) -> None:
        """Restore all vehicle classes on every lane of an edge."""
        try:
            num_lanes = traci.edge.getLaneNumber(edge_id)
            for lane_idx in range(num_lanes):
                lane_id = f"{edge_id}_{lane_idx}"
                traci.lane.setDisallowed(lane_id, [])
        except Exception as e:
            logger.warning(f"[Blockage] Failed to unblock edge '{edge_id}': {e}")