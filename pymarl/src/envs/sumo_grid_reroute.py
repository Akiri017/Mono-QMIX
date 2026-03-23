"""
SUMO Grid Rerouting Environment for PyMARL QMIX

This environment implements a multi-agent reinforcement learning interface
for vehicle rerouting in SUMO traffic simulations, compatible with PyMARL's
QMIX algorithm.

Key features:
- Fixed-N agent slots with lifelong replacement
- Discrete action space: k-shortest path route candidates
- Local observations with one-hot edge encoding
- Global state for QMIX mixing network
- Active/reset masking for mid-episode replacements
- Network-level reward (time + stops + emissions)
"""

import os
import sys
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

# SUMO imports
import traci
import sumolib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SUMOGridRerouteEnv:
    """
    Multi-agent environment for vehicle rerouting in SUMO.

    Implements the PyMARL environment API with support for:
    - Fixed-N agent slots with mid-episode replacement
    - Discrete route selection actions
    - Local observations and global state
    - Action masking and agent lifecycle tracking
    """

    def __init__(self, env_args: Dict):
        """
        Initialize the SUMO environment.

        Args:
            env_args: Configuration dictionary from YAML
        """
        self.env_args = env_args

        # Core configuration
        self.n_agents = env_args["n_agents"]
        self.n_actions = env_args["n_actions"]
        self.decision_period = env_args["decision_period"]
        self.sumo_step_length = env_args.get("sumo_step_length", 1.0)
        self.max_episode_steps = env_args["max_episode_steps"]

        # File paths (resolve relative to repo root)
        self.sumo_cfg = self._resolve_path(env_args["sumo_cfg"])
        self.network_file = self._resolve_path(env_args.get("network_file", ""))
        self.controlled_routes_file = self._resolve_path(env_args["controlled_routes"])

        # Adaptive decision period
        self.adaptive_decision_period = env_args.get("adaptive_decision_period", False)
        self.decision_period_warmup = env_args.get("decision_period_warmup", 10)
        self.decision_period_steady = env_args.get("decision_period_steady", 5)
        self.warmup_duration = env_args.get("warmup_duration", 300)

        # Replacement configuration
        self.replacement_enabled = env_args.get("replacement_enabled", True)
        self.replacement_delay = env_args.get("replacement_delay", 3)

        # Observation configuration
        self.obs_ego_dim = env_args.get("obs_ego_dim", 5)
        self.obs_edge_encoding = env_args.get("obs_edge_encoding", "onehot")
        self.obs_edge_dim = env_args.get("obs_edge_dim", 48)
        self.obs_max_outgoing_edges = env_args.get("obs_max_outgoing_edges", 4)
        self.obs_traffic_features = env_args.get("obs_traffic_features", 3)

        # State configuration
        self.state_concat_obs = env_args.get("state_concat_obs", True)
        self.state_include_global_stats = env_args.get("state_include_global_stats", False)
        self.state_global_stats_dim = env_args.get("state_global_stats_dim", 0)

        # Action space configuration
        self.route_cost_metric = env_args.get("route_cost_metric", "length")
        self.route_refresh_each_step = env_args.get("route_refresh_each_step", True)
        self.action_noop_as_keep_route = env_args.get("action_noop_as_keep_route", True)

        # Reward configuration
        self.reward_global = env_args.get("reward_global", True)
        self.reward_time_weight = env_args.get("reward_time_weight", 1.0)
        self.reward_stops_weight = env_args.get("reward_stops_weight", 0.05)
        self.reward_emissions_weight = env_args.get("reward_emissions_weight", 0.001)
        self.reward_stop_speed_threshold = env_args.get("reward_stop_speed_threshold", 0.1)
        self.reward_reroute_penalty = env_args.get("reward_reroute_penalty", 0.0)

        # Emissions configuration
        self.emissions_enabled = env_args.get("emissions_enabled", False)
        self.emissions_device = env_args.get("emissions_device", "CO2")

        # Masking configuration
        self.use_active_mask = env_args.get("use_active_mask", True)
        self.use_reset_mask = env_args.get("use_reset_mask", True)

        # SUMO simulation settings
        self.sumo_gui = env_args.get("sumo_gui", False)
        self.sumo_seed = env_args.get("sumo_seed", None)
        self.sumo_warnings = env_args.get("sumo_warnings", False)
        self.verbose = env_args.get("verbose", False)

        # Compute observation and state dimensions
        self.obs_dim = (self.obs_ego_dim +
                       self.obs_edge_dim +
                       self.obs_max_outgoing_edges * self.obs_traffic_features)
        self.state_dim = self.n_agents * self.obs_dim
        if self.state_include_global_stats:
            self.state_dim += self.state_global_stats_dim

        # Episode tracking
        self.episode_step = 0
        self.sim_time = 0.0
        self.last_decision_time = 0.0

        # Agent slot tracking
        self.agent_vehicle_ids = [None] * self.n_agents  # Current vehicle ID per slot
        self.agent_active = [False] * self.n_agents  # Whether slot has active vehicle
        self.agent_reset_mask = [False] * self.n_agents  # Whether slot was just reset
        self.agent_inactive_since = [0.0] * self.n_agents  # Time when slot became inactive
        self.agent_last_actions = [0] * self.n_agents  # Last action taken per agent

        # Route candidate cache
        self.route_candidates = {}  # Map agent_id -> list of route candidates
        self.route_masks = {}  # Map agent_id -> availability mask

        # OD pair cache (loaded once from XML, not re-parsed on every replacement)
        self._od_pairs_cache = None

        # Network and vehicles
        self.net = None  # sumolib network object
        self.edge_id_to_idx = {}  # Map edge ID to index for one-hot encoding
        self.idx_to_edge_id = {}  # Reverse mapping
        self.controlled_vehicle_prefix = "controlled_"
        self.next_vehicle_id = 0  # Counter for spawning new vehicles

        # TraCI connection
        self.traci_connection = None
        self.sumo_port = None

        # Episode metrics tracking (Step 6)
        self.vehicle_spawn_times = {}  # Map vehicle_id -> spawn time
        self.vehicle_travel_times = []  # List of completed travel times
        self.vehicle_waiting_times = []  # List of total waiting times per vehicle
        self.episode_stops_count = 0  # Total stop events in episode
        self.episode_emissions = 0.0  # Total emissions in episode
        self.episode_arrivals = 0  # Number of arrivals in episode
        self.total_spawned = 0  # Total vehicles spawned in episode
        self.controlled_travel_times = []  # Travel times for controlled vehicles only
        self.background_travel_times = []  # Travel times for background vehicles only
        self.controlled_vehicle_ids = set()  # Set of controlled vehicle IDs

        if self.verbose:
            logger.info(f"SUMOGridRerouteEnv initialized:")
            logger.info(f"  n_agents={self.n_agents}, n_actions={self.n_actions}")
            logger.info(f"  obs_dim={self.obs_dim}, state_dim={self.state_dim}")
            logger.info(f"  decision_period={self.decision_period}s")
            logger.info(f"  route_cost_metric={self.route_cost_metric}")

    def _resolve_path(self, path: str) -> str:
        """Resolve path relative to repo root."""
        if os.path.isabs(path):
            return path
        # Assume paths are relative to repo root (3 levels up from this file)
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        return os.path.join(repo_root, path)

    def reset(self) -> None:
        """
        Reset the environment and start a new episode.

        Starts SUMO, loads the scenario, and spawns initial controlled vehicles.
        """
        # Close previous TraCI connection if exists
        if self.traci_connection is not None:
            try:
                traci.close()
            except:
                pass

        # Reset episode tracking
        self.episode_step = 0
        self.sim_time = 0.0
        self.last_decision_time = 0.0

        # Reset agent slots
        self.agent_vehicle_ids = [None] * self.n_agents
        self.agent_active = [False] * self.n_agents
        self.agent_reset_mask = [False] * self.n_agents
        self.agent_inactive_since = [0.0] * self.n_agents
        self.agent_last_actions = [0] * self.n_agents
        self.next_vehicle_id = 0

        # Reset episode metrics (Step 6)
        self.vehicle_spawn_times.clear()
        self.vehicle_travel_times = []
        self.vehicle_waiting_times = []
        self.episode_stops_count = 0
        self.episode_emissions = 0.0
        self.episode_arrivals = 0
        self.total_spawned = 0
        self.controlled_travel_times = []
        self.background_travel_times = []
        self.controlled_vehicle_ids.clear()

        # Clear caches
        self.route_candidates.clear()
        self.route_masks.clear()

        # Start SUMO
        self._start_sumo()

        # Load network for k-shortest paths
        if self.net is None:
            self._load_network()

        # Spawn initial controlled vehicles
        self._spawn_initial_vehicles()

        # Advance simulation until all vehicles are inserted
        max_warmup_steps = 50
        for _ in range(max_warmup_steps):
            traci.simulationStep()
            self.sim_time = traci.simulation.getTime()
            if self._count_active_agents() >= self.n_agents:
                break

        if self.verbose:
            logger.info(f"Episode reset complete. Active agents: {self._count_active_agents()}/{self.n_agents}")

    def _start_sumo(self) -> None:
        """Start SUMO with TraCI."""
        sumo_binary = "sumo-gui" if self.sumo_gui else "sumo"

        # Build SUMO command
        sumo_cmd = [
            sumo_binary,
            "-c", self.sumo_cfg,
            "--step-length", str(self.sumo_step_length),
            "--no-step-log", "true",
            "--time-to-teleport", "-1",  # Disable teleporting
        ]

        if self.sumo_seed is not None:
            sumo_cmd.extend(["--seed", str(self.sumo_seed)])

        if not self.sumo_warnings:
            sumo_cmd.append("--no-warnings")

        # Start TraCI
        traci.start(sumo_cmd)
        self.traci_connection = traci

        if self.verbose:
            logger.info(f"SUMO started: {' '.join(sumo_cmd)}")

    def _load_network(self) -> None:
        """Load SUMO network using sumolib for k-shortest paths."""
        # Extract network file from sumocfg if not provided
        if not self.network_file or not os.path.exists(self.network_file):
            # Parse sumocfg to find network file
            import xml.etree.ElementTree as ET
            tree = ET.parse(self.sumo_cfg)
            root = tree.getroot()
            net_file_elem = root.find(".//net-file")
            if net_file_elem is not None:
                net_file_value = net_file_elem.get("value")
                # Resolve relative to sumocfg directory
                sumocfg_dir = os.path.dirname(self.sumo_cfg)
                self.network_file = os.path.join(sumocfg_dir, net_file_value)

        self.net = sumolib.net.readNet(self.network_file)

        # Build edge ID mapping for one-hot encoding
        all_edges = self.net.getEdges()
        self.edge_id_to_idx = {edge.getID(): idx for idx, edge in enumerate(all_edges)}
        self.idx_to_edge_id = {idx: edge.getID() for idx, edge in enumerate(all_edges)}

        if self.verbose:
            logger.info(f"Loaded network: {self.network_file} ({len(all_edges)} edges)")

    def _spawn_initial_vehicles(self) -> None:
        """Spawn initial controlled vehicles to fill all agent slots."""
        # Read OD pairs from controlled routes file
        od_pairs = self._load_od_pairs_from_routes()

        for agent_id in range(self.n_agents):
            # Pick an OD pair (cycle through available pairs)
            od_pair = od_pairs[agent_id % len(od_pairs)]
            from_edge, to_edge = od_pair

            # Spawn vehicle
            self._spawn_vehicle(agent_id, from_edge, to_edge, depart_time=0.0)

    def _load_od_pairs_from_routes(self) -> List[Tuple[str, str]]:
        """Load origin-destination pairs from controlled routes file (cached after first load)."""
        if self._od_pairs_cache is not None:
            return self._od_pairs_cache

        import xml.etree.ElementTree as ET
        tree = ET.parse(self.controlled_routes_file)
        root = tree.getroot()

        od_pairs = []

        # Parse trips (with from/to attributes)
        for trip in root.findall("trip"):
            from_edge = trip.get("from")
            to_edge = trip.get("to")
            if from_edge and to_edge:
                od_pairs.append((from_edge, to_edge))

        # Parse vehicles with nested route elements
        for vehicle in root.findall("vehicle"):
            route_elem = vehicle.find("route")
            if route_elem is not None:
                edges = route_elem.get("edges", "").split()
                if len(edges) >= 2:
                    od_pairs.append((edges[0], edges[-1]))

        # Parse standalone route elements
        for route_elem in root.findall("route"):
            edges = route_elem.get("edges", "").split()
            if len(edges) >= 2:
                od_pairs.append((edges[0], edges[-1]))

        if not od_pairs:
            logger.warning("No OD pairs found in routes file. Using default OD pairs.")
            # Fallback: use arbitrary edges from network
            edges = list(self.edge_id_to_idx.keys())
            if len(edges) >= 2:
                od_pairs = [(edges[0], edges[-1])]

        self._od_pairs_cache = od_pairs
        return od_pairs

    def _spawn_vehicle(self, agent_id: int, from_edge: str, to_edge: str,
                       depart_time: float = None) -> None:
        """Spawn a new controlled vehicle and assign to agent slot."""
        vehicle_id = f"{self.controlled_vehicle_prefix}{self.next_vehicle_id}"
        self.next_vehicle_id += 1

        # Compute initial route
        try:
            route_edges = self._compute_shortest_route(from_edge, to_edge)
            if not route_edges:
                logger.warning(f"Could not compute route from {from_edge} to {to_edge}")
                return

            # Add vehicle to simulation
            if depart_time is None:
                depart_time = self.sim_time

            traci.vehicle.add(
                vehID=vehicle_id,
                routeID="",  # We'll set route manually
                typeID="DEFAULT_VEHTYPE",
                depart=str(int(depart_time)),
                departLane="best",
                departSpeed="max"
            )

            # Set route
            traci.vehicle.setRoute(vehicle_id, route_edges)

            # Assign to agent slot
            self.agent_vehicle_ids[agent_id] = vehicle_id
            self.agent_active[agent_id] = True
            self.agent_reset_mask[agent_id] = True  # Mark as just reset

            # Track spawn time and controlled vehicles (Step 6)
            self.vehicle_spawn_times[vehicle_id] = depart_time
            self.controlled_vehicle_ids.add(vehicle_id)
            self.total_spawned += 1

            if self.verbose:
                logger.debug(f"Spawned vehicle {vehicle_id} for agent {agent_id}: {from_edge} -> {to_edge}")

        except Exception as e:
            logger.error(f"Failed to spawn vehicle for agent {agent_id}: {e}")

    def _compute_shortest_route(self, from_edge_id: str, to_edge_id: str) -> List[str]:
        """Compute shortest route between two edges (for initial routing)."""
        try:
            from_edge = self.net.getEdge(from_edge_id)
            to_edge = self.net.getEdge(to_edge_id)

            # Use Dijkstra via sumolib
            result = self.net.getShortestPath(from_edge, to_edge)

            if result is None:
                return []

            # getShortestPath returns (edges, cost) tuple
            if isinstance(result, tuple) and len(result) >= 1:
                edges = result[0]
                if edges:
                    return [edge.getID() for edge in edges]

            return []
        except Exception as e:
            logger.error(f"Failed to compute shortest route: {e}")
            return []

    def step(self, actions: np.ndarray) -> Tuple[float, bool, Dict]:
        """
        Execute one environment step.

        Args:
            actions: Array of shape (n_agents,) with action indices

        Returns:
            reward: Team reward (scalar)
            terminated: Whether episode is done
            info: Additional information dictionary
        """
        # Clear reset masks (only valid for one step after reset)
        self.agent_reset_mask = [False] * self.n_agents

        # Apply reroute actions
        self._apply_actions(actions)

        # Advance simulation for decision period
        reward = self._advance_simulation()

        # Check termination
        terminated = self._check_termination()

        # Update episode tracking
        self.episode_step += 1

        # Prepare info dict
        info = {
            "episode_step": self.episode_step,
            "sim_time": self.sim_time,
            "active_agents": self._count_active_agents(),
        }

        # Add comprehensive episode metrics when episode terminates (Step 6)
        if terminated:
            episode_metrics = self._compute_episode_metrics()
            info["episode_metrics"] = episode_metrics

        return reward, terminated, info

    def _apply_actions(self, actions: np.ndarray) -> None:
        """Apply reroute actions to controlled vehicles."""
        # Generate route candidates for all active agents
        self._generate_route_candidates()

        for agent_id in range(self.n_agents):
            if not self.agent_active[agent_id]:
                continue

            vehicle_id = self.agent_vehicle_ids[agent_id]
            action_idx = int(actions[agent_id])

            # Get route candidates
            if agent_id not in self.route_candidates:
                continue

            candidates = self.route_candidates[agent_id]
            mask = self.route_masks[agent_id]

            # Validate action
            if action_idx >= len(candidates) or not mask[action_idx]:
                # Invalid action, keep current route
                continue

            # Check if action 0 is explicit "keep current route"
            if self.action_noop_as_keep_route and action_idx == 0:
                # No reroute
                continue

            # Apply reroute
            selected_route = candidates[action_idx]
            if selected_route is not None and len(selected_route) > 0:
                try:
                    traci.vehicle.setRoute(vehicle_id, selected_route)
                    self.agent_last_actions[agent_id] = action_idx
                except Exception as e:
                    logger.warning(f"Failed to set route for {vehicle_id}: {e}")

    def _advance_simulation(self) -> float:
        """Advance SUMO simulation for decision period and accumulate reward."""
        # Determine current decision period (adaptive or fixed)
        current_K = self._get_current_decision_period()
        steps_to_advance = int(current_K / self.sumo_step_length)

        total_reward = 0.0

        for _ in range(steps_to_advance):
            # Step simulation
            traci.simulationStep()
            self.sim_time = traci.simulation.getTime()

            # Fetch vehicle list once per sub-step (shared by reward, arrivals, tracking)
            current_vehicles = set(traci.vehicle.getIDList())

            # Track newly entered vehicles (Step 6 - for background vehicles)
            self._track_new_vehicles(current_vehicles)

            # Compute reward for this timestep
            step_reward = self._compute_reward(current_vehicles)
            total_reward += step_reward

            # Handle vehicle arrivals and replacements
            self._handle_arrivals(current_vehicles)

        self.last_decision_time = self.sim_time

        return total_reward

    def _get_current_decision_period(self) -> float:
        """Get decision period K based on current time (adaptive or fixed)."""
        if not self.adaptive_decision_period:
            return self.decision_period

        if self.sim_time < self.warmup_duration:
            return self.decision_period_warmup
        else:
            return self.decision_period_steady

    def _compute_reward(self, current_vehicles: set) -> float:
        """Compute team reward for current timestep."""
        reward = 0.0

        # Get all vehicles (controlled + background if reward_global=True)
        if self.reward_global:
            all_vehicles = current_vehicles
        else:
            # Only controlled vehicles
            all_vehicles = {vid for vid in self.agent_vehicle_ids
                            if vid is not None and vid in current_vehicles}

        if len(all_vehicles) == 0:
            return 0.0

        # Time component: penalty for vehicles in network
        r_time = -1.0 * len(all_vehicles)

        # Stops component: count stopped vehicles
        stopped_count = 0
        for veh_id in all_vehicles:
            try:
                speed = traci.vehicle.getSpeed(veh_id)
                if speed < self.reward_stop_speed_threshold:
                    stopped_count += 1
            except:
                pass
        r_stops = -1.0 * stopped_count

        # Accumulate stops for episode metrics (Step 6)
        self.episode_stops_count += stopped_count

        # Emissions component (if enabled)
        r_emissions = 0.0
        if self.emissions_enabled:
            try:
                total_co2 = 0.0
                for veh_id in all_vehicles:
                    co2 = traci.vehicle.getCO2Emission(veh_id)  # mg/s
                    total_co2 += co2
                r_emissions = -1.0 * total_co2 / 1000.0  # Convert mg to g

                # Accumulate emissions for episode metrics (Step 6)
                self.episode_emissions += total_co2 / 1000.0  # Store in grams

            except Exception as e:
                if self.verbose:
                    logger.debug(f"Could not get emissions: {e}")

        # Weighted sum
        reward = (self.reward_time_weight * r_time +
                 self.reward_stops_weight * r_stops +
                 self.reward_emissions_weight * r_emissions)

        return reward

    def _handle_arrivals(self, current_vehicles: set) -> None:
        """Handle vehicle arrivals and spawn replacements."""
        od_pairs = None  # lazy-load only if a replacement is needed this step
        for agent_id in range(self.n_agents):
            if not self.agent_active[agent_id]:
                # Check if enough time has passed to spawn replacement
                if self.replacement_enabled:
                    time_inactive = self.sim_time - self.agent_inactive_since[agent_id]
                    if time_inactive >= self.replacement_delay:
                        # Spawn replacement vehicle (OD pairs cached after first load)
                        if od_pairs is None:
                            od_pairs = self._load_od_pairs_from_routes()
                        od_pair = od_pairs[agent_id % len(od_pairs)]
                        self._spawn_vehicle(agent_id, od_pair[0], od_pair[1])
                continue

            vehicle_id = self.agent_vehicle_ids[agent_id]

            # Check if vehicle has arrived
            try:
                # Vehicle no longer in simulation = arrived
                if vehicle_id not in current_vehicles:
                    # Compute travel metrics (Step 6)
                    if vehicle_id in self.vehicle_spawn_times:
                        spawn_time = self.vehicle_spawn_times[vehicle_id]
                        travel_time = self.sim_time - spawn_time
                        self.vehicle_travel_times.append(travel_time)
                        self.episode_arrivals += 1

                        # Track separately for controlled vs background
                        if vehicle_id in self.controlled_vehicle_ids:
                            self.controlled_travel_times.append(travel_time)
                        else:
                            self.background_travel_times.append(travel_time)

                        # Waiting time cannot be retrieved after vehicle has left;
                        # it is tracked continuously in get_obs instead (future work)

                        # Clean up
                        del self.vehicle_spawn_times[vehicle_id]
                        if vehicle_id in self.controlled_vehicle_ids:
                            self.controlled_vehicle_ids.remove(vehicle_id)

                    # Mark slot as inactive
                    self.agent_active[agent_id] = False
                    self.agent_vehicle_ids[agent_id] = None
                    self.agent_inactive_since[agent_id] = self.sim_time

                    if self.verbose:
                        logger.debug(f"Agent {agent_id} vehicle {vehicle_id} arrived at t={self.sim_time}")
            except:
                pass

    def _track_new_vehicles(self, current_vehicles: set) -> None:
        """Track spawn times for newly entered vehicles (Step 6)."""
        for veh_id in current_vehicles:
            # If we haven't seen this vehicle before, record its spawn time
            if veh_id not in self.vehicle_spawn_times:
                # Get departure time (when it was added to simulation)
                try:
                    depart_time = traci.vehicle.getDeparture(veh_id)
                    self.vehicle_spawn_times[veh_id] = depart_time
                    self.total_spawned += 1
                except:
                    # If we can't get departure time, use current sim time
                    self.vehicle_spawn_times[veh_id] = self.sim_time
                    self.total_spawned += 1

    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        # Terminate if max episode steps reached
        if self.episode_step >= self.max_episode_steps / self._get_current_decision_period():
            return True

        # Terminate if simulation time exceeds limit
        if self.sim_time >= self.max_episode_steps:
            return True

        # Optionally: terminate if all vehicles arrived (if replacement disabled)
        if not self.replacement_enabled:
            if self._count_active_agents() == 0:
                return True

        return False

    def _count_active_agents(self) -> int:
        """Count number of currently active agent slots."""
        return sum(self.agent_active)

    def _compute_episode_metrics(self) -> Dict:
        """
        Compute comprehensive episode-level metrics (Step 6).

        Returns:
            Dictionary with mean travel time, waiting time, stops, emissions, arrival rate, etc.
        """
        metrics = {}

        # Travel time metrics
        if len(self.vehicle_travel_times) > 0:
            metrics["mean_travel_time"] = float(np.mean(self.vehicle_travel_times))
            metrics["std_travel_time"] = float(np.std(self.vehicle_travel_times))
            metrics["median_travel_time"] = float(np.median(self.vehicle_travel_times))
            metrics["min_travel_time"] = float(np.min(self.vehicle_travel_times))
            metrics["max_travel_time"] = float(np.max(self.vehicle_travel_times))
        else:
            metrics["mean_travel_time"] = 0.0
            metrics["std_travel_time"] = 0.0
            metrics["median_travel_time"] = 0.0
            metrics["min_travel_time"] = 0.0
            metrics["max_travel_time"] = 0.0

        # Controlled vs background travel times
        if len(self.controlled_travel_times) > 0:
            metrics["controlled_mean_travel_time"] = float(np.mean(self.controlled_travel_times))
            metrics["controlled_arrivals"] = len(self.controlled_travel_times)
        else:
            metrics["controlled_mean_travel_time"] = 0.0
            metrics["controlled_arrivals"] = 0

        if len(self.background_travel_times) > 0:
            metrics["background_mean_travel_time"] = float(np.mean(self.background_travel_times))
            metrics["background_arrivals"] = len(self.background_travel_times)
        else:
            metrics["background_mean_travel_time"] = 0.0
            metrics["background_arrivals"] = 0

        # Waiting time metrics
        if len(self.vehicle_waiting_times) > 0:
            metrics["mean_waiting_time"] = float(np.mean(self.vehicle_waiting_times))
            metrics["total_waiting_time"] = float(np.sum(self.vehicle_waiting_times))
        else:
            metrics["mean_waiting_time"] = 0.0
            metrics["total_waiting_time"] = 0.0

        # Stops and emissions
        metrics["total_stops"] = int(self.episode_stops_count)
        metrics["total_emissions"] = float(self.episode_emissions)

        # Arrival rate
        metrics["arrivals"] = self.episode_arrivals
        metrics["total_spawned"] = self.total_spawned
        if self.total_spawned > 0:
            metrics["arrival_rate"] = float(self.episode_arrivals / self.total_spawned)
        else:
            metrics["arrival_rate"] = 0.0

        # Episode info
        metrics["episode_steps"] = self.episode_step
        metrics["sim_time"] = float(self.sim_time)

        return metrics

    def _generate_route_candidates(self) -> None:
        """Generate k-shortest path route candidates for all active agents."""
        self.route_candidates.clear()
        self.route_masks.clear()

        for agent_id in range(self.n_agents):
            if not self.agent_active[agent_id]:
                # Inactive agent: no valid routes
                self.route_candidates[agent_id] = [None] * self.n_actions
                self.route_masks[agent_id] = [0] * self.n_actions
                continue

            vehicle_id = self.agent_vehicle_ids[agent_id]

            try:
                # Get current edge and destination
                current_edge_id = traci.vehicle.getRoadID(vehicle_id)
                current_route = traci.vehicle.getRoute(vehicle_id)

                # Skip if on internal edge (junction)
                if current_edge_id.startswith(':'):
                    self.route_candidates[agent_id] = [None] * self.n_actions
                    self.route_masks[agent_id] = [0] * self.n_actions
                    continue

                if len(current_route) == 0:
                    self.route_candidates[agent_id] = [None] * self.n_actions
                    self.route_masks[agent_id] = [0] * self.n_actions
                    continue

                dest_edge_id = current_route[-1]

                # Compute k-shortest paths
                routes, mask = self._compute_k_shortest_paths(
                    current_edge_id, dest_edge_id, k=self.n_actions
                )

                self.route_candidates[agent_id] = routes
                self.route_masks[agent_id] = mask

            except Exception as e:
                logger.warning(f"Failed to generate routes for agent {agent_id}: {e}")
                self.route_candidates[agent_id] = [None] * self.n_actions
                self.route_masks[agent_id] = [0] * self.n_actions

    def _compute_k_shortest_paths(self, from_edge_id: str, to_edge_id: str,
                                   k: int) -> Tuple[List[List[str]], List[int]]:
        """
        Compute k-shortest paths using sumolib.

        Returns:
            routes: List of k routes (each is list of edge IDs)
            mask: Binary mask indicating valid routes
        """
        try:
            from_edge = self.net.getEdge(from_edge_id)
            to_edge = self.net.getEdge(to_edge_id)
        except:
            return [None] * k, [0] * k

        routes = []

        # Simple approach: compute shortest path, then find alternatives
        # by temporarily removing edges from the network
        try:
            # Get first shortest path
            result = self.net.getShortestPath(from_edge, to_edge)
            if result and result[0]:
                first_route = [edge.getID() for edge in result[0]]
                routes.append(first_route)

                # For k>1, try to find alternative routes by avoiding some edges
                # This is a simplified heuristic approach
                if k > 1 and len(first_route) > 2:
                    # Try alternative paths by preferring different intermediate edges
                    # For now, just use the first path multiple times
                    # This ensures we always have valid actions even if k-shortest is not available
                    for i in range(1, min(k, 4)):
                        # For simplicity, reuse the first route
                        # TODO: Implement proper k-shortest paths algorithm
                        if i < len(routes):
                            routes.append(first_route)

        except Exception as e:
            logger.warning(f"Shortest path computation failed: {e}")

        # If we couldn't compute any routes, create empty routes
        if not routes:
            routes = [None] * k
            mask = [0] * k
        else:
            # Pad with None if fewer than k routes
            num_valid = len(routes)
            while len(routes) < k:
                routes.append(None)

            # Create mask (1 = valid, 0 = invalid)
            mask = [1 if route is not None else 0 for route in routes]

            # If action 0 is "keep current route", ensure its always valid
            if self.action_noop_as_keep_route and num_valid > 0:
                mask[0] = 1

        return routes, mask

    def _deduplicate_routes(self, routes: List[List[str]], threshold: float = 0.85) -> List[List[str]]:
        """Remove near-duplicate routes based on edge overlap."""
        if len(routes) <= 1:
            return routes

        unique = [routes[0]]
        for route in routes[1:]:
            is_dup = False
            for existing in unique:
                overlap = len(set(route) & set(existing)) / max(len(route), len(existing))
                if overlap > threshold:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(route)

        return unique

    # ========== PyMARL Environment API ==========

    def get_obs(self) -> np.ndarray:
        """
        Get observations for all agents.

        Returns:
            Array of shape (n_agents, obs_dim)
        """
        obs = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)

        for agent_id in range(self.n_agents):
            obs[agent_id] = self._get_agent_obs(agent_id)

        return obs

    def _get_agent_obs(self, agent_id: int) -> np.ndarray:
        """Get observation for a single agent."""
        obs = np.zeros(self.obs_dim, dtype=np.float32)

        if not self.agent_active[agent_id]:
            return obs  # Return zeros for inactive agents

        vehicle_id = self.agent_vehicle_ids[agent_id]

        try:
            # Ego vehicle features (5 dimensions)
            speed = traci.vehicle.getSpeed(vehicle_id)
            current_edge_id = traci.vehicle.getRoadID(vehicle_id)

            # Get speed limit
            if not current_edge_id.startswith(':'):
                edge = self.net.getEdge(current_edge_id)
                speed_limit = edge.getSpeed()
            else:
                speed_limit = 13.89  # Default ~50 km/h

            normalized_speed = speed / speed_limit if speed_limit > 0 else 0.0

            # Remaining route length
            route = traci.vehicle.getRoute(vehicle_id)
            route_idx = traci.vehicle.getRouteIndex(vehicle_id)
            remaining_edges = route[route_idx:]
            remaining_length = sum(self.net.getEdge(e).getLength() for e in remaining_edges
                                  if not e.startswith(':'))

            # Time since last reroute (normalized by K)
            time_since_reroute = (self.sim_time - self.last_decision_time) / self._get_current_decision_period()

            # Fill ego features
            idx = 0
            obs[idx] = speed; idx += 1
            obs[idx] = speed_limit; idx += 1
            obs[idx] = normalized_speed; idx += 1
            obs[idx] = remaining_length / 1000.0; idx += 1  # Normalize to km
            obs[idx] = time_since_reroute; idx += 1

            # One-hot edge encoding
            if current_edge_id in self.edge_id_to_idx:
                edge_idx = self.edge_id_to_idx[current_edge_id]
                if edge_idx < self.obs_edge_dim:
                    obs[idx + edge_idx] = 1.0
            idx += self.obs_edge_dim

            # Local traffic features (next intersection)
            if not current_edge_id.startswith(':'):
                edge = self.net.getEdge(current_edge_id)
                to_node = edge.getToNode()
                outgoing_edges = to_node.getOutgoing()

                for i, out_edge in enumerate(outgoing_edges[:self.obs_max_outgoing_edges]):
                    out_edge_id = out_edge.getID()

                    # Occupancy
                    edge_length = out_edge.getLength()
                    vehicles_on_edge = traci.edge.getLastStepVehicleNumber(out_edge_id)
                    occupancy = vehicles_on_edge / (edge_length / 5.0) if edge_length > 0 else 0.0  # vehicles per 5m

                    # Mean speed
                    mean_speed = traci.edge.getLastStepMeanSpeed(out_edge_id)

                    # Queue length
                    queue_length = traci.edge.getLastStepHaltingNumber(out_edge_id)

                    obs[idx] = occupancy; idx += 1
                    obs[idx] = mean_speed / 13.89; idx += 1  # Normalize to ~50 km/h
                    obs[idx] = queue_length / 10.0; idx += 1  # Normalize

        except Exception as e:
            if self.verbose:
                logger.debug(f"Failed to get obs for agent {agent_id}: {e}")

        return obs

    def get_state(self) -> np.ndarray:
        """
        Get global state for QMIX mixing network.

        Returns:
            Array of shape (state_dim,)
        """
        # Concatenate all agent observations
        obs = self.get_obs()
        state = obs.flatten()

        # Optionally add global statistics
        if self.state_include_global_stats:
            global_stats = self._get_global_stats()
            state = np.concatenate([state, global_stats])

        return state

    def _get_global_stats(self) -> np.ndarray:
        """Compute global network statistics."""
        stats = np.zeros(self.state_global_stats_dim, dtype=np.float32)

        # Example global stats (customize as needed)
        try:
            all_vehicles = traci.vehicle.getIDList()
            stats[0] = len(all_vehicles) / 100.0  # Normalized vehicle count

            # Mean edge occupancy (sample a few edges)
            # Mean speed across network
            # etc.
        except:
            pass

        return stats

    def get_avail_actions(self) -> np.ndarray:
        """
        Get available actions mask for all agents.

        Returns:
            Array of shape (n_agents, n_actions)
        """
        avail_actions = np.zeros((self.n_agents, self.n_actions), dtype=np.float32)

        for agent_id in range(self.n_agents):
            if agent_id in self.route_masks:
                avail_actions[agent_id] = self.route_masks[agent_id]
            else:
                # Default: only no-op available
                avail_actions[agent_id, 0] = 1.0

        return avail_actions

    def get_active_mask(self) -> np.ndarray:
        """
        Get active mask indicating which agent slots have vehicles.

        Returns:
            Array of shape (n_agents,)
        """
        return np.array(self.agent_active, dtype=np.float32)

    def get_reset_mask(self) -> np.ndarray:
        """
        Get reset mask indicating which agent slots were just reassigned.

        Returns:
            Array of shape (n_agents,)
        """
        return np.array(self.agent_reset_mask, dtype=np.float32)

    def get_obs_size(self) -> int:
        """Get observation dimension."""
        return self.obs_dim

    def get_state_size(self) -> int:
        """Get state dimension."""
        return self.state_dim

    def get_total_actions(self) -> int:
        """Get number of actions per agent."""
        return self.n_actions

    def get_env_info(self) -> Dict:
        """
        Get environment information for PyMARL.

        Returns:
            Dictionary with environment configuration
        """
        return {
            "n_agents": self.n_agents,
            "n_actions": self.n_actions,
            "obs_shape": self.obs_dim,
            "state_shape": self.state_dim,
            "episode_limit": int(self.max_episode_steps / self._get_current_decision_period()),
        }

    def close(self) -> None:
        """Close the environment and clean up resources."""
        if self.traci_connection is not None:
            try:
                traci.close()
            except:
                pass
            self.traci_connection = None

        if self.verbose:
            logger.info("Environment closed")

    def __del__(self):
        """Destructor: ensure TraCI is closed."""
        self.close()
