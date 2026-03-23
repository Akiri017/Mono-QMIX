# SUMO Grid Reroute Environment - Implementation Guide

This document describes the implementation of the SUMO+TraCI environment wrapper for PyMARL QMIX.

## Overview

The `SUMOGridRerouteEnv` class implements a multi-agent reinforcement learning environment where vehicles in a SUMO traffic simulation learn to reroute cooperatively to minimize network congestion.

**File:** `pymarl/src/envs/sumo_grid_reroute.py`

## Key Features Implemented

### 1. ✅ Fixed-N Agent Slots with Lifelong Replacement

**Implementation:**
- Each agent slot (0 to N-1) controls at most one vehicle at a time
- When a vehicle arrives at its destination:
  - Slot becomes inactive
  - After `replacement_delay` seconds, a new vehicle spawns in that slot
  - New vehicle gets a (potentially different) origin-destination pair

**State tracking:**
```python
self.agent_vehicle_ids[agent_id]  # Current SUMO vehicle ID
self.agent_active[agent_id]       # Whether slot has active vehicle
self.agent_reset_mask[agent_id]   # Whether slot was just reassigned
self.agent_inactive_since[agent_id]  # Time when slot became inactive
```

### 2. ✅ Discrete Action Space: K-Shortest Paths

**Implementation:**
- At each decision timestep, compute M=4 candidate routes using `sumolib`
- Agent selects one route by action index (0-3)
- Action 0 can be reserved as "keep current route" (no-op)

**Route generation:**
```python
def _compute_k_shortest_paths(from_edge, to_edge, k=4):
    # Uses sumolib.route.kShortestPaths
    # Cost metric: "length" (static) or "traveltime" (dynamic)
    # Returns: (list of routes, availability mask)
```

**Route cost metrics:**
- **Static (default):** Edge length (deterministic, cached)
- **Dynamic (config flag):** Current travel time from TraCI (traffic-aware)

**Switching between static/dynamic:**
```yaml
# In sumo_grid4x4.yaml
route_cost_metric: "length"      # or "traveltime"
```

### 3. ✅ Local Observations (Per Agent)

**Observation structure (65 dimensions):**

1. **Ego vehicle features (5 dims):**
   - Current speed (m/s)
   - Speed limit of current edge (m/s)
   - Normalized speed (current / limit)
   - Remaining route length (meters)
   - Time since last reroute (normalized by K)

2. **Current edge encoding (48 dims):**
   - One-hot encoding of current edge ID
   - Maps edge to unique binary feature

3. **Local traffic features (12 dims = 4 edges × 3 features):**
   - For each outgoing edge at next intersection:
     - Occupancy (vehicles per unit length)
     - Mean speed (normalized)
     - Queue length (stopped vehicles)

**Implementation:**
```python
def _get_agent_obs(agent_id) -> np.ndarray:
    # Returns shape (obs_dim,)
    # Inactive agents return zeros
```

### 4. ✅ Global State for QMIX

**State construction:**
- Concatenate all agent observations: `[obs_0, obs_1, ..., obs_31]`
- Dimension: 32 × 65 = 2080
- Optional: add global network statistics (not enabled by default)

**Implementation:**
```python
def get_state() -> np.ndarray:
    obs = self.get_obs()  # (n_agents, obs_dim)
    state = obs.flatten()  # (state_dim,)
    return state
```

### 5. ✅ Network-Level Team Reward

**Reward components:**

1. **Time component (dominant):**
   ```python
   r_time = -1.0 * num_vehicles_in_network
   ```

2. **Stops component (congestion penalty):**
   ```python
   r_stops = -1.0 * num_stopped_vehicles
   # stopped = speed < 0.1 m/s
   ```

3. **Emissions component (environmental):**
   ```python
   r_emissions = -1.0 * total_CO2_grams
   # Requires emissions_enabled: true
   ```

**Weighted sum:**
```python
reward = (w_time * r_time +
         w_stops * r_stops +
         w_emissions * r_emissions)
```

**Default weights:** `(1.0, 0.05, 0.001)`

**Global vs controlled-only:**
- By default, reward computed over **all vehicles** (controlled + background)
- This optimizes network-level efficiency, not just agent performance

### 6. ✅ Action Masking

**Three types of masks:**

1. **Available actions mask:**
   ```python
   get_avail_actions() -> np.ndarray  # (n_agents, n_actions)
   # 1 = action valid, 0 = action invalid
   ```
   - Masks invalid route candidates (e.g., fewer than M routes exist)
   - Inactive agents have only no-op available

2. **Active mask:**
   ```python
   get_active_mask() -> np.ndarray  # (n_agents,)
   # 1 = slot has vehicle, 0 = slot inactive
   ```
   - Used to mask inactive agents in loss computation

3. **Reset mask:**
   ```python
   get_reset_mask() -> np.ndarray  # (n_agents,)
   # 1 = slot was just reassigned, 0 = no change
   ```
   - Signals when to reset RNN hidden states
   - Only valid for ONE step after reassignment

### 7. ✅ Decision Period Management

**Fixed decision period:**
```yaml
decision_period: 10  # K=10 seconds
```

**Adaptive decision period (optional):**
```yaml
adaptive_decision_period: true
decision_period_warmup: 10   # K=10s for first 300s
decision_period_steady: 5    # K=5s afterwards
warmup_duration: 300
```

**Implementation:**
```python
def _get_current_decision_period() -> float:
    if not adaptive_decision_period:
        return decision_period
    if sim_time < warmup_duration:
        return decision_period_warmup
    else:
        return decision_period_steady
```

### 8. ✅ Episode Termination

Episode ends when **any** of:
1. Simulation time reaches `max_episode_steps` (default: 1000s)
2. Episode step count reaches limit (computed from time / K)
3. All vehicles arrived (only if replacement disabled)

**Implementation:**
```python
def _check_termination() -> bool:
    # Check time limits and arrival status
```

## PyMARL Environment API

The environment implements the standard PyMARL interface:

```python
class SUMOGridRerouteEnv:
    # Core methods
    def reset() -> None
    def step(actions: np.ndarray) -> Tuple[reward, terminated, info]
    def close() -> None

    # Observation/state methods
    def get_obs() -> np.ndarray          # (n_agents, obs_dim)
    def get_state() -> np.ndarray        # (state_dim,)
    def get_obs_size() -> int
    def get_state_size() -> int

    # Action methods
    def get_avail_actions() -> np.ndarray  # (n_agents, n_actions)
    def get_total_actions() -> int

    # Mask methods (for replacement)
    def get_active_mask() -> np.ndarray  # (n_agents,)
    def get_reset_mask() -> np.ndarray   # (n_agents,)

    # Environment info
    def get_env_info() -> dict
```

## Configuration

All parameters are configurable via `pymarl/src/config/envs/sumo_grid4x4.yaml`:

### Key Parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_agents` | 32 | Number of agent slots |
| `n_actions` | 4 | Number of route candidates |
| `decision_period` | 10 | Seconds between decisions |
| `route_cost_metric` | "length" | "length" or "traveltime" |
| `emissions_enabled` | true | Enable emissions tracking |
| `reward_time_weight` | 1.0 | Weight for time component |
| `reward_stops_weight` | 0.05 | Weight for stops component |
| `reward_emissions_weight` | 0.001 | Weight for emissions |

## Testing

Run the test script to verify environment functionality:

```bash
python scripts/test_env.py
```

**Tests performed:**
1. Environment initialization
2. Reset functionality
3. Observation/state/mask shapes
4. Step execution with random actions
5. Episode completion
6. Environment info

**Expected output:**
```
==============================================================
TEST: Basic Environment Functionality
==============================================================

1. Initializing environment...
   ✓ Environment created
   - n_agents: 32
   - n_actions: 4
   - obs_dim: 65
   - state_dim: 2080

2. Resetting environment...
   ✓ Reset successful

...

✓ ALL TESTS PASSED
```

## Implementation Notes

### 1. Route Deduplication

K-shortest paths may return near-duplicate routes. These are filtered:

```python
def _deduplicate_routes(routes, threshold=0.85):
    # Remove routes with >85% edge overlap
```

### 2. Internal Edge Handling

Vehicles on internal edges (junctions) are skipped during rerouting:

```python
if current_edge_id.startswith(':'):
    # Skip: vehicle is on internal junction edge
    return no_routes
```

### 3. Vehicle Spawning

Vehicles spawn with origin-destination pairs from `controlled_init.rou.xml`:

```python
def _spawn_vehicle(agent_id, from_edge, to_edge):
    vehicle_id = f"controlled_{next_vehicle_id}"
    traci.vehicle.add(...)
    traci.vehicle.setRoute(vehicle_id, initial_route)
```

### 4. Observation Normalization

Features are normalized for stable learning:
- Speed: divide by speed limit
- Route length: divide by 1000 (convert to km)
- Time: divide by decision period K
- Queue length: divide by 10

### 5. Dynamic Routing (Future)

Currently, dynamic routing uses static edge length. To implement true dynamic routing:

```python
# TODO: Implement custom edge weight function for sumolib
def get_edge_traveltime(edge_id):
    return traci.edge.getAdaptedTraveltime(edge_id)

# Pass to kShortestPaths as weight function
```

## Troubleshooting

### Issue: "SUMO_HOME not found"
**Solution:** Set environment variable:
```bash
export SUMO_HOME=/path/to/sumo
```

### Issue: "No vehicles spawned"
**Solution:** Check `controlled_init.rou.xml` has valid origin-destination pairs

### Issue: "KeyError: edge not found"
**Solution:** Verify network file path in config matches the scenario

### Issue: TraCI connection errors
**Solution:** Ensure no other SUMO instances are running, check port availability

## Next Steps

With the environment implemented, proceed to **Step 5: Integration with PyMARL QMIX**:

1. Register environment in PyMARL's env registry
2. Configure QMIX algorithm parameters
3. Modify PyMARL's controller to handle reset masks
4. Test training loop with short episodes
5. Implement baseline comparisons

See `plan_MonoQMIX_prompt.md` for full step-by-step plan.
