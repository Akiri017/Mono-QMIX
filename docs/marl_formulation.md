# MARL Formulation for SUMO+QMIX Rerouting

This document specifies the Multi-Agent Reinforcement Learning (MARL) formulation for the SUMO grid rerouting task, designed to be compatible with PyMARL's QMIX algorithm.

## 1. Agents

**Fixed-N controlled fleet with lifelong slots:**
- **N_controlled = 32** agent slots (configurable)
- Each slot controls at most one vehicle at a time
- When a controlled vehicle arrives at its destination:
  - The slot becomes inactive for a short delay (e.g., 2-5 seconds)
  - A new controlled vehicle (new OD pair) is spawned and assigned to that slot
- This keeps PyMARL/QMIX tensors fixed-size while allowing realistic open-system traffic

**Agent identification:**
- Agent slots are indexed `0` to `N_controlled - 1`
- Each slot maintains a stable index throughout the episode
- Vehicle IDs in SUMO will change as slots are reassigned, but slot IDs remain constant

## 2. Decision Period

**Rerouting cadence:**
- **K = 10 seconds** (configurable)
- Agents select a new route every K seconds
- Between decisions, vehicles follow SUMO's default car-following and lane-changing behavior
- Only routing is modified by the learned policy

**Implementation:**
- Environment steps forward K simulation steps at a time (or accumulates reward over K steps)
- Actions are applied at `t % K == 0`

**Adaptive decision period (optional):**
- Can use different K values during warmup vs steady state
- Example: K=10s for first 300s (warmup), then K=5s afterwards
- Helps avoid initial hot-loading while maintaining responsiveness later
- Enabled via `adaptive_decision_period: true` in config

## 3. Action Space (Discrete)

**Per-agent action set:**
- **M = 4** candidate routes (configurable, keep small)
- Actions are discrete: `a ∈ {0, 1, 2, ..., M-1}`

**Candidate route generation:**
- Use **k-shortest paths** algorithm from `sumolib` to compute M candidate routes from the agent's current edge to its destination
- Cost metric options:
  - **Static:** edge length (deterministic, faster computation)
  - **Dynamic:** current travel time from TraCI (traffic-aware, more realistic)
  - **Implementation note:** Both are supported via `route_cost_metric` config flag - switching between them requires only changing this parameter and is designed to be seamless

**Route computation timing:**
- Compute candidates **at each decision timestep** (every K seconds)
- This allows routes to adapt as the vehicle progresses through the network
- For dynamic cost metric, travel times are queried from TraCI at each decision step, making routes traffic-responsive

**Action masking (availability):**
- If fewer than M feasible routes exist from current edge to destination:
  - Pad action set with "no-op" (keep current route)
  - Set `avail_actions[i][j] = 0` for invalid actions, `= 1` for valid actions
- **Mask structure:** `(n_agents, n_actions)` binary tensor
- PyMARL will use this mask to prevent selection of invalid actions

**Special case: no-op action**
- If M=4 and only 2 routes exist, actions 2 and 3 are masked as unavailable
- Optionally reserve action 0 as explicit "keep current route" (no reroute)

## 4. Observation Space (Local per agent)

Each agent observes **local information** about itself and its immediate neighborhood:

### 4.1 Ego vehicle features (5 dimensions)
1. **Current speed** (m/s): `v_current`
2. **Speed limit** of current edge (m/s): `v_max_edge`
3. **Normalized speed**: `v_current / v_max_edge`
4. **Remaining route length** (meters): distance from current position to destination along current route
5. **Time since last reroute** (seconds): normalized by K (ranges 0 to 1 within a decision period)

### 4.2 Current edge encoding (one-hot)
- **One-hot encoding** of current edge ID
  - Dimension: `n_edges` (size of network)
  - For 4x4 grid: ~48 edges (12 horizontal + 12 vertical, bidirectional)
  - Each edge gets a unique binary feature, allowing the network to learn location-specific behaviors
  - Standard approach used in QMIX and MARL literature for discrete spatial states

### 4.3 Local traffic features (next intersection)
For edges **outgoing from the next intersection** on the current route (typically 2-4 edges):
- **Occupancy** (vehicles/length): number of vehicles on edge / edge length
- **Mean speed** on edge (m/s): average speed of vehicles on that edge
- **Queue length** (vehicles): count of stopped/slow vehicles (speed < 0.1 m/s)

**Aggregation:**
- If multiple outgoing edges, compute features for each candidate outgoing direction
- Pad with zeros if fewer than `max_out_edges` (e.g., max 4)
- Dimension: `max_out_edges * 3` (occupancy, mean speed, queue length)

### 4.4 Total observation dimension
```
obs_dim = 5                          (ego features)
        + n_edges                    (current edge one-hot)
        + max_out_edges * 3          (local traffic)
```

**Example (4x4 grid, max_out_edges=4):**
```
obs_dim = 5 + 48 + 12 = 65
```

**Observation tensor shape:** `(n_agents, obs_dim)`

### 4.5 Inactive agents
- When an agent slot is inactive (no vehicle assigned), fill observation with **zeros**
- Use `active_mask` to indicate which slots have valid observations

## 5. Global State (for QMIX Mixing Network)

QMIX requires a global state `s` that the mixing network uses to combine agent Q-values.

**Composition:**
- **Concatenate all agent observations:** `[obs_0, obs_1, ..., obs_{N-1}]`
  - Dimension: `N_controlled * obs_dim`
- **(Optional) Add global network summary statistics:**
  - Mean edge occupancy across network
  - Total number of vehicles in network
  - Mean speed across all edges
  - Dimension: ~3-5 scalars

**Default choice:** Start with **concatenated observations only** (simpler, sufficient for small N and small grid).

**State tensor shape:** `(state_dim,)` where:
```
state_dim = n_agents * obs_dim  [+ optional_global_stats]
```

**Example (N=32, obs_dim=65):**
```
state_dim = 32 * 65 = 2080  (or + 5 if adding global stats)
```

## 6. Reward Structure

### 6.1 Primary objective: Network-level reward
The reward is computed over **all vehicles** (controlled fleet + background traffic) to optimize global network efficiency.

**Components:**
1. **Time component (dominant):**
   - Negative total time loss across all vehicles
   - Formula: `r_time = -1.0 * (sum of travel time for all active vehicles in this timestep)`
   - Alternatively: `r_time = -1.0 * sum(1.0 for each vehicle still in network)`

2. **Stops component (penalty for congestion):**
   - Count vehicles with speed < threshold (e.g., 0.1 m/s)
   - Formula: `r_stops = -0.05 * (number of stopped vehicles)`

3. **Emissions component (environmental objective):**
   - SUMO emissions model is enabled by default
   - Formula: `r_emissions = -0.001 * (total CO2 in grams this timestep)`
   - Can track CO2, NOx, PMx, or fuel consumption

**Reward aggregation:**
```python
reward = w_time * r_time + w_stops * r_stops + w_emissions * r_emissions
```

**Default weights (configurable via YAML):**
```
w_time = 1.0
w_stops = 0.05
w_emissions = 0.001
```

### 6.2 Stabilizer (optional secondary terms)
If training is too noisy, add:
- **Reroute penalty:** small cost for changing route (discourage oscillation)
  - `r_reroute = -0.01 if action != "keep current route"`
- **Anti-oscillation:** penalty for changing to a route recently used

### 6.3 Reward structure for PyMARL
- **Team reward:** All agents receive the same global reward `r`
- Shape: scalar (not per-agent, since QMIX uses a shared team reward)

## 7. Episode Termination

Episode ends when **either**:
1. **All vehicles arrive:** All controlled vehicles have reached their destinations
2. **Max simulation time:** `T_max` seconds (e.g., 600s or 1000s)

**Recommended:** Start with a generous `T_max` (e.g., 1000s) and monitor actual episode lengths. Adjust if episodes consistently end early or always hit the limit.

## 8. Masks for Mid-Episode Slot Resets

Because replacement occurs mid-episode, some agent slots will reset while the episode continues.

### 8.1 Active mask
- **`active_mask[i]`**: binary, 1 if slot `i` currently controls an in-network vehicle, 0 otherwise
- **Shape:** `(n_agents,)`
- **Usage:** Mask inactive slots in loss computation and action selection

### 8.2 Reset mask
- **`reset_mask[i]`**: binary, 1 if slot `i` was just reassigned to a new vehicle on this timestep, 0 otherwise
- **Shape:** `(n_agents,)`
- **Usage:** Reset RNN hidden states for slots that have been reassigned

### 8.3 Available actions mask
- **`avail_actions[i][j]`**: binary, 1 if action `j` is valid for agent `i`, 0 otherwise
- **Shape:** `(n_agents, n_actions)`
- **Usage:** Mask invalid route candidates and no-op actions for inactive agents

## 9. PyMARL Environment API

The SUMO environment must implement the standard PyMARL environment interface:

### Methods
```python
class SUMOGridRerouteEnv:
    def reset(self) -> None
    def step(self, actions: np.ndarray) -> Tuple[reward, terminated, info]
    def get_obs(self) -> np.ndarray          # shape: (n_agents, obs_dim)
    def get_state(self) -> np.ndarray        # shape: (state_dim,)
    def get_avail_actions(self) -> np.ndarray  # shape: (n_agents, n_actions)
    def get_total_actions(self) -> int       # M
    def get_obs_size(self) -> int            # obs_dim
    def get_state_size(self) -> int          # state_dim
    def get_env_info(self) -> dict
```

### Additional mask methods (for replacement support)
```python
    def get_active_mask(self) -> np.ndarray  # shape: (n_agents,)
    def get_reset_mask(self) -> np.ndarray   # shape: (n_agents,)
```

## 10. RNN State Management

PyMARL's default agent architecture uses an **RNN (GRU)** to handle partial observability over time.

**Critical requirement:**
- When a slot is reassigned to a new vehicle (`reset_mask[i] == 1`), the RNN hidden state for that slot must be **reset to zero**
- This prevents hidden-state leakage between unrelated vehicle trajectories

**Implementation:**
- The environment exposes `get_reset_mask()`
- PyMARL's agent/controller must check this mask and reset `h[i] = 0` for slots with `reset_mask[i] == 1`

## 11. Summary of Key Parameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Controlled agents | `N_controlled` | 32 | Number of fixed agent slots |
| Decision period | `K` | 10 | Seconds between rerouting decisions |
| Candidate routes | `M` | 4 | Discrete action space size |
| Observation dim | `obs_dim` | 65 | Per-agent observation vector size |
| State dim | `state_dim` | 2080 | Global state vector size (N × obs_dim) |
| Episode length | `T_max` | 1000 | Max simulation time (seconds) |
| Reward weights | `(w_time, w_stops, w_emissions)` | (1.0, 0.05, 0.001) | Reward component weights |
| Route cost metric | - | static (length) | "length" (static) or "traveltime" (dynamic) |
| Emissions tracking | - | enabled | Track CO2 emissions for environmental objective |

## 12. Next Steps (Implementation)

With this formulation defined, the next step (Step 4) is to implement the SUMO+TraCI environment wrapper:
- `pymarl/src/envs/sumo_grid_reroute.py` — environment class implementing the above API
- `pymarl/src/config/envs/sumo_grid4x4.yaml` — configuration file with these parameters
- Candidate route generation using `sumolib` k-shortest paths
- TraCI integration for vehicle control and observation collection
