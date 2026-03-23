# SUMO Environment Quick Reference

## Basic Usage

```python
import yaml
from pymarl.src.envs import SUMOGridRerouteEnv

# Load configuration
with open("pymarl/src/config/envs/sumo_grid4x4.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Create environment
env = SUMOGridRerouteEnv(config["env_args"])

# Reset
env.reset()

# Get initial observations
obs = env.get_obs()  # (32, 65)
state = env.get_state()  # (2080,)
avail_actions = env.get_avail_actions()  # (32, 4)

# Training loop
for episode in range(num_episodes):
    env.reset()
    episode_reward = 0

    while not done:
        # Select actions (your policy here)
        actions = select_actions(obs, avail_actions)  # (32,)

        # Step environment
        reward, done, info = env.step(actions)
        episode_reward += reward

        # Get next observations
        obs = env.get_obs()
        state = env.get_state()
        avail_actions = env.get_avail_actions()

        # Get masks for QMIX training
        active_mask = env.get_active_mask()  # (32,)
        reset_mask = env.get_reset_mask()  # (32,)

    print(f"Episode {episode}: Reward = {episode_reward:.2f}")

env.close()
```

## Configuration Switches

### Enable Dynamic Routing (Traffic-Aware)
```yaml
route_cost_metric: "traveltime"  # instead of "length"
```

### Enable Adaptive Decision Period
```yaml
adaptive_decision_period: true
decision_period_warmup: 10
decision_period_steady: 5
warmup_duration: 300
```

### Disable Emissions Tracking
```yaml
emissions_enabled: false
```

### Change Reward Weights
```yaml
reward_time_weight: 1.0      # Increase for more time emphasis
reward_stops_weight: 0.1     # Increase for more congestion penalty
reward_emissions_weight: 0.01  # Increase for environmental focus
```

### Enable SUMO GUI (for debugging)
```yaml
sumo_gui: true
verbose: true
```

## Common Tasks

### Get Environment Info
```python
info = env.get_env_info()
# Returns: {
#   "n_agents": 32,
#   "n_actions": 4,
#   "obs_shape": 65,
#   "state_shape": 2080,
#   "episode_limit": 100
# }
```

### Check Active Agents
```python
active_mask = env.get_active_mask()
num_active = np.sum(active_mask)
print(f"Active agents: {num_active}/32")
```

### Check Which Agents Need RNN Reset
```python
reset_mask = env.get_reset_mask()
agents_to_reset = np.where(reset_mask == 1)[0]
print(f"Reset RNN states for agents: {agents_to_reset}")
```

### Validate Actions
```python
avail_actions = env.get_avail_actions()
for i in range(env.n_agents):
    valid_actions = np.where(avail_actions[i] > 0)[0]
    action = np.random.choice(valid_actions)  # Sample valid action
```

## Observation Breakdown

**Total: 65 dimensions**

```python
# Indices
ego_start = 0                              # Ego features: 0-4
ego_end = 5

edge_start = ego_end                       # One-hot edge: 5-52
edge_end = edge_start + 48

traffic_start = edge_end                   # Local traffic: 53-64
traffic_end = traffic_start + 12

# Extract components
obs = env.get_obs()[0]  # Agent 0's observation
ego_features = obs[ego_start:ego_end]      # [speed, limit, norm_speed, route_len, time]
edge_encoding = obs[edge_start:edge_end]   # One-hot (48)
traffic_features = obs[traffic_start:traffic_end]  # [occ, speed, queue] × 4 edges
```

## Debugging Tips

### View SUMO GUI During Training
```python
env_args["sumo_gui"] = True
env_args["sumo_step_length"] = 0.1  # Slower for visualization
```

### Print Detailed Logs
```python
env_args["verbose"] = True
```

### Run Single Episode with Random Actions
```python
env.reset()
for _ in range(10):
    actions = np.random.randint(0, 4, size=32)
    reward, done, info = env.step(actions)
    print(f"Step: reward={reward:.2f}, active={info['active_agents']}")
    if done:
        break
```

### Check Route Candidates
```python
# After env.step(), route candidates are cached
print(env.route_candidates[0])  # Agent 0's candidate routes
print(env.route_masks[0])       # Agent 0's action mask
```

## Performance Tips

### Speed Up Testing
```yaml
max_episode_steps: 100      # Shorter episodes
decision_period: 10         # Longer decision period (fewer decisions)
n_agents: 16                # Fewer agents
```

### Realistic Training
```yaml
max_episode_steps: 1000     # Full episode
decision_period: 10         # or use adaptive
n_agents: 32                # Full fleet
emissions_enabled: true     # Track environmental impact
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Import Error: No module named 'traci'` | Install SUMO: `pip install sumo-tools` |
| `SUMO not found` | Set `SUMO_HOME` environment variable |
| `No vehicles spawned` | Check `controlled_init.rou.xml` file exists |
| `TraCI connection refused` | Close any open SUMO instances |
| `Observation shape mismatch` | Verify network has expected edge count |
| `All actions masked` | Vehicle on internal edge or arrived |

## Testing

```bash
# Test environment
python scripts/test_env.py

# Expected output: "✓ ALL TESTS PASSED"
```
