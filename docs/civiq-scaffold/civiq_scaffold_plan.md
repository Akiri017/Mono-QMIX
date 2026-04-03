# Civiq Scaffolding Plan — Building on Monolithic QMIX

**Date:** 2026-04-02  
**Status:** Implementation checklist — all design decisions locked  
**Prerequisite:** Mono QMIX smoke test passing (confirmed 2026-03-23)

---

## Design decisions (locked)

| Decision | Choice | Rationale |
|---|---|---|
| Backpropagation | Single end-to-end loss at Level 3 | Gradients flow Global Mixer → Local Mixers → DRQNs. Structurally identical to Mono QMIX, just with two mixing stages. |
| Variable agents per RSU | Pad + mask to `max_agents_per_rsu` | Keeps `LocalQMixer` structurally identical to vanilla `QMixer`. Simpler than attention, easier to debug. |
| RSU count per map | Different per map, pad Global Mixer to `max_rsus` | RSU placement follows intersection topology. Padding at Level 3 enables full weight transfer across curriculum stages. |
| Local state | Concat of observations of vehicles in RSU zone, padded to `max_agents_per_rsu × 65` | Same construction as Mono QMIX global state, spatially scoped. |
| Global state | Concat of all agent observations (same as Mono QMIX `get_state()`) | Full information for Global Mixer hypernetwork, consistent with Rashid et al. |
| Weight transfer (curriculum) | All 3 levels transfer directly | Padding/masking at both mixer levels makes architectures identical across maps. |

---

## Tensor shapes reference

### Current Mono QMIX shapes

```
obs_dim          = 65 (5 ego + 48 one-hot edge + 12 local traffic)
n_agents         = variable per episode (vehicles in simulation)
state_dim        = n_agents × 65 (flattened concat of all obs)
n_actions        = [check your config — number of routing actions]
mixing_embed_dim = 32 (default PyMARL, confirm in your config)
```

### Civiq shapes (new)

```
max_agents_per_rsu  = [TBD — empirically from LOS E peak vehicle count per zone]
max_rsus            = [TBD — BGC Full RSU count, largest across maps]
n_rsus_this_map     = varies (4 for Synthetic, ~6 for BGC Core, ~16 for BGC Full)

# Level 1: Agent DRQN (unchanged)
agent_input:   (batch, timestep, obs_dim=65)
agent_output:  (batch, timestep, n_actions)    — Q_i per agent

# Level 2: Local Mixer (per RSU)
local_mixer_input_qs:    (batch, timestep, max_agents_per_rsu)   — padded Q-values
local_mixer_input_state: (batch, timestep, max_agents_per_rsu × 65)  — padded local state
local_mixer_mask:        (batch, timestep, max_agents_per_rsu)   — 1 for real, 0 for padded
local_mixer_output:      (batch, timestep, 1)    — local Q_tot scalar

# Level 3: Global Mixer
global_mixer_input_qs:    (batch, timestep, max_rsus)    — padded local Q_tots
global_mixer_input_state: (batch, timestep, n_agents × 65)  — full global state (same as Mono)
global_mixer_mask:        (batch, timestep, max_rsus)    — 1 for active RSUs, 0 for padded
global_mixer_output:      (batch, timestep, 1)    — global Q_tot scalar → loss
```

---

## Phase 0: RSU zone manager

**Goal:** At every timestep, map each vehicle to exactly one RSU zone.  
**New files:** `src/components/rsu_zone_manager.py`  
**Config additions:** Per-map RSU config files

### Step 0.1 — RSU position configs

Create a config per map. Example structure:

```yaml
# config/envs/rsu/synthetic_4x4.yaml
rsu_positions:
  - id: 0
    x: 250.0
    y: 250.0
    radius: 300.0
  - id: 1
    x: 750.0
    y: 250.0
    radius: 300.0
  # ... etc
max_agents_per_rsu: 30   # set after empirical check
max_rsus: 16             # set to BGC Full count, used across ALL maps
```

**Action item:** For each map, run a SUMO episode at LOS E and log vehicle positions per timestep. Determine: (a) where to place RSUs (at major intersections), (b) the peak vehicle count in any single zone (sets `max_agents_per_rsu`), and (c) the total number of RSUs per map (BGC Full count sets `max_rsus`).

### Step 0.2 — Zone assignment function

```python
# src/components/rsu_zone_manager.py

class RSUZoneManager:
    def __init__(self, rsu_config):
        self.rsu_positions = rsu_config['rsu_positions']  # list of {id, x, y, radius}
        self.n_rsus = len(self.rsu_positions)
        self.max_rsus = rsu_config['max_rsus']
        self.max_agents_per_rsu = rsu_config['max_agents_per_rsu']

    def assign_vehicles_to_zones(self, vehicle_positions):
        """
        Args:
            vehicle_positions: dict {vehicle_id: (x, y)}
        Returns:
            zone_assignments: dict {rsu_id: [vehicle_ids]}
        """
        # Assign each vehicle to nearest RSU
        # Handle: vehicle outside all radii → assign to nearest
        # Handle: deterministic tiebreak (lower RSU id wins)
        pass

    def get_rsu_mask(self):
        """Returns mask of shape (max_rsus,) — 1 for active RSUs, 0 for padded"""
        mask = np.zeros(self.max_rsus)
        mask[:self.n_rsus] = 1.0
        return mask
```

### Step 0.3 — Standalone validation

Run a SUMO episode with no RL. At each timestep, log zone assignments. Verify:
- [ ] Every vehicle is assigned to exactly one RSU
- [ ] Assignments make geographic sense (check against SUMO GUI)
- [ ] No RSU exceeds `max_agents_per_rsu` at LOS E
- [ ] Zone assignment computation time is negligible relative to SUMO step time

---

## Phase 1: Local mixer module (Level 2)

**Goal:** A QMIX mixing network that handles variable agent counts via padding/masking.  
**New file:** `src/modules/mixers/local_qmixer.py`  
**Template:** Fork from existing `QMixer` class

### Step 1.1 — Implement `LocalQMixer`

Key differences from vanilla `QMixer`:

```python
# src/modules/mixers/local_qmixer.py

class LocalQMixer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.max_agents = args.max_agents_per_rsu
        self.state_dim = args.max_agents_per_rsu * args.obs_dim  # 65 per agent, padded
        self.embed_dim = args.local_mixing_embed_dim  # new config param

        # Hypernetworks — identical structure to QMixer
        # but sized for max_agents_per_rsu and local_state_dim
        self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.max_agents)
        self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)
        self.V = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )

    def forward(self, agent_qs, local_states, agent_mask):
        """
        Args:
            agent_qs:     (batch*T, max_agents_per_rsu) — padded with 0s
            local_states: (batch*T, max_agents_per_rsu * 65) — padded with 0s
            agent_mask:   (batch*T, max_agents_per_rsu) — 1=real, 0=padded
        Returns:
            local_qtot:   (batch*T, 1)
        """
        bs = agent_qs.size(0)

        # Mask the Q-values before mixing
        agent_qs = agent_qs * agent_mask

        agent_qs = agent_qs.view(-1, 1, self.max_agents)
        local_states = local_states.reshape(-1, self.state_dim)

        # First layer
        w1 = th.abs(self.hyper_w_1(local_states))
        b1 = self.hyper_b_1(local_states)
        w1 = w1.view(-1, self.max_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)

        # Second layer
        w_final = th.abs(self.hyper_w_final(local_states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        v = self.V(local_states).view(-1, 1, 1)

        y = th.bmm(hidden, w_final) + v
        local_qtot = y.view(bs, 1)
        return local_qtot
```

**Critical detail on masking:** The mask is applied to `agent_qs` *before* the matrix multiply with `w1`. Since padded Q-values become 0 and the hypernetwork weights for padded positions still exist but multiply by 0, the padded agents don't contribute to the output. The bias terms (`b1`, `v`) still contribute, which is correct — they represent the baseline value of the zone independent of agent count.

### Step 1.2 — Config additions

```yaml
# In your algorithm config (e.g., civiq.yaml)
local_mixing_embed_dim: 32    # same as Mono QMIX default, can tune later
max_agents_per_rsu: 30        # TBD from Phase 0 empirical check
```

---

## Phase 2: Global mixer module (Level 3)

**Goal:** A standard QMIX mixing network operating on RSU-level Q_tot values.  
**New file:** `src/modules/mixers/global_qmixer.py`  
**Template:** Fork from existing `QMixer` class

### Step 2.1 — Implement `GlobalQMixer`

This is the closest to vanilla `QMixer`. Its "agents" are RSUs.

```python
# src/modules/mixers/global_qmixer.py

class GlobalQMixer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.max_rsus = args.max_rsus
        # Global state = same as Mono QMIX: all agent obs concatenated
        # But this varies per episode, so pad to a fixed max
        self.state_dim = args.global_state_dim  # max_agents_total * 65, padded
        self.embed_dim = args.global_mixing_embed_dim

        # Same hypernetwork structure as QMixer
        self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.max_rsus)
        self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)
        self.V = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )

    def forward(self, rsu_qtots, global_states, rsu_mask):
        """
        Args:
            rsu_qtots:     (batch*T, max_rsus) — padded local Q_tots
            global_states: (batch*T, global_state_dim) — padded global state
            rsu_mask:      (batch*T, max_rsus) — 1=active RSU, 0=padded
        Returns:
            global_qtot:   (batch*T, 1)
        """
        rsu_qtots = rsu_qtots * rsu_mask
        # ... rest identical to QMixer.forward()
```

### Step 2.2 — Global state dimension

**Important:** In Mono QMIX, `state_dim = n_agents × 65` and `n_agents` varies per episode. For Civiq's Global Mixer, you need a *fixed* `global_state_dim` so the hypernetwork architecture is constant. Options:

**Option A (recommended):** Set `global_state_dim = max_total_agents × 65` where `max_total_agents` is the peak vehicle count across all LOS scenarios. Pad with zeros when fewer vehicles are present. This is consistent with your padding strategy everywhere else.

**Option B:** Use a fixed-size global state representation (e.g., per-edge statistics for all edges in the network). This decouples state dim from vehicle count but requires building a new state representation.

Go with Option A for consistency. You already pad at every other level — doing the same at the global state level keeps the system uniform.

### Step 2.3 — Config additions

```yaml
global_mixing_embed_dim: 32
max_rsus: 16                    # BGC Full RSU count
global_state_dim: 6500          # example: 100 max vehicles × 65. TBD from LOS E peak.
```

---

## Phase 3: Hierarchical learner

**Goal:** A training loop that wires Levels 1-2-3 together with a single loss.  
**New file:** `src/learners/hierarchical_q_learner.py`  
**Template:** Fork from existing `QLearner` (or `q_learner.py`)

### Step 3.1 — Constructor

```python
class HierarchicalQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.mac = mac
        self.args = args
        self.logger = logger

        # Level 2: Local Mixers (shared weights — one network, applied per RSU)
        self.local_mixer = LocalQMixer(args)
        self.target_local_mixer = copy.deepcopy(self.local_mixer)

        # Level 3: Global Mixer
        self.global_mixer = GlobalQMixer(args)
        self.target_global_mixer = copy.deepcopy(self.global_mixer)

        # RSU zone manager
        self.zone_manager = RSUZoneManager(args.rsu_config)

        # Single optimizer for all parameters
        self.params = list(mac.parameters())
        self.params += list(self.local_mixer.parameters())
        self.params += list(self.global_mixer.parameters())
        self.optimiser = Adam(params=self.params, lr=args.lr, eps=args.optim_eps)

        self.last_target_update_episode = 0
```

### Step 3.2 — Training forward pass (the core logic)

```python
def train(self, batch, t_env, episode_num):
    # 1. Get per-agent Q-values from MAC (same as Mono QMIX)
    mac_out = []  # list of (batch, n_actions) per timestep
    self.mac.init_hidden(batch.batch_size)
    for t in range(batch.max_seq_length):
        agent_outs = self.mac.forward(batch, t=t)
        mac_out.append(agent_outs)
    mac_out = th.stack(mac_out, dim=1)  # (batch, T, n_agents, n_actions)

    # Get chosen action Q-values
    chosen_actions = batch["actions"][:, :-1]  # (batch, T-1, n_agents, 1)
    chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=chosen_actions).squeeze(3)
    # chosen_action_qvals: (batch, T-1, n_agents)

    # 2. Get zone assignments from batch data (stored during episode collection)
    zone_assignments = batch["zone_assignments"]  # (batch, T, n_agents) — RSU id per vehicle
    agent_masks_per_rsu = batch["agent_masks_per_rsu"]  # pre-computed padded masks
    local_states = batch["local_states"]  # (batch, T, max_rsus, max_agents_per_rsu * 65)

    # 3. For each RSU, gather Q-values and run Local Mixer
    local_qtots = []
    for rsu_id in range(self.args.max_rsus):
        # Gather Q-values of agents in this RSU's zone (pre-padded in episode runner)
        rsu_agent_qs = batch["rsu_agent_qs"][:, :, rsu_id]  # (batch, T, max_agents_per_rsu)
        rsu_mask = agent_masks_per_rsu[:, :, rsu_id]  # (batch, T, max_agents_per_rsu)
        rsu_local_state = local_states[:, :, rsu_id]  # (batch, T, local_state_dim)

        local_qtot = self.local_mixer(
            rsu_agent_qs[:, :-1],
            rsu_local_state[:, :-1],
            rsu_mask[:, :-1]
        )
        local_qtots.append(local_qtot)

    local_qtots = th.stack(local_qtots, dim=2)  # (batch, T-1, max_rsus)

    # 4. Run Global Mixer
    global_states = batch["state"][:, :-1]  # (batch, T-1, global_state_dim)
    rsu_mask = self.zone_manager.get_rsu_mask()  # (max_rsus,) — same for all timesteps
    rsu_mask = rsu_mask.expand_as(local_qtots)

    global_qtot = self.global_mixer(local_qtots, global_states, rsu_mask)
    # global_qtot: (batch, T-1, 1)

    # 5. Compute targets (same structure, using target networks)
    # ... [target MAC forward, target local mixer, target global mixer]
    # ... [TD target: r + gamma * max_a' target_global_qtot]

    # 6. Loss (identical to Mono QMIX)
    td_error = global_qtot - targets.detach()
    mask = batch["filled"][:, :-1].float()
    td_error = td_error * mask
    loss = (td_error ** 2).sum() / mask.sum()

    # 7. Backward (gradients flow through all 3 levels)
    self.optimiser.zero_grad()
    loss.backward()
    grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
    self.optimiser.step()

    # 8. Target network updates
    if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
        self._update_targets()
        self.last_target_update_episode = episode_num
```

**Note:** The pseudocode above shows the conceptual flow. The actual implementation needs careful attention to tensor reshaping for the `bmm` operations. The pre-computation of `rsu_agent_qs`, `agent_masks_per_rsu`, and `local_states` in the episode runner (Phase 4) is critical — computing them inside `train()` is possible but messy and slow.

### Step 3.3 — Target network update

```python
def _update_targets(self):
    self.target_local_mixer.load_state_dict(self.local_mixer.state_dict())
    self.target_global_mixer.load_state_dict(self.global_mixer.state_dict())
    self.mac.target_mac()  # or however your MAC handles target updates
```

---

## Phase 4: Episode runner and replay buffer modifications

**Goal:** Collect and store zone assignment data alongside existing episode data.  
**Modified files:** Your episode runner, your environment wrapper, your replay buffer scheme

### Step 4.1 — Environment wrapper additions

Add to your SUMO environment class:

```python
def get_zone_assignments(self):
    """Returns zone assignment for each vehicle at current timestep."""
    vehicle_positions = {}
    for veh_id in self.active_vehicles:
        x, y = traci.vehicle.getPosition(veh_id)
        vehicle_positions[veh_id] = (x, y)
    return self.zone_manager.assign_vehicles_to_zones(vehicle_positions)

def get_local_states(self, zone_assignments):
    """Returns padded local state tensor for each RSU."""
    local_states = th.zeros(self.max_rsus, self.max_agents_per_rsu * self.obs_dim)
    for rsu_id, vehicle_ids in zone_assignments.items():
        for i, veh_id in enumerate(vehicle_ids[:self.max_agents_per_rsu]):
            obs = self.get_obs_agent(veh_id)  # 65-dim
            local_states[rsu_id, i*65:(i+1)*65] = th.tensor(obs)
    return local_states

def get_rsu_agent_qs(self, zone_assignments, all_agent_qs):
    """Rearranges per-agent Q-values into per-RSU padded tensors."""
    rsu_qs = th.zeros(self.max_rsus, self.max_agents_per_rsu)
    rsu_masks = th.zeros(self.max_rsus, self.max_agents_per_rsu)
    for rsu_id, vehicle_ids in zone_assignments.items():
        for i, veh_id in enumerate(vehicle_ids[:self.max_agents_per_rsu]):
            agent_idx = self.vehicle_id_to_agent_idx(veh_id)
            rsu_qs[rsu_id, i] = all_agent_qs[agent_idx]
            rsu_masks[rsu_id, i] = 1.0
    return rsu_qs, rsu_masks
```

### Step 4.2 — Episode data scheme additions

Add these fields to the episode batch scheme:

```python
# In your scheme definition
new_fields = {
    "zone_assignments": {"vshape": (max_rsus, max_agents_per_rsu), "dtype": th.long},
    "local_states": {"vshape": (max_rsus, max_agents_per_rsu * 65), "dtype": th.float32},
    "rsu_agent_qs": {"vshape": (max_rsus, max_agents_per_rsu), "dtype": th.float32},
    "agent_masks_per_rsu": {"vshape": (max_rsus, max_agents_per_rsu), "dtype": th.float32},
}
```

### Step 4.3 — Replay buffer impact

The episodic replay buffer stores full episodes. Adding the new fields increases per-episode memory. Estimate:

```
Per timestep additional storage:
  local_states:        max_rsus × max_agents_per_rsu × 65 × 4 bytes
  rsu_agent_qs:        max_rsus × max_agents_per_rsu × 4 bytes
  agent_masks_per_rsu: max_rsus × max_agents_per_rsu × 4 bytes

Example with max_rsus=16, max_agents_per_rsu=30:
  local_states:  16 × 30 × 65 × 4 = 124,800 bytes/step
  rsu_agent_qs:  16 × 30 × 4 = 1,920 bytes/step
  masks:         16 × 30 × 4 = 1,920 bytes/step
  Total: ~128 KB/step

  Per episode (100 steps): ~12.8 MB
  Replay buffer (400 episodes): ~5.1 GB additional
```

This is significant. If your 16GB RAM is already tight with the Mono QMIX buffer (we calculated ~6.65GB previously), the additional 5.1GB may not fit. Options:
- Reduce `buffer_size` from 400 to ~200 episodes
- Reduce `max_agents_per_rsu` if empirically safe
- Move to a more memory-efficient storage format (e.g., sparse storage for masks)

**Action item:** Re-run the RAM budget calculation with the new field sizes before committing to buffer parameters.

---

## Phase 5: Smoke test and validation

### Step 5.1 — Gradient flow test (before any training)

```python
# Run 1 episode, 1 training step
# After loss.backward(), check:
for name, param in mac.named_parameters():
    assert param.grad is not None, f"No gradient for {name}"
    assert not th.isnan(param.grad).any(), f"NaN gradient for {name}"

for name, param in local_mixer.named_parameters():
    assert param.grad is not None, f"No gradient for local_mixer.{name}"

for name, param in global_mixer.named_parameters():
    assert param.grad is not None, f"No gradient for global_mixer.{name}"

print(f"Agent grad norm: {agent_grad_norm:.4f}")
print(f"Local mixer grad norm: {local_mixer_grad_norm:.4f}")
print(f"Global mixer grad norm: {global_mixer_grad_norm:.4f}")
```

All three must show non-zero gradients. If the local mixer has zero gradients, the mask is killing all signal. If the global mixer has zero gradients, the local-to-global wiring is broken.

### Step 5.2 — Shape assertion test

Insert assertions at every interface:

```python
assert agent_qs.shape == (batch_size, T, n_agents, n_actions)
assert rsu_agent_qs.shape == (batch_size, T, max_rsus, max_agents_per_rsu)
assert local_states.shape == (batch_size, T, max_rsus, max_agents_per_rsu * 65)
assert local_qtots.shape == (batch_size, T, max_rsus)
assert global_qtot.shape == (batch_size, T, 1)
```

### Step 5.3 — 50k step smoke test (1 seed)

Same protocol as Mono QMIX smoke test:
- [ ] Training loop runs without crash for 50k steps
- [ ] Loss decreases from initial value (even slightly)
- [ ] No NaN/Inf in Q-values at any level
- [ ] Checkpoint saves correctly
- [ ] Checkpoint loads and evaluation runs
- [ ] Zone assignments logged and visually verified

### Step 5.4 — Equivalence sanity check

**Optional but valuable:** Temporarily set `max_rsus = 1` (single RSU covering the whole map) and verify that Civiq produces equivalent behavior to Mono QMIX. With one RSU, the Local Mixer sees all agents and the Global Mixer sees one input — this should approximate Mono QMIX's behavior. If results diverge significantly, there's a bug in the wiring.

---

## File change summary

| File | Action | Description |
|---|---|---|
| `src/components/rsu_zone_manager.py` | NEW | Zone assignment logic |
| `src/modules/mixers/local_qmixer.py` | NEW | Level 2 Local Mixer with pad+mask |
| `src/modules/mixers/global_qmixer.py` | NEW | Level 3 Global Mixer |
| `src/learners/hierarchical_q_learner.py` | NEW | Wires L1+L2+L3, single loss |
| `config/envs/rsu/*.yaml` | NEW | RSU positions per map |
| `config/algs/civiq.yaml` | NEW | Civiq-specific hyperparams |
| Episode runner | MODIFY | Add zone assignment collection per step |
| Environment wrapper | MODIFY | Add `get_zone_assignments()`, `get_local_states()` |
| Replay buffer scheme | MODIFY | Add new fields for RSU data |
| `run_experiments.py` (or equivalent) | MODIFY | Support `--learner hierarchical_q_learner` |

---

## Implementation order (critical path)

```
Phase 0 (RSU zone manager)          ← START HERE
  │
  ├── 0.1: RSU position configs
  ├── 0.2: Zone assignment function
  └── 0.3: Standalone validation     ← GATE: zone assignments verified
          │
Phase 1 (Local Mixer)
  │
  ├── 1.1: Implement LocalQMixer
  └── 1.2: Unit test forward pass    ← GATE: forward produces valid output
          │
Phase 2 (Global Mixer)
  │
  ├── 2.1: Implement GlobalQMixer
  └── 2.2: Unit test forward pass    ← GATE: forward produces valid output
          │
Phase 3 (Hierarchical Learner)
  │
  ├── 3.1: Constructor + param collection
  ├── 3.2: Training forward pass
  └── 3.3: Target network updates    ← GATE: loss.backward() succeeds
          │
Phase 4 (Episode data)
  │
  ├── 4.1: Environment wrapper additions
  ├── 4.2: Scheme additions
  └── 4.3: RAM budget re-check       ← GATE: fits in memory
          │
Phase 5 (Smoke test)
  │
  ├── 5.1: Gradient flow test
  ├── 5.2: Shape assertions
  ├── 5.3: 50k step smoke test       ← GATE: pipeline runs end-to-end
  └── 5.4: Single-RSU equivalence    ← GATE: matches Mono QMIX behavior
```

---

## Open items requiring empirical data

| Item | How to determine | Blocks |
|---|---|---|
| `max_agents_per_rsu` | Run LOS E episode, log peak vehicles per zone | Phase 1 (LocalQMixer init) |
| `max_rsus` (BGC Full) | Place RSUs on BGC Full map intersections, count | Phase 2 (GlobalQMixer init) |
| RSU positions per map | Manual placement at major intersections | Phase 0 |
| `global_state_dim` | `max_total_agents × 65`, from LOS E peak | Phase 2 |
| Replay buffer size | RAM budget with new fields | Phase 4 |
| `n_actions` | Confirm from current config | Shape verification |
