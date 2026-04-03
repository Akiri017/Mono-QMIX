# Civiq Scaffold — Sprint Task List for Claude Code

**Purpose:** Each sprint below is a self-contained coding task. Execute them in order — each sprint depends on the output of the previous one. All config values that depend on empirical SUMO data use placeholders that the developer will replace later. No sprints require running SUMO or training. The goal is a fully wired pipeline that compiles and connects end-to-end.

**Repo root:** `Mono_QMIX/`
**Primary source directory:** `pymarl/src/`

**Key existing files Claude Code must read before starting:**
- `pymarl/src/modules/mixers/qmix.py` — the mixer to fork for Local and Global mixers
- `pymarl/src/learners/q_learner.py` — the learner to fork for the hierarchical learner
- `pymarl/src/envs/sumo_grid_reroute.py` — the SUMO environment wrapper (contains `get_state()`, `get_obs()`)
- `pymarl/src/runners/episode_runner.py` — the episode data collection loop
- `pymarl/src/components/episode_buffer.py` — the replay buffer and data scheme
- `pymarl/src/controllers/basic_controller.py` — the MAC (multi-agent controller)
- `pymarl/src/config/algs/qmix_sumo.yaml` — current algorithm config
- `pymarl/src/config/envs/sumo_grid4x4.yaml` — current environment config
- `pymarl/src/main.py` — entry point, where learners/envs are registered

**Architecture context for Claude Code:**

Civiq is a hierarchical extension of QMIX with three levels:
- **Level 1 (Vehicle Agents):** DRQN agents compute individual Q-values. UNCHANGED from Mono QMIX.
- **Level 2 (Local RSU Mixers):** Each RSU covers a geographic zone. A Local Mixer (QMIX-style mixing network) aggregates Q-values from vehicles in its zone into a local Q_tot. The Local Mixer's hypernetwork is conditioned on a **local state** (concatenated observations of vehicles in the zone, padded to fixed size).
- **Level 3 (Global Mixer):** A Global Mixer (QMIX-style mixing network) aggregates local Q_tots from all RSUs into a global Q_tot. The Global Mixer's hypernetwork is conditioned on the **global state** (same as Mono QMIX: concatenated observations of all vehicles, padded to fixed size).

**Key design decisions (do not deviate from these):**
- Variable agent counts per RSU are handled by **padding to `max_agents_per_rsu` with zero-masking**
- Variable RSU counts across maps are handled by **padding to `max_rsus` with zero-masking**
- **Single end-to-end loss** at Level 3. Gradients flow backward through Global Mixer → Local Mixers → Agent DRQNs. One optimizer covers all parameters.
- **Local Mixer weights are shared** — one `LocalQMixer` network instance applied to every RSU zone (same weights, different inputs per zone).
- The existing global state (`get_state()`) is the concatenation of all agent observations flattened. Each agent observation is 65-dim (5 ego features + 48 one-hot edge encoding + 12 local traffic features).

---

## Sprint 1 — RSU config files and zone manager class

### New files to create:
- `pymarl/src/config/rsu/synthetic_4x4.yaml`
- `pymarl/src/config/rsu/bgc_core.yaml`
- `pymarl/src/config/rsu/bgc_full.yaml`
- `pymarl/src/components/rsu_zone_manager.py`

### Task 1.1 — RSU config files

Create the directory `pymarl/src/config/rsu/` and three YAML config files. All three must share the same `max_agents_per_rsu` and `max_rsus` values (these define fixed neural network dimensions for curriculum weight transfer).

Schema for each file:
```yaml
max_agents_per_rsu: 30       # placeholder — developer sets from LOS E peak
max_rsus: 16                 # placeholder — set to BGC Full RSU count
rsu_coverage_radius: 300.0   # meters

rsu_positions:
  - id: 0
    x: 0.0      # placeholder — SUMO network coordinates in meters
    y: 0.0      # placeholder
  # ... more RSUs
```

- `synthetic_4x4.yaml`: 4 placeholder RSU positions
- `bgc_core.yaml`: 6 placeholder RSU positions
- `bgc_full.yaml`: 16 placeholder RSU positions

Add a comment at the top of each file noting that coordinates are placeholders.

### Task 1.2 — RSU zone manager class

Create `pymarl/src/components/rsu_zone_manager.py`.

This class:
- Loads an RSU config (passed as a dict from YAML)
- At each timestep, assigns vehicles to RSU zones based on Euclidean distance from vehicle `(x, y)` position to RSU positions
- Each vehicle is assigned to exactly one RSU (the nearest one). If a vehicle is outside all RSU radii, assign it to the nearest RSU anyway.
- Deterministic tiebreak: if equidistant from two RSUs, assign to the lower RSU id.
- Provides methods to build padded tensors for the Local Mixer:
  - Given a dict of `{vehicle_id: agent_index}` and a flat tensor of all agent Q-values, produce `rsu_agent_qs` of shape `(max_rsus, max_agents_per_rsu)` and `agent_masks_per_rsu` of shape `(max_rsus, max_agents_per_rsu)` where mask is 1.0 for real agents and 0.0 for padding.
  - Given per-agent observations and zone assignments, produce `local_states` of shape `(max_rsus, max_agents_per_rsu * obs_dim)` — concatenated observations of vehicles in each zone, padded with zeros.
- Provides a method `get_rsu_mask()` returning shape `(max_rsus,)` — 1.0 for active RSUs on this map, 0.0 for padded RSU slots.

Use numpy for the distance calculations. Return torch tensors for the padded outputs.

Read the existing environment file `pymarl/src/envs/sumo_grid_reroute.py` to understand how vehicle positions and observations are currently accessed. The zone manager will be called from the episode runner, receiving position data that the environment already provides via TraCI.

---

## Sprint 2 — Local Mixer (Level 2)

### New file to create:
- `pymarl/src/modules/mixers/local_qmixer.py`

### Task

Read the existing `pymarl/src/modules/mixers/qmix.py` first. The `LocalQMixer` is a fork of `QMixer` with these changes:

1. **Constructor takes different dimensions:**
   - `self.n_agents` → `self.max_agents = args.max_agents_per_rsu`
   - `self.state_dim` → `self.local_state_dim = args.max_agents_per_rsu * args.obs_dim` (where `obs_dim` = 65)
   - `self.embed_dim` = `args.local_mixing_embed_dim` (new config param, default 32)

2. **`forward()` signature changes to accept a mask:**
   ```python
   def forward(self, agent_qs, local_states, agent_mask):
       """
       Args:
           agent_qs:     (batch_size, max_agents_per_rsu) — zero-padded
           local_states: (batch_size, local_state_dim) — zero-padded
           agent_mask:   (batch_size, max_agents_per_rsu) — 1.0=real, 0.0=padded
       Returns:
           local_qtot:   (batch_size, 1)
       """
   ```

3. **Masking:** Apply `agent_qs = agent_qs * agent_mask` before the first matrix multiply. This zeros out padded agent Q-values so they don't contribute to the mixing.

4. **Everything else is identical to QMixer:** Same hypernetwork structure (absolute activation on weights, unconstrained biases), same V(s) state-dependent bias on the final layer, same ELU activation on the hidden layer. Match the number of hypernet layers used in the existing `qmix.py` (check `getattr(args, "hypernet_layers", 1)`).

Do NOT modify the existing `qmix.py`. The `LocalQMixer` is a new file alongside it.

---

## Sprint 3 — Global Mixer (Level 3)

### New file to create:
- `pymarl/src/modules/mixers/global_qmixer.py`

### Task

Read the existing `pymarl/src/modules/mixers/qmix.py` first. The `GlobalQMixer` is a fork of `QMixer` with these changes:

1. **Constructor takes different dimensions:**
   - `self.n_agents` → `self.max_rsus = args.max_rsus`
   - `self.state_dim` → `self.global_state_dim = args.global_state_dim` (new config param: `max_total_agents * obs_dim`, placeholder value 6500 = 100 agents × 65)
   - `self.embed_dim` = `args.global_mixing_embed_dim` (new config param, default 32)

2. **`forward()` signature changes to accept a mask:**
   ```python
   def forward(self, rsu_qtots, global_states, rsu_mask):
       """
       Args:
           rsu_qtots:     (batch_size, max_rsus) — zero-padded local Q_tots
           global_states: (batch_size, global_state_dim) — zero-padded
           rsu_mask:      (batch_size, max_rsus) — 1.0=active RSU, 0.0=padded
       Returns:
           global_qtot:   (batch_size, 1)
       """
   ```

3. **Masking:** Apply `rsu_qtots = rsu_qtots * rsu_mask` before the first matrix multiply. Same pattern as LocalQMixer.

4. **Everything else is identical to QMixer.** Same hypernetwork structure, same activation functions, same V(s) bias.

Do NOT modify the existing `qmix.py`.

---

## Sprint 4 — Civiq algorithm config

### New file to create:
- `pymarl/src/config/algs/civiq_sumo.yaml`

### Task

Read the existing `pymarl/src/config/algs/qmix_sumo.yaml` first. Create `civiq_sumo.yaml` as a copy with these additions/changes:

```yaml
# --- Civiq-specific parameters ---
learner: "hierarchical_q_learner"
mixer: "civiq"    # signals to use the hierarchical mixer setup

# Level 2 — Local Mixer
max_agents_per_rsu: 30          # placeholder
local_mixing_embed_dim: 32
obs_dim: 65                     # per-agent observation dimension

# Level 3 — Global Mixer
max_rsus: 16                    # placeholder
global_mixing_embed_dim: 32
global_state_dim: 6500          # placeholder (100 max agents × 65 obs_dim)

# RSU config path (relative to src/config/)
rsu_config: "rsu/synthetic_4x4.yaml"    # switch per map

# --- Everything below carried over from qmix_sumo.yaml ---
# (copy all existing params: gamma, lr, epsilon settings, buffer size,
#  target_update_interval, grad_norm_clip, etc.)
```

Carry over ALL existing parameters from `qmix_sumo.yaml`. The Civiq config must be a superset — same training hyperparameters plus the new hierarchical params. Do not remove or rename any existing params.

---

## Sprint 5 — Hierarchical Q-Learner

### New file to create:
- `pymarl/src/learners/hierarchical_q_learner.py`

### Task

This is the most complex new file. Read these files first:
- `pymarl/src/learners/q_learner.py` — the base to fork from
- `pymarl/src/modules/mixers/local_qmixer.py` — created in Sprint 2
- `pymarl/src/modules/mixers/global_qmixer.py` — created in Sprint 3
- `pymarl/src/components/rsu_zone_manager.py` — created in Sprint 1

Fork `q_learner.py` into `hierarchical_q_learner.py`. The class name should be `HierarchicalQLearner`. Key changes:

### 5.1 — Constructor (`__init__`)

Instead of one mixer, create two mixers and their target copies:
```python
self.local_mixer = LocalQMixer(args)
self.target_local_mixer = copy.deepcopy(self.local_mixer)

self.global_mixer = GlobalQMixer(args)
self.target_global_mixer = copy.deepcopy(self.global_mixer)
```

Load the RSU config from the path in `args.rsu_config` and create the zone manager:
```python
self.zone_manager = RSUZoneManager(rsu_config)
```

Collect parameters from ALL three levels into a single parameter list:
```python
self.params = list(mac.parameters())
self.params += list(self.local_mixer.parameters())
self.params += list(self.global_mixer.parameters())
```

Use the same optimizer type and settings as `q_learner.py` but over the combined params.

### 5.2 — Training method (`train`)

The forward pass logic. Follow the existing `q_learner.py` structure but insert the hierarchical mixing between the MAC output and the loss computation.

**Step-by-step flow:**

1. **Get per-agent Q-values from MAC** — identical to `q_learner.py`. Run the MAC forward for each timestep, stack outputs, gather chosen action Q-values. Result: `chosen_action_qvals` of shape `(batch, T-1, n_agents)`.

2. **Retrieve pre-computed RSU data from the batch** — the episode runner (Sprint 6) will have stored these in the batch:
   - `batch["rsu_agent_qs"]`: shape `(batch, T, max_rsus, max_agents_per_rsu)` — the Q-values of agents assigned to each RSU, padded
   - `batch["agent_masks_per_rsu"]`: shape `(batch, T, max_rsus, max_agents_per_rsu)` — masks
   - `batch["local_states"]`: shape `(batch, T, max_rsus, local_state_dim)` — padded local states
   
   **IMPORTANT:** At training time, we need the Q-values *from the current network*, not the ones stored during collection (which used the behavior policy). So the pre-computed `rsu_agent_qs` from the batch are the *zone assignments and masks* — but the actual Q-values must come from the current MAC forward pass. The approach:
   - Use `batch["agent_masks_per_rsu"]` and `batch["local_states"]` as-is (these don't depend on the network).
   - For the Q-values, use `chosen_action_qvals` from step 1 and scatter them into the RSU structure using the zone assignment indices stored in the batch.
   
   Store zone assignments as `batch["zone_assignments"]`: shape `(batch, T, n_agents)` — the RSU id each agent is assigned to at each timestep. Use this to scatter the fresh Q-values into per-RSU padded tensors at training time.

3. **Run Local Mixer for each RSU** — iterate over RSU indices 0 to `max_rsus-1`. For each RSU, gather the Q-values of its assigned agents (using zone_assignments to index into chosen_action_qvals), pad to `max_agents_per_rsu`, and pass through `self.local_mixer()` along with the local state and mask. Collect the `local_qtot` scalars. Stack into shape `(batch, T-1, max_rsus)`.

4. **Run Global Mixer** — pass the stacked `local_qtots`, the global state from `batch["state"]` (padded to `global_state_dim`), and the RSU mask through `self.global_mixer()`. Output: `global_qtot` of shape `(batch, T-1, 1)`.

5. **Compute targets** — same logic as `q_learner.py` but using `self.target_local_mixer` and `self.target_global_mixer`. Run the target MAC, compute max Q-values, scatter into RSU structure, run target local mixers, run target global mixer. TD target: `r + gamma * target_global_qtot`.

6. **Loss** — identical to `q_learner.py`: squared TD error, masked by `batch["filled"]`.

7. **Backward** — `loss.backward()` flows gradients through Global Mixer → Local Mixer → MAC.

8. **Gradient clipping** — same as `q_learner.py`.

9. **Target network updates** — update all target networks on the same schedule as `q_learner.py`:
   ```python
   self.target_local_mixer.load_state_dict(self.local_mixer.state_dict())
   self.target_global_mixer.load_state_dict(self.global_mixer.state_dict())
   ```
   Plus whatever target MAC update the existing code does.

### 5.3 — Checkpoint save/load

Override or extend the save/load methods to include `local_mixer`, `global_mixer`, and their target copies. Look at how `q_learner.py` saves/loads the mixer and follow the same pattern for both new mixers.

### 5.4 — Logging

Log the same metrics as `q_learner.py` (loss, grad_norm, td_error_abs, q_taken_mean). Additionally log:
- `local_qtot_mean`: mean of local Q_tot values across RSUs
- `global_qtot_mean`: mean of global Q_tot

---

## Sprint 6 — Episode runner modifications

### File to modify:
- `pymarl/src/runners/episode_runner.py`

### Task

Read the existing `episode_runner.py` carefully first. Understand how it collects data at each timestep and stores it in the episode batch.

Add the following data collection at each timestep, **only when running in Civiq mode** (check `self.args.learner == "hierarchical_q_learner"` or a similar flag). When running vanilla QMIX, the runner should behave identically to before — do not break the existing pipeline.

At each timestep, after the environment step:

1. **Get vehicle positions** from the environment. The environment wrapper (`sumo_grid_reroute.py`) already queries TraCI for vehicle data. Add a method (or use an existing one) that returns `{vehicle_id: (x, y)}` for all active vehicles.

2. **Run zone assignment** using the `RSUZoneManager` instance (initialized once when the runner starts, from the RSU config).

3. **Build and store per-RSU data:**
   - `zone_assignments`: for each agent, which RSU id it's assigned to. Shape per timestep: `(n_agents,)`. Pad to `(max_total_agents,)` with -1 for inactive agent slots.
   - `agent_masks_per_rsu`: shape `(max_rsus, max_agents_per_rsu)`. From zone manager.
   - `local_states`: shape `(max_rsus, max_agents_per_rsu * obs_dim)`. From zone manager using per-agent observations.

4. **Store in the episode batch** using the pre-transition or post-transition data insertion pattern already used by the runner.

### Important constraints:
- The RSU zone manager must be initialized once per episode runner instantiation, not per timestep.
- The new data fields must be added to the episode batch scheme (see Sprint 7).
- Guard all new code behind a conditional so the runner still works for vanilla QMIX without any RSU config.

---

## Sprint 7 — Episode buffer scheme modifications

### File to modify:
- `pymarl/src/components/episode_buffer.py`

### Task

Read the existing `episode_buffer.py` to understand how the data scheme is defined. Add the new fields required by Civiq.

Add these fields to the scheme, guarded by a conditional (only added when running Civiq):

```python
# New fields for Civiq hierarchical data
"zone_assignments":     {"vshape": (max_total_agents,), "group": "agents", "dtype": th.long},
"agent_masks_per_rsu":  {"vshape": (max_rsus, max_agents_per_rsu), "dtype": th.float32},
"local_states":         {"vshape": (max_rsus, max_agents_per_rsu * obs_dim), "dtype": th.float32},
```

Where `max_total_agents`, `max_rsus`, `max_agents_per_rsu`, and `obs_dim` come from `args`.

**Key constraint:** The scheme modifications must not affect vanilla QMIX operation. When running with `--config=qmix_sumo`, these fields should not be allocated.

Read how the existing scheme defines fields like `"state"`, `"obs"`, and `"actions"` and follow the exact same pattern for the new fields. Pay attention to whether fields use `"group": "agents"` or not, and whether they need `"episode_const": True`.

---

## Sprint 8 — Environment wrapper additions

### File to modify:
- `pymarl/src/envs/sumo_grid_reroute.py`

### Task

Read the existing environment file. Add methods that the episode runner (Sprint 6) will call. These methods expose vehicle position data that TraCI already provides but that the environment doesn't currently return to the runner.

Add the following methods to the environment class:

1. **`get_vehicle_positions()`** — Returns a dict `{vehicle_id: (x, y)}` for all currently active vehicles. Use `traci.vehicle.getPosition(veh_id)` which returns `(x, y)` in SUMO network coordinates. The environment likely already calls this internally — expose it as a public method.

2. **`get_agent_index_mapping()`** — Returns a dict `{vehicle_id: agent_index}` mapping each active vehicle to its integer index in the agent ordering (the same ordering used by `get_obs()` and the MAC). This is needed so the zone manager can map Q-values to the correct RSU.

Read the existing `get_obs()` and `get_state()` methods to understand how vehicles are ordered (the agent index ordering). The mapping must be consistent with that ordering.

**Do NOT modify any existing methods.** Only add new methods. The environment must continue to work identically for vanilla QMIX.

---

## Sprint 9 — Main entry point registration

### File to modify:
- `pymarl/src/main.py`

### Task

Read `main.py` to see how learners and mixers are registered/loaded. Add registration so that:

1. When `args.learner == "hierarchical_q_learner"`, the `HierarchicalQLearner` class is imported and used.
2. When `args.mixer == "civiq"` (or however the config signals Civiq mode), the correct initialization path is followed.

Follow the exact pattern used for the existing `q_learner` and `qmix` mixer registration. Do not refactor the existing registration logic — just add the new entries alongside.

Also ensure that the RSU config YAML is loaded and made available in `args`. The path is specified in `civiq_sumo.yaml` as `rsu_config: "rsu/synthetic_4x4.yaml"`. Load it relative to `pymarl/src/config/` and merge its contents into `args` or attach it as `args.rsu_config_data`.

---

## Sprint 10 — Curriculum weight transfer utility

### New file to create:
- `pymarl/src/utils/curriculum_transfer.py`

### Task

Create a utility function that loads a Civiq checkpoint from a previous curriculum stage and transfers weights to a new model for the next stage.

The function should:
1. Accept a checkpoint path and a fresh (newly initialized) `HierarchicalQLearner` instance.
2. Load the saved state dicts for: MAC (agent DRQN), `local_mixer`, `global_mixer`.
3. Load them into the new learner's corresponding modules. Since all three levels use the same architecture across maps (thanks to padding/masking), the state dicts are directly compatible — no dimension adaptation needed.
4. Also load into the target network copies.
5. **Do NOT load the optimizer state** — the optimizer resets at each curriculum stage.
6. Print a summary of what was transferred (parameter counts, any mismatches).

Additionally, create a simple CLI interface:
```bash
python -m utils.curriculum_transfer \
    --checkpoint_path results/experiments/.../models/best.pth \
    --config civiq_sumo \
    --rsu_config rsu/bgc_core.yaml
```

This lets the developer verify weight compatibility between stages before starting a training run.

---

## Summary — Execution Order

```
Sprint 1:  RSU configs + zone manager class         (no dependencies)
Sprint 2:  LocalQMixer                               (reads existing qmix.py)
Sprint 3:  GlobalQMixer                              (reads existing qmix.py)
Sprint 4:  Civiq algorithm config                    (reads existing qmix_sumo.yaml)
Sprint 5:  HierarchicalQLearner                      (depends on Sprints 1-4)
Sprint 6:  Episode runner modifications              (depends on Sprints 1, 7, 8)
Sprint 7:  Episode buffer scheme modifications       (depends on Sprint 4 for config values)
Sprint 8:  Environment wrapper additions             (no dependencies on new code)
Sprint 9:  Main entry point registration             (depends on Sprints 4, 5)
Sprint 10: Curriculum weight transfer utility        (depends on Sprint 5)
```

**Suggested batch groupings if Claude Code can do parallel tasks:**
- **Batch A (independent):** Sprints 1, 2, 3, 4, 8 — these don't depend on each other
- **Batch B (depends on A):** Sprints 5, 7
- **Batch C (depends on B):** Sprints 6, 9, 10
