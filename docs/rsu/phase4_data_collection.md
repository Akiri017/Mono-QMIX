# Phase 4 — Data Collection (Episode Runner + Replay Buffer)

## Purpose

Phase 4 wires the Civiq batch fields that `HierarchicalQLearner.train()` has been waiting for since Phase 3. The episode runner now collects per-zone observation tensors at every timestep and stores them in the replay buffer alongside the existing state/obs/actions fields.

`rsu_agent_qs` is intentionally NOT populated in the runner — it is computed inside `HierarchicalQLearner.train()` by slicing `chosen_action_qvals` according to `zone_assignments`.

---

## Files Created / Modified

| File | Action | Description |
|------|--------|-------------|
| `pymarl/src/envs/sumo_grid_reroute.py` | Updated | Added zone_manager, 4 new Civiq observation methods |
| `pymarl/src/runners/episode_runner.py` | Updated | Zone data collection at every timestep; civiq key forwarding in `_make_env` |
| `pymarl/src/main.py` | Updated | `get_scheme()` adds Civiq fields when `mixer == "civiq"` |
| `pymarl/src/config/algs/civiq_sumo.yaml` | Updated | `global_state_dim` corrected to 2080 |
| `tests/test_phase4_data_collection.py` | Created | Phase 4 gate — 4-test assertion suite |

---

## Environment Changes (`sumo_grid_reroute.py`)

### Zone manager instantiation

`RSUZoneManager` is now instantiated in `__init__` when `rsu_config` is present in `env_args`. If the key is absent (plain QMIX run without civiq), `self.zone_manager = None` and all new methods are skipped by the runner.

```python
self.zone_manager = None
_rsu_cfg_path = env_args.get("rsu_config", None)
if _rsu_cfg_path is not None:
    _rsu_cfg_path = self._resolve_path(_rsu_cfg_path)
    with open(_rsu_cfg_path) as _f:
        _rsu_cfg_dict = yaml.safe_load(_f)
    self.zone_manager = RSUZoneManager(_rsu_cfg_dict)
```

### New methods

| Method | Returns | Shape |
|--------|---------|-------|
| `get_zone_assignments()` | `{rsu_id: [vehicle_ids]}` for ALL vehicles | dict |
| `get_local_obs_padded(zone)` | padded per-RSU agent observations | `(max_rsus, max_agents_per_rsu * obs_dim)` |
| `get_agent_masks_padded(zone)` | 1.0=real, 0.0=padding | `(max_rsus, max_agents_per_rsu)` |
| `get_zone_assignments_flat(zone)` | RSU id per agent slot, −1=unassigned | `(n_agents,)` int32 |

**TraCI call pattern** — identical to existing env usage:
```python
veh_ids = traci.vehicle.getIDList()
positions = {v: traci.vehicle.getPosition(v) for v in veh_ids}
```
`traci` here is the `sumo_backend` proxy, which works identically for both libsumo and TraCI backends.

**Zone → agent mapping**: `get_zone_assignments()` returns all vehicles (controlled + background). The env's `agent_vehicle_ids` list (index = agent slot, value = SUMO vehicle ID) is used to reverse-map from vehicle IDs back to agent slot indices. Background vehicles are ignored.

---

## Runner Changes (`episode_runner.py`)

### `_make_env` fix

The runner previously passed `args["env_args"]` (the nested env YAML section) directly to the env, excluding Civiq keys like `rsu_config` that live in the alg config. Fixed by forwarding them explicitly:

```python
for key in ("rsu_config", "max_rsus", "max_agents_per_rsu", "obs_dim"):
    if key in self.args and key not in env_args:
        env_args[key] = self.args[key]
```

### Zone data collection

At each timestep (both ts=0 and ts=t+1 inside the episode loop), when `zone_manager` is present:

```python
_civiq = hasattr(self.env, "zone_manager") and self.env.zone_manager is not None
if _civiq:
    _zone = self.env.get_zone_assignments()
    pre_transition_data["local_states"]          = [self.env.get_local_obs_padded(_zone)]
    pre_transition_data["agent_masks_per_rsu"]   = [self.env.get_agent_masks_padded(_zone)]
    pre_transition_data["zone_assignments"]      = [self.env.get_zone_assignments_flat(_zone)]
    # rsu_agent_qs NOT populated — computed in HierarchicalQLearner.train()
```

---

## Scheme Changes (`main.py`)

`get_scheme(env_info, args)` adds Civiq fields when `args["mixer"] == "civiq"`:

| Field | Shape | dtype | Notes |
|-------|-------|-------|-------|
| `zone_assignments` | `(n_agents,)` | int32 | RSU id per agent slot, −1=unassigned |
| `rsu_agent_qs` | `(max_rsus, max_agents_per_rsu)` | float32 | Zeroes in buffer; filled in train() |
| `agent_masks_per_rsu` | `(max_rsus, max_agents_per_rsu)` | float32 | 1.0=real, 0.0=padding |
| `local_states` | `(max_rsus, max_agents_per_rsu * obs_dim)` | float32 | Per-RSU concatenated agent obs |

**`global_states` is NOT a separate field.** `batch["state"]` (shape `state_dim = n_agents × obs_dim = 2080`) serves as the GlobalQMixer input. `global_state_dim` has been updated to 2080 in `civiq_sumo.yaml`.

---

## `global_state_dim` Correction

| Previous value | Corrected value | Rationale |
|----------------|-----------------|-----------|
| 4485 (69 × 65) | **2080 (32 × 65)** | `batch["state"]` = concatenated controlled-agent observations; `state_dim = n_agents × obs_dim` |

The earlier 4485 assumed all 69 peak-network vehicles have observations, but only the 32 controlled agents have the 65-dim observation vector. GlobalQMixer input = `batch["state"]` with no duplication.

---

## RAM Budget

`episode_limit = 1000`, so T+1 = 1001 timesteps per episode stored.

| Field | Per-episode | ×400 |
|-------|------------|------|
| `local_states` (dominant) | 83.4 MB | 33.4 GB |
| `rsu_agent_qs` | 1.3 MB | 0.5 GB |
| `agent_masks_per_rsu` | 1.3 MB | 0.5 GB |
| `zone_assignments` | 0.1 MB | 0.04 GB |
| Existing fields | 17.4 MB | 7.0 GB |
| **Total** | **~103.5 MB** | **~41.4 GB** |

**Recommendation: reduce `buffer_size` from 400 → 100** in `civiq_sumo.yaml`.

| buffer_size | Total RAM | Verdict |
|-------------|-----------|---------|
| 400 | ~41 GB | Exceeds 14 GB usable |
| **100** | **~10.4 GB** | Fits comfortably |
| 64 | ~6.6 GB | Safe with margin |

Tradeoff: 4× less experience diversity per training batch at buffer_size=100 vs 400.

---

## Phase 4 Gate Results

```
PHASE 4 GATE PASSED
```

Episode: 50 steps, `train_med.sumocfg`, libsumo backend.

| Test | Description | Result |
|------|-------------|--------|
| 1 | `zone_assignments` shape `(1,51,32)`, dtype int32, range `[-1, 11]` | PASSED |
| 2 | `local_states` shape `(1,51,12,1820)`, no NaN | PASSED |
| 3 | `agent_masks_per_rsu` binary, max_per_rsu=14 across all timesteps | PASSED |
| 4 | Conservation: sum(masks)==assigned agents, mean 26.6 agents/step | PASSED |

Mean 26.6 of 32 agent slots actively assigned to RSU zones at each step — consistent with a LOS B scenario where most (but not all) controlled vehicles are active.

---

## Next Step

Phase 5 — **Complete `HierarchicalQLearner.train()`**: replace the `NotImplementedError` stub with the full hierarchical mixing forward pass using the batch fields now available.
