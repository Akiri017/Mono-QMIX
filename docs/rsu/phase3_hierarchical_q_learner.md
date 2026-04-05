# Phase 3 — HierarchicalQLearner

## Purpose

Phase 3 implements the **HierarchicalQLearner**: a fork of QLearner that wires LocalQMixer (Level 2) and GlobalQMixer (Level 3) together with a single end-to-end TD loss. It replaces the single-mixer QLearner as the training engine for Civiq.

The MAC forward pass, double-Q logic, TD loss structure, gradient clipping, target network updates, and save/load patterns are **identical to QLearner** — only the mixer section changes.

The mixing forward pass in `train()` is stubbed with `NotImplementedError` until Phase 4 populates the required batch fields from the episode runner.

---

## Files Created / Modified

| File | Action | Description |
|------|--------|-------------|
| `pymarl/src/learners/hierarchical_q_learner.py` | Created | HierarchicalQLearner class |
| `pymarl/src/learners/__init__.py` | Updated | Added `HierarchicalQLearner` export |
| `tests/test_hierarchical_learner.py` | Created | Phase 3 gate — 4-test assertion suite |

---

## Design: What Changes from QLearner

| Aspect | QLearner | HierarchicalQLearner |
|--------|----------|----------------------|
| Mixer | `self.mixer` (QMixer) | `self.local_mixer` (LocalQMixer) + `self.global_mixer` (GlobalQMixer) |
| Target mixer | `self.target_mixer` | `self.target_local_mixer` + `self.target_global_mixer` |
| Optimizer params | MAC + mixer | MAC + local_mixer + global_mixer |
| Zone manager | — | `self.zone_manager` (RSUZoneManager, loaded from `args["rsu_config"]`) |
| Mixing forward | single `self.mixer(qs, states)` | local then global (stubbed — Phase 4) |
| Save/load | `mixer.pth` | `local_mixer.th` + `global_mixer.th` |

Everything else — `_update_targets()`, `cuda/cpu`, optimizer settings, TD loss formula, mask handling, logging — is identical to QLearner.

---

## train() Status

The MAC forward pass and double-Q target computation are fully implemented. The mixer section raises `NotImplementedError` with a complete TODO block showing the Phase 4 implementation:

```python
# TODO (Phase 4): requires these batch fields from the episode runner:
#   batch["rsu_agent_qs"]         (batch, T, max_rsus, max_agents_per_rsu)
#   batch["agent_masks_per_rsu"]  (batch, T, max_rsus, max_agents_per_rsu)
#   batch["local_states"]         (batch, T, max_rsus, max_agents_per_rsu * obs_dim)
#   batch["zone_assignments"]     (batch, T, n_agents)
#   batch["global_states"]        (batch, T+1, global_state_dim)
```

When Phase 4 lands, the mixing forward will be:

```python
BT = batch_size * max_t
R  = max_rsus

# Online path
local_qtots = self.local_mixer(
    rsu_agent_qs.view(BT * R, max_agents_per_rsu),
    local_states.view(BT * R, -1),
    agent_masks.view(BT * R, max_agents_per_rsu)
).view(BT, R)

q_tot = self.global_mixer(
    local_qtots,
    global_states.view(BT, -1),
    rsu_mask.view(BT, R)
).view(batch_size, max_t, 1)

# Target path — same reshape, use target mixers
```

---

## Constructor Args

All read from the same `args` dict as QLearner, plus Civiq-specific keys:

| Key | Value | Source |
|-----|-------|--------|
| `max_agents_per_rsu` | 28 | civiq_sumo.yaml |
| `obs_dim` | 65 | civiq_sumo.yaml |
| `local_mixing_embed_dim` | 32 | civiq_sumo.yaml |
| `max_rsus` | 12 | civiq_sumo.yaml |
| `global_state_dim` | 4485 | civiq_sumo.yaml (LOS E validated) |
| `global_mixing_embed_dim` | 32 | civiq_sumo.yaml |
| `rsu_config` | `"config/envs/rsu/synthetic_4x4.yaml"` | civiq_sumo.yaml |

---

## Parameter Counts (synthetic 4×4 map)

| Component | Parameters |
|-----------|------------|
| MAC (StubAgent in test) | 110 |
| LocalQMixer | 528,641 |
| GlobalQMixer | 1,177,601 |
| **Total in optimizer** | **1,706,352** |

---

## Phase 3 Gate Results

```
PHASE 3 GATE PASSED
```

| Test | Description | Result |
|------|-------------|--------|
| 1 | Construction — all components present (mixers, targets, zone_manager, params) | PASSED |
| 2 | Parameter grouping — MAC, LocalQMixer, GlobalQMixer all in `self.params` | PASSED |
| 3 | Target independence — modifying mixer weights does not change target copies | PASSED |
| 4 | Save/load round-trip — `local_mixer.th` and `global_mixer.th` weights match after reload | PASSED |

---

## Next Step

Phase 4 — **Episode runner / replay buffer changes**: populate `rsu_agent_qs`, `agent_masks_per_rsu`, `local_states`, `zone_assignments`, and `global_states` in the batch at each timestep, then complete the `train()` mixing section in HierarchicalQLearner.
