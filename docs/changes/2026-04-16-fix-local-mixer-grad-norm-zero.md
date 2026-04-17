# Debug Session + Hotfix: Restore gradient flow to LocalQMixer and MAC on BGC Full map

**Date:** 2026-04-16
**Status:** Partial — gradient path restored; gradient scale imbalance deferred
**Affected areas:**
- `pymarl/src/envs/sumo_grid_reroute.py`
- `pymarl/src/learners/hierarchical_q_learner.py`

---

## Context

During training on the BGC Full map with the Civiq hierarchical architecture,
`local_mixer_grad_norm` and `agent_grad_norm` were consistently logging `0.0000`
while `global_mixer_grad_norm` was non-zero (~1398). This meant the LocalQMixer
and MAC parameters were receiving no learning signal — the hierarchical structure
was effectively not training, only the GlobalQMixer's state-independent output
paths (bias, V, w_final) were fitting.

---

## Discussion and decisions

Three possible root causes were identified through code inspection and diagnostic logging:

1. **`rsu_mask` all-zeros in replay buffer** — if `agent_masks_per_rsu` was zero-filled,
   `rsu_mask = (agent_masks.sum(-1) > 0).float()` would be all-zero, multiplying out
   `local_qtots` in `GlobalQMixer.forward` and severing all gradient paths downstream.

2. **Warmup loop bug** — the original warmup in `reset()` exited as soon as
   `agent_active = True` (set when a vehicle is *scheduled*), not when it physically
   entered the SUMO network. At ts=0 of each episode, no controlled vehicles were in
   `traci.vehicle.getIDList()`, so `get_agent_masks_padded` returned all-zeros.

3. **`scatter_add_` on a non-grad leaf tensor** — `_build_rsu_agent_qs` used
   `out = torch.zeros(...); out[:,:,r,:].scatter_add_(...)`. PyTorch in-place scatter
   on a `requires_grad=False` leaf does not propagate gradients from `src` to `out`.
   So `rsu_agent_qs.requires_grad = False`, severing the gradient path from
   `local_qtots → rsu_agent_qs → chosen_action_qvals → MAC params`.

Root cause 1 was confirmed via a `rsu_mask_mean` diagnostic log (`0.6532` on BGC Full —
masks were non-zero, so this was not the primary cause). Root causes 2 and 3 were
both real bugs and were fixed. A precision diagnostic (`{:.6e}`) confirmed the
gradient path was intact after fixes: `local_mixer_grad_norm = 9.84e-06`,
`agent_grad_norm = 1.13e-06` — grads exist but are ~142 million× smaller than
`global_mixer_grad_norm` (1398) due to a scale imbalance, not a broken path.

---

## What changed

### `pymarl/src/envs/sumo_grid_reroute.py`

- **Added `agent_last_rsu` field** (`__init__` and `reset()`) — caches the last
  confirmed RSU assignment per agent slot. Used as a fallback when a replacement
  vehicle has been scheduled via `traci.vehicle.add()` but hasn't yet entered the
  SUMO network (absent from `getIDList()`).

- **Fixed warmup loop in `reset()`** — replaced `if _count_active_agents() >= n_agents`
  (tests `agent_active`, set at scheduling time) with
  `if active_vids.issubset(traci.vehicle.getIDList())` (tests physical SUMO entry).
  Ensures `get_zone_assignments()` at ts=0 finds all controlled vehicles, producing
  non-zero `agent_masks_per_rsu` from the first timestep.

- **Added fallback in `_build_agents_per_rsu()`** — when a vehicle is active but
  not yet in SUMO, falls back to `agent_last_rsu[agent_id]` so the RSU slot
  remains masked as occupied. Prevents mid-episode replacement transitions from
  producing all-zero masks.

- **Added matching fallback in `get_zone_assignments_flat()`** — keeps
  `zone_assignments` and `agent_masks_per_rsu` consistent during replacement
  transitions. Without this, the mask would say "agent in RSU X" but
  `zone_assignments` would return -1, causing `rsu_agent_qs` to contribute 0
  Q-value for that slot despite the mask being 1.

### `pymarl/src/learners/hierarchical_q_learner.py`

- **Fixed `_build_rsu_agent_qs()` to use out-of-place `scatter_add`** — replaced
  in-place `out[:,:,r,:].scatter_add_(...)` (which does not propagate gradients
  from `src` through a non-grad leaf) with `base.scatter_add(dim, index, src)` per
  RSU, collected into `out_list` and stacked. This restores the gradient path from
  `rsu_agent_qs → chosen_action_qvals → MAC params`. Small memory overhead:
  `out_list` holds R separate `(B, T, A_rsu)` tensors before stacking (~2× the
  output tensor size at peak), plus `slot_idx_all` / `in_rsu_all` are retained
  for backward.

- **Added `rsu_mask_mean` and `rsu_mask_nonzero_frac` diagnostic stats** — logged
  once per `log_interval` to verify `agent_masks_per_rsu` is being populated in
  the replay buffer. Confirmed non-zero on both 4x4 and BGC Full.

- **Added precision grad diagnostic** (`[GRAD DIAG]` print) — logs
  `local_mixer_has_grad`, `local_mixer_grad_norm` in `{:.6e}` format, and same
  for agent. Used to distinguish "grad path broken (None)" from "grad exists but
  tiny". Confirmed grad path intact on BGC Full: `9.84e-06` and `1.13e-06`.

---

## Implementation notes

- The out-of-place `scatter_add` fix changes memory behaviour during backward:
  `slot_idx_all` (shape `(B, T, n_agents, max_rsus)`) and `in_rsu_all` (same shape)
  must now be retained until backward completes, since `scatter_add` backward uses
  index for a gather. For BGC Full dims this adds ~10–15 MB of retained tensors.

- The `agent_last_rsu` cache is reset to -1 at `reset()`. It is populated at the
  first `get_agent_masks_padded()` call of each episode (after the warmup loop
  guarantees vehicles are in SUMO), so the fallback is always available from ts=1
  onwards for mid-episode replacements.

- The warmup loop has a hard cap of 50 simulation steps. If vehicles fail to enter
  within 50 steps (e.g., route scheduling failure), the loop exits and ts=0 masks
  may still be partially zero for those slots. This is acceptable and covered by
  the fallback in subsequent timesteps.

---

## Status and follow-up

**Status:** Partial — gradient path confirmed working; training scale issue unresolved.

**Remaining problem:** On BGC Full, `global_mixer_grad_norm ≈ 1398` while
`local_mixer_grad_norm ≈ 9.84e-6` — a ~142 million× imbalance. The GlobalQMixer
receives strong gradient signal through its state-independent paths (bias `b1`,
output `V`, `w_final`), while the backward signal reaching LocalQMixer is
proportional to `w1 = abs(hyper_w_1(global_states))`, which is weak on BGC Full
due to the large, unnormalized `global_state_dim = 24032` (raw SUMO obs values).

**Follow-up items:**
- Implement global state normalization via EMA (same pattern as reward normalization)
  before `global_states` enters `GlobalQMixer` — principled fix, map-invariant.
  Requires adding EMA state to `save_models`/`load_models`.
- OR: split optimizer into two param groups (local_mixer + MAC at higher lr,
  global_mixer at lower lr) — faster to implement, compensates the symptom without
  fixing the root cause.
- Remove `[GRAD DIAG]` print once the scale issue is resolved.
- Remove `rsu_mask_mean` / `rsu_mask_nonzero_frac` diagnostic stats once
  confirmed stable across maps and seeds.
- Consider PopArt on the output of `GlobalQMixer` for long runs (>200k steps)
  as reward scale grows with improving policy.
