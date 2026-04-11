# Phase 2 — GlobalQMixer

## Purpose

Phase 2 implements the **GlobalQMixer**: the top-level mixing network of the Civiq hierarchy. It operates one level above LocalQMixer — its "agents" are RSU zones, not individual vehicles. It takes the `local_Q_tot` scalar output from each RSU's LocalQMixer, mixes them using the global state, and produces a single global `Q_tot`.

Weights are **shared across training** — a single `GlobalQMixer` instance is applied to all RSU zone outputs at each timestep.

---

## Files Created / Modified

| File | Action | Description |
|------|--------|-------------|
| `pymarl/src/modules/mixers/global_qmixer.py` | Verified (pre-existing) | GlobalQMixer class |
| `pymarl/src/config/algs/civiq_sumo.yaml` | Updated | Added `global_state_dim` key |
| `tests/test_global_qmixer.py` | Created | Phase 2 gate — 4-test assertion suite |

---

## Design: Mixer Hierarchy

| Level | Mixer | Input | Output |
|-------|-------|-------|--------|
| 2 | LocalQMixer | Vehicle Q-values per RSU zone `(batch, max_agents_per_rsu)` | `local_Q_tot` per zone `(batch, 1)` |
| 3 | GlobalQMixer | RSU-level Q_tots `(batch, max_rsus)` | `global_Q_tot` `(batch, 1)` |

GlobalQMixer mirrors LocalQMixer's structure exactly. The only dimensional difference is that `max_rsus` replaces `max_agents_per_rsu` as the padded agent dimension.

| Dimension | LocalQMixer | GlobalQMixer |
|-----------|-------------|--------------|
| Agent count | `max_agents_per_rsu` (28) | `max_rsus` (12) |
| State input | local state `(batch, max_agents * obs_dim)` | global state `(batch, global_state_dim)` |
| Extra input | `agent_mask (batch, max_agents)` | `rsu_mask (batch, max_rsus)` |

The mask is applied **before** the first `bmm`:

```python
rsu_qtots = rsu_qtots * rsu_mask   # zero out padded RSU slots
rsu_qtots = rsu_qtots.view(batch, 1, max_rsus)
hidden    = F.elu(torch.bmm(rsu_qtots, w1) + b1)
```

An all-zero mask (no active RSUs) produces a non-zero but finite output — bias terms (`b1`, `V`) still fire. No NaN risk.

---

## Hypernetwork Structure

Identical pattern to QMixer and LocalQMixer (`hypernet_layers=2` default):

```
global_states (batch, 4485)
      │
      ├─▶ hyper_w_1     → Linear(4485,64) → ReLU → Linear(64, 12*32)  → abs → w1   (batch,12,32)
      ├─▶ hyper_b_1     → Linear(4485,64) → ReLU → Linear(64, 32)            → b1   (batch,1,32)
      ├─▶ hyper_w_final → Linear(4485,64) → ReLU → Linear(64, 32)    → abs → wf   (batch,32,1)
      └─▶ V             → Linear(4485,64) → ReLU → Linear(64, 1)            → v    (batch,1,1)

hidden      = ELU( rsu_qtots @ w1 + b1 )   # (batch,1,32)
global_qtot = hidden @ wf + v               # (batch,1,1) → (batch,1)
```

Global state dimension: `4485` — LOS E validated (`peak_total_agents=69` × `obs_dim=65`).

---

## Config Parameters (`civiq_sumo.yaml`)

Civiq-specific keys after Phase 2:

```yaml
local_mixing_embed_dim: 32      # embed dim for LocalQMixer
max_agents_per_rsu: 28          # LOS E validated (peak 23 + buffer 5)
obs_dim: 65                     # per-agent observation size

global_mixing_embed_dim: 32     # embed dim for GlobalQMixer
max_rsus: 12                    # TEMPORARY — update to BGC Full node count
global_state_dim: 4485          # peak_total_agents (69) * obs_dim (65) — LOS E validated

rsu_config: "config/envs/rsu/synthetic_4x4.yaml"
```

### global_state_dim

`global_state_dim = 4485` = `peak_total_agents (69) × obs_dim (65)`.

Determined by adding a network-wide peak tracker to `los_e_zone_validation.py` and running it against `train_high.sumocfg` (LOS E, 3600 steps). The peak simultaneous vehicle count across all 12 RSU zones was **69** at step ~1800 (63 vehicles visible at the 1800-step log; actual peak captured per-step between log points).

---

## Phase 2 Gate Results

```
PHASE 2 GATE PASSED
```

| Test | Description | Result |
|------|-------------|--------|
| 1 | Full batch, all 12 RSUs active — shape `(4,1)`, no NaN/Inf | PASSED |
| 2 | Partial batch, 8/12 active RSUs, 4 padded — shape `(4,1)`, no NaN | PASSED |
| 3 | Empty network (all-zero mask) — shape `(4,1)`, no NaN, output ≠ 0 (bias active) | PASSED |
| 4 | Gradient flow — `rsu_qtots.grad` present, all param grads non-NaN | PASSED |

Empty-network output sample: `0.2360` — confirms bias terms (`b1`, `V`) fire correctly even when no RSUs are active.

---

## Next Step

Phase 3 — **HierarchicalQLearner**: wire `LocalQMixer` and `GlobalQMixer` together in a `CiviqLearner` that applies `LocalQMixer` per zone, collects `local_qtots`, then runs `GlobalQMixer` to produce the final global `Q_tot`.
