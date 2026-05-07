# Phase 1 — LocalQMixer

## Purpose

Phase 1 implements the **LocalQMixer**: the per-RSU mixing network at level 2 of the Civiq hierarchy. Each RSU zone runs one LocalQMixer instance that aggregates the Q-values of its assigned vehicles into a local `Q_tot`. Weights are **shared across all RSU zones** — a single `LocalQMixer` object is applied to every zone with different inputs at each timestep.

---

## Files Created / Modified

| File | Action | Description |
|------|--------|-------------|
| `pymarl/src/modules/mixers/local_qmixer.py` | Verified (pre-existing) | LocalQMixer class |
| `pymarl/src/modules/mixers/__init__.py` | Updated | Added `LocalQMixer`, `GlobalQMixer` exports |
| `pymarl/src/config/algs/civiq_sumo.yaml` | Created | Civiq algorithm config (based on qmix_sumo.yaml) |
| `tests/test_local_qmixer.py` | Created | Phase 1 gate — 4-test assertion suite |

---

## Design: QMixer Relationship

LocalQMixer mirrors QMixer's structure exactly, with three differences:

| Dimension | QMixer | LocalQMixer |
|-----------|--------|-------------|
| Agent count | `n_agents` (32) | `max_agents_per_rsu` (28) |
| State input | global state `(batch, state_shape)` | local state `(batch, max_agents * obs_dim)` |
| Extra input | — | `agent_mask (batch, max_agents)` |

The mask is applied **before** the first `bmm`:

```python
agent_qs = agent_qs * agent_mask   # zero out padded slots
agent_qs = agent_qs.view(batch, 1, max_agents)
hidden   = F.elu(torch.bmm(agent_qs, w1) + b1)
```

This ensures padded vehicle slots contribute zero to the monotonic mix. Bias terms (`b1`, `V`) are still conditioned on `local_states`, so an all-zero mask (empty zone) produces a non-zero but finite output — no NaN risk.

---

## Hypernetwork Structure

Identical to QMixer (`hypernet_layers=2` default):

```
local_states (batch, 1820)
      │
      ├─▶ hyper_w_1    → Linear(1820,64) → ReLU → Linear(64, 28*32)  → abs → w1   (batch,28,32)
      ├─▶ hyper_b_1    → Linear(1820,64) → ReLU → Linear(64, 32)            → b1   (batch,1,32)
      ├─▶ hyper_w_final → Linear(1820,64) → ReLU → Linear(64, 32)   → abs → wf   (batch,32,1)
      └─▶ V            → Linear(1820,64) → ReLU → Linear(64, 1)            → v    (batch,1,1)

hidden = ELU( agent_qs @ w1 + b1 )   # (batch,1,32)
q_tot  = hidden @ wf + v              # (batch,1,1) → (batch,1)
```

Local state dimension: `28 × 65 = 1820`

---

## Config Parameters (`civiq_sumo.yaml`)

Only Civiq-specific additions — all other hyperparameters are identical to `qmix_sumo.yaml`:

```yaml
local_mixing_embed_dim: 32      # embed dim for LocalQMixer
max_agents_per_rsu: 28          # LOS E validated (peak 23 + buffer 5)
obs_dim: 65                     # per-agent observation size

global_mixing_embed_dim: 32     # embed dim for GlobalQMixer (Phase 2)
max_rsus: 12                    # TEMPORARY — update to BGC Full count

rsu_config: "config/envs/rsu/synthetic_4x4.yaml"
```

---

## Phase 1 Gate Results

```
PHASE 1 GATE PASSED
```

| Test | Description | Result |
|------|-------------|--------|
| 1 | Full batch, all 28 agents real — shape `(4,1)`, no NaN/Inf | PASSED |
| 2 | Partial batch, 10/28 real agents, 18 padded — shape `(4,1)`, no NaN | PASSED |
| 3 | Empty zone (all-zero mask) — shape `(4,1)`, no NaN, output ≠ 0 (bias active) | PASSED |
| 4 | Gradient flow — `agent_qs.grad` present, all param grads non-NaN | PASSED |

Empty-zone output sample: `-0.2392` — confirms bias terms (`b1`, `V`) fire correctly even when no agents are active.

---

## Next Step

Phase 2 — **GlobalQMixer integration**: wire `LocalQMixer` and `GlobalQMixer` together in a `CiviqLearner` that applies `LocalQMixer` per zone, collects `local_qtots`, then runs `GlobalQMixer` to produce the final `Q_tot`.
