# Civiq Scaffold — Sprint 3: Global Mixer (Level 3)

**Status:** Complete  
**Date:** 2026-04-03  
**File created:** `pymarl/src/modules/mixers/global_qmixer.py`

---

## What was implemented

`GlobalQMixer` — a QMIX-style mixing network that operates at the top level (Level 3 of the Civiq hierarchy). It aggregates local Q_tot values from all RSU zones into a single global Q_tot scalar.

### Key design decisions

| Decision | Detail |
|----------|--------|
| **Forked from** | `pymarl/src/modules/mixers/qmix.py` (not modified) |
| **Input dimensions** | `max_rsus` (16) replaces `n_agents`; `global_state_dim` (6500 = 100 agents × 65 obs_dim) replaces `state_dim` |
| **Embed dim** | Controlled by `global_mixing_embed_dim` (default 32) |
| **Hypernet structure** | Identical to QMixer — configurable 1 or 2 layer hypernets via `hypernet_layers` (default 2), with `hypernet_embed` (default 64) |
| **Masking** | `rsu_qtots = rsu_qtots * rsu_mask` applied before the first matrix multiply, zeroing padded RSU slots |
| **Monotonicity** | Preserved via `torch.abs()` on hypernetwork-generated weights (same as QMixer) |
| **Activations** | ELU on hidden layer, unconstrained bias on first layer, V(s) state-dependent bias on final layer |

### Forward signature

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

### Required config keys (from `civiq_sumo.yaml`, Sprint 4)

- `max_rsus`: int (placeholder 16)
- `global_state_dim`: int (placeholder 6500)
- `global_mixing_embed_dim`: int (default 32)
- `hypernet_layers`: int (default 2)
- `hypernet_embed`: int (default 64)

---

## Verification

Ran the following checks:

1. **Instantiation** — `GlobalQMixer(args)` creates successfully with default config. Total parameters: **1,701,761**.
2. **Forward pass** — Produces output shape `(batch, 1)` as expected.
3. **Masking correctness** — Changing Q_tot values in padded RSU slots (where `rsu_mask=0`) does not change the output. Confirmed via `torch.allclose`.

---

## Dependencies

- **Upstream:** Reads existing `qmix.py` for reference (not modified). Receives local Q_tot values produced by `LocalQMixer` (Sprint 2).
- **Downstream:** Used by `HierarchicalQLearner` (Sprint 5), which passes stacked local Q_tots through `self.global_mixer.forward()`.
