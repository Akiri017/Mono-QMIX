# Civiq Scaffold — Sprint 2: Local Mixer (Level 2)

**Status:** Complete  
**Date:** 2026-04-03  
**File created:** `pymarl/src/modules/mixers/local_qmixer.py`

---

## What was implemented

`LocalQMixer` — a QMIX-style mixing network that operates at the RSU zone level (Level 2 of the Civiq hierarchy). It aggregates Q-values from vehicles assigned to a single RSU zone into a local Q_tot scalar.

### Key design decisions

| Decision | Detail |
|----------|--------|
| **Forked from** | `pymarl/src/modules/mixers/qmix.py` (not modified) |
| **Input dimensions** | `max_agents = max_agents_per_rsu` (30), `local_state_dim = max_agents_per_rsu * obs_dim` (30 × 65 = 1950) |
| **Embed dim** | Controlled by `local_mixing_embed_dim` (default 32) |
| **Hypernet structure** | Identical to QMixer — configurable 1 or 2 layer hypernets via `hypernet_layers` (default 2), with `hypernet_embed` (default 64) |
| **Weight sharing** | One `LocalQMixer` instance is applied to every RSU zone with different inputs — same weights, different per-zone data |
| **Masking** | `agent_qs = agent_qs * agent_mask` applied before the first matrix multiply, zeroing padded agent Q-values |
| **Monotonicity** | Preserved via `torch.abs()` on hypernetwork-generated weights (same as QMixer) |
| **Activations** | ELU on hidden layer, unconstrained bias on first layer, V(s) state-dependent bias on final layer |

### Forward signature

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

### Required config keys (from `civiq_sumo.yaml`, Sprint 4)

- `max_agents_per_rsu`: int (placeholder 30)
- `obs_dim`: int (65)
- `local_mixing_embed_dim`: int (default 32)
- `hypernet_layers`: int (default 2)
- `hypernet_embed`: int (default 64)

---

## Verification

Ran the following checks:

1. **Instantiation** — `LocalQMixer(args)` creates successfully with default config. Total parameters: **566,081**.
2. **Forward pass** — Produces output shape `(batch, 1)` as expected.
3. **Masking correctness** — Changing Q-values in padded agent slots (where `agent_mask=0`) does not change the output. Confirmed via `torch.allclose`.

---

## Dependencies

- **Upstream:** Reads existing `qmix.py` for reference (not modified).
- **Downstream:** Used by `HierarchicalQLearner` (Sprint 5), which iterates over RSU zones and calls `self.local_mixer.forward()` for each.
