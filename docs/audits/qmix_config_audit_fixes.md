# QMIX Config Audit & Fixes

**Date:** 2026-03-28
**Context:** Pre-500k training run audit. The QMIX training pipeline was checked against the canonical QMIX paper (Rashid et al., 2018) checklist. Five deviations were identified; three were intentional and documented, two were corrected.

---

## 1. Background

A full pipeline audit was performed across the five canonical QMIX components: agent network (DRQN), mixing network, replay buffer, loss & optimization, and exploration. The audit output was saved to `docs/qmix_pipeline_audit.md`.

The audit identified five deviations from paper defaults:

| # | Deviation | Intentional? | Action |
|---|-----------|--------------|--------|
| 1 | Agent input: obs only (no last action, no agent ID) | Yes | Documented only |
| 2 | Replay buffer: 200 episodes vs. 5000 | Yes — memory | Resized to 400 |
| 3 | Optimizer: Adam vs. RMSprop | Yes | Dead keys cleaned |
| 4 | Target update: per gradient step vs. per episode | No | Fixed |
| 5 | ε anneal: 500k vs. 50k steps | Yes — complex env | Documented only |

---

## 2. Target Network Update — Fixed

### 2.1 Problem

The target network update counter was tracking **gradient steps**, not episodes. `self.training_steps` incremented by 1 on every `train()` call (every gradient update). With `batch_size=32` and `target_update_interval=200`, this meant the target network updated every 200 gradient steps — roughly every 6,400 environment steps — not every 200 episodes as intended.

### 2.2 Fix

**File:** `pymarl/src/learners/q_learner.py`

Replaced the step-based counter with `episode_num`, which is already passed in as the third argument to `train()` from `main.py`. No call site changes were needed.

**Before:**
```python
# __init__
self.training_steps = 0
self.last_target_update = 0

# train()
self.training_steps += 1
if self.training_steps - self.last_target_update >= self.target_update_interval:
    self._update_targets()
    self.last_target_update = self.training_steps
```

**After:**
```python
# __init__
self.last_target_update_episode = 0

# train()
if episode_num - self.last_target_update_episode >= self.target_update_interval:
    self._update_targets()
    self.last_target_update_episode = episode_num
```

`self.training_steps` is fully removed — no dead variables remain.

### 2.3 Effect

Target network now hard-updates every 200 episodes, matching the intended semantics. With `episode_limit=1000` steps per episode, this is one update every 200,000 environment steps.

---

## 3. Replay Buffer Size — Resized

### 3.1 Memory Footprint

The buffer was at 200 episodes. To understand the safe ceiling, the per-episode footprint was calculated from the actual tensor scheme in `main.py`:

| Field | Shape | dtype | Bytes/episode |
|-------|-------|-------|---------------|
| obs | (1001, 32, 65) | float32 | 8,328,320 |
| state | (1001, 2080) | float32 | 8,328,320 |
| actions | (1001, 32, 1) | int64 | 256,256 |
| avail_actions | (1001, 32, 4) | int32 | 512,512 |
| reward | (1001, 1) | float32 | 4,004 |
| terminated | (1001, 1) | uint8 | 1,001 |
| filled | (1001, 1) | uint8 | 1,001 |
| **Total** | | | **~16.6 MB** |

`max_seq_length = episode_limit + 1 = 1001` (extra timestep for s_{t+1} in TD target).

**Size at various capacities:**

| Episodes | RAM |
|----------|-----|
| 200 | ~3.2 GB |
| 400 | ~6.5 GB |
| 500 | ~8.1 GB |
| 5000 | ~81 GB |

Note: the previous inline comment claimed `5000 ≈ 41GB` — this was wrong by ~2×.

### 3.2 Change

**File:** `pymarl/src/config/algs/qmix_sumo.yaml`

```diff
- buffer_size: 200  # Replay buffer size (episodes). 5000 requires ~41GB RAM; 200 ≈ 3.5GB
+ buffer_size: 400  # Replay buffer size (episodes). ~16.6MB/episode → 400 ≈ 6.5GB RAM
```

400 episodes was chosen to double sample diversity from the previous 200 while leaving headroom below the machine's RAM ceiling.

---

## 4. Dead RMSprop Config Keys — Cleaned

### 4.1 Problem

The optimizer was switched to Adam at some point but two RMSprop-specific keys were left in `qmix_sumo.yaml`:

```yaml
optim_alpha: 0.99  # RMSprop alpha (if using RMSprop)
optim_eps: 0.00001  # Optimizer epsilon
```

`optim_alpha` is never read anywhere in the codebase. `optim_eps` is a valid Adam parameter and was kept.

### 4.2 Change

**File:** `pymarl/src/config/algs/qmix_sumo.yaml`

```diff
- optim_alpha: 0.99  # RMSprop alpha (if using RMSprop)
  optim_eps: 0.00001  # Optimizer epsilon
+ optim_eps: 0.00001  # Adam epsilon
```

---

## 5. File Change Summary

| File | Change |
|------|--------|
| `pymarl/src/learners/q_learner.py` | Replaced gradient-step counter with episode-based counter for target network updates |
| `pymarl/src/config/algs/qmix_sumo.yaml` | Buffer size 200 → 400; removed dead `optim_alpha` key; corrected buffer memory comment |
| `docs/qmix_pipeline_audit.md` | New — full pipeline audit against canonical QMIX checklist |

---

## 6. Documented Deviations (No Action Taken)

These deviations from the paper were confirmed intentional and are documented here for thesis reference.

**Agent input (obs only):**
`obs_last_action: false` and `obs_agent_id: false` are set in `qmix_sumo.yaml`. The paper includes both for agent differentiation. Disabled here to reduce input dimensionality. Should be explicitly addressed when comparing results against paper baselines.

**Agent input (obs only) — parameter sharing consequence:**
Parameter sharing is enabled via a single `RNNAgent` instance. All 32 agents are 
processed through the same weights via observation reshaping to 
`(batch_size * n_agents, obs_dim)` in the controller. With `obs_agent_id: false`, 
agents have no differentiation signal during the forward pass — the network cannot 
distinguish which agent it is acting for.

In the traffic domain this is a deliberate tradeoff: vehicles are functionally 
homogeneous (identical action spaces, observation structures, and objectives), unlike 
the heterogeneous unit types in the original StarCraft benchmark. Agent ID would 
provide no meaningful role signal and may introduce noise. The shared policy is 
therefore a feature of the domain, not a limitation of the implementation.

**Optimizer (Adam vs. RMSprop):**
Adam is used at `lr=5e-4`. The paper specifies RMSprop. Both are valid adaptive optimizers; Adam is standard in modern MARL implementations.

**Epsilon annealing (500k vs. 50k steps):**
`epsilon_anneal_time: 500000` — 10× slower than the paper's 50k. This is appropriate for a 32-agent traffic environment with a more complex exploration landscape than the paper's StarCraft maps.
