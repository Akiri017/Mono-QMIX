# TensorBoard Logging Improvements and Gradient Clipping Fix

**Date:** 2026-04-03
**Branch:** libsumo
**Relevant files:** `pymarl/src/main.py`, `pymarl/src/learners/q_learner.py`, `pymarl/src/config/algs/qmix_sumo.yaml`

---

## 1. TensorBoard Logging Improvements

### Motivation

Prior to this change, TensorBoard only logged coarse metrics (loss, grad_norm, test return,
validation travel time). For the 500k training run, we needed higher signal density to detect
learning onset, reward scale problems, and buffer warmup state.

### What was added

Three new stats are logged every `log_interval` (5000 steps) in `main.py`:

| Tag | Description | Why it matters |
|-----|-------------|----------------|
| `epsilon` | Current exploration rate from the linear annealing schedule | Shows exactly when QMIX transitions from exploration to exploitation. At 500k steps, epsilon crosses ~0.3 around t=350k — the point where learning should visibly start outperforming baselines. Without this, learning onset is invisible in TensorBoard. |
| `buffer_fill` | `episodes_in_buffer / buffer_size` (0.0 → 1.0) | Loss before the buffer is warm is meaningless noise. This makes it explicit when replay has enough diversity to trust the loss curve. |
| `train_return_mean` | Mean episode return over the last `log_interval` window | Per-episode training returns (already logged) are noisy. This smoothed mean shows the trend clearly. Resets each interval so it reflects the current window only. |

One new stat was added in `q_learner.py` at each `log_interval`:

| Tag | Description | Why it matters |
|-----|-------------|----------------|
| `q_taken_std` | Standard deviation of Q-values for taken actions (valid timesteps only) | Detect Q-value collapse (std → 0, all agents converge to same value) or divergence (std exploding). Complements the existing `q_taken_mean`. |

### Implementation note — `q_taken_mean` fix

The previous `q_taken_mean` computation multiplied `q_log` by `mask` before averaging,
which included padded (zero) timesteps in the denominator-equivalent and slightly biased
the mean downward. The fix extracts only valid entries first:

```python
# Before (biased — zeros from masked steps affect the distribution)
(q_log * mask).sum() / (mask.sum() * n_agents)

# After (correct — only valid timesteps)
valid_q = q_log[mask.expand_as(q_log).bool()]
valid_q.mean()   # q_taken_mean
valid_q.std()    # q_taken_std
```

---

## 2. Gradient Clipping Scale Fix

### Diagnosis

Running `python pymarl/src/main.py --t_max 5000 --seed 42` with the new logging produced:

```
loss:       8,173,261
grad_norm:  1,223,389
q_taken_mean:  -0.65
target_mean:   -2,772
```

The `grad_norm_clip` was set to `10` (inherited from standard PyMARL, designed for
unit-scale rewards). The reward structure in this project is:

```python
# sumo_grid_reroute.py:709
r_time = -1.0 * len(all_vehicles)   # ~-300 per sub-step at medium density
```

With `reward_global: true`, all vehicles (controlled + background, ~300+) contribute
to every step. Over `episode_limit = 1000` steps:

```
~300 vehicles × 1000 steps = ~300,000 per episode
```

This matches the observed `test_return_mean ≈ -315,000`.

### Why `grad_norm_clip: 10` was harmful

`clip_grad_norm_` clips gradients to `max_norm` and returns the **pre-clip norm**.
The logged `grad_norm: 1,223,389` is the pre-clip value — clipping to 10 was
happening every step, scaling gradients down by:

```
10 / 1,223,389 ≈ 0.000008  (0.0008% of the natural gradient)
```

At this scale, the Q-network could barely move toward the TD targets per update,
which is why `q_taken_mean ≈ -0.65` while `target_mean ≈ -2,772` even after
5,000 training steps. Learning was technically occurring but at negligible rate.

### Fix

`grad_norm_clip` scaled proportionally to the reward magnitude:

```yaml
# qmix_sumo.yaml
# Before
grad_norm_clip: 10

# After
grad_norm_clip: 10000
```

**Rationale:** Standard QMIX uses `grad_norm_clip: 10` for environments with
rewards in `[-1, 1]`. Our reward scale is ~300,000× larger, so the clip threshold
is scaled by the same order of magnitude (10 × 1000 = 10,000). This allows the
optimizer to take meaningful steps while still preventing runaway gradient spikes
from outlier batches.

The fix is conservative — `10,000` is still a meaningful ceiling given that
natural grad norms of ~1.2M were observed. It allows roughly 1% of the natural
gradient through rather than 0.0008%.

---

## 3. Expected Changes in TensorBoard After Fix

After `grad_norm_clip: 10000`, the following changes are expected compared to the
5k diagnostic run:

| Metric | Before fix | Expected after fix |
|--------|------------|-------------------|
| `q_taken_mean` | Barely moves from 0 | Should track toward `target_mean` faster |
| `target_mean` | -2,772 at t=5k | Gap to `q_taken_mean` should close over time |
| `loss` | Stays very high | Should decrease as Q-values converge to targets |
| `grad_norm` | ~1.2M constant | May still be high but loss should start declining |
| `train_return_mean` | Flat | Should show improvement after epsilon decay begins |
