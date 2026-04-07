# Experiment + Hotfix: Running Reward Normalization
**Date:** 2026-04-07
**Status:** Partial — 20k sanity check with normalization live is pending
**Affected areas:** `pymarl/src/learners/q_learner.py`, `pymarl/src/config/algs/qmix_sumo.yaml`

## Context
The 500k smoke test (seed=0, reward_scale=250) revealed slow but monotonic
divergence: loss grew from 8.19 at t=5k to 384 at t=500k, and `target_mean`
drifted from -22 to -219. Root cause: as the policy improved throughput,
more vehicles occupied the network simultaneously, increasing raw reward
magnitude per step. The fixed `reward_scale=250` was calibrated for ~50
mean vehicles (LOS A baseline) and could not adapt to higher-density episodes.

## Discussion and decisions
Post-500k analysis identified two candidate fixes:

- **Option A — Running mean/std normalization:** Track a running estimate of
  reward magnitude and normalize before TD computation. Fully adaptive;
  scale adjusts as vehicle density grows throughout training. Requires
  implementation in `QLearner` since the existing `reward_standardization`
  flag in PyMARL is present in the config but not implemented.

- **Option B — Increase `reward_scale` to 375:** Blunt fix based on the
  hypothesis that peak vehicle count during high-throughput episodes is ~75
  (5 × 75 = 375). Faster to test but still a fixed scalar that would
  eventually be outpaced.

Decision: run a 20k diagnostic with Option B first to confirm the hypothesis,
then implement Option A as the permanent fix regardless of whether B held for
the full 500k range. Both options were run in sequence rather than jumping
straight to A, because a failed 375 test would have invalidated the vehicle
density hypothesis before investing implementation time.

The 20k diagnostic (seed=2, reward_scale=375) confirmed the hypothesis:
`target_mean` was completely flat at ~-15.5 across all 20k steps, and loss
decreased from 3.27 → 1.36. Option A was then implemented.

`grad_norm_clip` was also reduced from 10000 to 10 (PyMARL default) as part
of this fix. The original value of 10000 was set to "match reward magnitude"
but was effectively no clipping, allowing gradient spikes up to 2812 during
the 500k run.

## What changed
- `pymarl/src/config/algs/qmix_sumo.yaml` — `reward_scale: 375.0` added as
  diagnostic step (Option B), replacing the original 250
- `pymarl/src/config/algs/qmix_sumo.yaml` — `reward_scale` removed and
  replaced with `reward_ema_decay: 0.99` after Option A was implemented
- `pymarl/src/config/algs/qmix_sumo.yaml` — `grad_norm_clip` reduced from
  10000 to 10
- `pymarl/src/learners/q_learner.py` — removed `self.reward_scale`; added
  `self.ema_decay`, `self.reward_running_mean`, `self.reward_running_var`
- `pymarl/src/learners/q_learner.py` — replaced single `/reward_scale` line
  in `train()` with per-batch EMA update over valid timesteps + in-place
  normalization: `(r - mean) / (std + 1e-8)`
- `pymarl/src/learners/q_learner.py` — added `reward_running_mean` and
  `reward_running_std` to logged stats for monitoring
- `pymarl/src/learners/q_learner.py` — `save_models()` / `load_models()`
  now persist `reward_stats.pth` alongside the model so the EMA state
  survives checkpointing and resume

## Implementation notes
- `mask` is now moved to device before the other tensors in `train()` so it
  is available for valid-timestep indexing during the EMA update. Order
  change is harmless but intentional.
- Running variance is initialised to 1.0 (not 0.0) to prevent divide-by-zero
  on the very first batch before any real variance has accumulated.
- EMA decay of 0.99 weights recent batches more heavily than early noisy
  episodes. This is intentional — early training has high variance from
  random exploration and should not permanently anchor the running scale.
- `reward_stats.pth` loading is guarded with `os.path.exists` so old
  checkpoints without the file still load cleanly.
- `target_mean` is now expected to stabilize around -1 to -2 with normalized
  rewards, compared to -22 at t=5k with the old fixed scalar.

## Status and follow-up
- **Status:** Partial — implementation complete, sanity check pending
- **Follow-up items:**
  - Run 20k sanity check with running normalization + grad_norm_clip=10 live;
    confirm `target_mean` holds flat, loss decreases, and `reward_running_std`
    settles to a stable value
  - If sanity check passes, run full 500k training run
  - Investigate `q_taken_mean` being anchored at ~-0.62 throughout training
    while `target_mean` is significantly more negative — may indicate the
    mixer is handling most of the scale difference, but worth confirming
    Q-values are actually tracking targets post-normalization
