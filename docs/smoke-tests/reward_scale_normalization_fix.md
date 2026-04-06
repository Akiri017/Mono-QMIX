# Hotfix: Reward Scale Normalization Before TD Computation

**Date:** 2026-04-06
**Status:** Complete — validation pending (20k smoke test not yet run)
**Affected areas:** `pymarl/src/learners/q_learner.py`, `pymarl/src/config/algs/qmix_sumo.yaml`

## Context

Post-500k smoke test analysis revealed three correlated training failures: TD loss monotonically increasing from 6M → 31M over 500k steps, `q_taken_mean` frozen at -0.71 for the entire run, and no improvement in `test_return_mean` despite epsilon annealing to 0.05. Root cause traced to raw rewards entering the TD computation at a scale of ~-250 per decision step (5 SUMO sub-steps × ~50 vehicles × `reward_time_weight=1.0`), producing TD targets on the order of -25,000 and MSE loss on the order of 10⁸–10¹⁰ per sample. The frozen Q-values were identified as GRU hidden state saturation downstream of the large-gradient regime.

## Discussion and decisions

**Option A (normalize at environment level, in `_compute_reward`)** was considered and rejected. Normalizing at the env level would scale the `episode_return` logged in `episode_runner.py` before it reaches the learner, making all future TensorBoard runs incomparable to the 500k baseline eval summaries (`summary_20260405_152355.json`) without re-running all baselines. It would also bake normalization into the env, making the env behavior implicit rather than explicit.

**Option B (normalize at learner level, in `q_learner.py` before TD computation)** was chosen. Raw rewards are preserved in the replay buffer and in all upstream logging. The normalization is a single, explicit divide that lives in the alg config alongside the other training hyperparameters. The `reward_standardization: false` flag already in the alg config was confirmed to be a dead flag — never read by any code — so a new `reward_scale` param was introduced instead.

Before implementing, the `rewards` variable was traced through `train()` to confirm `batch["reward"]` is read exactly once (line 89), and the same local variable flows directly into the TD target at line 153 with no second raw read. No inconsistency risk.

The `grad_norm_clip: 10000` setting was **not changed**. That value was set deliberately in a prior diagnostic run (documented in `docs/smoke-tests/tensorboard_logging_and_grad_clip_fix.md`) after measuring a natural grad norm of ~1.2M against the original clip of 10, which reduced effective gradient to 0.0008%. The 10000 value is a proportional compensation, not circular reasoning. It should be re-evaluated empirically after the reward scale fix stabilizes training.

## What changed

- `pymarl/src/learners/q_learner.py` (`__init__`) — added `self.reward_scale = args.get("reward_scale", 1.0)` alongside other learning parameters. Default of `1.0` means no behavior change on runs that don't set the config key — safe for checkpoint resume.
- `pymarl/src/learners/q_learner.py` (`train()`) — added `rewards = rewards / self.reward_scale` immediately after the device transfer (`rewards.to(self.device)`), before any TD computation. The normalized variable is what flows into `targets = rewards + self.gamma * (1 - terminated) * target_q_tot`.
- `pymarl/src/config/algs/qmix_sumo.yaml` — added `reward_scale: 250.0` under the reward configuration block, with a comment explaining the derivation (`decision_period_steady=5` × ~50 mean vehicles in network).

## Implementation notes

The scale value of 250 is a first-pass estimate derived from: `decision_period_steady (5 sub-steps) × mean vehicle count (~50)`. This brings per-step rewards to approximately [-1, 0], which is the range standard QMIX hyperparameters were tuned for. The actual mean vehicle count varies by LOS level and episode phase, so 250 is an approximation. It should be treated as an initial value to tune from, not a calibrated constant.

After normalization, expected TD target magnitude drops from ~-25,000 to ~-100 (discounted return over ~100 effective steps). Expected MSE loss at step 5k should be in the hundreds to low thousands — four orders of magnitude below the pre-fix 6–8M.

## Status and follow-up

- **Status:** Implementation complete. Validation pending.
- **Follow-up items:**
  - Run 20k step smoke test. Primary go/no-go at step 5k: `loss` should be hundreds–low thousands (not millions), `grad_norm` should be in thousands (not millions), `target_mean` should be ~-10 to -50.
  - If loss is decreasing by step 20k and `q_taken_mean` is visibly drifting from -0.71, fix is confirmed working.
  - After validation, re-evaluate `grad_norm_clip: 10000` with a new 5k diagnostic run. With normalized rewards, natural grad norms should drop significantly and the clip may be reducible.
  - Tune `reward_scale` value if the 250 estimate is too coarse — check `target_mean` at 5k steps to calibrate.
