# Feature: Reward Normalization Integration into Civiq (HierarchicalQLearner)

**Date:** 2026-04-10
**Status:** Partial — normalization implemented, smoke test not yet run
**Affected areas:** `pymarl/src/learners/hierarchical_q_learner.py`, `pymarl/src/config/algs/civiq_sumo.yaml`, `run_experiments.py`

## Context

Phase 5 smoke test flagged `global_mixer_grad_norm` at ~2–3M due to raw reward
magnitudes of ~300k/episode. `grad_norm_clip=10000` kept training stable but
the gradient scale was too aggressive for a full training run. Reward
normalization had already been validated on Mono QMIX (commit
`0f07be88ff07cb10c7c1330595da807f8ca71447`) but was not yet integrated into the
Civiq learner. The `main` branch was merged into `rsu-placement`
(commit `69cc795`) to bring the implementation across.

## Discussion and decisions

- **EMA running normalization** was the only approach considered. Fixed reward
  scaling was not revisited — it was already tried on Mono QMIX and discarded
  in favor of the adaptive EMA approach, which handles changing reward
  distributions as traffic density grows with an improving policy.
- The normalization is applied **globally to episode rewards** before TD target
  computation, not per-RSU. This is correct because rewards are a single scalar
  per timestep; the hierarchical mixing operates on Q-values, not rewards.
- `grad_norm_clip` was reduced from `10000` to `10` in `civiq_sumo.yaml` to
  match the normalized reward scale, consistent with Mono QMIX post-normalization
  config.

## What changed

- **`hierarchical_q_learner.py` `__init__`** — added `self.ema_decay`,
  `self.reward_running_mean = 0.0`, `self.reward_running_var = 1.0`; mirrors
  `q_learner.py` lines 47–52 exactly
- **`hierarchical_q_learner.py` `train()`** — EMA update + normalization block
  inserted after `mask.to(device)`, before the MAC forward pass; filters to
  valid timesteps via `rewards[mask.bool()]` before computing batch statistics
- **`hierarchical_q_learner.py` `train()` logging** — added
  `reward_running_mean` and `reward_running_std` stats to the log interval block
- **`hierarchical_q_learner.py` `save_models()`** — saves
  `reward_stats.pth` (`running_mean`, `running_var`) alongside mixer and
  optimizer state; ported from `q_learner.py` lines 253–258
- **`hierarchical_q_learner.py` `load_models()`** — restores `reward_stats.pth`
  with `os.path.exists` guard so old checkpoints without the file still load
  cleanly; ported from `q_learner.py` lines 268–272
- **`civiq_sumo.yaml`** — added `reward_ema_decay: 0.99`; changed
  `grad_norm_clip: 10000` → `10`
- **`run_experiments.py`** — added `--alg_config` CLI flag forwarded to the
  training subprocess; added `Alg config` line to the experiment plan printout
  so the active config is visible in terminal logs

## Implementation notes

- The hierarchical structure is fully agnostic to reward normalization —
  `LocalQMixer` and `GlobalQMixer` operate on Q-values, not rewards, so no
  changes were needed in either mixer.
- Checkpointing (`reward_stats.pth`) was ported explicitly from Mono QMIX to
  prevent a silent scale discontinuity on resume: without it, EMA state resets
  to `mean=0, var=1` mid-training, destabilizing TD targets from that point on.
- Terminal experiment logs when running via `run_experiments.py` now include the
  active `Alg config` in the plan header, standardizing Civiq output to match
  the Mono QMIX log format.
- `reward_standardization: false` is retained in `civiq_sumo.yaml` as a PyMARL
  compatibility stub — it is not used; normalization is handled directly inside
  `HierarchicalQLearner`.

## Status and follow-up

- **Status:** Partial — implementation complete, smoke test not yet run
- **Follow-up items:**
  - Run pipeline smoke test (`--t_max 10000`) to confirm end-to-end integration
    with `run_experiments.py` and `--alg_config civiq_sumo.yaml`
  - Verify `global_mixer_grad_norm` is significantly reduced vs. Phase 5
    baseline (~2–3M pre-normalization)
  - Verify `reward_running_mean` and `reward_running_std` appear in terminal
    logs and TensorBoard
  - If grad norms are stable, proceed to a full divergence-detection smoke test
    at higher `t_max` (100k–500k)
