# Debug Session: 20k Smoke Test — Reward Normalization Validation

**Date:** 2026-04-06
**Status:** Complete — fix confirmed working
**Affected areas:** `results/eval/summary_20260406_214841.json`, `pymarl/src/results/logs/events.out.tfevents.1775480318.Ianne.1900.0`
**Commit:** `7c0ae1e7282ba0e98846ae61ec27e721bf35d3de`
**Seed:** 1 (fresh run, not resumed from 500k checkpoint)

## Context

Validation run to confirm the reward scale normalization fix (`reward_scale: 250.0` applied in `q_learner.py` before TD computation) resolved the three training failures identified in the 500k smoke test: monotonically increasing loss, frozen Q-values at -0.71, and flat `test_return_mean` throughout. Ran 20k steps — the agreed minimum to see all primary diagnostic signals (loss trend, grad norm, q_taken_mean, and two test evaluations at test_interval=10k).

## What to look for (pre-run criteria)

| Signal | Go condition | Step visible |
|---|---|---|
| `loss` at step 5k | Hundreds–low thousands (not millions) | 5k |
| `grad_norm` at step 5k | Thousands, not millions | 5k |
| `loss` trend | Decreasing by step 20k | 20k |
| `q_taken_mean` | Drifting from -0.71 init | 10k |
| `target_mean` | ~-10 to -50 (normalized range) | 5k |

## Results

### Training diagnostics

| Step | `loss` | `grad_norm` | `q_taken_mean` | `target_mean` |
|---|---|---|---|---|
| 5k | **10.34** | **981** | -0.49 | -22.27 |
| 10k | 3.73 | 51.0 | -0.54 | -22.55 |
| 15k | 3.43 | 63.3 | -0.55 | -22.26 |
| 20k | 3.32 | 13.7 | -0.54 | -22.61 |

**Pre-fix comparison at step 5k:** loss was 6,035,818 and grad_norm was 2,030,675. Post-fix loss is 583,000× lower and grad_norm is 2,070× lower. All four go conditions met.

- Loss decreasing monotonically: ✅ (10.34 → 3.32 over 20k steps)
- Grad norm collapsing: ✅ (981 → 13.7, well below clip threshold of 10000)
- Q-values moving: ✅ (-0.49 at 5k, no longer frozen at -0.71)
- Target mean in normalized range: ✅ (-22.27 ≈ raw -5,568 / 250, consistent with discounted return early in training)

### Eval summary vs. baselines (t_max=20k, seed=1, 5 episodes each)

| Policy | Return (mean) | Travel Time | Waiting Time | Total Stops |
|---|---|---|---|---|
| **QMIX** | **-238,522** | **13.80s** | **22.83s** | **62,594** |
| noop | -241,525 | 14.05s | 24.21s | 64,945 |
| greedy_shortest | -241,718 | 14.14s | 23.93s | 64,406 |
| random | -218,953 | 15.78s | 19.53s | 48,945 |

QMIX beats noop and greedy_shortest in return at only 20k steps. In the 500k pre-fix run, QMIX was losing to both (QMIX -247k vs noop -234k). Random still leads on raw return, but epsilon at step 20k is ~0.905 — that comparison is not meaningful until post-anneal.

`train_return_mean` correctly stays at raw scale (~-248k) throughout — confirms Option B normalization (learner-only, buffer untouched) is working as designed.

## Implementation notes

`grad_norm` fell to 13.7 by step 20k, far below `grad_norm_clip: 10000`. The clip is entirely inactive post-normalization. This was anticipated as a follow-up item in `reward_scale_normalization_fix.md` but is not causing any current problem — it simply means the clip is doing nothing. Do not reduce it until a longer run (100k+) confirms the grad norm stays low.

`reward_scale: 250.0` was derived as `decision_period_steady (5) × mean_vehicles (~50)`. The `target_mean` settling at ~-22 (normalized) is consistent with this estimate being reasonable for the current traffic density.

## Status and follow-up

- **Status:** Complete — all pre-run go conditions met, fix confirmed.
- **Follow-up items:**
  - Re-evaluate `grad_norm_clip: 10000` after a 100k+ run — it is now inactive and can likely be reduced to standard QMIX value (10), but needs empirical confirmation the norm stays low across the full epsilon anneal.
  - Run a longer budget (100k–200k) to determine the training ceiling now that the divergence is resolved and learning is occurring.
  - Tune `reward_scale` if needed — check whether -22 normalized target is the right scale for the full 500k run or if a different divisor would better center the value range.
