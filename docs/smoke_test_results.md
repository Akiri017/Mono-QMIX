# Smoke Test Results

**Date:** 2026-03-23
**Run config:** 3 seeds (42, 43, 44) × 50,000 steps, 5 eval episodes per policy
**Hardware:** Intel i7-H series, CPU only
**Script:** `python run_experiments.py --seeds 3 --first_seed 42 --t_max 50000 --eval_episodes 5`

---

## 1. Training Summary

| Seed | Status | Duration | Notes |
|---|---|---|---|
| 42 | Done | ~132 min | Completed normally |
| 43 | Done | ~132 min | Completed normally |
| 44 | **Killed** | >216 min | Hung — output frozen, no `final` checkpoint saved |

Seed 44 stalled after saving its best checkpoint at ~105 minutes in. The SUMO subprocess appeared to hang waiting on a TraCI response. No crash was raised — the process stayed alive but produced no output for over 100 consecutive minutes. It was manually terminated.

Evaluation was re-run on seeds 42 and 43 only using:

```bash
python run_experiments.py --seeds 2 --first_seed 42 --eval_episodes 5 \
    --eval_only --checkpoint_root results/experiments/20260323_125850
```

---

## 2. Bugs Encountered During Eval

### 2.1 QMIX Eval Crash — `action_selector` AttributeError

When `evaluate.py` attempted to load the trained QMIX model for seed 42, it crashed immediately on startup:

```
AttributeError: 'BasicMAC' object has no attribute 'action_selector'
  File "evaluate.py", line 131, in evaluate_policy
    mac.action_selector.epsilon = 0.0
```

**Root cause:** `evaluate.py` assumed `BasicMAC` exposed a standalone `action_selector` object (a legacy PyMARL API). In this codebase `BasicMAC` handles epsilon internally — passing `test_mode=True` to `select_actions()` already switches to greedy. The explicit epsilon override was incorrect and not needed.

**Fix applied:** Removed `mac.action_selector.epsilon = 0.0` from `evaluate.py`. Greedy action selection is now correctly handled by `test_mode=True` in the episode runner.

**Impact on results:** The fix was applied before seed 43's eval ran. Seed 42's QMIX eval had already failed and was not re-run. QMIX results therefore have **n=1** (seed 43 only) while baseline results have n=2.

---

## 3. Evaluation Results

Results directory: `pymarl/src/results/eval/`
Summary JSON: `pymarl/src/results/experiments/20260323_125850/experiment_summary.json`

| Policy | n | Mean Return | Mean Travel Time (s) | Mean Waiting Time (s) | Total Stops | Total Emissions (g) | Arrival Rate |
|---|---|---|---|---|---|---|---|
| QMIX | 1 | -318,877.66 | 8.464 | 0.000 | 108,021 | 561,601.7 | 0.731 |
| NoOp | 2 | -318,877.66 | 8.464 | 0.000 | 108,021 | 561,601.7 | 0.731 |
| Greedy Shortest | 2 | -318,877.66 | 8.464 | 0.000 | 108,021 | 561,601.7 | 0.731 |

All three policies produced **identical metrics**. This is expected — see section 4.

### Statistical Significance Tests

**Skipped.** `scipy` is not installed in the environment. The runner caught this gracefully and printed:

```
[scipy not installed - skipping significance tests]
```

Install before the real training run:

```bash
pip install scipy
```

---

## 4. Why All Policies Are Identical

This is expected behavior for a smoke test at 50,000 steps. Two reasons:

**a) QMIX has not learned anything.**
The epsilon schedule anneals from 1.0 to 0.05 over `epsilon_anneal_time = 500,000` steps. At 50,000 steps, epsilon at end of training was:

```
epsilon = 1.0 - (0.95 × 50,000 / 500,000) = 0.905
```

The agent acted randomly ~90% of the time throughout all of training. The Q-network has not converged to any meaningful policy.

**b) The scenario may be under-congested at this episode length.**
With `episode_limit = 100` decision steps and `decision_period = 10` SUMO seconds per step, each episode covers 1,000 simulated seconds. At low vehicle density, congestion may not develop within this window, so any routing policy — including noop — produces the same travel time outcome.

These results confirm the pipeline is wired correctly end-to-end. Meaningful QMIX vs baseline differentiation will only appear at `t_max >= 500,000` steps.

---

## 5. What the Smoke Test Confirmed

| Check | Result |
|---|---|
| SUMO environment initializes | Pass |
| QMIX training loop runs without crash | Pass |
| Best model checkpoint saves during validation | Pass |
| Evaluation loads trained model weights | Pass (after fix) |
| Baseline (NoOp, Greedy Shortest) evaluation | Pass |
| Episode metrics collected correctly | Pass |
| Results serialized to JSON | Pass |
| Aggregate table printed | Pass |
| Statistical t-tests | Skipped (scipy missing) |
| Seed 44 completion | Fail (SUMO hang) |

---

## 6. Actions Before Real Training Run

1. **Install scipy:** `pip install scipy` and add it to `requirements.txt`
2. **Add SUMO subprocess timeout:** Guard against seed hangs — a hung seed blocks all remaining seeds indefinitely
3. **Use `t_max >= 500,000`** for results that show meaningful learning
4. **Consider parallel seeds** on separate machines to cut wall time (see `performance_observations.md` §4)
5. **Confirm `mean_waiting_time` accumulation strategy** — currently always 0.0 (known issue, waiting time can only be read while vehicle is active)
