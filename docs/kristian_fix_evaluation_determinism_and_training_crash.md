# Fix: Evaluation Determinism, Training Crash, and Travel-Time Bias

**Date:** 2026-04-02  
**Files changed:**
- `pymarl/src/envs/sumo_grid_reroute.py`
- `pymarl/src/learners/q_learner.py`

---

## Background

Three bugs were identified after reviewing the seed0 and seed42 evaluation result files
(`pymarl/src/results/eval/`). All deterministic policies (noop, greedy_shortest, QMIX)
showed std=0.0 across 5 episodes, QMIX training crashed at t=5000 leaving a useless
checkpoint, and mean travel time was systematically underestimated for replacement vehicles.

---

## Fix 1 — Zero variance: per-episode SUMO seed

**File:** `pymarl/src/envs/sumo_grid_reroute.py`  
**Location:** `reset()`, just before `self._start_sumo()`

### Root cause

`sumo_seed: null` in `sumo_grid4x4.yaml` means no `--seed` flag is ever passed to SUMO.
Without an explicit seed, SUMO falls back to its compiled-in default (seed `23423`) on
every process start. Every call to `reset()` kills the old SUMO process and starts a new
one — deterministic background-traffic demand, identical vehicle insertion times, zero
variance in returns across episodes.

The Python-level seed (`np.random.seed(args["seed"])` in `evaluate.py`) controls
NumPy/PyTorch but has no effect on SUMO's internal RNG.

Evidence from results: `noop_seed0_seed0.json` and `qmix_seed0_seed0.json` both show
`"std": 0.0` across all 5 raw values. Only `random_seed0_seed0.json` shows variance
(std=7306) because its variance comes from Python action choices, not the environment.

### Fix

```python
# pymarl/src/envs/sumo_grid_reroute.py  —  reset(), before _start_sumo()
self.sumo_seed = int(np.random.randint(0, 2**31 - 1))
```

NumPy's RNG is already seeded by `args["seed"]` in the evaluation harness, so the
sequence of episode seeds is reproducible across identical evaluation runs while still
giving each episode distinct demand. The existing `if self.sumo_seed is not None` guard
in `_start_sumo()` (line ~319) picks it up automatically — no other change needed.

---

## Fix 2 — Training crash at t=5000: tensor shape mismatch in logging

**File:** `pymarl/src/learners/q_learner.py`  
**Location:** `train()`, inside the `if t_env - self.log_stats_t >= log_interval` block

### Root cause

`chosen_action_qvals` starts as `(batch, T, n_agents)` but is reshaped to
`(batch*T, n_agents)` at line 140 for the mixer forward pass:

```python
chosen_action_qvals = chosen_action_qvals.view(batch_size * max_t, self.n_agents)
```

It is never reshaped back. The logging line at the end of the function tried to multiply
it against `mask` which is `(batch, T, 1)`:

```python
# BEFORE (crashes):
(chosen_action_qvals * mask).sum()   # (batch*T, n_agents) × (batch, T, 1) → shape error
```

With the default training `batch_size=32` and `episode_limit=1000`, the concrete shapes
are `(32000, 32) × (32, 1000, 1)`. PyTorch expands the 2-D tensor to `(1, 32000, 32)`;
`32000 ≠ 1000` makes the broadcast illegal → RuntimeError.

This crash does NOT occur during evaluation (where the single-episode runner uses
`batch_size=1`, which happens to broadcast without error). It only fires during training
when the replay buffer is sampled at the configured `batch_size=32`.

The first logging event fires at `t_env >= log_interval` (default 5000), which is
exactly why training crashed at t=5000. The saved checkpoint represents only the first
half of the intended 10000-step run.

### Fix

```python
# pymarl/src/learners/q_learner.py  —  inside logging block
q_log = chosen_action_qvals.view(batch_size, max_t, self.n_agents)
self.logger.log_stat(
    "q_taken_mean",
    (q_log * mask).sum().item() / (mask.sum().item() * self.n_agents),
    t_env)
```

`q_log` is `(batch, T, n_agents)`. Multiplying by `mask` `(batch, T, 1)` broadcasts
cleanly. The denominator gains `* self.n_agents` so the stat is the mean per
(timestep, agent) pair rather than per timestep.

### What this unlocks

With the crash fixed, retraining from scratch reaches t=10000 and produces a properly
trained checkpoint. The current `results/models/seed0/best/` files come from an
undertrained model (5000 steps) that was effectively selecting random actions and
achieved a return of −324,584 vs noop's −316,434. A fully trained model should close
and reverse that gap.

---

## Fix 3 — Travel-time bias: depart_time float/int mismatch

**File:** `pymarl/src/envs/sumo_grid_reroute.py`  
**Location:** `_spawn_vehicle()`, spawn-time recording after `traci.vehicle.add()`

### Root cause

`traci.vehicle.add()` receives `depart=str(int(depart_time))` — SUMO inserts the vehicle
at the integer-truncated simulation step. The spawn time stored for travel-time
measurement was the raw float:

```python
# BEFORE:
traci.vehicle.add(..., depart=str(int(depart_time)), ...)
self.vehicle_spawn_times[vehicle_id] = depart_time          # e.g. 50.7
```

For a replacement vehicle spawned when `sim_time = 50.7`, SUMO starts the vehicle at
step 50, but the denominator of the travel-time calculation was `50.7`. This
systematically underestimates travel time by up to 1.0 second per vehicle. For the
short controlled routes in the current scenario (~9–10 s trips), that is a ~10% error.

### Fix

```python
# AFTER:
self.vehicle_spawn_times[vehicle_id] = float(int(depart_time))   # e.g. 50.0
```

Aligns the recorded spawn time with SUMO's actual integer departure step. Initial
vehicles (`depart_time=0.0`) are unaffected since `float(int(0.0)) == 0.0`.

---

## Additional finding (no code change needed)

`greedy_shortest` returns identical results to `noop` across all episodes. This is
expected given the scenario: `_spawn_vehicle()` always initialises routes via
`_compute_shortest_route()`, so every controlled vehicle is already on its shortest path
when the greedy_shortest baseline runs. Re-applying the shortest path is a no-op. The
baseline is not broken — it just has no room to improve over noop in this setup.

---

## Verification checklist

1. **Issue 1** — Re-run `evaluate.py --baseline noop --episodes 5 --seed 0`. Confirm
   `"std"` in the returned JSON is non-zero.
2. **Issue 2** — Retrain from scratch (`main.py`). Confirm training reaches t=10000
   without a crash. Check logs for `q_taken_mean` appearing at intervals.
3. **Issue 3** — Re-run evaluation. `mean_travel_time` should increase by ~0.5–1 s for
   replacement-heavy episodes. Initial-vehicle trips (depart=0) are unchanged.
