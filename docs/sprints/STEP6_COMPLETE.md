# Step 6 Implementation: Training + Evaluation Protocol — COMPLETE

## Overview

Step 6 adds a comprehensive training and evaluation protocol on top of the PyMARL QMIX
framework built in Step 5. The goal is to make it possible to measure whether the learned
QMIX policy actually improves traffic efficiency relative to non-learning baselines, using
statistically sound multi-seed experiments.

---

## What Was Built

### 1. Episode-Level Metrics Tracking
**File:** `pymarl/src/envs/sumo_grid_reroute.py`

The environment previously only exposed a scalar reward signal per step. It now
accumulates rich per-episode statistics and returns them in the `info` dict when
the episode terminates.

**New instance variables (initialised in `__init__`, reset in `reset()`):**

| Variable | Type | Purpose |
|---|---|---|
| `vehicle_spawn_times` | `dict` | Maps vehicle ID → simulation time when it entered |
| `vehicle_travel_times` | `list` | Completed travel times for all vehicles |
| `vehicle_waiting_times` | `list` | Total waiting times for completed vehicles |
| `controlled_travel_times` | `list` | Travel times for agent-controlled vehicles only |
| `background_travel_times` | `list` | Travel times for background traffic only |
| `controlled_vehicle_ids` | `set` | Tracks which vehicle IDs are under agent control |
| `episode_stops_count` | `int` | Cumulative stop events across the whole episode |
| `episode_emissions` | `float` | Cumulative CO₂ in grams across the whole episode |
| `episode_arrivals` | `int` | Number of vehicles that reached their destination |
| `total_spawned` | `int` | Total vehicles that entered the simulation |

**New / modified methods:**

- `_spawn_vehicle()` — now records `vehicle_spawn_times[vehicle_id]` and adds the ID to
  `controlled_vehicle_ids` when a new agent vehicle is created.
- `_advance_simulation()` — calls `_track_new_vehicles()` every simulation micro-step so
  background vehicles are also registered.
- `_track_new_vehicles()` — new method; iterates `traci.vehicle.getIDList()` and records
  the departure time of any vehicle not yet seen, using `traci.vehicle.getDeparture()`.
- `_compute_reward()` — now also accumulates `episode_stops_count` and
  `episode_emissions` in addition to computing the reward.
- `_handle_arrivals()` — now computes travel time (arrival − spawn), retrieves
  `traci.vehicle.getWaitingTime()`, categorises the vehicle as controlled or background,
  and appends to the appropriate lists before marking the slot inactive.
- `_compute_episode_metrics()` — new method; called by `step()` on termination.
  Returns a dict with:
  - `mean_travel_time`, `std_travel_time`, `median_travel_time`, `min_travel_time`, `max_travel_time`
  - `controlled_mean_travel_time`, `controlled_arrivals`
  - `background_mean_travel_time`, `background_arrivals`
  - `mean_waiting_time`, `total_waiting_time`
  - `total_stops`, `total_emissions`
  - `arrivals`, `total_spawned`, `arrival_rate`
  - `episode_steps`, `sim_time`
- `step()` — now appends `info["episode_metrics"]` when `terminated=True`.

---

### 2. Baseline Controllers
**New file:** `pymarl/src/controllers/baseline_controller.py`
**Modified file:** `pymarl/src/controllers/__init__.py`

`BaselineMAC` is a drop-in replacement for `BasicMAC` that implements three
non-learning policies. It matches the `BasicMAC` interface (same method signatures)
so it can be used anywhere a standard MAC is expected.

| Policy (`baseline_policy` key) | Behaviour |
|---|---|
| `"noop"` | Always returns action `0` (keep current route). Equivalent to SUMO default routing when `action_noop_as_keep_route=True`. |
| `"greedy_shortest"` | For each active agent, reads `env.route_candidates` and `env.route_masks`, computes the length of every available route using `sumolib`, and selects the shortest one. Requires `set_env(env)` to be called once before evaluation begins. |
| `"random"` | Samples uniformly from the set of available actions (`avail_actions > 0`) for each agent each step. |

**`set_env(env)`** — must be called for the `greedy_shortest` policy so the controller
can access the environment's route candidate cache directly.

`__init__.py` now exports both `BasicMAC` and `BaselineMAC`.

---

### 3. Evaluation Script
**New file:** `pymarl/src/evaluate.py`

Standalone script that loads either a trained QMIX checkpoint or a baseline policy,
runs evaluation episodes, and saves results.

**Key features:**

- Loads algorithm and environment configs from `config/algs/qmix_sumo.yaml` and
  `config/envs/sumo_grid4x4.yaml`.
- For QMIX: loads `agent.pth` and `mixer.pth` from a checkpoint directory and sets
  `epsilon = 0.0` (greedy evaluation).
- For baselines: instantiates `BaselineMAC` with the chosen policy.
- Runs `eval_episodes` episodes in `test_mode=True`, unpacking the
  `(batch, ep_metrics)` tuple returned by the runner.
- Collects per-episode values for six traffic metrics defined in `METRIC_KEYS`:
  `mean_travel_time`, `mean_waiting_time`, `total_stops`, `total_emissions`,
  `arrival_rate`, `controlled_mean_travel_time`.
- Computes `mean`, `std`, `median`, `min`, `max` across episodes.
- Saves a JSON result file to `results/eval/<policy>_seed<N>.json`.
- Prints a formatted traffic-metrics summary table.

**`compare_results(result_files)`** — comparison mode (`--compare`):
- Loads multiple JSON result files.
- Prints a side-by-side table (return, travel time, arrival rate).
- Runs pairwise Welch's t-tests for return and travel time, annotated with
  significance stars (`*`, `**`, `***`, `ns`).

**CLI:**
```
python evaluate.py --model  results/models/best    --episodes 100 --seed 42
python evaluate.py --baseline noop                 --episodes 100 --seed 42
python evaluate.py --baseline greedy_shortest      --episodes  50
python evaluate.py --compare results/eval/qmix_seed42.json results/eval/noop_seed42.json
```

---

### 4. Validation & Best-Model Selection
**Modified file:** `pymarl/src/main.py`

The training loop now runs periodic **validation** episodes (separate from the existing
test episodes) and saves the best model based on validation performance.

**New parameters (from config or CLI):**

| Parameter | Default | Meaning |
|---|---|---|
| `use_validation` | `True` | Enable/disable validation runs |
| `validation_interval` | `50000` | Steps between validation runs |
| `validation_nepisode` | `10` | Episodes per validation run |

**New state variables:**
- `last_validation_t` — tracks when the last validation ran.
- `best_validation_return` — the best validation metric seen so far (initialised to `-inf`).
- `best_model_t` — the timestep at which the best model was found.

**Validation metric selection:**
- Primary: **mean travel time** (lower is better → negated internally so `>` comparison works).
- Fallback: episode return if travel time is unavailable (e.g. no vehicles completed trips).

**Model saving:**
- Best model is saved to `<checkpoint_path>/best/` whenever the validation metric improves.
- Periodic saves and the final save at `<checkpoint_path>/step_<t>/` and `final/` are unchanged.

**New CLI flags:**
- `--no_validation` — disables validation entirely.
- `--validation_interval <N>` — override interval.
- `--validation_nepisode <N>` — override number of validation episodes.

**Logging additions:**
- `validation_return_mean` — mean return across validation episodes.
- `validation_mean_travel_time` — mean travel time across validation episodes.
- `best_validation_metric` — best metric seen (logged each time a new best is found).
- Aggregate test-time traffic metrics (`test_mean_travel_time_avg`, `test_arrival_rate_avg`, etc.)
  are now logged alongside the existing `test_return_mean`.

---

### 5. Multi-Seed Experiment Runner
**New file:** `pymarl/src/run_experiments.py`

Orchestrates a full multi-seed experiment end-to-end: train → evaluate → aggregate → report.

**Three phases:**

**Phase 1 — Training**
Spawns one `main.py` subprocess per seed. Each seed's models are saved under
`<checkpoint_root>/seed_<N>/`. The best model (`best/`) is preferred over the final
model (`final/`) when selecting the checkpoint to evaluate.

**Phase 2 — Evaluation**
For each seed, calls `evaluate.py` as a subprocess for:
- The QMIX model trained in Phase 1 (or loaded from disk in `--eval_only` mode).
- Each baseline policy (`noop`, `greedy_shortest`, optionally `random`).

Result JSON files are written to `results/eval/`.

**Phase 3 — Aggregation & Reporting**
- Loads all per-seed result JSONs and computes **cross-seed** statistics:
  `mean`, `std`, `ci95` (95% confidence interval = 1.96 × SE) for episode return
  and all six traffic metrics.
- Prints a formatted summary table for each policy.
- Runs pairwise t-tests between all pairs of policies for return, travel time,
  and arrival rate.
- Saves a `experiment_summary.json` to the checkpoint root.

**CLI:**
```
# Full experiment
python run_experiments.py --seeds 5 --t_max 1000000 --eval_episodes 50

# Quick smoke test (2 seeds, 10k steps)
python run_experiments.py --seeds 2 --t_max 10000 --eval_episodes 5

# Evaluation only (reuse existing models)
python run_experiments.py --eval_only --checkpoint_root results/experiments/20260323_120000
```

---

## File Change Summary

| File | Status | Description |
|---|---|---|
| `pymarl/src/envs/sumo_grid_reroute.py` | Modified | Episode-level metrics accumulation and reporting |
| `pymarl/src/main.py` | Modified | Validation loop, best-model saving, new CLI flags |
| `pymarl/src/runners/episode_runner.py` | Modified | Returns `(batch, episode_metrics)` tuple; logs per-episode traffic metrics |
| `pymarl/src/controllers/__init__.py` | Modified | Exports `BaselineMAC` alongside `BasicMAC` |
| `pymarl/src/controllers/baseline_controller.py` | **New** | `BaselineMAC`: noop, greedy_shortest, random policies |
| `pymarl/src/evaluate.py` | **New** | Evaluation script for QMIX and baselines |
| `pymarl/src/run_experiments.py` | **New** | Multi-seed experiment orchestrator |

---

## Metrics Tracked

| Metric | Unit | Source |
|---|---|---|
| Mean travel time | seconds | Arrival time − spawn time, averaged over completed vehicles |
| Mean waiting time | seconds | `traci.vehicle.getWaitingTime()` at arrival, averaged |
| Total stops | count | Cumulative vehicles with speed < threshold per step |
| Total emissions | grams CO₂ | `traci.vehicle.getCO2Emission()` accumulated over episode |
| Arrival rate | 0–1 | Vehicles arrived ÷ total vehicles spawned |
| Controlled mean travel time | seconds | Travel time for agent-controlled vehicles only |

---

## Usage Example: Full Comparison

```bash
# 1 — Train QMIX with validation (single seed)
python main.py --seed 42 --t_max 500000

# 2 — Evaluate QMIX best model
python evaluate.py --model results/models/best --episodes 50 --output qmix

# 3 — Evaluate baselines
python evaluate.py --baseline noop           --episodes 50 --output noop
python evaluate.py --baseline greedy_shortest --episodes 50 --output greedy

# 4 — Compare all three
python evaluate.py --compare \
    results/eval/qmix_seed42.json \
    results/eval/noop_seed42.json \
    results/eval/greedy_seed42.json

# 5 — Full multi-seed experiment (5 seeds)
python run_experiments.py --seeds 5 --t_max 1000000 --eval_episodes 50
```
