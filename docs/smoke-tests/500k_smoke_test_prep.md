# 500k Smoke Test Prep: Watchdog, Waiting Time Fix, and Congestion Diagnostics

**Date:** 2026-03-28
**Context:** Follow-up to the 50k smoke test (2026-03-23). Three issues identified from that run were addressed before running the next smoke test at 500,000 steps with a single seed.

---

## 1. Background

The 50k smoke test surfaced three problems that needed to be resolved before increasing training length:

| Issue | Impact |
|---|---|
| Seed 44 hung indefinitely with no timeout guard | Any of the 3 seeds in a 2M-step run could block for days without detection |
| `mean_waiting_time` always 0.0 | One of the primary KPIs in thesis Table 3.3 was missing from all evaluation results |
| No way to confirm the network was actually congesting | Could not distinguish "QMIX hasn't learned yet" from "reward signal is flat by design" |

All three were addressed in `pymarl/src/envs/sumo_grid_reroute.py`.

---

## 2. SUMO Subprocess Watchdog

### 2.1 Problem

Seed 44 in the 50k smoke test hung after ~105 minutes. The SUMO subprocess stalled waiting on a TraCI response — `traci.simulationStep()` blocked indefinitely. Python's `subprocess.run()` has no default timeout, so the process stayed alive and produced no output for over 100 minutes. It had to be killed manually. At 2M-step scale across 3 seeds, an undetected hang on seed 1 could waste days.

### 2.2 Design Decision: Heartbeat Thread vs. Per-Step Timeout

An earlier approach wrapped each `traci.simulationStep()` in a new `threading.Thread` with a per-call timeout. This was rejected for two reasons:

1. **Thread overhead** — spawning and joining a thread on every simulation sub-step (up to 10 per decision step × 1000 steps per episode) creates thousands of threads per episode.
2. **False positives** — a per-step wall-clock limit cannot distinguish a legitimately slow step (high vehicle density) from a hung step. Both exceed a fixed threshold.

The accepted design uses a **single background watchdog thread** that monitors a rolling heartbeat timestamp. The main thread updates the heartbeat *before* each `traci.simulationStep()`. The watchdog polls every 5 seconds and fires only when the silence between heartbeat updates exceeds `sumo_step_timeout`. A slow-but-progressing step continuously advances the heartbeat, so it never triggers; only a step that never returns will.

### 2.3 Implementation

**File:** `pymarl/src/envs/sumo_grid_reroute.py`

**New state fields (added to `__init__`):**

| Field | Type | Purpose |
|---|---|---|
| `sumo_step_timeout` | `int` | Seconds of TraCI silence before watchdog fires (default: `60`, configurable via `env_args["sumo_step_timeout"]`) |
| `_traci_heartbeat` | `float` | Monotonic timestamp of the last heartbeat write |
| `_watchdog_thread` | `Thread \| None` | Reference to the running watchdog thread |
| `_watchdog_stop` | `threading.Event` | Signalled to shut the watchdog down cleanly |
| `_watchdog_fired` | `bool` | Set to `True` when the watchdog triggers |

**New methods:**

- **`_start_watchdog()`** — Calls `_stop_watchdog()` first (ensures any stale thread from a previous episode is gone), resets the heartbeat timestamp, clears `_watchdog_stop`, and starts the daemon thread.
- **`_stop_watchdog()`** — Sets `_watchdog_stop` and joins the thread with a 2-second timeout.
- **`_watchdog_loop()`** — Background thread body. Polls `_watchdog_stop` every 5 seconds. If `time.monotonic() - _traci_heartbeat > sumo_step_timeout`, sets `_watchdog_fired = True`, calls `traci.close()` (which unblocks any pending `simulationStep()` call on the main thread with a socket error), and exits.

**Modified methods:**

- **`reset()`** — Calls `_start_watchdog()` immediately after `_start_sumo()`, so the watchdog is active during warmup steps as well as the main episode loop.
- **`close()`** — Calls `_stop_watchdog()` before closing the TraCI connection.
- **`_advance_simulation()`** — Writes `self._traci_heartbeat = time.monotonic()` before each `traci.simulationStep()` call. No change to the call itself.

### 2.4 Failure Propagation

When the watchdog fires:
1. `traci.close()` is called from the watchdog thread.
2. The blocked `traci.simulationStep()` on the main thread raises a TraCI socket exception.
3. The exception propagates up: `_advance_simulation()` → `step()` → `EpisodeRunner.run()` → `main.py`'s `while` training loop.
4. `main.py`'s `finally` block runs unconditionally, calls `runner.close_env()` → `env.close()` → `_stop_watchdog()`. The watchdog thread has already exited (called `break`), so the join returns immediately.
5. The training subprocess exits with a non-zero return code.
6. `run_experiments.py` logs the warning and continues to the next seed.

No manual intervention is required.

### 2.5 Lifecycle Safety

Two layers protect against a stale watchdog firing on a subsequent valid episode:

- `_start_watchdog()` always calls `_stop_watchdog()` first. Even if `close()` was not called after a previous episode, the next `reset()` cleans up the stale thread.
- `main.py`'s `finally` block covers all exit paths (normal completion, `KeyboardInterrupt`, and any uncaught exception), ensuring `close()` is always called before the process exits.

---

## 3. Waiting Time Fix (`mean_waiting_time`)

### 3.1 Problem

`mean_waiting_time` was always `0.0` in all evaluation results. The original `_handle_arrivals()` attempted to call `traci.vehicle.getWaitingTime(vehicle_id)` after the vehicle had already left the simulation (detection condition: `vehicle_id not in current_vehicles`). TraCI cannot query a vehicle that has already been unloaded, so the call was either silently failing or was removed in an earlier cleanup. The metric was left at its default zero.

This matters because `mean_waiting_time` is a primary KPI in thesis Table 3.3. It was missing from all policy comparisons.

### 3.2 Design Decision: Manual Sub-Step Accumulation

Three TraCI approaches were considered:

| Approach | Problem |
|---|---|
| `getWaitingTime()` at arrival | Vehicle already unloaded — call fails |
| `getWaitingTime()` at each sub-step | Returns current *continuous* wait streak; resets when vehicle moves. Summing gives the total of wait streaks, not total waiting time across the episode. |
| `getAccumulatedWaitingTime()` | Returns a windowed sum (default window: 100s), not episode-total |

The accepted approach: maintain a `vehicle_accumulated_waiting` dict keyed by vehicle ID. At each simulation sub-step, for each vehicle whose speed is below `reward_stop_speed_threshold`, add `sumo_step_length` to that vehicle's accumulator. This gives a true episode-total waiting time that:
- Persists across move/stop cycles within the episode
- Is readable at any point while the vehicle is active
- Does not depend on a TraCI call after the vehicle has left

### 3.3 Implementation

**File:** `pymarl/src/envs/sumo_grid_reroute.py`

**New state field:**

| Field | Type | Purpose |
|---|---|---|
| `vehicle_accumulated_waiting` | `dict[str, float]` | Running waiting-time sum (seconds) per vehicle ID for the current episode |

Cleared in `reset()`. Never cleared mid-episode.

**Modified methods:**

- **`_track_new_vehicles()`** — Added `self.vehicle_accumulated_waiting.setdefault(veh_id, 0.0)` for each new vehicle. This initialises an entry for every spawned vehicle, including those that never stop, so the denominator in the episode mean includes all vehicles, not just those that waited.

- **`_advance_simulation()`** — The per-vehicle speed loop (previously only used for the reward stops count) now also increments `vehicle_accumulated_waiting[veh_id]` by `sumo_step_length` for each vehicle with speed below threshold. This is folded into an existing loop — no additional TraCI calls.

- **`_handle_arrivals()`** — When a vehicle is detected as departed (any reason: normal arrival, SUMO teleport, collision), calls `self.vehicle_accumulated_waiting.pop(vehicle_id, 0.0)` and appends the value to `self.vehicle_waiting_times`. The value is captured while the dict entry still exists, before the vehicle leaves.

- **`_compute_episode_metrics()`** — Replaced the previous `vehicle_waiting_times`-only mean with a combined list:
  ```python
  all_waiting = list(self.vehicle_waiting_times)         # arrived during episode
  all_waiting.extend(self.vehicle_accumulated_waiting.values())  # still active at episode end
  ```
  The `if wt > 0.0` filter that was present in an intermediate version was removed. Vehicles that never stopped have `wt = 0.0` and must be included in the denominator for the mean to be correct.

### 3.4 Coverage

All vehicles that appear during an episode are counted:

| Vehicle fate | How waiting time is captured |
|---|---|
| Arrives normally | Popped from `vehicle_accumulated_waiting` in `_handle_arrivals()`, appended to `vehicle_waiting_times` |
| Removed mid-episode (teleport, collision, etc.) | Same as above — `_handle_arrivals()` fires on any disappearance from `current_vehicles` |
| Still active at episode end (truncation) | Included via `vehicle_accumulated_waiting.values()` in `_compute_episode_metrics()` |
| Spawned but never stopped | Initialised to `0.0` by `setdefault` in `_track_new_vehicles()`, contributing `0.0` to the mean |

---

## 4. Congestion Diagnostics

### 4.1 Problem

With all three evaluation policies producing identical results at 50k steps, it was unclear whether the reward signal would ever differentiate them at longer training. One possible explanation was that the BGC scenario was under-congested: if vehicles routed through freely with no meaningful delays, any routing policy — including no-op — would produce the same travel time. QMIX would have nothing to learn regardless of training length.

No episode-level aggregate statistics (mean vehicle count, mean speed) were previously being tracked, so this could not be confirmed from existing output.

### 4.2 Implementation

**File:** `pymarl/src/envs/sumo_grid_reroute.py`

**New state fields (added to `__init__`, reset in `reset()`):**

| Field | Type | Purpose |
|---|---|---|
| `_sub_step_count` | `int` | Number of simulation sub-steps completed in the episode |
| `_total_vehicle_steps` | `int` | Sum of vehicle counts across all sub-steps |
| `_total_speed_sum` | `float` | Sum of all individual vehicle speeds across all sub-steps |
| `_total_speed_veh_steps` | `int` | Count of (vehicle, sub-step) pairs where a speed was successfully read |

These are accumulated inside the existing per-vehicle speed loop in `_advance_simulation()` — the same loop that handles waiting time accumulation. No additional TraCI calls are made.

**Modified method — `_compute_episode_metrics()`:**

Two new keys are added to the metrics dict:

| Key | Formula | Interpretation |
|---|---|---|
| `mean_vehicle_count` | `_total_vehicle_steps / _sub_step_count` | Average number of vehicles in the network per simulation second |
| `mean_speed` | `_total_speed_sum / _total_speed_veh_steps` | Average speed across all vehicles and all sub-steps (m/s) |

If `mean_speed` is near the network's free-flow speed limit and `mean_vehicle_count` is low, the scenario is under-congested and the reward signal will be flat regardless of policy.

**Verbose per-step logging:**

When `verbose: true` is set in the environment config, `_advance_simulation()` logs one line at the start of each decision period:

```
[congestion] t=120s  vehicles=28  mean_speed=8.42m/s
```

This is logged once per decision step (not per sub-step) to avoid flooding output.

---

## 5. File Change Summary

| File | Change |
|---|---|
| `pymarl/src/envs/sumo_grid_reroute.py` | All changes — watchdog, waiting time accumulation, congestion tracking |

All other files are unchanged.

**New instance variables added to `sumo_grid_reroute.py`:**

| Variable | Section |
|---|---|
| `sumo_step_timeout` | Watchdog config |
| `_traci_heartbeat`, `_watchdog_thread`, `_watchdog_stop`, `_watchdog_fired` | Watchdog state |
| `vehicle_accumulated_waiting` | Waiting time accumulation |
| `_sub_step_count`, `_total_vehicle_steps`, `_total_speed_sum`, `_total_speed_veh_steps` | Congestion tracking |

**New methods added:**

| Method | Purpose |
|---|---|
| `_start_watchdog()` | Start background watchdog for current episode |
| `_stop_watchdog()` | Signal watchdog to stop and join |
| `_watchdog_loop()` | Watchdog thread body — polls heartbeat, fires on silence |

---

## 6. Updated Metrics Table

`_compute_episode_metrics()` now returns the following keys:

| Key | Unit | Status |
|---|---|---|
| `mean_travel_time` | seconds | Unchanged |
| `std_travel_time` | seconds | Unchanged |
| `median_travel_time` | seconds | Unchanged |
| `min_travel_time` | seconds | Unchanged |
| `max_travel_time` | seconds | Unchanged |
| `controlled_mean_travel_time` | seconds | Unchanged |
| `controlled_arrivals` | count | Unchanged |
| `background_mean_travel_time` | seconds | Unchanged |
| `background_arrivals` | count | Unchanged |
| `mean_waiting_time` | seconds | **Fixed** — was always 0.0 |
| `total_waiting_time` | seconds | **Fixed** — was always 0.0 |
| `total_stops` | count | Unchanged |
| `total_emissions` | grams CO₂ | Unchanged |
| `arrivals` | count | Unchanged |
| `total_spawned` | count | Unchanged |
| `arrival_rate` | 0–1 | Unchanged |
| `episode_steps` | count | Unchanged |
| `sim_time` | seconds | Unchanged |
| `mean_vehicle_count` | count | **New** |
| `mean_speed` | m/s | **New** |

---

## 7. 500k Smoke Test Run Command

```bash
python run_experiments.py --seeds 1 --first_seed 42 --t_max 500000 --eval_episodes 5
```

One seed is sufficient to confirm that reward curves diverge past the epsilon annealing cutoff (~500k steps) and that the three policies produce differentiated results. Running 3 seeds is not necessary until the full 2M-step training run.

**To validate congestion before training:**

Set `verbose: true` in `config/envs/sumo_grid4x4.yaml` and run a single short episode. Check `mean_vehicle_count` and `mean_speed` in the printed episode metrics. If `mean_speed` is close to the free-flow speed limit throughout, the network is not congesting and the vehicle density should be increased before the full run.

```bash
python run_experiments.py --seeds 1 --first_seed 42 --t_max 2000 --eval_episodes 1
```
