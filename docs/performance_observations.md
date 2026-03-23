# Performance Observations and Bug Fixes

**Date:** 2026-03-23
**Context:** First smoke test run — 3 seeds × 50,000 steps, on Intel i7-H (CPU only)

---

## 1. Observed Throughput

Measured during seed 42 training:

| Metric | Value |
|---|---|
| Throughput | ~16.6 steps/sec |
| Time per seed (50k steps) | ~50 min |
| Time per seed (2M steps, full training) | ~33.5 hrs |
| Total wall time for 3 seeds sequential (2M steps) | ~100 hrs |
| Evaluation overhead (50 eps × 3 seeds × 3 policies) | ~7.5 hrs |
| **Estimated full experiment wall time** | **~107 hrs (~4.5 days)** |

One "step" here is one agent decision step, which internally advances SUMO by `decision_period / sumo_step_length = 10` simulation sub-steps.

---

## 2. Root Cause of Slowness

The bottleneck is **SUMO**, not the neural network. The QMIX model (GRU hidden=64, mixer=32) is tiny — a forward+backward pass is under 1ms. The process spends the majority of its time:

1. **Waiting on SUMO via TraCI IPC** — every `traci.*` call is a round-trip to a separate SUMO subprocess over a local socket.
2. **k-shortest path computation** — `sumolib.net.getShortestPath()` called for all 32 agents at every decision step.
3. **XML parsing in the hot loop** — `_load_od_pairs_from_routes()` was re-parsing the routes XML file from disk every time a replacement vehicle was spawned.

A GPU does **not** address this bottleneck. The simulation is single-threaded C++ (SUMO), and TraCI IPC is the wall. GPU would accelerate the network forward/backward pass, which is not the limiting factor.

---

## 3. Bugs Fixed (2026-03-23)

### 3.1 "Vehicle not known" Error Flood

**File:** `pymarl/src/envs/sumo_grid_reroute.py`, `_handle_arrivals()`

**Problem:** `traci.vehicle.getWaitingTime(vehicle_id)` was called immediately after confirming the vehicle had already left the simulation (`vehicle_id not in traci.vehicle.getIDList()`). Python's `try/except` caught the resulting exception silently, but SUMO itself still printed the error to stderr, flooding the output with thousands of lines per episode:

```
Error: Answered with error to command 0xa4: Vehicle 'controlled_X' is not known.
```

This made it impossible to spot real errors in the logs.

**Fix:** Removed the `getWaitingTime` call on departed vehicles. Waiting time cannot be retrieved after a vehicle has left the simulation; the vehicle is already unloaded from SUMO's internal state by the time arrival is detected.

**Note:** `mean_waiting_time` in episode metrics will now always be 0.0 until a proper mid-episode accumulation strategy is implemented (e.g. calling `getWaitingTime` at each step while the vehicle is still active).

---

### 3.2 XML File Parsed on Every Vehicle Replacement

**File:** `pymarl/src/envs/sumo_grid_reroute.py`, `_load_od_pairs_from_routes()`

**Problem:** `_load_od_pairs_from_routes()` used `xml.etree.ElementTree.parse()` to read the controlled routes XML file from disk every time it was called. It was called inside `_handle_arrivals()`, which runs every SUMO sub-step for every inactive agent slot. With 32 agents and replacement enabled, this could be called dozens of times per episode step — all reading and parsing the same file repeatedly.

**Fix:** Added an instance-level cache `self._od_pairs_cache`. The file is parsed once on first call; all subsequent calls return the cached list immediately.

---

### 3.3 `getIDList()` Called 34+ Times Per SUMO Sub-Step

**File:** `pymarl/src/envs/sumo_grid_reroute.py`, `_advance_simulation()`

**Problem:** Three methods each independently called `traci.vehicle.getIDList()`:

| Method | Calls per sub-step |
|---|---|
| `_compute_reward()` | 1 |
| `_handle_arrivals()` | 1 per agent slot (up to 32) |
| `_track_new_vehicles()` | 1 |

Each call is a TraCI round-trip (Python → socket → SUMO → socket → Python). With `decision_period=10` sub-steps per agent step, this amounted to ~340 redundant IPC calls per agent decision step.

**Fix:** `_advance_simulation()` now calls `getIDList()` exactly once per sub-step and passes the result (as a Python `set`) to `_compute_reward()`, `_handle_arrivals()`, and `_track_new_vehicles()` as a parameter. All three methods were updated to accept `current_vehicles: set` instead of fetching it themselves.

---

### 3.4 Unicode Characters Breaking Output on Windows

**Files:** `pymarl/src/run_experiments.py`, `pymarl/src/main.py`

**Problem:** Both files used Unicode box-drawing characters (`─` U+2500) and arrows (`→` U+2192) in print statements. Windows uses cp1252 encoding by default for stdout, which does not support these characters. The first smoke test run crashed immediately:

```
UnicodeEncodeError: 'charmap' codec can't encode characters in position 2-61: character maps to <undefined>
```

**Fix:** Replaced all `─` with `-` and `→` with `->` throughout both files.

---

## 4. Cloud / Hardware Recommendations

Since SUMO is the bottleneck (not the network), the most effective strategies are:

### Run seeds in parallel on separate machines
Currently seeds run sequentially via `subprocess.run()`. The biggest wall-time win is running each seed on a separate VM simultaneously:

- 3 seeds sequential: ~100 hrs wall time
- 3 seeds parallel (3 VMs): ~34 hrs wall time, same total cost

### Use compute-optimized CPU instances (not GPU)
A GPU does not help at current model sizes. Recommended cloud instances:

| Provider | Instance | Notes |
|---|---|---|
| AWS | `c7i.xlarge` | Intel, high single-core clock |
| GCP | `c3-highcpu-4` | Good single-thread performance |
| Azure | `Fsv2` series | Compute-optimized |
| Hetzner | `CCX13` | Low cost option |

### Reduce wall time by tuning config
Without changing architecture:

| Lever | Effect |
|---|---|
| Reduce `episode_limit` (e.g. 500) | Proportional speedup, shorter episodes |
| Reduce `decision_period` (e.g. 5s) | 2× fewer SUMO sub-steps per agent step |
| Set `route_refresh_each_step: false` | Skip k-shortest recomputation mid-episode |
| Set `emissions_enabled: false` for smoke tests | Removes per-vehicle `getCO2Emission` call |

---

---

## 5. Smoke Test Incidents (2026-03-23)

### 5.1 Seed 44 Never Finished — Killed

**What happened:** The smoke test ran 3 seeds (42, 43, 44) sequentially. Seeds 42 and 43 each completed in ~132 minutes. Seed 44 ran for over 216 minutes without saving a `final` checkpoint, and its output file stopped growing entirely (line count frozen at 8,278 for multiple consecutive checks). The process was still alive (no crash notification received) but appeared hung — likely a SUMO subprocess that stalled waiting on a TraCI response.

**Action taken:** Seed 44 was manually killed. Evaluation was re-run using `--eval_only` on seeds 42 and 43 only:

```bash
python run_experiments.py --seeds 2 --first_seed 42 --eval_episodes 5 \
    --eval_only --checkpoint_root results/experiments/20260323_125850
```

**Impact:** Results table has n=2 seeds instead of n=3. Statistical tests (t-tests) are less reliable at n=2. For the full training run, monitor seed health and consider adding a timeout guard.

---

### 5.2 QMIX Evaluation Crash — `action_selector` AttributeError

**File:** `pymarl/src/evaluate.py`, line 131

**What happened:** When `evaluate.py` loaded a trained QMIX model and attempted to force greedy action selection, it crashed immediately:

```
AttributeError: 'BasicMAC' object has no attribute 'action_selector'
mac.action_selector.epsilon = 0.0
```

**Root cause:** `evaluate.py` assumed `BasicMAC` had a standalone `action_selector` object (matching an older PyMARL API). In this codebase, `BasicMAC` handles epsilon-greedy internally via `_get_epsilon()` and already switches to pure greedy when `test_mode=True` is passed to `select_actions()`. The explicit `epsilon = 0.0` assignment was unnecessary and incorrect.

**Fix:** Removed the offending line in `evaluate.py`:

```python
# Before (broken):
mac.action_selector.epsilon = 0.0

# After (correct):
# Greedy action selection is handled by test_mode=True in the runner
```

**Impact:** Seed 42's QMIX evaluation had already failed before the fix was applied and could not be recovered in this run. Seed 43's QMIX evaluation will use the fixed code. The aggregate QMIX results table will therefore have n=1 for QMIX (seed 43 only) vs n=2 for baselines.

---

## 6. Remaining Known Issues

| Issue | Location | Status |
|---|---|---|
| `mean_waiting_time` always 0.0 | `_handle_arrivals()` | Known — fix requires mid-episode accumulation |
| k-shortest paths duplicates first route for k>1 | `_compute_k_shortest_paths()` | Known — marked TODO in code |
| Seeds run sequentially | `run_experiments.py` | By design; parallelize via separate VMs |
| No timeout guard on SUMO subprocess | `sumo_grid_reroute.py` | Can cause seed to hang indefinitely (see §5.1) |
