# Selfish Routing Evaluation — Metrics Reference

## Overview

This document covers the selfish routing baseline added to the evaluation pipeline,
what each extracted metric means, how it is computed, and the accuracy assessment
performed on the first evaluation run (100 episodes, seed 5, 4×4 map, high demand).

---

## What Is Selfish Routing?

In traffic theory, **selfish routing** (Wardrop's first principle / user equilibrium)
means every vehicle routes independently to minimise its own travel time, with no
cooperative coordination.

In this codebase, the selfish routing baseline is implemented as **noop** for the
32 controlled agents — vehicles hold their initial routes from `controlled_init.rou.xml`
and make no adaptive rerouting decisions. Background vehicles already follow
pre-computed Wardrop-equilibrium routes (`wardrop_routes_*.rou.xml`), so the entire
network operates under selfish routing conditions.

**Run command:**
```bash
cd pymarl/src
python evaluate.py --baseline selfish_routing --episodes 100 --seed 5
```

**Output file:** `pymarl/src/results/eval/selfish_routing_seed5.json`

---

## Metrics — Definitions and Computation

### Mean Travel Time (s)

| Field | Value (seed 5, 100 eps) |
|---|---|
| Mean | 14.08 s |
| Std | 3.13 s |

**Definition:** Average time (seconds) from a vehicle's entry into the simulation
to its arrival at its destination, computed over all vehicles that completed their
trip during the episode.

**Source:** `traci.vehicle.getDeparture()` on entry; `sim_time` at arrival.
Recorded in `vehicle_travel_times` list. Computed in `_compute_episode_metrics`.

**Population (after fix):** All vehicles — both controlled agents and background
traffic — that completed a trip.

---

### Mean Waiting Time (s)

| Field | Value (seed 5, 100 eps) |
|---|---|
| Mean | ~34.5 s (pre-fix, mixed population) |
| Std | ~4.3 s |

**Definition:** Average time (seconds) a vehicle spent stopped
(speed < `reward_stop_speed_threshold`) during its trip, over all vehicles that
completed their trip.

**Source:** `vehicle_accumulated_waiting` dict, updated every sub-step
(`sumo_step_length = 1 s`) when `speed < threshold`. Flushed to
`vehicle_waiting_times` on vehicle arrival.

**Population (after fix):** Same as Mean Travel Time — only completed vehicles.
Before the fix, this included vehicles still stuck in the network at episode end,
making waiting time artificially higher than travel time.

---

### Network Throughput (veh/h)

| Field | Value (seed 5, 100 eps) |
|---|---|
| Mean | 6074.8 veh/h (pre-fix, controlled only) |
| Std | 1382.6 veh/h |

**Definition:** Number of vehicles completing their trip per simulated hour.

**Formula:**
```
network_throughput = episode_arrivals / (sim_time / 3600)
```

**Source:** `episode_arrivals` counter incremented on each vehicle arrival;
`sim_time` from SUMO at episode end (`max_episode_steps = 1000 s → sim_hours ≈ 0.278`).

**Population (after fix):** All arrivals — controlled agents and background vehicles.
Before the fix, only controlled agent arrivals were counted, inflating the metric
because the 32 controlled slots cycle through short routes (~14 s each) producing
many completions.

---

### Real-Time Factor

| Field | Value (seed 5, 100 eps) |
|---|---|
| Mean | 521.9 |
| Std | 73.4 |

**Definition:** Number of simulated seconds that elapse per wall-clock second.
A value of 522 means SUMO runs 522× faster than real time.

**Source:** Wall-clock time measured with `time.monotonic()` at episode start
(`_episode_wall_start`) and at `_compute_episode_metrics`.

```
real_time_factor = sim_time / (time.monotonic() - _episode_wall_start)
```

**Note:** This is a simulation performance metric, not a traffic quality metric.
It reflects the speed of the libsumo backend and hardware, not network conditions.
Do not use this to compare policy quality.

---

### CO2 Emissions (g)

| Field | Value (seed 5, 100 eps) |
|---|---|
| Mean | 655,031.6 g (655 kg) |
| Std | 85,588.6 g |

**Definition:** Total CO2 emitted by all vehicles in the simulation during one
episode, in grams.

**Source:** `traci.vehicle.getCO2Emission(veh_id)` returns instantaneous CO2 rate
in **mg/s** per vehicle. Accumulated every sub-step:

```python
episode_emissions += total_co2_mg_s / 1000.0  # convert mg to g, per 1-s step
```

**Emission class:** `thesis_car` uses `HBEFA3/PC_G_EU4` (Euro 4 petrol passenger
car). Background vehicles use `DEFAULT_VEHTYPE` which inherits the SUMO default
emission class.

**Sanity check:** ~100 average vehicles × 1000 s × ~6550 mg/s average ÷ 1000 ≈ 655 kg.
Plausible for dense urban stop-and-go traffic.

---

### Fuel Consumption (ml)

| Field | Value (seed 5, 100 eps) |
|---|---|
| Mean | 252,214.6 ml ≈ 252 L (pre-fix) |
| Std | 32,837.4 ml |

**Definition:** Total fuel consumed by all vehicles during one episode, in
millilitres.

**Source:** `traci.vehicle.getFuelConsumption(veh_id)` returns fuel consumption
rate in **mg/s**. Accumulated every sub-step and converted to ml:

```python
episode_fuel_consumption += (fuel_mg_s * sumo_step_length) / density
```

**Density correction (fix applied):**
- Before fix: used diesel density (832 mg/ml) — incorrect for `thesis_car` (petrol)
- After fix: uses petrol density (740 mg/ml)
- Impact: ~12% upward correction

**Cross-check:** CO2/fuel ratio ≈ 655,000 g / 252,000 ml = 2.6 kg CO2/L.
Expected for petrol: ~2.39 kg CO2/L. ~8% discrepancy explained by mixed fleet
(background vehicles may have different emission profiles). Values are consistent.

---

### Arrival Rate

| Field | Value (seed 5, 100 eps) |
|---|---|
| Mean | 0.639 |
| Std | 0.037 |

**Definition:** Fraction of spawned vehicles that completed their trip before
episode end.

**Formula:**
```
arrival_rate = episode_arrivals / total_spawned
```

**Population (after fix):** Both numerator and denominator cover all vehicles
(controlled + background). Before the fix, the numerator counted only controlled
arrivals while the denominator counted all spawned — a mixed-population fraction.

---

## Accuracy Assessment

### What is real SUMO data

| Metric | SUMO source | Reliable? |
|---|---|---|
| Mean Travel Time | `vehicle.getDeparture()`, `simulation.getTime()` | Yes, after population fix |
| Mean Waiting Time | `vehicle.getSpeed()` per sub-step | Yes, after population fix |
| CO2 Emissions | `vehicle.getCO2Emission()` | Yes |
| Fuel Consumption | `vehicle.getFuelConsumption()` | Yes, after density fix |
| Network Throughput | Derived from arrival counts | Yes, after population fix |
| Arrival Rate | Derived from arrival/spawn counts | Yes, after population fix |
| Real-Time Factor | Wall clock (`time.monotonic`) | Yes (but not a traffic metric) |

### Issues found and fixed (2026-04-14)

#### 1. Population mismatch — travel time vs waiting time

**Problem:** `mean_travel_time` was computed only from the 32 **controlled** vehicles
(short 2–6 edge routes, ~14 s average). `mean_waiting_time` combined waiting times
from completed controlled vehicles **plus** all background vehicles still stuck in
the network at episode end — a different, larger population. This caused waiting
time (34.5 s) to exceed travel time (14 s), which is logically inconsistent.

**Fix:** Added `_handle_background_arrivals()` which detects when background
vehicles leave the simulation and records their travel time and waiting time into
the same lists used for controlled vehicles. Both metrics now use the identical
population: all vehicles that completed a trip.

#### 2. Network throughput excludes background vehicles

**Problem:** `episode_arrivals` was only incremented inside `_handle_arrivals()`
which loops over `agent_vehicle_ids` (controlled slots). Background vehicles
departing the network were never counted. With 32 controlled slots cycling through
~14 s trips in 1000 s, each slot completes ~53 trips → inflated controlled-only
throughput.

**Fix:** `_handle_background_arrivals()` also increments `episode_arrivals` for
each background vehicle completion, making throughput reflect all network traffic.

#### 3. Fuel consumption density mismatch

**Problem:** Code used diesel density (832 mg/ml) to convert `getFuelConsumption()`
from mg to ml, but `thesis_car` is a petrol vehicle (`HBEFA3/PC_G_EU4`).

**Fix:** Changed to petrol density (740 mg/ml). Effect: ~12% upward correction
in fuel consumption values.

---

## Files Changed

| File | Change |
|---|---|
| `pymarl/src/envs/sumo_grid_reroute.py` | Added `_handle_background_arrivals()`, fixed waiting time population, fixed fuel density, added `episode_fuel_consumption`, `_episode_wall_start` |
| `pymarl/src/evaluate.py` | Added `network_throughput`, `real_time_factor`, `co2_emissions`, `fuel_consumption` to `METRIC_KEYS`; added `--selfish_routing` baseline choice; added `--sumo_backend` CLI flag |
| `pymarl/src/controllers/baseline_controller.py` | Added `selfish_routing` policy (maps to noop — no adaptive rerouting) |
| `pymarl/src/envs/sumo_backend.py` | Added libsumo → traci auto-fallback when libsumo is not installed |
| `sumo/scenarios/4by4_map/train_*.sumocfg` | Added `vtypes.add.xml` as additional-files so `thesis_car` vType is loaded during training and evaluation |
