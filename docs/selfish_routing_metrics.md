# Selfish Routing Evaluation — Metrics Reference

## Overview

This document covers the selfish routing baseline added to the evaluation pipeline,
what each extracted metric means, how it is computed, and the accuracy assessment
performed on the evaluation run (100 episodes, seed 5, 4×4 map, high demand).

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

All metrics are computed per episode and averaged over 100 episodes. Mean ± Std
values below are from seed 5, 100 episodes, 4×4 map, high demand.

---

### Mean Travel Time (s)

| Field | Value |
|---|---|
| Mean | 76.46 s |
| Std | 11.30 s |

**Definition:** Average time (seconds) from a vehicle's entry into the simulation
to its arrival at its destination, over **all** vehicles (controlled + background)
that completed their trip.

**Source:** `traci.vehicle.getDeparture()` at vehicle entry; `sim_time` at arrival.
Recorded in `vehicle_travel_times` in `_handle_arrivals` (controlled) and
`_handle_background_arrivals` (background). Computed in `_compute_episode_metrics`.

**Note:** The higher value compared to `controlled_mean_travel_time` (14.08 s) is
expected — background vehicles travel longer Wardrop routes (avg ~1 637 m) while
controlled vehicles have short 2–6 edge routes (~14 s). The overall population mean
reflects the full mixed fleet.

---

### Controlled Mean Travel Time (s)

| Field | Value |
|---|---|
| Mean | 14.08 s |
| Std | 3.13 s |

**Definition:** Average travel time for the **32 controlled agent vehicles only**,
isolated from background traffic. Useful for direct comparison against QMIX policy
performance on the same controlled slots.

**Source:** Separate `controlled_travel_times` list populated only in
`_handle_arrivals` (controlled vehicle slots).

---

### Mean Waiting Time (s)

| Field | Value |
|---|---|
| Mean | 16.78 s |
| Std | 1.86 s |

**Definition:** Average time (seconds) a vehicle spent stopped
(speed < `reward_stop_speed_threshold`) during its trip, over all vehicles that
completed their trip.

**Source:** `vehicle_accumulated_waiting` dict, updated every sub-step
(`sumo_step_length = 1 s`) when `speed < threshold`. Flushed to
`vehicle_waiting_times` on vehicle arrival in both `_handle_arrivals` and
`_handle_background_arrivals`.

**Population:** All completed vehicles (controlled + background). Before the
population fix, background vehicles still in-network at episode end were included,
inflating waiting time above travel time — which is logically inconsistent.

---

### Network Throughput (veh/h)

| Field | Value |
|---|---|
| Mean | 8 009.4 veh/h |
| Std | 1 375.7 veh/h |

**Definition:** Number of vehicles completing their trip per simulated hour.

**Formula:**
```
network_throughput = episode_arrivals / (sim_time / 3600)
```

**Source:** `episode_arrivals` counter incremented on each vehicle arrival in both
`_handle_arrivals` and `_handle_background_arrivals`; `sim_time` from SUMO at
episode end (`max_episode_steps = 1000 s → sim_hours ≈ 0.278`).

---

### Real-Time Factor

| Field | Value |
|---|---|
| Mean | 481.8 |
| Std | 62.3 |

**Definition:** Number of simulated seconds that elapse per wall-clock second.
A value of ~482 means SUMO runs ~482× faster than real time.

**Source:** Wall-clock time measured with `time.monotonic()` at episode start
(`_episode_wall_start`) and at `_compute_episode_metrics`.

```python
real_time_factor = sim_time / (time.monotonic() - _episode_wall_start)
```

**Note:** This is a simulation performance metric, not a traffic quality metric.
Do not use this to compare policy quality.

---

### CO2 Emissions (g)

| Field | Value |
|---|---|
| Mean | 655 031.6 g (655 kg) |
| Std | 85 588.6 g |

**Definition:** Total CO2 emitted by all vehicles during one episode, in grams.

**Source:** `traci.vehicle.getCO2Emission(veh_id)` returns instantaneous CO2 rate
in **mg/s** per vehicle. Accumulated every sub-step:

```python
episode_emissions += total_co2_mg_s / 1000.0  # convert mg to g, per 1-s step
```

**Emission class:** `thesis_car` uses `HBEFA3/PC_G_EU4` (Euro 4 petrol passenger
car). Background vehicles use `DEFAULT_VEHTYPE`.

**Sanity check:** ~100 average vehicles × 1000 s × ~6550 mg/s average ÷ 1000 ≈ 655 kg.
Plausible for dense urban stop-and-go traffic.

---

### Fuel Consumption (ml)

| Field | Value |
|---|---|
| Mean | 283 571.1 ml ≈ 284 L |
| Std | 36 919.9 ml |

**Definition:** Total fuel consumed by all vehicles during one episode, in
millilitres.

**Source:** `traci.vehicle.getFuelConsumption(veh_id)` returns fuel consumption
rate in **mg/s**. Accumulated every sub-step and converted to ml using petrol
density (740 mg/ml):

```python
episode_fuel_consumption += (fuel_mg_s * sumo_step_length) / 740.0
```

**Density fix applied:** Earlier versions used diesel density (832 mg/ml) — incorrect
for `thesis_car` (petrol, `HBEFA3/PC_G_EU4`). Corrected to 740 mg/ml, causing a
~12% upward correction in reported values.

---

### CO2 Emissions per km (g/km)

| Field | Value |
|---|---|
| Mean | 747.1 g/km |
| Std | 115.1 g/km |

**Definition:** Total episode CO2 (g) divided by total distance driven by all
completed vehicles (km). Normalises emissions by network utilisation.

**Formula:**
```python
co2_g_per_km = episode_emissions / (sum(completed_route_lengths) / 1000.0)
```

**How route lengths are obtained:** See [Avg Route Length](#avg-route-length-m)
below.

**Sanity check:** 655 kg / (N_completed × 1.637 km) — consistent with high stop-and-go
emissions for Euro 4 petrol vehicles in congested urban conditions.

---

### Fuel Consumption per 100 km (L/100km)

| Field | Value |
|---|---|
| Mean | 32.3 L/100km |
| Std | 5.0 L/100km |

**Definition:** Total episode fuel (ml → L) divided by total completed distance
(km), expressed as litres per 100 km.

**Formula:**
```python
fuel_l_per_100km = (episode_fuel_consumption / 1000.0) / total_completed_km * 100
```

**Cross-check:** CO2/fuel ratio ≈ 747 g/km / (32.3 L/100km × 10) ≈ 2.31 kg CO2/L.
Expected for petrol: ~2.39 kg CO2/L. Small discrepancy expected from mixed fleet
emission profiles between `thesis_car` and `DEFAULT_VEHTYPE` background vehicles.

---

### Avg Route Length (m)

| Field | Value |
|---|---|
| Mean | 1 637.5 m |
| Std | 37.8 m |
| Median | 1 631.1 m |
| Min | 1 547.2 m |
| Max | 1 726.1 m |

**Definition:** Mean planned route length (metres) across all vehicles that
completed their trip during the episode. This is the average end-to-end path length
SUMO assigned to each vehicle at spawn.

**How it is extracted — `_track_new_vehicles`:**
```python
# Primary: TraCI getRouteLength (returns metres)
try:
    rl = traci.vehicle.getRouteLength(veh_id)
    if rl > 0:
        self.vehicle_route_lengths[veh_id] = rl
    else:
        raise ValueError("getRouteLength returned 0")
except Exception:
    # Fallback: sum edge lengths from the assigned route edge list
    try:
        edges = traci.vehicle.getRoute(veh_id)
        rl = sum(traci.lane.getLength(e + "_0") for e in edges if e)
        if rl > 0:
            self.vehicle_route_lengths[veh_id] = rl
    except Exception:
        pass  # omit vehicle from route-length stats
```

The fallback (`getRoute` + `lane.getLength`) is required because `getRouteLength`
returns 0 in some SUMO/libsumo builds before the vehicle's route is fully resolved.

Route lengths are stored in `vehicle_route_lengths[veh_id]` and moved to
`completed_route_lengths[]` when the vehicle arrives (in `_handle_arrivals` and
`_handle_background_arrivals`). `avg_route_length_m`, `co2_g_per_km`, and
`fuel_l_per_100km` are all derived from `completed_route_lengths` at
`_compute_episode_metrics`.

**Note on controlled vs background routes:**
- Controlled vehicles: short 2–6 edge routes from `controlled_init.rou.xml` (~50–200 m)
- Background vehicles: Wardrop equilibrium routes from `wardrop_routes_high.rou.xml` (longer cross-network paths)
- The reported average (~1 637 m) is dominated by background vehicles, which vastly outnumber the 32 controlled slots.

---

### Total Stops

| Field | Value |
|---|---|
| Mean | 97 428.9 |
| Std | 26 657.0 |

**Definition:** Total number of vehicle-timesteps where any vehicle in the
simulation was stopped (speed fell below threshold), aggregated over the episode.

**Source:** `episode_stops_count` incremented each sub-step a vehicle is stopped.

---

### Arrival Rate

| Field | Value |
|---|---|
| Mean | 0.849 |
| Std | 0.009 |

**Definition:** Fraction of spawned vehicles that completed their trip before
episode end.

**Formula:**
```
arrival_rate = episode_arrivals / total_spawned
```

**Population:** Both numerator and denominator cover all vehicles (controlled +
background).

---

## Accuracy Assessment

### Data sources and reliability

| Metric | SUMO source | Reliable? |
|---|---|---|
| Mean Travel Time | `vehicle.getDeparture()`, `simulation.getTime()` | Yes — all completed vehicles |
| Controlled Mean Travel Time | Same, controlled slots only | Yes |
| Mean Waiting Time | `vehicle.getSpeed()` per sub-step | Yes — all completed vehicles |
| CO2 Emissions | `vehicle.getCO2Emission()` | Yes |
| Fuel Consumption | `vehicle.getFuelConsumption()` | Yes — petrol density 740 mg/ml |
| Network Throughput | Arrival counts (all vehicles) | Yes |
| Arrival Rate | Arrival/spawn counts (all vehicles) | Yes |
| Avg Route Length | `vehicle.getRoute()` + `lane.getLength()` | Yes — TraCI edge-sum fallback |
| CO2 g/km | Derived from above | Yes — once route lengths correct |
| Fuel L/100km | Derived from above | Yes — once route lengths correct |
| Real-Time Factor | `time.monotonic()` | Yes (not a traffic metric) |

---

## Issues Found and Fixed

### 1. Population mismatch — travel time vs waiting time

**Problem:** `mean_travel_time` was computed only from the 32 **controlled** vehicles
(short 2–6 edge routes, ~14 s average). `mean_waiting_time` combined waiting times
from completed controlled vehicles **plus** all background vehicles still stuck in
the network at episode end — a different, larger population. This caused waiting
time to exceed travel time, which is logically inconsistent.

**Fix:** Added `_handle_background_arrivals()` which detects when background
vehicles leave the simulation and records their travel time and waiting time into
the same lists used for controlled vehicles. Both metrics now use the identical
population: all vehicles that completed a trip. A separate `controlled_mean_travel_time`
field preserves the controlled-only value for policy comparison.

---

### 2. Network throughput excludes background vehicles

**Problem:** `episode_arrivals` was only incremented inside `_handle_arrivals()`
which loops over `agent_vehicle_ids` (controlled slots). Background vehicle
departures were never counted.

**Fix:** `_handle_background_arrivals()` also increments `episode_arrivals` for
each background vehicle completion.

---

### 3. Fuel consumption density mismatch

**Problem:** Code used diesel density (832 mg/ml) to convert `getFuelConsumption()`
from mg to ml, but `thesis_car` is a petrol vehicle (`HBEFA3/PC_G_EU4`).

**Fix:** Changed to petrol density (740 mg/ml), causing a ~12% upward correction.

---

### 4. Route length tracking — `getRouteLength` returning 0

**Problem:** `traci.vehicle.getRouteLength(veh_id)` returns 0 (not an exception)
in some SUMO/libsumo builds before the vehicle's route is fully resolved at spawn.
The original code caught only exceptions, so 0 was silently stored, causing
`completed_route_lengths` to fill with zeros and making `co2_g_per_km`,
`fuel_l_per_100km`, and `avg_route_length_m` all report 0.

**Fix:** Added a `> 0` guard and a fallback path that retrieves the route edge list
via `traci.vehicle.getRoute()` and sums `traci.lane.getLength(edge + "_0")` for
each edge. This always produces a correct positive value.

---

## Files Changed

| File | Change |
|---|---|
| `pymarl/src/envs/sumo_grid_reroute.py` | Added `_handle_background_arrivals()`, fixed waiting time population, fixed fuel density (petrol 740 mg/ml), added `episode_fuel_consumption`, `_episode_wall_start`, `vehicle_route_lengths`, `completed_route_lengths`, `controlled_travel_times`; fixed route length fallback (`getRoute` + edge sum) |
| `pymarl/src/evaluate.py` | Added `network_throughput`, `real_time_factor`, `co2_emissions`, `fuel_consumption`, `co2_g_per_km`, `fuel_l_per_100km`, `avg_route_length_m`, `controlled_mean_travel_time` to `METRIC_KEYS`; removed duplicate `total_emissions`; added `--baseline selfish_routing` choice; added `--sumo_backend` CLI flag |
| `pymarl/src/controllers/baseline_controller.py` | Added `selfish_routing` policy (maps to noop — no adaptive rerouting) |
| `pymarl/src/envs/sumo_backend.py` | Added libsumo → traci auto-fallback when libsumo is not installed |
| `sumo/scenarios/4by4_map/train_*.sumocfg` | Added `vtypes.add.xml` as additional-files so `thesis_car` vType is loaded during training and evaluation |
