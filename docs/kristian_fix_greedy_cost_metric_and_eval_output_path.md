# Fix: Greedy Cost Metric and Eval Output Path

**Date:** 2026-04-03  
**Files changed:**
- `pymarl/src/evaluate.py`
- `pymarl/src/controllers/baseline_controller.py`

---

## Background

After running a 10k-step experiment (seed=0), two issues were identified:

1. The final comparison table printed by `run_experiments.py` was empty (`"policies": {}`)
   despite per-policy JSON files being written successfully.
2. `greedy_shortest` produced results identical to `noop` across all 5 episodes
   (same return, same travel time, same stops — episode by episode).

---

## Fix 1 — Empty comparison table: double-seed suffix in output path

**File:** `pymarl/src/evaluate.py`  
**Location:** bottom of `__main__` block, line ~381

### Root cause

`run_experiments.py` builds a stem that already encodes the seed:

```python
stem = f"qmix_seed{seed}"   # e.g. "qmix_seed0"
```

It passes this as `--output qmix_seed0` to `evaluate.py`. Inside `evaluate.py`, the
save path was:

```python
# BEFORE:
output_path = f"results/eval/{policy_stem}_seed{args['seed']}.json"
# → "results/eval/qmix_seed0_seed0.json"
```

The seed suffix was appended a second time. `run_experiments.py` then looked for:

```python
out_file = eval_dir / f"{stem}.json"   # "results/eval/qmix_seed0.json"
```

File not found → `None` returned → `policy_aggs` empty → blank comparison table and
`"policies": {}` in the summary JSON.

### Fix

```python
# AFTER:
output_path = f"results/eval/{policy_stem}.json"
# → "results/eval/qmix_seed0.json"  ✓
```

The stem already contains the seed; `evaluate.py` now uses it as-is.

---

## Fix 2 — greedy_shortest identical to noop: static route cost metric

**File:** `pymarl/src/controllers/baseline_controller.py`  
**Location:** `_compute_route_cost()`

### Root cause

`_compute_route_cost` computed route cost using static network data
(`edge.getLength()` / `edge.getMeanSpeed()` from the `.net.xml` file):

```python
# BEFORE:
edge = self.env.net.getEdge(edge_id)
if self.env.route_cost_metric == "length":
    total_cost += edge.getLength()
elif self.env.route_cost_metric == "traveltime":
    travel_time = edge.getMeanSpeed()       # free-flow speed, not current
    total_cost += edge.getLength() / travel_time
```

The environment assigns vehicles their initial route via `_compute_shortest_route()`,
which already picks the static shortest path. The k=1 candidate (action 0, keep current)
is therefore always ≤ the k=2 candidate (action 1, alternative) by static length.
Greedy always picked action 0 — identical to noop regardless of network state.

### Fix

Replace the static lookup with a live SUMO query:

```python
# AFTER:
import traci

# in _compute_route_cost():
try:
    total_cost += traci.edge.getTraveltime(edge_id)
except Exception:
    try:
        edge = self.env.net.getEdge(edge_id)
        total_cost += edge.getLength()
    except Exception:
        pass
```

`traci.edge.getTraveltime()` returns the current congestion-aware travel time maintained
by SUMO's online edge data. When a route alternative is genuinely faster given current
traffic, greedy will now prefer it over the current route. The fallback to static length
preserves behaviour when traci is unavailable (e.g., unit tests outside SUMO).

### Expected effect

`greedy_shortest` should now differ from `noop`. In congested conditions it may
reroute vehicles around bottlenecks; in light traffic it will still keep the original
route (since free-flow alternatives are rarely shorter in travel time). Results will no
longer be episode-for-episode identical to noop.

---

## Verification checklist

1. **Fix 1** — Re-run `run_experiments.py`. Confirm the final comparison table is
   populated and `summary_*.json` has non-empty `"policies"`.
2. **Fix 2** — Run `evaluate.py --baseline greedy_shortest --episodes 5 --seed 0`.
   Confirm returns differ from the `noop` baseline at least for congested episodes.
