# libsumo Migration: TraCI to libsumo Backend

## Motivation

Training runs at scale (500k+ timesteps) were bottlenecked by TraCI's client-server
architecture. TraCI communicates with SUMO over a TCP socket, adding ~0.1-0.5ms of
IPC latency per API call. With 35+ TraCI calls per decision step and K=10 simulation
sub-steps per decision, this overhead compounds to significant wall-clock waste across
long training budgets.

**libsumo** is SUMO's in-process C++ library binding (via SWIG). It exposes nearly the
same Python API as TraCI but eliminates socket overhead entirely by running the
simulation inside the Python process. Expected speedup: **2-5x** on step-heavy
workloads.

---

## Architecture: Thin Adapter Module

Rather than scattering `if libsumo ... else traci ...` conditionals across every call
site, the migration uses a **single adapter module** (`sumo_backend.py`) that the rest
of the codebase imports in place of `traci`.

```
                  +--------------------+
                  |  sumo_backend.py   |
                  |  (_BackendProxy)   |
                  +--------+-----------+
                           |
              config flag: sumo_backend
                    /              \
            "libsumo"            "traci"
           (in-process)        (TCP socket)
```

Consumer files import the adapter aliased as `traci`, so all 35 existing call sites
(`traci.vehicle.getSpeed(...)`, `traci.edge.getTraveltime(...)`, etc.) work unchanged.

### Why an adapter instead of direct replacement

- Only **3 API calls** actually differ between TraCI and libsumo; the remaining 32 are
  identical. A direct find-and-replace would add risk for no benefit.
- The adapter centralises the backend decision, making rollback a one-line config change.
- It keeps the diff minimal: 2 import lines changed in consumer files.

---

## Files Changed

| File | Change |
|------|--------|
| `pymarl/src/envs/sumo_backend.py` | **New** -- adapter module |
| `pymarl/src/envs/sumo_grid_reroute.py` | Import swapped, backend config + GUI fallback in `__init__`, watchdog guard |
| `pymarl/src/controllers/baseline_controller.py` | Import swapped |
| `pymarl/src/config/envs/sumo_grid4x4.yaml` | Added `sumo_backend` key |

---

## Adapter Module (`sumo_backend.py`)

### Backend selection

The backend is selected once via `set_backend(name)`, called during env `__init__`.
Falls back to the `SUMO_BACKEND` environment variable, defaulting to `"libsumo"`.

```python
from envs.sumo_backend import backend as traci
from envs.sumo_backend import set_backend, is_libsumo
```

### API normalisation

The adapter normalises the three calls that differ between backends:

| Call | TraCI | libsumo | Adapter |
|------|-------|---------|---------|
| Start simulation | `traci.start(cmd)` | `libsumo.start(cmd)` | `backend.start(cmd)` -- strips `sumo-gui` binary for libsumo |
| Advance one step | `traci.simulationStep()` | `libsumo.simulation.step()` | `backend.simulationStep()` -- dispatches to correct method |
| Close simulation | `traci.close()` | `libsumo.close()` | `backend.close()` -- pass-through |

All other calls (`vehicle.*`, `edge.*`, `simulation.getTime()`, etc.) pass through
unchanged via `__getattr__`.

---

## Environment Changes (`sumo_grid_reroute.py`)

### Import change (line 27-28)

```python
# Before
import traci

# After
from envs.sumo_backend import backend as traci
from envs.sumo_backend import set_backend as _set_sumo_backend, is_libsumo as _is_libsumo
```

### Backend config in `__init__`

Reads `sumo_backend` from `env_args` (defaults to `"libsumo"`), applies GUI
auto-fallback, then initialises the adapter:

```python
self.sumo_backend = env_args.get("sumo_backend", "libsumo")

# Auto-fallback: sumo-gui requires TraCI (libsumo has no GUI support)
if self.sumo_gui and self.sumo_backend == "libsumo":
    logger.warning("sumo-gui requires the TraCI backend; falling back to 'traci'.")
    self.sumo_backend = "traci"

_set_sumo_backend(self.sumo_backend)
```

### Watchdog guard

The watchdog thread is **skipped** when using libsumo because:

1. libsumo runs in-process -- there is no TCP socket to deadlock.
2. libsumo is **not thread-safe** -- calling `close()` from a background thread while
   `simulation.step()` is executing would be undefined behaviour.
3. In-process hangs are much rarer than socket-layer deadlocks.

```python
def _start_watchdog(self) -> None:
    if _is_libsumo():
        return  # unsafe and unnecessary for libsumo
    # ... existing TraCI watchdog logic unchanged ...
```

The watchdog remains fully functional when using the TraCI backend.

---

## Baseline Controller Changes (`baseline_controller.py`)

Import swapped to adapter (line 12). The `traci.edge.getTraveltime()` call on line 175
is API-identical in libsumo, so no further changes needed.

---

## Configuration

### YAML config (`sumo_grid4x4.yaml`)

```yaml
env_args:
  sumo_backend: "libsumo"   # "libsumo" (default, faster) or "traci" (supports GUI)
  sumo_gui: false            # sumo-gui requires "traci" backend; auto-fallback if mismatch
```

### Environment variable override

If `set_backend()` is never called explicitly (e.g., in test scripts), the adapter
auto-initialises from:

```bash
export SUMO_BACKEND=libsumo   # or "traci"
```

### Rollback

To revert to the original TraCI behaviour, change one line:

```yaml
sumo_backend: "traci"
```

No code changes required. All original TraCI code paths remain intact.

---

## Limitations and Constraints

| Constraint | Detail |
|-----------|--------|
| No GUI with libsumo | libsumo cannot launch `sumo-gui`. The adapter auto-falls back to TraCI when `sumo_gui: true`. |
| One simulation per process | libsumo is not thread-safe and only supports a single simulation instance. The current architecture (one env per process) satisfies this. Do not use `SubprocVecEnv` with libsumo. |
| Crash isolation | With TraCI, a SUMO crash only kills the subprocess. With libsumo, it crashes the Python process. Accepted trade-off for performance; SUMO crashes are rare in steady-state. |
| Watchdog disabled | The heartbeat watchdog is skipped for libsumo (thread-safety constraint). If a simulation hangs with libsumo, the process must be killed externally. |

---

## Verification Checklist

1. **Regression test (TraCI)**: Run 1 episode with `sumo_backend: "traci"` to confirm
   existing behaviour is preserved after the import changes.

2. **Deterministic comparison**: Run the same episode (identical seed) under both
   backends. Diff vehicle speeds, route assignments, rewards at every step -- they
   should be identical.

3. **Smoke training**: Run ~1000 timesteps with `sumo_backend: "libsumo"`. Verify
   reward curve and episode metrics (mean travel time, stops, emissions) match the
   TraCI baseline.

4. **Performance benchmark**: Time 50 episodes under each backend, compare wall-clock
   per episode. Expected improvement: 2-5x.

5. **GUI fallback**: Set `sumo_gui: true` with `sumo_backend: "libsumo"`. Confirm the
   warning is logged and the backend automatically falls back to TraCI.

6. **Baseline evaluation**: Run `evaluate.py` with `--baseline greedy_shortest` under
   libsumo to verify `baseline_controller.py` works correctly through the adapter.

---

## Note:
- When downloading libsumo, use pip install libsumo with the minor mismatch with the SUMO 1.25.0 we use
- the libsumo included in the PyPI packagae only has 1.26.0. The version mismatch is minor  
  and the Python API is stable across patch versions
- Downloading libsumo to match the existingg configs does not work atm
