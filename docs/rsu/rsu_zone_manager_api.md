# RSUZoneManager — API Reference

**File:** `pymarl/src/components/rsu_zone_manager.py`

---

## Constructor

```python
RSUZoneManager(rsu_config: dict)
```

Loads RSU positions from the config dict (as returned by `yaml.safe_load` on `config/envs/rsu/<map>.yaml`).

### Parameters

| Key in config | Type | Description |
|---------------|------|-------------|
| `rsu_positions` | list of dicts | Each dict has `id` (str), `x` (float), `y` (float) |
| `max_rsus` | int | Total padded RSU slots (≥ `len(rsu_positions)`) |
| `max_agents_per_rsu` | int | Cap used for assertions and masking |

### Attributes set

| Attribute | Type | Description |
|-----------|------|-------------|
| `n_rsus` | int | Number of active RSUs (`len(rsu_positions)`) |
| `max_rsus` | int | Padded slot count |
| `max_agents_per_rsu` | int | Per-zone agent cap |
| `rsu_ids` | list[str] | Ordered RSU id strings; index = RSU index |
| `_rsu_xy` | np.ndarray (n_rsus, 2) | RSU coordinates for distance computation |

---

## Methods

### `assign_vehicles_to_zones`

```python
def assign_vehicles_to_zones(
    self, vehicle_positions: dict[str, tuple[float, float]]
) -> dict[int, list[str]]
```

Assigns every vehicle to its nearest RSU using Euclidean distance.

**Parameters:**
- `vehicle_positions` — `{vehicle_id: (x, y)}`. May be empty.

**Returns:**
- `{rsu_index: [vehicle_id, ...]}` — All `n_rsus` keys present. Lists may be empty.

**Complexity:** O(n_vehicles × n_rsus) via vectorised numpy distance matrix.

**Tie-breaking:** Lower RSU index wins (numpy `argmin` left-to-right default).

---

### `get_rsu_mask`

```python
def get_rsu_mask(self) -> np.ndarray
```

Returns a float32 array of shape `(max_rsus,)`:
- Indices `0 .. n_rsus-1` → `1.0` (active)
- Indices `n_rsus .. max_rsus-1` → `0.0` (padded, unused)

Used by LocalQMixer to ignore padding slots in the mixing network.

---

## Usage Example

```python
import yaml, sys
sys.path.insert(0, "pymarl/src")
from components.rsu_zone_manager import RSUZoneManager

with open("config/envs/rsu/synthetic_4x4.yaml") as f:
    cfg = yaml.safe_load(f)

zm = RSUZoneManager(cfg)

# At each simulation step:
vehicle_positions = {v: traci.vehicle.getPosition(v) for v in traci.vehicle.getIDList()}
zones = zm.assign_vehicles_to_zones(vehicle_positions)

for rsu_idx, veh_list in zones.items():
    print(f"RSU {rsu_idx} ({zm.rsu_ids[rsu_idx]}): {len(veh_list)} vehicles")

mask = zm.get_rsu_mask()  # shape (12,) — all 1.0 for synthetic_4x4
```

---

## Design Notes

- **No radius gating.** Every vehicle is assigned to *some* RSU regardless of distance. `radius` in the YAML is reserved for future observation-space filtering.
- **No state between steps.** `assign_vehicles_to_zones` is stateless — call it every step with fresh positions.
- **Thread safety.** Not thread-safe (numpy operations mutate no shared state, but no lock is held). Safe for single-process use.
