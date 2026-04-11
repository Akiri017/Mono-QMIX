# Phase 7 — BGC Full Wiring (In Progress)

## Status: BLOCKED — map editing in progress

BGC Full RSU placement and zone validation are complete. Wiring is blocked until
the BGC Full network is cleaned up (residential areas + C5 road segments removed)
to reduce the edge count to a tractable level for the one-hot edge encoding.

---

## Why the Map Needs Fixing First

BGC Full currently has ~2,723 edges. The Civiq observation encodes the vehicle's
current edge as a one-hot over all network edges:

```
obs_dim         = 5 (ego) + 2,723 (one-hot) + 12 (traffic stats) = 2,740
local_state_dim = max_agents_per_rsu × obs_dim
               = 384 × 2,740 = 1,052,160   ← infeasible
```

Storing `local_states` in the replay buffer at this dimensionality would require
~170 GB RAM. Training cannot proceed.

**Target:** get edges below ~300 (ideally close to BGC Core's 184) so that
one-hot encoding remains tractable without architecture changes.

**Map edits planned:**
- Remove residential areas (low-traffic, not relevant to BGC routing decisions)
- Remove C5 road segments — keep only BGC entrance/exit connection points
  (C5 is a controlled-access arterial; agents make no routing decisions on it;
   OD demand via C5 access points is preserved)

---

## What Is Done

| Item | Status |
|------|--------|
| 20-RSU placement in `bgc_full.yaml` | Done — ISO 250m compliant, all at proper junctions |
| Block A (`block_a_bgc_full.py`) | Done — CSV generated (`vehicle_positions_bgc_full_los_e.csv`) |
| Block B (`block_b_bgc_full.py`) | Done — peak validated at RSU17 = 379 vehicles |
| `max_agents_per_rsu: 384` in `bgc_full.yaml` | Done |
| Cached routes file (`bgc_full_los_e_routed.rou.xml`) | Exists in `scripts/bgc/` — stale after map edit |

---

## Resume Checklist (after map is fixed)

### Step 1 — Validate RSU positions against new network
```bash
python scripts/bgc/gen_bgc_full_visualization.py
# Open in netedit:
netedit --net-file bgc_full/final_map.net.xml \
        --additional-files bgc_full/rsu_placed.add.xml
```
Any RSU that falls in a removed zone must be moved to the nearest valid junction.
Update `config/envs/rsu/bgc_full.yaml` accordingly.

### Step 2 — Delete stale cached routes
```bash
rm scripts/bgc/bgc_full_los_e_routed.rou.xml
```
Block A will regenerate this via duarouter against the new network.

### Step 3 — Re-run Block A
```bash
python scripts/bgc/block_a_bgc_full.py
```
Outputs: `vehicle_positions_bgc_full_los_e.csv` + heatmap PNG.

### Step 4 — Re-run Block B
```bash
python scripts/bgc/block_b_bgc_full.py
```
Update `max_agents_per_rsu` in `config/envs/rsu/bgc_full.yaml` with new peak + 5.

### Step 5 — Get edge count from new network
```python
import sumolib
net = sumolib.net.readNet("bgc_full/final_map.net.xml")
edges = [e for e in net.getEdges() if not e.getID().startswith(":")]
print(len(edges))
```
This gives the new `obs_edge_dim` value.

### Step 6 — Propagate `max_rsus: 20` to all YAMLs
Files that still use the old placeholder value:

| File | Key to update |
|------|---------------|
| `pymarl/src/config/algs/civiq_sumo.yaml` | `max_rsus` |
| `pymarl/src/config/envs/sumo_bgc_core.yaml` | `max_rsus` |
| `pymarl/src/config/envs/sumo_grid4x4.yaml` | `max_rsus` (if present) |

### Step 7 — Create `sumo_bgc_full.yaml`
Mirror of `sumo_bgc_core.yaml` with BGC Full values:

| Key | BGC Core | BGC Full |
|-----|----------|----------|
| `network_file` | `bgc_core/final_map.net.xml` | `bgc_full/final_map.net.xml` |
| `obs_edge_dim` | 184 | **TBD — from Step 5** |
| `obs_dim` | 201 | **TBD — 5 + edges + 12** |
| `state_shape` | 6,432 | **TBD — 32 × obs_dim** |
| `global_state_dim` | 6,432 | **TBD** |
| `max_agents_per_rsu` | 50 | **TBD — from Step 4** |
| `max_rsus` | 20 | 20 |
| `rsu_config` | `bgc_core.yaml` | `bgc_full.yaml` |
| `n_actions` | 4 | **TBD — check BGC Full route options** |
| `los_cfg` | bgc_core sumocfgs | bgc_full sumocfgs |

### Step 8 — Create BGC Full sumocfgs
```
bgc_full/train_low.sumocfg
bgc_full/train_med.sumocfg
bgc_full/train_high.sumocfg
```
Mirror `bgc_core/train_*.sumocfg`, pointing to `bgc_full/final_map.net.xml`
and the corresponding trips files.

### Step 9 — 1k smoke test
```bash
cd pymarl/src
python main.py --alg_config civiq_sumo.yaml --env_config sumo_bgc_full.yaml \
               --t_max 1000 --no_validation --batch_size 1 --buffer_size 2
```
Gate criteria: no crash, no NaN, episodes complete.

### Step 10 — Write Phase 7 gate doc
`docs/rsu/phase7_bgc_full_wiring.md` — only after smoke test passes.

---

## Open Items Beyond Phase 7

| Item | Status |
|------|--------|
| Reward normalization (grad norms ~2–3M) | Pending — needed before full training |
| 10k smoke test on BGC Core | Pending |
| Full 200k training run — BGC Core vs Mono QMIX baseline | Pending |
| Full 200k training run — BGC Full | Pending (after Phase 7) |
