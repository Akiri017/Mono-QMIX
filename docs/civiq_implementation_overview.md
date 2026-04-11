# Civiq — Implementation Overview (Phases 0–6)

> Curated summary for research documentation.  
> For per-phase implementation details, see `docs/rsu/phase{0–6}_*.md`.

---

## 1. Problem Setting

Urban traffic congestion in dense networks (e.g., Bonifacio Global City, Manila) demands coordinated rerouting of connected vehicles at scale. We frame this as a **cooperative multi-agent reinforcement learning (MARL)** problem: each vehicle is an agent that chooses among candidate routes, and the collective goal is to minimise network-wide travel time.

Standard QMIX operates with a flat agent set and a single global mixer. It does not scale to networks with 100+ simultaneous agents across geographically distributed zones. **Civiq** (Cooperative Infrastructure-based Vehicle IQ) extends QMIX with a two-level hierarchical mixer structured around Road-Side Units (RSUs) — fixed infrastructure nodes that partition the network into local zones.

---

## 2. Architecture

```
                        Global Q_tot
                             ▲
                    ┌────────┴────────┐
                    │  GlobalQMixer   │   ← Level 3
                    └────────┬────────┘
                             │  (R local Q_tot scalars)
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
   LocalQMixer 0      LocalQMixer 1  …   LocalQMixer R   ← Level 2  (weight-shared)
          ▲                  ▲                  ▲
    [agents in          [agents in          [agents in
      RSU 0 zone]         RSU 1 zone]         RSU R zone]            ← Level 1 (MAC)
```

**Level 1 — Multi-Agent Controller (MAC).**  
Each vehicle carries a GRU-based policy network (identical to standard QMIX). It observes local ego features, a one-hot edge encoding for its current route, and outgoing-edge traffic statistics (occupancy, mean speed, queue length). The MAC outputs per-action Q-values for the vehicle's candidate routes.

**Level 2 — LocalQMixer (per RSU).**  
Within each RSU zone, vehicle Q-values are aggregated by a LocalQMixer — a monotonic hypernetwork conditioned on the local state (concatenated agent observations in the zone). One LocalQMixer instance exists but is applied independently to each RSU zone with shared weights. The output is a scalar `local_Q_tot` per zone.

**Level 3 — GlobalQMixer.**  
A second monotonic hypernetwork takes the vector of `local_Q_tot` scalars from all RSU zones and mixes them — conditioned on the global state (concatenated observations of all controlled agents) — to produce a single `global_Q_tot`. The TD loss is computed on this final scalar and backpropagated end-to-end through both mixers and the MAC.

### Key Structural Properties

| Property | Value |
|----------|-------|
| Mixer monotonicity | Enforced at both levels (absolute-value hypernetwork weights) |
| Weight sharing | LocalQMixer: one instance, applied per-zone; GlobalQMixer: one instance |
| Padded zones | Empty RSU zones output a non-zero bias (no NaN risk) |
| End-to-end training | Single Adam optimizer covers MAC + LocalQMixer + GlobalQMixer |

---

## 3. Zone Management (RSUZoneManager)

**Assignment rule:** at each simulation timestep, every vehicle is assigned to the RSU whose centroid is geometrically nearest. Ties (equidistant RSUs) are broken by ascending RSU index. No vehicle is assigned to more than one zone.

**Zone radius** is stored per-RSU config and used for capacity validation (Block B peak count) but not for assignment — assignment is purely nearest-centroid.

**Padding:** each zone's agent tensor is padded to `max_agents_per_rsu`. Padded slots have an agent mask of 0, zeroing their contribution in the LocalQMixer's first `bmm` step.

---

## 4. Observation and State Spaces

### Per-agent observation (`obs_dim`)

| Component | Synthetic 4×4 | BGC Core |
|-----------|---------------|----------|
| Ego features | 5 | 5 |
| Edge one-hot | 48 | 184 |
| Outgoing traffic (4 edges × 3 features) | 12 | 12 |
| **Total** | **65** | **201** |

### Local state (LocalQMixer input)

`local_state_dim = max_agents_per_rsu × obs_dim`

| Map | Value |
|-----|-------|
| Synthetic 4×4 | 28 × 65 = 1,820 |
| BGC Core | 50 × 201 = 10,050 |

### Global state (GlobalQMixer input)

`global_state_dim = n_agents × obs_dim` (concatenated controlled-agent observations from `batch["state"]`)

| Map | Value |
|-----|-------|
| Synthetic 4×4 | 32 × 65 = 2,080 |
| BGC Core | 32 × 201 = 6,432 |

---

## 5. RSU Placement (BGC Core)

RSUs are placed at traffic-light intersections. Placement for BGC Core went through one revision:

**Final configuration (8 RSUs):**

| RSU | x (m) | y (m) | LOS E peak |
|-----|--------|--------|------------|
| RSU00 | 37.6 | 221.4 | 26 |
| RSU01 | 228.0 | 159.4 | 49 |
| RSU02 | 417.3 | 95.0 | 37 |
| RSU03 | 606.9 | 30.2 | 28 |
| RSU04 | 676.0 | 220.0 | 24 |
| RSU05 | 481.0 | 284.8 | 42 |
| RSU06 | 386.5 | 317.4 | **50** |
| RSU07 | 133.0 | 190.0 | 29 |

**Peak validation method (Block B):**  
SUMO was run at LOS E (period = 0.7 s, ~5,143 veh/hr) for 3,600 seconds, logging every vehicle position at every timestep. The resulting CSV (179,470 rows) was post-processed by assigning each vehicle at each timestep to its nearest RSU within 300 m and tracking the maximum simultaneous occupancy per zone.

Overall peak: **50 vehicles (RSU06)** — used as `max_agents_per_rsu = 50`.

**Network facts:**

| Property | BGC Core |
|----------|----------|
| Edges (no internal) | 184 |
| Junctions | 129 |
| Traffic-light junctions | 8 |
| Map extent | ~700 × 450 m |

---

## 6. Training Pipeline

### Replay Buffer Fields (Civiq-specific)

| Field | Shape | Description |
|-------|-------|-------------|
| `zone_assignments` | `(B, T, n_agents)` | RSU index per agent slot; −1 = unassigned |
| `local_states` | `(B, T, max_rsus, max_agents_per_rsu × obs_dim)` | Per-zone padded agent obs |
| `agent_masks_per_rsu` | `(B, T, max_rsus, max_agents_per_rsu)` | 1.0 = real agent, 0.0 = padded |
| `rsu_agent_qs` | not stored | Computed in `train()` by scattering `chosen_action_qvals` |

### `rsu_agent_qs` Construction in `train()`

`chosen_action_qvals` (shape `B × T × n_agents`) is scattered into `rsu_agent_qs` (shape `B × T × max_rsus × max_agents_per_rsu`) using cumsum-based slot assignment per RSU. Non-RSU agents contribute zero via masking before the scatter.

### Loss and Optimization

Standard double-DQN TD loss on `global_Q_tot`:

```
targets = r + γ (1 − done) · target_global_Q_tot
loss    = mean[ (global_Q_tot − targets.detach())² · episode_mask ]
```

All parameters (MAC + LocalQMixer + GlobalQMixer) share a single Adam optimizer with gradient norm clipping.

---

## 7. Multi-Map Support

The pipeline is map-agnostic. Switching between the synthetic 4×4 map and BGC Core requires only two CLI arguments:

```bash
python main.py --alg_config civiq_sumo.yaml --env_config sumo_bgc_core.yaml
```

Key design decisions enabling this:
- `los_cfg` map in the env YAML selects per-map SUMO config files at runtime (not hardcoded)
- All Civiq dimension constants (`obs_dim`, `global_state_dim`, `max_agents_per_rsu`, `rsu_config`) are overridden at the top level of the env YAML, shadowing the alg config defaults
- `RSUZoneManager` is instantiated from the RSU YAML specified by `rsu_config`; if absent, the env runs in standard QMIX mode

---

## 8. Validated Milestones

| Phase | Milestone | Gate Result |
|-------|-----------|-------------|
| 0 | RSUZoneManager — zone assignment, masking, API | PASSED (100-step assertion suite) |
| 1 | LocalQMixer — padded mixing, gradient flow, empty-zone safety | PASSED (4 unit tests) |
| 2 | GlobalQMixer — RSU-level mixing, padded RSU mask | PASSED (4 unit tests) |
| 3 | HierarchicalQLearner — construction, target independence, save/load | PASSED (4 unit tests) |
| 4 | Data collection — zone fields in replay buffer, conservation check | PASSED (50-step libsumo run) |
| 5 | End-to-end training — `train()` loss + gradient flow, 10k smoke test | PASSED (loss ↓ 12.6% over 10k steps) |
| 6 | BGC Core wiring — real-world map, env config, sumocfgs, CLI args | PASSED (1k-step smoke test, no crash) |

---

## 9. Parameter Counts

| Component | Synthetic 4×4 |
|-----------|---------------|
| MAC (GRU + Q-head) | ~29,000 |
| LocalQMixer | 528,641 |
| GlobalQMixer | 561,921 |
| **Total** | **~1,120,000** |

---

## 10. Open Items

| Item | Status |
|------|--------|
| Reward normalization (grad norms ~2–3M) | Pending — needed before full training |
| 10k smoke test on BGC Core | Pending |
| Full 200k training run — BGC Core vs Mono QMIX baseline | Pending |
| `max_rsus` global lock to BGC Full node count | Pending — currently placeholder `12` |
| BGC Full map wiring | Pending |
