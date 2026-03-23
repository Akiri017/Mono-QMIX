## Plan: QMIX Smart Rerouting in SUMO Grid

Build a PyMARL-compatible SUMO+TraCI environment where each vehicle is an agent that periodically selects among a small discrete set of candidate routes to its fixed destination. Use QMIX for centralized training (mixing network over per-agent Qs) and decentralized execution (each vehicle chooses its own action), evaluated on a 4x4 synthetic SUMO grid with fixed-N vehicles.

**Steps**
1. Environment + tooling setup
   1. Install SUMO (with `sumo`/`netgenerate`) and Python deps (`traci`, `sumolib`, `numpy`, `torch`). Confirm `SUMO_HOME` is set and that `python -c "import traci"` works.
   2. Decide whether to vendor PyMARL into this repo (recommended for easiest env integration) or treat it as an external repo. Recommended structure: keep PyMARL sources in `pymarl/` and add the custom env under `pymarl/src/envs/`.

2. Create the 4x4 synthetic SUMO map
   1. Reuse the existing SUMO network + scenario assets already copied into this repo (no need to regenerate a “true” 4x4 grid net for now).
   2. Produce a canonical SUMO config (`.sumocfg`) for training that references:
      - the network (`.net.xml`)
      - a *controlled-fleet* demand file (fixed-N with replacement)
      - a *background-traffic* demand file (realistic departures over time)
      and explicitly sets the simulation step length (e.g., 1s).
   3. Controlled fleet (agents): maintain a fixed number of concurrently-active controlled vehicles (e.g., N=32) using **replacement/lifelong slots**:
      - vehicles depart early (t=0 or within a small insertion window) to avoid long warm-up
      - when a controlled vehicle arrives, its slot becomes inactive for a short delay, then a new controlled vehicle is spawned into the same slot
      - each vehicle instance has a fixed destination for the duration of its trip
   4. Background traffic (non-agents): use realistic trips/flows (e.g., low/med/high demand files) with departures spread across the episode horizon; these vehicles follow the default routing/baseline.

3. Define the MARL formulation (QMIX-friendly)
   1. Agents: vehicles (fixed N per episode).
   2. Decision period: every K seconds (your choice; start with K=5 or 10). Between decisions, vehicles follow SUMO’s default driving logic; only routing is changed.
   3. Discrete action set per agent: choose among M candidate routes from current edge to destination (keep M small and fixed; start with M=4).
      - Precompute candidates via k-shortest paths on the SUMO network (sumolib), using either edge length or current travel time as cost.
      - If fewer than M feasible routes exist, pad with a “no-op / keep current route” action and mask invalid actions.
   4. Observations (per agent, local): own speed, current edge id (encoded), remaining route length to destination (scalar), and queue/occupancy features for outgoing edges at the next intersection (limited neighborhood).
   5. Global state for QMIX: concatenate all agent observations (fixed N) plus optional global edge-density summary if desired. Keep it simple initially: concat obs only.

4. Implement the SUMO+TraCI environment wrapper (PyMARL interface)
   1. Implement `reset()` to start SUMO with TraCI, load the scenario, insert vehicles, and advance until all N are present.
   2. Implement `step(actions)`:
      - If `t % K == 0`, apply reroute actions by setting each vehicle’s route to the chosen candidate (TraCI route update).
      - Advance SUMO for K internal steps (or 1 step repeatedly while accumulating reward) to match the decision period.
      - Compute reward, termination, and collect next obs/state.
   3. Reward (multi-objective, team reward for QMIX):
      - Time component: negative mean speed loss or negative per-step travel time proxy (e.g., `-1` per vehicle per sim-step while not arrived), aggregated over all vehicles.
      - Stops component: penalty for stop-and-go (e.g., count vehicles with speed < threshold).
      - Emissions component: penalty from TraCI emission getters (CO2/NOx) if enabled.
      - Start with a weighted sum and make weights configurable (YAML).
   4. Availability masking: expose `get_avail_actions()` with valid route candidates for each agent at the current state.
   5. Termination: episode ends when all vehicles arrive OR max simulation time reached. Provide both.
   6. Robustness: handle vehicles arriving mid-episode by freezing their actions to “no-op” and masking others; still keep fixed-N tensors.

5. Integrate with PyMARL QMIX
   1. Register the environment in PyMARL’s env registry (where other envs are registered) so `--env-config` can select it.
   2. Create an env config YAML specifying: N agents, obs/state dimensions, action count M, decision period K, scenario paths, reward weights, and episode length.
   3. Use the standard QMIX config (mixer=qmixer, mac=basic_mac, agent=rnn) and adjust batch size / buffer sizes for this task.

6. Training + evaluation protocol
   1. Baselines:
      - SUMO default routing (no re-routing).
      - Greedy shortest-path reroute baseline every K seconds (non-learning).
   2. Metrics:
      - Mean travel time, arrival rate by horizon, mean waiting time, number of stops, total emissions.
   3. Run multiple random seeds; log metrics per episode; save best model by validation metric (e.g., travel time).

7. Reproducibility + packaging
   1. Add a single entry script to launch training with fixed seed, scenario, and configs.
   2. Add a short README with: install steps, how to run SUMO headless, and how to reproduce baseline vs QMIX.

**Relevant files**
- `sumo/scenarios/grid4x4/grid.net.xml` — generated SUMO network
- `sumo/scenarios/grid4x4/routes.rou.xml` — fixed-N vehicle routes/demand
- `sumo/scenarios/grid4x4/grid.sumocfg` — SUMO configuration
- `pymarl/src/envs/sumo_grid_reroute.py` — new TraCI-based environment implementing PyMARL env API
- `pymarl/src/envs/__init__.py` or env registry file — register `sumo_grid_reroute`
- `pymarl/src/config/envs/sumo_grid4x4.yaml` — environment config (N, M, K, reward weights)
- `pymarl/src/config/algs/qmix_sumo.yaml` (or reuse QMIX config) — algorithm hyperparams
- `README.md` — run instructions and experiment protocol

**Verification**
1. SUMO sanity: run the `.sumocfg` in `sumo-gui` and confirm all N vehicles spawn and reach destinations.
2. TraCI sanity: a small script that starts SUMO headless, steps 50 ticks, and prints a sample vehicle’s speed and current edge.
3. Env API: call `reset()` and `step()` for a few episodes with random actions; verify tensor shapes match `(n_agents, obs_dim)` and `(n_agents, n_actions)` for masks.
4. Training smoke test: run QMIX for a short duration (e.g., 1000 env steps) and verify loss decreases or at least runs without NaNs.
5. Baseline comparison: confirm learned policy does not underperform default routing on basic metrics after a modest training budget.

**Decisions**
- Use PyMARL-style QMIX (centralized training, decentralized execution).
- Core defaults: N=32 agents, M=4 candidate routes/action size, decision period K=5s.
- Episode termination: end when all vehicles arrive (no hard cap initially).
- Rerouting actions every K seconds via k-shortest candidate routes + action masking.
- Local observations; global state as concatenated observations.
- Team reward weights (time, stops, emissions) = (1.0, 0.05, 0.001), configurable.

**Further Considerations**
1. Reward weights: pick initial weights (e.g., time dominates) and treat emissions/stops as regularizers; tune once training is stable.
2. Candidate-route generation: decide cost metric (static length vs dynamic travel time) and whether to refresh candidates each decision.
3. Demand design: OD pairs and congestion level strongly affect learnability; start with moderate load before scaling up.
