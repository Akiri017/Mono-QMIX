# Refactor: Compute local_states on-the-fly instead of buffering
**Date:** 2026-04-14
**Status:** Complete — pending smoketest validation by user
**Affected areas:** `pymarl/src/learners/hierarchical_q_learner.py`, `pymarl/src/runners/episode_runner.py`, `pymarl/src/main.py`, `pymarl/src/evaluate.py`, `pymarl/src/config/algs/civiq_sumo.yaml`

## Context
Civiq's replay buffer pre-allocated a `local_states` field of shape `(max_rsus=17, max_agents_per_rsu × obs_dim)` per timestep. At `buffer_size=400` and `episode_limit=100` this consumed ~25–40 GB depending on the map — making `buffer_size=400` impossible on any available hardware, including Kaggle's 15 GB VRAM. Mono-QMIX runs at `buffer_size=400` on all maps without issue, so a Civiq cap of 32 would have created an unfair experimental comparison.

## Discussion and decisions

### Why buffer_size matters for comparison
A smaller replay buffer means less experience diversity when sampling training batches, more temporal correlation between samples, and less stable learning. Running Civiq at `buffer_size=32` vs Mono-QMIX at `buffer_size=400` would confound the comparison — any performance gap could be attributed to the buffer difference rather than the algorithm.

### Chosen approach: compute on-the-fly
`local_states` is only consumed in `HierarchicalQLearner.train()` (two reads: online path `[:, :-1]` and target path `[:, 1:]`). It is never read anywhere else. It can be reconstructed from `batch["obs"]` and `batch["zone_assignments"]`, which are already in the buffer and are ~15× smaller.

The reconstruction uses the same slot ordering as `env._build_agents_per_rsu`: agents are placed into RSU slots in **ascending agent_id order** (cumsum over the agent dimension), matching the env exactly. The implementation mirrors `_build_rsu_agent_qs` but scatters `obs_dim`-dimensional observation vectors instead of scalar Q-values.

### Alternatives rejected
- **Reduce buffer_size permanently** — creates unfair comparison with Mono-QMIX; rejected.
- **Store local_states as a compressed format** — adds complexity with no clear benefit over on-the-fly; rejected.
- **Reduce episode_limit** — already done (1000 → 100, commit `14851f1`), which helped but was insufficient to bring BGC Full under 15 GB at `buffer_size=400`.

### Correctness verification
Before implementing, the slot ordering in `env._build_agents_per_rsu` (ascending `agent_id` via `for agent_id in range(self.n_agents)`) was verified to match `_build_rsu_agent_qs`'s `cumsum(dim=2)` logic. Both assign slot 0 to the lowest-indexed agent in each RSU, slot 1 to the next, etc. The on-the-fly reconstruction is byte-for-byte equivalent.

## What changed
- `hierarchical_q_learner.py` — added `_build_local_states(obs, zone_assignments, batch_size, max_t)` method; replaced both `batch["local_states"]` reads (lines 187 and 219) with calls to this method
- `runners/episode_runner.py` — removed two `pre_transition_data["local_states"] = [self.env.get_local_obs_padded(_zone)]` calls (ts=0 and ts=t+1 blocks inside `if _civiq:`)
- `main.py` — removed `local_states` entry from the Civiq scheme block in `get_scheme()`; updated docstring
- `evaluate.py` — removed `local_states` entry from the Civiq scheme block; removed `obs_dim` variable (no longer needed in that block); updated comment
- `civiq_sumo.yaml` — `buffer_size` restored to 400 with updated comment reflecting the memory improvement

## Implementation notes
- `get_local_obs_padded()` in `sumo_grid_reroute.py` is now dead code (never called by the runner). It was left in place intentionally — removing env methods is out of scope and the method is harmless.
- The `_build_local_states` loop iterates over `max_rsus=17` RSUs. Each iteration does one scatter_add_ over `(B×T, n_agents×obs_dim)` tensors. At batch_size=32, max_t=100, this is a small kernel. Estimated overhead: <5ms per training step — negligible.
- Mono-QMIX is completely unaffected. All changes are gated by `if _civiq:` (runner) or `mixer == "civiq"` (scheme), and `HierarchicalQLearner` is never instantiated by Mono-QMIX.
- The stale docstring at line 9 of `hierarchical_q_learner.py` still lists `local_states` as a batch field. This is cosmetic and was left to minimize diff size.

## Status and follow-up
- **Status:** Complete — implementation done, awaiting user smoketest confirmation on 4x4 and BGC Full
- **Follow-up items:**
  - Run `python run_experiments.py --seeds 0 --t_max 1000 --eval_episodes 3 --alg_config civiq_sumo --env_config sumo_grid4x4 --los_level low` to verify no OOM, no NaN loss
  - Run BGC Full smoke: `python run_experiments.py --seeds 0 --t_max 5000 --eval_episodes 3 --alg_config civiq_sumo --env_config sumo_bgc_full --los_level low`
  - Update stale docstring at `hierarchical_q_learner.py` line 9 to remove `local_states` from listed batch fields
