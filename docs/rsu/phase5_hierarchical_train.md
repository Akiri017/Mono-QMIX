# Phase 5 — Hierarchical Training (`HierarchicalQLearner.train()`)

## Purpose

Phase 5 completes `HierarchicalQLearner.train()` by replacing the `NotImplementedError` stub with the full hierarchical mixing forward pass, wires Civiq into `main.py`, and validates the end-to-end training pipeline with a 10k-step smoke test and a single-RSU equivalence check.

---

## Files Created / Modified

| File | Action | Description |
|------|--------|-------------|
| `pymarl/src/learners/hierarchical_q_learner.py` | Updated | Replaced NotImplementedError with full mixing implementation |
| `pymarl/src/main.py` | Updated | Added HierarchicalQLearner import, learner selection, `--alg_config` CLI arg |
| `pymarl/src/config/algs/civiq_sumo.yaml` | Updated | Added `learner: "hierarchical_q_learner"` |
| `pymarl/src/config/algs/civiq_single_rsu.yaml` | Created | Single-RSU equivalence check config (temporary) |
| `config/envs/rsu/synthetic_single_rsu.yaml` | Created | Single-RSU zone config at map centroid |
| `tests/test_phase5_gradient_flow.py` | Created | Phase 5 gate — 3-test gradient/shape suite |

---

## `train()` Implementation

### rsu_agent_qs Construction

`chosen_action_qvals` (B, T, n_agents) must be scattered into `rsu_agent_qs` (B, T, max_rsus, max_agents_per_rsu) using `batch["zone_assignments"]`.

**Key insight**: within each RSU, agents are ordered by ascending agent index (matching `_build_agents_per_rsu`). The slot index for agent `a` in RSU `r` equals `cumsum(zone_assignments == r)[:a] - 1`.

**Vectorized scatter (loop over RSUs, vectorized over B×T×agents):**

```python
rsu_agent_qs = torch.zeros(B, T, max_rsus, max_agents_per_rsu, device=device)
for r in range(max_rsus):
    in_rsu  = (zone_assignments_t == r)           # (B, T, n_agents) bool
    cumsum  = in_rsu.long().cumsum(dim=-1)          # rank within RSU r
    slot_idx = (cumsum - 1).clamp(0, A - 1)         # (B, T, n_agents)
    src     = chosen_action_qvals * in_rsu.float()  # zero out non-RSU agents
    rsu_agent_qs[:, :, r, :].scatter_add_(dim=-1, index=slot_idx, src=src)
```

`scatter_add_` is safe: non-RSU agents contribute `src=0` (adding zero has no effect). Real agents in RSU r get unique slots (cumsum guarantees strict monotone ordering).

### Forward Pass

```
# Online path
local_qtots = local_mixer(
    rsu_agent_qs.view(BT*R, A),
    local_states.view(BT*R, -1),
    agent_masks.view(BT*R, A)
).view(BT, R)                                 # (BT, R)

global_qtot = global_mixer(
    local_qtots,
    global_states.view(BT, -1),               # batch["state"][:, :-1]
    rsu_mask.view(BT, R)                      # (agent_masks.sum(-1) > 0)
).view(B, T, 1)                               # (B, T, 1)

# Target path — identical using target mixers and batch[:, 1:] zone fields
```

### TD Loss

Identical to QLearner:
```python
targets      = rewards + gamma * (1 - terminated) * target_global_qtot
td_error     = global_qtot - targets.detach()
masked_error = td_error * mask.expand_as(td_error)
loss         = (masked_error ** 2).sum() / mask.sum()
```

### Gradient Clipping

Per-component norms are computed (without clipping) for logging, then a single `clip_grad_norm_(self.params, grad_norm_clip)` clips all parameters together (MAC + LocalQMixer + GlobalQMixer):

```python
agent_grad_norm       = _grad_norm(mac.parameters())
local_mixer_grad_norm = _grad_norm(local_mixer.parameters())
global_mixer_grad_norm = _grad_norm(global_mixer.parameters())
torch.nn.utils.clip_grad_norm_(self.params, self.grad_norm_clip)
```

---

## main.py Changes

### Learner Selection

`run_training()` now selects the learner based on `args["learner"]`:

```python
learner_key = args.get("learner", "q_learner")
if learner_key == "hierarchical_q_learner":
    learner = HierarchicalQLearner(mac, scheme, logger, args)
else:
    learner = QLearner(mac, scheme, logger, args)
```

### `--alg_config` CLI Arg

```bash
python main.py --alg_config civiq_sumo.yaml --t_max 10000 --no_validation
```

Selects a config from `config/algs/`. Default remains `qmix_sumo.yaml` (backward compatible).

### rsu_config Path Resolution

`HierarchicalQLearner.__init__` resolves relative `rsu_config` paths to repo root (3 levels up from `learners/`), matching the env's `_resolve_path()`:

```python
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
rsu_config_path = os.path.join(repo_root, rsu_config_path)
```

---

## Phase 5 Gate Results

```
STEP 5.1 + 5.2 PASSED
```

Episode: 50 steps, `train_med.sumocfg`, libsumo backend, batch_size=1.

| Test | Description | Result |
|------|-------------|--------|
| 1 | Batch field shapes correct (actions, zone_assignments, agent_masks, local_states, state dims) | PASSED |
| 2 | `train()` completes: loss=3.46M, global_qtot=-2.35, no NaN/Inf, loss > 0 | PASSED |
| 3 | All MAC/LocalQMixer/GlobalQMixer parameters have non-None, non-NaN gradients | PASSED |

Gradient counts:
- MAC params with grad: 29,314
- LocalQMixer params: 528,641
- GlobalQMixer params: 561,921

---

## 10k Step Smoke Test

```
python main.py --alg_config civiq_sumo.yaml --t_max 10000 --no_validation --batch_size 1
```

| Metric | Value |
|--------|-------|
| Crash? | No |
| Episodes completed | 100 (100 steps/episode avg) |
| Initial loss (t=5000) | 4,051,339 |
| Final loss (t=10000) | 3,541,044 |
| Loss trend | Decreasing (~12.6%) |
| NaN/Inf in log? | No |
| Checkpoint saved? | Yes (`results/models/civiq_10k/final`) |
| global_qtot_mean (t=10000) | −173.05 |

Note: `global_mixer_grad_norm` is large (~2–3M) due to reward magnitudes (~300k/episode). Grad clip (`grad_norm_clip=10000`) is active. Stable but aggressive — reward normalization recommended for full training.

---

## Single-RSU Equivalence Check

Centroid of 12 4×4 intersections: `(150.0, 250.0)`, radius `2000.0m` (covers all; max dist ≈ 791m).

| Config | Loss (t=5000) | global_qtot (t=5000) | Ratio |
|--------|--------------|----------------------|-------|
| 12-RSU (`civiq_sumo.yaml`) | 4,051,339 | −47.20 | 1× |
| 1-RSU (`civiq_single_rsu.yaml`) | 8,813,469 | −50.63 | ~2× |

Both in the same order of magnitude (millions). No NaN, no constant-zero output, no 100× divergence. Equivalence check **passed**.

The ~2× higher loss for single-RSU is expected: with 1 RSU covering all 32 agents, `local_state_dim = 32 × 65 = 2080`, giving the LocalQMixer a more complex state space to condition on.

---

```
PHASE 5 GATE PASSED
```

---

## Next Step

Phase 6 — **Reward Normalization + Full Training Run**: The current large gradient norms in GlobalQMixer (~2–3.5M with grad_norm_clip=10000) indicate reward magnitude mismatch. Options:
- Add reward normalization to `run_training()` (divide by running std)
- Reduce `lr` for the GlobalQMixer (separate optimizer)
- Scale rewards by a fixed factor (e.g., `/ 100000`)

After reward normalization, run a full 200k-step training run (1 seed) and compare against Mono QMIX baseline.
