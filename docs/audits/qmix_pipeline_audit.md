# QMIX Training Pipeline Audit

**Date:** 2026-03-28
**Project:** Mono-QMIX — Traffic Rerouting via HMARL
**Checklist Reference:** Canonical QMIX Paper (Rashid et al., 2018)

---

## 1. Agent Network (DRQN)

| Item | Expected | Actual | Status |
|------|----------|--------|--------|
| GRU hidden dim | 64 | 64 | ✓ |
| Input: obs + last action + agent ID | All three | obs only | ⚠ |
| Output: per-action Q-values | Q per action | Q per action | ✓ |
| Hidden state across timesteps | Maintained | Maintained, supports mid-episode resets | ✓ |

**Deviation:** `obs_last_action: false` and `obs_agent_id: false` in `qmix_sumo.yaml`. The paper uses all three inputs for better agent differentiation. This is a deliberate simplification — worth noting when comparing to paper results.

---

## 2. Mixing Network

| Item | Expected | Actual | Status |
|------|----------|--------|--------|
| Hypernetworks from state | Yes | Yes, 2-layer | ✓ |
| Non-negative weights | `abs()` | `torch.abs()` on both layers | ✓ |
| Hidden size | 32 units | 32 (`mixing_embed_dim`) | ✓ |
| Activation | ELU | ELU | ✓ |
| Output | Scalar Q_tot | Scalar Q_tot | ✓ |

No deviations.

---

## 3. Replay Buffer

| Item | Expected | Actual | Status |
|------|----------|--------|--------|
| Episode storage | Full episodes | Full episodes via `EpisodeBatch` | ✓ |
| Tuple contents | obs/act/rew/next_obs/state/next_state/done | All present | ✓ |
| Capacity | 5000 episodes | 200 episodes | ⚠ |
| Sampling | Uniform random | Uniform random, no replacement | ✓ |

**Deviation:** `buffer_size: 200` — reduced from 5000 due to memory constraints (32 agents × 1000 timesteps ≈ 41 GB at full capacity). Reduces sample diversity. Intentional tradeoff.

---

## 4. Loss & Optimization

| Item | Expected | Actual | Status |
|------|----------|--------|--------|
| TD target: `r + γ(1-done)Q'_tot` | Correct form | Correct, with double Q-learning | ✓ |
| Loss | Squared TD error | Masked MSE | ✓ |
| Optimizer | RMSprop | Adam | ⚠ |
| Learning rate | 5e-4 | 5e-4 | ✓ |
| Target update | Every 200 episodes | Every 200 training steps | ⚠ |

**Deviations:**
- **Optimizer:** Adam instead of RMSprop. The config still has dead `optim_alpha`/`optim_eps` keys from RMSprop that go unused. Functionally fine for modern RL, but worth acknowledging in the hyperparameter section of a thesis.
- **Target update timing:** 200 gradient steps, not 200 episodes. With `batch_size=32`, that translates to ~6,400 env steps per update. The semantics differ from the paper but the scale is similar.

---

## 5. Exploration

| Item | Expected | Actual | Status |
|------|----------|--------|--------|
| ε-greedy per agent | Independent | Independent per agent | ✓ |
| ε start | 1.0 | 1.0 | ✓ |
| ε finish | 0.05 | 0.05 | ✓ |
| Anneal time | 50k steps | 500k steps | ⚠ |

**Deviation:** `epsilon_anneal_time: 500000` in config — 10× slower than the paper's 50k. The code default is 50k, but the SUMO config overrides it. Likely intentional for a more complex 32-agent environment, but should be justified explicitly.

---

## Summary of Deviations

| # | Deviation | Severity | Likely Intentional? |
|---|-----------|----------|---------------------|
| 1 | Agent input: obs only (no last action, no agent ID) | Medium | Yes |
| 2 | Buffer: 200 episodes vs 5000 | Medium | Yes — memory constraint |
| 3 | Optimizer: Adam vs RMSprop | Low | Probably yes |
| 4 | Target update: per gradient step vs per episode | Low | Unclear |
| 5 | ε anneal: 500k vs 50k steps | Low–Medium | Likely yes |

---

## Notes

The core QMIX invariants are all correct:
- Monotonicity constraint (non-negative mixing weights via `abs()`)
- Hypernetwork architecture
- Episode-based replay
- TD target structure

The deviations are either environment-driven (buffer size, anneal time) or modern-practice substitutions (Adam). The most defensible point of concern is **Deviation #1** — disabling last action and agent ID removes information the paper relies on for agent differentiation. This should be explicitly addressed if submitting results against paper baselines.

---

## Files Audited

| File | Component |
|------|-----------|
| `pymarl/src/modules/agents/rnn_agent.py` | Agent network |
| `pymarl/src/modules/mixers/qmix.py` | Mixing network |
| `pymarl/src/components/episode_buffer.py` | Replay buffer |
| `pymarl/src/learners/q_learner.py` | Loss & optimization |
| `pymarl/src/controllers/basic_controller.py` | Exploration & action selection |
| `pymarl/src/runners/episode_runner.py` | Episode collection |
| `pymarl/src/main.py` | Training loop |
| `pymarl/src/config/algs/qmix_sumo.yaml` | Algorithm config |
| `pymarl/src/config/envs/sumo_grid4x4.yaml` | Environment config |
