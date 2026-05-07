# Architecture: Civiq vs Mono-QMIX Inheritance Audit
**Date:** 2026-04-14
**Status:** Complete
**Affected areas:** `pymarl/src/modules/agents/rnn_agent.py`, `pymarl/src/modules/mixers/qmix.py`, `pymarl/src/modules/mixers/local_qmixer.py`, `pymarl/src/modules/mixers/global_qmixer.py`, `pymarl/src/learners/q_learner.py`, `pymarl/src/learners/hierarchical_q_learner.py`

## Context
Before running BGC full map smoketests, an audit was conducted to verify that Civiq's internal components — DRQN agent (RNNAgent) and mixer hypernetwork structure — faithfully inherit from Mono-QMIX. The intended design constraint is that Civiq differs from Mono-QMIX in exactly one way: it uses two mixers (LocalQMixer + GlobalQMixer) instead of one, managing Q-values hierarchically across RSU zones.

## Discussion and decisions

### Audit findings

**DRQN Agent:** Fully reused. Both Mono-QMIX and Civiq use the same `RNNAgent` class from `pymarl/src/modules/agents/rnn_agent.py`. HierarchicalQLearner receives the MAC (which wraps RNNAgent) as a parameter, identical to QLearner. No custom agent class is defined in Civiq.

**Mixer internals:** LocalQMixer and GlobalQMixer are structural clones of QMixer's hypernetwork. All three share the same architecture: 1–2 layer hypernetworks, `hypernet_embed=64`, `mixing_embed_dim=32`, `abs(w1)` for monotonicity, `F.elu` activation in the mixing layer, and a 2-layer linear V(s) network. The only differences are input/output dimensions tied to the hierarchy level (n_agents → max_agents_per_rsu → max_rsus) and the absence of PopArt parameters in the two Civiq mixers.

**Learner:** HierarchicalQLearner does not inherit from QLearner — it is a fork. All non-mixer logic (MAC forward pass, double Q-learning, target network updates, reward normalization, device management) is directly mirrored from QLearner. The docstring acknowledges this: "Fork of QLearner — all non-mixer logic is identical." The deliberate divergences are the two-mixer hierarchical forward pass, RSUZoneManager, per-component gradient norm logging, and removal of PopArt.

### PopArt vs. EMA reward normalization
QMixer supports PopArt (Preserving Outputs Precisely while Adapting Reward Targets). LocalQMixer and GlobalQMixer do not — Civiq uses EMA-based reward normalization instead.

**Why EMA was chosen:** Applying PopArt to a hierarchical mixer is non-trivial. PopArt must rescale output layer weights whenever normalization statistics change to preserve the network's predictions. In a two-mixer setup, this rescaling would need to be coordinated across GlobalQMixer and LocalQMixer without breaking the other's gradient signal. EMA reward normalization — normalizing rewards before they enter the TD update — is simpler and sufficient to address the immediate problem (GlobalQMixer grad norms of 2–3M).

**Known bias introduced by EMA:** Replay buffer transitions stored under earlier reward statistics are sampled later under updated statistics, creating non-stationarity in the normalized targets. PopArt avoids this by rescaling the network output to always match the current normalization. The bias is real but small relative to the instability EMA was fixing. PopArt remains the more principled long-term recommendation (noted in 500k findings).

### Gradient flow through the hierarchical architecture
In Civiq, `loss.backward()` traverses a single computational graph in one pass — not sequential per-component descents. The gradient signal originates at the MSE loss and flows: Loss → GlobalQMixer → LocalQMixer (per RSU) → RNNAgent. All parameters are updated by the optimizer simultaneously after this single backward pass.

**Attenuation risk:** In standard Mono-QMIX, the agent gradient passes through one mixing layer. In Civiq, it passes through two (LocalQMixer, then GlobalQMixer), each applying learned nonlinear weight transformations. Depending on the learned weights, the gradient reaching RNNAgent can be weaker than in Mono-QMIX. The per-component grad norm logging (`agent_grad_norm`, `local_mixer_grad_norm`, `global_mixer_grad_norm`) exists specifically to monitor this. A pattern of `global >> local >> agent` grad norms would indicate the hierarchy is bottlenecking agent learning.

## What changed
- No code was changed. This was a read-only audit.

## Implementation notes
- All divergences between HierarchicalQLearner and QLearner are justified by the two-mixer design. No unintended divergences were found.
- `max_rsus` in `civiq_sumo.yaml` is correctly set to 17 (BGC Full). The audit agent incorrectly reported this as 12 — verified by direct file read.

## Status and follow-up
- **Status:** Complete — audit passed. Civiq correctly inherits Mono-QMIX's internal components within the bounds of the two-mixer hierarchy.
- **Follow-up items:**
  - Monitor `agent_grad_norm` vs `global_mixer_grad_norm` during smoketests to detect gradient attenuation through the two-mixer chain.
  - Consider PopArt integration at the GlobalQMixer output level as a future improvement over EMA normalization, per the 500k findings recommendation.
