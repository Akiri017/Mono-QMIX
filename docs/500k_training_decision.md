# Architecture Decision: Accept Mono QMIX Training Limitations, Fix Training Config for Final Run

**Date:** 2026-04-08
**Status:** Complete (decision made; final training run pending)
**Affected areas:** Training configuration (`qmix_sumo.yaml`), thesis framing, results documentation

## Context

A 500k smoke test was run on Kaggle (seed=4) with reward normalization enabled to validate whether the EMA reward normalizer fixed the aggressive GlobalQMixer gradient norms (~2–3.5M) observed in prior runs. The goal was to determine whether training could proceed to the original 2M-step budget and establish a defensible training ceiling for Mono QMIX.

## Discussion and decisions

### Reward normalization outcome — partially worked, root cause misdiagnosed

The EMA running stats converged and stabilized by t=40k (mean≈-2480, std≈790) and remained flat for the rest of the run. The normalization itself functioned correctly. However, loss still grew from ~3 at t=20k to ~9,417 at t=500k and grad_norm reached 2.4M by t=500k. Reward normalization alone was insufficient because the divergence is not in the reward scale — it is in the mixer.

### Root cause identified: mixer bootstrap amplification

Individual agent Q-values (`q_taken_mean`) stabilized at ≈-2.4 from t=200k onwards. Q_tot targets (`target_mean`) grew from -15 to -1471 over the same period — a 600x gap. The GlobalQMixer weights diverge through the TD bootstrap loop: as Q_tot grows slightly, the target `r_norm + γ*Q_tot(s')` grows, which drives Q_tot further. Reward normalization only addresses the `r` term; the dominant `γ*Q_tot` bootstrap term is unaffected.

### Best checkpoint falls at t=200k — not epsilon-driven

The t=200k peak (validation travel_time=10.87s) was questioned as potentially coincidental or correlated with ε=0.62. Analysis ruled out epsilon as the causal factor: the validation curve is W-shaped (10.87s at t=200k, regression to 15.1s at t=250k, partial recovery to 11.5s at t=400k–450k). A monotonically decreasing epsilon cannot explain a regression immediately after the peak. The real explanation is timing — t=200k falls just before mixer divergence crosses a critical threshold:

| Step | Loss | Grad Norm | Validation Travel Time |
|---|---|---|---|
| t=150k | 494 | 21,559 | 12.6s |
| t=200k | 684 | 26,595 | **10.87s (best)** |
| t=250k | 984 | 116,925 | 15.1s |
| t=300k | 1,178 | 83,294 | 13.3s |
| t=400k | 3,453 | 535,416 | 11.5s |
| t=450k | 6,347 | 1,437,455 | 11.5s |
| t=500k | 9,417 | 2,406,942 | — |

### Training budget ceiling considered — rejected in favor of full 500k

Options evaluated:

| Option | Assessment | Rejected because |
|---|---|---|
| Cap at 300k, anneal at 300k | Pragmatic | ε=0.43 at stop — policy never trains in near-greedy conditions; epsilon_anneal_time mismatch is a methodological error |
| Cap at 200k, anneal at 200k | Stops at known peak | Consolidation phase entirely unobserved; no low-epsilon training |
| Fix mixer (gamma reduction, PopArt, per-module clipping) | Correct fix | High implementation risk given thesis deadline |
| **Keep 500k, anneal 500k, report best checkpoint** | **Decided** | Smoke test already ran this full schedule; anneal completed as designed (ε≈0.06 at t=495k); best checkpoint identifiable via validation-based model selection |

The late-stage low-epsilon window confirmed the full anneal has value: t=450k (ε=0.145) produced the best observed test return of the entire run (-217k) and stable 11.5s validation. The full schedule is not wasted — the policy survives the divergence because individual Q-networks remain healthy. Only the mixer weights diverge.

## What changed

- **Training configuration:** No changes to `qmix_sumo.yaml` — the existing 500k/500k anneal config is confirmed correct for the final run
- **Thesis framing:** Best checkpoint reported via validation-based model selection, not final checkpoint; divergence documented as an observed failure mode with full diagnostic evidence
- **Recommendation section:** PopArt / Q_tot-level target normalization added as the identified fix for mixer bootstrap divergence

## Implementation notes

The smoke test (seed=4) produced a 5-episode final evaluation against three baselines using the best checkpoint (t=200k):

| Policy | Mean Return | Travel Time | Waiting Time | Arrival Rate |
|---|---|---|---|---|
| QMIX (best, t=200k) | -243,444 ± 25,849 | 13.46s ± 2.28 | 24.29s | 0.700 |
| noop | -239,259 ± 34,259 | 14.53s ± 3.88 | 23.32s | 0.691 |
| greedy_shortest | -241,762 ± 31,381 | 13.53s ± 3.11 | 21.74s | 0.700 |
| random | -254,069 ± 29,711 | 12.54s ± 2.64 | 24.82s | 0.714 |

Differences are within noise at n=5. The evaluation design requires ≥20 episodes per policy for statistically valid comparisons.

The best checkpoint at t=200k (validation 10.87s) measurably outperformed the initial random policy (12.2s) on validation — the meaningful signal is the validation trajectory, not the noisy 5-episode final eval.

## Status and follow-up

**Status:** Complete — decision finalized, final training run pending

**Follow-up items:**
- Run final training (500k, seed TBD) and collect best checkpoint via validation
- Increase final evaluation to ≥20 episodes per policy for statistically valid comparisons
- Document mixer divergence diagnostics (`loss`, `grad_norm`, `target_mean`, `q_taken_mean` trajectories) in thesis limitations section
- **Recommendation — PopArt:** Normalize Q_tot targets using a running estimate of their scale and de-normalize mixer output at inference. Addresses bootstrap amplification at the correct level (Q_tot, not reward). This is the identified next step for the architecture beyond thesis scope.
