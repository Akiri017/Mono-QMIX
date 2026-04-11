# Discussion: BGC Full-Map Disqualification, Civiq Inheritance, and Thesis Framing Tradeoffs

**Date:** 2026-04-09
**Status:** Ongoing — decision not yet made
**Affected areas:** Thesis scope, evaluation plan, Civiq architecture framing, BGC full-map evaluation

## Context

Following the 500k smoke test confirming mixer divergence across all LOS levels on the synthetic 4x4 grid, a broader set of tradeoffs was raised before committing to a final scope decision. Three interconnected problems were identified and are currently being weighed.

## Discussion

### Problem 1: Is LOS A divergence sufficient evidence to disqualify the BGC full-map run?

The argument for disqualifying: the synthetic 4x4 grid is the easiest possible test case — controlled topology, uniform demand, fixed agent count (32 intersections regardless of LOS level). Divergence here is structural, driven by bootstrap amplification through the GlobalQMixer hypernetwork, not by traffic scenario complexity. The BGC full map adds larger state/observation space, more agents, irregular topology, and higher return variance — all of which would accelerate divergence, not mitigate it. The standard justification for moving from synthetic to full map is that the system works on the controlled case. That justification does not hold if the synthetic run is itself a failure mode.

The counterargument: a smoke test is one seed, one run. Divergence at LOS A does not guarantee divergence at the same rate on every seed or scenario. The BGC result might still yield a meaningful best-checkpoint result even if the learning window is shorter.

**Not yet decided.** The tradeoff being weighed: ruling out BGC entirely based on synthetic evidence vs. running it and risking a null result with no reportable output under deadline constraints.

### Problem 2: Civiq inherits and likely amplifies the mixer instability

Civiq uses a two-level mixer stack:

```
q_individual → LocalMixer_i → Q_RSU_i → GlobalMixer → Q_tot
```

The TD bootstrap propagates through GlobalMixer using Q_tot targets. If Q_tot diverges, the gradient signal reaching LocalMixers is corrupted. Since LocalMixers produce Q_RSU values that feed GlobalMixer, the amplification is cascading — two divergence paths rather than one. Civiq's meaningful pre-divergence learning window is likely shorter than Mono QMIX's ~200k steps, though the exact threshold is unknown without a Civiq smoke test.

A secondary question: Civiq's local state is the concatenation of its member agents' states, but the reward is likely global (localized observation ≠ localized reward — independent design choices). If the global network reward is passed to LocalMixers, they face the same reward scale problem as Mono QMIX's single mixer. A localized per-RSU reward would reduce the scale problem, but true localization is non-trivial — waiting time and queue length can be computed per-RSU (point metrics), but mean travel time and emissions cannot be cleanly attributed without boundary artifacts and double-counting.

**Implication being weighed:** If Civiq diverges earlier than Mono QMIX, the comparison between them may reflect divergence rate more than architectural capability. PopArt may be needed at both mixer levels for Civiq, not just the GlobalMixer.

### Problem 3: Thesis framing breaks if both systems diverge

Three compounding problems were identified:

**Broken baseline.** Mono QMIX is the baseline for Civiq. If Mono QMIX's result is a constrained best-checkpoint at t=200k before mixer divergence, comparing Civiq against it measures which architecture finds a better policy in its narrow pre-divergence window — not scalability or architectural capability.

**Broken comparison.** If both systems diverge for the same root cause, a reviewer can correctly point out the comparison is confounded by training instability. Performance differences are as easily explained by divergence rate as by design.

**Broken BGC justification chain.** Moving from synthetic to full map requires demonstrating the system works on the controlled case. If the synthetic result is itself a failure mode, the justification for BGC evaluation does not hold.

## Tradeoffs being weighed

| Option | Upside | Risk |
|---|---|---|
| Contain scope to synthetic map only, reframe contribution | Honest, defensible, avoids null results | Thesis contribution is narrower; BGC was a stated goal |
| Run Civiq on synthetic map and compare against Mono QMIX best checkpoint | Produces a baseline-vs-extension comparison even if limited | Civiq might diverge early enough that there is nothing meaningful to report |
| Run BGC regardless and report what you get | Maximizes scope, fulfills original plan | Likely produces an earlier and more severe divergence with nothing reportable |
| Fix the mixer before evaluation (PopArt, gamma reduction) | Addresses root cause, unlocks full scope | High implementation risk under current deadline |

## Open questions before deciding

- Is Civiq currently runnable on the synthetic map? If not, the scope narrows further regardless.
- Does Civiq use global reward at all mixing levels, or is per-RSU reward already implemented?
- Is there enough time to run a Civiq smoke test on the synthetic map before the deadline?
- If Civiq produces no meaningful pre-divergence result, is Mono QMIX synthetic results alone a sufficient thesis?
