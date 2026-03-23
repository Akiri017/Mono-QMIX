# Design choices (locked in — 2026-03-17)

This document captures the decisions we finalized for the Mono_QMIX SUMO+QMIX project.

## Scenario realism vs fixed-size MARL

### Partial adoption model
- Only a subset of vehicles use the learned navigation policy (these are the **controlled fleet** / agents).
- The remaining vehicles are **background traffic** and follow default routing or a baseline.
- This mirrors real navigation apps (e.g., Waze/Google Maps): not everyone cooperates, but the policy should still improve overall network efficiency.

### Fixed-N controlled fleet with replacement (lifelong agent slots)
- The MARL interface exposes a fixed number of agent slots `N_controlled` (e.g., 32).
- Each slot controls at most one vehicle at a time.
- When a controlled vehicle arrives at its destination:
  - the slot becomes inactive for a short configurable delay, then
  - a new controlled vehicle (new OD/destination) is spawned and assigned to that slot.

Rationale:
- Keeps PyMARL/QMIX tensors fixed-size while allowing realistic open-system traffic.
- Avoids long warm-up periods that occur when departures are spread across the full horizon.

## Reward (network objective)

### Primary objective: global/system reward
- Train primarily on a **network-level objective** measured over *all* vehicles (controlled + background).
- Example: negative total time loss / waiting time proxy aggregated over the network.

### Stabilization
- Add a small stabilizer term (regularizer) to reduce variance and discourage pathological routing behavior, e.g.:
  - penalty for frequent route changes (reroute cost)
  - smoothing/anti-oscillation penalty
- If training remains too noisy even with the stabilizer, optionally include a **tiny** controlled-only component as a secondary term, but keep the global term dominant.

## Mid-episode slot resets and masking

### Alive/reset masking (chosen approach)
- Because replacement can occur mid-episode, some agent slots will reset while the episode continues.
- The environment will expose masks that make this correct for learning:
  - `active_mask[i]`: whether slot `i` currently controls an in-network vehicle
  - `reset_mask[i]`: whether slot `i` was just reassigned to a new vehicle on this timestep

Expected training semantics:
- Inactive slots should have only a no-op action available.
- Loss/TD targets should ignore inactive slots via masking.

### RNN state reset on replacement
- If the agent network is recurrent (e.g., PyMARL’s default RNN agent), the hidden state for a slot must be reset when a new vehicle occupies that slot.

Rationale:
- Prevents hidden-state leakage between unrelated vehicle trajectories.
- Preserves stability while keeping learned knowledge in shared network weights.

## Notes / implications
- Metrics should be reported for both:
  - controlled fleet (to see what the policy is doing), and
  - the whole network (to validate network-level benefit).
- Penetration/adoption studies become straightforward by varying `N_controlled` and/or background demand level.
