# Phase 8 — Mono-QMIX Full 1M Training: BGC Full Results

**Date:** 2026-04-16 → 2026-04-18
**Branch:** `bgc-full-mono-qmix`
**Status:** Complete — all three LOS levels trained and evaluated

---

## Objective

Establish the Mono-QMIX baseline on BGC Full at full training scale (1M steps)
across three levels of service: LOS A (low), LOS C (medium), LOS E (high).

These results serve as the primary comparison point for CiViQ — the hierarchical
QMIX architecture this thesis proposes. Mono-QMIX uses standard QMIX with all
32 agents sharing a single global mixer, no RSU zone partitioning.

---

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Algorithm | `qmix_sumo` |
| Environment | `sumo_bgc_full` |
| t_max | 1,000,000 steps |
| Eval episodes | 30 per policy |
| Baselines | noop, greedy_shortest |
| Batch size | 32 |
| Buffer size | 400 episodes |
| Learning rate | 0.0005 |
| Epsilon decay | 1.0 → 0.05 over 500k steps |
| Save interval | Every 50,000 steps |

| LOS | Seed | Traffic Level |
|-----|------|---------------|
| A   | 1801 | Low (`trips_low.xml`) |
| C   | 1802 | Medium (`trips_med.xml`) |
| E   | 1803 | High (`trips_highEnough.xml`) |

---

## Training Details

### Wall-Clock Times

| LOS | Start | Finish | Duration |
|-----|-------|--------|----------|
| A | 5:26 AM Apr 16 | 10:00 AM Apr 17 | ~28h 34min |
| C | 3:09 AM Apr 16 | 5:00 PM Apr 17 | ~37h 51min |
| E | 6:00 AM Apr 16 | 12:00 PM Apr 17 | ~30h 00min |

LOS C wall-clock is inflated by three crashes and restarts. Compute time
per run was similar to LOS A and E (~28–30h).

### Best Model Checkpoints

| LOS | Best Model t | Notes |
|-----|-------------|-------|
| A | 900,100 | Peaked late; slight degradation from 900k→1M |
| C | — | Tracked in Drive; multiple resume sessions |
| E | 500,100 | Policy frozen after 500k; final = best |

### Training Platform

- Google Colab T4 GPU (1.19 compute units/hr)
- Drive-backed checkpointing every 50k steps
- libsumo backend (in-process, no watchdog)
- LOS C required 3 crash recoveries (200k, 600k, 700k) due to OOM from
  7.4 GB replay buffer on resume; resolved by deleting buffer before reload
  (acceptable at >500k steps — epsilon already at 0.05)

---

## Results

### Return (30-episode mean, best model)

| LOS | QMIX | Noop | Greedy | QMIX vs Noop | QMIX vs Greedy |
|-----|------|------|--------|--------------|----------------|
| A | -45,791 | -45,378 | -45,320 | -0.9% | -1.0% |
| C | -54,641 | -52,550 | -53,272 | -4.0% | -2.6% |
| E | -105,961 | -94,370 | -96,345 | -12.3% | -10.0% |

Negative = worse. QMIX underperforms both baselines across all LOS levels.

### Mean Travel Time (seconds)

| LOS | QMIX | Noop | Greedy | QMIX vs Noop |
|-----|------|------|--------|--------------|
| A | 124.0 | — | — | — |
| C | 108.3 | 122.2 | 123.3 | **+11.4% better** |
| E | 124.8 | 125.5 | 126.4 | +0.6% better |

### Arrival Rate

| LOS | QMIX | Noop | Greedy |
|-----|------|------|--------|
| A | 0.857 | — | — |
| C | 0.859 | 0.871 | 0.867 |
| E | 0.786 | 0.860 | 0.850 |

### Total Stops

| LOS | QMIX | Noop | Greedy |
|-----|------|------|--------|
| C | 22,785 | 16,434 | 17,495 |
| E | 47,937 | 29,663 | 32,614 |

QMIX consistently generates more vehicle stops than baselines. This is the
primary driver of worse returns despite comparable or better travel times.

---

## Key Findings

### LOS A — Marginal difference, QMIX nearly matches baselines
At low traffic density, there is limited congestion for QMIX to exploit.
The reward signal provides weak gradient because most episodes have similar
outcomes regardless of rerouting decisions. QMIX converges late (peak at
t=900k) but still finishes ~1% below baseline.

### LOS C — Better travel time, worse return
QMIX achieves 11% improvement in mean travel time (108s vs 122s noop) and
14% improvement for controlled vehicles specifically (98s vs 114s). However
the reward function penalises total stops heavily, and QMIX produces 39%
more stops than noop. Net effect: worse return despite faster movement.

This suggests QMIX is learning to route vehicles through faster-moving
corridors at the cost of more stop-and-go behaviour elsewhere. A potential
multi-objective conflict in the reward design.

### LOS E — Cascading congestion failure
At high traffic density, QMIX is significantly worse across all metrics.
The policy plateaued completely at t=500k — the second 500k steps produced
zero improvement (final = best). Throughput collapsed: 1,675 veh/h vs
2,187 veh/h for noop (23% lower). Stops increased 62% over noop.

The single global mixer cannot coordinate 32 agents effectively at highdensity — rerouting decisions by one agent cascade into congestion for

others. This is the central motivation for CiViQ's zone-partitioned
hierarchical architecture.

---

## Motivation for CiViQ

| Failure mode | Mono-QMIX | CiViQ response |
|---|---|---|
| Global mixer scales poorly at high density | All 32 agents in one joint Q-value | Zone-local mixers (RSU level) + global coordinator |
| Cascading rerouting at LOS E | No spatial structure | Zone boundaries limit cascade radius |
| Stop penalty dominates reward | No spatial awareness of downstream effects | RSU zones give agents local congestion context |

Mono-QMIX's MAC (agent Q-network + GRU) weights can be transferred directly
to CiViQ as a warm start — obs_dim=751 and the RNN architecture are identical.
Only the mixer layers differ.

---

## Result Files

```
results/mono-qmix/
├── mono-qmix-los-a/
│   ├── experiment_summary.json       # aggregated A/C/E eval
│   ├── noop_exp_1801.json
│   ├── greedy_shortest_exp_1801.json
│   ├── qmix_exp_1801.json            # best model eval
│   ├── seed_1801/best/               # t=900,100 checkpoint
│   └── seed_1801/final/              # t=1,000,000 checkpoint
├── mono-qmix-los-c/
│   ├── experiment_summary.json
│   ├── noop_exp_1802.json
│   ├── greedy_shortest_exp_1802.json
│   ├── seed_1802/best/qmix_exp_1802_best.json
│   └── seed_1802/final/qmix_exp_1802_final.json
└── mono-qmix-los-e/
    ├── experiment_summary.json
    ├── noop_exp_1803.json
    ├── greedy_shortest_exp_1803.json
    ├── seed_1803/best/qmix_exp_1803_los_e_best.json   # t=500,100
    └── seed_1803/final/qmix_exp_1803_los_e_final.json # t=1,000,000 (identical)
```

---

## Next Steps

| Item | Notes |
|------|-------|
| CiViQ warm-start from Mono-QMIX weights | Transfer MAC weights; mixer layers trained from scratch |
| CiViQ training — BGC Full LOS A/C/E | Same seeds (1801/1802/1803) for direct comparison |
| Reward normalization | Consider reweighting stop penalty vs travel time for LOS C |
