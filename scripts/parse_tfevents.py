"""
Parse TensorBoard tfevents files for Mono-QMIX training runs and update
the frontend metrics JSON files with high-resolution training curves.

Strategy:
  - Use text-log training.curve for the pre-tfevents range (loss, gradNorm, etc.)
  - Use tfevents episode_return for the high-res range (per-episode returns)
  - Attach nearest-neighbor loss/gradNorm/qTaken/targetMean for tfevents points
  - Recompute moving average across the merged curve

Usage:
  python scripts/parse_tfevents.py
"""

import json
import os
import math
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT         = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR  = os.path.join(ROOT, 'results', 'mono-qmix')
FRONTEND_DIR = os.path.join(ROOT, 'frontend', 'data', 'mono_qmix')

SCENARIOS = {
    'los_a': {
        'logs_dir':    os.path.join(RESULTS_DIR, 'mono-qmix-los-a', 'logs'),
        'metrics_file': os.path.join(FRONTEND_DIR, 'metrics_los_a.json'),
    },
    'los_c': {
        'logs_dir':    os.path.join(RESULTS_DIR, 'mono-qmix-los-c', 'logs'),
        'metrics_file': os.path.join(FRONTEND_DIR, 'metrics_los_c.json'),
    },
    'los_e': {
        'logs_dir':    os.path.join(RESULTS_DIR, 'mono-qmix-los-e', 'logs'),
        'metrics_file': os.path.join(FRONTEND_DIR, 'metrics_los_e.json'),
    },
}

MA_WINDOW = 20


def moving_average(values, w):
    result = []
    for i, v in enumerate(values):
        window = values[max(0, i - w): i + 1]
        result.append(sum(window) / len(window))
    return result


def nearest(scalars, step):
    """Return value of nearest scalar event to `step`, or None if list empty."""
    if not scalars:
        return None
    best = min(scalars, key=lambda e: abs(e.step - step))
    # Only attach if within 10k steps
    if abs(best.step - step) <= 10000:
        return round(float(best.value), 6)
    return None


def load_tfevents(logs_dir):
    ea = EventAccumulator(logs_dir)
    ea.Reload()
    tags = ea.Tags().get('scalars', [])

    def get(tag):
        return ea.Scalars(tag) if tag in tags else []

    return {
        'episode_return': get('episode_return'),
        'loss':           get('loss'),
        'grad_norm':      get('grad_norm'),
        'q_taken_mean':   get('q_taken_mean'),
        'target_mean':    get('target_mean'),
    }


def normalize_entry(p):
    """Normalize text-log curve entry to canonical format (handles 'return' vs 'reward')."""
    t = p['t']
    reward = p.get('reward', p.get('return'))
    return {
        't':         t,
        'episode':   p.get('episode', round(t / 100)),
        'reward':    reward,
        'loss':      p.get('loss'),
        'gradNorm':  p.get('gradNorm'),
        'qTaken':    p.get('qTaken'),
        'targetMean': p.get('targetMean'),
    }


def build_merged_curve(text_curve, tb):
    ep_events = tb['episode_return']
    tb_start  = ep_events[0].step if ep_events else float('inf')

    # -- Text-log segment: all points strictly before tfevents coverage --------
    pre = [
        normalize_entry(p)
        for p in text_curve
        if p['t'] < tb_start
    ]

    # -- TensorBoard segment: per-episode returns + nearest sparse metrics -----
    tb_seg = []
    for ev in ep_events:
        t = ev.step
        tb_seg.append({
            't':         t,
            'episode':   round(t / 100),
            'reward':    round(float(ev.value), 2),
            'loss':      nearest(tb['loss'],       t),
            'gradNorm':  nearest(tb['grad_norm'],  t),
            'qTaken':    nearest(tb['q_taken_mean'], t),
            'targetMean': nearest(tb['target_mean'], t),
        })

    merged = pre + tb_seg

    # -- Recompute moving average across entire curve --------------------------
    rewards = [p['reward'] for p in merged]
    ma      = moving_average(rewards, MA_WINDOW)
    for i, p in enumerate(merged):
        p['ma'] = round(ma[i], 2)

    return merged


def note_for(scenario, tb_start):
    step_str = f't={tb_start:,}'
    return (
        f'Merged: text-log before {step_str}, '
        f'TensorBoard episode_return from {step_str} to t=1,000,000 '
        f'(per-episode resolution in TB range)'
    )


def process(scenario, cfg):
    print(f'\n=== {scenario.upper()} ===')

    # Load existing metrics JSON
    with open(cfg['metrics_file'], encoding='utf-8') as f:
        metrics = json.load(f)

    text_curve = metrics.get('training', {}).get('curve', [])
    print(f'  Text-log curve points: {len(text_curve)}')

    # Load tfevents
    print(f'  Loading tfevents from: {cfg["logs_dir"]}')
    tb = load_tfevents(cfg['logs_dir'])
    ep = tb['episode_return']
    print(f'  TensorBoard episode_return points: {len(ep)}'
          + (f', step {ep[0].step}–{ep[-1].step}' if ep else ''))

    if not ep:
        print('  No episode_return data — skipping')
        return

    merged = build_merged_curve(text_curve, tb)
    tb_start = ep[0].step

    metrics['training']['curve'] = merged
    metrics['training']['note']  = note_for(scenario, tb_start)

    with open(cfg['metrics_file'], 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f'  Merged curve: {len(merged)} points '
          f'({len([p for p in merged if p["t"] < tb_start])} text-log '
          f'+ {len(ep)} TB)')
    print(f'  Written to {cfg["metrics_file"]}')


if __name__ == '__main__':
    for scenario, cfg in SCENARIOS.items():
        process(scenario, cfg)
    print('\nDone.')
