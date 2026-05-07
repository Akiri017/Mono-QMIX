"""
Parse CiViQ training data and update frontend metrics JSON files.

Sources:
  LOS A — TensorBoard tfevents (7557 episode_return pts, step 100–755700)
  LOS C — Text log civiq-los-c.txt (complete, t=5000–1000000)
  LOS E — Text log civiq-los-e.txt (complete, t=5000–1000000)

Output fields (matches QmixTrainingEntry in page.tsx):
  t, episode, reward, ma, loss, gradNorm, qTaken, targetMean

Field mapping (CiViQ → canonical):
  episode_return  → reward
  loss            → loss
  agent_grad_norm → gradNorm
  global_qtot_mean→ qTaken
  targetMean      → null (not logged by CiViQ)

Usage:
  python scripts/parse_civiq_training.py
"""

import json, re, os
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT        = os.path.dirname(SCRIPT_DIR)
CIVIQ_DIR   = os.path.join(ROOT, 'results', 'civiq')
FRONTEND    = os.path.join(ROOT, 'frontend', 'data', 'civiq')

MA_WINDOW = 20


def moving_average(values, w):
    result = []
    for i, v in enumerate(values):
        window = values[max(0, i - w): i + 1]
        result.append(sum(window) / len(window))
    return result


def nearest(rows, step, field, tol=10000):
    """Return field value from nearest row within tol steps, or None."""
    best, best_d = None, tol + 1
    for r in rows:
        d = abs(r['t'] - step)
        if d < best_d and r.get(field) is not None:
            best_d, best = d, r[field]
    return round(best, 6) if best is not None else None


# ── Text log parser ────────────────────────────────────────────────────────────

STAT_RE = re.compile(r'^([\w]+):\s*([-\d.eE+]+)\s*\(t=(\d+)\)')

WANTED = {
    'episode_return':   'reward',
    'loss':             'loss',
    'agent_grad_norm':  'gradNorm',
    'global_qtot_mean': 'qTaken',
}

def parse_text_log(path):
    """Parse a CiViQ text log into a list of per-step dicts."""
    by_t = defaultdict(dict)
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            m = STAT_RE.match(line.strip())
            if not m:
                continue
            key, val, t = m.group(1), float(m.group(2)), int(m.group(3))
            if key in WANTED:
                by_t[t][WANTED[key]] = val

    rows = []
    for t in sorted(by_t):
        r = by_t[t]
        if 'reward' not in r:
            continue
        rows.append({
            't':         t,
            'episode':   round(t / 100),
            'reward':    round(r['reward'], 2),
            'loss':      round(r['loss'], 6)     if 'loss'     in r else None,
            'gradNorm':  round(r['gradNorm'], 6) if 'gradNorm' in r else None,
            'qTaken':    round(r['qTaken'], 6)   if 'qTaken'   in r else None,
            'targetMean': None,
        })
    return rows


# ── TensorBoard parser ─────────────────────────────────────────────────────────

def parse_tfevents(logs_dir):
    """Load tfevents and return (ep_events, sparse_rows)."""
    ea = EventAccumulator(logs_dir)
    ea.Reload()
    tags = ea.Tags().get('scalars', [])

    def get(tag):
        return ea.Scalars(tag) if tag in tags else []

    ep_events = get('episode_return')
    loss_ev   = get('loss')
    gn_ev     = get('agent_grad_norm')
    qt_ev     = get('global_qtot_mean')

    # Build sparse rows for loss/gradNorm/qTaken
    sparse = []
    all_t  = sorted(set([e.step for e in loss_ev] +
                         [e.step for e in gn_ev]   +
                         [e.step for e in qt_ev]))
    loss_d = {e.step: e.value for e in loss_ev}
    gn_d   = {e.step: e.value for e in gn_ev}
    qt_d   = {e.step: e.value for e in qt_ev}
    for t in all_t:
        sparse.append({
            't':         t,
            'loss':      round(loss_d[t], 6) if t in loss_d else None,
            'gradNorm':  round(gn_d[t],   6) if t in gn_d   else None,
            'qTaken':    round(qt_d[t],   6) if t in qt_d   else None,
        })

    return ep_events, sparse


# ── Curve builders ─────────────────────────────────────────────────────────────

def build_from_tfevents(ep_events, sparse):
    rows = []
    for ev in ep_events:
        t = ev.step
        rows.append({
            't':         t,
            'episode':   round(t / 100),
            'reward':    round(float(ev.value), 2),
            'loss':      nearest(sparse, t, 'loss'),
            'gradNorm':  nearest(sparse, t, 'gradNorm'),
            'qTaken':    nearest(sparse, t, 'qTaken'),
            'targetMean': None,
        })
    return rows


def add_ma(rows):
    rewards = [r['reward'] for r in rows]
    ma      = moving_average(rewards, MA_WINDOW)
    for r, m in zip(rows, ma):
        r['ma'] = round(m, 2)
    return rows


def downsample(rows, max_pts):
    if len(rows) <= max_pts:
        return rows
    step = -(-len(rows) // max_pts)   # ceiling division
    return [r for i, r in enumerate(rows) if i % step == 0 or i == len(rows) - 1]


# ── Per-scenario processing ────────────────────────────────────────────────────

def process_los_a():
    logs_dir = os.path.join(CIVIQ_DIR, 'civiq-los-a', 'logs')
    print('  Loading tfevents from:', logs_dir)
    ep_events, sparse = parse_tfevents(logs_dir)
    print(f'  episode_return: {len(ep_events)} pts  '
          f'(step {ep_events[0].step}–{ep_events[-1].step})')
    rows = build_from_tfevents(ep_events, sparse)
    return rows, f'TensorBoard: {len(ep_events)} per-episode returns, step 100–{ep_events[-1].step}'


def process_los_text(log_path, label):
    print('  Parsing text log:', log_path)
    rows = parse_text_log(log_path)
    print(f'  Parsed {len(rows)} entries')
    return rows, f'Text log: {len(rows)} entries, t={rows[0]["t"]}–{rows[-1]["t"]}'


def process(los, rows, note):
    metrics_file = os.path.join(FRONTEND, f'metrics_los_{los}.json')
    with open(metrics_file, encoding='utf-8') as f:
        metrics = json.load(f)

    rows = add_ma(rows)
    metrics['training']['curve'] = rows
    metrics['training']['note']  = note

    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f'  Written {len(rows)} pts to {metrics_file}')


if __name__ == '__main__':
    print('\n=== CiViQ LOS A (tfevents) ===')
    rows_a, note_a = process_los_a()
    process('a', rows_a, note_a)

    print('\n=== CiViQ LOS C (text log) ===')
    rows_c, note_c = process_los_text(
        os.path.join(CIVIQ_DIR, 'civiq-los-c', 'civiq-los-c.txt'), 'LOS C')
    process('c', rows_c, note_c)

    print('\n=== CiViQ LOS E (text log) ===')
    rows_e, note_e = process_los_text(
        os.path.join(CIVIQ_DIR, 'civiq-los-e', 'civiq-los-e.txt'), 'LOS E')
    process('e', rows_e, note_e)

    print('\nDone.')
