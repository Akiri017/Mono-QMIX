# Checkpointing Pipeline

Implemented in preparation for the 500k Kaggle training run, where the 12hr session limit makes it likely that training will be interrupted before completion.

---

## What Was Changed

### `pymarl/src/learners/q_learner.py`

**`save_models(path)`** — extended to also save optimizer state:
```
agent.pth       — RNN agent weights
mixer.pth       — QMIX mixer weights
optimizer.pth   — Adam optimizer state (momentum + variance)
```
Saving the optimizer state matters because without it, Adam restarts with zero momentum on resume, which disrupts the learning trajectory.

**`load_models(path)`** — extended to restore optimizer state if `optimizer.pth` exists. Gracefully skips if loading an older checkpoint that predates this change.

---

### `pymarl/src/main.py`

**Checkpoint saves now write `training_state.json`** alongside the model files:
```json
{"t_env": 100000, "episode_num": 1000}
```
This records exactly where training was when the checkpoint fired.

**New `--resume_from <path>` CLI argument** — when provided:
- Calls `learner.load_models(path)` to restore weights + optimizer
- Reads `training_state.json` to restore `t_env` and `episode_num`
- Patches `runner.t_env` so epsilon annealing, target network updates, and all intervals continue from the correct step rather than restarting from 0

---

## Checkpoint Directory Layout

Each periodic save produces a directory like:

```
results/models/seed0/
  step_100000/
    agent.pth
    mixer.pth
    optimizer.pth
    training_state.json
  step_200000/
    ...
  best/          ← saved by validation (no training_state.json)
  final/         ← saved on training end (no training_state.json)
```

`save_model_interval` in `qmix_sumo.yaml` controls how often step checkpoints fire (default: `100000`).

---

## Resuming a Run

```bash
python pymarl/src/main.py \
  --t_max 500000 \
  --resume_from results/models/seed0/step_400000 \
  --checkpoint_path results/models/seed0
```

Expected output at startup:
```
[Resuming from checkpoint: .../step_400000]
  Resumed at t_env=400000, episode_num=4000
```

**Important:** always point `--checkpoint_path` to the same seed folder the original run used. This ensures the `best/` model gets updated by the resumed run and is available for evaluation afterward.

---

## Running Evaluation After a Resumed Run

`run_experiments.py` has an `--eval_only` flag that skips training entirely, loads the existing `best` model, evaluates QMIX against all baselines, and prints the full comparison table.

```bash
python run_experiments.py \
  --seeds 0 \
  --eval_episodes 20 \
  --eval_only \
  --checkpoint_root results/models
```

It expects the model at `<checkpoint_root>/seed<seed>/best`. As long as `--checkpoint_path` pointed to `results/models/seed0` during training (including the resumed segment), this will find the right model automatically.

---

## Local Verification Procedure

Before a long Kaggle run, verify the pipeline with a short local test:

1. Temporarily set `save_model_interval: 2000` in `qmix_sumo.yaml`
2. Run a short training pass:
   ```bash
   python run_experiments.py --seeds 0 --t_max 10000 --eval_episodes 5
   ```
3. Confirm checkpoint dirs appear with all four files (`agent.pth`, `mixer.pth`, `optimizer.pth`, `training_state.json`)
4. Resume from any checkpoint:
   ```bash
   python pymarl/src/main.py --t_max 10000 --resume_from results/models/seed0/step_2000 --checkpoint_path results/models/resume_test
   ```
5. Confirm the resume message prints the correct `t_env`
6. Reset `save_model_interval: 100000` before the Kaggle run
