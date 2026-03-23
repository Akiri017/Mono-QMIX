# Mono_QMIX

## Step 1 — Environment + tooling setup (Windows)

### Prereqs
- Install **Eclipse SUMO** and ensure these are available in your terminal:
  - `sumo`, `sumo-gui`, `netgenerate`
- Set `SUMO_HOME` to the SUMO install root (the folder that contains `bin/` and `tools/`).

### Python
This repo uses a local virtual environment in `.venv/`.

Install Python deps:
- `python -m pip install -r requirements.txt`

Verify everything:
- `python scripts/check_setup.py`

Expected output includes:
- SUMO binaries found on `PATH`
- `SUMO_HOME` printed
- Successful `import traci` and `import sumolib`

### Notes
- If `traci`/`sumolib` imports fail, SUMO also ships those modules under `$SUMO_HOME/tools`.
  The checker attempts to add that directory to `sys.path` as a fallback.

### PyMARL
For easiest integration later, vendor PyMARL into this repo under `pymarl/`:
- `powershell -ExecutionPolicy Bypass -File scripts/clone_pymarl.ps1`
