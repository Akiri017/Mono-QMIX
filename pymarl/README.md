# PyMARL vendoring

Recommended for this project: vendor PyMARL into this repo under `pymarl/` so the custom SUMO env can live at `pymarl/src/envs/` and be selected via PyMARL configs.

## Option A (recommended): vendor PyMARL here
From the repo root:
- `git clone https://github.com/oxwhirl/pymarl.git pymarl`

Then you can add the custom env under `pymarl/src/envs/` and keep all experiment configs in `pymarl/src/config/`.

## Option B: keep PyMARL external
Keep PyMARL in a separate folder/repo and add this repo's env code to that checkout.

Notes:
- Either way, SUMO must be installed separately and `SUMO_HOME` should be set.
