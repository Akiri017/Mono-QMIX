$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"
Set-Location "c:\Users\otaku\Documents\VSC\Mono QMIX\Mono-QMIX"

$BASE = "results/mono-qmix/mono-qmix-los-c"

# QMIX - Best
python pymarl/src/evaluate.py `
  --model $BASE/best `
  --env_config sumo_bgc_full `
  --alg_config qmix_sumo `
  --los_level med `
  --episodes 30 `
  --seed 1802 `
  --output $BASE/qmix_exp_1802_best

# QMIX - Final
python pymarl/src/evaluate.py `
  --model $BASE/final `
  --env_config sumo_bgc_full `
  --alg_config qmix_sumo `
  --los_level med `
  --episodes 30 `
  --seed 1802 `
  --output $BASE/qmix_exp_1802_final

# Noop
python pymarl/src/evaluate.py `
  --baseline noop `
  --env_config sumo_bgc_full `
  --alg_config qmix_sumo `
  --los_level med `
  --episodes 30 `
  --seed 1802 `
  --output $BASE/noop_exp_1802

# Greedy
python pymarl/src/evaluate.py `
  --baseline greedy_shortest `
  --env_config sumo_bgc_full `
  --alg_config qmix_sumo `
  --los_level med `
  --episodes 30 `
  --seed 1802 `
  --output $BASE/greedy_shortest_exp_1802
