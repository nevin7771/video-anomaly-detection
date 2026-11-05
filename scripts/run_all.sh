#!/usr/bin/env bash
# Full preliminary pipeline: data -> EDA -> train/score all models -> evaluate.
# CPU-only (4 cores): ~2 h for everything.
# Quick smoke test:  EPOCHS=2 SEEDS="0" bash scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SEEDS=${SEEDS:-"0 1 2"}
EP=${EPOCHS:+--epochs $EPOCHS}

bash scripts/download_ped2.sh
python scripts/prepare_data.py
python scripts/eda.py
python scripts/train.py --model pca
python scripts/train.py --model cae --no-motion --seed 0 $EP   # ablation (1 seed)
for s in $SEEDS; do
  python scripts/train.py --model cae --seed "$s" $EP
  python scripts/train.py --model vit --seed "$s" $EP
done
python scripts/evaluate.py
python scripts/analyse_errors.py
