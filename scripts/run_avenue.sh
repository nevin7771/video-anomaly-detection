#!/usr/bin/env bash
# CUHK Avenue with the identical Ped2 pipeline. Needs a GPU for a sensible run time
# (Colab T4: roughly 1-2 h for everything). Outputs go to results/avenue/.
# Quick smoke test:  EPOCHS=1 SEEDS="0" bash scripts/run_avenue.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SEEDS=${SEEDS:-"0 1 2"}
EP=${EPOCHS:+--epochs $EPOCHS}
CFG=configs/avenue.yaml

bash scripts/download_avenue.sh
python scripts/prepare_data.py --config $CFG
python scripts/train.py --config $CFG --model pca
python scripts/train.py --config $CFG --model cae --no-motion --seed 0 $EP
for s in $SEEDS; do
  python scripts/train.py --config $CFG --model cae --seed "$s" $EP
  python scripts/train.py --config $CFG --model vit --seed "$s" $EP
done
python scripts/evaluate.py --config $CFG
python scripts/analyse_errors.py --config $CFG
