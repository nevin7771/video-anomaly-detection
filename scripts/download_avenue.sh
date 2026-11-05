#!/usr/bin/env bash
# Download CUHK Avenue (~1.9 GB videos + ground-truth masks) into data/raw/Avenue.
# Source: official CUHK project page (Lu, Shi & Jia, ICCV 2013). Run this in Colab or
# locally; some sandboxed environments block the CUHK host.
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=data/raw/Avenue
if [ -d "$DEST/training_videos" ] && [ -d "$DEST/testing_videos" ] && \
   find "$DEST" -path '*testing_label_mask/*_label.mat' | grep -q .; then
  echo "CUHK Avenue already present at $DEST"; exit 0
fi
BASE=http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal
mkdir -p data/raw "$DEST"
for f in Avenue_Dataset.zip ground_truth_demo.zip; do
  [ -f "data/raw/$f" ] || curl -fL --connect-timeout 30 --retry 3 -o "data/raw/$f" "$BASE/$f"
done
TMP=$(mktemp -d)
unzip -q data/raw/Avenue_Dataset.zip -d "$TMP"
unzip -q data/raw/ground_truth_demo.zip -d "$TMP"
# The zip nests everything under "Avenue Dataset/"; flatten into $DEST
SRC=$(dirname "$(find "$TMP" -type d -name training_videos | head -1)")
mv "$SRC/training_videos" "$SRC/testing_videos" "$DEST/"
mv "$(find "$TMP" -type d -name testing_label_mask | head -1)" "$DEST/testing_label_mask"
rm -rf "$TMP"
echo "Done. Verify with: python scripts/prepare_data.py --config configs/avenue.yaml"
