#!/usr/bin/env bash
# Download UCSD Ped2 (~460 MB of frames) into data/raw/UCSDped2.
# Tries the official UCSD server first, then a GitHub mirror of the unmodified
# v1p2 release (sparse checkout so only Ped2 is fetched).
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=data/raw/UCSDped2
if [ -d "$DEST/Train" ] && [ -d "$DEST/Test" ]; then
  echo "UCSD Ped2 already present at $DEST"; exit 0
fi
mkdir -p data/raw
OFFICIAL=http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz
echo "Trying official source: $OFFICIAL"
if curl -fL --connect-timeout 20 -o data/raw/UCSD_Anomaly_Dataset.tar.gz "$OFFICIAL"; then
  tar -xzf data/raw/UCSD_Anomaly_Dataset.tar.gz -C data/raw UCSD_Anomaly_Dataset.v1p2/UCSDped2
  mv data/raw/UCSD_Anomaly_Dataset.v1p2/UCSDped2 "$DEST"
  rm -rf data/raw/UCSD_Anomaly_Dataset.v1p2 data/raw/UCSD_Anomaly_Dataset.tar.gz
else
  echo "Official server unreachable; using GitHub mirror (junaidwahid/UCSD-Anomaly-dataset)"
  TMP=$(mktemp -d)
  git clone --depth 1 --filter=blob:none --sparse https://github.com/junaidwahid/UCSD-Anomaly-dataset.git "$TMP/ucsd"
  git -C "$TMP/ucsd" sparse-checkout set UCSD_Anomaly_Dataset.v1p2/UCSDped2
  mv "$TMP/ucsd/UCSD_Anomaly_Dataset.v1p2/UCSDped2" "$DEST"
  rm -rf "$TMP"
fi
find "$DEST" \( -name '.DS_Store' -o -name '._*' -o -name '*.m~' \) -delete
echo "Done. Verify with: python scripts/prepare_data.py"
