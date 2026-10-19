"""Verify the raw UCSD Ped2 download, then build the cached, resized tensor file.

Usage: python scripts/prepare_data.py [--config configs/ped2.yaml]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vad.data import load_ped2  # noqa: E402

# Official UCSD Ped2 v1p2 counts (Mahadevan et al., CVPR 2010)
EXPECTED = {"train_videos": 16, "test_videos": 12, "train_frames": 2550, "test_frames": 2010}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/ped2.yaml"))
    cfg = yaml.safe_load(open(ap.parse_args().config))
    root = ROOT / cfg["data_root"]
    if not (root / "Train").exists():
        sys.exit(f"Dataset not found at {root}. Run: bash scripts/download_ped2.sh")

    data = load_ped2(root, tuple(cfg["img_size"]), ROOT / cfg["cache_dir"])
    found = {
        "train_videos": len(data["train"]), "test_videos": len(data["test"]),
        "train_frames": sum(len(v) for v in data["train"].values()),
        "test_frames": sum(len(v) for v in data["test"].values()),
    }
    ok = all(found[k] == v for k, v in EXPECTED.items())
    for k, v in EXPECTED.items():
        print(f"  {k:13s} expected {v:5d}  found {found[k]:5d}  {'OK' if found[k] == v else 'MISMATCH'}")
    # Every test video must have a frame label and a pixel mask per frame
    for name, frames in data["test"].items():
        assert len(data["labels"][name]) == len(frames), name
        assert name in data["masks"] and len(data["masks"][name]) == len(frames), name
    anom = sum(int(v.sum()) for v in data["labels"].values())
    found["test_anomalous_frames"] = anom
    found["test_anomalous_pct"] = round(100 * anom / found["test_frames"], 1)
    found["verified"] = ok
    out = ROOT / "results/metrics"; out.mkdir(parents=True, exist_ok=True)
    json.dump(found, open(out / "dataset_check.json", "w"), indent=2)
    print(f"Integrity check {'PASSED' if ok else 'FAILED'}; {anom} anomalous test frames ({found['test_anomalous_pct']}%).")
    print(f"Cached tensors in {cfg['cache_dir']}/")


if __name__ == "__main__":
    main()
