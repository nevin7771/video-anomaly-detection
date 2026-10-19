"""Exploratory data analysis of UCSD Ped2: composition, anomaly prevalence,
motion statistics and anomaly region size. Writes figures + JSON to results/.

Usage: python scripts/eda.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vad import plotstyle as ps  # noqa: E402
from vad.data import load_ped2  # noqa: E402
from vad.metrics import separation_stats  # noqa: E402

ps.apply()
import matplotlib.pyplot as plt  # noqa: E402


def motion_energy(frames):
    f = frames.astype(np.float32) / 255
    d = np.abs(np.diff(f, axis=0)).mean((1, 2))
    return np.concatenate([[d[0]], d])


def main():
    cfg = yaml.safe_load(open(ROOT / "configs/ped2.yaml"))
    data = load_ped2(ROOT / cfg["data_root"], tuple(cfg["img_size"]), ROOT / cfg["cache_dir"])
    fig_dir = ps.figure_dir(ROOT)
    names = list(data["test"])

    # 1. Sample frames: normal training frame vs anomalous test frames with GT outline
    picks = [("Train001", None, 60), ("Test002", "Test002", 120), ("Test004", "Test004", 90), ("Test007", "Test007", 120)]
    fig, axes = plt.subplots(1, 4, figsize=(14, 2.9))
    for ax, (vid, gt, t) in zip(axes, picks):
        frames = data["train"][vid] if gt is None else data["test"][vid]
        ax.imshow(frames[t], cmap="gray"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        if gt is not None:
            ax.contour(data["masks"][gt][t], levels=[0.5], colors=[ps.ANOM], linewidths=1.8)
            ax.set_title(f"{vid} · frame {t + 1} · anomaly", fontsize=10, color=ps.INK)
        else:
            ax.set_title(f"{vid} · frame {t + 1} · normal", fontsize=10, color=ps.INK)
    fig.savefig(fig_dir / "eda_sample_frames.png"); plt.close(fig)

    # 2. Anomalous-frame share per test video
    frac = [100 * data["labels"][n].mean() for n in names]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar([n.replace("Test", "") for n in names], frac, color=ps.SERIES["cae"], width=0.6)
    ax.set_ylabel("% frames anomalous"); ax.set_xlabel("Test video")
    ax.set_title("Anomaly prevalence per test video"); ax.set_ylim(0, 105); ax.grid(axis="x", visible=False)
    for i, v in enumerate(frac):
        ax.text(i, v + 2, f"{v:.0f}", ha="center", fontsize=9, color=ps.INK2)
    fig.savefig(fig_dir / "eda_anomaly_prevalence.png"); plt.close(fig)

    # 3. Motion energy: normal vs anomalous frames (justifies motion channels)
    y = np.concatenate([data["labels"][n] for n in names])
    me = np.concatenate([motion_energy(data["test"][n]) for n in names])
    me_train = np.concatenate([motion_energy(v) for v in data["train"].values()])
    st = separation_stats(y, me)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    bins = np.linspace(0, np.percentile(me, 99.5), 50)
    ax.hist(me_train, bins, density=True, alpha=0.55, color=ps.SERIES["cae-nomotion"], label="Train (normal)")
    ax.hist(me[y == 0], bins, density=True, histtype="step", lw=2, color=ps.SERIES["cae"], label="Test normal")
    ax.hist(me[y == 1], bins, density=True, histtype="step", lw=2, color=ps.ANOM, label="Test anomalous")
    ax.set_xlabel("Mean |frame difference| (motion energy)"); ax.set_ylabel("Density")
    ax.set_title("Anomalous frames carry more motion"); ax.legend()
    fig.savefig(fig_dir / "eda_motion_energy.png"); plt.close(fig)

    # 4. Anomaly region size (fraction of frame covered by GT mask)
    area = np.concatenate([data["masks"][n].reshape(len(data["masks"][n]), -1).mean(1) for n in names])
    area = 100 * area[y == 1]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.hist(area, bins=40, color=ps.SERIES["cae"])
    ax.axvline(np.median(area), color=ps.INK, lw=1.5, ls="--")
    ax.text(np.median(area), ax.get_ylim()[1] * 0.9, f"  median {np.median(area):.1f}% of frame", fontsize=10)
    ax.set_xlabel("Anomalous region, % of frame area"); ax.set_ylabel("Frames")
    ax.set_title("Anomalies are small and local"); ax.grid(axis="x", visible=False)
    fig.savefig(fig_dir / "eda_anomaly_area.png"); plt.close(fig)

    summary = {
        "train_videos": len(data["train"]), "test_videos": len(names),
        "train_frames": int(sum(len(v) for v in data["train"].values())),
        "test_frames": int(len(y)), "test_anomalous_frames": int(y.sum()),
        "test_anomalous_pct": round(100 * float(y.mean()), 1),
        "fully_anomalous_test_videos": [n for n in names if data["labels"][n].all()],
        "frame_size_native": [240, 360], "frame_size_model": cfg["img_size"],
        "motion_energy": {"train_mean": float(me_train.mean()), "test_normal_mean": float(me[y == 0].mean()),
                          "test_anomalous_mean": float(me[y == 1].mean()), **st},
        "anomaly_area_pct": {"median": float(np.median(area)), "p90": float(np.percentile(area, 90)),
                             "max": float(area.max())},
    }
    out = ROOT / "results/metrics"; out.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(out / "eda_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
