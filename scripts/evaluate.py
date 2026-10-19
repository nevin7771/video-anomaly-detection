"""Evaluate every scored run in results/scores/ and produce tables + figures.

Protocols reported (all frame-level, Gaussian-smoothed over time):
  * global     : raw scores pooled over all videos, one threshold for every video.
                 PRIMARY protocol: it matches deployment and the unsupervised threshold tau.
  * per-video  : scores min-max normalised within each test video (protocol of Liu et al.
                 2018 / MNAD 2020). Reported for comparison with the literature, but on Ped2 it
                 is distorted: 4 of the 12 test videos are 100 % anomalous, so normalising
                 within them stretches anomalous frames down to a score of 0.
Scoring variants: 'mean' (mean squared error over the frame) and 'patch_max'
(max over 16x16 patches; better for small, local anomalies).

Usage: python scripts/evaluate.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vad import plotstyle as ps  # noqa: E402
from vad.data import load_ped2  # noqa: E402
from vad.metrics import (bootstrap_auc, frame_metrics, minmax, separation_stats, smooth,  # noqa: E402
                         threshold_metrics)

ps.apply()
import matplotlib.pyplot as plt  # noqa: E402

ORDER = ["pca", "cae-nomotion", "cae", "vit"]
PRIMARY = ("patch_max", "global")


def load_runs():
    runs = {}
    for f in sorted((ROOT / "results/scores").glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        runs[f.stem] = {k: z[k].item() for k in ("test", "train", "maps", "meta")}
    return runs


def video_scores(run, labels, key, protocol, sigma):
    ys, ss = [], []
    for name, y in labels.items():
        s = smooth(run["test"][name][key], sigma)
        ss.append(minmax(s) if protocol == "per-video" else s)
        ys.append(y)
    return ys, ss


def main():
    cfg = yaml.safe_load(open(ROOT / "configs/ped2.yaml"))
    sigma = cfg["smoothing_sigma"]
    data = load_ped2(ROOT / cfg["data_root"], tuple(cfg["img_size"]), ROOT / cfg["cache_dir"])
    labels = data["labels"]
    runs = load_runs()
    if not runs:
        sys.exit("No runs in results/scores/. Train first (bash scripts/run_all.sh).")

    rows = []
    for rname, run in runs.items():
        variant = re.sub(r"_s\d+$", "", rname)
        for key in ("mean", "patch_max"):
            for protocol in ("per-video", "global"):
                ys, ss = video_scores(run, labels, key, protocol, sigma)
                y, s = np.concatenate(ys), np.concatenate(ss)
                rows.append({"run": rname, "variant": variant, "score": key, "protocol": protocol,
                             **frame_metrics(y, s)})
    # Aggregate across seeds
    agg = defaultdict(list)
    for r in rows:
        agg[(r["variant"], r["score"], r["protocol"])].append(r)
    table = []
    for (variant, key, protocol), rs in agg.items():
        entry = {"variant": variant, "label": ps.LABELS.get(variant, variant), "score": key, "protocol": protocol,
                 "n_seeds": len(rs)}
        for m in ("auc", "eer", "pr_auc"):
            v = np.array([r[m] for r in rs])
            entry[m] = float(v.mean()); entry[m + "_std"] = float(v.std())
        table.append(entry)
    table.sort(key=lambda e: (ORDER.index(e["variant"]) if e["variant"] in ORDER else 99, e["score"], e["protocol"]))

    # Primary protocol. Pick the seed-0 run of each variant for detail.
    primary = {}
    for variant in ORDER:
        rname = next((r for r in runs if re.sub(r"_s\d+$", "", r) == variant), None)
        if rname is None:
            continue
        run = runs[rname]
        ys, ss = video_scores(run, labels, *PRIMARY, sigma)
        y, s = np.concatenate(ys), np.concatenate(ss)
        lo, hi = bootstrap_auc(ys, ss)
        # Unsupervised operating threshold: 99th percentile of smoothed TRAIN (normal) scores
        tr = np.concatenate([smooth(v["patch_max"], sigma) for v in run["train"].values()])
        gys, gss = video_scores(run, labels, "patch_max", "global", sigma)
        thr = threshold_metrics(np.concatenate(gys), np.concatenate(gss), float(np.percentile(tr, 99)))
        meta = run["meta"]
        primary[variant] = {"run": rname, **frame_metrics(y, s), "auc_ci95": [lo, hi],
                            **separation_stats(y, s), "threshold_p99_train": thr,
                            "params": meta["params"], "fps_cpu": meta["fps"], "train_sec": meta["train_sec"],
                            "epochs": meta["epochs"], "history": meta["history"]}

    out = ROOT / "results/metrics"; out.mkdir(parents=True, exist_ok=True)
    json.dump({"primary_protocol": list(PRIMARY), "table": table, "primary": primary, "runs": rows}, open(out / "summary.json", "w"), indent=2)

    # Markdown table
    md = ["| Model | Score | Protocol | Seeds | AUC-ROC | EER | PR-AUC |", "|---|---|---|---|---|---|---|"]
    for e in table:
        md.append(f"| {e['label']} | {e['score']} | {e['protocol']} | {e['n_seeds']} | "
                  f"{e['auc']:.3f} ± {e['auc_std']:.3f} | {e['eer']:.3f} | {e['pr_auc']:.3f} |")
    (out / "results_table.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    fig_dir = ps.figure_dir(ROOT)
    variants = [v for v in ORDER if v in primary]

    # Fig: AUC per model (primary protocol, mean ± std across seeds)
    prim = {e["variant"]: e for e in table if (e["score"], e["protocol"]) == PRIMARY}
    fig, ax = plt.subplots(figsize=(8, 3.6))
    vals = [100 * prim[v]["auc"] for v in variants]
    errs = [100 * prim[v]["auc_std"] for v in variants]
    ax.barh([ps.LABELS[v] for v in variants], vals, xerr=errs, color=[ps.SERIES[v] for v in variants],
            height=0.55, error_kw={"ecolor": ps.INK2, "capsize": 3, "lw": 1})
    for i, v in enumerate(vals):
        ax.text(v + errs[i] + 0.6, i, f"{v:.1f}", va="center", fontsize=11, color=ps.INK, fontweight="bold")
    ax.axvline(90, color=ps.INK2, ls="--", lw=1); ax.text(90.2, len(vals) - 0.45, "target 90", fontsize=9, color=ps.INK2)
    ax.set_xlim(50, 102); ax.set_xlabel("Frame-level AUC-ROC (%)"); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_title("UCSD Ped2 · frame-level AUC by model")
    fig.savefig(fig_dir / "auc_by_model.png"); plt.close(fig)

    # Fig: ROC curves
    fig, ax = plt.subplots(figsize=(5, 4.6))
    for v in variants:
        ys, ss = video_scores(runs[primary[v]["run"]], labels, *PRIMARY, sigma)
        fpr, tpr, _ = roc_curve(np.concatenate(ys), np.concatenate(ss))
        ax.plot(fpr, tpr, color=ps.SERIES[v], label=f"{ps.LABELS[v]} ({100 * primary[v]['auc']:.1f})")
    ax.plot([0, 1], [0, 1], color=ps.GRID, ls="--", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.set_title("ROC · UCSD Ped2")
    ax.legend(loc="lower right", fontsize=9)
    fig.savefig(fig_dir / "roc_curves.png"); plt.close(fig)

    best = max(variants, key=lambda v: primary[v]["auc"])
    brun = runs[primary[best]["run"]]

    # Fig: score timelines with ground truth shading
    vids = cfg["keep_maps"]
    fig, axes = plt.subplots(len(vids), 1, figsize=(9, 2.1 * len(vids)), sharex=False)
    for ax, name in zip(np.atleast_1d(axes), vids):
        y = labels[name]
        s = minmax(smooth(brun["test"][name]["patch_max"], sigma))
        t = np.arange(1, len(y) + 1)
        ax.fill_between(t, 0, 1, where=y == 1, color=ps.ANOM, alpha=0.15, step="mid", lw=0, label="Ground-truth anomaly")
        ax.plot(t, s, color=ps.SERIES[best], label="Anomaly score")
        ax.set_ylim(0, 1.05); ax.set_xlim(1, len(y)); ax.set_ylabel(name, fontsize=10)
    np.atleast_1d(axes)[0].legend(loc="upper left", fontsize=9, ncol=2)
    np.atleast_1d(axes)[0].set_title(f"Frame anomaly score over time · {ps.LABELS[best]}")
    np.atleast_1d(axes)[-1].set_xlabel("Frame")
    fig.savefig(fig_dir / "score_timelines.png"); plt.close(fig)

    # Fig: localisation heat maps (frame | error map | overlay with GT)
    rows_ = []
    for name in vids:
        if name in brun["maps"]:
            y = labels[name]; idx = np.where(y == 1)[0]
            rows_.append((name, int(idx[len(idx) // 2])))
    if rows_:
        fig, axes = plt.subplots(len(rows_), 3, figsize=(11, 2.7 * len(rows_)))
        axes = np.atleast_2d(axes)
        for r, (name, t) in enumerate(rows_):
            frame = data["test"][name][t]; err = brun["maps"][name][t].astype(np.float32)
            gt = data["masks"][name][t]
            axes[r, 0].imshow(frame, cmap="gray"); axes[r, 0].set_title(f"{name} · frame {t + 1}", fontsize=10)
            axes[r, 1].imshow(err, cmap="magma"); axes[r, 1].set_title("Reconstruction error", fontsize=10)
            axes[r, 2].imshow(frame, cmap="gray"); axes[r, 2].imshow(err, cmap="magma", alpha=0.55)
            axes[r, 2].contour(gt, levels=[0.5], colors=[ps.OK], linewidths=1.6)
            axes[r, 2].set_title("Overlay + ground-truth outline", fontsize=10)
            for a in axes[r]:
                a.set_xticks([]); a.set_yticks([]); a.grid(False)
        fig.savefig(fig_dir / "localisation_heatmaps.png"); plt.close(fig)

    # Fig: single clean overlay (title-slide image) from the best deep model
    deep = [v for v in ("cae", "vit") if v in primary]
    if deep:
        drun = runs[primary[max(deep, key=lambda v: primary[v]["auc"])]["run"]]
        name = "Test004" if "Test004" in drun["maps"] else next(iter(drun["maps"]), None)
        if name:
            idx = np.where(labels[name] == 1)[0]; t = int(idx[len(idx) // 2])
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.imshow(data["test"][name][t], cmap="gray")
            ax.imshow(drun["maps"][name][t].astype(np.float32), cmap="magma", alpha=0.55)
            ax.axis("off")
            fig.savefig(fig_dir / "cover_frame.png", bbox_inches="tight", pad_inches=0); plt.close(fig)

    # Fig: training curves
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for v in variants:
        h = primary[v]["history"]
        if h:
            ax.plot(np.arange(1, len(h) + 1), np.array(h) / h[0], color=ps.SERIES[v], label=ps.LABELS[v])
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (relative to epoch 1)"); ax.set_title("Training convergence")
    ax.legend(fontsize=9)
    fig.savefig(fig_dir / "training_curves.png"); plt.close(fig)

    print(f"\nBest model: {ps.LABELS[best]}  AUC {primary[best]['auc']:.3f} "
          f"(95% CI {primary[best]['auc_ci95'][0]:.3f}-{primary[best]['auc_ci95'][1]:.3f})")


if __name__ == "__main__":
    main()
