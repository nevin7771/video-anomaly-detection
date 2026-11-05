"""Event-level error analysis and deployment numbers for the best run of each model.

For each model (seed 0, patch-max score, global protocol, label-free threshold tau =
99th percentile of smoothed training scores) this reports:
  * event recall: an anomalous event (a contiguous run of anomalous frames) is detected
    if any of its frames scores above tau; detection delay = frames to the first alarm
  * missed events and the size of their anomalous region (from the pixel masks)
  * false-alarm episodes (contiguous runs above tau on normal frames) per hour of video
  * inference throughput (fps measured during scoring, on whatever device ran train.py)

Usage: python scripts/analyse_errors.py [--config configs/ped2.yaml] [--fps-video 10]
Writes <results_dir>/metrics/error_analysis.json and error_analysis.md.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vad.data import load_dataset  # noqa: E402
from vad.metrics import smooth  # noqa: E402

ORDER = ["pca", "cae-nomotion", "cae", "vit"]
LABELS = {"pca": "PCA baseline", "cae-nomotion": "CAE (appearance only)", "cae": "CAE + motion",
          "vit": "Masked ViT-AE + motion"}


def runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous [start, end) runs where mask is True."""
    m = np.concatenate([[0], mask.astype(np.int8), [0]])
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def analyse(run, data, sigma, fps_video):
    tr = np.concatenate([smooth(v["patch_max"], sigma) for v in run["train"].values()])
    tau = float(np.percentile(tr, 99))
    events, false_alarms, normal_frames = [], 0, 0
    for name, y in data["labels"].items():
        s = smooth(run["test"][name]["patch_max"], sigma)
        alarm = s > tau
        for a, b in runs_of(y == 1):
            hit = np.where(alarm[a:b])[0]
            area = None
            if name in data["masks"]:
                area = float(data["masks"][name][a:b].mean(axis=(1, 2)).mean() * 100)
            events.append({"video": name, "start": int(a) + 1, "end": int(b), "frames": int(b - a),
                           "detected": bool(len(hit)), "delay_frames": int(hit[0]) if len(hit) else None,
                           "frac_frames_flagged": float(alarm[a:b].mean()), "area_pct": area,
                           "peak_over_tau": float(s[a:b].max() / tau)})
        normal = y == 0
        normal_frames += int(normal.sum())
        # a false-alarm episode = an alarm run that lies entirely on normal frames
        false_alarms += sum(1 for a, b in runs_of(alarm) if normal[a:b].all())
    det = [e for e in events if e["detected"]]
    hours = normal_frames / fps_video / 3600
    return {
        "tau": tau,
        "events": len(events), "events_detected": len(det),
        "event_recall": len(det) / max(len(events), 1),
        "median_delay_frames": float(np.median([e["delay_frames"] for e in det])) if det else None,
        "median_delay_sec": float(np.median([e["delay_frames"] for e in det]) / fps_video) if det else None,
        "false_alarm_episodes": false_alarms,
        "normal_hours": hours,
        "false_alarms_per_hour": false_alarms / hours if hours else None,
        "missed": [e for e in events if not e["detected"]],
        "event_detail": events,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/ped2.yaml"))
    ap.add_argument("--fps-video", type=float, default=None,
                    help="recording frame rate (default: cfg video_fps, else 10 for Ped2 / 25 for Avenue)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    fps_video = args.fps_video or cfg.get("video_fps") or (25 if cfg.get("dataset") == "avenue" else 10)
    res = ROOT / cfg.get("results_dir", "results")
    data = load_dataset(cfg, ROOT)
    files = sorted((res / "scores").glob("*.npz"))
    if not files:
        sys.exit(f"No runs in {res}/scores/. Train first.")
    out = {"fps_video": fps_video, "smoothing_sigma": cfg["smoothing_sigma"], "models": {}}
    for variant in ORDER:
        f = next((f for f in files if re.sub(r"_s\d+$", "", f.stem) == variant), None)
        if f is None:
            continue
        z = np.load(f, allow_pickle=True)
        run = {k: z[k].item() for k in ("test", "train", "meta")}
        r = analyse(run, data, cfg["smoothing_sigma"], fps_video)
        r["run"] = f.stem
        r["fps_cpu"] = run["meta"].get("fps")
        out["models"][variant] = r
        print(f"{LABELS[variant]:24s} events {r['events_detected']}/{r['events']}  "
              f"delay {r['median_delay_sec']}s  false alarms/h {r['false_alarms_per_hour']:.1f}")

    m = res / "metrics"; m.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(m / "error_analysis.json", "w"), indent=2)
    md = ["| Model | Events detected | Median delay (s) | False-alarm episodes / hour | Inference fps |",
          "|---|---|---|---|---|"]
    for v, r in out["models"].items():
        delay = f"{r['median_delay_sec']:.1f}" if r["median_delay_sec"] is not None else "–"
        fpsc = f"{r['fps_cpu']:.0f}" if r["fps_cpu"] else "–"
        md.append(f"| {LABELS[v]} | {r['events_detected']}/{r['events']} ({100 * r['event_recall']:.0f} %) | "
                  f"{delay} | {r['false_alarms_per_hour']:.1f} | {fpsc} |")
    md += ["", "Missed events (all models):", ""]
    for v, r in out["models"].items():
        for e in r["missed"]:
            area = f"{e['area_pct']:.2f} % of frame" if e["area_pct"] is not None else "area n/a"
            md.append(f"- {LABELS[v]}: {e['video']} frames {e['start']}–{e['end']} ({e['frames']} frames, {area}, "
                      f"peak score {e['peak_over_tau']:.2f}× tau)")
    (m / "error_analysis.md").write_text("\n".join(md) + "\n")
    print(f"Wrote {m}/error_analysis.json and error_analysis.md")


if __name__ == "__main__":
    main()
