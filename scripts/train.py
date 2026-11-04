"""Train one model on normal-only training clips, then score every test frame.

Usage:
    python scripts/train.py --config configs/ped2.yaml --model cae --seed 0
    python scripts/train.py --config configs/ped2.yaml --model vit --seed 0
    python scripts/train.py --config configs/ped2.yaml --model pca
    python scripts/train.py --config configs/ped2.yaml --model cae --no-motion   # ablation

Writes results/scores/<run>.npz (per-frame scores) and checkpoints/<run>.pt.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, RandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vad.data import ClipDataset, load_dataset, n_channels  # noqa: E402
from vad.models import build_model, count_params  # noqa: E402


def error_maps_to_scores(err: torch.Tensor, patch: int) -> tuple[np.ndarray, np.ndarray]:
    """err: (B,H,W) squared error -> (mean score, max patch-pooled score) per frame."""
    mean = err.flatten(1).mean(1)
    pooled = F.avg_pool2d(err[:, None], patch, stride=patch // 2).flatten(1).amax(1)
    return mean.cpu().numpy(), pooled.cpu().numpy()


@torch.no_grad()
def score_videos(predict, videos, cfg, motion, keep_maps=()):
    out, maps = {}, {}
    for name, frames in videos.items():
        ds = ClipDataset({name: frames}, cfg["clip_length"], motion)
        dl = DataLoader(ds, batch_size=64, shuffle=False)
        means, pooled, vmaps = [], [], []
        for x in dl:
            err = ((predict(x) - x) ** 2).mean(1)  # (B,H,W), averaged over channels
            m, p = error_maps_to_scores(err, cfg["score_patch"])
            means.append(m); pooled.append(p)
            if name in keep_maps:
                vmaps.append(err.numpy().astype(np.float16))
        out[name] = {"mean": np.concatenate(means), "patch_max": np.concatenate(pooled)}
        if vmaps:
            maps[name] = np.concatenate(vmaps)
    return out, maps


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_torch(model, train_videos, cfg, motion, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model.to(DEVICE)
    ds = ClipDataset(train_videos, cfg["clip_length"], motion)
    # Each epoch visits a random subset of clips (adjacent frames are near-duplicates)
    sampler = RandomSampler(ds, num_samples=int(len(ds) * cfg.get("epoch_fraction", 1.0)))
    dl = DataLoader(ds, batch_size=cfg["batch_size"], sampler=sampler, num_workers=0, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    epochs = cfg["epochs"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=epochs * len(dl))
    history = []
    for ep in range(epochs):
        model.train(); t0 = time.time(); tot = 0.0
        for x in dl:
            x = x.to(DEVICE)
            out = model(x)
            if isinstance(out, tuple):  # masked ViT: loss on hidden patches only
                rec, mask = out
                pe = ((model.patchify(rec) - model.patchify(x)) ** 2).mean(-1)
                loss = (pe * mask).sum() / mask.sum()
            else:
                loss = F.mse_loss(out, x)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(x)
        history.append(tot / (len(dl) * cfg["batch_size"]))
        print(f"  epoch {ep + 1:02d}/{epochs}  loss {history[-1]:.5f}  ({time.time() - t0:.0f}s)", flush=True)
    model.eval()
    return history


def run_pca(train_videos, test_videos, cfg, motion, keep_maps):
    """Classical baseline: PCA reconstruction of down-sampled clips."""
    from sklearn.decomposition import PCA

    ds = ClipDataset(train_videos, cfg["clip_length"], motion, stride=2)
    X = torch.stack([ds[i] for i in range(len(ds))])
    shape = X.shape[1:]
    small = F.avg_pool2d(X, 4).flatten(1).numpy()
    pca = PCA(n_components=cfg["pca_components"], svd_solver="randomized", random_state=0).fit(small)

    def predict(x):
        s = F.avg_pool2d(x, 4).flatten(1).numpy()
        r = pca.inverse_transform(pca.transform(s))
        r = torch.from_numpy(r).float().reshape(len(x), shape[0], shape[1] // 4, shape[2] // 4)
        return F.interpolate(r, size=shape[1:], mode="bilinear", align_corners=False).clamp(0, 1)

    return predict, pca


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/ped2.yaml"))
    ap.add_argument("--model", choices=["cae", "vit", "pca"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-motion", action="store_true", help="appearance-only input (ablation)")
    ap.add_argument("--epochs", type=int, help="override epochs in config")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cfg["epochs"] = args.epochs or cfg.get(f"epochs_{args.model}", cfg["epochs"])
    torch.set_num_threads(cfg.get("threads", 4))
    motion = not args.no_motion
    run = f"{args.model}{'' if motion else '-nomotion'}_s{args.seed}"
    print(f"[{run}] loading data ...", flush=True)
    data = load_dataset(cfg, ROOT)
    res = ROOT / cfg.get("results_dir", "results")
    keep = tuple(cfg.get("keep_maps", []))
    t0 = time.time()

    if args.model == "pca":
        predict, _ = run_pca(data["train"], data["test"], cfg, motion, keep)
        history, n_params = [], cfg["pca_components"] * n_channels(cfg["clip_length"], motion) * (cfg["img_size"][0] // 4) * (cfg["img_size"][1] // 4)
    else:
        torch.manual_seed(args.seed)  # seed controls weight init as well as sampling/masking
        model = build_model(args.model, n_channels(cfg["clip_length"], motion), tuple(cfg["img_size"]))
        n_params = count_params(model)
        print(f"[{run}] {n_params / 1e6:.2f}M parameters, training {cfg['epochs']} epochs on normal clips only ({DEVICE})", flush=True)
        history = train_torch(model, data["train"], cfg, motion, args.seed)
        ckpt = ROOT / cfg.get("checkpoint_dir", "checkpoints"); ckpt.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "cfg": cfg, "motion": motion}, ckpt / f"{run}.pt")
        def predict(x):
            return model(x.to(DEVICE)).cpu()
    train_time = time.time() - t0

    print(f"[{run}] scoring test videos ...", flush=True)
    t1 = time.time()
    test_scores, maps = score_videos(predict, data["test"], cfg, motion, keep)
    infer_time = time.time() - t1
    n_test = sum(len(v) for v in data["test"].values())
    train_scores, _ = score_videos(predict, data["train"], cfg, motion)

    out = res / "scores"; out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / f"{run}.npz",
        test=np.array(test_scores, dtype=object), train=np.array(train_scores, dtype=object),
        maps=np.array(maps, dtype=object),
        meta=np.array({"run": run, "model": args.model, "motion": motion, "seed": args.seed,
                       "params": int(n_params), "history": history, "train_sec": train_time,
                       "fps": n_test / infer_time, "epochs": cfg["epochs"] if args.model != "pca" else 0}, dtype=object),
    )
    print(f"[{run}] done: train {train_time:.0f}s, inference {n_test / infer_time:.1f} FPS (CPU)", flush=True)
    print(json.dumps({"run": run, "params": int(n_params)}))


if __name__ == "__main__":
    main()
