"""Dataset loading, caching and clip sampling for video anomaly detection.

UCSD Ped2 layout (official release v1p2):
    UCSDped2/Train/Train001..Train016/*.tif   normal-only training videos
    UCSDped2/Test/Test001..Test012/*.tif      test videos (normal + anomalous)
    UCSDped2/Test/TestXXX_gt/*.bmp            pixel-level anomaly masks
    UCSDped2/Test/UCSDped2.m                  frame-level ground truth ranges
"""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _read_frames(folder: Path, size: tuple[int, int], exts=(".tif", ".bmp", ".jpg", ".png")) -> np.ndarray:
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)
    h, w = size
    frames = []
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Could not read frame {f}")
        frames.append(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA))
    return np.stack(frames).astype(np.uint8)


def parse_ped_gt(m_file: Path, n_frames: list[int]) -> list[np.ndarray]:
    """Parse UCSDpedX.m into per-video binary frame labels (1 = anomalous)."""
    text = m_file.read_text()
    labels = []
    for i, block in enumerate(re.findall(r"gt_frame\s*=\s*\[([^\]]*)\]", text)):
        y = np.zeros(n_frames[i], dtype=np.uint8)
        for a, b in re.findall(r"(\d+)\s*:\s*(\d+)", block):
            y[int(a) - 1:int(b)] = 1
        labels.append(y)
    return labels


def load_ped2(root: str | Path, size=(128, 192), cache_dir: str | Path | None = None) -> dict:
    """Load UCSD Ped2 as grayscale uint8 arrays, caching to .npz for fast reuse."""
    root = Path(root)
    cache = None
    if cache_dir is not None:
        cache = Path(cache_dir) / f"ped2_{size[0]}x{size[1]}.npz"
        if cache.exists():
            z = np.load(cache, allow_pickle=True)
            return z["data"].item()

    train_dirs = sorted(d for d in (root / "Train").iterdir() if d.is_dir() and re.fullmatch(r"Train\d+", d.name))
    test_dirs = sorted(d for d in (root / "Test").iterdir() if d.is_dir() and re.fullmatch(r"Test\d+", d.name))
    train = {d.name: _read_frames(d, size) for d in train_dirs}
    test = {d.name: _read_frames(d, size) for d in test_dirs}
    masks = {}
    for d in test_dirs:
        gt_dir = d.parent / f"{d.name}_gt"
        if gt_dir.exists():
            masks[d.name] = (_read_frames(gt_dir, size) > 127).astype(np.uint8)
    labels = parse_ped_gt(root / "Test" / "UCSDped2.m", [len(v) for v in test.values()])
    data = {
        "train": train,
        "test": test,
        "labels": dict(zip(test.keys(), labels)),
        "masks": masks,
        "size": size,
    }
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, data=np.array(data, dtype=object))
    return data


def clip_indices(n: int, t: int, length: int) -> np.ndarray:
    """Indices of a clip of `length` frames centred on frame t (edges clamped)."""
    half = length // 2
    idx = np.arange(t - half, t - half + length)
    return np.clip(idx, 0, n - 1)


def make_clip(frames: np.ndarray, t: int, length: int, motion: bool) -> np.ndarray:
    """Build a (C,H,W) float32 clip: `length` frames, plus |frame diffs| if motion."""
    x = frames[clip_indices(len(frames), t, length)].astype(np.float32) / 255.0
    if motion:
        d = np.abs(np.diff(x, axis=0))
        d = np.clip(d * 4.0, 0, 1)  # amplify small motion cues
        x = np.concatenate([x, d], axis=0)
    return x


class ClipDataset(Dataset):
    """Every frame of every video becomes the centre of one clip."""

    def __init__(self, videos: dict[str, np.ndarray], length=5, motion=True, stride=1):
        self.videos = videos
        self.length = length
        self.motion = motion
        self.items = [(k, t) for k, v in videos.items() for t in range(0, len(v), stride)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        k, t = self.items[i]
        return torch.from_numpy(make_clip(self.videos[k], t, self.length, self.motion))


def n_channels(length: int, motion: bool) -> int:
    return length + (length - 1 if motion else 0)
