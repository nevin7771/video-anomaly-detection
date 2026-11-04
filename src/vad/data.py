"""Dataset loading, caching and clip sampling for video anomaly detection.

UCSD Ped2 layout (official release v1p2):
    UCSDped2/Train/Train001..Train016/*.tif   normal-only training videos
    UCSDped2/Test/Test001..Test012/*.tif      test videos (normal + anomalous)
    UCSDped2/Test/TestXXX_gt/*.bmp            pixel-level anomaly masks
    UCSDped2/Test/UCSDped2.m                  frame-level ground truth ranges

CUHK Avenue layout (official Avenue_Dataset.zip + ground_truth_demo.zip):
    Avenue/training_videos/01.avi..16.avi              normal-only training videos
    Avenue/testing_videos/01.avi..21.avi               test videos
    Avenue/**/testing_label_mask/<n>_label.mat         per-frame pixel masks (volLabel)
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


def _read_video(path: Path, size: tuple[int, int]) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    h, w = size
    frames = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        frames.append(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA))
    cap.release()
    if not frames:
        raise IOError(f"Could not read video {path}")
    return np.stack(frames).astype(np.uint8)


def load_avenue(root: str | Path, size=(128, 224), cache_dir: str | Path | None = None,
                keep_masks: tuple[str, ...] = ()) -> dict:
    """Load CUHK Avenue in the same dict format as load_ped2.

    Frame labels come from the pixel masks (a frame is anomalous if any pixel is).
    Full-resolution masks for ~15k frames do not fit comfortably in memory, so masks
    are only kept for the videos named in `keep_masks` (used for heat-map figures).
    """
    from scipy.io import loadmat

    root = Path(root)
    cache = None
    if cache_dir is not None:
        cache = Path(cache_dir) / f"avenue_{size[0]}x{size[1]}.npz"
        if cache.exists():
            return np.load(cache, allow_pickle=True)["data"].item()

    train = {p.stem: _read_video(p, size) for p in sorted((root / "training_videos").glob("*.avi"))}
    test = {p.stem: _read_video(p, size) for p in sorted((root / "testing_videos").glob("*.avi"))}
    gt_files = {p.name.split("_")[0].zfill(2): p for p in root.rglob("*_label.mat")
                if p.parent.name == "testing_label_mask"}
    labels, masks = {}, {}
    h, w = size
    for name, frames in test.items():
        if name not in gt_files:
            raise FileNotFoundError(f"No ground-truth mask file for Avenue test video {name}")
        vol = loadmat(gt_files[name])["volLabel"].ravel()
        n = min(len(vol), len(frames))  # a few videos decode one frame more than the masks
        test[name] = frames[:n]
        labels[name] = np.array([int(np.asarray(m).any()) for m in vol[:n]], dtype=np.uint8)
        if name in keep_masks:
            masks[name] = np.stack([cv2.resize(np.asarray(m, np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                                    for m in vol[:n]]).astype(np.uint8)
    data = {"train": train, "test": test, "labels": labels, "masks": masks, "size": size}
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, data=np.array(data, dtype=object))
    return data


def load_dataset(cfg: dict, root: Path) -> dict:
    """Dispatch on cfg['dataset'] (default 'ped2'); paths in cfg are relative to `root`."""
    name = cfg.get("dataset", "ped2")
    size, cache = tuple(cfg["img_size"]), root / cfg["cache_dir"]
    if name == "ped2":
        return load_ped2(root / cfg["data_root"], size, cache)
    if name == "avenue":
        return load_avenue(root / cfg["data_root"], size, cache, tuple(cfg.get("keep_maps", [])))
    raise ValueError(f"Unknown dataset {name!r}")


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
