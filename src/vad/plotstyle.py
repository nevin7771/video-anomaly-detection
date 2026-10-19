"""Shared matplotlib style so every figure in the report and slides matches.

Set VAD_THEME=dark to render slide-ready dark figures into results/figures/dark/.
"""
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DARK = os.environ.get("VAD_THEME", "light") == "dark"

# Categorical slots in fixed order (validated colour-blind-safe reference palette; dark steps for dark bg)
if DARK:
    SERIES = {"pca": "#c98500", "cae-nomotion": "#199e70", "cae": "#3987e5", "vit": "#d95926"}
    INK, INK2, GRID, BG, ANOM, OK = "#eaeef2", "#8a94a3", "#2a323f", "#0e1116", "#e66767", "#22d3ee"
else:
    SERIES = {"pca": "#eda100", "cae-nomotion": "#1baf7a", "cae": "#2a78d6", "vit": "#eb6834"}
    INK, INK2, GRID, BG, ANOM, OK = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff", "#e34948", "#1baf7a"
LABELS = {"pca": "PCA baseline", "cae-nomotion": "CAE (appearance only)", "cae": "CAE + motion",
          "vit": "Masked ViT-AE + motion"}


def figure_dir(root: Path) -> Path:
    d = root / "results/figures" / ("dark" if DARK else "")
    d.mkdir(parents=True, exist_ok=True)
    return d


def apply():
    plt.rcParams.update({
        "font.family": ["Inter", "DejaVu Sans"], "font.size": 11, "text.color": INK, "axes.edgecolor": INK2,
        "axes.labelcolor": INK, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlecolor": INK, "axes.facecolor": BG, "figure.facecolor": BG, "savefig.facecolor": BG,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False,
        "legend.labelcolor": INK, "figure.dpi": 150, "savefig.bbox": "tight", "lines.linewidth": 2,
    })
