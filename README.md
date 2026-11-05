# Video Anomaly Detection for Intelligent Surveillance

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nevin7771/video-anomaly-detection/blob/main/notebooks/VAD_Colab.ipynb)

MSc Data Science final-year project (24EMDT7218). Unsupervised video anomaly detection: models
learn *normal* spatio-temporal behaviour from normal-only footage and flag frames they cannot
reconstruct. Compares a Convolutional Autoencoder (CAE) with a masked Vision-Transformer
autoencoder (ViT-AE), plus a PCA baseline, under one fixed preprocessing pipeline.

## Project layout

```
configs/ped2.yaml          experiment settings (image size, clip length, epochs, ...)
src/vad/data.py            dataset loading, ground-truth parsing, clip sampling
src/vad/models.py          ConvAutoencoder and MaskedViTAutoencoder
src/vad/metrics.py         AUC-ROC, EER, PR-AUC, bootstrap CI, Mann-Whitney U, thresholding
src/vad/plotstyle.py       shared figure style
scripts/download_ped2.sh   downloads UCSD Ped2 into data/raw/
scripts/download_avenue.sh downloads CUHK Avenue into data/raw/ (official CUHK host)
scripts/prepare_data.py    integrity check + cached tensors
scripts/eda.py             exploratory analysis figures
scripts/train.py           train one model on normal clips, score every test frame
scripts/evaluate.py        metrics tables + result figures
scripts/analyse_errors.py  event-level misses, detection delay, false alarms per hour
scripts/run_all.sh         the whole Ped2 pipeline end to end
scripts/run_avenue.sh      the same pipeline on CUHK Avenue (outputs in results/avenue/)
docs/literature_review.md  literature review, conceptual framework, references
results/                   metrics (JSON/Markdown) and figures from the last run
```

## Quick start

```bash
git clone https://github.com/nevin7771/video-anomaly-detection.git
cd video-anomaly-detection
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
bash scripts/run_all.sh                               # everything: data -> EDA -> train -> evaluate
```

Smoke test (2 epochs, 1 seed, ~10 min on CPU): `EPOCHS=2 SEEDS="0" bash scripts/run_all.sh`

CUHK Avenue (GPU recommended, e.g. Colab section 10): `bash scripts/run_avenue.sh`. Every script takes
`--config configs/avenue.yaml`; results go to `results/avenue/` so the Ped2 results stay untouched.

Step by step:

```bash
bash scripts/download_ped2.sh                    # ~460 MB
python scripts/prepare_data.py                   # verifies 16 train / 12 test videos, 2550 / 2010 frames
python scripts/eda.py                            # results/figures/eda_*.png
python scripts/train.py --model pca              # classical baseline
python scripts/train.py --model cae --seed 0     # CAE + motion channels
python scripts/train.py --model cae --no-motion  # ablation: appearance only
python scripts/train.py --model vit --seed 0     # masked ViT-AE
python scripts/evaluate.py                       # results/metrics/summary.json, results_table.md, figures
```

On Google Colab: open `notebooks/VAD_Colab.ipynb` (badge above, or upload it), switch to a T4 GPU and run
all cells. `train.py` uses CUDA automatically when a GPU is available.

## Rebuilding the Review 2 slides

```bash
VAD_THEME=dark python scripts/eda.py          # dark, slide-ready figures in results/figures/dark/
VAD_THEME=dark python scripts/evaluate.py
cd slides && npm install && node build_deck.js   # writes slides/Video_Anomaly_Detection_Second_Review.pptx
```

## Method (preliminary)

* **Input**: 5 consecutive grayscale frames at 128×192 + 4 amplified |frame-difference| motion channels.
* **CAE**: 3-level conv encoder → 64-channel bottleneck → symmetric decoder; MSE on normal clips.
* **Masked ViT-AE**: 16×16 patch tokens, 4 transformer layers, trained to inpaint 50 % randomly
  hidden patches; at test time two complementary checkerboard masks so every patch is predicted
  while hidden.
* **Score**: squared error map → max over 16×16 patches → Gaussian temporal smoothing (σ = 3).
* **Evaluation**: frame-level AUC-ROC, EER, PR-AUC; per-video normalised (literature protocol) and
  global (deployment-like); 95 % bootstrap CI over videos; Mann-Whitney U and Cohen's d;
  unsupervised threshold τ = 99th percentile of training scores.

## Datasets

| Dataset | Status | Notes |
|---|---|---|
| UCSD Ped2 (Mahadevan et al., 2010) | downloaded, verified | development benchmark, 16 train / 12 test videos |
| CUHK Avenue (Lu et al., 2013) | pipeline ready, run in Colab | same pipeline and hyper-parameters, 16 train / 21 test videos |
| ComplexVAD (Mumcu et al., 2025, MERL) | licence request | 39 GB, 34 h, 40 interaction anomaly types |
