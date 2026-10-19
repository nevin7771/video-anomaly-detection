# How to start and run the project

## 1. Put the code on GitHub (one time)

1. On github.com, create an **empty** repository named `video-anomaly-detection`
   (no README, no .gitignore, no licence). Private is fine.
2. Unzip `video-anomaly-detection.zip` and push it:

```bash
cd video-anomaly-detection
git remote add origin https://github.com/nevin7771/video-anomaly-detection.git
git push -u origin main
```

(The zip already contains a git history with one commit, so `git push` is all you need.)

## 2. Set up your computer (one time)

Needs Python 3.10+ and git. On Windows use Git Bash or WSL for the `.sh` scripts.

```bash
cd video-anomaly-detection
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 3. Run everything

```bash
bash scripts/run_all.sh
```

This downloads UCSD Ped2 (~460 MB), checks it, makes the EDA figures, trains PCA, the CAE
(with and without motion) and the masked ViT-AE over 3 seeds, and writes:

* `results/metrics/results_table.md` – the AUC / EER / PR-AUC table
* `results/metrics/summary.json` – every number used in the slides
* `results/figures/*.png` – all figures

Time: about 2 h on a 4-core laptop CPU, about 10–15 min on a Colab T4 GPU.
Quick check that everything works (about 10 min): `EPOCHS=2 SEEDS="0" bash scripts/run_all.sh`

## 4. Run it on Google Colab instead (free GPU)

New notebook → `Runtime → Change runtime type → T4 GPU`, then in cells:

```python
!git clone https://github.com/nevin7771/video-anomaly-detection.git
%cd video-anomaly-detection
!pip install -q -r requirements.txt
!bash scripts/run_all.sh
!cat results/metrics/results_table.md
```

## 5. Run one piece at a time

```bash
bash scripts/download_ped2.sh                    # data
python scripts/prepare_data.py                   # integrity check
python scripts/eda.py                            # EDA figures
python scripts/train.py --model cae --seed 0     # one model (cae | vit | pca), add --no-motion for the ablation
python scripts/evaluate.py                       # tables + figures from whatever has been trained
```

Change image size, clip length, epochs, etc. in `configs/ped2.yaml`.

## 6. Rebuild the slides after new results

```bash
VAD_THEME=dark python scripts/eda.py
VAD_THEME=dark python scripts/evaluate.py
cd slides && npm install && node build_deck.js
```

Needs Node.js 18+. Output: `slides/Video_Anomaly_Detection_Second_Review.pptx`.
