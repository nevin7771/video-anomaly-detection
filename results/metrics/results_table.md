| Model | Score | Protocol | Seeds | AUC-ROC | EER | PR-AUC |
|---|---|---|---|---|---|---|
| PCA baseline | mean | global | 1 | 0.762 ± 0.000 | 0.375 | 0.938 |
| PCA baseline | mean | per-video | 1 | 0.877 ± 0.000 | 0.193 | 0.964 |
| PCA baseline | patch_max | global | 1 | 0.868 ± 0.000 | 0.193 | 0.970 |
| PCA baseline | patch_max | per-video | 1 | 0.862 ± 0.000 | 0.145 | 0.955 |
| CAE (appearance only) | mean | global | 1 | 0.642 ± 0.000 | 0.488 | 0.891 |
| CAE (appearance only) | mean | per-video | 1 | 0.837 ± 0.000 | 0.257 | 0.964 |
| CAE (appearance only) | patch_max | global | 1 | 0.769 ± 0.000 | 0.294 | 0.936 |
| CAE (appearance only) | patch_max | per-video | 1 | 0.702 ± 0.000 | 0.360 | 0.898 |
| CAE + motion | mean | global | 3 | 0.708 ± 0.003 | 0.447 | 0.920 |
| CAE + motion | mean | per-video | 3 | 0.878 ± 0.003 | 0.185 | 0.968 |
| CAE + motion | patch_max | global | 3 | 0.876 ± 0.002 | 0.200 | 0.973 |
| CAE + motion | patch_max | per-video | 3 | 0.871 ± 0.003 | 0.187 | 0.970 |
| Masked ViT-AE + motion | mean | global | 3 | 0.730 ± 0.000 | 0.434 | 0.925 |
| Masked ViT-AE + motion | mean | per-video | 3 | 0.891 ± 0.003 | 0.175 | 0.973 |
| Masked ViT-AE + motion | patch_max | global | 3 | 0.882 ± 0.000 | 0.215 | 0.973 |
| Masked ViT-AE + motion | patch_max | per-video | 3 | 0.844 ± 0.001 | 0.209 | 0.961 |
