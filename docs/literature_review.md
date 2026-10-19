# Literature review and theoretical background

## 1. Conceptual framework

Video anomaly detection (VAD) is framed here as **one-class (unsupervised) novelty detection**:
a model is fitted only to normal footage, learns the distribution *p(x | normal)* of
spatio-temporal patterns, and scores a test clip by how poorly that distribution explains it
(Chandola et al., 2009; Pang et al., 2021). In reconstruction-based VAD the score is the
reconstruction error ‖x − x̂‖²; in prediction-based VAD it is the error of predicting the next frame.

```
Normal-only video ──► Representation learner (CAE / ViT-AE) ──► Model of normality
                                                                   │
Test video ──► same learner ──► reconstruction / prediction error ─┴► anomaly score s(t)
                                                                        │
                                       temporal smoothing + threshold τ ┴► frame decision + heat map
```

Three design questions organise the literature and this project:

| Question | Options in the literature | Choice in this project |
|---|---|---|
| What signal is modelled? | appearance, motion (optical flow / frame difference), objects, scene graphs | appearance + frame-difference motion channels |
| What learner? | hand-crafted + sparse coding, CAE, ConvLSTM-AE, GAN/prediction, memory-AE, transformers / MAE | CAE (baseline) vs masked ViT-AE |
| How is error turned into a decision? | global threshold, per-video normalisation, patch-level max, object-level | patch-max score, temporal smoothing, τ from training-score percentile |

## 2. Review of approaches

| # | Study | Venue | Family | Key idea | Ped2 AUC | Avenue AUC |
|---|---|---|---|---|---|---|
| 1 | Mahadevan et al. 2010 | CVPR | Classical | Mixtures of dynamic textures; introduced UCSD Ped1/Ped2 | — | — |
| 2 | Lu, Shi & Jia 2013 | ICCV | Classical | Sparse combination learning at 150 FPS; introduced CUHK Avenue | — | — |
| 3 | Hasan et al. 2016 | CVPR | Reconstruction | Fully convolutional AE on stacked frames; "regularity score" | 90.0 | 70.2 |
| 4 | Luo, Liu & Gao 2017 | ICME | Reconstruction | ConvLSTM-AE remembers temporal history | 88.1 | 77.0 |
| 5 | Luo, Liu & Gao 2017 | ICCV | Sparse coding | Stacked-RNN sparse coding; introduced ShanghaiTech | 92.2 | 81.7 |
| 6 | Sabokrou et al. 2017 | IEEE TIP | Cascade CNN | Deep-Cascade: cascaded 3D networks for fast localisation | — | — |
| 7 | Sabokrou et al. 2018 | CVIU | FCN | Deep-anomaly: fully convolutional feature description | — | — |
| 8 | Liu et al. 2018 | CVPR | Prediction | Future-frame prediction with flow + adversarial constraints | 95.4 | 85.1 |
| 9 | Sultani, Chen & Shah 2018 | CVPR | Weakly supervised | Multiple-instance ranking; introduced UCF-Crime | — | — |
| 10 | Gong et al. 2019 | ICCV | Memory AE | MemAE: memory module stops AE reconstructing anomalies | 94.1 | 83.3 |
| 11 | Park, Noh & Ham 2020 | CVPR | Memory AE | MNAD: memory of normal prototypes, recon + prediction | 97.0 | 88.5 |
| 12 | Liu et al. 2021 | ICCV | Hybrid | HF²-VAD: memory-augmented flow reconstruction + flow-guided prediction | 99.3 | 91.1 |
| 13 | Georgescu et al. 2021 | CVPR | Self-supervised | Multi-task proxy tasks on object tracks | 99.8 | 91.5 |
| 14 | Dosovitskiy et al. 2021 | ICLR | Transformer | ViT: images as 16×16 patch tokens | — | — |
| 15 | He et al. 2022 | CVPR | Transformer | MAE: masked patch reconstruction as self-supervision | — | — |
| 16 | Ristea et al. 2024 | CVPR | Transformer | Self-distilled masked AE for efficient VAD | 95.4 | 91.3 |
| 17 | Wu et al. 2024 | AAAI | Vision-language | VadCLIP: CLIP adapted for weakly supervised VAD | — | — |
| 18 | Mumcu et al. 2025 | WACV-W | Benchmark | ComplexVAD: 34 h, 40 interaction-anomaly types; best frame AUC ≈ 0.60 | — | — |

Survey / review sources: Chandola et al. 2009 (ACM CSUR), Kiran et al. 2018 (J. Imaging),
Pang et al. 2021 (ACM CSUR), Ramachandra et al. 2022 (IEEE TPAMI), Liu et al. 2024 (ACM CSUR).

AUC values are those reported by the original authors (frame-level, micro or per-video protocol as
in each paper); they are for orientation, not a like-for-like comparison. Values for rows 4, 5, 13
and 16 should be double-checked against the papers before the final thesis.

## 3. Gaps that motivate this project

1. **Reconstruction AEs generalise too well**: a plain CAE can reconstruct anomalies (Gong et al. 2019).
   Memory modules and *masking* are two remedies; this project tests masking via a ViT-AE.
2. **CNN vs transformer under identical preprocessing** is rarely isolated: most papers change
   inputs, losses and datasets at once. This project holds the pipeline fixed and swaps only the learner.
3. **Small, local anomalies** (bikes, carts in Ped2 cover a median of only ~1.2 % of the frame) are
   diluted by frame-mean error, motivating patch-level max scoring and heat-map localisation.
4. **Complex, interaction anomalies** (ComplexVAD) remain unsolved: the best reported frame AUC is
   ~0.60, so a > 0.90 target is realistic for Ped2/Avenue but not for ComplexVAD.

## 4. References

1. Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly detection: A survey. *ACM Computing Surveys, 41*(3), 1–58.
2. Mahadevan, V., Li, W., Bhalodia, V., & Vasconcelos, N. (2010). Anomaly detection in crowded scenes. *CVPR*, 1975–1981.
3. Lu, C., Shi, J., & Jia, J. (2013). Abnormal event detection at 150 FPS in MATLAB. *ICCV*, 2720–2727.
4. Hasan, M., Choi, J., Neumann, J., Roy-Chowdhury, A. K., & Davis, L. S. (2016). Learning temporal regularity in video sequences. *CVPR*, 733–742.
5. Luo, W., Liu, W., & Gao, S. (2017). Remembering history with convolutional LSTM for anomaly detection. *IEEE ICME*, 439–444.
6. Luo, W., Liu, W., & Gao, S. (2017). A revisit of sparse coding based anomaly detection in stacked RNN framework. *ICCV*, 341–349.
7. Sabokrou, M., Fayyaz, M., Fathy, M., & Klette, R. (2017). Deep-Cascade: Cascading 3D deep neural networks for fast anomaly detection and localization in crowded scenes. *IEEE Transactions on Image Processing, 26*(4), 1992–2004.
8. Sabokrou, M., Fayyaz, M., Fathy, M., Moayed, Z., & Klette, R. (2018). Deep-anomaly: Fully convolutional neural network for fast anomaly detection in crowded scenes. *Computer Vision and Image Understanding, 172*, 88–97.
9. Liu, W., Luo, W., Lian, D., & Gao, S. (2018). Future frame prediction for anomaly detection – a new baseline. *CVPR*, 6536–6545.
10. Sultani, W., Chen, C., & Shah, M. (2018). Real-world anomaly detection in surveillance videos. *CVPR*, 6479–6488.
11. Kiran, B. R., Thomas, D. M., & Parakkal, R. (2018). An overview of deep learning based methods for unsupervised and semi-supervised anomaly detection in videos. *Journal of Imaging, 4*(2), 36.
12. Gong, D., Liu, L., Le, V., Saha, B., Mansour, M. R., Venkatesh, S., & van den Hengel, A. (2019). Memorizing normality to detect anomaly: Memory-augmented deep autoencoder for unsupervised anomaly detection. *ICCV*, 1705–1714.
13. Park, H., Noh, J., & Ham, B. (2020). Learning memory-guided normality for anomaly detection. *CVPR*, 14372–14381.
14. Pang, G., Shen, C., Cao, L., & van den Hengel, A. (2021). Deep learning for anomaly detection: A review. *ACM Computing Surveys, 54*(2), 1–38.
15. Liu, Z., Nie, Y., Long, C., Zhang, Q., & Li, G. (2021). A hybrid video anomaly detection framework via memory-augmented flow reconstruction and flow-guided frame prediction. *ICCV*, 13588–13597.
16. Georgescu, M.-I., Barbalau, A., Ionescu, R. T., Khan, F. S., Popescu, M., & Shah, M. (2021). Anomaly detection in video via self-supervised and multi-task learning. *CVPR*, 12742–12752.
17. Dosovitskiy, A., et al. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. *ICLR*.
18. He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). Masked autoencoders are scalable vision learners. *CVPR*, 16000–16009.
19. Ramachandra, B., Jones, M. J., & Vatsavai, R. R. (2022). A survey of single-scene video anomaly detection. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 44*(5), 2293–2312.
20. Ristea, N.-C., Croitoru, F.-A., Ionescu, R. T., Popescu, M., Khan, F. S., & Shah, M. (2024). Self-distilled masked auto-encoders are efficient video anomaly detectors. *CVPR*, 15984–15995.
21. Wu, P., Zhou, X., Pang, G., Zhou, L., Yan, Q., Wang, P., & Zhang, Y. (2024). VadCLIP: Adapting vision-language models for weakly supervised video anomaly detection. *AAAI, 38*(6), 6074–6082.
22. Liu, Y., Yang, D., Wang, Y., Liu, J., Liu, J., Boukerche, A., Sun, P., & Song, L. (2024). Generalized video anomaly event detection: Systematic taxonomy and comparison of deep models. *ACM Computing Surveys, 56*(7), 1–38.
23. Mumcu, F., Jones, M. J., Cherian, A., & Yilmaz, Y. (2025). ComplexVAD: Detecting interaction anomalies in video. *WACV Workshops* (ASTAD). arXiv:2501.09733.

Page ranges are from memory of the published versions; verify each against Scopus / Google
Scholar before submitting the thesis bibliography.
