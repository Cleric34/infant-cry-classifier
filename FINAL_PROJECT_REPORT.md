# Final Project Report: 4-Class Infant Cry Classification Pipeline

**Authors / Engineering Team**: Infant Cry AI Research & Development  
**Date**: October 2026 | **Evaluation Protocol Version**: 2.0 (Stratified Group Cross-Validation)

---

## Executive Summary

This report documents the design, implementation, and rigorous empirical validation of a modernized machine learning pipeline for 4-class acoustic infant cry classification (`burping`, `needs`, `physical_pain`, `scared`). 

Addressing severe class imbalance (63.7% `needs`, 6.4% `burping`) and historical data leakage pitfalls, we implemented an honest 5-fold Stratified Group Cross-Validation protocol across all 4,871 clean audio recordings, using agglomerative cosine clustering of deep audio embeddings to prevent session leakage.

Our final declared architecture—a **Logit-Adjusted Multinomial Logistic Stacker** combining a 4-way foundation model neural head (6,656-d embeddings from PANNs, AST, WavLM, and wav2vec2) with multi-domain handcrafted LightGBM boosting models—achieved **87.66% Accuracy** (95% CI: [86.57%, 88.75%]), **0.7993 Macro F1** (95% CI: [0.7812, 0.8162]), and **49.0% Burping Recall**. A 2,000-sample paired group-bootstrap hypothesis test confirmed that the accuracy gain (+3.57%) and macro-F1 gain (+0.0607) over baseline are **statistically significant ($p < 0.05$)**.

---

## 1. Problem Statement & Dataset Demographics

Infant crying serves as a primary biological communication mechanism. Accurately distinguishing between physiological states is critical for clinical monitoring and parental assistance:
- **Burping (Class 0)**: 310 recordings (6.36%) — Gastrointestinal distress / gas.
- **Needs (Class 1)**: 3,103 recordings (63.70%) — Generalized discomfort / hunger (Dominant majority).
- **Physical Pain (Class 2)**: 774 recordings (15.89%) — Somatosensory / acute nociceptive pain.
- **Scared (Class 3)**: 684 recordings (14.04%) — Startle reflex / sudden acoustic or environmental fear.
- **Total Corpus**: 4,871 real audio files sampled at 22,050 Hz, denoised via Dual-Tree Complex Wavelet Transform (DTCWT).

---

## 2. Methodology & Modernized Pipeline Architecture

```
                                  [ 4,871 Raw Audio Clips ]
                                             │
                                             ▼
                               [ DTCWT Wavelet Denoising ]
                                             │
                                             ▼
                      [ Stratified Group K-Fold (5 Folds, Repeat 0) ]
                                             │
                    ┌────────────────────────┴────────────────────────┐
                    ▼                                                 ▼
     [ Handcrafted Feature Stream ]                   [ Pre-Trained Foundation Models ]
  • Scattering Transform (J=6, Q=16)               • AST (Audio Spectrogram Transformer)
  • Spectral Rolloff/Bandwidth/Centroid            • WavLM Base+ (Microsoft, Layers 4-8)
  • F0 Pitch Jitter / Shimmer / HNR                • wav2vec2-base (Facebook, Layers 4-8)
  • Envelope Shape Statistics                      • PANNs CNN14 (AudioSet 2048-d)
                    │                                                 │
                    ▼                                                 ▼
        [ LightGBM Classifiers ]                        [ 4-Way Concatenated MLP Head ]
  (drop-MFCC, V2CV, V3 Multi-Domain)               (6656-d, GPU, Focal Loss γ=2, Mixup α=0.2)
                    │                                                 │
                    └────────────────────────┬────────────────────────┘
                                             ▼
                              [ Multinomial Logistic Stacker ]
                                             │
                                             ▼
                            [ Inner-Fold Logit Offset Search ]
                                             │
                                             ▼
                             [ Final Calibrated 4-Class Output ]
                               (87.66% Acc | 0.7993 Macro-F1)
```

---

## 3. Comprehensive Experimental Results

### 3.1 Headline Benchmark Leaderboard (5-Fold Stratified Group CV, n=4,871)

| Model Architecture | Accuracy (95% CI) | Balanced Acc (95% CI) | Macro F1 (95% CI) | Burping Recall | Pain Recall | Scared Recall |
|---|---|---|---|---|---|---|
| **Baseline (LGBM v2cv Policy P1)** | 84.09% [82.82, 85.31] | 70.17% [68.26, 72.06] | 0.7388 [0.7182, 0.7579] | 38.7% [32.9, 44.8] | 54.0% | 94.0% |
| **LightGBM Full V3 (695 feats)** | 85.36% [84.23, 86.49] | 68.02% [66.13, 69.96] | 0.7350 [0.7147, 0.7559] | 31.0% [25.2, 37.3] | 50.3% | 93.0% |
| **LightGBM V3 (Drop MFCC, 344 feats)** | 85.67% [84.50, 86.85] | 69.28% [67.35, 71.25] | 0.7475 [0.7267, 0.7674] | 35.2% [29.2, 41.2] | 52.2% | 92.1% |
| **Linear SVM on AST Embeddings** | 83.49% [82.23, 84.73] | 69.87% [67.92, 71.82] | 0.7201 [0.6992, 0.7414] | 33.1% [27.4, 39.4] | 54.6% | 94.3% |
| **MLP Concat Head (6656-d, CUDA)** | 84.19% [82.95, 85.37] | 78.00% [76.25, 79.73] | **0.7696 [0.7524, 0.7867]** | **55.5% [49.7, 61.4]** | **70.4%** | **98.8%** |
| **Soft Voting Top-2 Ensemble** | 86.49% [85.36, 87.62] | 71.53% [69.68, 73.48] | 0.7638 [0.7441, 0.7831] | 38.4% [32.3, 44.6] | 55.4% | 95.2% |
| **🏆 FINAL MODEL: Stacker + Offsets** | **87.66% [86.57, 88.75]** | **77.72% [75.76, 79.53]** | **0.7993 [0.7812, 0.8162]** | **49.0% [42.6, 55.1]** | **69.4%** | **98.8%** |

---

### 3.2 Paired Group-Bootstrap Difference (Final Model vs. Baseline)

- **Accuracy Improvement**: **+3.57%** (95% Confidence Interval: `[+2.65%, +4.48%]`)
- **Macro-F1 Improvement**: **+0.0607** (95% Confidence Interval: `[+0.0459, +0.0752]`)
- **Statistical Significance**: Both confidence intervals strictly exclude 0.0, validating genuine architectural improvement over random variation.

---

### 3.3 Historical Baseline Comparison (Old Split 489-File Test Set)

| Benchmark / Model | Test Accuracy | Test Macro F1 | Source / Protocol |
|---|---|---|---|
| **Majority Class Guess (`needs`)** | 63.60% | 0.1943 | Empirical baseline |
| **K-Nearest Neighbors (KNN)** | 59.30% | 0.5450 | Historical script 08 |
| **Random Forest (RF)** | 68.10% | 0.6029 | Historical script 07 |
| **Old LightGBM (features_v2)** | 87.32% | 0.8082 | Historical script 15 |
| **Modernized Stacker (Secondary View)** | **87.32%** | **0.7913** | Evaluated on same 489 holdout |
| **Modernized Stacker (Full 5-Fold CV)** | **87.66%** | **0.7993** | Leak-free across all 4,871 files |

---

## 4. Key Scientific & Engineering Findings

1. **True-Length Audio Extraction Eliminated Silence Contamination**:
   Zero-padding short audio clips to 3.98s previously diluted time-pooled feature representations with zero-energy silence. Extracting true-length audio boosted PANNs embedding accuracy by **+25.25%** ($53.69\% \to 78.94\%$).

2. **Scattering Transforms Render MFCCs Redundant**:
   Ablation studies confirmed that Scattering Wavelet coefficients capture time-frequency cry morphology with mathematical stability against time-warping. Dropping MFCCs reduced feature dimensionality from 695 to 344 while improving Macro F1 by **+0.0125** and raising Burping recall by **+4.2%**.

3. **AST Transformer is the Superior Foundation Model**:
   Google's Audio Spectrogram Transformer (AST) achieved 83.49% accuracy and 94.3% Scared recall, outperforming human speech models (WavLM, wav2vec2) due to its comprehensive patch-based attention across high-frequency acoustic transients.

4. **Focal Loss & Mixup Regularization on 6,656-d Embeddings**:
   Training a 3-layer MLP on the concatenated 4-embedding space with Focal Loss ($\gamma=2.0$) achieved **55.5% Burping recall** and **0.7696 Macro F1**, demonstrating that deep foundation models capture subtle acoustic nuances that shallow tree models miss.

---

## 5. File Manifest & Reproducibility Verification

All code and evaluation results are packaged in the repository:
- `21_build_cv.py` & `21_cv_utils.py`: Master dataset manifest (`master.csv`) and group-bootstrap scoring engine.
- `22_build_aug_pool.py`: Deterministic 2-copy augmentation pool (`aug_manifest.csv`).
- `23_features_v3.py` & `23_eval_v3.py`: 695-d multi-domain feature extractor and ablation benchmarks.
- `24_embeddings_v2.py` & `24_embeddings_cv.py`: GPU foundation model extractor (PANNs, AST, WavLM, wav2vec2).
- `25_neural_cv.py`: 4-way concatenated neural head with Focal Loss and Mixup.
- `26_boost_cv.py`: Boosters evaluation on handcrafted feature spaces.
- `27_stack_cv.py`: Meta-learning multinomial stacker with logit offset optimization.
- `28_final_report.py`: Paired hypothesis testing suite and automated markdown report generator.
