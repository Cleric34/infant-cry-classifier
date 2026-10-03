# INFANT CRY CLASSIFICATION PIPELINE: TECHNICAL HANDOFF & CONTEXT REPORT (AI-to-AI)

**Document Purpose**: This document is an exhaustive, mathematically precise technical specification and context state document. It is designed to allow another autonomous AI agent (e.g., Claude 3.7 / Claude 4.6 Sonnet) to immediately understand the exact mathematical foundations, data contracts, failure modes, experimental results, and code architecture of this repository without needing to re-explore or re-evaluate past experiments.

---

## 1. PROJECT SPECIFICATION & CONSTRAINTS

- **Domain**: 4-Class Acoustic Infant Cry Classification
- **Target Classes** (Zero-indexed alphabetical order):
  0. `burping` ($N = 310$, ~6.4%)
  1. `needs` ($N = 3,103$, ~63.7%) — dominant majority class
  2. `physical_pain` ($N = 774$, ~15.9%)
  3. `scared` ($N = 684$, ~14.0%)
  - **Total Real Dataset Size ($N$)**: 4,871 audio files at 22,050 Hz sampling rate.
- **Primary Optimization Goal**: Maximize Macro-Averaged F1-Score ($\text{Macro-F1}$) and Balanced Accuracy while simultaneously raising Overall Accuracy beyond 87.5% under strict leak-free evaluation.
- **Environment & Hardware**: Windows OS, Python 3.12, PyTorch 2.5.1 with CUDA 12.1 acceleration, NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM).

---

## 2. EVALUATION PROTOCOL CONTRACT (STRICT LEAK-FREE GROUP CV)

### 2.1 The Master Dataset Manifest (`infant_cry_dataset/cv/master.csv`)
- **Row count**: Exactly 4,871 rows.
- **Columns**: `key`, `path`, `y` (0-3), `class`, `group` (0-4870), `old_split` (`train`, `val`, `test`), `fold_r0`, `fold_r1`, `fold_r2`.
- **Clustering / Pseudo-Sessions**: Because original filenames lack metadata IDs, pseudo-sessions were constructed by running true-length audio through PANNs CNN14, $L_2$-normalizing embeddings, and running agglomerative cosine clustering with max group size $\le 0.31\%$ of $N$ to group near-duplicate recordings into unified clusters.
- **Fold Assignment**: 5-Fold `StratifiedGroupKFold` across repeats 0, 1, 2 (random seeds 42, 43, 44). **Primary benchmark is Repeat 0.** No group ever spans train and test partitions.

### 2.2 Vectorized Group-Bootstrap Confidence Intervals (`21_cv_utils.py`)
- Uncertainty is quantified via **2,000-sample group bootstrap** (resampling cluster groups with replacement).
- All reported metrics (Accuracy, Balanced Accuracy, Macro F1, Per-Class Recalls) must include $[2.5\%, 97.5\%]$ percentile intervals.
- Statistical significance between models is tested via **paired group bootstrap difference distributions** ($\Delta = \text{Metric}_{\text{model}} - \text{Metric}_{\text{baseline}}$); significance requires the 95% CI of $\Delta$ to completely exclude 0.0.

---

## 3. SUMMARY LEADERBOARD & COMPARATIVE RESULTS

All results below are evaluated on the exact 4,871 real files under 5-Fold Stratified Group CV (Repeat 0):

| Model / Architecture | Accuracy (95% CI) | Balanced Acc (95% CI) | Macro F1 (95% CI) | Burping Recall (95% CI) | Pain Recall | Scared Recall |
|---|---|---|---|---|---|---|
| **Baseline: LGBM v2cv (Policy P1)** | 84.09% [82.82, 85.31] | 70.17% [68.26, 72.06] | 0.7388 [0.7182, 0.7579] | 38.7% [32.9, 44.8] | 54.0% | 94.0% |
| **LGBM Full V3 (695 feats)** | 85.36% [84.23, 86.49] | 68.02% [66.13, 69.96] | 0.7350 [0.7147, 0.7559] | 31.0% [25.2, 37.3] | 50.3% | 93.0% |
| **LGBM V3 (Drop MFCC, 344 feats)** | 85.67% [84.50, 86.85] | 69.28% [67.35, 71.25] | 0.7475 [0.7267, 0.7674] | 35.2% [29.2, 41.2] | 52.2% | 92.1% |
| **SVM on AST Embeddings (1536-d)** | 83.49% [82.23, 84.73] | 69.87% [67.92, 71.82] | 0.7201 [0.6992, 0.7414] | 33.1% [27.4, 39.4] | 54.6% | 94.3% |
| **MLP Concat Head (6656-d, GPU)** | 84.19% [82.95, 85.37] | 78.00% [76.25, 79.73] | **0.7696 [0.7524, 0.7867]** | **55.5% [49.7, 61.4]** | **70.4%** | **98.8%** |
| **Soft Vote Top-2** | 86.49% [85.36, 87.62] | 71.53% [69.68, 73.48] | 0.7638 [0.7441, 0.7831] | 38.4% [32.3, 44.6] | 55.4% | 95.2% |
| **🏆 FINAL: Stacker + Logit Offsets** | **87.66% [86.57, 88.75]** | **77.72% [75.76, 79.53]** | **0.7993 [0.7812, 0.8162]** | **49.0% [42.6, 55.1]** | **69.4%** | **98.8%** |

### Paired Hypothesis Testing (Final Model vs. Baseline)
- **$\Delta \text{Accuracy}$**: **+3.57%** (95% CI: `[+2.65%, +4.48%]`) $\implies$ **Statistically Significant ($p < 0.05$)**.
- **$\Delta \text{Macro F1}$**: **+0.0607** (95% CI: `[+0.0459, +0.0752]`) $\implies$ **Statistically Significant ($p < 0.05$)**.
- **Secondary View (Old 489-file Test Holdout)**: **87.32% Accuracy, 0.7913 Macro-F1** (vs. historical 59.30% KNN and 68.10% Random Forest baselines).

---

## 4. ARCHITECTURAL BREAKDOWN (SCRIPTS 01–28)

### Era 1: Preprocessing & Historical Foundations (`01_` to `20_`)
1. `01_dtcwt_denoise.py`: Dual-Tree Complex Wavelet Transform denoising. Cleans raw wavs into `infant_cry_dataset/audio_denoised/`.
2. `02_split_classwise.py`: Historical 80/10/10 split generation (identified as prone to group leakage; kept for reference only).
3. `03_controlled_augmentation.py`: Historical augmentation script (superseded by `22_build_aug_pool.py`).
4. `04_augmentation_visualization.py`: Generates augmentation validation spectrogram plots in `augmentation_plots/`.
5. `05_graph_scattering_features.py`: Computes wavelets & graph scattering transform coefficients.
6. `06_statistical_feature_testing.py`: ANOVA / Kruskal-Wallis feature significance ranking.
7. `07_train_random_forest.py` & `08_train_knn.py`: Historical baseline classifiers (68.1% and 59.3% acc).
8. `09_compare_rf_knn.py`: Baseline comparative metric printer.
9. `10_single_file_trace.py`: End-to-end debugging trace on a single audio clip.
10. `11_kymatio_mfcc_features.py`: Historical feature extractor (identified with fixed 3.98s zero-padding flaw).
11. `12_eval_utils.py`: Evaluation helper module for Era 1 scripts.
12. `13_pitch_features.py`: Fundamental frequency (F0), jitter, shimmer, HNR extraction.
13. `14_embeddings.py`: Era 1 foundation model extraction (last-layer only).
14. `15_train_boosting.py`, `16_train_cnn.py`, `17_ablations.py`, `18_ensemble_shap.py`, `19_final_report.py`, `20_diagnostics.py`: Historical tuning and diagnostic scripts.

### Era 2: Honest Cross-Validation & Modernized Pipeline (`21_` to `28_`)
1. `21_build_cv.py` & `21_cv_utils.py`: Constructs `master.csv` (4,871 files), agglomerative clustering pseudo-sessions, StratifiedGroupKFold splits, vectorized group-bootstrap scoring engine.
2. `22_build_aug_pool.py`: Generates deterministic 2-copies-per-clip pool (`aug_pool/`) for non-Needs classes ($N=3,536$ files) with `aug_manifest.csv`.
3. `22_extract_v2cv.py` & `22_aug_ablation.py`: Extracts true-length `features_v2cv/real_X.npy` (4,098-d) and evaluates Policies P0–P4 (P1 Real+Balanced established as winning policy).
4. `23_features_v3.py` & `23_eval_v3.py`: Generates multi-domain 695-d feature space (`scat_`, `mfcc_`, `spec_`, `pitch_`, `shape_`). Discovered that dropping `mfcc_` boosts Macro F1 (+0.0125) and Burping recall (+4.2%).
5. `24_embeddings_v2.py` & `24_embeddings_cv.py`: Extracts true-length foundation model embeddings:
   - `features_panns_v2`: PANNs CNN14 (2,048-d, 32 kHz) $\to$ 78.94% Acc (+25.25% gain over padded version).
   - `features_ast`: MIT Audio Spectrogram Transformer (1,536-d, 16 kHz) $\to$ **83.49% Acc / 0.7201 F1 (Best Foundation Model)**.
   - `features_wavlm_mid`: Microsoft WavLM Base+ layers 4–8 mean+std (1,536-d, 16 kHz) $\to$ 81.79% Acc.
   - `features_w2v_mid`: Facebook wav2vec2-base layers 4–8 mean+std (1,536-d, 16 kHz) $\to$ 82.08% Acc.
6. `25_neural_cv.py`: 4-Way Multimodal MLP Head trained on concatenated 6,656-d embeddings on GPU with Focal Loss ($\gamma=2.0$) and Mixup ($\alpha=0.2$) across 5 seeds $\times$ 5 folds = 25 neural runs. Achieved **0.7696 Macro F1** and **55.5% Burping Recall** (Best Single Model).
7. `26_boost_cv.py`: Outer CV booster evaluation on `features_v2cv` and `features_v3`.
8. `27_stack_cv.py`: Multinomial Logistic Regression stacker on outer fold log-probabilities combining top-4 diverse models (`mlp_concat_head`, `lgbm_v3_drop_mfcc`, `lgbm_v2cv_default`, `lgbm_v3_default`) with grid-searched logit offsets.
9. `28_final_report.py`: Runs paired group-bootstrap hypothesis testing and generates `results/summary_v2.md`.

---

## 5. CRITICAL SCIENTIFIC INSIGHTS

1. **Silence Padding Dilution Effect**:
   - Zero-padding variable-length baby cries to a fixed 3.98s duration corrupted time-pooled summary statistics (mean, variance, energy) by mixing genuine acoustic crying with zero-energy silence.
   - Removing silence padding and extracting true-length features lifted PANNs accuracy by **+25.25%** ($53.69\% \to 78.94\%$).
2. **Scattering Transform vs. MFCC Redundancy**:
   - Scattering coefficients ($J=6, Q=16$) provide deformation-stable multi-scale representations of cry morphology that strictly outperform MFCCs.
   - Ablation confirmed that dropping MFCCs while keeping Scattering + Spectral + Pitch + Shape improved LightGBM Macro F1 from 0.7350 to 0.7475 and raised Burping recall from 31.0% to 35.2%.
3. **AST Transformer Superiority**:
   - Despite being pre-trained on generic environmental audio (AudioSet), AST significantly outperformed speech-specific models (wav2vec2, WavLM) on infant cry categorization ($83.49\%$ Acc vs $82.08\%$ / $81.79\%$).
4. **Focal Loss + Multimodal Concatenation on Imbalanced Minorities**:
   - Direct concatenation of the 4 foundation model embeddings (6,656 features) paired with Focal Loss ($\gamma=2$) in an MLP head achieved **55.5% Burping recall**—an absolute improvement of +16.8% over the best single tree model.
5. **Stacking with Logit Offsets**:
   - Logistic regression meta-learning on out-of-fold log-probabilities combined with training-fold-only logit offset tuning yielded the winning performance: **87.66% Accuracy, 0.7993 Macro F1, 49.0% Burping Recall, 69.4% Pain Recall, and 98.8% Scared Recall**.

---

## 6. INSTRUCTIONS FOR CONTINUED WORK (FOR FUTURE AI SESSIONS)

- **To run the full evaluation suite**: `python 28_final_report.py`
- **To re-evaluate stacking with new candidate models**: Add candidate OOF array `results/cv/<new_model>_oof.npy` (must be shape `(3, 4871, 4)` float32) and run `python 27_stack_cv.py`.
- **To test additional neural architectures**: Use `25_neural_cv.py` as template; ensure `21_cv_utils.get_folds(repeat=0)` is strictly used with inner-only validation.
- **Never modify**: `master.csv`, `audio_denoised/`, or existing numbered scripts `01_` through `28_`. All extensions must be additive (`30_*.py`, etc.).
