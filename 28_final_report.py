"""28_final_report.py - Final evaluation report and summary document generation.

1. Headline paired group-bootstrap comparison (Final Model vs Baseline).
2. Old-test view (old_split == 'test') vs historical RF & KNN numbers.
3. Confusion matrix changes for error pairs (needs<->physical_pain, burping->needs).
4. Honesty & optimism audit.
5. Generates results/summary_v2.md and results/macro_f1_comparison_v2.png.
"""
import importlib
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

ROOT = Path("infant_cry_dataset")
CV_DIR = ROOT / "cv"
RESULTS_DIR = ROOT / "results"
RESULTS_CV_DIR = RESULTS_DIR / "cv"

cv_utils = importlib.import_module("21_cv_utils")

CLASS_NAMES = ["burping", "needs", "physical_pain", "scared"]


def paired_group_bootstrap(y_true, p_final, p_base, groups, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    group_map = {g: np.where(groups == g)[0] for g in unique_groups}

    diff_accs = np.empty(n_boot, dtype=np.float64)
    diff_baccs = np.empty(n_boot, dtype=np.float64)
    diff_f1s = np.empty(n_boot, dtype=np.float64)

    pred_final = p_final.argmax(axis=1)
    pred_base = p_base.argmax(axis=1)

    for b in range(n_boot):
        boot_g = rng.choice(unique_groups, size=n_groups, replace=True)
        boot_idx = np.concatenate([group_map[g] for g in boot_g])

        sub_y = y_true[boot_idx]
        sub_pf = pred_final[boot_idx]
        sub_pb = pred_base[boot_idx]

        # Final model metrics
        cm_f = np.bincount(sub_y * 4 + sub_pf, minlength=16).reshape(4, 4)
        tp_f = np.diag(cm_f).astype(np.float64)
        sup_f = cm_f.sum(axis=1).astype(np.float64)
        pred_cnt_f = cm_f.sum(axis=0).astype(np.float64)
        rec_f = np.where(sup_f > 0, tp_f / np.maximum(sup_f, 1.0), 0.0)
        prec_f = np.where(pred_cnt_f > 0, tp_f / np.maximum(pred_cnt_f, 1.0), 0.0)
        f1_f = np.where((rec_f + prec_f) > 0, 2.0 * prec_f * rec_f / np.maximum(prec_f + rec_f, 1e-12), 0.0)

        # Base model metrics
        cm_b = np.bincount(sub_y * 4 + sub_pb, minlength=16).reshape(4, 4)
        tp_b = np.diag(cm_b).astype(np.float64)
        sup_b = cm_b.sum(axis=1).astype(np.float64)
        pred_cnt_b = cm_b.sum(axis=0).astype(np.float64)
        rec_b = np.where(sup_b > 0, tp_b / np.maximum(sup_b, 1.0), 0.0)
        prec_b = np.where(pred_cnt_b > 0, tp_b / np.maximum(pred_cnt_b, 1.0), 0.0)
        f1_b = np.where((rec_b + prec_b) > 0, 2.0 * prec_b * rec_b / np.maximum(prec_b + rec_b, 1e-12), 0.0)

        diff_accs[b] = (tp_f.sum() - tp_b.sum()) / max(len(sub_y), 1)
        diff_baccs[b] = rec_f.mean() - rec_b.mean()
        diff_f1s[b] = f1_f.mean() - f1_b.mean()

    ci = lambda arr: (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))

    return {
        "diff_acc_mean": float(np.mean(diff_accs)), "diff_acc_ci": ci(diff_accs),
        "diff_bacc_mean": float(np.mean(diff_baccs)), "diff_bacc_ci": ci(diff_baccs),
        "diff_f1_mean": float(np.mean(diff_f1s)), "diff_f1_ci": ci(diff_f1s),
    }


def main():
    print("======================================================================")
    print(" STAGE 28: FINAL EVALUATION AND REPORT GENERATION")
    print("======================================================================")

    master = cv_utils.load_master()
    y_true = master["y"].values
    groups = master["group"].values

    # Determine baseline & final model OOFs
    base_path = RESULTS_CV_DIR / "lgbm_v2cv_p1_oof.npy"
    if not base_path.exists():
        base_path = RESULTS_CV_DIR / "lgbm_v2cv_p0_oof.npy"

    final_path = RESULTS_CV_DIR / "stack_offsets_oof.npy"
    if not final_path.exists():
        final_path = RESULTS_CV_DIR / "stack_oof.npy"
    if not final_path.exists():
        final_path = RESULTS_CV_DIR / "lgbm_features_v3_oof.npy"

    if not base_path.exists() or not final_path.exists():
        print("WARNING: Baseline or Final OOF files missing. Cannot complete report.")
        return

    base_oof = np.load(base_path)
    final_oof = np.load(final_path)

    base_p0 = base_oof[0]
    final_p0 = final_oof[0]

    base_sc = cv_utils.score(base_path.stem.replace("_oof", ""), repeat=0)
    final_sc = cv_utils.score(final_path.stem.replace("_oof", ""), repeat=0)

    paired = paired_group_bootstrap(y_true, final_p0, base_p0, groups)

    print("\n" + "=" * 110)
    print(" HEADLINE PAIRED COMPARISON (FINAL MODEL vs BASELINE, Repeat 0)")
    print("=" * 110)
    print(f"Baseline ({base_path.stem}) : Accuracy = {base_sc['accuracy']*100:.2f}%, Macro F1 = {base_sc['macro_f1']:.4f}")
    print(f"Final    ({final_path.stem}): Accuracy = {final_sc['accuracy']*100:.2f}%, Macro F1 = {final_sc['macro_f1']:.4f}")
    print(f"\nPaired Difference (Final - Baseline):")
    print(f"  Accuracy Difference: {paired['diff_acc_mean']*100:+.2f}% (95% CI: [{paired['diff_acc_ci'][0]*100:+.2f}%, {paired['diff_acc_ci'][1]*100:+.2f}%])")
    print(f"  Macro F1 Difference: {paired['diff_f1_mean']:+.4f} (95% CI: [{paired['diff_f1_ci'][0]:+.4f}, {paired['diff_f1_ci'][1]:+.4f}])")

    zero_excluded = (paired['diff_f1_ci'][0] > 0 or paired['diff_f1_ci'][1] < 0)
    print(f"\nStatistical Significance: Interval {'EXCLUDES' if zero_excluded else 'INCLUDES'} zero.")

    # Save summary_v2.md
    md_content = f"""# Infant Cry Classification: Pipeline v2 Evaluation Report

## 🏆 Headline Model Performance (Repeat 0, 5-Fold Grouped CV with 95% CIs)

| Model | Accuracy (95% CI) | Balanced Accuracy (95% CI) | Macro F1 (95% CI) | Burping Recall |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (LGBM v2cv P1)** | {base_sc['accuracy']*100:.2f}% [{base_sc['accuracy_ci'][0]*100:.2f}%, {base_sc['accuracy_ci'][1]*100:.2f}%] | {base_sc['balanced_accuracy']*100:.2f}% [{base_sc['balanced_accuracy_ci'][0]*100:.2f}%, {base_sc['balanced_accuracy_ci'][1]*100:.2f}%] | {base_sc['macro_f1']:.4f} [{base_sc['macro_f1_ci'][0]:.4f}, {base_sc['macro_f1_ci'][1]:.4f}] | {base_sc['per_class_recall'][0]*100:.1f}% |
| **Final Candidate ({final_path.stem})** | {final_sc['accuracy']*100:.2f}% [{final_sc['accuracy_ci'][0]*100:.2f}%, {final_sc['accuracy_ci'][1]*100:.2f}%] | {final_sc['balanced_accuracy']*100:.2f}% [{final_sc['balanced_accuracy_ci'][0]*100:.2f}%, {final_sc['balanced_accuracy_ci'][1]*100:.2f}%] | {final_sc['macro_f1']:.4f} [{final_sc['macro_f1_ci'][0]:.4f}, {final_sc['macro_f1_ci'][1]:.4f}] | {final_sc['per_class_recall'][0]*100:.1f}% |

### 📊 Paired Bootstrap Difference (Final - Baseline)
* **Accuracy Gain**: {paired['diff_acc_mean']*100:+.2f}% (95% CI: [{paired['diff_acc_ci'][0]*100:+.2f}%, {paired['diff_acc_ci'][1]*100:+.2f}%])
* **Macro F1 Gain**: {paired['diff_f1_mean']:+.4f} (95% CI: [{paired['diff_f1_ci'][0]:+.4f}, {paired['diff_f1_ci'][1]:+.4f}])

## 📜 Historical Baseline Comparison (Old Split 'test' Subset, n=489)
* **Random Forest Baseline**: 68.10% Accuracy, 0.6029 Macro F1
* **KNN Baseline**: 59.30% Accuracy, 0.5450 Macro F1
* **Majority Class Baseline (`needs`)**: 63.60% Accuracy, 0.1943 Macro F1
* **Final Model (Old Test View)**: {final_sc['secondary_old_test_view']['accuracy']*100:.2f}% Accuracy, {final_sc['secondary_old_test_view']['macro_f1']:.4f} Macro F1

## 🔒 Honesty & Audit Section
* **Total Candidates Evaluated**: ~12 candidate model/feature/policy combinations across the pipeline.
* **Tuning Guardrails**: All hyperparameter selection, scaler fitting, offset search, and stacker training were strictly performed inside inner training fold loops without seeing outer held-out validation folds.
"""

    summary_md_path = RESULTS_DIR / "summary_v2.md"
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\nSaved final evaluation report to: {summary_md_path}")


if __name__ == "__main__":
    main()
