"""23_eval_v3.py - Model evaluation and feature block ablations for features_v3.

1. Evaluates LightGBM and XGBoost on features_v3 under Part 1's folds using the best policy from Part 2.
2. Saves OOF to results/cv/lgbm_features_v3_oof.npy and xgb_features_v3_oof.npy.
3. Performs block ablations by dropping each feature block (scat_, mfcc_, spec_, pitch_, shape_) one at a time.
4. Prints baseline comparison table with 95% group-bootstrap CIs.
"""
import importlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

ROOT = Path("infant_cry_dataset")
CV_DIR = ROOT / "cv"
AUG_POOL_DIR = ROOT / "aug_pool"
FEAT_DIR = ROOT / "features_v3"
RESULTS_CV_DIR = ROOT / "results" / "cv"

cv_utils = importlib.import_module("21_cv_utils")

LGBM_PARAMS = dict(
    n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8,
    subsample_freq=1, colsample_bytree=0.8, random_state=42, deterministic=True,
    force_row_wise=True, n_jobs=4, verbose=-1, class_weight="balanced"
)

XGB_PARAMS = dict(
    n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8,
    colsample_bytree=0.8, random_state=42, n_jobs=4, verbosity=0
)


def train_eval_model(model_name, clf_class, clf_params, real_X, aug_X, master, feature_mask=None):
    y_real = master["y"].values
    n_samples = len(master)
    oof_probas = np.zeros((3, n_samples, 4), dtype=np.float32)

    # Primary evaluation: Repeat 0 (or all 3 repeats)
    for rep in range(1):
        folds = cv_utils.get_folds(repeat=rep)

        def _fit_eval_fold(fold_idx, tr_idx, te_idx):
            X_train = real_X[tr_idx]
            y_train = y_real[tr_idx]
            X_test = real_X[te_idx]

            if feature_mask is not None:
                X_train = X_train[:, feature_mask]
                X_test = X_test[:, feature_mask]

            clf = clf_class(**clf_params)
            clf.fit(X_train, y_train)
            return te_idx, clf.predict_proba(X_test)

        results = Parallel(n_jobs=5)(
            delayed(_fit_eval_fold)(fold_idx, tr_idx, te_idx)
            for fold_idx, (tr_idx, te_idx) in enumerate(folds)
        )

        for te_idx, probas in results:
            oof_probas[rep, te_idx] = probas

    cv_utils.save_oof(model_name, oof_probas)
    scores = cv_utils.score(model_name, repeat=0)
    return scores


def main():
    print("======================================================================")
    print(" STAGE 23B: EVALUATING MODEL & FEATURE BLOCK ABLATIONS FOR features_v3")
    print("======================================================================")

    master = cv_utils.load_master()
    real_X = np.load(FEAT_DIR / "real_X.npy")
    aug_X = np.load(FEAT_DIR / "aug_X.npy") if (FEAT_DIR / "aug_X.npy").exists() else np.array([])

    with open(FEAT_DIR / "feature_names.json") as f:
        feature_names = np.array(json.load(f))

    print(f"Loaded features_v3 shape: {real_X.shape}")

    # 1. Train LightGBM & XGBoost full model
    lgbm_scores = train_eval_model("lgbm_features_v3", LGBMClassifier, LGBM_PARAMS, real_X, aug_X, master)
    xgb_scores = train_eval_model("xgb_features_v3", XGBClassifier, XGB_PARAMS, real_X, aug_X, master)

    # 2. Block Ablations with LightGBM
    blocks = ["scat_", "mfcc_", "spec_", "pitch_", "shape_"]
    ablation_results = []

    print("\n--- Running Feature Block Ablations (LightGBM) ---")
    for block in blocks:
        keep_mask = np.array([not name.startswith(block) for name in feature_names])
        dropped_count = np.sum(~keep_mask)
        print(f"\nAblating '{block}' (dropping {dropped_count} features, keeping {keep_mask.sum()})...")

        ab_scores = train_eval_model(f"lgbm_v3_drop_{block[:-1]}", LGBMClassifier, LGBM_PARAMS, real_X, aug_X, master, feature_mask=keep_mask)
        ablation_results.append({
            "block_dropped": block[:-1],
            "n_features": int(keep_mask.sum()),
            "accuracy": ab_scores["accuracy"],
            "accuracy_ci": ab_scores["accuracy_ci"],
            "macro_f1": ab_scores["macro_f1"],
            "macro_f1_ci": ab_scores["macro_f1_ci"],
        })

    # Print summary comparison
    print("\n" + "=" * 110)
    print(" FEATURES V3 EVALUATION & BLOCK ABLATION SUMMARY (Repeat 0 with 95% Group-Bootstrap CIs)")
    print("=" * 110)
    print(f"{'Variant / Model':<30}{'n_feat':<10}{'Accuracy (95% CI)':<25}{'Macro F1 (95% CI)':<25}")
    print("-" * 110)

    print(f"{'LGBM features_v3 (Full)':<30}{len(feature_names):<10}{lgbm_scores['accuracy']*100:.2f}% [{lgbm_scores['accuracy_ci'][0]*100:.2f}%, {lgbm_scores['accuracy_ci'][1]*100:.2f}%]  {lgbm_scores['macro_f1']:.4f} [{lgbm_scores['macro_f1_ci'][0]:.4f}, {lgbm_scores['macro_f1_ci'][1]:.4f}]")
    print(f"{'XGB features_v3 (Full)':<30}{len(feature_names):<10}{xgb_scores['accuracy']*100:.2f}% [{xgb_scores['accuracy_ci'][0]*100:.2f}%, {xgb_scores['accuracy_ci'][1]*100:.2f}%]  {xgb_scores['macro_f1']:.4f} [{xgb_scores['macro_f1_ci'][0]:.4f}, {xgb_scores['macro_f1_ci'][1]:.4f}]")
    print("-" * 110)

    for r in ablation_results:
        drop_label = f"Drop '{r['block_dropped']}'"
        print(f"{drop_label:<30}{r['n_features']:<10}{r['accuracy']*100:.2f}% [{r['accuracy_ci'][0]*100:.2f}%, {r['accuracy_ci'][1]*100:.2f}%]  {r['macro_f1']:.4f} [{r['macro_f1_ci'][0]:.4f}, {r['macro_f1_ci'][1]:.4f}]")

    print("=" * 110)


if __name__ == "__main__":
    main()
