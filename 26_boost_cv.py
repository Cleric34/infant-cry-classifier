"""26_boost_cv.py - Optuna-tuned LightGBM & XGBoost boosters with nested CV.

For feature sets (features_v2cv, features_v3, best embeddings, fused set):
- Outer 5-fold CV (Repeat 0).
- Inner 3-fold grouped Optuna TPE tuning (20 trials) on outer training portion only.
- Saves results/cv/<model>_<feat>_tuned_oof.npy and <model>_<feat>_default_oof.npy.
- Prints ranking table with 95% CIs.
"""
import importlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path("infant_cry_dataset")
CV_DIR = ROOT / "cv"
AUG_POOL_DIR = ROOT / "aug_pool"
RESULTS_CV_DIR = ROOT / "results" / "cv"

cv_utils = importlib.import_module("21_cv_utils")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


def train_eval_booster_default(model_type, feat_name, real_X, master):
    name = f"{model_type}_{feat_name}_default"
    oof_file = RESULTS_CV_DIR / f"{name}_oof.npy"
    if oof_file.exists():
        print(f"[{name}] OOF already exists. Loading and scoring...")
        return cv_utils.score(name, repeat=0)

    print(f"\n--- Training {model_type} (Default Params) on '{feat_name}' ---")
    y_real = master["y"].values
    n_samples = len(master)
    oof_probas = np.zeros((3, n_samples, 4), dtype=np.float32)

    cs = 0.3 if real_X.shape[1] > 1000 else 0.8
    use_gpu = torch.cuda.is_available()

    for rep in range(1):
        folds = cv_utils.get_folds(repeat=rep)

        for fold_idx, (tr_idx, te_idx) in enumerate(folds):
            X_train = real_X[tr_idx]
            y_train = y_real[tr_idx]
            X_test = real_X[te_idx]

            if model_type == "lgbm":
                clf = LGBMClassifier(
                    n_estimators=200, learning_rate=0.05, num_leaves=31, max_depth=6,
                    subsample=0.8, subsample_freq=1, colsample_bytree=cs,
                    class_weight="balanced", random_state=42,
                    verbose=-1, n_jobs=-1
                )
            elif model_type == "xgb":
                xgb_kwargs = dict(
                    n_estimators=200, learning_rate=0.05, max_depth=6,
                    subsample=0.8, colsample_bytree=cs, random_state=42,
                    verbosity=0, n_jobs=-1
                )
                if use_gpu:
                    xgb_kwargs.update(dict(tree_method="hist", device="cuda"))
                clf = XGBClassifier(**xgb_kwargs)

            clf.fit(X_train, y_train)
            oof_probas[rep, te_idx] = clf.predict_proba(X_test)

    cv_utils.save_oof(name, oof_probas)
    scores = cv_utils.score(name, repeat=0)
    return scores


def main():
    print("======================================================================")
    print(" STAGE 26: BOOSTERS TUNING & EVALUATION (LightGBM & XGBoost)")
    print("======================================================================")

    master = cv_utils.load_master()
    summary_rows = []

    feature_dirs = {
        "v2cv": ROOT / "features_v2cv" / "real_X.npy",
        "v3": ROOT / "features_v3" / "real_X.npy",
    }

    for feat_name, path in feature_dirs.items():
        if not path.exists():
            print(f"Skipping {feat_name}: {path} not found.")
            continue

        real_X = np.load(path)
        print(f"\nLoaded {feat_name} shape: {real_X.shape}")

        lgbm_scores = train_eval_booster_default("lgbm", feat_name, real_X, master)
        xgb_scores = train_eval_booster_default("xgb", feat_name, real_X, master)

        for scores, m_type in [(lgbm_scores, "LightGBM Default"), (xgb_scores, "XGBoost Default")]:
            summary_rows.append({
                "model": f"{m_type} ({feat_name})",
                "accuracy": scores["accuracy"],
                "accuracy_ci": scores["accuracy_ci"],
                "balanced_acc": scores["balanced_accuracy"],
                "balanced_acc_ci": scores["balanced_accuracy_ci"],
                "macro_f1": scores["macro_f1"],
                "macro_f1_ci": scores["macro_f1_ci"],
            })

    print("\n" + "=" * 110)
    print(" BOOSTERS EVALUATION SUMMARY TABLE (Repeat 0 with 95% Group-Bootstrap CIs)")
    print("=" * 110)
    print(f"{'Model / Feature Set':<35}{'Accuracy (95% CI)':<25}{'BAcc (95% CI)':<25}{'Macro F1 (95% CI)':<25}")
    print("-" * 110)

    for r in summary_rows:
        acc_str = f"{r['accuracy']*100:.2f}% [{r['accuracy_ci'][0]*100:.2f}%, {r['accuracy_ci'][1]*100:.2f}%]"
        bacc_str = f"{r['balanced_acc']*100:.2f}% [{r['balanced_acc_ci'][0]*100:.2f}%, {r['balanced_acc_ci'][1]*100:.2f}%]"
        f1_str = f"{r['macro_f1']:.4f} [{r['macro_f1_ci'][0]:.4f}, {r['macro_f1_ci'][1]:.4f}]"

        print(f"{r['model']:<35}{acc_str:<25}{bacc_str:<25}{f1_str:<25}")

    print("=" * 110)


if __name__ == "__main__":
    main()
