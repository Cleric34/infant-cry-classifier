"""22_aug_ablation.py - Baseline pipeline evaluation & training policy ablations.

Evaluates 5 training policies (P0 to P4) using features_v2cv under 5-fold x 3-repeat CV:
- P0: real files only, no class weights
- P1: real files only, class_weight="balanced"
- P2: real + 1 aug copy for non-Needs, class_weight="balanced"
- P3: real + 2 aug copies for non-Needs, class_weight="balanced"
- P4: real + 2 aug copies for Burping only, class_weight="balanced"

Saves OOF predictions to results/cv/lgbm_v2cv_<policy>_oof.npy.
Prints new BASELINE evaluation table with 95% group-bootstrap CIs.
"""
import importlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from tqdm import tqdm

ROOT = Path("infant_cry_dataset")
CV_DIR = ROOT / "cv"
AUG_POOL_DIR = ROOT / "aug_pool"
FEAT_DIR = ROOT / "features_v2cv"
RESULTS_CV_DIR = ROOT / "results" / "cv"

cv_utils = importlib.import_module("21_cv_utils")

LGBM_PARAMS = dict(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=5,
    num_leaves=20,
    subsample=0.8,
    colsample_bytree=0.3,
    random_state=42,
    n_jobs=1,
    verbose=-1,
)

POLICIES = {
    "P0": {"aug_copies": None, "class_weight": None, "desc": "real only, no weights"},
    "P1": {"aug_copies": None, "class_weight": "balanced", "desc": "real only, balanced"},
    "P2": {"aug_copies": 1, "class_weight": "balanced", "desc": "real + 1 aug (non-Needs), balanced"},
    "P3": {"aug_copies": 2, "class_weight": "balanced", "desc": "real + 2 aug (non-Needs), balanced"},
    "P4": {"aug_copies": {"burping": 2, "needs": 0, "physical_pain": 0, "scared": 0}, "class_weight": "balanced", "desc": "real + 2 aug (Burping only), balanced"},
}


def fit_single_fold(fold_idx, tr_idx, te_idx, real_X, aug_X, y_real, aug_df, aug_copies, class_weight, params):
    X_tr_real = real_X[tr_idx]
    y_tr_real = y_real[tr_idx]

    if aug_copies is not None and len(aug_X) > 0 and len(aug_df) > 0:
        aug_row_idx = cv_utils.aug_rows_for(tr_idx, max_copies=aug_copies)
        if len(aug_row_idx) > 0:
            X_tr_aug = aug_X[aug_row_idx]
            y_tr_aug = aug_df.iloc[aug_row_idx]["y"].values
            X_train = np.vstack([X_tr_real, X_tr_aug])
            y_train = np.concatenate([y_tr_real, y_tr_aug])
        else:
            X_train, y_train = X_tr_real, y_tr_real
    else:
        X_train, y_train = X_tr_real, y_tr_real

    X_test = real_X[te_idx]
    clf = LGBMClassifier(class_weight=class_weight, **params)
    clf.fit(X_train, y_train)
    preds = clf.predict_proba(X_test)
    return fold_idx, te_idx, preds


def train_eval_policy(policy_name, config, real_X, aug_X, master, aug_df):
    print(f"\n======================================================================")
    print(f" EVALUATING POLICY {policy_name}: {config['desc']}")
    print(f"======================================================================")

    y_real = master["y"].values
    n_samples = len(master)
    oof_probas = np.zeros((3, n_samples, 4), dtype=np.float32)

    aug_copies = config["aug_copies"]
    class_weight = config["class_weight"]

    model_name = f"lgbm_v2cv_{policy_name.lower()}"
    oof_file = RESULTS_CV_DIR / f"{model_name}_oof.npy"
    if oof_file.exists():
        print(f"[{policy_name}] OOF already exists at {oof_file}. Loading and scoring...")
        return cv_utils.score(model_name, repeat=0)

    # Primary evaluation: Repeat 0 (bonus repeats 1, 2 can be run later)
    for rep in range(1):
        folds = cv_utils.get_folds(repeat=rep)
        n_workers = min(len(folds), max(1, os.cpu_count() or 4))

        print(f"Fitting {len(folds)} folds in parallel using {n_workers} CPU workers...")
        results = Parallel(n_jobs=n_workers)(
            delayed(fit_single_fold)(
                fold_idx, tr_idx, te_idx, real_X, aug_X, y_real, aug_df, aug_copies, class_weight, LGBM_PARAMS
            )
            for fold_idx, (tr_idx, te_idx) in enumerate(folds)
        )

        for fold_idx, te_idx, preds in results:
            oof_probas[rep, te_idx] = preds

    # Save OOF
    cv_utils.save_oof(model_name, oof_probas)

    # Score repeat 0
    scores = cv_utils.score(model_name, repeat=0)
    return scores


def main():
    print("======================================================================")
    print(" STAGE 22C: TRAINING POLICY ABLATION & NEW BASELINE EVALUATION")
    print("======================================================================")

    master = cv_utils.load_master()
    real_X = np.load(FEAT_DIR / "real_X.npy")

    aug_X_path = FEAT_DIR / "aug_X.npy"
    aug_X = np.load(aug_X_path) if aug_X_path.exists() else np.array([])
    aug_manifest_path = AUG_POOL_DIR / "aug_manifest.csv"
    aug_df = pd.read_csv(aug_manifest_path) if aug_manifest_path.exists() else pd.DataFrame()

    summary_rows = []

    for p_name, p_config in POLICIES.items():
        res = train_eval_policy(p_name, p_config, real_X, aug_X, master, aug_df)
        summary_rows.append({
            "policy": p_name,
            "description": p_config["desc"],
            "accuracy": res["accuracy"],
            "accuracy_ci": res["accuracy_ci"],
            "balanced_acc": res["balanced_accuracy"],
            "balanced_acc_ci": res["balanced_accuracy_ci"],
            "macro_f1": res["macro_f1"],
            "macro_f1_ci": res["macro_f1_ci"],
            "burping_recall": res["per_class_recall"][0],
            "burping_recall_ci": res["per_class_recall_ci"][0],
        })

    # Print summary table
    print("\n" + "=" * 120)
    print(" NEW CV BASELINE TABLE (Policy Comparison on features_v2cv, Repeat 0 with 95% Group-Bootstrap CIs)")
    print("=" * 120)
    print(f"{'Policy':<8}{'Description':<38}{'Accuracy (95% CI)':<22}{'BAcc (95% CI)':<22}{'Macro F1 (95% CI)':<22}{'Burping Recall':<15}")
    print("-" * 120)

    best_p_name = None
    best_f1 = -1.0

    for r in summary_rows:
        acc_str = f"{r['accuracy']*100:.2f}% [{r['accuracy_ci'][0]*100:.2f}%, {r['accuracy_ci'][1]*100:.2f}%]"
        bacc_str = f"{r['balanced_acc']*100:.2f}% [{r['balanced_acc_ci'][0]*100:.2f}%, {r['balanced_acc_ci'][1]*100:.2f}%]"
        f1_str = f"{r['macro_f1']:.4f} [{r['macro_f1_ci'][0]:.4f}, {r['macro_f1_ci'][1]:.4f}]"
        burp_str = f"{r['burping_recall']*100:.1f}% [{r['burping_recall_ci'][0]*100:.1f}%, {r['burping_recall_ci'][1]*100:.1f}%]"

        print(f"{r['policy']:<8}{r['description']:<38}{acc_str:<22}{bacc_str:<22}{f1_str:<22}{burp_str:<15}")

        if r["macro_f1"] > best_f1:
            best_f1 = r["macro_f1"]
            best_p_name = r["policy"]

    print("=" * 120)
    print(f"\nBEST BASELINE POLICY: {best_p_name} ({POLICIES[best_p_name]['desc']}) with Macro F1 = {best_f1:.4f}")


if __name__ == "__main__":
    main()
