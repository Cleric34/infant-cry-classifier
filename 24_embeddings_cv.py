"""24_embeddings_cv.py - Cross-validation evaluation of pretrained embeddings.

Evaluates each embedding directory under Part 1's folds:
- Fits StandardScaler + optional PCA(128) inside each fold.
- Fits (a) LogisticRegression (inner 3-fold CV for C in [0.01, 0.1, 1.0, 10.0])
- Fits (b) SVC(kernel="rbf", probability=True, class_weight="balanced").
- Saves OOF to results/cv/<emb>_lr_oof.npy and <emb>_svm_oof.npy.
- Prints ranking table with 95% group-bootstrap CIs.
"""
import importlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

ROOT = Path("infant_cry_dataset")
CV_DIR = ROOT / "cv"
AUG_POOL_DIR = ROOT / "aug_pool"
RESULTS_CV_DIR = ROOT / "results" / "cv"

cv_utils = importlib.import_module("21_cv_utils")

EMBEDDING_DIRS = [
    "features_panns_v2",
    "features_panns",
    "features_clap",
    "features_ast",
    "features_wavlm_mid",
    "features_w2v_mid",
]


def train_eval_embedding(emb_name, real_X, master):
    lr_name = f"{emb_name}_lr"
    svm_name = f"{emb_name}_svm"
    lr_oof_file = RESULTS_CV_DIR / f"{lr_name}_oof.npy"
    svm_oof_file = RESULTS_CV_DIR / f"{svm_name}_oof.npy"

    if lr_oof_file.exists() and svm_oof_file.exists():
        print(f"[{emb_name}] OOF already exists for LR and SVM. Loading and scoring...")
        return cv_utils.score(lr_name, repeat=0), cv_utils.score(svm_name, repeat=0)

    print(f"\n======================================================================")
    print(f" EVALUATING EMBEDDINGS: {emb_name} (Shape: {real_X.shape})")
    print(f"======================================================================")

    y_real = master["y"].values
    groups = master["group"].values
    n_samples = len(master)

    lr_oof = np.zeros((3, n_samples, 4), dtype=np.float32)
    svm_oof = np.zeros((3, n_samples, 4), dtype=np.float32)

    c_candidates = [0.01, 0.1, 1.0, 10.0]

    # Primary evaluation: Repeat 0 (bonus repeats 1, 2 can be run later)
    for rep in range(1):
        folds = cv_utils.get_folds(repeat=rep)

        for fold_idx, (tr_idx, te_idx) in enumerate(folds):
            X_tr_raw = real_X[tr_idx]
            y_tr = y_real[tr_idx]
            g_tr = groups[tr_idx]

            X_te_raw = real_X[te_idx]

            # Fit scaler on train only
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr_raw)
            X_te_s = scaler.transform(X_te_raw)

            # Optional PCA if dimension > 256
            if X_tr_s.shape[1] > 256:
                pca = PCA(n_components=min(128, X_tr_s.shape[1]), random_state=42)
                X_tr_f = pca.fit_transform(X_tr_s)
                X_te_f = pca.transform(X_te_s)
            else:
                X_tr_f = X_tr_s
                X_te_f = X_te_s

            # Inner 3-fold grouped CV for LR C selection
            inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
            best_c = 1.0
            best_inner_score = -1.0

            for c_val in c_candidates:
                inner_f1s = []
                for in_tr, in_val in inner_cv.split(X_tr_f, y_tr, groups=g_tr):
                    clf_in = LogisticRegression(C=c_val, class_weight="balanced", max_iter=500, random_state=42)
                    clf_in.fit(X_tr_f[in_tr], y_tr[in_tr])
                    pred_in = clf_in.predict(X_tr_f[in_val])
                    from sklearn.metrics import f1_score
                    inner_f1s.append(f1_score(y_tr[in_val], pred_in, average="macro", zero_division=0))

                mean_in_f1 = float(np.mean(inner_f1s))
                if mean_in_f1 > best_inner_score:
                    best_inner_score = mean_in_f1
                    best_c = c_val

            # Fit best LR on full outer train
            clf_lr = LogisticRegression(C=best_c, class_weight="balanced", max_iter=500, random_state=42)
            clf_lr.fit(X_tr_f, y_tr)
            lr_oof[rep, te_idx] = clf_lr.predict_proba(X_te_f)

            # Fit SVM on full outer train
            clf_svm = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)
            clf_svm.fit(X_tr_f, y_tr)
            svm_oof[rep, te_idx] = clf_svm.predict_proba(X_te_f)

    # Save OOF
    lr_name = f"{emb_name}_lr"
    svm_name = f"{emb_name}_svm"
    cv_utils.save_oof(lr_name, lr_oof)
    cv_utils.save_oof(svm_name, svm_oof)

    lr_scores = cv_utils.score(lr_name, repeat=0)
    svm_scores = cv_utils.score(svm_name, repeat=0)

    return lr_scores, svm_scores


def main():
    print("======================================================================")
    print(" STAGE 24B: PRETRAINED EMBEDDINGS EVALUATION & RANKING")
    print("======================================================================")

    master = cv_utils.load_master()
    summary_rows = []

    for emb_dir in EMBEDDING_DIRS:
        real_path = ROOT / emb_dir / "real_X.npy"
        if not real_path.exists():
            # Check if features_panns has train_X.npy etc.
            if emb_dir == "features_panns" and (ROOT / emb_dir / "train_X.npy").exists():
                # Load real_X from cached panns_real_all
                cache_p = ROOT / "features" / "panns_real_all.npy"
                if cache_p.exists():
                    real_X = np.load(cache_p)
                else:
                    print(f"Skipping {emb_dir}: real_X.npy missing.")
                    continue
            else:
                print(f"Skipping {emb_dir}: real_X.npy missing.")
                continue
        else:
            real_X = np.load(real_path)

        lr_res, svm_res = train_eval_embedding(emb_dir, real_X, master)

        for res, clf_type in [(lr_res, "LogisticRegression"), (svm_res, "SVM (RBF)")]:
            summary_rows.append({
                "embedding": emb_dir,
                "classifier": clf_type,
                "accuracy": res["accuracy"],
                "accuracy_ci": res["accuracy_ci"],
                "balanced_acc": res["balanced_accuracy"],
                "balanced_acc_ci": res["balanced_accuracy_ci"],
                "macro_f1": res["macro_f1"],
                "macro_f1_ci": res["macro_f1_ci"],
                "burping_recall": res["per_class_recall"][0],
                "burping_recall_ci": res["per_class_recall_ci"][0],
            })

    # Sort by Macro F1
    summary_rows.sort(key=lambda x: x["macro_f1"], reverse=True)

    print("\n" + "=" * 120)
    print(" PRETRAINED EMBEDDINGS RANKING TABLE (Repeat 0 with 95% Group-Bootstrap CIs)")
    print("=" * 120)
    print(f"{'Embedding':<22}{'Classifier':<20}{'Accuracy (95% CI)':<25}{'BAcc (95% CI)':<25}{'Macro F1 (95% CI)':<25}")
    print("-" * 120)

    for r in summary_rows:
        acc_str = f"{r['accuracy']*100:.2f}% [{r['accuracy_ci'][0]*100:.2f}%, {r['accuracy_ci'][1]*100:.2f}%]"
        bacc_str = f"{r['balanced_acc']*100:.2f}% [{r['balanced_acc_ci'][0]*100:.2f}%, {r['balanced_acc_ci'][1]*100:.2f}%]"
        f1_str = f"{r['macro_f1']:.4f} [{r['macro_f1_ci'][0]:.4f}, {r['macro_f1_ci'][1]:.4f}]"

        print(f"{r['embedding']:<22}{r['classifier']:<20}{acc_str:<25}{bacc_str:<25}{f1_str:<25}")

    print("=" * 120)


if __name__ == "__main__":
    main()
