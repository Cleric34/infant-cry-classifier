"""27_stack_cv.py - Stacking ensemble & Class-wise logit offsets.

Ensemble pipeline:
1. Selects top models by out-of-fold Macro F1.
2. Soft voting ensemble (top-2, top-3, all).
3. Multinomial LogisticRegression stacker on out-of-fold log-probabilities.
4. Class-wise logit offset tuning (grid search -1.0 to +1.0 step 0.25) subject to Macro F1 constraint.
5. Saves results/cv/stack_oof.npy and stack_offsets_oof.npy.
6. Prints final comparison table with 95% CIs.
"""
import importlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

ROOT = Path("infant_cry_dataset")
CV_DIR = ROOT / "cv"
RESULTS_CV_DIR = ROOT / "results" / "cv"

cv_utils = importlib.import_module("21_cv_utils")

CLASS_NAMES = ["burping", "needs", "physical_pain", "scared"]


def train_eval_stacker(member_names, master):
    print(f"\n--- Training Multinomial Stacker on {len(member_names)} Models ---")
    y_real = master["y"].values
    groups = master["group"].values
    n_samples = len(master)

    oof_stack = np.zeros((3, n_samples, 4), dtype=np.float32)
    oof_offsets = np.zeros((3, n_samples, 4), dtype=np.float32)

    # Load member OOFs
    member_oofs = []
    for m in member_names:
        path = RESULTS_CV_DIR / f"{m}_oof.npy"
        if not path.exists():
            raise FileNotFoundError(f"Member OOF file {path} not found.")
        member_oofs.append(np.load(path))

    for rep in range(1):
        folds = cv_utils.get_folds(repeat=rep)

        for fold_idx, (tr_idx, te_idx) in enumerate(folds):
            # Form feature representation of log-probabilities for outer train
            tr_features = []
            te_features = []

            for m_oof in member_oofs:
                p_tr = np.clip(m_oof[rep, tr_idx], 1e-7, 1.0 - 1e-7)
                p_te = np.clip(m_oof[rep, te_idx], 1e-7, 1.0 - 1e-7)

                tr_features.append(np.log(p_tr))
                te_features.append(np.log(p_te))

            X_tr_meta = np.hstack(tr_features)
            X_te_meta = np.hstack(te_features)
            y_tr = y_real[tr_idx]

            # Fit stacker (LogisticRegression)
            stacker = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
            stacker.fit(X_tr_meta, y_tr)

            pred_te_probas = stacker.predict_proba(X_te_meta)
            oof_stack[rep, te_idx] = pred_te_probas

            # Logit offset search on training portion
            tr_pred_probas = stacker.predict_proba(X_tr_meta)
            tr_log_probas = np.log(np.clip(tr_pred_probas, 1e-7, 1.0 - 1e-7))

            baseline_tr_f1 = f1_score(y_tr, tr_pred_probas.argmax(axis=1), average="macro", zero_division=0)
            baseline_tr_acc = accuracy_score(y_tr, tr_pred_probas.argmax(axis=1))

            best_offset = np.zeros(4)
            best_offset_acc = baseline_tr_acc

            # Grid search for offsets -1.0 to 1.0
            grid = np.linspace(-1.0, 1.0, 9)
            for o0 in grid:
                for o1 in grid:
                    for o2 in grid:
                        for o3 in grid:
                            offset = np.array([o0, o1, o2, o3])
                            cand_log = tr_log_probas + offset
                            cand_pred = cand_log.argmax(axis=1)

                            cand_f1 = f1_score(y_tr, cand_pred, average="macro", zero_division=0)
                            cand_acc = accuracy_score(y_tr, cand_pred)

                            # Constraint: macro-F1 within 0.005 of baseline
                            if cand_f1 >= (baseline_tr_f1 - 0.005) and cand_acc > best_offset_acc:
                                best_offset_acc = cand_acc
                                best_offset = offset

            # Apply tuned offset to held-out fold
            te_log_probas = np.log(np.clip(pred_te_probas, 1e-7, 1.0 - 1e-7)) + best_offset
            te_offset_probas = np.exp(te_log_probas)
            te_offset_probas /= te_offset_probas.sum(axis=1, keepdims=True)

            oof_offsets[rep, te_idx] = te_offset_probas

    # Save OOFs
    cv_utils.save_oof("stack", oof_stack)
    cv_utils.save_oof("stack_offsets", oof_offsets)

    stack_scores = cv_utils.score("stack", repeat=0)
    offsets_scores = cv_utils.score("stack_offsets", repeat=0)

    return stack_scores, offsets_scores


def main():
    print("======================================================================")
    print(" STAGE 27: STACKING ENSEMBLE & LOGIT OFFSETS EVALUATION")
    print("======================================================================")

    master = cv_utils.load_master()

    # Find available OOF models
    available_oofs = []
    cv_res_dir = ROOT / "results" / "cv"
    if cv_res_dir.exists():
        for p in cv_res_dir.glob("*_oof.npy"):
            m_name = p.stem.replace("_oof", "")
            if m_name not in ("dummy_test_model", "stack", "stack_offsets"):
                available_oofs.append(m_name)

    print(f"Found {len(available_oofs)} available candidate models: {available_oofs}")

    if len(available_oofs) < 2:
        print("WARNING: At least 2 candidate models required for stacking.")
        return

    # Select top models by Macro F1
    model_scores = []
    for m_name in available_oofs:
        try:
            sc = cv_utils.score(m_name, repeat=0)
            model_scores.append((m_name, sc["macro_f1"], sc))
        except Exception as e:
            print(f"Error scoring {m_name}: {e}")

    model_scores.sort(key=lambda x: x[1], reverse=True)
    top_members = [m[0] for m in model_scores[:min(4, len(model_scores))]]

    print(f"\nSelected Top Ensemble Members for Stacking: {top_members}")

    stack_scores, offsets_scores = train_eval_stacker(top_members, master)

    # Soft Vote Baseline (Top-2)
    top2_oof = (np.load(cv_res_dir / f"{top_members[0]}_oof.npy") + np.load(cv_res_dir / f"{top_members[1]}_oof.npy")) / 2.0
    cv_utils.save_oof("soft_vote_top2", top2_oof)
    top2_scores = cv_utils.score("soft_vote_top2", repeat=0)

    print("\n" + "=" * 125)
    print(" FINAL STACKING & ENSEMBLE COMPARISON TABLE (Repeat 0 with 95% Group-Bootstrap CIs)")
    print("=" * 125)
    print(f"{'Ensemble Candidate':<25}{'Accuracy (95% CI)':<25}{'BAcc (95% CI)':<25}{'Macro F1 (95% CI)':<25}{'Burping Recall':<15}")
    print("-" * 125)

    candidates = [
        ("Soft Vote Top-2", top2_scores),
        ("Logistic Stacker", stack_scores),
        ("Stacker + Offsets", offsets_scores),
    ]

    for c_name, sc in candidates:
        acc_str = f"{sc['accuracy']*100:.2f}% [{sc['accuracy_ci'][0]*100:.2f}%, {sc['accuracy_ci'][1]*100:.2f}%]"
        bacc_str = f"{sc['balanced_accuracy']*100:.2f}% [{sc['balanced_accuracy_ci'][0]*100:.2f}%, {sc['balanced_accuracy_ci'][1]*100:.2f}%]"
        f1_str = f"{sc['macro_f1']:.4f} [{sc['macro_f1_ci'][0]:.4f}, {sc['macro_f1_ci'][1]:.4f}]"
        burp_ci = sc['per_class_recall_ci'][0]  # CI for class 0 (burping): (low, high)
        burp_str = f"{sc['per_class_recall'][0]*100:.1f}% [{burp_ci[0]*100:.1f}%, {burp_ci[1]*100:.1f}%]"

        print(f"{c_name:<25}{acc_str:<25}{bacc_str:<25}{f1_str:<25}{burp_str:<15}")

    print("=" * 125)

    # Final declared model rule
    print("\nFINAL DECLARED MODEL SELECTION RULE:")
    print("Select the candidate with the highest OOF Macro F1 among all candidates whose Accuracy >= Baseline Accuracy.")


if __name__ == "__main__":
    main()
