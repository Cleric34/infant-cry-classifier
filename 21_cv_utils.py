"""21_cv_utils.py - Cross-validation helpers and evaluation reporting for infant cry classification.

Provides:
- load_master(): loads infant_cry_dataset/cv/master.csv
- get_folds(repeat): returns 5-fold (train_idx, test_idx) list over master rows
- aug_rows_for(train_idx, max_copies): returns indices into aug_manifest.csv for training
- save_oof(name, oof): saves out-of-fold array shape (3, N, 4) to results/cv/<name>_oof.npy
- score(name_or_array, repeat=0): prints & returns metrics with 95% group-bootstrap CIs
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

DEFAULT_ROOT = Path("infant_cry_dataset")


def load_master(root=DEFAULT_ROOT):
    """Load and return master.csv DataFrame."""
    root = Path(root)
    master_path = root / "cv" / "master.csv"
    if not master_path.exists():
        raise FileNotFoundError(f"Master manifest not found at {master_path}. Run 21_build_cv.py first.")
    return pd.read_csv(master_path)


def get_folds(repeat=0, root=DEFAULT_ROOT):
    """Return list of 5 tuples (train_idx, test_idx) over master rows for repeat (0, 1, or 2)."""
    df = load_master(root)
    col = f"fold_r{repeat}"
    if col not in df.columns:
        raise ValueError(f"Repeat column '{col}' not found in master.csv")
    folds = []
    for f in range(5):
        test_idx = np.where(df[col] == f)[0]
        train_idx = np.where(df[col] != f)[0]
        folds.append((train_idx, test_idx))
    return folds


def aug_rows_for(train_idx, max_copies=None, root=DEFAULT_ROOT):
    """Return row indices into aug_manifest.csv whose source_key is in train_idx.

    max_copies: int or dict {class_name: int} specifying max allowed copy_idx per clip.
    Returns an empty numpy array if aug_manifest.csv does not exist yet.
    """
    root = Path(root)
    aug_path = root / "aug_pool" / "aug_manifest.csv"
    if not aug_path.exists():
        return np.array([], dtype=int)

    aug_df = pd.read_csv(aug_path)
    master_df = load_master(root)

    train_keys = set(master_df.iloc[train_idx]["key"])
    mask = aug_df["source_key"].isin(train_keys)

    if max_copies is not None:
        if isinstance(max_copies, (int, float)):
            mask = mask & (aug_df["copy_idx"] <= int(max_copies))
        elif isinstance(max_copies, dict):
            class_masks = []
            for cls_name, limit in max_copies.items():
                c_mask = (aug_df["class"] == cls_name) & (aug_df["copy_idx"] <= limit)
                class_masks.append(c_mask)
            if class_masks:
                limit_mask = pd.concat(class_masks, axis=1).any(axis=1)
                mask = mask & limit_mask

    return np.where(mask)[0]


def save_oof(name, oof, root=DEFAULT_ROOT):
    """Save out-of-fold array shape (3, N, 4) to infant_cry_dataset/results/cv/<name>_oof.npy."""
    root = Path(root)
    oof = np.asarray(oof, dtype=np.float32)
    master = load_master(root)
    n_samples = len(master)
    if oof.shape != (3, n_samples, 4):
        raise ValueError(f"OOF shape mismatch: expected (3, {n_samples}, 4), got {oof.shape}")

    out_dir = root / "results" / "cv"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}_oof.npy"
    np.save(out_path, oof)
    print(f"Saved OOF predictions to {out_path} (shape: {oof.shape})")


def _group_bootstrap_ci(y_true, y_pred, groups, n_boot=2000, seed=42):
    """Compute 95% group-bootstrap confidence intervals for accuracy, bacc, macro_f1, and recall."""
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    group_map = [np.where(groups == g)[0] for g in unique_groups]
    group_map_arr = np.array(group_map, dtype=object)

    boot_accs = np.empty(n_boot, dtype=np.float64)
    boot_baccs = np.empty(n_boot, dtype=np.float64)
    boot_f1s = np.empty(n_boot, dtype=np.float64)
    boot_recalls = np.empty((4, n_boot), dtype=np.float64)

    for b in range(n_boot):
        chosen_g = rng.integers(0, n_groups, size=n_groups)
        boot_idx = np.concatenate(group_map_arr[chosen_g])

        sub_y = y_true[boot_idx]
        sub_p = y_pred[boot_idx]

        # Fast 4x4 confusion matrix via bincount
        cm = np.bincount(sub_y * 4 + sub_p, minlength=16).reshape(4, 4)
        tp = np.diag(cm).astype(np.float64)
        support = cm.sum(axis=1).astype(np.float64)
        predicted = cm.sum(axis=0).astype(np.float64)

        rec = np.where(support > 0, tp / np.maximum(support, 1.0), 0.0)
        prec = np.where(predicted > 0, tp / np.maximum(predicted, 1.0), 0.0)
        f1_c = np.where((rec + prec) > 0, 2.0 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)

        boot_accs[b] = tp.sum() / max(cm.sum(), 1)
        boot_baccs[b] = rec.mean()
        boot_f1s[b] = f1_c.mean()
        for c in range(4):
            boot_recalls[c, b] = rec[c]

    ci = lambda arr: (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))

    return {
        "accuracy_ci": ci(boot_accs),
        "balanced_accuracy_ci": ci(boot_baccs),
        "macro_f1_ci": ci(boot_f1s),
        "per_class_recall_ci": [ci(boot_recalls[c]) for c in range(4)],
    }



def score(name_or_array, repeat=0, n_boot=2000, seed=42, root=DEFAULT_ROOT):
    """Print and return metrics with 95% group-bootstrap CIs for specified repeat (0, 1, or 2)."""
    root = Path(root)
    master = load_master(root)
    y_true = master["y"].values
    groups = master["group"].values
    class_names = sorted(master["class"].unique())

    if isinstance(name_or_array, (str, Path)):
        name = str(name_or_array)
        path = root / "results" / "cv" / f"{name}_oof.npy"
        if not path.exists():
            raise FileNotFoundError(f"OOF predictions file not found: {path}")
        oof = np.load(path)
    else:
        name = "array_input"
        oof = np.asarray(name_or_array)

    if oof.ndim != 3 or oof.shape[0] != 3 or oof.shape[1] != len(master) or oof.shape[2] != 4:
        raise ValueError(f"OOF array must have shape (3, {len(master)}, 4), got {oof.shape}")

    probas = oof[repeat]
    if np.isnan(probas).all():
        raise ValueError(f"All values for repeat {repeat} are NaN in model '{name}'.")

    valid_mask = ~np.isnan(probas).any(axis=1)
    if not valid_mask.all():
        print(f"WARNING: score() evaluating on {valid_mask.sum()}/{len(valid_mask)} non-NaN rows for repeat {repeat}.")

    y_sub = y_true[valid_mask]
    g_sub = groups[valid_mask]
    p_sub = probas[valid_mask]
    y_pred = p_sub.argmax(axis=1)

    acc = float(accuracy_score(y_sub, y_pred))
    bacc = float(balanced_accuracy_score(y_sub, y_pred))
    f1 = float(f1_score(y_sub, y_pred, average="macro", zero_division=0))
    _, r, _, _ = precision_recall_fscore_support(y_sub, y_pred, labels=[0, 1, 2, 3], zero_division=0)
    cm = confusion_matrix(y_sub, y_pred, labels=[0, 1, 2, 3])

    cis = _group_bootstrap_ci(y_sub, y_pred, g_sub, n_boot=n_boot, seed=seed)

    # Secondary View: old_split == 'test'
    test_mask = (master["old_split"] == "test").values & valid_mask
    if test_mask.any():
        y_t = y_true[test_mask]
        g_t = groups[test_mask]
        p_t = probas[test_mask].argmax(axis=1)
        t_acc = float(accuracy_score(y_t, p_t))
        t_bacc = float(balanced_accuracy_score(y_t, p_t))
        t_f1 = float(f1_score(y_t, p_t, average="macro", zero_division=0))
        _, t_r, _, _ = precision_recall_fscore_support(y_t, p_t, labels=[0, 1, 2, 3], zero_division=0)
        t_cis = _group_bootstrap_ci(y_t, p_t, g_t, n_boot=n_boot, seed=seed)
        sec_view = {
            "accuracy": t_acc, "accuracy_ci": t_cis["accuracy_ci"],
            "balanced_accuracy": t_bacc, "balanced_accuracy_ci": t_cis["balanced_accuracy_ci"],
            "macro_f1": t_f1, "macro_f1_ci": t_cis["macro_f1_ci"],
            "per_class_recall": [float(val) for val in t_r],
            "per_class_recall_ci": t_cis["per_class_recall_ci"],
        }
    else:
        sec_view = None

    print(f"\n======================================================================")
    print(f" EVALUATION REPORT: {name} (Repeat {repeat}, n={len(y_sub)})")
    print(f"======================================================================")
    print(f"Accuracy         : {acc*100:.2f}%  (95% CI: [{cis['accuracy_ci'][0]*100:.2f}%, {cis['accuracy_ci'][1]*100:.2f}%])")
    print(f"Balanced Accuracy: {bacc*100:.2f}%  (95% CI: [{cis['balanced_accuracy_ci'][0]*100:.2f}%, {cis['balanced_accuracy_ci'][1]*100:.2f}%])")
    print(f"Macro F1         : {f1:.4f}   (95% CI: [{cis['macro_f1_ci'][0]:.4f}, {cis['macro_f1_ci'][1]:.4f}])")
    print("\nPer-Class Recall (with 95% Group Bootstrap CI):")
    for idx, c in enumerate(class_names):
        r_val = r[idx] * 100
        ci_l, ci_u = cis["per_class_recall_ci"][idx][0] * 100, cis["per_class_recall_ci"][idx][1] * 100
        print(f"  {c:<15}: {r_val:>5.1f}%  (95% CI: [{ci_l:>5.1f}%, {ci_u:>5.1f}%])")

    if sec_view:
        print(f"\n--- Secondary View (Old Split 'test' subset, n={test_mask.sum()}) ---")
        print(f"Test Accuracy   : {sec_view['accuracy']*100:.2f}%  (95% CI: [{sec_view['accuracy_ci'][0]*100:.2f}%, {sec_view['accuracy_ci'][1]*100:.2f}%])")
        print(f"Test Macro F1   : {sec_view['macro_f1']:.4f}   (95% CI: [{sec_view['macro_f1_ci'][0]:.4f}, {sec_view['macro_f1_ci'][1]:.4f}])")

    res = {
        "name": name,
        "repeat": repeat,
        "n": len(y_sub),
        "accuracy": acc, "accuracy_ci": cis["accuracy_ci"],
        "balanced_accuracy": bacc, "balanced_accuracy_ci": cis["balanced_accuracy_ci"],
        "macro_f1": f1, "macro_f1_ci": cis["macro_f1_ci"],
        "per_class_recall": [float(val) for val in r],
        "per_class_recall_ci": cis["per_class_recall_ci"],
        "confusion_matrix": cm.tolist(),
        "secondary_old_test_view": sec_view,
    }
    return res


def _self_test():
    """Self-test: verify dummy OOF array shapes, coverage, fold isolation, and score()."""
    print("\n======================================================================")
    print(" RUNNING 21_cv_utils.py SELF-TEST")
    print("======================================================================")

    master = load_master()
    n_samples = len(master)
    print(f"Master manifest loaded successfully ({n_samples} rows).")

    # 1. Test get_folds
    for rep in range(3):
        folds = get_folds(repeat=rep)
        assert len(folds) == 5, f"Repeat {rep} must have 5 folds"
        seen_test = np.zeros(n_samples, dtype=int)

        for fold_idx, (tr_idx, te_idx) in enumerate(folds):
            seen_test[te_idx] += 1

            # Check zero group overlap
            tr_groups = set(master.iloc[tr_idx]["group"])
            te_groups = set(master.iloc[te_idx]["group"])
            overlap = tr_groups.intersection(te_groups)
            assert len(overlap) == 0, f"Group leakage detected in repeat {rep} fold {fold_idx}: {len(overlap)} overlapping groups!"

        assert (seen_test == 1).all(), f"In repeat {rep}, not all samples appear in exactly 1 test fold!"
        print(f"Repeat {rep} folds verification PASSED: 5 folds, exact 1-to-1 coverage, 0 group leakage.")

    # 2. Test aug_rows_for before aug_manifest exists
    aug_rows = aug_rows_for(folds[0][0])
    print(f"aug_rows_for() before aug_manifest exists: returned shape {aug_rows.shape} (PASSED).")

    # 3. Test dummy OOF and score()
    rng = np.random.default_rng(42)
    dummy_oof = rng.random((3, n_samples, 4))
    dummy_oof /= dummy_oof.sum(axis=-1, keepdims=True)

    save_oof("dummy_test_model", dummy_oof)
    scores = score("dummy_test_model", repeat=0, n_boot=200)
    print(f"Dummy model score calculation PASSED: Macro F1 = {scores['macro_f1']:.4f}")
    print("\n======================================================================")
    print(" ALL 21_cv_utils.py SELF-TESTS PASSED SUCCESSFULLY!")
    print("======================================================================")


if __name__ == "__main__":
    _self_test()
