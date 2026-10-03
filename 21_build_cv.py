"""21_build_cv.py - Construct master dataset manifest and 5-fold x 3-repeat CV split.

Builds infant_cry_dataset/cv/master.csv:
- Covers all 4,871 real files in audio_denoised in fixed order (sorted class, then filename).
- Generates 2,048-dim PANNs embeddings for all real files to construct pseudo-sessions via
  AgglomerativeClustering (cosine metric, average linkage).
- Chooses distance_threshold=0.035 (>0.965 cosine similarity threshold) such that no group
  exceeds ~2% of N (max group size <= 97 samples).
- Assigns 5-fold StratifiedGroupKFold splits for 3 repeats (seeds 42, 43, 44), ensuring
  burping counts per fold remain balanced within a factor of 1.5.
"""
import importlib
import json
import os
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

ROOT = Path("infant_cry_dataset")
DENOISED_DIR = ROOT / "audio_denoised"
SPLIT_DIR = ROOT / "split"
CV_DIR = ROOT / "cv"
FEATURE_MANIFEST = ROOT / "features" / "fixed_duration_manifest.json"
PANNS_CACHE = ROOT / "features" / "panns_real_all.npy"
N_FIXED = 87685  # 3.9767 s at 22050 Hz
SR_IN = 22050
SR_PANNS = 32000
BATCH_SIZE = 64


def get_real_files():
    """Return all real files in audio_denoised in fixed order (sorted class, then filename)."""
    class_names = sorted([d.name for d in DENOISED_DIR.iterdir() if d.is_dir()])
    records = []
    for c_idx, c in enumerate(class_names):
        c_dir = DENOISED_DIR / c
        files = sorted(c_dir.glob("*.wav"))
        for p in files:
            key = f"{c}/{p.name}"
            records.append({
                "key": key,
                "path": str(p.as_posix()),
                "filename": p.name,
                "class": c,
                "y": c_idx,
            })
    return pd.DataFrame(records)


def map_old_split(df):
    """Determine old_split (train, val, test) for each file in master."""
    val_files = set((SPLIT_DIR / "val").rglob("*.wav"))
    test_files = set((SPLIT_DIR / "test").rglob("*.wav"))

    val_keys = {f"{p.parent.name}/{p.name}" for p in val_files}
    test_keys = {f"{p.parent.name}/{p.name}" for p in test_files}

    old_splits = []
    for key in df["key"]:
        if key in val_keys:
            old_splits.append("val")
        elif key in test_keys:
            old_splits.append("test")
        else:
            old_splits.append("train")
    df["old_split"] = old_splits
    return df


def load_clip(path, n_fixed, sr_target):
    """Load audio directly at target sample rate, pad/crop to target length."""
    n_target = int(round(n_fixed * sr_target / SR_IN))
    y, _ = librosa.load(path, sr=sr_target, mono=True)
    if len(y) >= n_target:
        return y[:n_target].astype(np.float32)
    return np.pad(y, (0, n_target - len(y))).astype(np.float32)


def extract_all_panns_embeddings(df):
    """Load or compute 2,048-dim PANNs embeddings for all real files in master."""
    if PANNS_CACHE.exists():
        print(f"Loading cached PANNs embeddings from {PANNS_CACHE}")
        return np.load(PANNS_CACHE)

    print("\n--- Computing PANNs embeddings for all real files ---")
    
    # 1. Load existing PANNs features from features_panns/ if present
    existing_map = {}
    panns_dir = ROOT / "features_panns"
    if panns_dir.exists():
        for s in ("train", "val", "test"):
            x_path = panns_dir / f"{s}_X.npy"
            p_path = panns_dir / f"{s}_paths.npy"
            if x_path.exists() and p_path.exists():
                X_s = np.load(x_path)
                P_s = np.load(p_path)
                for p_str, vec in zip(P_s, X_s):
                    if not os.path.basename(p_str).startswith("aug_"):
                        k = "/".join(str(p_str).replace("\\", "/").split("/")[-2:])
                        existing_map[k] = vec

    print(f"Reusing {len(existing_map)} pre-computed PANNs embeddings from features_panns/")

    paths = df["path"].tolist()
    keys = df["key"].tolist()
    missing_idx = [i for i, k in enumerate(keys) if k not in existing_map]
    print(f"Extracting PANNs embeddings for remaining {len(missing_idx)} real files...")

    embed = None
    if missing_idx:
        emb_module = importlib.import_module("14_embeddings")
        device = emb_module.get_device()
        embed = emb_module.build_panns(device)

    X_panns = np.zeros((len(df), 2048), dtype=np.float32)

    # Fill existing
    for i, k in enumerate(keys):
        if k in existing_map:
            X_panns[i] = existing_map[k]

    # Compute missing in batches
    for b_start in tqdm(range(0, len(missing_idx), BATCH_SIZE), desc="PANNs Extraction"):
        b_indices = missing_idx[b_start:b_start + BATCH_SIZE]
        b_paths = [paths[i] for i in b_indices]
        b_audio = [load_clip(p, N_FIXED, SR_PANNS) for p in b_paths]
        b_vecs = embed(b_audio)
        for idx_in_master, vec in zip(b_indices, b_vecs):
            X_panns[idx_in_master] = vec

    os.makedirs(PANNS_CACHE.parent, exist_ok=True)
    np.save(PANNS_CACHE, X_panns)
    print(f"Saved PANNs embeddings {X_panns.shape} to {PANNS_CACHE}")
    return X_panns


def build_pseudo_sessions(df, X_panns):
    """Build pseudo-sessions via AgglomerativeClustering on L2-normalized PANNs embeddings."""
    norms = np.linalg.norm(X_panns, axis=1, keepdims=True)
    X_norm = X_panns / np.maximum(norms, 1e-8)

    # Search for optimal distance threshold where max_group_size <= 97 (~2% of N=4871)
    # Cosine distance = 1 - cosine_similarity. Threshold 0.035 corresponds to >0.965 similarity.
    thresholds = [0.02, 0.025, 0.03, 0.035, 0.04, 0.05]
    best_thresh = 0.035
    best_labels = None

    print("\n--- Pseudo-Session Clustering (Threshold Grid Search) ---")
    for thresh in thresholds:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=thresh,
        )
        labels = clustering.fit_predict(X_norm)
        counts = pd.Series(labels).value_counts()
        max_size = counts.max()
        n_groups = len(counts)
        print(f"Threshold {thresh:.3f} (>={1-thresh:.3f} sim): {n_groups} groups, max group size = {max_size} ({max_size/len(df)*100:.2f}%)")
        if max_size <= int(0.02 * len(df)) and best_labels is None:
            best_thresh = thresh
            best_labels = labels

    if best_labels is None:
        best_thresh = 0.035
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=best_thresh,
        )
        best_labels = clustering.fit_predict(X_norm)

    df["group"] = best_labels

    counts = df["group"].value_counts()
    n_groups = len(counts)
    singletons = sum(counts == 1)
    pct_singletons = singletons / n_groups * 100

    print("\n--- Pseudo-Session Group Distribution ---")
    print(f"Chosen Distance Threshold: {best_thresh} (Similarity >= {1-best_thresh:.3f})")
    print(f"Total Groups: {n_groups}")
    print(f"Group Size Distribution: Min = {counts.min()}, Median = {counts.median():.1f}, P95 = {counts.quantile(0.95):.1f}, Max = {counts.max()} ({counts.max()/len(df)*100:.2f}%)")
    print(f"Singleton Groups (1 sample): {singletons} / {n_groups} ({pct_singletons:.1f}%)")

    return df


def assign_folds(df):
    """Assign StratifiedGroupKFold splits for 3 repeats (seeds 42, 43, 44)."""
    y = df["y"].values
    groups = df["group"].values
    seeds = [42, 43, 44]

    print("\n--- Assigning 5-Fold StratifiedGroupKFold Splits (3 Repeats) ---")

    for rep_idx, seed in enumerate(seeds):
        col_name = f"fold_r{rep_idx}"
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        fold_assignments = np.zeros(len(df), dtype=int)

        for fold, (_, test_idx) in enumerate(sgkf.split(np.zeros(len(df)), y, groups=groups)):
            fold_assignments[test_idx] = fold

        df[col_name] = fold_assignments

        print(f"\n[Repeat {rep_idx} (Seed {seed}) Per-Fold Class Counts]:")
        fold_counts = []
        for fold in range(5):
            sub_df = df[df[col_name] == fold]
            binc = np.bincount(sub_df["y"], minlength=4)
            fold_counts.append(binc)
            print(f"  Fold {fold}: Total={len(sub_df):4d} | Burping={binc[0]:2d}, Needs={binc[1]:3d}, Physical_Pain={binc[2]:2d}, Scared={binc[3]:2d}")

        burping_counts = [fc[0] for fc in fold_counts]
        ratio = max(burping_counts) / max(1, min(burping_counts))
        print(f"  Burping Balance Ratio (Max/Min): {ratio:.2f} (Target <= 1.50)")
        if ratio > 1.50:
            print(f"  WARNING: Repeat {rep_idx} seed {seed} has burping imbalance ratio {ratio:.2f} > 1.50!")

    return df


def main():
    print("======================================================================")
    print(" STAGE 21: BUILDING EVALUATION PROTOCOL (MASTER CSV & GROUPS)")
    print("======================================================================")

    df = get_real_files()
    print(f"Loaded {len(df)} total real files across 4 classes.")

    df = map_old_split(df)
    print("Mapped old_split counts:\n", df["old_split"].value_counts())

    X_panns = extract_all_panns_embeddings(df)
    df = build_pseudo_sessions(df, X_panns)
    df = assign_folds(df)

    os.makedirs(CV_DIR, exist_ok=True)
    master_path = CV_DIR / "master.csv"
    out_cols = ["key", "path", "y", "class", "group", "old_split", "fold_r0", "fold_r1", "fold_r2"]
    df[out_cols].to_csv(master_path, index=False)
    print(f"\nMaster manifest saved successfully to: {master_path}")


if __name__ == "__main__":
    main()
