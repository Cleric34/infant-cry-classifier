"""22_extract_v2cv.py - Feature extraction for features_v2cv (real and aug pool).

Re-runs the exact Kymatio scattering + MFCC feature extractor from 11_kymatio_mfcc_features.py:
- J=13, Q=8, fixed_length=87685, 22050 Hz.
- Pools mean, var, skew, kurtosis, max, min over time -> 4,098 features per clip.
- Saves infant_cry_dataset/features_v2cv/real_X.npy, aug_X.npy, feature_names.json.
- Chunked, resumable, parallelized with joblib, progress & ETA printed.
"""
import json
import os
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from kymatio.scattering1d.frontend.numpy_frontend import ScatteringNumPy1D as Scattering1D
from scipy.stats import kurtosis, skew
from tqdm import tqdm

ROOT = Path("infant_cry_dataset")
CV_DIR = ROOT / "cv"
AUG_POOL_DIR = ROOT / "aug_pool"
OUT_DIR = ROOT / "features_v2cv"
CACHE_DIR = OUT_DIR / "_cache"

SR, J, Q = 22050, 13, 8
N_FIXED = 87685  # 3.9767 s at 22050 Hz
CHUNK_SIZE = 32
STATS = ["mean", "var", "skew", "kurt", "max", "min"]


def pool(M):
    return np.concatenate([M.mean(1), M.var(1), skew(M, 1), kurtosis(M, 1), M.max(1), M.min(1)])


def process_chunk(paths, cache_file):
    if cache_file.exists():
        return np.load(cache_file)

    S = Scattering1D(J=J, shape=N_FIXED, Q=Q)
    chunk_feats = []
    for p in paths:
        x, _ = librosa.load(p, sr=SR, mono=True)
        if len(x) >= N_FIXED:
            x = x[:N_FIXED]
        else:
            x = np.pad(x, (0, N_FIXED - len(x)))

        sx = np.log1p(S(x.astype(np.float32)))
        m = librosa.feature.mfcc(y=x, sr=SR, n_mfcc=13, n_fft=1024, hop_length=256)
        m = np.vstack([m, librosa.feature.delta(m), librosa.feature.delta(m, order=2)])

        feat = np.nan_to_num(np.concatenate([pool(sx), pool(m)]))
        chunk_feats.append(feat.astype(np.float32))

    arr = np.array(chunk_feats, dtype=np.float32)

    tmp_file = cache_file.with_suffix(".tmp.npy")
    np.save(tmp_file, arr)
    tmp_file.replace(cache_file)
    return arr


def preseed_cache_from_features_v2(name, paths):
    """Pre-fill chunk cache from features_v2 for files that already exist in split."""
    v2_dir = ROOT / "features_v2"
    if not v2_dir.exists():
        return

    v2_map = {}
    for s in ("train", "val", "test"):
        x_p = v2_dir / f"{s}_X.npy"
        paths_p = v2_dir / f"{s}_paths.npy"
        if x_p.exists() and paths_p.exists():
            X_s = np.load(x_p)
            P_s = np.load(paths_p)
            for p_str, vec in zip(P_s, X_s):
                k = "/".join(str(p_str).replace("\\", "/").split("/")[-2:])
                v2_map[k] = vec

    chunks = [paths[i:i + CHUNK_SIZE] for i in range(0, len(paths), CHUNK_SIZE)]
    cache_files = [CACHE_DIR / f"{name}_chunk_{i:04d}.npy" for i in range(len(chunks))]

    for chunk, cfile in zip(chunks, cache_files):
        if cfile.exists():
            continue
        chunk_keys = ["/".join(str(p).replace("\\", "/").split("/")[-2:]) for p in chunk]
        if all(k in v2_map for k in chunk_keys):
            arr = np.array([v2_map[k] for k in chunk_keys], dtype=np.float32)
            np.save(cfile, arr)


def extract_set(name, paths):
    print(f"\n--- Extracting features_v2cv for '{name}' ({len(paths)} files) ---")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    preseed_cache_from_features_v2(name, paths)

    n_jobs = max(1, (os.cpu_count() or 4) - 1)
    chunks = [paths[i:i + CHUNK_SIZE] for i in range(0, len(paths), CHUNK_SIZE)]
    cache_files = [CACHE_DIR / f"{name}_chunk_{i:04d}.npy" for i in range(len(chunks))]

    cached_count = sum(f.exists() for f in cache_files)
    print(f"Total chunks: {len(chunks)} ({cached_count} already cached). Using {n_jobs} CPU cores.")

    t0 = time.perf_counter()
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_chunk)(chunk, cfile)
        for chunk, cfile in zip(tqdm(chunks, desc=f"v2cv [{name}]"), cache_files)
    )
    elapsed = time.perf_counter() - t0

    X = np.concatenate(results, axis=0).astype(np.float32)
    out_path = OUT_DIR / f"{name}_X.npy"
    np.save(out_path, X)
    print(f"Saved {name}_X.npy shape {X.shape} to {out_path} (Time: {elapsed:.1f}s)")
    return X


def main():
    print("======================================================================")
    print(" STAGE 22B: FEATURE EXTRACTION FOR features_v2cv")
    print("======================================================================")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    master_path = CV_DIR / "master.csv"
    if not master_path.exists():
        raise FileNotFoundError("master.csv missing. Run 21_build_cv.py first.")
    master = pd.read_csv(master_path)

    aug_manifest_path = AUG_POOL_DIR / "aug_manifest.csv"
    if not aug_manifest_path.exists():
        raise FileNotFoundError("aug_manifest.csv missing. Run 22_build_aug_pool.py first.")
    aug_manifest = pd.read_csv(aug_manifest_path)

    real_paths = master["path"].tolist()
    aug_paths = aug_manifest["path"].tolist()

    real_X = extract_set("real", real_paths)
    aug_X = extract_set("aug", aug_paths)

    # Save feature names
    n_sx_features = real_X.shape[1] - (39 * 6)
    n_coeffs = n_sx_features // 6
    names = [f"sx{c}_{s}" for s in STATS for c in range(n_coeffs)] + \
            [f"{kk}{c}_{s}" for s in STATS for kk in ("mfcc", "dmfcc", "ddmfcc") for c in range(13)]

    assert len(names) == real_X.shape[1], f"Feature names count mismatch: {len(names)} vs {real_X.shape[1]}"
    with open(OUT_DIR / "feature_names.json", "w") as f:
        json.dump(names, f, indent=2)

    print("\nFeature names saved to:", OUT_DIR / "feature_names.json")
    print("features_v2cv extraction complete!")


if __name__ == "__main__":
    main()
