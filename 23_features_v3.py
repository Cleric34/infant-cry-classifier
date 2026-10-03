"""23_features_v3.py - Handcrafted audio features v3 with active-frame VAD pooling.

Fixes padding/cropping pitfalls:
1. Loads clips at 22050 Hz at true length (capped at 8.0 s, no zero-padding to 3.98 s).
2. Voice-activity trimming: frame RMS (hop 256, frame 1024), active frames = RMS > -35 dB
   relative to 95th percentile RMS. Minimum 0.3 s active audio required (else fallback to whole clip).
3. Feature blocks:
   - scat_: Kymatio scattering 1D (J=8, Q=8) on 2 s active windows (hop 1 s).
   - mfcc_: MFCC(13) + delta + delta2 (39 channels).
   - spec_: Centroid, bandwidth, rolloff, flatness, contrast (11 channels).
   - pitch_: pyin F0 (250-1000 Hz) stats & voiced ratio on active frames.
   - shape_: duration, active duration, active ratio, burst count, mean burst length, mean gap.
4. Pooled over ACTIVE frames using 9 statistics: mean, std, skew, kurtosis, min, max, p10, p50, p90.
5. Saves infant_cry_dataset/features_v3/real_X.npy, aug_X.npy, feature_names.json.
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
OUT_DIR = ROOT / "features_v3"
CACHE_DIR = OUT_DIR / "_cache"

SR = 22050
MAX_LEN_SEC = 8.0
MAX_SAMPLES = int(MAX_LEN_SEC * SR)
HOP_LENGTH = 256
FRAME_LENGTH = 1024
CHUNK_SIZE = 32
VAD_DB_THRESH = -35.0  # relative to 95th percentile RMS
MIN_ACTIVE_SEC = 0.3
MIN_ACTIVE_FRAMES = int(round(MIN_ACTIVE_SEC * SR / HOP_LENGTH))


def pool_active(M):
    """Pool 9 statistics over active frames for 2D array M (channels, frames)."""
    if M.ndim == 1:
        M = M.reshape(1, -1)
    if M.shape[1] == 0:
        return np.zeros(M.shape[0] * 9, dtype=np.float32)

    mean = M.mean(axis=1)
    std = M.std(axis=1)
    sk = skew(M, axis=1)
    kt = kurtosis(M, axis=1)
    mn = M.min(axis=1)
    mx = M.max(axis=1)
    p10 = np.percentile(M, 10, axis=1)
    p50 = np.percentile(M, 50, axis=1)
    p90 = np.percentile(M, 90, axis=1)

    return np.concatenate([mean, std, sk, kt, mn, mx, p10, p50, p90]).astype(np.float32)


# Global process-level cache for Scattering1D
_SCAT_CACHE = {}


def get_scattering(win_samples):
    if win_samples not in _SCAT_CACHE:
        _SCAT_CACHE[win_samples] = Scattering1D(J=8, shape=win_samples, Q=8)
    return _SCAT_CACHE[win_samples]


def extract_clip_v3(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    if len(y) > MAX_SAMPLES:
        y = y[:MAX_SAMPLES]
    total_dur = len(y) / SR

    # VAD RMS check
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    p95_rms = np.percentile(rms, 95) if len(rms) else 1e-6
    db_rel = 20 * np.log10(np.maximum(rms, 1e-8) / np.maximum(p95_rms, 1e-8))

    active_mask = db_rel > VAD_DB_THRESH
    if active_mask.sum() < MIN_ACTIVE_FRAMES:
        active_mask = np.ones(len(rms), dtype=bool)

    # Active audio reconstruction
    active_frame_indices = np.where(active_mask)[0]
    active_samples = []
    for idx in active_frame_indices:
        start = idx * HOP_LENGTH
        end = min(len(y), start + HOP_LENGTH)
        active_samples.extend(y[start:end])
    active_y = np.array(active_samples, dtype=np.float32) if len(active_samples) else y

    active_dur = len(active_y) / SR
    active_ratio = active_dur / max(total_dur, 1e-4)

    # Bursts & gaps statistics
    diffs = np.diff(np.pad(active_mask.astype(int), (1, 1)))
    burst_starts = np.where(diffs == 1)[0]
    burst_ends = np.where(diffs == -1)[0]
    num_bursts = len(burst_starts)
    burst_lens = [(e - s) * HOP_LENGTH / SR for s, e in zip(burst_starts, burst_ends)]
    mean_burst_len = float(np.mean(burst_lens)) if burst_lens else 0.0

    if num_bursts > 1:
        gaps = [(burst_starts[i] - burst_ends[i-1]) * HOP_LENGTH / SR for i in range(1, num_bursts)]
        mean_gap = float(np.mean(gaps))
    else:
        mean_gap = 0.0

    shape_feats = np.array([total_dur, active_dur, active_ratio, float(num_bursts), mean_burst_len, mean_gap], dtype=np.float32)

    # 1. Kymatio Scattering on 2 s active windows
    win_samples = int(2.0 * SR)
    hop_win = int(1.0 * SR)

    sc_input = active_y
    if len(sc_input) < win_samples:
        sc_input = np.pad(sc_input, (0, win_samples - len(sc_input)), mode="reflect")

    S = get_scattering(win_samples)
    sc_window_feats = []

    for w_start in range(0, len(sc_input) - win_samples + 1, hop_win):
        w_audio = sc_input[w_start:w_start + win_samples]
        sx = np.log1p(S(w_audio))
        sc_window_feats.append(sx.mean(axis=-1))

    if sc_window_feats:
        scat_feats = np.mean(sc_window_feats, axis=0).astype(np.float32)
    else:
        scat_feats = np.zeros(S(np.zeros(win_samples, dtype=np.float32)).shape[0], dtype=np.float32)

    # 2. MFCC + delta + delta2
    m = librosa.feature.mfcc(y=active_y, sr=SR, n_mfcc=13, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    m_full = np.vstack([m, librosa.feature.delta(m), librosa.feature.delta(m, order=2)])
    mfcc_feats = pool_active(m_full)

    # 3. Spectral Descriptors
    cent = librosa.feature.spectral_centroid(y=active_y, sr=SR, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    band = librosa.feature.spectral_bandwidth(y=active_y, sr=SR, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    roll = librosa.feature.spectral_rolloff(y=active_y, sr=SR, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    flat = librosa.feature.spectral_flatness(y=active_y, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    cont = librosa.feature.spectral_contrast(y=active_y, sr=SR, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    spec_full = np.vstack([cent, band, roll, flat, cont])
    spec_feats = pool_active(spec_full)

    # 4. Fast Yin F0 statistics (250 - 1000 Hz)
    f0 = librosa.yin(active_y, fmin=250, fmax=1000, sr=SR, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    f0_valid = f0[(f0 >= 250) & (f0 <= 1000) & ~np.isnan(f0)]
    voiced_ratio = float(len(f0_valid) / max(len(f0), 1))

    if len(f0_valid) > 0:
        f0_stats = np.array([f0_valid.mean(), f0_valid.std(), f0_valid.min(), f0_valid.max()], dtype=np.float32)
    else:
        f0_stats = np.zeros(4, dtype=np.float32)
    pitch_feats = np.concatenate([f0_stats, [voiced_ratio]]).astype(np.float32)

    # Combine all blocks
    all_feats = np.nan_to_num(np.concatenate([scat_feats, mfcc_feats, spec_feats, pitch_feats, shape_feats]))
    return all_feats.astype(np.float32)


def process_chunk_v3(paths, cache_file):
    if cache_file.exists():
        return np.load(cache_file)

    chunk_feats = [extract_clip_v3(p) for p in paths]
    arr = np.array(chunk_feats, dtype=np.float32)

    tmp_file = cache_file.with_suffix(".tmp.npy")
    np.save(tmp_file, arr)
    tmp_file.replace(cache_file)
    return arr


def extract_set_v3(name, paths):
    print(f"\n--- Extracting features_v3 for '{name}' ({len(paths)} files) ---")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    n_jobs = min(6, max(1, (os.cpu_count() or 4) - 2))

    chunks = [paths[i:i + CHUNK_SIZE] for i in range(0, len(paths), CHUNK_SIZE)]
    cache_files = [CACHE_DIR / f"{name}_v3_chunk_{i:04d}.npy" for i in range(len(chunks))]

    cached_count = sum(f.exists() for f in cache_files)
    print(f"Total chunks: {len(chunks)} ({cached_count} cached). Using {n_jobs} CPU cores.")

    t0 = time.perf_counter()
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_chunk_v3)(chunk, cfile)
        for chunk, cfile in zip(tqdm(chunks, desc=f"v3 [{name}]"), cache_files)
    )
    elapsed = time.perf_counter() - t0

    X = np.concatenate(results, axis=0).astype(np.float32)
    out_path = OUT_DIR / f"{name}_X.npy"
    np.save(out_path, X)
    print(f"Saved {name}_X.npy shape {X.shape} to {out_path} (Time: {elapsed:.1f}s)")
    return X


def main():
    print("======================================================================")
    print(" STAGE 23A: FEATURE EXTRACTION FOR features_v3 (VAD & True Length)")
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

    real_X = extract_set_v3("real", real_paths)
    aug_X = extract_set_v3("aug", aug_paths)

    # Build feature names
    sample_feat = extract_clip_v3(real_paths[0])
    n_scat = len(sample_feat) - (39 * 9) - (11 * 9) - 5 - 6

    names = [f"scat_{i}" for i in range(n_scat)] + \
            [f"mfcc_{i}" for i in range(39 * 9)] + \
            [f"spec_{i}" for i in range(11 * 9)] + \
            [f"pitch_{i}" for i in range(5)] + \
            ["shape_total_dur", "shape_active_dur", "shape_active_ratio", "shape_num_bursts", "shape_mean_burst_len", "shape_mean_gap"]

    assert len(names) == real_X.shape[1], f"Feature names count mismatch: {len(names)} vs {real_X.shape[1]}"
    with open(OUT_DIR / "feature_names.json", "w") as f:
        json.dump(names, f, indent=2)

    print("\nFeature names saved to:", OUT_DIR / "feature_names.json")
    print(f"Total features_v3 per clip: {real_X.shape[1]}")
    print(f"Block breakdown: scat={n_scat}, mfcc={39*9}, spec={11*9}, pitch=5, shape=6")


if __name__ == "__main__":
    main()
