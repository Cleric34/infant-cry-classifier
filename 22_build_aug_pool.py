"""22_build_aug_pool.py - Generate deterministic augmented pool for minority classes.

For every real non-Needs file in master.csv (or Needs if NEEDS_COPIES > 0), creates
exactly 2 deterministic augmented copies in infant_cry_dataset/aug_pool/<Class>/.
Writes infant_cry_dataset/aug_pool/aug_manifest.csv.
"""
import hashlib
import json
import os
import random
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

ROOT = Path("infant_cry_dataset")
DENOISED_DIR = ROOT / "audio_denoised"
CV_DIR = ROOT / "cv"
AUG_POOL_DIR = ROOT / "aug_pool"

NEEDS_COPIES = 0  # Default 0. Can be set > 0 if desired.
NON_NEEDS_COPIES = 2

SHIFT_MAX_SECONDS = 0.08
STRETCH_MIN = 0.92
STRETCH_MAX = 1.08
GAIN_MIN = 0.80
GAIN_MAX = 1.20
NOISE_MIN_REL_RMS = 0.001
NOISE_MAX_REL_RMS = 0.003


def peak_normalize(x):
    peak = np.max(np.abs(x)) if len(x) else 0.0
    return (x / peak if peak > 1e-8 else x).astype(np.float32)


def add_noise(x, rng):
    rms = np.sqrt(np.mean(x ** 2) + 1e-12)
    scale = rng.uniform(NOISE_MIN_REL_RMS, NOISE_MAX_REL_RMS)
    return x + rng.normal(0.0, scale * max(rms, 1e-4), size=x.shape)


def time_shift(x, sr, rng):
    max_shift = max(1, int(SHIFT_MAX_SECONDS * sr))
    shift = rng.integers(-max_shift, max_shift + 1)
    return np.roll(x, shift)


def time_stretch(x, sr, rng):
    rate = rng.uniform(STRETCH_MIN, STRETCH_MAX)
    y = librosa.effects.time_stretch(y=x, rate=float(rate))
    if len(y) > len(x):
        start = rng.integers(0, len(y) - len(x) + 1)
        y = y[start:start + len(x)]
    else:
        y = np.pad(y, (0, len(x) - len(y)))
    return y


def gain_scale(x, rng):
    return x * rng.uniform(GAIN_MIN, GAIN_MAX)


def apply_ops(x, sr, ops, rng):
    y = x.copy()
    for op in ops:
        if op == "noise":
            y = add_noise(y, rng)
        elif op == "shift":
            y = time_shift(y, sr, rng)
        elif op == "stretch":
            y = time_stretch(y, sr, rng)
        elif op == "gain":
            y = gain_scale(y, rng)
    return peak_normalize(y)


def main():
    print("======================================================================")
    print(" STAGE 22A: BUILDING DETERMINISTIC AUGMENTATION POOL")
    print("======================================================================")

    master_path = CV_DIR / "master.csv"
    if not master_path.exists():
        raise FileNotFoundError(f"Master manifest not found at {master_path}. Run 21_build_cv.py first.")

    master = pd.read_csv(master_path)
    AUG_POOL_DIR.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    # Op combinations for copy 1 and copy 2 to ensure diversity
    copy_op_combos = {
        1: ["shift", "gain"],
        2: ["stretch", "noise"],
    }

    for _, row in tqdm(master.iterrows(), total=len(master), desc="Generating Aug Pool"):
        key = row["key"]
        cls = row["class"]
        y_val = row["y"]
        src_path = Path(row["path"])

        num_copies = NEEDS_COPIES if cls == "needs" else NON_NEEDS_COPIES
        if num_copies <= 0:
            continue

        x, sr = librosa.load(src_path, sr=None, mono=True)
        base_name = src_path.stem

        cls_aug_dir = AUG_POOL_DIR / cls
        cls_aug_dir.mkdir(parents=True, exist_ok=True)

        for copy_idx in range(1, num_copies + 1):
            # Deterministic seed per clip and copy_idx
            seed_str = f"{key}_copy{copy_idx}"
            seed_int = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed_int)

            ops = copy_op_combos.get(copy_idx, ["shift", "stretch", "gain", "noise"][:copy_idx])
            ops_str = "-".join(ops)

            aug_x = apply_ops(x, sr, ops, rng)

            out_filename = f"aug_c{copy_idx}_{ops_str}_{base_name}.wav"
            out_path = cls_aug_dir / out_filename
            sf.write(out_path, aug_x, sr)

            manifest_rows.append({
                "path": str(out_path.as_posix()),
                "source_key": key,
                "copy_idx": copy_idx,
                "class": cls,
                "y": y_val,
                "ops": ops_str,
            })

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_csv = AUG_POOL_DIR / "aug_manifest.csv"
    manifest_df.to_csv(manifest_csv, index=False)

    print(f"\nGenerated {len(manifest_df)} total augmented clips.")
    print("Augmentation pool manifest saved to:", manifest_csv)
    print("\nAugmented clips per class:")
    print(manifest_df["class"].value_counts())


if __name__ == "__main__":
    main()
