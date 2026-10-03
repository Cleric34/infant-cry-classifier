"""
STAGE 1 — DTCWT DENOISING

DTCWT gives near shift-invariance (unlike a plain DWT) by running two
wavelet trees offset from each other so their output magnitude is stable
even if the cry starts a few samples earlier/later in the clip. Soft-
thresholds each complex highpass subband using a robust MAD-based universal
threshold (Donoho-Johnstone), shrinking magnitude while preserving phase.

Handles odd-length input (dtcwt.Transform1d requires even length) by
edge-padding one sample and trimming back after reconstruction — verified
against synthetic odd-length files.

Also writes duration_manifest.json: per-file and summary duration stats
for the denoised set, used downstream by 05_ to justify the fixed-length
cutoff and to answer "what's the average length / cutoff" directly from
data.
"""

import os
import glob
import json
import numpy as np
import librosa
import soundfile as sf
import dtcwt

SRC = "infant_cry_dataset/audio"
DST = "infant_cry_dataset/audio_denoised"
NLEVELS = 4
SEED = 42

# Compatibility for newer NumPy versions used by some dtcwt releases.
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)


def dtcwt_denoise(x, nlevels=NLEVELS):
    """DTCWT denoising using robust noise estimation + soft thresholding."""
    orig_len = len(x)
    if orig_len % 2 != 0:
        x = np.pad(x, (0, 1), mode="edge")

    xfm = dtcwt.Transform1d()
    coeffs = xfm.forward(x, nlevels=nlevels)

    highpasses = list(coeffs.highpasses)

    for i, band in enumerate(highpasses):
        # Robust noise estimate based on the median absolute deviation.
        sigma = np.median(np.abs(band)) / 0.6745
        threshold = sigma * np.sqrt(2.0 * np.log(max(band.size, 2)))

        magnitude = np.abs(band)
        shrink = np.maximum(
            1.0 - threshold / np.maximum(magnitude, 1e-12),
            0.0,
        )
        highpasses[i] = band * shrink

    coeffs.highpasses = tuple(highpasses)
    y = xfm.inverse(coeffs)
    return np.asarray(y[:orig_len], dtype=np.float32)


def main():
    if not os.path.isdir(SRC):
        raise FileNotFoundError(f"Source dataset not found: {SRC}")

    os.makedirs(DST, exist_ok=True)

    duration_records = []

    for cls in sorted(os.listdir(SRC)):
        src_cls = os.path.join(SRC, cls)
        if not os.path.isdir(src_cls):
            continue

        dst_cls = os.path.join(DST, cls)
        os.makedirs(dst_cls, exist_ok=True)
        files = sorted(glob.glob(os.path.join(src_cls, "*.wav")))

        for i, path in enumerate(files, 1):
            x, sr = librosa.load(path, sr=None, mono=True)
            y = dtcwt_denoise(x)
            out = os.path.join(dst_cls, os.path.basename(path))
            sf.write(out, y, sr)

            duration_records.append({
                "class": cls,
                "file": os.path.basename(path),
                "sampling_rate_hz": int(sr),
                "n_samples": int(len(x)),
                "duration_seconds": float(len(x) / sr),
            })

            if i % 100 == 0 or i == len(files):
                print(f"{cls}: {i}/{len(files)}")

    # Duration manifest: used downstream to justify the fixed-length
    # cutoff/average applied before feature extraction (see script 05),
    # and to answer "what is the average length / cutoff limit" directly
    # from data rather than from memory.
    durations = np.array([r["duration_seconds"] for r in duration_records], dtype=np.float64)
    duration_summary = {
        "n_files": int(len(durations)),
        "mean_seconds": float(durations.mean()) if len(durations) else None,
        "median_seconds": float(np.median(durations)) if len(durations) else None,
        "std_seconds": float(durations.std()) if len(durations) else None,
        "min_seconds": float(durations.min()) if len(durations) else None,
        "max_seconds": float(durations.max()) if len(durations) else None,
        "p95_seconds": float(np.percentile(durations, 95)) if len(durations) else None,
    }
    with open(os.path.join(DST, "duration_manifest.json"), "w") as f:
        json.dump({"per_file": duration_records, "summary": duration_summary}, f, indent=2)

    print("\nDuration summary (seconds):")
    print(json.dumps(duration_summary, indent=2))
    print("DTCWT denoising completed.")


if __name__ == "__main__":
    main()
