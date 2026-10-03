# Infant Cry Classification Dataset (4 Classes)

A cleaned, deduplicated, speaker-aware dataset of 4,871 infant cry recordings, organized into 4 acoustically distinct classes for training cry-type classifiers.

## 📊 Dataset Statistics

| Class | Description | Files |
|-------|-------------|-------|
| `scared` | Sudden startle / fear cries (sharp, high-pitched) | 684 |
| `needs` | General fussing — hungry, tired, or uncomfortable | 3,103 |
| `physical_pain` | Pain-related cries (e.g., colic — rhythmic, intense) | 774 |
| `burping` | Needs to expel air after feeding (short bursts) | 310 |
| **Total** | | **4,871** |

## 🎵 Audio Specifications

- **Sample rate**: 16 kHz
- **Format**: WAV (16-bit PCM)
- **Channels**: Mono
- **Duration**: variable (most clips 3–10 seconds)
- **Normalization**: −23 LUFS loudness-normalized
- **Preprocessing**: high-pass filtered at 50 Hz, voice-activity trimmed

## 📁 Folder Structure
audio/
├── scared/           ← 684 files
├── needs/            ← 3,103 files
├── physical_pain/    ← 774 files
└── burping/          ← 310 files

Folder names serve as class labels — compatible with `torchaudio.datasets.DatasetFolder`, Keras `audio_dataset_from_directory`, and similar APIs.

## 🧪 Data Quality

This dataset was built with deliberate attention to common pitfalls:

- **Cross-dataset deduplication**: 1,297 near-duplicate clips were removed via MFCC fingerprint analysis. Public infant cry corpora often share underlying recordings across "different" datasets — this is corrected here.
- **Speaker-aware curation**: Where speaker IDs could be extracted from filenames, the dataset was checked for speaker overlap.
- **Standardized preprocessing**: All audio passes through the same pipeline (resampling, high-pass filter, VAD, loudness normalization).

## ⚠️ Important Notes for Users

### Creating splits
This release does **not** provide pre-defined train/validation/test splits, allowing users to apply their own splitting strategy. If you build a split, please consider:
- **Speaker-aware splitting** to avoid leakage (the same baby in both train and test inflates results)
- **Stratified splitting** by class to handle imbalance

### Class taxonomy
The `needs` class is intentionally broad. It combines what other datasets label separately as "hungry," "tired," and "discomfort." These categories are not reliably separable from short audio clips alone — experienced parents typically use context (time since last feed, sleep schedule) rather than acoustic features. Merging them produces a more honest and accurate label.

### Known biases
- The dataset combines 3 source corpora with imbalanced contributions. The dominant source is over-represented relative to minority sources.
- Real-world recordings from different microphones, rooms, or families may perform differently from in-corpus evaluation.
- This is **not a medical dataset**. Models trained on it should not be used for clinical diagnosis.

## 🔬 Source Datasets

This dataset is a cleaned, deduplicated, and re-labeled combination of three public corpora:

1. **Donate-a-Cry Corpus** (warcoder/infant-cry-audio-corpus on Kaggle)
2. **Baby Crying Dataset** (mennaahmed23/baby-crying-dataset on Kaggle)
3. **Baby Crying Sounds Dataset** (mennaahmed23/baby-crying-sounds-dataset on Kaggle)

Original sources retain their respective licenses. This combined dataset is released under CC BY-NC 4.0.

## 📜 License

This dataset is released under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**.

You are free to:
- ✅ Share — copy and redistribute the material in any medium or format
- ✅ Adapt — remix, transform, and build upon the material

Under the following terms:
- 📝 **Attribution** — You must give appropriate credit
- 🚫 **NonCommercial** — You may not use the material for commercial purposes

See `LICENSE.txt` for the full license text.

## 📖 Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{eletriby2026_infant_cry,
  author       = {Karim A. Eletriby},
  title        = {Infant Cry Classification Dataset (4 Classes)},
  year         = {2026},
  publisher    = {Kaggle},
  note         = {Faculty of Artificial Intelligence, Delta University},
  howpublished = {Kaggle Datasets}
}
```

## 🙏 Acknowledgments

- The original contributors to the Donate-a-Cry corpus and the Kaggle baby-crying datasets
- The HuggingFace Transformers and PyTorch communities
- Faculty of Artificial Intelligence, Delta University

## 📧 Contact

For questions, corrections, or research collaborations, please contact through the Kaggle profile or leave a comment on this dataset.

---

**Author**: Karim A. Eletriby
**Affiliation**: Faculty of Artificial Intelligence, Delta University
**Version**: 2.0 (4-class clean release)
