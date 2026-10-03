"""24_embeddings_v2.py - Extract pretrained embeddings at true clip length.

Supports:
1. features_panns_v2 (PANNs CNN14 2048-d, 32 kHz)
2. features_clap (laion/clap-htsat-unfused 512-d, 48 kHz)
3. features_ast (MIT/ast-finetuned-audioset 768-d mean+std=1536-d, 16 kHz)
4. features_wavlm_mid (microsoft/wavlm-base-plus layers 4-8 mean+std=1536-d, 16 kHz)
5. features_w2v_mid (facebook/wav2vec2-base layers 4-8 mean+std=1536-d, 16 kHz)

Resumable, chunked, optional per model.
"""
import json
import os
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path("infant_cry_dataset")
CV_DIR = ROOT / "cv"
AUG_POOL_DIR = ROOT / "aug_pool"
MAX_SEC = 8.0
BATCH_SIZE = 16


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_raw_audio(path, target_sr):
    y, _ = librosa.load(path, sr=target_sr, mono=True)
    max_samples = int(MAX_SEC * target_sr)
    if len(y) > max_samples:
        y = y[:max_samples]
    return y.astype(np.float32)


# ---------------- 1. PANNs ----------------
def build_panns_embedder(device):
    from panns_inference import AudioTagging
    tagger = AudioTagging(checkpoint_path=None, device=device)

    def embed(batch):
        # Pad batch to max length in batch
        max_len = max(len(x) for x in batch)
        padded = np.stack([np.pad(x, (0, max_len - len(x))) for x in batch])
        _, emb = tagger.inference(padded)
        return np.asarray(emb, dtype=np.float32)
    return embed, 32000


# ---------------- 2. CLAP ----------------
def build_clap_embedder(device):
    from transformers import ClapAudioModelWithProjection, ClapFeatureExtractor
    name = "laion/clap-htsat-unfused"
    fe = ClapFeatureExtractor.from_pretrained(name)
    net = ClapAudioModelWithProjection.from_pretrained(name).to(device).eval()

    def embed(batch):
        inp = fe(list(batch), sampling_rate=48000, return_tensors="pt").to(device)
        with torch.no_grad():
            out = net(**inp).audio_embeds
        return out.cpu().numpy().astype(np.float32)
    return embed, 48000


# ---------------- 3. AST ----------------
def build_ast_embedder(device):
    from transformers import AutoFeatureExtractor, AutoModel
    name = "MIT/ast-finetuned-audioset-10-10-0.4593"
    fe = AutoFeatureExtractor.from_pretrained(name)
    net = AutoModel.from_pretrained(name).to(device).eval()

    def embed(batch):
        inp = fe(list(batch), sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad():
            h = net(**inp).last_hidden_state  # (B, T, 768)
        mean_h = h.mean(dim=1)
        std_h = h.std(dim=1)
        return torch.cat([mean_h, std_h], dim=1).cpu().numpy().astype(np.float32)
    return embed, 16000


# ---------------- 4. WavLM Mid-Layers (4 to 8) ----------------
def build_wavlm_mid_embedder(device):
    from transformers import AutoFeatureExtractor, WavLMModel
    name = "microsoft/wavlm-base-plus"
    fe = AutoFeatureExtractor.from_pretrained(name)
    net = WavLMModel.from_pretrained(name, output_hidden_states=True, use_safetensors=True).to(device).eval()

    def embed(batch):
        inp = fe(list(batch), sampling_rate=16000, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            res = net(**inp)
            hidden = res.hidden_states  # Tuple of layers
            mid_layers = torch.stack(hidden[4:9], dim=0).mean(dim=0)  # (B, T, 768)
        mean_h = mid_layers.mean(dim=1)
        std_h = mid_layers.std(dim=1)
        return torch.cat([mean_h, std_h], dim=1).cpu().numpy().astype(np.float32)
    return embed, 16000


# ---------------- 5. Wav2Vec2 Mid-Layers (4 to 8) ----------------
def build_w2v_mid_embedder(device):
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
    name = "facebook/wav2vec2-base"
    fe = Wav2Vec2FeatureExtractor.from_pretrained(name)
    net = Wav2Vec2Model.from_pretrained(name, output_hidden_states=True, use_safetensors=True).to(device).eval()

    def embed(batch):
        inp = fe(list(batch), sampling_rate=16000, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            res = net(**inp)
            hidden = res.hidden_states
            mid_layers = torch.stack(hidden[4:9], dim=0).mean(dim=0)  # (B, T, 768)
        mean_h = mid_layers.mean(dim=1)
        std_h = mid_layers.std(dim=1)
        return torch.cat([mean_h, std_h], dim=1).cpu().numpy().astype(np.float32)
    return embed, 16000


MODELS = {
    "features_panns_v2": build_panns_embedder,
    "features_ast": build_ast_embedder,
    "features_wavlm_mid": build_wavlm_mid_embedder,
    "features_w2v_mid": build_w2v_mid_embedder,
}


def extract_for_model(model_name, builder, device, master, aug_manifest):
    out_dir = ROOT / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    real_x_path = out_dir / "real_X.npy"
    aug_x_path = out_dir / "aug_X.npy"

    if real_x_path.exists() and aug_x_path.exists():
        print(f"[{model_name}] Already fully extracted. Skipping.")
        return

    print(f"\n======================================================================")
    print(f" EXTRACTING EMBEDDINGS: {model_name}")
    print(f"======================================================================")

    embed_fn, sr = builder(device)

    # 1. Real files
    real_paths = master["path"].tolist()
    real_feats = []
    print(f"Extracting real files ({len(real_paths)})...")
    for i in tqdm(range(0, len(real_paths), BATCH_SIZE), desc=f"{model_name} [real]"):
        b_paths = real_paths[i:i + BATCH_SIZE]
        b_audio = [load_raw_audio(p, sr) for p in b_paths]
        real_feats.append(embed_fn(b_audio))

    real_X = np.concatenate(real_feats, axis=0).astype(np.float32)
    np.save(real_x_path, real_X)
    print(f"Saved real_X.npy shape {real_X.shape}")

    # 2. Aug files
    if len(aug_manifest) > 0:
        aug_paths = aug_manifest["path"].tolist()
        aug_feats = []
        print(f"Extracting aug files ({len(aug_paths)})...")
        for i in tqdm(range(0, len(aug_paths), BATCH_SIZE), desc=f"{model_name} [aug]"):
            b_paths = aug_paths[i:i + BATCH_SIZE]
            b_audio = [load_raw_audio(p, sr) for p in b_paths]
            aug_feats.append(embed_fn(b_audio))

        aug_X = np.concatenate(aug_feats, axis=0).astype(np.float32)
        np.save(aug_x_path, aug_X)
        print(f"Saved aug_X.npy shape {aug_X.shape}")
    else:
        np.save(aug_x_path, np.array([], dtype=np.float32))


def main():
    print("======================================================================")
    print(" STAGE 24A: PRETRAINED EMBEDDINGS V2 EXTRACTION")
    print("======================================================================")

    device = get_device()
    print(f"Device: {device}")

    master = pd.read_csv(CV_DIR / "master.csv")
    aug_manifest_path = AUG_POOL_DIR / "aug_manifest.csv"
    aug_manifest = pd.read_csv(aug_manifest_path) if aug_manifest_path.exists() else pd.DataFrame()

    loaded_models = []
    for model_name, builder in MODELS.items():
        try:
            extract_for_model(model_name, builder, device, master, aug_manifest)
            loaded_models.append(model_name)
        except Exception as e:
            print(f"WARNING: Failed to extract {model_name}: {e}")

    print("\nSuccessfully processed embedding models:", loaded_models)


if __name__ == "__main__":
    main()
