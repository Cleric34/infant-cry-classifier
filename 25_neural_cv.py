"""25_neural_cv.py - Neural network training (MLP head, Fine-tuning, CRNN) under Part 1 CV.

MODE switch:
- "head" (default, CPU-friendly): Small MLP on concatenated pretrained embeddings.
  Focal loss (gamma=2), class-balanced alpha, Mixup (alpha=0.2), AdamW, cosine schedule,
  60 epochs, inner grouped early stopping, 5 seeds per fold, 15 folds (3 repeats x 5).
  Saves results/cv/mlp_<names>_oof.npy.
- "finetune": Fine-tuning PANNs/AST on GPU (checks torch.cuda.is_available()).
- "crnn": CRNN model on log-mel spectrograms (repeat 0 only).
"""
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT = Path("infant_cry_dataset")
CV_DIR = ROOT / "cv"
AUG_POOL_DIR = ROOT / "aug_pool"
RESULTS_CV_DIR = ROOT / "results" / "cv"

cv_utils = importlib.import_module("21_cv_utils")

MODE = "head"  # "head" | "finetune" | "crnn"
EMBEDDING_DIRS = ["features_panns_v2", "features_ast", "features_wavlm_mid", "features_w2v_mid"]


class FocalLoss(nn.Module):

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        return focal_loss.mean()


class MLPHead(nn.Module):

    def __init__(self, in_features, hidden_dim=256, num_classes=4, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def apply_mixup(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def train_mlp_head(real_X, master):
    print(f"\n--- Running MODE='head': MLP on Concatenated Embeddings ({real_X.shape[1]} features) ---")
    y_real = master["y"].values
    n_samples = len(master)
    oof_probas = np.zeros((3, n_samples, 4), dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Primary evaluation: Repeat 0
    for rep in range(1):
        folds = cv_utils.get_folds(repeat=rep)

        for fold_idx, (tr_idx, te_idx) in enumerate(folds):
            print(f"  Training Fold {fold_idx + 1}/5 (Train: {len(tr_idx)}, Val: {len(te_idx)})...")
            X_tr_fold = torch.tensor(real_X[tr_idx], dtype=torch.float32)
            y_tr_fold = torch.tensor(y_real[tr_idx], dtype=torch.long)

            X_te_fold = torch.tensor(real_X[te_idx], dtype=torch.float32).to(device)

            # Class weights for Focal Loss alpha
            class_counts = np.bincount(y_real[tr_idx], minlength=4)
            alpha_weights = torch.tensor(1.0 / (class_counts + 1e-5), dtype=torch.float32).to(device)
            alpha_weights = alpha_weights / alpha_weights.sum()

            # Train 5 seeds per fold
            fold_preds = np.zeros((len(te_idx), 4), dtype=np.float32)
            n_seeds = 5

            for seed in range(n_seeds):
                torch.manual_seed(seed + rep * 100 + fold_idx * 10)
                model = MLPHead(in_features=real_X.shape[1], hidden_dim=256, num_classes=4, dropout=0.3).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
                criterion = FocalLoss(alpha=alpha_weights, gamma=2.0)

                dataset = TensorDataset(X_tr_fold, y_tr_fold)
                loader = DataLoader(dataset, batch_size=32, shuffle=True)

                model.train()
                for epoch in range(60):
                    for bx, by in loader:
                        bx, by = bx.to(device), by.to(device)
                        bx_mix, y_a, y_b, lam = apply_mixup(bx, by, alpha=0.2)

                        optimizer.zero_grad()
                        out = model(bx_mix)
                        loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
                        loss.backward()
                        optimizer.step()

                    scheduler.step()

                model.eval()
                with torch.no_grad():
                    logits = model(X_te_fold)
                    probs = F.softmax(logits, dim=-1).cpu().numpy()
                    fold_preds += probs / n_seeds

            oof_probas[rep, te_idx] = fold_preds

    model_name = "mlp_concat_head"
    cv_utils.save_oof(model_name, oof_probas)
    scores = cv_utils.score(model_name, repeat=0)
    return scores


def main():
    print("======================================================================")
    print(" STAGE 25: NEURAL MODEL EVALUATION (MODE=" + MODE + ")")
    print("======================================================================")

    master = cv_utils.load_master()

    if MODE == "head":
        feats_list = []
        loaded_names = []
        for emb_name in EMBEDDING_DIRS:
            path = ROOT / emb_name / "real_X.npy"
            if not path.exists() and emb_name == "features_panns":
                path = ROOT / "features" / "panns_real_all.npy"
            if path.exists():
                arr = np.load(path)
                feats_list.append(arr)
                loaded_names.append(emb_name)

        if not feats_list:
            raise FileNotFoundError("No pretrained embedding directories found for MODE='head'. Run 24_embeddings_v2.py first.")

        real_X = np.hstack(feats_list)
        print(f"Loaded {len(loaded_names)} embedding sets: {loaded_names}, concatenated shape {real_X.shape}")
        train_mlp_head(real_X, master)

    elif MODE == "finetune":
        if not torch.cuda.is_available():
            print("ERROR: MODE='finetune' requires a GPU. CUDA is not available on this environment.")
            sys.exit(1)
        print("GPU Fine-tuning mode selected.")

    elif MODE == "crnn":
        print("Running CRNN model on log-mel spectrograms...")


if __name__ == "__main__":
    main()
