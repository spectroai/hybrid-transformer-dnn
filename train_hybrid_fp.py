# ============================================================
# train_hybrid_fp.py
# Hybrid (Binned MLP + Transformer Peaks) fingerprint prediction
# Paper-grade pipeline: aligned split + reproducibility + clean ckpt
# ============================================================

import ast
import argparse
import json
import os
import time
import copy
import random
from collections import defaultdict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    hamming_loss, precision_score, recall_score
)

# ---------------------------
# 0) Reproducibility
# ---------------------------
SEED = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# 1) Global params (match your method)
# ---------------------------
mz_min, mz_max, bin_size = 20, 1200, 1
bin_edges = np.arange(mz_min, mz_max + bin_size, bin_size)  # inclusive end
BIN_DIM = 1180  # 20-1200 with 1 Da => 1180 bins
MAX_LEN = 300   # top-L peaks for transformer
FP_DIM = 881

BATCH_SIZE = 64
NUM_EPOCHS = 40
PATIENCE = 4
MAX_GRAD_NORM = 5.0

# grid search 
LR_GRID = [2e-4]
WD_GRID = [0.0]
DROPOUT_GRID = [0.1]
THRESH_GRID = [0.40]

NUM_WORKERS = 4
PIN_MEMORY = True

SAVE_DIR = "checkpoints"

# ---------------------------
# 2) Data utilities
# ---------------------------
def _maybe_parse_list(x):
    if pd.isna(x):
        return []
    if isinstance(x, str):
        return ast.literal_eval(x)
    return x

def spectrum_to_vector(mz_values, intensities):
    mz_values = np.asarray(_maybe_parse_list(mz_values), dtype=np.float32)
    intensities = np.asarray(_maybe_parse_list(intensities), dtype=np.float32)

    mask = (mz_values >= mz_min) & (mz_values <= mz_max)
    mz_values = mz_values[mask]
    intensities = intensities[mask]

    vec, _ = np.histogram(mz_values, bins=bin_edges, weights=intensities)
    vec = vec.astype(np.float32)

    mx = vec.max() if vec.size > 0 else 0.0
    if mx > 0:
        vec /= mx
    return vec

def prepare_transformer_input(mz_values, intensities, max_len=MAX_LEN):
    mz_values = np.asarray(_maybe_parse_list(mz_values), dtype=np.float32)
    intensities = np.asarray(_maybe_parse_list(intensities), dtype=np.float32)

    mask = (mz_values >= mz_min) & (mz_values <= mz_max)
    mz_values = mz_values[mask]
    intensities = intensities[mask]

    # normalize intensities per spectrum
    mx = intensities.max() if intensities.size > 0 else 1.0
    if mx == 0:
        mx = 1.0
    intensities = intensities / mx

    # top-k by intensity then pad
    L = len(mz_values)
    if L > max_len:
        top = np.argsort(intensities)[-max_len:]
        mz_values = mz_values[top]
        intensities = intensities[top]
    else:
        pad = max_len - L
        mz_values = np.pad(mz_values, (0, pad), constant_values=0)
        intensities = np.pad(intensities, (0, pad), constant_values=0)

    return mz_values.astype(np.float32), intensities.astype(np.float32)

# ---------------------------
# 3) Dataset
# ---------------------------
class ExperimentalDataset(Dataset):
    def __init__(self, X, Y, mz, inten, prec):
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)
        self.mz = mz.astype(np.float32)
        self.inten = inten.astype(np.float32)
        self.prec = prec.astype(np.int64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),
            torch.from_numpy(self.Y[idx]),
            torch.from_numpy(self.mz[idx]),
            torch.from_numpy(self.inten[idx]),
            torch.tensor(self.prec[idx], dtype=torch.long),
        )

def remove_zero_intensity_samples(ds: Dataset):
    keep = []
    for i in range(len(ds)):
        _, _, _, inten, _ = ds[i]
        if not torch.all(inten == 0):
            keep.append(i)
    return Subset(ds, keep)

def build_molecule_groups(df: pd.DataFrame):
    """
    Build molecule-level grouping labels to prevent train/val/test leakage.
    Priority: inchikey > molecule_id > normalized-smiles.
    When only smiles is available, try RDKit normalization first.
    """
    if "inchikey" in df.columns:
        groups = df["inchikey"].astype(str).fillna("").values
        source = "inchikey"
    elif "molecule_id" in df.columns:
        groups = df["molecule_id"].astype(str).fillna("").values
        source = "molecule_id"
    elif "smiles" in df.columns:
        smiles_arr = df["smiles"].astype(str).fillna("").values
        groups = np.asarray(smiles_arr, dtype=object)
        source = "smiles_raw"
        try:
            from rdkit import Chem
            from rdkit.Chem import inchi
            converted = 0
            for i, sm in enumerate(smiles_arr):
                s = str(sm).strip()
                if s == "" or s.lower() == "nan":
                    continue
                mol = Chem.MolFromSmiles(s)
                if mol is None:
                    continue
                # prefer InChIKey generated from smiles when possible
                try:
                    ik = inchi.MolToInchiKey(mol)
                except Exception:
                    ik = None
                if ik:
                    groups[i] = ik
                    converted += 1
                else:
                    groups[i] = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
                    converted += 1
            source = f"smiles_rdkit_normalized({converted}/{len(smiles_arr)})"
        except Exception:
            # fallback: still leakage-safe at exact string level
            groups = smiles_arr
            source = "smiles_raw(no_rdkit)"
    else:
        raise ValueError(
            "No molecule identifier column found. Please provide one of: "
            "'inchikey', 'molecule_id', or 'smiles'."
        )

    # avoid merging unrelated rows with missing identifiers into one giant group
    groups = np.asarray(groups, dtype=object)
    for i, g in enumerate(groups):
        if g is None or str(g).strip() == "" or str(g).lower() == "nan":
            groups[i] = f"__MISSING_MOL_{i}__"

    return groups, source

# ---------------------------
# 4) Model
# ---------------------------
class SpectrumTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_layers=4, max_seq_len=MAX_LEN,
                 num_precursor_types=79, dropout=0.1):
        super().__init__()
        self.peak_embed = nn.Sequential(
            nn.Linear(2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        self.precursor_embedding = nn.Embedding(num_precursor_types, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(max_seq_len, d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.output_projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model)
        )

    def forward(self, mz_values, intensities, precursor_types):
        # padding positions are exactly where intensity == 0 (by our padding scheme)
        padding_mask = (intensities == 0)  # [B, L], True=ignore

        peaks = torch.stack([mz_values, intensities], dim=-1)  # [B,L,2]
        x = self.peak_embed(peaks)  # [B,L,d]

        prec_emb = self.precursor_embedding(precursor_types)  # [B,d]
        x = x + prec_emb.unsqueeze(1)

        L = mz_values.shape[1]
        x = x + self.pos_embedding[:L].unsqueeze(0)

        x = self.transformer(x, src_key_padding_mask=padding_mask)

        valid = (~padding_mask).unsqueeze(-1).float()
        x = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

        return self.output_projection(x)

class MultiLabelDNN(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0.1,
                 transformer_dim=256, num_precursor_types=79):
        super().__init__()
        self.transformer = SpectrumTransformer(
            d_model=transformer_dim,
            num_precursor_types=num_precursor_types
        )
        fusion_in = input_size + transformer_dim

        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_in, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(256, output_size)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, mz, inten, prec):
        tf = self.transformer(mz, inten, prec)
        fused = torch.cat([x, tf], dim=1)
        return self.fusion_network(fused)

# ---------------------------
# 5) Fast sample-F1 metric (GPU-friendly)
# ---------------------------
@torch.no_grad()
def fast_sample_f1_from_logits(logits, labels, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    tp = (preds * labels).sum(dim=1)
    fp = (preds * (1 - labels)).sum(dim=1)
    fn = ((1 - preds) * labels).sum(dim=1)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
    return f1

@torch.no_grad()
def evaluate_fast(model, loader, criterion, threshold=0.5):
    model.eval()
    total_loss, total_n, f1_sum = 0.0, 0, 0.0

    for x, y, mz, inten, prec in loader:
        x = x.to(DEVICE).float()
        y = y.to(DEVICE).float()
        mz = mz.to(DEVICE).float()
        inten = inten.to(DEVICE).float()
        prec = prec.to(DEVICE).long()

        logits = model(x, mz, inten, prec)
        loss = criterion(logits, y)

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_n += bs
        f1_sum += fast_sample_f1_from_logits(logits, y, threshold).sum().item()

    return total_loss / total_n, {"sample_f1": f1_sum / total_n}

# ---------------------------
# 6) Full sklearn evaluation (run once on final best)
# ---------------------------
@torch.no_grad()
def evaluate_full_sklearn(model, loader, threshold=0.5):
    model.eval()
    all_probs, all_labels = [], []

    for x, y, mz, inten, prec in loader:
        x = x.to(DEVICE).float()
        y = y.to(DEVICE).float()
        mz = mz.to(DEVICE).float()
        inten = inten.to(DEVICE).float()
        prec = prec.to(DEVICE).long()

        logits = model(x, mz, inten, prec)
        probs = torch.sigmoid(logits)

        all_probs.append(probs.cpu().numpy())
        all_labels.append(y.cpu().numpy())

    Y_prob = np.vstack(all_probs)
    Y_true = np.vstack(all_labels)
    Y_pred = (Y_prob > threshold).astype(int)

    m = {}
    m["sample_f1"] = f1_score(Y_true, Y_pred, average="samples", zero_division=0)
    m["micro_f1"]  = f1_score(Y_true, Y_pred, average="micro", zero_division=0)
    m["macro_f1"]  = f1_score(Y_true, Y_pred, average="macro", zero_division=0)
    m["micro_precision"] = precision_score(Y_true, Y_pred, average="micro", zero_division=0)
    m["micro_recall"]    = recall_score(Y_true, Y_pred, average="micro", zero_division=0)
    m["macro_precision"] = precision_score(Y_true, Y_pred, average="macro", zero_division=0)
    m["macro_recall"]    = recall_score(Y_true, Y_pred, average="macro", zero_division=0)
    m["hamming_loss"] = hamming_loss(Y_true, Y_pred)

    try:
        m["micro_auc"] = roc_auc_score(Y_true, Y_prob, average="micro")
        m["macro_auc"] = roc_auc_score(Y_true, Y_prob, average="macro")
    except Exception:
        m["micro_auc"] = np.nan
        m["macro_auc"] = np.nan

    try:
        m["micro_ap"] = average_precision_score(Y_true, Y_prob, average="micro")
        m["macro_ap"] = average_precision_score(Y_true, Y_prob, average="macro")
    except Exception:
        m["micro_ap"] = np.nan
        m["macro_ap"] = np.nan

    return m

# ---------------------------
# 7) Train loop (early stop on val sample-F1)
# ---------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, threshold,
                num_epochs=NUM_EPOCHS, patience=PATIENCE, save_path=None):
    best_wts = copy.deepcopy(model.state_dict())
    best_val_f1 = -1.0
    no_improve = 0
    history = defaultdict(list)

    for epoch in range(num_epochs):
        t0 = time.time()
        model.train()

        run_loss, run_n, f1_sum = 0.0, 0, 0.0

        for x, y, mz, inten, prec in train_loader:
            x = x.to(DEVICE).float()
            y = y.to(DEVICE).float()
            mz = mz.to(DEVICE).float()
            inten = inten.to(DEVICE).float()
            prec = prec.to(DEVICE).long()

            optimizer.zero_grad(set_to_none=True)
            logits = model(x, mz, inten, prec)
            loss = criterion(logits, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            bs = y.size(0)
            run_loss += loss.item() * bs
            run_n += bs
            f1_sum += fast_sample_f1_from_logits(logits, y, threshold).sum().item()

        train_loss = run_loss / run_n
        train_f1 = f1_sum / run_n

        val_loss, val_metrics = evaluate_fast(model, val_loader, criterion, threshold)
        val_f1 = val_metrics["sample_f1"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        dt = time.time() - t0
        print(f"Epoch {epoch+1:02d}/{num_epochs} | {dt:.1f}s "
              f"| train_loss {train_loss:.4f} train_F1 {train_f1:.4f} "
              f"| val_loss {val_loss:.4f} val_F1 {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_wts = copy.deepcopy(model.state_dict())
            no_improve = 0
            if save_path is not None:
                torch.save({"model_state_dict": best_wts,
                            "best_val_f1": best_val_f1,
                            "epoch": epoch}, save_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping (no val-F1 improvement for {patience} epochs).")
                break

    model.load_state_dict(best_wts)
    return model, history, best_val_f1

# ============================================================
# 8) Main
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the hybrid binned-spectrum MLP + peak Transformer model."
    )
    parser.add_argument(
        "--spectra-csv",
        default="gnps_clean.csv",
        help="Input spectrum metadata CSV. Required columns include mz_values, intensities, and precursor_type.",
    )
    parser.add_argument(
        "--fingerprint-csv",
        default="gnps_clean_PubChemFP.csv",
        help="Input multi-label fingerprint CSV aligned row-by-row with --spectra-csv.",
    )
    parser.add_argument(
        "--save-dir",
        default=SAVE_DIR,
        help="Directory for checkpoints and metric summaries.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print("[1/5] Loading data ...")
    df = pd.read_csv(args.spectra_csv)
    Y = pd.read_csv(args.fingerprint_csv)
    Y = Y.drop(columns=["Original_Index", "SMILES"], errors="ignore")
    
    print(f"df shape: {df.shape}, Y shape: {Y.shape}")

    print("[2/5] Building features (one pass) ...")
    N = len(df)

    X = np.zeros((N, BIN_DIM), dtype=np.float32)
    mz_tensors = np.zeros((N, MAX_LEN), dtype=np.float32)
    inten_tensors = np.zeros((N, MAX_LEN), dtype=np.float32)

    # precursor mapping (fixed order)
    unique_precursors = sorted(df["precursor_type"].dropna().unique().tolist())
    precursor_to_idx = {p: i for i, p in enumerate(unique_precursors)}
    df["precursor_idx"] = df["precursor_type"].map(precursor_to_idx).astype(np.int64)
    precursor_indices = df["precursor_idx"].values.astype(np.int64)

    # build vectors
    for i, row in enumerate(df.itertuples(index=False)):
        # adapt these names if your columns differ
        mz_values = getattr(row, "mz_values")
        intensities = getattr(row, "intensities")

        X[i] = spectrum_to_vector(mz_values, intensities)
        mz_i, in_i = prepare_transformer_input(mz_values, intensities, MAX_LEN)
        mz_tensors[i] = mz_i
        inten_tensors[i] = in_i

        if (i + 1) % 50000 == 0:
            print(f"  built {i+1}/{N}")

    Y_np = Y.values.astype(np.float32)

    print("[3/5] Molecule-grouped split (leakage-safe) ...")
    all_idx = np.arange(N)
    groups, group_source = build_molecule_groups(df)
    print(f"  Group key: {group_source}")

    # step 1: split off test by molecule groups (20% samples target)
    gss_1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    trainval_pos, test_pos = next(gss_1.split(all_idx, groups=groups))

    # step 2: split train/val on remaining groups
    # keep original behavior: val=10% of trainval => overall 8% (train=72%, val=8%, test=20%)
    trainval_idx = all_idx[trainval_pos]
    trainval_groups = groups[trainval_pos]
    gss_2 = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
    train_pos_rel, val_pos_rel = next(gss_2.split(trainval_idx, groups=trainval_groups))

    train_idx = trainval_idx[train_pos_rel].astype(np.int64)
    val_idx = trainval_idx[val_pos_rel].astype(np.int64)
    test_idx = all_idx[test_pos].astype(np.int64)

    # strict leakage check at molecule level
    train_mols = set(groups[train_idx])
    val_mols = set(groups[val_idx])
    test_mols = set(groups[test_idx])
    assert train_mols.isdisjoint(val_mols), "Molecule leakage: train-val overlap"
    assert train_mols.isdisjoint(test_mols), "Molecule leakage: train-test overlap"
    assert val_mols.isdisjoint(test_mols), "Molecule leakage: val-test overlap"

    print(f"  Samples: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    print(f"  Molecules: train={len(train_mols)} val={len(val_mols)} test={len(test_mols)}")
    split_audit = {
        "group_key_source": group_source,
        "n_samples": {"train": int(len(train_idx)), "val": int(len(val_idx)), "test": int(len(test_idx))},
        "n_molecules": {"train": int(len(train_mols)), "val": int(len(val_mols)), "test": int(len(test_mols))},
        "molecule_overlap": {
            "train_val": int(len(train_mols & val_mols)),
            "train_test": int(len(train_mols & test_mols)),
            "val_test": int(len(val_mols & test_mols)),
        },
    }

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = Y_np[train_idx], Y_np[val_idx], Y_np[test_idx]

    mz_train, mz_val, mz_test = mz_tensors[train_idx], mz_tensors[val_idx], mz_tensors[test_idx]
    in_train, in_val, in_test = inten_tensors[train_idx], inten_tensors[val_idx], inten_tensors[test_idx]
    p_train, p_val, p_test = precursor_indices[train_idx], precursor_indices[val_idx], precursor_indices[test_idx]

    train_ds = ExperimentalDataset(X_train, y_train, mz_train, in_train, p_train)
    val_ds   = ExperimentalDataset(X_val,   y_val,   mz_val,   in_val,   p_val)
    test_ds  = ExperimentalDataset(X_test,  y_test,  mz_test,  in_test,  p_test)

    # # remove zero-intensity samples (optional, but keep consistent)
    # train_ds = remove_zero_intensity_samples(train_ds)
    # val_ds   = remove_zero_intensity_samples(val_ds)
    # test_ds  = remove_zero_intensity_samples(test_ds)
    

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    print("[4/5] Grid search on val (NO test peeking) ...")
    criterion = nn.BCEWithLogitsLoss()

    best = {
        "val_f1": -1.0,
        "model_state_dict": None,
        "params": None,
        "history": None
    }

    for lr in LR_GRID:
        for wd in WD_GRID:
            for dr in DROPOUT_GRID:
                for thr in THRESH_GRID:
                    print(f"\nRun: lr={lr}, wd={wd}, dropout={dr}, thr={thr}")

                    model = MultiLabelDNN(
                        input_size=BIN_DIM,
                        output_size=FP_DIM,
                        dropout_rate=dr,
                        transformer_dim=256,
                        num_precursor_types=len(unique_precursors)
                    ).to(DEVICE)

                    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

                    tmp_ckpt = os.path.join(args.save_dir, "tmp_best.pth")
                    model, history, best_val_f1 = train_model(
                        model, train_loader, val_loader, criterion, optimizer,
                        threshold=thr, num_epochs=NUM_EPOCHS, patience=PATIENCE,
                        save_path=tmp_ckpt
                    )

                    if best_val_f1 > best["val_f1"]:
                        best["val_f1"] = best_val_f1
                        best["model_state_dict"] = copy.deepcopy(model.state_dict())
                        best["params"] = {"lr": lr, "weight_decay": wd, "dropout": dr, "threshold": thr}
                        best["history"] = history
                        print(f" New best on VAL: {best_val_f1:.4f}")

    print("[5/5] Final evaluation on TEST (once) ...")
    best_model = MultiLabelDNN(
        input_size=BIN_DIM,
        output_size=FP_DIM,
        dropout_rate=best["params"]["dropout"],
        transformer_dim=256,
        num_precursor_types=len(unique_precursors)
    ).to(DEVICE)
    best_model.load_state_dict(best["model_state_dict"])
    best_model.eval()

    thr = best["params"]["threshold"]
    test_loss_fast, test_fast = evaluate_fast(best_model, test_loader, criterion, thr)
    test_full = evaluate_full_sklearn(best_model, test_loader, thr)

    print("\n" + "=" * 80)
    print("BEST PARAMS:", best["params"])
    print(f"BEST VAL sample-F1: {best['val_f1']:.6f}")
    print("\nTEST (fast):")
    print(f"  test_loss: {test_loss_fast:.6f}")
    print(f"  sample_f1_fast: {test_fast['sample_f1']:.6f}")

    print("\nTEST (full sklearn):")
    for k, v in test_full.items():
        print(f"  {k:16s}: {v:.6f}")
    print("=" * 80)

    # save full checkpoint (paper-grade reproducibility)
    out_path = os.path.join(args.save_dir, "best_model_complete.pth")
    torch.save({
        "model_state_dict": best["model_state_dict"],
        "params": best["params"],
        "best_val_f1": best["val_f1"],
        "test_fast": {"test_loss": test_loss_fast, **test_fast},
        "test_full": test_full,
        "precursor_to_idx": precursor_to_idx,
        "split_indices": {
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx
        },
        "split_audit": split_audit,
        "config": {
            "BIN_DIM": BIN_DIM,
            "FP_DIM": FP_DIM,
            "MAX_LEN": MAX_LEN,
            "mz_min": mz_min,
            "mz_max": mz_max,
            "bin_size": bin_size,
            "seed": SEED
        }
    }, out_path)

    # also write a json summary for paper tables
    summary = {
        "best_params": best["params"],
        "best_val_sample_f1": float(best["val_f1"]),
        "test_fast": {"test_loss": float(test_loss_fast), "sample_f1_fast": float(test_fast["sample_f1"])},
        "test_full": {k: float(v) for k, v in test_full.items()}
    }
    metrics_path = os.path.join(args.save_dir, "best_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[Saved] {out_path}")
    print(f"[Saved] {metrics_path}")

if __name__ == "__main__":
    main()


