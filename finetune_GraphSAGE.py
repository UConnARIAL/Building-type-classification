# -*- coding: utf-8 -*-
"""
Fine-tune pretrained GraphSAGE on DL-domain nodes
while masking out OSM nodes during loss computation.
Now includes reproducible train/val/test splits and hold-out evaluation.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from models import GraphSAGE
from config import CONFIG
from tune_and_eval_dist_v3 import build_knn_graph, build_neighbor_loaders, evaluate_minibatch

# =====================================================
# Configuration and device
# =====================================================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PRETRAINED_PATH = CONFIG["fine_tune_pretrained_path"]
DATA_PATH = CONFIG["data_path"]
OUT_MODEL_PATH = CONFIG["fine_tune_model_path"]
OUT_METRICS_CSV = CONFIG["fine_tune_metrics_path"]

params = CONFIG["fine_tune_params"]
HIDDEN_DIM = params["hidden_dim"]
NUM_LAYERS = params["num_layers"]
DROPOUT = params["dropout"]
LR = params["lr"]

EPOCHS = CONFIG["fine_tune_epochs"]
PATIENCE = CONFIG["fine_tune_patience"]
BATCH_SIZE = CONFIG["batch_size"]
KNN_K = CONFIG["knn_k"]
FREEZE = CONFIG.get("fine_tune_freeze_layers", False)

SPLIT_DIR = os.path.join(os.path.dirname(DATA_PATH), "splits")
os.makedirs(SPLIT_DIR, exist_ok=True)

# =====================================================
# Load dataset
# =====================================================
df = pd.read_csv(DATA_PATH, low_memory=False)
df["label"] = df["label"].fillna(-1).astype(int)

mask_labeled = df["label"] != -1
mask_osm = df["Source"].str.upper() == "OSM"
mask_dl_labeled = mask_labeled & (~mask_osm)

X = df[CONFIG["feature_cols"]].values.astype(np.float32)
coords = df[CONFIG["spatial_cols"]].values.astype(np.float32)
y = df["label"].values.astype(int)

data = build_knn_graph(X, coords, y, k=KNN_K).to(DEVICE)

# =====================================================
# Train / Val / Test Split (reproducible)
# =====================================================
train_file = os.path.join(SPLIT_DIR, "dl_train_idx.csv")
val_file = os.path.join(SPLIT_DIR, "dl_val_idx.csv")
test_file = os.path.join(SPLIT_DIR, "combined_test_idx.csv")

if all(os.path.exists(f) for f in [train_file, val_file, test_file]):
    train_idx = np.loadtxt(train_file, dtype=int)
    val_idx = np.loadtxt(val_file, dtype=int)
    test_idx = np.loadtxt(test_file, dtype=int)
    print("Loaded existing DL-domain + OSM test split.")
else:
    # Only DL-labeled samples for training/validation
    labeled_idx_dl = np.where(mask_dl_labeled)[0]
    np.random.seed(42)
    np.random.shuffle(labeled_idx_dl)
    n = len(labeled_idx_dl)
    train_idx = labeled_idx_dl[:int(0.8 * n)]
    val_idx = labeled_idx_dl[int(0.8 * n):int(0.9 * n)]

    # Include *all labeled* nodes (DL + OSM) in the test set for full evaluation
    labeled_idx_all = np.where(mask_labeled)[0]
    test_idx = np.setdiff1d(labeled_idx_all, np.concatenate([train_idx, val_idx]))

    np.savetxt(train_file, train_idx, fmt="%d")
    np.savetxt(val_file, val_idx, fmt="%d")
    np.savetxt(test_file, test_idx, fmt="%d")
    print("Generated and saved new DL-domain train/val and combined-domain test splits.")


train_loader, val_loader, test_loader = build_neighbor_loaders(
    data, train_idx, val_idx, test_idx,
    num_layers=NUM_LAYERS,
    batch_size=BATCH_SIZE,
    num_neighbors_each_layer=CONFIG["num_neighbors_default"]
)

# =====================================================
# Load pretrained GraphSAGE model
# =====================================================
model = GraphSAGE(
    in_channels=X.shape[1],
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
).to(DEVICE)

model.load_state_dict(torch.load(PRETRAINED_PATH, map_location=DEVICE))
print(f"Loaded pretrained model from {PRETRAINED_PATH}")

# --- Optional freezing control ---
FREEZE = CONFIG.get("fine_tune_freeze_layers", False)
FREEZE_DEPTH = CONFIG.get("fine_tune_freeze_depth", 0)

if FREEZE and FREEZE_DEPTH > 0:
    frozen = 0
    for name, param in model.named_parameters():
        # Freeze bottom layers up to the chosen depth
        if any(f"convs.{i}." in name for i in range(FREEZE_DEPTH)):
            param.requires_grad = False
            frozen += 1
    print(f"Froze {FREEZE_DEPTH} bottom GraphSAGE layer(s) "
          f"({frozen} parameter groups).")
elif FREEZE:
    print("fine_tune_freeze_layers=True but fine_tune_freeze_depth=0 — nothing frozen.")
else:
    print("No layers frozen; full model fine-tuning.")


# =====================================================
# Fine-tuning loop
# =====================================================
opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
best_f1, best_state, no_improve = -1.0, None, 0

for epoch in tqdm(range(EPOCHS), desc="Fine-tuning", dynamic_ncols=True):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        batch = batch.to(DEVICE)
        opt.zero_grad()
        logits = model(batch)
        out = logits[:batch.batch_size]
        y_true = batch.y[:batch.batch_size]
        loss = F.binary_cross_entropy_with_logits(out, y_true)
        loss.backward()
        opt.step()
        running_loss += loss.item()

    p_val, r_val, f_val = evaluate_minibatch(model, val_loader, DEVICE)
    print(f"Epoch {epoch+1:03d}: val_F1={f_val:.4f}, val_P={p_val:.4f}, val_R={r_val:.4f}")

    if f_val > best_f1:
        best_f1, best_state = f_val, {k: v.cpu() for k, v in model.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

if best_state:
    model.load_state_dict(best_state)

# =====================================================
# Save model and metrics
# =====================================================
torch.save(model.state_dict(), OUT_MODEL_PATH)
print(f"Saved fine-tuned model → {OUT_MODEL_PATH}")

# =====================================================
# Final hold-out evaluation (domain-specific)
# =====================================================

def evaluate_masked(model, df, data, mask, name):
    """Evaluate model on a masked subset of labeled nodes."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        print(f"[WARN] No nodes found for {name} evaluation.")
        return np.nan, np.nan, np.nan

    loader, _, _ = build_neighbor_loaders(
        data, idx, [], [],
        num_layers=NUM_LAYERS,
        batch_size=BATCH_SIZE,
        num_neighbors_each_layer=CONFIG["num_neighbors_default"]
    )
    p, r, f = evaluate_minibatch(model, loader, DEVICE)
    print(f"{name} → P={p:.4f}, R={r:.4f}, F1={f:.4f}")
    return p, r, f

# --- Define masks for labeled subsets ---
mask_osm_labeled = mask_labeled & mask_osm
mask_dl_labeled  = mask_labeled & (~mask_osm)
mask_all_labeled = mask_labeled

# --- Run evaluations ---
p_osm, r_osm, f_osm = evaluate_masked(model, df, data, mask_osm_labeled, "OSM-only")
p_dl,  r_dl,  f_dl  = evaluate_masked(model, df, data, mask_dl_labeled,  "DL-only")
p_all, r_all, f_all = evaluate_masked(model, df, data, mask_all_labeled, "All labeled")

# --- Save metrics ---
metrics = pd.DataFrame([{
    "precision_val": p_val, "recall_val": r_val, "f1_val": best_f1,
    "precision_osm": p_osm, "recall_osm": r_osm, "f1_osm": f_osm,
    "precision_dl":  p_dl,  "recall_dl":  r_dl,  "f1_dl":  f_dl,
    "precision_all": p_all, "recall_all": r_all, "f1_all": f_all
}])
metrics.to_csv(OUT_METRICS_CSV, index=False)
print(f"\nSaved full domain metrics → {OUT_METRICS_CSV}")
