# -*- coding: utf-8 -*-
"""
Fine-tune pretrained GraphSAGE on DL-domain nodes
while masking out OSM nodes during loss computation.
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

# =====================================================
# Load dataset
# =====================================================
df = pd.read_csv(DATA_PATH)
df["label"] = df["label"].fillna(-1).astype(int)

mask_labeled = df["label"] != -1
mask_osm = df["Source"].str.upper() == "OSM"
mask_dl_labeled = mask_labeled & (~mask_osm)

X = df[CONFIG["feature_cols"]].values.astype(np.float32)
coords = df[CONFIG["spatial_cols"]].values.astype(np.float32)
y = df["label"].values.astype(int)

data = build_knn_graph(X, coords, y, k=KNN_K).to(DEVICE)

# =====================================================
# Define DL-domain train/val split
# =====================================================
labeled_idx = np.where(mask_dl_labeled)[0]
np.random.shuffle(labeled_idx)
n = len(labeled_idx)
train_idx = labeled_idx[:int(0.9 * n)]
val_idx = labeled_idx[int(0.9 * n):]

train_loader, val_loader, _ = build_neighbor_loaders(
    data, train_idx, val_idx, val_idx,
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

if FREEZE:
    # Freeze all layers except the final one for stable domain adaptation
    for name, param in model.named_parameters():
        if "lin_layers" not in name:
            param.requires_grad = False
    print("Froze lower layers for fine-tuning.")

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

metrics = pd.DataFrame([{"precision_val": p_val, "recall_val": r_val, "f1_val": best_f1}])
metrics.to_csv(OUT_METRICS_CSV, index=False)
print(f"Saved validation metrics → {OUT_METRICS_CSV}")
