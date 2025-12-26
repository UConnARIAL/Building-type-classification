# -*- coding: utf-8 -*-
"""
Inference script for dual GraphSAGE models:
- Uses OSM-trained model for 'OSM' nodes
- Uses DL-finetuned model for 'HABITAT' nodes
- Adds binary 'Value' predictions (0/1) to CSV
- Exports shapefile of predicted points

Author: Elias Manos
"""

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from shapely.geometry import Point
import geopandas as gpd

from config import CONFIG
from models import GraphSAGE
from graph_utils import build_knn_graph
from tune_and_eval_dist_v3 import build_neighbor_loaders


# =====================================================
# Configuration and device
# =====================================================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

DATA_PATH = CONFIG["data_path"]
FEATURE_COLS = CONFIG["feature_cols"]
SPATIAL_COLS = CONFIG["spatial_cols"]

OSM_MODEL_PATH = CONFIG["fine_tune_pretrained_path"]
DL_MODEL_PATH = CONFIG["fine_tune_model_path"]

OUT_CSV = os.path.join(CONFIG["output_results_path"], "inference_OSM_HABITAT_predictions.csv")
OUT_SHP = os.path.join(CONFIG["output_results_path"], "inference_OSM_HABITAT_predictions.shp")

params = CONFIG["fine_tune_params"]
HIDDEN_DIM = params["hidden_dim"]
NUM_LAYERS = params["num_layers"]
DROPOUT = params["dropout"]


# =====================================================
# Load data
# =====================================================
print(f"Loading data from {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH, low_memory=False)

mask_osm = df["Source"].str.upper() == "OSM"
mask_habitat = df["Source"].str.upper() == "HABITAT"

X = df[FEATURE_COLS].values.astype(np.float32)
coords = df[SPATIAL_COLS].values.astype(np.float32)
y_dummy = np.zeros(len(df), dtype=int)


# =====================================================
# Build shared KNN graph
# =====================================================
print("Building KNN graph ...")
data = build_knn_graph(X, coords, y_dummy, k=CONFIG["knn_k"]).to(DEVICE)


# =====================================================
# Load GraphSAGE models
# =====================================================
def load_model(path):
    model = GraphSAGE(
        in_channels=X.shape[1],
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded model from {path}")
    return model


model_osm = load_model(OSM_MODEL_PATH)
model_dl = load_model(DL_MODEL_PATH)


# =====================================================
# Inference helper
# =====================================================
def infer_subset(model, idx_subset, name):
    """Run binary inference for a given subset of nodes."""
    if len(idx_subset) == 0:
        print(f"[WARN] No nodes found for {name}")
        return np.zeros(0)
    loader, _, _ = build_neighbor_loaders(
        data, idx_subset, [], [],
        num_layers=NUM_LAYERS,
        batch_size=CONFIG["batch_size"],
        num_neighbors_each_layer=CONFIG["num_neighbors_default"]
    )
    preds = []
    for batch in tqdm(loader, desc=f"Inference ({name})", dynamic_ncols=True):
        batch = batch.to(DEVICE)
        logits = model(batch)
        probs = torch.sigmoid(logits[:batch.batch_size])
        binary = (probs >= 0.5).float()
        preds.append(binary.cpu().numpy())
    return np.concatenate(preds)


# =====================================================
# Run inference for each domain
# =====================================================
idx_osm = np.where(mask_osm.values)[0]
idx_dl = np.where(mask_habitat.values)[0]

print(f"Running inference on {len(idx_osm)} OSM and {len(idx_dl)} HABITAT nodes ...")

preds_osm = infer_subset(model_osm, idx_osm, "OSM")
preds_dl = infer_subset(model_dl, idx_dl, "HABITAT")


# =====================================================
# Merge predictions and save CSV
# =====================================================
Value = np.zeros(len(df), dtype=int)
Value[idx_osm] = preds_osm
Value[idx_dl] = preds_dl

df["Value"] = Value.astype(int)
df.to_csv(OUT_CSV, index=False)

print(f"\nSaved binary inference results → {OUT_CSV}")


# =====================================================
# Export shapefile of predictions
# =====================================================
if {"centroid_x", "centroid_y", "Value"}.issubset(df.columns):
    print("Creating GeoDataFrame and exporting shapefile ...")
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df["centroid_x"], df["centroid_y"])],
        crs="EPSG:3995"  # North Pole LAEA projection
    )
    gdf[["centroid_x", "centroid_y", "Value", "geometry"]].to_file(OUT_SHP)
    print(f"Saved shapefile → {OUT_SHP}")
else:
    print("Required columns (centroid_x, centroid_y, Value) not found — shapefile skipped.")
