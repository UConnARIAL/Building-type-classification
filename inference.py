# -*- coding: utf-8 -*-
"""
Dual GraphSAGE inference script (domain-isolated)
-------------------------------------------------
- Builds separate graphs for OSM and HABITAT nodes
- Applies each trained model to its own domain
- Writes both probability and binary Value outputs
- Exports shapefile with EPSG:3995 coordinates

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
from torch_geometric.loader import NeighborLoader  # <-- use directly


# =====================================================
# Configuration / device
# =====================================================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

DATA_PATH = CONFIG["data_path"]
FEATURE_COLS = CONFIG["feature_cols"]
SPATIAL_COLS = CONFIG["spatial_cols"]

OSM_MODEL_PATH = CONFIG["fine_tune_pretrained_path"]
DL_MODEL_PATH = CONFIG["fine_tune_model_path"]

OUT_DIR = CONFIG["output_results_path"]
OUT_CSV = os.path.join(OUT_DIR, "inference_OSM_HABITAT_predictions.csv")
OUT_SHP = os.path.join(OUT_DIR, "inference_OSM_HABITAT_predictions.shp")

params = CONFIG["fine_tune_params"]
HIDDEN_DIM = params["hidden_dim"]
NUM_LAYERS = params["num_layers"]
DROPOUT = params["dropout"]
K = CONFIG["knn_k"]
BATCH_SIZE = CONFIG["batch_size"]
NUM_NEIGHBORS = CONFIG["num_neighbors_default"]
NUM_WORKERS = CONFIG.get("loader_num_workers", 0)


# =====================================================
# Load input data
# =====================================================
print(f"Loading data from {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH, low_memory=False)

mask_osm = df["Source"].str.upper() == "OSM"
mask_habitat = df["Source"].str.upper() == "HABITAT"

print(f"OSM nodes: {mask_osm.sum():,} | HABITAT nodes: {mask_habitat.sum():,}")


# =====================================================
# Model loader
# =====================================================
def load_model(path):
    model = GraphSAGE(
        in_channels=len(FEATURE_COLS),
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded model → {path}")
    return model


# =====================================================
# Inference loader builder
# =====================================================
def build_inference_loader(data, idx_subset):
    """Consistent NeighborLoader for inference (no val/test placeholders)."""
    nodes = torch.tensor(idx_subset, dtype=torch.long)
    num_neighbors = [NUM_NEIGHBORS] * NUM_LAYERS
    return NeighborLoader(
        data,
        input_nodes=nodes,
        num_neighbors=num_neighbors,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )


# =====================================================
# Inference helper
# =====================================================
@torch.no_grad()
def infer_subset(model, X, coords, name):
    """Run binary + probability inference for one domain subset."""
    print(f"\nBuilding KNN graph for {name} ({len(X):,} nodes) ...")
    data = build_knn_graph(X, coords, np.zeros(len(X)), k=K).to(DEVICE)

    loader = build_inference_loader(data, np.arange(len(X)))

    probs_all = []
    for batch in tqdm(loader, desc=f"Inference ({name})", dynamic_ncols=True):
        batch = batch.to(DEVICE)
        logits = model(batch)
        probs = torch.sigmoid(logits[:batch.batch_size])
        probs_all.append(probs.cpu().numpy())

    probs_all = np.concatenate(probs_all)
    preds_bin = (probs_all >= 0.5).astype(int)
    return preds_bin, probs_all


# =====================================================
# Load models
# =====================================================
model_osm = load_model(OSM_MODEL_PATH)
model_dl = load_model(DL_MODEL_PATH)


# =====================================================
# Prepare subsets
# =====================================================
X_osm = df.loc[mask_osm, FEATURE_COLS].values.astype(np.float32)
coords_osm = df.loc[mask_osm, SPATIAL_COLS].values.astype(np.float32)

X_dl = df.loc[mask_habitat, FEATURE_COLS].values.astype(np.float32)
coords_dl = df.loc[mask_habitat, SPATIAL_COLS].values.astype(np.float32)


# =====================================================
# Run inference
# =====================================================
preds_osm, probs_osm = infer_subset(model_osm, X_osm, coords_osm, "OSM")
preds_dl, probs_dl = infer_subset(model_dl, X_dl, coords_dl, "HABITAT")


# =====================================================
# Merge results back to master dataframe
# =====================================================
Value = np.zeros(len(df), dtype=int)
Prob = np.zeros(len(df), dtype=float)

Value[mask_osm.values] = preds_osm
Prob[mask_osm.values] = probs_osm

Value[mask_habitat.values] = preds_dl
Prob[mask_habitat.values] = probs_dl

df["Value"] = Value
df["Prob"] = Prob


# =====================================================
# Save CSV + shapefile
# =====================================================
os.makedirs(OUT_DIR, exist_ok=True)
df.to_csv(OUT_CSV, index=False)
print(f"\n✅ Saved predictions → {OUT_CSV}")

if {"centroid_x", "centroid_y"}.issubset(df.columns):
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df["centroid_x"], df["centroid_y"])],
        crs="EPSG:3995"
    )
    gdf[["centroid_x", "centroid_y", "Value", "Prob", "geometry"]].to_file(OUT_SHP)
    print(f"✅ Saved shapefile → {OUT_SHP}")
