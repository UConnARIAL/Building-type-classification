"""
Single-GPU (no distribution) tuning for Decision Tree, Random Forest, and FCN.

- Uses the SAME fixed splits as GNN script (train/val/test from NPZ).
- Sequential random search (no multiprocessing/DDP). FCN uses cuda:0 if available.
- **Macro** P/R/F1 for tuning *and* final evaluation.
- Final models also write a **per-class classification_report** CSV.
- Saves per-model tuning CSVs, a final eval CSV, and serialized models under output_dir.
- Logs runtime of each final model's training/evaluation.

Run:
  python classical_models_tune_and_eval_single_gpu.py

Edit CONFIG below if needed.
"""
import os
import json
import time
import random
import pickle
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ============================
# CONFIG
# ============================
CONFIG: Dict[str, Any] = {
    "seed": 42,
    # I/O
    "data_path": r"D:\PhD_main\chapter_2\data\building_node_features_with_admin.csv",
    "split_path": r"D:\PhD_main\chapter_2\code\GNN\results\splits_GraphSAGE.npz",
    "output_dir": r"D:\PhD_main\chapter_2\outputs\results\classical_results",

    # Columns
    "feature_cols": None,  # if None, infer numeric columns excluding exclude_cols + label
    "exclude_cols": [
        "label", "node_id", "building_id", "layer0", "layer1", "layer2",
        "x", "y", "lon", "lat", "geometry"
    ],
    "label_col": "label",

    # Search
    "n_trials": 30,  # per model

    # FCN
    "fcn_max_epochs": 200,
    "fcn_patience": 20,
}

# ============================
# Utils
# ============================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def infer_feature_cols(df: pd.DataFrame, label_col: str, exclude_cols: List[str]) -> List[str]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = set([c for c in exclude_cols if c in df.columns] + [label_col])
    feats = [c for c in num_cols if c not in exclude]
    if not feats:
        raise ValueError("No numeric feature columns found. Set CONFIG['feature_cols'] explicitly.")
    return feats


def load_splits(split_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sp = np.load(split_path, allow_pickle=True)
    for k in ["train_idx", "val_idx", "test_idx"]:
        if k not in sp.files:
            raise KeyError(f"Split file missing '{k}'. Found: {list(sp.files)}")
    return sp["train_idx"], sp["val_idx"], sp["test_idx"]


def coerce_split_indices(train_idx, val_idx, test_idx, mask_labeled: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Make split indices compatible with the labeled subset ordering.
    If indices are already within [0, n_labeled), return as-is.
    Otherwise assume they refer to absolute positions in the full DataFrame and map -> labeled order.
    """
    n_lab = int(mask_labeled.sum())
    max_idx = int(max([np.max(train_idx), np.max(val_idx), np.max(test_idx)]))
    if max_idx < n_lab:
        return train_idx.astype(int), val_idx.astype(int), test_idx.astype(int)
    labeled_positions = np.where(mask_labeled)[0]
    pos_to_lab = {int(pos): i for i, pos in enumerate(labeled_positions)}
    def convert(arr):
        return np.array([pos_to_lab[int(p)] for p in arr if int(p) in pos_to_lab], dtype=int)
    return convert(train_idx), convert(val_idx), convert(test_idx)


def metrics_from_preds(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> Tuple[float, float, float]:
    """Return P/R/F1 using the requested averaging (default=macro)."""
    p = precision_score(y_true, y_pred, average=average, zero_division=0)
    r = recall_score(y_true, y_pred, average=average, zero_division=0)
    f = f1_score(y_true, y_pred, average=average, zero_division=0)
    return p, r, f

# ============================
# FCN (simple MLP)
# ============================
class FCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for _ in range(num_layers):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden_dim
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_fcn(x_train, y_train, x_val, y_val, hp: Dict[str, Any], device: torch.device,
              max_epochs: int, patience: int) -> Tuple[nn.Module, Tuple[float,float,float]]:
    model = FCN(
        in_dim=x_train.shape[1],
        hidden_dim=hp["hidden_dim"],
        num_layers=hp["num_layers"],
        dropout=hp["dropout"],
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    crit = nn.BCEWithLogitsLoss()

    tr_ds = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float())
    va_ds = TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val).float())
    tr_dl = DataLoader(tr_ds, batch_size=hp["batch_size"], shuffle=True, num_workers=0)
    va_dl = DataLoader(va_ds, batch_size=hp["batch_size"], shuffle=False, num_workers=0)

    best_f1 = -1.0
    best_state = None
    no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logit = model(xb)
            loss = crit(logit, yb)
            loss.backward()
            opt.step()

        # val
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in va_dl:
                xb = xb.to(device)
                logit = model(xb)
                pr = (torch.sigmoid(logit) >= 0.5).cpu().numpy()
                preds.append(pr)
                trues.append(yb.numpy())
        y_pred = np.concatenate(preds)
        y_true = np.concatenate(trues)
        p, r, f = metrics_from_preds(y_true, y_pred, average="macro")

        if f > best_f1:
            best_f1 = f
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[FCN] Early stop at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, (p, r, best_f1)

# ============================
# Search spaces
# ============================
DT_SPACE = {
    "criterion": ["gini", "entropy"],  # removed log_loss for compat with sklearn<=1.0
    "max_depth": [None, 8, 12, 20, 30, 50],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 5],
    "class_weight": [None, "balanced"],
}
RF_SPACE = {
    "n_estimators": [100, 200, 400, 800],
    "max_depth": [None, 10, 20, 40],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False],
    "class_weight": [None, "balanced"],
    "max_features": ["sqrt", "log2", None],
}
FCN_SPACE = {
    "hidden_dim": [64, 128, 256, 512],
    "num_layers": [1, 2, 3],
    "dropout": [0.0, 0.1, 0.25, 0.5],
    "lr": [5e-4, 1e-3, 2e-3],
    "batch_size": [256, 512, 1024],
}


def sample(space: Dict[str, List[Any]]) -> Dict[str, Any]:
    return {k: random.choice(v) for k, v in space.items()}

# ============================
# Tuning loops (sequential)
# ============================

def tune_DT(Xtr, ytr, Xva, yva, n_trials: int, out_csv: str) -> Dict[str, Any]:
    rows = []
    best = {"f1": -1.0, "params": None}
    for t in range(n_trials):
        hp = sample(DT_SPACE)
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", DecisionTreeClassifier(random_state=CONFIG["seed"] + t, **hp)),
        ])
        pipe.fit(Xtr, ytr)
        ypred = pipe.predict(Xva)
        p, r, f = metrics_from_preds(yva, ypred, average="macro")
        rows.append({"trial": t, "params": json.dumps(hp), "precision_val": p, "recall_val": r, "f1_val": f})
        if f > best["f1"]:
            best = {"f1": f, "params": hp}
        print(f"[DT] t={t:03d} F1={f:.4f} P={p:.4f} R={r:.4f} | {hp}")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return best


def tune_RF(Xtr, ytr, Xva, yva, n_trials: int, out_csv: str) -> Dict[str, Any]:
    rows = []
    best = {"f1": -1.0, "params": None}
    for t in range(n_trials):
        hp = sample(RF_SPACE)
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(random_state=CONFIG["seed"] + t, n_jobs=-1, **hp)),
        ])
        pipe.fit(Xtr, ytr)
        ypred = pipe.predict(Xva)
        p, r, f = metrics_from_preds(yva, ypred, average="macro")
        rows.append({"trial": t, "params": json.dumps(hp), "precision_val": p, "recall_val": r, "f1_val": f})
        if f > best["f1"]:
            best = {"f1": f, "params": hp}
        print(f"[RF] t={t:03d} F1={f:.4f} P={p:.4f} R={r:.4f} | {hp}")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return best


def tune_FCN(Xtr, ytr, Xva, yva, n_trials: int, out_csv: str) -> Dict[str, Any]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    imp = SimpleImputer(strategy="median")
    sca = StandardScaler()
    Xtr_p = sca.fit_transform(imp.fit_transform(Xtr))
    Xva_p = sca.transform(imp.transform(Xva))

    rows = []
    best = {"f1": -1.0, "params": None}
    for t in range(n_trials):
        hp = sample(FCN_SPACE)
        model, (p, r, f) = train_fcn(Xtr_p, ytr, Xva_p, yva, hp, device, CONFIG["fcn_max_epochs"], CONFIG["fcn_patience"])
        rows.append({"trial": t, "params": json.dumps(hp), "precision_val": p, "recall_val": r, "f1_val": f})
        if f > best["f1"]:
            best = {"f1": f, "params": hp}
        print(f"[FCN] t={t:03d} F1={f:.4f} P={p:.4f} R={r:.4f} | {hp}")
        # cleanup
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return best

# ============================
# Final train & test
# ============================

def _save_per_class_report(y_true, y_pred, out_dir: str, name: str):
    rep = classification_report(y_true, y_pred, digits=4, output_dict=True, zero_division=0)
    pd.DataFrame(rep).transpose().to_csv(os.path.join(out_dir, f"classification_report_{name}.csv"))


def final_DT_RF(name: str, params: Dict[str, Any], Xtrv, ytrv, Xte, yte, out_dir: str) -> Dict[str, Any]:
    clf = RandomForestClassifier(n_jobs=-1, random_state=CONFIG["seed"], **params) if name == "RF" \
          else DecisionTreeClassifier(random_state=CONFIG["seed"], **params)
    pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("clf", clf)])
    t0 = time.time()
    pipe.fit(Xtrv, ytrv)
    runtime = time.time() - t0
    ypred = pipe.predict(Xte)
    p, r, f = metrics_from_preds(yte, ypred, average="macro")
    _save_per_class_report(yte, ypred, out_dir, name)
    path = os.path.join(out_dir, f"best_{name}.pkl")
    with open(path, "wb") as fobj:
        pickle.dump(pipe, fobj)
    print(f"FINAL {name}: P_macro={p:.4f} R_macro={r:.4f} F1_macro={f:.4f} | runtime={runtime:.2f}s")
    return {"precision_test_macro": p, "recall_test_macro": r, "f1_test_macro": f, "runtime_sec": runtime, "model_path": path}


def final_FCN(params: Dict[str, Any], Xtrv, ytrv, Xte, yte, out_dir: str) -> Dict[str, Any]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    imp = SimpleImputer(strategy="median")
    sca = StandardScaler()
    Xtrv_p = sca.fit_transform(imp.fit_transform(Xtrv))
    Xte_p = sca.transform(imp.transform(Xte))

    model = FCN(Xtrv_p.shape[1], params["hidden_dim"], params["num_layers"], params["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"])
    crit = nn.BCEWithLogitsLoss()

    ds = TensorDataset(torch.from_numpy(Xtrv_p).float(), torch.from_numpy(ytrv).float())
    dl = DataLoader(ds, batch_size=params["batch_size"], shuffle=True, num_workers=0)

    best_loss = float("inf")
    best_state = None
    no_imp = 0
    t0 = time.time()
    for ep in range(CONFIG["fcn_max_epochs"]):
        model.train()
        run = 0.0; nb = 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); logit = model(xb); loss = crit(logit, yb); loss.backward(); opt.step()
            run += float(loss.item()); nb += 1
        eloss = run / max(1, nb)
        if eloss < best_loss:
            best_loss = eloss; best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}; no_imp = 0
        else:
            no_imp += 1
            if no_imp >= CONFIG["fcn_patience"]:
                print(f"[FCN final] Early stop at epoch {ep+1} (best train loss={best_loss:.4f})")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    runtime = time.time() - t0

    model.eval()
    with torch.no_grad():
        ylog = model(torch.from_numpy(Xte_p).float().to(device)).cpu().numpy()
        ypred = (1.0 / (1.0 + np.exp(-ylog)) >= 0.5).astype(int)
    p, r, f = metrics_from_preds(yte, ypred, average="macro")
    _save_per_class_report(yte, ypred, out_dir, "FCN")

    bundle = {"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
              "imputer": imp, "scaler": sca, "params": params, "in_dim": Xtrv_p.shape[1]}
    path = os.path.join(out_dir, "best_FCN.pkl")
    with open(path, "wb") as fobj:
        pickle.dump(bundle, fobj)
    print(f"FINAL FCN: P_macro={p:.4f} R_macro={r:.4f} F1_macro={f:.4f} | runtime={runtime:.2f}s")
    return {"precision_test_macro": p, "recall_test_macro": r, "f1_test_macro": f, "runtime_sec": runtime, "model_path": path}

# ============================
# Main
# ============================

def main():
    cfg = CONFIG
    set_seed(cfg["seed"])
    ensure_dir(cfg["output_dir"])

    # Data
    df = pd.read_csv(cfg["data_path"])
    if cfg["label_col"] not in df.columns:
        raise KeyError(f"Label column '{cfg['label_col']}' not in data")
    df[cfg["label_col"]] = df[cfg["label_col"]].fillna(-1).astype(int)
    mask_lab = (df[cfg["label_col"]].values != -1)
    df_lab = df.loc[mask_lab].reset_index(drop=True)

    feats = cfg["feature_cols"] if cfg["feature_cols"] is not None else infer_feature_cols(df_lab, cfg["label_col"], cfg["exclude_cols"])
    print(f"Using {len(feats)} feature columns.")

    X_all = df_lab[feats].values.astype(np.float32)
    y_all = df_lab[cfg["label_col"]].values.astype(int)

    tr_idx_raw, va_idx_raw, te_idx_raw = load_splits(cfg["split_path"])
    tr_idx, va_idx, te_idx = coerce_split_indices(tr_idx_raw, va_idx_raw, te_idx_raw, mask_lab)

    Xtr, ytr = X_all[tr_idx], y_all[tr_idx]
    Xva, yva = X_all[va_idx], y_all[va_idx]
    Xte, yte = X_all[te_idx], y_all[te_idx]

    print(f"Split sizes -> train: {len(ytr)}, val: {len(yva)}, test: {len(yte)}")

    # ===== Tuning (sequential) =====
    best_dt = tune_DT(Xtr, ytr, Xva, yva, cfg["n_trials"], os.path.join(cfg["output_dir"], "tuning_DT.csv"))
    best_rf = tune_RF(Xtr, ytr, Xva, yva, cfg["n_trials"], os.path.join(cfg["output_dir"], "tuning_RF.csv"))
    best_fcn = tune_FCN(Xtr, ytr, Xva, yva, cfg["n_trials"], os.path.join(cfg["output_dir"], "tuning_FCN.csv"))

    # ===== Final train+val & test =====
    Xtrv = np.concatenate([Xtr, Xva]); ytrv = np.concatenate([ytr, yva])

    eval_dt = final_DT_RF("DT", best_dt["params"], Xtrv, ytrv, Xte, yte, cfg["output_dir"]) if best_dt["params"] else None
    eval_rf = final_DT_RF("RF", best_rf["params"], Xtrv, ytrv, Xte, yte, cfg["output_dir"]) if best_rf["params"] else None
    eval_fcn = final_FCN(best_fcn["params"], Xtrv, ytrv, Xte, yte, cfg["output_dir"]) if best_fcn["params"] else None

    rows = []
    if eval_dt: rows.append({"model": "DecisionTree", **eval_dt, "best_params": json.dumps(best_dt["params"])})
    if eval_rf: rows.append({"model": "RandomForest", **eval_rf, "best_params": json.dumps(best_rf["params"])})
    if eval_fcn: rows.append({"model": "FCN", **eval_fcn, "best_params": json.dumps(best_fcn["params"])})

    final_csv = os.path.join(cfg["output_dir"], "final_eval_all_models_single_gpu.csv")
    pd.DataFrame(rows).to_csv(final_csv, index=False)
    print(f"Saved summary -> {final_csv}")

if __name__ == "__main__":
    main()
