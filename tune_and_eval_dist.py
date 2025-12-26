import os
import time
import random
import multiprocessing as mp
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
import sys

# Ensure real-time stdout printing
sys.stdout.reconfigure(line_buffering=True)

from config import CONFIG
from models import GraphTransformer, GraphAttentionNetwork as GAT, GCN, GraphSAGE
from graph_utils import build_knn_graph
from torch_geometric.loader import NeighborLoader

# -----------------------------
# Model registry
# -----------------------------
MODEL_REGISTRY = {
    "GraphTransformer": GraphTransformer,
    "GAT": GAT,
    "GCN": GCN,
    "GraphSAGE": GraphSAGE,
}

# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def sample_params(space: Dict[str, Any]) -> Dict[str, Any]:
    return {k: random.choice(v) for k, v in space.items()}

def split_indices(labeled_idx: np.ndarray, fracs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_frac, val_frac, test_frac = fracs
    N = len(labeled_idx)
    n_train = int(train_frac * N)
    n_val = int(val_frac * N)
    train_idx = labeled_idx[:n_train]
    val_idx = labeled_idx[n_train:n_train + n_val]
    test_idx = labeled_idx[n_train + n_val:]
    return train_idx, val_idx, test_idx

def build_neighbor_loaders(data, train_idx, val_idx, test_idx, num_layers, batch_size, num_neighbors_each_layer, num_workers=0):
    train_nodes = torch.tensor(train_idx, dtype=torch.long)
    val_nodes = torch.tensor(val_idx, dtype=torch.long)
    test_nodes = torch.tensor(test_idx, dtype=torch.long)
    num_neighbors = [num_neighbors_each_layer] * num_layers

    train_loader = NeighborLoader(data, input_nodes=train_nodes, num_neighbors=num_neighbors,
                                  batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = NeighborLoader(data, input_nodes=val_nodes, num_neighbors=num_neighbors,
                                batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = NeighborLoader(data, input_nodes=test_nodes, num_neighbors=num_neighbors,
                                 batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader

@torch.no_grad()
def evaluate_minibatch(model, loader, device):
    model.eval()
    preds_all, trues_all = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        out = torch.sigmoid(logits[:batch.batch_size])
        preds_all.append((out >= 0.5).cpu().numpy())
        trues_all.append(batch.y[:batch.batch_size].cpu().numpy())
    preds_all = np.concatenate(preds_all)
    trues_all = np.concatenate(trues_all)
    p = precision_score(trues_all, preds_all)
    r = recall_score(trues_all, preds_all)
    f = f1_score(trues_all, preds_all)
    return p, r, f

def train_minibatch_keep_best_on_val(model, train_loader, val_loader, device, epochs, lr, patience):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_f1, best_state, best_metrics = -1.0, None, (0.0, 0.0, 0.0)
    no_improve_counter = 0
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            out = logits[:batch.batch_size]
            y = batch.y[:batch.batch_size]
            loss = F.binary_cross_entropy_with_logits(out, y)
            loss.backward()
            optimizer.step()
        p, r, f = evaluate_minibatch(model, val_loader, device)
        if f > best_f1:
            best_f1, best_metrics = f, (p, r, f)
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            no_improve_counter = 0
        else:
            no_improve_counter += 1
            if no_improve_counter >= patience:
                print(f"[GPU {device}] Early stopping on val after {epoch + 1} epochs.", flush=True)
                break
    if best_state:
        model.load_state_dict(best_state)
    return model, best_metrics, best_state

def train_minibatch_keep_best_on_trainloss(model, train_loader, device, epochs, lr, patience):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss, best_state = float('inf'), None
    no_improve_counter = 0
    for epoch in tqdm(range(epochs), desc="Final training (train loss, early stop)", dynamic_ncols=True):
        model.train()
        running_loss, n_batches = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            out = logits[:batch.batch_size]
            y = batch.y[:batch.batch_size]
            loss = F.binary_cross_entropy_with_logits(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        epoch_loss = running_loss / max(n_batches, 1)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            no_improve_counter = 0
        else:
            no_improve_counter += 1
            if no_improve_counter >= patience:
                print(f"Early stopping on train loss after {epoch + 1} epochs.", flush=True)
                break
    if best_state:
        model.load_state_dict(best_state)
    return model, best_loss, best_state

def worker(gpu_id: int,
           trial_indices: List[int],
           return_dict,
           seed_base: int,
           train_idx: np.ndarray,
           val_idx: np.ndarray,
           test_idx: np.ndarray):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    cfg = CONFIG

    # Different seed per GPU for HP sampling, BUT NOT for the split (which is passed in).
    set_seed(cfg["seed"] + seed_base + gpu_id)

    df = pd.read_csv(cfg["data_path"])
    df['label'] = df['label'].fillna(-1).astype(int)
    X = df[cfg["feature_cols"]].values.astype(np.float32)
    coords = df[cfg["spatial_cols"]].values.astype(np.float32)
    y = df['label'].values.astype(int)

    data = build_knn_graph(X, coords, y, k=cfg["knn_k"])

    arch = cfg["model"]
    model_cls = MODEL_REGISTRY[arch]
    space = cfg["architectures"][arch]["param_space"]
    local_trials = []
    best_local = {"f1": -1.0, "params": None}

    print(f"[GPU {gpu_id}] Running trials: {trial_indices}", flush=True)

    for t in trial_indices:
        hp = sample_params(space)
        lr = hp.pop("lr")
        print(f"[GPU {gpu_id}] Starting trial {t} with hyperparameters: {hp}, lr={lr}", flush=True)
        model_kwargs = {"in_channels": X.shape[1]}
        if arch == "GraphTransformer":
            model_kwargs.update({"hidden_dim": hp["hidden_dim"], "num_heads": hp["num_heads"],
                                 "num_layers": hp["num_layers"], "dropout": hp["dropout"]})
            num_layers_for_loader = hp["num_layers"]
        elif arch == "GAT":
            model_kwargs.update({"hidden_dim": hp["hidden_dim"], "heads": hp["heads"],
                                 "num_layers": hp["num_layers"], "dropout": hp["dropout"]})
            num_layers_for_loader = hp["num_layers"]
        else:
            model_kwargs.update({"hidden_dim": hp["hidden_dim"], "num_layers": hp["num_layers"],
                                 "dropout": hp["dropout"]})
            num_layers_for_loader = hp["num_layers"]

        train_loader, val_loader, _ = build_neighbor_loaders(
            data, train_idx, val_idx, test_idx,
            num_layers_for_loader, cfg["batch_size"],
            cfg["num_neighbors_default"], cfg["loader_num_workers"]
        )
        model = model_cls(**model_kwargs).to(device)
        model, (p, r, f), _ = train_minibatch_keep_best_on_val(
            model, train_loader, val_loader, device, cfg["max_epochs"], lr, cfg["early_stopping_patience"]
        )
        local_trials.append({"trial": t, **hp, "lr": lr, "precision_val": p, "recall_val": r, "f1_val": f})
        if f > best_local["f1"]:
            best_local.update({"f1": f, "params": {**hp, "lr": lr}})
        print(f"[GPU {gpu_id}] Trial {t}: F1={f:.4f} (P={p:.4f}, R={r:.4f})", flush=True)

    return_dict[gpu_id] = {"trials": local_trials, "best": best_local}
    print(f"[GPU {gpu_id}] Done. Local best F1={best_local['f1']:.4f}", flush=True)

def distribute_trials(total_trials: int, n_gpus: int) -> List[List[int]]:
    indices = list(range(total_trials))
    splits = [[] for _ in range(n_gpus)]
    for i, idx in enumerate(indices):
        splits[i % n_gpus].append(idx)
    return splits

def main():
    cfg = CONFIG
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required but not available.")
    n_gpus = torch.cuda.device_count()
    if n_gpus < 1:
        raise RuntimeError(f"Expected >=1 GPU, found {n_gpus}.")

    # -----------------------------
    # Load data ONCE, build a SINGLE deterministic split
    # -----------------------------
    set_seed(cfg["seed"])  # <-- THIS guarantees deterministic split everywhere

    df = pd.read_csv(cfg["data_path"])
    df['label'] = df['label'].fillna(-1).astype(int)

    X = df[cfg["feature_cols"]].values.astype(np.float32)
    coords = df[cfg["spatial_cols"]].values.astype(np.float32)
    y = df['label'].values.astype(int)
    mask_labeled = y != -1

    labeled_idx = np.where(mask_labeled)[0]
    np.random.shuffle(labeled_idx)
    train_idx, val_idx, test_idx = split_indices(labeled_idx, cfg["split_fracs"])

    # Save split for audit/reuse
    split_path = os.path.join(cfg["output_results_path"], f"splits_{cfg['run_name']}.npz")
    np.savez(split_path, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, labeled_idx=labeled_idx)
    print(f"Saved deterministic split to {split_path}", flush=True)

    # -----------------------------
    # Tuning (multi-GPU)
    # -----------------------------
    total_trials = cfg["n_trials"]
    splits = distribute_trials(total_trials, n_gpus)

    print(f"Distributing {total_trials} trials across {n_gpus} GPUs:")
    for gpu_id, chunk in enumerate(splits):
        print(f"  GPU {gpu_id}: {chunk}")

    manager = mp.Manager()
    return_dict = manager.dict()
    procs = []

    for gpu_id in range(n_gpus):
        p = mp.Process(
            target=worker,
            args=(gpu_id, splits[gpu_id], return_dict, 12345, train_idx, val_idx, test_idx)
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    # -----------------------------
    # Merge tuning results
    # -----------------------------
    all_trials = []
    best_overall = {"f1": -1.0, "params": None}
    for gpu_id in range(n_gpus):
        gpu_out = return_dict.get(gpu_id, None)
        if gpu_out is None:
            raise RuntimeError(f"No results returned for GPU {gpu_id}")
        all_trials.extend(gpu_out["trials"])
        if gpu_out["best"]["f1"] > best_overall["f1"]:
            best_overall = gpu_out["best"]

    arch = cfg["model"]
    tuning_csv = os.path.join(cfg["output_results_path"], f"tuning_results_{cfg['run_name']}_{arch}.csv")
    pd.DataFrame(all_trials).to_csv(tuning_csv, index=False)
    print(f"Saved merged tuning results to {tuning_csv}", flush=True)
    print(f"Best overall val F1={best_overall['f1']:.4f}, params={best_overall['params']}", flush=True)

    # -----------------------------
    # FINAL TRAINING & TEST EVAL (single GPU, e.g. cuda:0)
    # -----------------------------
    print("\n===== Final training on train+val and evaluation on test =====", flush=True)

    # Re-build the graph cleanly
    data = build_knn_graph(X, coords, y, k=cfg["knn_k"])

    full_train_idx = np.concatenate([train_idx, val_idx])

    best_hp = best_overall["params"].copy()
    best_lr = best_hp.pop("lr")

    model_cls = MODEL_REGISTRY[arch]
    model_kwargs = {"in_channels": X.shape[1]}
    if arch == "GraphTransformer":
        model_kwargs.update({
            "hidden_dim": best_hp["hidden_dim"],
            "num_heads": best_hp["num_heads"],
            "num_layers": best_hp["num_layers"],
            "dropout": best_hp["dropout"],
        })
        num_layers_for_loader = best_hp["num_layers"]
    elif arch == "GAT":
        model_kwargs.update({
            "hidden_dim": best_hp["hidden_dim"],
            "heads": best_hp["heads"],
            "num_layers": best_hp["num_layers"],
            "dropout": best_hp["dropout"],
        })
        num_layers_for_loader = best_hp["num_layers"]
    else:
        model_kwargs.update({
            "hidden_dim": best_hp["hidden_dim"],
            "num_layers": best_hp["num_layers"],
            "dropout": best_hp["dropout"],
        })
        num_layers_for_loader = best_hp["num_layers"]

    full_train_loader, _, test_loader = build_neighbor_loaders(
        data=data,
        train_idx=full_train_idx,
        val_idx=val_idx,     # not used in final train loop, but required by API
        test_idx=test_idx,
        num_layers=num_layers_for_loader,
        batch_size=cfg["batch_size"],
        num_neighbors_each_layer=cfg["num_neighbors_default"],
        num_workers=cfg["loader_num_workers"],
    )

    final_device = torch.device("cuda:0")
    best_model = model_cls(**model_kwargs).to(final_device)

    start_time = time.time()
    best_model, _, _ = train_minibatch_keep_best_on_trainloss(
        best_model, full_train_loader, final_device, cfg["max_epochs"], best_lr, cfg["early_stopping_patience"]
    )
    runtime_sec = time.time() - start_time

    p_test, r_test, f_test = evaluate_minibatch(best_model, test_loader, final_device)
    print(f"FINAL TEST -> P={p_test:.4f} R={r_test:.4f} F1={f_test:.4f}", flush=True)
    print(f"Final training runtime (s): {runtime_sec:.2f}", flush=True)

    # Save final model + metrics
    model_path = os.path.join(cfg["output_model_path"], f"model_best_{cfg['run_name']}_{arch}.pt")
    torch.save(best_model.state_dict(), model_path)

    final_eval_path = os.path.join(cfg["output_results_path"], f"final_eval_{cfg['run_name']}_{arch}.csv")
    pd.DataFrame([{
        "arch": arch,
        "precision_test": p_test,
        "recall_test": r_test,
        "f1_test": f_test,
        "training_runtime_sec": runtime_sec,
        "best_params": str(best_overall["params"]),
        "split_path": split_path
    }]).to_csv(final_eval_path, index=False)

    print(f"Saved best model to {model_path}", flush=True)
    print(f"Saved final test metrics to {final_eval_path}", flush=True)
    print(f"Best val F1 during tuning: {best_overall['f1']:.4f}", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
