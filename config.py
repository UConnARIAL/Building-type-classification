import os

CONFIG = {
    # =====================================================
    # General metadata
    # =====================================================
    "run_name": "inference",
    "seed": 42,

    # =====================================================
    # Data and I/O paths
    # =====================================================
    "data_path": "/work2/08968/eliasm1/frontera/GNN/data/All_OSM_HABITAT_building_node_features.csv",
    "output_model_path": "/work2/08968/eliasm1/frontera/GNN/models",
    "output_results_path": "/work2/08968/eliasm1/frontera/GNN/results",
    "split_path": "/work2/08968/eliasm1/frontera/GNN/results/splits_GraphSAGE.npz",

    # =====================================================
    # Feature and coordinate definitions
    # =====================================================
    "feature_cols": [
        "area", "perim", "length", "width", "elong", "aspect", "compact", "border_ind",
        "shape_ind", "asymmetry", "max_radius", "min_radius", "circularity",
        "density", "complexity", "main_direction", "n_vertices", "vertices_per_area"
    ],
    "spatial_cols": ["centroid_x", "centroid_y"],

    # =====================================================
    # Model definitions and hyperparameter search spaces
    # =====================================================
    "model": "GraphSAGE",
    "architectures": {
        "GraphTransformer": {
            "param_space": {
                "hidden_dim": [64, 128],
                "num_heads": [4, 8],
                "num_layers": [2, 3, 4],
                "dropout": [0.1, 0.3, 0.5],
                "lr": [1e-4, 5e-4, 1e-3]
            }
        },
        "GAT": {
            "param_space": {
                "hidden_dim": [64, 128],
                "heads": [4, 8],
                "num_layers": [2, 3, 4],
                "dropout": [0.1, 0.3, 0.5],
                "lr": [1e-4, 5e-4, 1e-3]
            }
        },
        "GCN": {
            "param_space": {
                "hidden_dim": [32, 64, 128],
                "num_layers": [2, 3, 4],
                "dropout": [0.1, 0.3, 0.5],
                "lr": [1e-4, 5e-4, 1e-3]
            }
        },
        "GraphSAGE": {
            "param_space": {
                "hidden_dim": [32, 64, 128],
                "num_layers": [2, 3, 4],
                "dropout": [0.1, 0.3, 0.5],
                "lr": [1e-4, 5e-4, 1e-3]
            }
        },
    },

    # =====================================================
    # Training and tuning configuration
    # =====================================================
    "split_fracs": [0.7, 0.2, 0.1],          # train, val, test fractions
    "batch_size": 256,
    "num_neighbors_default": 10,
    "loader_num_workers": 0,
    "knn_k": 10,
    "max_epochs": 100,
    "early_stopping_patience": 15,
    "n_trials": 20,

    # =====================================================
    # Fine-tuning configuration
    # =====================================================
    # Pretrained model (OSM-trained)
    "fine_tune_pretrained_path": "/work2/08968/eliasm1/frontera/GNN/models/model_best_GraphSAGE_GraphSAGE.pt",

    # Output paths for fine-tuning results
    "fine_tune_model_path": "/work2/08968/eliasm1/frontera/GNN/models/model_finetuned_GraphSAGE_DL.pt",
    "fine_tune_metrics_path": "/work2/08968/eliasm1/frontera/GNN/results/fine_tune_GraphSAGE_DL_metrics.csv",

    # Best hyperparameters from tuning (used for fine-tuning)
    "fine_tune_params": {
        "hidden_dim": 128,
        "num_layers": 4,
        "dropout": 0.3,
        "lr": 1e-4
    },

    # Fine-tuning control
    "fine_tune_epochs": 50,
    "fine_tune_patience": 15,

    # Optionally freeze lower layers for domain transfer
    "fine_tune_freeze_layers": False,
    "fine_tune_freeze_depth": 3,


    # =====================================================
    # Optional experiment notes / provenance
    # =====================================================
    "notes": "This config unifies OSM-based tuning, final GraphSAGE training, and DL-domain fine-tuning for domain adaptation."
}

# Ensure output directories exist
os.makedirs(CONFIG["output_model_path"], exist_ok=True)
os.makedirs(CONFIG["output_results_path"], exist_ok=True)
