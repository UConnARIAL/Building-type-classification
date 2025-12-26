## Download data and setup environment 
1. Log into an HPC system (e.g., TACC Frontera).
2. Create new environment with required libraries using environment.yaml
3. Set up "data", "models", and "results" folders in working directory.
4. Download training and inference CSV files at https://drive.google.com/drive/folders/16ADbxJFL5sefXztszD-ireyr0Ca9m4d1?usp=sharing
   
## Initial hyperparameter tuning experiment
1. Set paths for data input/output in config.py (lines 13-16). ```OSM_building_node_features.csv``` contains the nodes of all OpenStreetMap building footprints across the Arctic circumpolar permafrost region. This will be used here, since the OSM building type classifier serves as the base model for the classifier that will later be finetuned on footprints derived from deep learning:
```
"data_path": "/.../data/OSM_building_node_features.csv",
"output_model_path": "/.../models",
"output_results_path": "/.../results",
"split_path": "/.../results/splits_GraphSAGE.npz",
```

2. Define hyperparameter search space for GraphCNN model selection config.py (lines 31-68).
3. Define training and tuning configuration (e.g., train/val/test split ratio, batch size, number of trials) in config (lines 72-79).
4. Choose GraphCNN architecture that will be tuned in config.py (line 31). For example:
   ```
       "model": "GraphSAGE",
   ```
   Pick a relevant experiment name in config.py (line 7). For example:
   ```
       "run_name": "GraphSAGE_tune",
   ```
5. Edit run_tune_and_eval_dist.py to match specific HPC system requirements. Partition (line 28) and allocation (line 29) MUST be set appropriately.
6. Run run_tune_and_eval_dist.py.
7. Repeat until all architectures have gone through hyperparameter tuning and final model performances on held-out evaluation data are saved to results folder. Take note of model with best performance.

## Model finetuning
1. Noting the model with the best performance, set the path to its weights in config.py (line 85).
2. Set paths at which the finetuned model (line 88) and it performance metrics (line 89) will be saved.
3. Specify the hyperparameters from the best performing model in lines 92-97:
4. 
