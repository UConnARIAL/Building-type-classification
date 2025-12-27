## Download data and setup environment 
1. Log into an HPC system (e.g., TACC Frontera).
2. Create new environment with required libraries using environment.yaml
3. Set up "data", "models", and "results" folders in working directory.
4. Download training and inference CSV files at https://drive.google.com/drive/folders/16ADbxJFL5sefXztszD-ireyr0Ca9m4d1?usp=sharing
   
## Initial hyperparameter tuning experiment
1. Set paths for data input/output in ```config.py (lines 13-16)```. ```OSM_building_node_features.csv``` contains the nodes of all OpenStreetMap building footprints across the Arctic circumpolar permafrost region. This will be used here, since the OSM building type classifier serves as the base model for the classifier that will later be finetuned on footprints derived from deep learning:
   ```
   "data_path": "/.../data/OSM_building_node_features.csv",
   "output_model_path": "/.../models",
   "output_results_path": "/.../results",
   "split_path": "/.../results/splits_GraphSAGE.npz",
   ```

2. Define hyperparameter search space for GraphCNN model selection ```config.py (lines 31-68)```. Leave the code as is if you want to reproduce the authors' findings.
3. Define training and tuning configuration (e.g., train/val/test split ratio, batch size, number of trials) in ```config (lines 72-79)```. Leave the code as is if you want to reproduce the authors' findings.
4. Choose GraphCNN architecture that will be tuned in ```config.py (line 31)```. For example:
   ```
       "model": "GraphSAGE",
   ```
   Pick a relevant experiment name in config.py (line 7). For example:
   ```
       "run_name": "GraphSAGE_tune",
   ```
5. Edit run_tune_and_eval_dist.py to match specific HPC system requirements. Partition (line 28) and allocation (line 29) MUST be set appropriately.
6. Run ```python run_tune_and_eval_dist.py```.
7. Repeat until all architectures have gone through hyperparameter tuning and final model performances on held-out evaluation data are saved to results folder. Take note of model with best performance.

## Model finetuning
1. Noting the model with the best performance, set the path to its weights in ```config.py (line 85)```.
2. We will now finetune the best model to building footprints predicted by a deep learning model (as opposed to those digitized by humans). In ```config.py (line 13)```, change ```"data_path": "/.../OSM_HABITAT_building_node_features.csv"```, as this file contains the deep learning-derived building footprint nodes along with the OSM building footprint nodes. These deep learning-derived footprints took the place and use type label of overlapping OSM footprints.
3. Set paths at which the finetuned model (line 88) and it performance metrics (line 89) will be saved.
4. Specify the hyperparameters from the best performing model in lines 92-97. For example:
   ```
    "fine_tune_params": {
        "hidden_dim": 128,
        "num_layers": 4,
        "dropout": 0.3,
        "lr": 1e-4
    },
   ```
5. Set finetuning options (e.g, epochs, frozen/unfrozen layers) in lines 100-105. Leave the code as is if you want to reproduce the authors' findings.
6. As done before, edit run_tune_and_eval_dist.py to match specific HPC system requirements. Change filename in line 36 from ```tune_and_eval_dist.py``` to ```finetune_GraphSAGE.py```
  
## Classical ML classifier model comparison
Since the classical ML classifiers used here are much lighter than the GraphCNNs assessed above, this experiment can be (and was) run on a local computer. 
1. Similar to the previous experiment, adjust the I/O paths in ```classical_models_tune_and_eval.py (lines 40-45)```.
2. From your HPC system, download the .npz file that contains the indices used to split the data into train/val/test for the GraphCNN model selection experiment. Set ```"split_path"``` in line 44 to this file.
3. Run ```python classical_models_tune_and_eval.py``` to obtain results of classical ML model performance.
