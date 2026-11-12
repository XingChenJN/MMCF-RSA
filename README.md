# MMCF-RSA

MMCF-RSA is a deep learning-based model for predicting the binding affinity between RNA and small molecules.

## Overview

This project implements a multi-modal deep learning framework designed to predict the interaction affinity (e.g., pKd and pIC50 values) between RNAs and small molecules. By integrating sequence, 2D graph, and 3D structural features, MMCF-RSA leverages cross-modal fusion techniques to achieve high-accuracy predictions.

## Model Architecture

The MMCF-RSA model consists of three key components:

1. **RNA Feature Extraction Module**:
   
   * **PointNet Encoder**: Processes RNA 3D point cloud structures.
   * **LSTM Extractor**: Captures RNA sequence features.
   * **Transformer GNN**: Processes RNA 2D graph structures.

2. **Small Molecule Feature Extraction Module**:
   
   * **PointNet Encoder**: Processes small molecule 3D point cloud structures.
   * **LSTM Extractor**: Captures small molecule sequence features.
   * **Transformer GNN**: Processes small molecule 2D graph structures.

3. **Feature Fusion Module**:
   
   * **Multi-layer Fusion Network**: Integrates features from different modalities.
   * **Gating Mechanism**: Adaptively weights different feature inputs.
   * **Attention Mechanism**: Focuses on critical interaction features.

## Requirements

* Python >= 3.8
* PyTorch >= 1.12.0
* PyTorch Geometric >= 2.0.0
* RDKit >= 2022.03.0
* NumPy >= 1.21.0
* Pandas >= 1.4.0
* Scikit-learn >= 1.1.0
* SciPy >= 1.8.0
* Lion-PyTorch >= 0.0.2

## Installation

1. **Clone the repository**:
   
   ```bash
   git clone https://github.com/wfxcode/MMCF-RSA.git
   cd MMCF-RSA
   ```

2. **Create a virtual environment (Recommended)**:
   
   ```bash
   # Linux/Mac
   python -m venv venv
   source venv/bin/activate
   
   # Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**:
   
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

Please organize your data files in the following directory structure:

* `data/RSM_data/`: RNA-small molecule interaction datasets
* `data/R-SIM/R-SIM_RNA_3D/`: RNA 3D point cloud data
* `data/R-SIM/R-SIM_RNA_seq/`: RNA sequence embeddings
* `data/R-SIM/R-SIM_RNA_2D/`: RNA 2D graph data
* `data/R-SIM/R-SIM_drug_3D/`: Small molecule 3D point cloud data
* `data/R-SIM/R-SIM_drug_seq/`: Small molecule sequence embeddings
* `data/R-SIM/R-SIM_drug_2D/`: Small molecule 2D graph data

## Usage

### Training Scripts

Run the appropriate script based on your task:

* **Standard Training (pKd)**:
  
  ```bash
  python main_base.py
  ```
* **IC50 Prediction**:
  
  ```bash
  python main_base_IC50.py
  ```
* **Blind Test Validation**:
  
  ```bash
  python main_blind.py
  ```
* **Independent Testing**:
  
  ```bash
  python main_independent.py
  ```

### Pretrained 10-Fold Cross-Validation Weights

This repository includes the best model checkpoint from each fold of the 10-fold cross-validation experiments. These checkpoints can be used for result reproduction, evaluation, or further development.

* **Kd/pKd prediction**: the checkpoints are available in `model/weight/kd-set 10CV best_model/`.
  Files `model_All_sf2_1_1.pth` through `model_All_sf2_10_1.pth` correspond to folds 1 through 10, respectively.
* **IC50/pIC50 prediction**: the checkpoints are available in `model/weight/IC50-set 10CV best_model/`.
  Files `model_2_1_1.pth` through `model_2_10_1.pth` correspond to folds 1 through 10, respectively.

Each checkpoint contains a PyTorch `state_dict`. Instantiate the model with the same architecture and hyperparameters as the corresponding training script before loading a checkpoint:

```python
weight_path = "model/weight/kd-set 10CV best_model/model_All_sf2_1_1.pth"

model = MMCF_RSA(device).to(device)
state_dict = torch.load(weight_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()
```

For Kd/pKd prediction, use the model definition and settings in `main_base.py`. For IC50/pIC50 prediction, use `main_base_IC50.py` and replace `weight_path` with the appropriate checkpoint in `model/weight/IC50-set 10CV best_model/`. The cross-validation experiments use `n_splits = 10`, `seed_dataset = 2`, and `seed = 1`.

### Hyperparameters

Key hyperparameters can be configured in the training scripts:

* `BATCH_SIZE`: Batch size (Default: 32)
* `EPOCH`: Number of training epochs (Default: 1500)
* `LR`: Learning rate (Default: 1e-3)
* `weight_decay`: Weight decay (Default: 5e-3)
* `dropout`: Dropout rate (Default: 0.5)
* `hidden_dim`: Hidden layer dimension (Default: 256)

# 

## Evaluation Metrics

The model is evaluated using the following metrics:

* **PCC**: Pearson Correlation Coefficient
* **SCC**: Spearman Correlation Coefficient
* **RMSE**: Root Mean Square Error
* **MAE**: Mean Absolute Error
