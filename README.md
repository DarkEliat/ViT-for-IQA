# ViT-for-IQA

**Vision Transformer for Full-Reference Image Quality Assessment**

A deep learning-based image quality assessment system using Vision Transformer (ViT) architecture. This project implements a full-reference IQA model that predicts perceptual quality scores by comparing distorted images against their pristine references.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Supported Datasets](#supported-datasets)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
  - [Download Datasets](#1-download-datasets)
  - [Create Experiment](#2-create-experiment)
  - [Run Training](#3-run-training)
  - [Run Prediction & Evaluation](#4-run-evaluation)
- [Configuration System](#configuration-system)
- [Project Structure](#project-structure)
- [Evaluation Metrics](#evaluation-metrics)
- [Technical Details](#technical-details)
- [Requirements](#requirements)
- [Author](#author)

---

## <a id="overview"></a>🎯 Overview

Image Quality Assessment (IQA) is crucial in many computer vision applications. This project leverages the power of Vision Transformers to accurately predict perceptual quality scores for distorted images in a full-reference (FR-IQA) setting.

**What is Full-Reference IQA?**

- Takes both a pristine **reference image** and its **distorted version** as input.
- Predicts a **quality score** that correlates with human perception (MOS/DMOS).
- Useful for evaluating compression algorithms, transmission systems, and image processing pipelines.

---

## <a id="key-features"></a>✨ Key Features

- **🏗️ Modern Architecture**: Utilizes pretrained Vision Transformer (ViT) backbones from `timm`.
- **🛠️ Powerful CLI**: Unified command-line interface for all tasks (training, evaluation, prediction).
- **⚙️ Modular Configuration**: Split configuration system (`dataset`, `model`, `training`) for maximum flexibility.
- **🔄 Cross-Dataset Evaluation**: Train on one dataset (e.g., KADID-10k) and evaluate on another (e.g., TID2013) seamlessly.
- **📊 Robust Metrics**: Implements standard IQA metrics (PLCC, SRCC, KRCC) with **5-parameter logistic regression** for PLCC.
- **💾 Advanced Checkpointing**: Automatic saving of best models based on SRCC improvement.
- **📈 Logging**: Full TensorBoard integration and structured Markdown/JSON reports.
- **📥 Auto-Download**: Utility script to automatically fetch and organize supported datasets.

---

## <a id="architecture"></a>🏛️ Architecture

The model architecture consists of:

1.  **Backbone**: Pretrained Vision Transformer (default: `vit_base_patch16_224` from `timm`).
    *   Processes both reference and distorted images independently (Siamese network style with shared weights).
    *   Extracts 768-dimensional embeddings for each image.
2.  **Regression Head**:
    *   Concatenates embeddings: `[reference_emb, distorted_emb]` → 1536 dimensions.
    *   Fully-connected layers: `Linear(1536→512) → ReLU → Linear(512→1)`.
    *   Outputs a single quality score.
3.  **Unified Quality Score**:
    *   All internal processing uses a normalized `[0, 1]` score range (where 1.0 = best quality).
    *   Automatically handles MOS (Mean Opinion Score) and DMOS (Differential Mean Opinion Score) conversions.

```mermaid
graph TD
    Ref[Reference Image] --> ViT_Shared[ViT Backbone\n(Shared Weights)]
    Dist[Distorted Image] --> ViT_Shared
    
    ViT_Shared -->|768-d| EmbRef[Reference Embedding]
    ViT_Shared -->|768-d| EmbDist[Distorted Embedding]
    
    EmbRef --> Concat[Concatenation\n(1536-d)]
    EmbDist --> Concat
    
    Concat --> Head[Regression Head\nFC 1536 -> 512 -> 1]
    Head --> Score[Quality Score\nRange 0-1]
```

---

## <a id="supported-datasets"></a>📚 Supported Datasets

The project supports automatic downloading and loading of the following datasets (defined in `src/datasets/dataset_map.py`):

| Dataset Name | Config Name | Reference Images | Distorted Images | Score Type |
| :--- | :--- | :--- | :--- | :--- |
| **KADID-10k** | `kadid10k` | 81 | 10,125 | MOS |
| **TID2013** | `tid2013` | 25 | 3,000 | MOS |
| **TID2008** | `tid2008` | 25 | 1,700 | MOS |
| **LIVE** | `live` | 80 | 320 | MOS |

**Unified Score Logic**:
- **MOS (e.g., TID2013)**: Normalized to `[0, 1]`.
- **DMOS**: Inverted and normalized to `[0, 1]` so that higher is always better.

---

## <a id="installation"></a>🚀 Installation

### Prerequisites

- **Python 3.12+**
- **CUDA-capable GPU** (highly recommended)
- **Poetry** (package manager)

### Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/ViT-for-IQA.git
    cd ViT-for-IQA
    ```

2.  **Activate the environment**:
    ```bash
    poetry env activate
    ```

3.  **Install dependencies**:
    ```bash
    poetry install
    ```

---

## <a id="quick-start"></a>⚡ Quick Start

### <a id="1-download-datasets"></a>1. Download Datasets

Use the included helper script to download and extract KADID-10k, TID2013, and TID2008 automatically:

```bash
poetry run python -m scripts.download_datasets
```

### <a id="2-create-experiment"></a>2. Create an Experiment

Create a new experiment structure based on a predefined training configuration:

```bash
poetry run python -m scripts.create_experiment \
    --experiment-name "my_first_run" \
    --training-config-name "training_tid2013_vit_base_patch16_224_baseline.yaml"
```

### <a id="3-run-training"></a>3. Run Training

Start the training loop:

```bash
poetry run python -m scripts.run_training \
    --experiment-path "tid2013/my_first_run/"
```

### <a id="4-run-evaluation"></a>4. Predict & Evaluate

Use prediction and evaluate the best model checkpoint on the test split:

```bash
poetry run python -m scripts.run_prediction \
    --checkpoint-path "tid2013/my_first_run/checkpoints/best.pth" \
    --split-name "test"
```

```bash
poetry run python -m scripts.run_evaluation \
    --checkpoint-path "tid2013/my_first_run/checkpoints/best.pth" \
    --split-name "test"
```

---

## <a id="cli-reference"></a>📖 CLI Reference

The project functionality is exposed via scripts in the `scripts/` directory, which wrap the CLI commands defined in `src/cli/`.

### Common Arguments
- Paths are relative to the project root or `experiments/` directory where applicable.
- Use `--help` on any script to see full options.
- `--skip-checkpoint-consistency-check`: Skips validation of checkpoint metadata (available in training, prediction, evaluation).

### `create_experiment.py`

Creates a new experiment workspace with configs, logs, and split indices.

**Modes:**
1.  **From Config** (New Training):
    ```bash
    poetry run python -m scripts.create_experiment \
        --experiment-name <experiment_name> \
        --training-config-name training_<dataset_name>_vit_base_patch16_224_baseline.yaml
    ```
2.  **From Checkpoint** (Resume/Fine-tune):
    ```bash
    poetry run python -m scripts.create_experiment \
        --experiment-name <experiment_name> \
        --checkpoint-path <dataset_name>/<experiment_name>/checkpoints/best.pth
    ```

### `run_training.py`

Executes the training pipeline for a given experiment.

```bash
poetry run python -m scripts.run_training \
    --experiment-path <dataset_name>/<experiment_name>/
```
*   **Resuming**: Automatically detects existing checkpoints in the experiment folder and asks to resume.

### `run_evaluation.py`

Calculates performance metrics (PLCC, SRCC, KRCC, RMSE).

**Modes:**
1.  **Split Evaluation** (Evaluate on a split of the training dataset):
    ```bash
    poetry run python -m scripts.run_evaluation \
        --checkpoint-path <dataset_name>/<experiment_name>/checkpoints/best.pth \
        --split-name <split_name>  # Options: train, validation, test
    ```
2.  **Cross-Dataset Evaluation** (Evaluate on a completely different dataset):
    ```bash
    poetry run python -m scripts.run_evaluation \
        --checkpoint-path <dataset_name>/<experiment_name>/checkpoints/best.pth \
        --dataset-name <dataset_name>  # Options: live, tid2008, tid2013, kadid10k
    ```

### `run_prediction.py`

Runs inference and outputs raw predicted scores.

**Modes:**
1.  **Split Evaluation** (Predict on a split of the training dataset):
    ```bash
    poetry run python -m scripts.run_prediction \
        --checkpoint-path <dataset_name>/<experiment_name>/checkpoints/best.pth \
        --split-name <split_name>  # Options: train, validation, test
    ```
2.  **Cross-Dataset Evaluation** (Predict on a full completely different dataset):
    ```bash
    poetry run python -m scripts.run_predicition \
        --checkpoint-path <dataset_name>/<experiment_name>/checkpoints/best.pth \
        --dataset-name <dataset_name>  # Options: live, tid2008, tid2013, kadid10k
    ```

---

## <a id="configuration-system"></a>⚙️ Configuration System

The configuration is modular, split into three YAML files found in `configs/` (source) and `experiments/<dataset_name>/<experiment_name>/configs/` (runtime):

### 1. Dataset Config (`dataset_*.yaml`)
Defines image paths, counts, and original score characteristics (MOS/DMOS ranges).
*   *Example:* `configs/dataset_kadid10k.yaml`

### 2. Model Config (`model_*.yaml`)
Defines the network architecture and input requirements.
*   *Example:* `configs/model_vit_base_patch16_224.yaml`

### 3. Training Config (`training_*.yaml`)
Links a Model to a Dataset and defines the training hyperparameters.
*   *Example:* `configs/training_kadid10k_vit_base_patch16_224_baseline.yaml`

---

## <a id="project-structure"></a>📂 Project Structure

```text
ViT-for-IQA/
├── configs/                              # Source Global Configurations
│   ├── dataset_*.yaml                    # Dataset definitions
│   ├── model_*.yaml                      # Model architecture definitions
│   └── training_*.yaml                   # Training scenarios
├── datasets/                             # Raw Datasets (downloaded here)
├── experiments/                          # Experiment Workspaces
│   └── <dataset_name>/
│       └── <experiment_name>/
│           ├── configs/                  # Frozen configs for this run
│           ├── checkpoints/              # Saved models (.pth)
│           ├── logs/                     # TensorBoard & Text logs
│           ├── splits/                   # Deterministic split indices (.csv)
│           └── metrics.md                # Evaluation report
├── scripts/                              # CLI Entrypoints
├── src/                                  # Source Code
│   ├── cli/                              # CLI Implementation & Validators
│   ├── configs/                          # Config Loading & Logic
│   ├── datasets/                         # Data Loading & Splitting
│   ├── evaluation/                       # Metrics (PLCC, SRCC, KRCC)
│   ├── experiments/                      # Experiment Management
│   ├── inference/                        # Prediction Logic
│   ├── models/                           # ViT Implementation
│   ├── training/                         # Trainer Loop
│   └── utils/                            # Helpers
└── pyproject.toml                        # Dependencies
```

---

## <a id="evaluation-metrics"></a>📊 Evaluation Metrics

The system computes correlation metrics between the **Ground Truth** (subjective human scores) and **Predicted Scores**:

1.  **PLCC (Pearson Linear Correlation Coefficient)**:
    *   Measures linear correlation.
    *   Computed **after** fitting a 5-parameter logistic function (Sheikh et al.) to the predictions to account for nonlinearity.
2.  **SRCC (Spearman Rank Correlation Coefficient)**:
    *   Measures monotonic relationship (rank order).
    *   Primary metric used for monitoring "best" checkpoints.
3.  **KRCC (Kendall Rank Correlation Coefficient)**:
    *   Another rank-based correlation metric.

**Results Output**:
Results are saved to `experiments/<dataset_name>/<experiment_name>/metrics.md` (Markdown report), `.json` (Machine readable), and `.csv` (Data analysis).

---

## <a id="technical-details"></a>🔧 Technical Details

### Data Splitting Strategy
To prevent **Data Leakage**, splitting is performed by **Reference Image**:
1.  All reference images are shuffled deterministically (seed-based).
2.  References are assigned to Train, Validation, or Test sets.
3.  All distorted versions of a specific reference image are forced into the same set as their reference.
4.  This ensures the model never sees a distortion of a "Test" reference image during training.

### Checkpointing
Saved `.pth` files are pickle objects (`CheckpointPickle`) containing:
- `model_state_dict`: Weights.
- `optimizer_state_dict`: Optimizer state.
- `best_epoch`: Statistics of the best epoch so far.
- `last_epoch`: Statistics of the current epoch.
- `app_version`: App version string (e.g., `0.1.2`).
- `dataset_config`: Dataset configuration used.
- `model_config`: Model configuration used.
- `training_config`: Training configuration used.

---

## <a id="requirements"></a>📦 Requirements

*   **torch**, **torchvision**, **timm** (Deep Learning)
*   **numpy**, **pandas**, **scipy** (Data & Metrics)
*   **Pillow**, **opencv-python** (Image Processing)
*   **pyyaml** (Configuration)
*   **tensorboard** (Logging)

See `pyproject.toml` for exact version constraints.

---

## <a id="author"></a>✍️ Author

**Jan Kostka (aka DarkEliat)**
