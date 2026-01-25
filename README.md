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
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Creating Experiments](#creating-experiments)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Inference](#inference)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Technical Details](#technical-details)
- [Requirements](#requirements)
- [License](#license)
- [Author](#author)

---

## 🎯 Overview

Image Quality Assessment (IQA) is crucial in many computer vision applications. This project leverages the power of Vision Transformers to accurately predict perceptual quality scores for distorted images in a full-reference (FR-IQA) setting.

**What is Full-Reference IQA?**

- Takes both a pristine reference image and its distorted version as input
- Predicts a quality score that correlates with human perception
- Useful for evaluating compression algorithms, transmission systems, and image processing pipelines

---

## ✨ Key Features

- **🏗️ Modern Architecture**: Utilizes pretrained Vision Transformer (ViT) backbones from `timm`
- **📊 Multiple Datasets**: Supports KADID-10k, TID2008, TID2013, and LIVE databases
- **🔬 Rigorous Evaluation**: Implements standard IQA metrics (PLCC, SRCC, KRCC)
- **⚙️ Flexible Configuration**: YAML-based configuration system with extensive validation
- **💾 Experiment Management**: Organized experiment structure with automatic checkpointing
- **🔄 Resume Training**: Intelligent checkpoint loading and training resumption
- **📈 Logging**: TensorBoard integration for training visualization
- **🎯 Smart Data Splitting**: Reference-based splitting to prevent data leakage

---

## 🏛️ Architecture

The model architecture consists of:

1. **Backbone**: Pretrained Vision Transformer (default: `vit_base_patch16_224`)

   - Processes both reference and distorted images independently
   - Extracts 768-dimensional embeddings for each image
2. **Regression Head**:

   - Concatenates embeddings: `[reference_emb, distorted_emb]` → 1536 dimensions
   - Fully-connected layers: `Linear(1536→512) → ReLU → Linear(512→1)`
   - Outputs a single quality score
3. **Training**:

   - Loss function: Mean Squared Error (MSE)
   - Optimizer: Adam
   - Configurable learning rate, batch size, and epochs

```
┌─────────────────┐     ┌─────────────────┐
│ Reference Image │     │ Distorted Image │
│   (224×224×3)   │     │   (224×224×3)   │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────┐       ┌───────┘
                 ▼       ▼
         ┌───────────────────────┐
         │   ViT Backbone        │
         │   (Shared Weights)    │
         └───────┬───────────────┘
                 │
         [768-d] │ [768-d]
                 │
                 ▼
         ┌───────────────────────┐
         │   Concatenation       │
         │      (1536-d)         │
         └───────┬───────────────┘
                 │
                 ▼
         ┌───────────────────────┐
         │   Regression Head     │
         │   FC(1536→512→1)      │
         └───────┬───────────────┘
                 │
                 ▼
         ┌───────────────────────┐
         │  Quality Score [0,1]  │
         └───────────────────────┘
```

---

## 📚 Supported Datasets

| Dataset             | Reference Images | Distorted Images | Score Type | Range    |
| ------------------- | ---------------- | ---------------- | ---------- | -------- |
| **KADID-10k** | 81               | 10,125           | DMOS       | [1, 5]   |
| **TID2008**   | 25               | 1,700            | MOS        | [0, 9]   |
| **TID2013**   | 25               | 3,000            | MOS        | [0, 9]   |
| **LIVE**      | 80               | 320              | MOS        | [1, 100] |

**Note**: All scores are automatically normalized to [0, 1] for training.

---

## 🚀 Installation

### Prerequisites

- Python 3.12.x
- CUDA-capable GPU (recommended)
- Poetry (package manager)

### Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/ViT-for-IQA.git
   cd ViT-for-IQA
   ```
2. **Install dependencies**:

   ```bash
   poetry install
   ```
3. **Activate the environment**:

   ```bash
   poetry env activate
   ```
4. **Download datasets** (manual):

   - Place datasets in the `datasets/` directory
   - Expected structure:
     ```
     datasets/
     ├── kadid10k/
     │   ├── images/
     │   └── dmos.csv
     ├── tid2008/
     │   ├── reference_images/
     │   ├── distorted_images/
     │   └── mos_with_names.txt
     ├── tid2013/
     │   ├── reference_images/
     │   ├── distorted_images/
     │   └── mos_with_names.txt
     └── live/
         ├── Images/
         └── MOS.mat
     ```

---

## ⚡ Quick Start

### 1. Create an Experiment

```bash
python scripts/create_experiment.py
```

This will:

- Prompt you to select a dataset configuration
- Ask for an experiment name
- Generate train/validation/test splits
- Create experiment directory structure

### 2. Train the Model

Edit `scripts/run_training.py` to point to your experiment:

```python
from src.training.trainer import Trainer
from src.utils.paths import EXPERIMENTS_LIVE_PATH

trainer = Trainer(experiment_path=(EXPERIMENTS_LIVE_PATH / 'my_experiment'))
trainer.train()
```

Run training:

```bash
python scripts/run_training.py
```

### 3. Evaluate

Edit `scripts/run_evaluation.py`:

```python
from src.evaluation.evaluator import Evaluator
from src.utils.paths import EXPERIMENTS_LIVE_PATH

evaluator = Evaluator(
    experiment_path=(EXPERIMENTS_LIVE_PATH / 'my_experiment'),
    split_name='test',
    checkpoint_name='last.pth'
)

results = evaluator.evaluate(
    apply_nonlinear_regression_for_plcc=True,
    save_outputs=True
)
```

Run evaluation:

```bash
python scripts/run_evaluation.py
```

---

## 📂 Project Structure

```
ViT-for-IQA/
├── configs/                              # Dataset configurations
│   ├── train_kadid10k_vit_base_patch16_224_baseline.yaml
│   ├── train_live_vit_base_patch16_224_baseline.yaml
│   ├── train_tid2008_vit_base_patch16_224_baseline.yaml
│   └── train_tid2013_vit_base_patch16_224_baseline.yaml
├── datasets/                             # Dataset storage (gitignored)
├── experiments/                          # Experiment outputs (gitignored)
│   └── {dataset_name}/
│       └── {experiment_name}/
│           ├── config.yaml               # Experiment configuration
│           ├── checkpoints/              # Model checkpoints (.pth)
│           ├── logs/
│           │   ├── tensorboard/          # TensorBoard logs
│           │   └── train.log             # Training logs
│           ├── splits/                   # Dataset split indices
│           │   ├── train_indices.csv
│           │   ├── validation_indices.csv
│           │   └── test_indices.csv
│           ├── metrics.json              # Evaluation metrics
│           ├── metrics.csv
│           └── summary.md                # Experiment summary
├── scripts/                              # Executable scripts
│   ├── create_experiment.py              # Create new experiment
│   ├── run_training.py                   # Train model
│   ├── run_evaluation.py                 # Evaluate model
│   └── run_prediction.py                 # Run inference
├── src/                                  # Source code
│   ├── datasets/                         # Dataset loaders
│   │   ├── base_dataset.py               # Abstract base class
│   │   ├── factory.py                    # Dataset factory
│   │   ├── file_map.py                   # File mapping utility
│   │   ├── kadid_dataset.py              # KADID-10k loader
│   │   ├── live_dataset.py               # LIVE loader
│   │   ├── tid_dataset.py                # TID2008/TID2013 loader
│   │   └── splits.py                     # Data splitting logic
│   ├── evaluation/                       # Evaluation tools
│   │   ├── correlation_metrics.py        # IQA metrics (PLCC, SRCC, KRCC)
│   │   └── evaluator.py                  # Evaluation pipeline
│   ├── inference/                        # Inference tools
│   │   └── predictor.py                  # Prediction pipeline
│   ├── models/                           # Model architectures
│   │   └── vit_regressor.py              # ViT-based regressor
│   ├── training/                         # Training logic
│   │   └── trainer.py                    # Training pipeline
│   └── utils/                            # Utilities
│       ├── checkpoints.py                # Checkpoint management
│       ├── configs.py                    # Configuration validation
│       ├── data_types.py                 # Type definitions
│       ├── image_preprocessing.py        # Image preprocessing
│       ├── paths.py                      # Path constants
│       └── quality_scores.py             # Score normalization
├── pyproject.toml                        # Poetry dependencies
├── TODO.md                               # Development roadmap
└── README.md                             # This file
```

---

## ⚙️ Configuration

Configuration files are in YAML format and define all experiment parameters.

### Key Configuration Sections

```yaml
config_name: "live_vit_base_patch16_224_baseline"

app:
  version: "0.1.0"

dataset:
  name: "live"                            # Dataset identifier
  representative_name: "LIVE Wild Compressed Picture Quality Database"
  images:
    reference:
      path: "datasets/live/Images/"
      count: 80
    distorted:
      path: "datasets/live/Images/"
      count: 320
  quality_label:
    type: "mos"                           # "mos" or "dmos"
    min: 1
    max: 100
  labels_path: "datasets/live/MOS.mat"

model:
  name: "vit_base_patch16_224"            # Model from timm
  input:
    image_size:
      width: 224
      height: 224
    keep_original_aspect_ratio: true
  embedding_dimension: 768
  output:
    type: "normalized_mos"                # Output normalization
    min: 0
    max: 1

training:
  splits:
    train: 0.6
    validation: 0.2
    test: 0.2
    random_seed: 42
  batch_size: 8
  num_of_epochs: 5
  learning_rate: 0.0001
  device: "cuda"
  num_of_workers: 4
  early_stopping:
    enabled: false
    max_epochs_without_improvement: 5
    min_improvement_delta: 0.001

checkpointing:
  enabled: true
  save_every_n_epochs: 1
  save_last_epoch: true
  save_best_epoch: true

logging:
  tensorboard: true
```

### Configuration Validation

The system performs extensive validation:

- ✅ File and directory existence checks
- ✅ Value range validation
- ✅ Cross-section consistency (e.g., MOS → normalized_mos)
- ✅ Type checking for all parameters

---

## 📖 Usage

### Creating Experiments

The experiment creation script provides an interactive interface:

```bash
python scripts/create_experiment.py
```

**Options**:

1. Create from scratch using a global config file
2. Create from existing checkpoint (for fine-tuning)

**What it does**:

- Generates reference-based train/val/test splits
- Creates directory structure
- Copies configuration
- Initializes checkpoint (if specified)

### Training

The `Trainer` class handles the complete training pipeline:

```python
from pathlib import Path
from src.training.trainer import Trainer

# Initialize trainer
trainer = Trainer(experiment_path=Path("experiments/live/my_experiment"))

# Start/resume training
trainer.train()
```

**Features**:

- Automatic checkpoint resumption
- TensorBoard logging
- Validation after each epoch
- Configurable checkpoint saving

**Training outputs**:

- `checkpoints/last.pth` - Latest checkpoint
- `checkpoints/epoch_N.pth` - Periodic checkpoints
- `checkpoints/best.pth` - Best performing checkpoint
- `logs/tensorboard/` - TensorBoard logs

### Evaluation

The `Evaluator` class computes IQA metrics:

```python
from src.evaluation.evaluator import Evaluator

evaluator = Evaluator(
    experiment_path=Path("experiments/live/my_experiment"),
    split_name='test',                    # 'train', 'validation', or 'test'
    checkpoint_name='last.pth'
)

results = evaluator.evaluate(
    apply_nonlinear_regression_for_plcc=True,  # Fit 5-parameter logistic
    save_outputs=True                          # Save metrics to files
)

print(f"PLCC: {results.plcc:.4f}")
print(f"SRCC: {results.srcc:.4f}")
print(f"KRCC: {results.krcc:.4f}")
```

### Inference

The `Predictor` class enables predictions on custom data:

```python
from src.inference.predictor import Predictor

predictor = Predictor(
    experiment_path=Path("experiments/live/my_experiment"),
    checkpoint_name='last.pth'
)

# Predict on training dataset
predictions = predictor.predict_on_training_dataset()

# Predict with custom DataLoader
from torch.utils.data import DataLoader
custom_loader = DataLoader(...)
predictions = predictor.predict(data_loader=custom_loader)
```

---

## 📊 Evaluation Metrics

### Correlation Metrics

1. **PLCC (Pearson Linear Correlation Coefficient)**

   - Measures linear correlation with human perception
   - Optional 5-parameter logistic regression fitting
   - Range: [-1, 1], higher is better
2. **SRCC (Spearman Rank Correlation Coefficient)**

   - Measures monotonic relationship
   - More robust to outliers than PLCC
   - Range: [-1, 1], higher is better
3. **KRCC (Kendall Rank Correlation Coefficient)**

   - Alternative rank-based correlation
   - Measures ordinal association
   - Range: [-1, 1], higher is better

### Error Metrics

- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error

### Nonlinear Regression

For PLCC calculation, the system can apply a 5-parameter logistic function:

```
f(x) = β₂ + (β₁ - β₂) / (1 + exp(-(x - β₃) / |β₄|)) + β₅·x
```

This accounts for nonlinear mapping between predicted scores and subjective ratings.

---

## 🏆 Results

Results will vary based on:

- Dataset used
- Number of training epochs
- Model architecture
- Hyperparameters

Example results structure (after evaluation):

```markdown
# Experiment summary

## Identification:
- Dataset: `live`
- Config: `live_vit_base_patch16_224_baseline`
- Split: `test`
- Checkpoint: `last.pth`
- Device: `cuda`
- Number of samples: 64

## Correlation metrics:
- PLCC: `0.9234`
- SRCC: `0.9156`
- KRCC: `0.7891`

## Error metrics:
- MSE: 0.0234
- RMSE: 0.1530
- MAE: 0.1123
```

---

## 🔧 Technical Details

### Data Splitting Strategy

To prevent data leakage, the system splits data **by reference images**:

1. Shuffle reference images deterministically (using random seed)
2. Split references into train/val/test
3. Assign all distortions of each reference to the same split

This ensures that distortions of the same reference never appear in different splits.

### Image Preprocessing

1. **Resize**: Images resized to 224×224
2. **Aspect Ratio**: Option to preserve aspect ratio (with padding)
3. **Normalization**: ToTensor() converts to [0, 1] range
4. **Padding**: Black padding (0, 0, 0) for aspect ratio preservation

### Quality Score Normalization

- **MOS → Unified**: `(value - min) / (max - min)`
- **DMOS → Unified**: `1 - ((value - min) / (max - min))`
- All unified scores in [0, 1] where 1 = best quality

### Checkpoint Structure

Checkpoints contain:

```python
{
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': OrderedDict,
    'train_loss': float,
    'validation_loss': float
}
```

---

## 📦 Requirements

### Core Dependencies

- **torch** - Deep learning framework
- **torchvision** - Computer vision utilities
- **timm** - Pretrained vision models
- **numpy** - Numerical computing
- **pandas** - Data manipulation
- **scipy** - Scientific computing (for metrics)
- **Pillow** - Image processing
- **scikit-image** - Image processing
- **opencv-python** - Computer vision
- **scikit-learn** - Machine learning metrics
- **einops** - Tensor operations
- **safetensors** - Safe tensor serialization
- **pyyaml** - Configuration parsing

### Development Dependencies

- **matplotlib** - Plotting
- **tqdm** - Progress bars
- **tensorboard** - Visualization

### System Requirements

- **Python**: 3.12.x
- **GPU**: CUDA-capable (recommended)
- **RAM**: 16GB+ recommended
- **Storage**: Depends on datasets (KADID-10k ~10GB, others smaller)

---

## 🗺️ Roadmap

### Upcoming Features

- [ ] Automatic dataset downloading
- [ ] Enhanced checkpoint management
- [ ] Early stopping implementation
- [ ] Extended logging to `train.log`
- [ ] Experiment consistency verification module
- [ ] Support for no-reference IQA
- [ ] Additional backbone architectures
- [ ] Hyperparameter optimization tools

---
