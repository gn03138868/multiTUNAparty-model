# multiTUNAparty-model

[![PyTorch](https://img.shields.io/badge/PyTorch-2.10.0.dev20251122+cu128-red)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-green)](https://developer.nvidia.com/cuda-toolkit)

A hybrid Multi-Task TransUNet architecture for a precise area recognition tool ya (multiparty).

---

## Packages in Environment

### Core Dependencies

| Package | Version |
|---------|---------|
| Python | 3.11.14 |
| PyTorch | 2.10.0.dev20251122+cu128 |
| torchvision | 0.25.0.dev20251122+cu128 |
| torchaudio | 2.10.0.dev20251122+cu128 |
| CUDA | 12.8 |

### Image Processing and Augmentation

| Package | Version |
|---------|---------|
| albumentations | 2.0.8 |
| opencv-python | 4.11.0.86 |
| opencv-python-headless | 4.11.0.86 |
| Pillow | 12.0.0 |
| scikit-image | 0.26.0 |
| imageio | 2.37.2 |

### Deep Learning and Transformers

| Package | Version |
|---------|---------|
| timm | 1.0.22 |
| transformers | 4.54.1 |
| safetensors | 0.7.0 |
| tokenizers | 0.21.4 |

### Scientific Computing

| Package | Version |
|---------|---------|
| numpy | 1.26.4 |
| scipy | 1.16.3 |
| pandas | 2.3.3 |
| scikit-learn | 1.6.1 |

### Visualisation and Interface

| Package | Version |
|---------|---------|
| matplotlib | 3.9.4 |
| gradio | 6.1.0 |
| streamlit | 1.51.0 |

### Utilities

| Package | Version |
|---------|---------|
| tqdm | 4.67.1 |
| PyYAML | 6.0.3 |
| requests | 2.32.5 |

### Installation

To replicate this environment:

```bash
# Create conda environment
conda create -n multitask python=3.11.14
conda activate multitask

# Install PyTorch with CUDA 12.8
pip install torch==2.10.0.dev20251122+cu128 torchvision==0.25.0.dev20251122+cu128 torchaudio==2.10.0.dev20251122+cu128 --index-url https://download.pytorch.org/whl/nightly/cu128

# Install other dependencies
pip install albumentations==2.0.8 opencv-python==4.11.0.86 pillow==12.0.0
pip install timm==1.0.22 transformers==4.54.1
pip install numpy==1.26.4 scipy==1.16.3 pandas==2.3.3 scikit-learn==1.6.1 scikit-image==0.26.0
pip install matplotlib==3.9.4 gradio==6.1.0 tqdm==4.67.1 pyyaml==6.0.3
```

---

## Introduction

This is an enhanced TransUNet model specifically designed to handle **three markedly different types of biological images**:

1. **Plant cells** - Polygonal cell wall structures
2. **Blood cells** - Circular cells
3. **Others** - Other structures

### Key Improvements

- **Task Conditioning** - The model recognises the type of image currently being processed
- **Multi-scale Feature Fusion (ASPP)** - Captures features at different scales simultaneously
- **Attention Mechanism (CBAM)** - Automatically focuses on important features
- **Task-specific Loss Weighting** - Optimised for different image characteristics
- **Task-balanced Sampling** - Ensures balanced training across all tasks

**Computational Overhead**: Approximately 10-15% increase compared to the original TransUNet

---

## Quick Start

### 1. Preparing Your Data

Two data structure options are available:

#### Option One: Subdirectories by Task (Recommended)

```
data/
├── train/
│   ├── cell/          # Plant cells
│   │   ├── images/
│   │   │   ├── img001.jpg
│   │   │   ├── img002.jpg
│   │   │   └── ...
│   │   └── masks/
│   │       ├── img001.png
│   │       ├── img002.png
│   │       └── ...
│   ├── blood/         # Blood cells
│   │   ├── images/
│   │   └── masks/
│   └── Other/          # Others
│       ├── images/
│       └── masks/
└── val/
    ├── cell/
    │   ├── images/
    │   └── masks/
    ├── blood/
    │   ├── images/
    │   └── masks/
    └── Other/
        ├── images/
        └── masks/
```

#### Option Two: Task Identification from Filenames

```
data/
├── train/
│   ├── images/
│   │   ├── cell_001.jpg    # Filename begins with cell_
│   │   ├── cell_002.jpg
│   │   ├── blood_001.jpg   # Filename begins with blood_
│   │   ├── blood_002.jpg
│   │   ├── Other_001.jpg    # Filename begins with Other_
│   │   └── Other_002.jpg
│   └── masks/
│       ├── cell_001.png
│       ├── cell_002.png
│       ├── blood_001.png
│       ├── blood_002.png
│       ├── Other_001.png
│       └── Other_002.png
└── val/
    ├── images/
    └── masks/
```

### 2. Configuration

Edit `config_multitask.yaml`:

```yaml
# Basic parameters
batch_size: 4          # Adjust according to GPU memory
epochs: 200
lr: 1.0e-5
patch_size: 400

# Data structure type
task_structure: subfolder  # or 'filename'

# Loss weights (adjust according to your data)
boundary_weights:
  0: 2.0    # Cell
  1: 3.0    # Blood
  2: 5.0    # Other (increase to 8.0 or 10.0 if Other performance remains poor)

foreground_weights:
  0: 1.0    # Cell
  1: 1.5    # Blood
  2: 3.0    # Other (increase to 5.0 if foreground is too sparse)
```

### 3. Training

```bash
python train_multitask.py --config config_multitask.yaml
```

During training, the system will:
- Automatically utilise the GPU if available
- Save validation samples every 10 epochs
- Save checkpoints every 20 epochs
- Automatically save the best model
- Plot training curves

### 4. Monitoring Training

Training output includes:

```
Epoch 50/200: 100%|████████| 150/150 [02:30<00:00]
  loss: 0.1234, lr: 1.0e-5

Validation Loss: 0.1123

Overall:
  IoU:       0.8234
  Dice:      0.9012
  Precision: 0.8956
  Recall:    0.8734

Cell:
  IoU:       0.8456
  Dice:      0.9134
  Precision: 0.9023
  Recall:    0.8892

Blood:
  IoU:       0.8234
  Dice:      0.9012
  Precision: 0.8956
  Recall:    0.8734

Other:
  IoU:       0.8012
  Dice:      0.8890
  Precision: 0.8889
  Recall:    0.8576

Best model saved (IoU: 0.8234)
```

### 5. Output Files

```
outputs/
├── models/
│   ├── best_model.pth              # Best model
│   ├── final_model.pth             # Final model
│   ├── checkpoint_epoch020.pth     # Checkpoints
│   └── checkpoint_epoch040.pth
├── predictions/
│   ├── val_epoch010.png            # Validation samples
│   ├── val_epoch020.png
│   └── ...
└── training_history.png            # Training curves
```

---

## Tuning Guide

### If a Particular Task Performs Poorly

#### Poor Other Performance (Most Common)

**Symptoms**: Others are barely detected, IoU < 0.5

**Solutions**:

1. **Increase loss weights**:
```yaml
boundary_weights:
  2: 8.0    # Increase from 5.0 to 8.0

foreground_weights:
  2: 5.0    # Increase from 3.0 to 5.0
```

2. **Increase patch_size** (Others are long linear structures):
```yaml
patch_size: 600  # Increase from 400 to 600
```

3. **Adjust data augmentation** (in dataset_multitask.py):
```python
# Others: reduce deformation, increase contrast adjustment
configs[2] = A.Compose([
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(
        brightness_limit=0.4,  # Increase
        contrast_limit=0.5,    # Increase
        p=0.6                  # Increase probability
    ),
])
```

#### Inaccurate Blood Cell Boundaries

**Symptoms**: Blood cells are detected but boundaries are blurred

**Solution**:

```yaml
boundary_weights:
  1: 4.0    # Increase from 3.0 to 4.0
```

#### Over-segmentation of Cells

**Symptoms**: Cells are split into multiple small fragments

**Solution**:

```yaml
boundary_weights:
  0: 1.5    # Reduce from 2.0 to 1.5
```

### Insufficient GPU Memory

**Symptoms**: `CUDA out of memory`

**Solutions**:

1. Reduce batch_size:
```yaml
batch_size: 2  # or 1
```

2. Reduce patch_size:
```yaml
patch_size: 320  # Reduce from 400 to 320
```

3. Reduce decoder layers:
```yaml
num_decoder_conv_layers: 60  # Reduce from 80 to 60
```

### Training Is Too Slow

**Solutions**:

1. Reduce patch overlap ratio (in dataset_multitask.py):
```python
stride = self.patch_size  # Change from patch_size//2 to patch_size (no overlap)
```

2. Reduce epochs:
```yaml
epochs: 150  # Reduce from 200 to 150
```

3. Use fewer decoder layers:
```yaml
num_decoder_conv_layers: 60
```

---

## Prediction and Evaluation

### Loading a Trained Model

```python
from model_multitask import MultiTaskTransUNet
import torch

# Load model
model = MultiTaskTransUNet(
    img_size=400,
    patch_size=16,
    num_decoder_layers=80,
    num_tasks=3
)

model.load_state_dict(torch.load('outputs/models/best_model.pth'))
model.eval()
model.to('cuda')

# Prediction
with torch.no_grad():
    # Plant cells
    output_cell = model(image, task_id=0)
    
    # Blood cells
    output_blood = model(image, task_id=1)
    
    # Others
    output_Other = model(image, task_id=2)
```

### Batch Prediction

```python
import cv2
import numpy as np
from pathlib import Path

def predict_image(model, image_path, task_id, patch_size=400):
    # Read image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    
    # Prediction result
    result = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)
    
    stride = patch_size // 2
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y:y+patch_size, x:x+patch_size]
            
            # Convert to tensor
            patch_tensor = torch.from_numpy(
                patch.astype(np.float32) / 255.0
            ).permute(2, 0, 1).unsqueeze(0).to('cuda')
            
            # Predict
            with torch.no_grad():
                pred = model(patch_tensor, task_id=task_id)
                pred = torch.sigmoid(pred)[0, 0].cpu().numpy()
            
            # Accumulate results
            result[y:y+patch_size, x:x+patch_size] += pred
            count[y:y+patch_size, x:x+patch_size] += 1
    
    # Average
    result = result / (count + 1e-7)
    
    return result

# Usage example
result = predict_image(model, 'test_image.jpg', task_id=2)  # Others
result_binary = (result > 0.5).astype(np.uint8) * 255
cv2.imwrite('prediction.png', result_binary)
```

---

## Frequently Asked Questions

### Q: Why use multi-task learning?

A: Because resources are limited. Training three separate models requires:
- Three times the training time
- Three times the storage space
- Three times the prediction time

Multi-task learning allows you to handle three tasks with **a single model**, with only a 10-15% increase in computational cost.

### Q: How does task conditioning work?

A: During training and prediction, we inform the model of the type of image currently being processed (0=cell, 1=blood, 2=Other). The model adjusts its internal feature representation according to the task.

### Q: If my three image types are more similar, can I simplify matters?

A: Yes. If the images are highly similar, you can:
1. Use identical loss weights
2. Use the same data augmentation strategy
3. Even forgo task conditioning altogether

### Q: Can it be extended to more tasks?

A: Yes. Simply:
1. Add the new task to `TASK_MAPPING`
2. Set the corresponding loss weights
3. Adjust the `num_tasks` parameter

### Q: How do I use pre-trained weights?

A: Place the pre-trained weights in:
```
data/pretrained model/pretrained_model.pth
```

The training script will load them automatically.

### Q: How do I know which task requires adjustment?

A: Check the task-specific IoU in the training log:
- IoU > 0.85: Excellent
- IoU 0.70-0.85: Acceptable
- IoU < 0.70: Adjustment required

---

## Technical Details

### Model Architecture

```
Input Image (400x400x3)
    |
    v
[Patch Embedding] -> (25x25x768)
    |
    v
[ViT Encoder (12 layers)] -> (25x25x768)
    |
    v
[ASPP Multi-scale Feature] -> (25x25x256)
    |
    v
[Task-conditioned Decoder] -> (400x400x256)
  |-- Task Embedding
  |-- CBAM Attention
  +-- Residual Connections
    |
    v
[Output] -> (400x400x1)
```

### Loss Function Composition

```python
Total Loss = base_weight * (Dice + BCE)
           + boundary_weight * Boundary Loss
           + foreground_weight * Foreground Loss
```

Each task uses a different combination of weights.

---

## Performance Benchmarks

Training times on RTX 5080:

| Configuration | Time per Epoch | Total Training Time (200 epochs) |
|---------------|----------------|----------------------------------|
| Batch Size 4, Patch 400 | Approx. 3 minutes | Approx. 10 hours |
| Batch Size 2, Patch 400 | Approx. 4 minutes | Approx. 13 hours |
| Batch Size 4, Patch 320 | Approx. 2 minutes | Approx. 7 hours |

---

## Licence and Citation

If you use this code, please cite:

```
multiTUNAparty-model: Multi-Task TransUNet for Heterogeneous Biological Image Segmentation
A task-conditioned multi-task TransUNet for biological image segmentation
```

---

## Contact and Support

Should you have any questions or suggestions, feedback is most welcome.

Best of luck with your training.
