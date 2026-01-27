# multiparty-model
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10.0.dev20251122+cu128-red)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-green)](https://developer.nvidia.com/cuda-toolkit)

A hybrid Multi-Task TransUNet architecture for a precise area recognition tool ya (multiparty).

## Table of Contents
- [Features](#-features)
- [Installation](#-installation)
- [Data Preparation](#-data-preparation)
- [Training](#-training)  
- [Inference](#-inference)
- [Configuration](#⚙-configuration)
- [Performance](#-performance)
- [Project Structure](#-project-structure)
- [License](#-license)

## Features
- **Dual-mode Architecture**: Combines Vision Transformer encoder with CNN decoder
- **Industrial-grade**: Supports any size of high-resolution input images
- **Real-time Processing**: 45ms inference time on NVIDIA T4 GPU
- **Adaptive Learning**: Automatic mixed precision training support

## Installation
### Prerequisites
- NVIDIA GPU with CUDA 12.1+ support
- Python 3.9+
- NVIDIA GeForce RTX5080 16G


### Setup Environment
```bash
conda create -n partymodel python=3.9
conda activate partymodel
pip install -r requirementsRTX5060.txt
cd party-model

```
#Note: If your GPU is RTX5080 with Blackwell, it is better to install PyTorch CUDA 12.8

```bash
conda create -n partymodel_cuda128 python=3.11
conda activate partymodel_cuda128
pip install -r requirementsRTX5080.txt
cd party-model
```
Then remove ", verbose=True" in L85 in the file "src/train.py" and add cuda to scaler = GradScaler() in L91, then it will like scaler = GradScaler('cuda')  


## Data Preparation
### Directory Structure
```
├── data/                
│   ├── pretrained model/
│        ├── pretrained_model.pth
│   ├── train/
│        ├── images/
│        ├── masks/
│   ├── test/
│        ├── images/
│        ├── masks/
│   ├── val/
│        ├── images/
│        ├── masks/
```

### Naming Convention
- Image/Mask pairs must have matching filenames:
  - `sieve_123.jpg` ↔ `sieve_123.png`
- Supported resolutions: 400x400 to 2000x2000

## Training
### Start Training by UI
```bash
streamlit run partymodel_ui.py
```
### Start Training
```bash
python src/train.py --config configs/default.yaml
```

### Training Options
| Parameter          | Default | Description                     |
|--------------------|---------|---------------------------------|
| `--batch_size`     | 8       | GPU memory dependent           |
| `--epochs`         | 100     | Typical range: 100-300          |
| `--lr`             | 1e-4    | AdamW initial learning rate    |
| `--image_size`     | 224     | ViT input dimension            |

### Training Monitoring
TensorBoard integration:
```bash
tensorboard --logdir outputs/tensorboard
```

## Inference
### Single Image Prediction
```bash
python src/predict.py \
    --model_path outputs/models/sievenet_epoch100.pth \
    --input data/test/images/demo.jpg \
    --output outputs/predictions/result.png
```

### Batch Prediction
```bash
python src/predict_batch.py \
    --model_path outputs/models/sievenet_epoch100.pth \
    --input_dir data/test/images/ \
    --output_dir outputs/predictions/
```

## Configuration
Edit `configs/default.yaml`:
```yaml
data_path: "data/"
batch_size: 4
epochs: 250
lr: 1.0e-5         # 使用科學計數法表示
patch_size: 400    # 可調整，值越小識別越精細
val_split: 0.2     # 小數形式

# 模型架構選項：
#  - "TransUNet": 使用原始 TransUNet 模型
#  - "TransUNetWithTimm": 使用 timm 載入的 ViT encoder，並保持原有 decoder 結構
model_type: "TransUNet"

num_decoder_conv_layers: 80   # 可隨意修改 decoder 中卷積層數量（例如預設 30 層），根據過往的研究，層數太高會丟失細節，太低會分辨不佳，大概20-30層之間
#25層好像也不太可以，目前測試80-120對篩管的辨識還不錯

```

## Performance
### Evaluation Metrics (Test Set)
| Metric        | Value  |
|---------------|--------|
| Dice Score    | 0.92   |
| IoU           | 0.87   |
| Precision     | 0.94   |
| Recall        | 0.91   |

### Hardware Benchmark
| GPU           | Inference Time | Memory Usage |
|---------------|----------------|--------------|
| NVIDIA T4     | 45ms           | 3.2GB        |
| RTX 3090      | 28ms           | 4.1GB        |
| A100 (40GB)   | 18ms           | 5.8GB        |

## Project Structure
```
party model/
├── configs/              # Configuration templates
│   ├── default.yaml
├── data/                # Dataset storage
│   ├── pretrained model/
│        ├── pretrained_model.pth
│   ├── train/
│        ├── images/
│        ├── masks/
│   ├── test/
│        ├── images/
│        ├── masks/
│   ├── val/
│        ├── images/
│        ├── masks/
├── outputs/             # Training artifacts
│   ├── models/          # Checkpoints
│   ├── predictions/     # Inference results
│   └── tensorboard/     # Training metrics
├── src/                 # Core implementation
│   ├── dataset.py        # 數據讀取與增強
│   ├── model.py          # 原始 TransUNet 模型
│   ├── model_with_timm.py# 基於 timm 的 ViT encoder 模型
│   ├── losses.py         # 損失函數實現
│   ├── train.py          # 模型訓練流程
│   ├── predict.py        # 預測與推理模塊
│   ├── utils.py          # 輔助工具（例如可視化）
│   └── postprocess.py    # 分割結果後處理模塊
└── requirements.txt     # Dependency list
└── README,md
```

## License
Apache 2.0 License. See [LICENSE](LICENSE) for details.
