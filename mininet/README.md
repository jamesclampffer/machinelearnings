
# CNN Training Toolkit

Minimalist framework for training CNNs on image classification tasks. Originally built for self-teaching experiments that needed fast iteration on public datasets like CIFAR and Food101.

Designed for:
- Quick end-to-end experimentation with reasonable defaults
- Hackable by design — an example architecture is prewired; others can be dropped in with minimal changes.

---

## Highlights

- **GPU-ready**: mixed precision (AMP), OneCycleLR, gradient clipping
- **Intended to fully utilize H100-class hardware** during training 
- **Dataset abstraction**: single loader interface hides dataset quirks  
- **Modular CLI training**: batching, learning rate, dataset, and activation type are all configurable  
- **ResNet-inspired "MiniNet" included**: Six residual blocks - depthwise-separable convs. SE blocks between residual blocks.
- **Portable**: no pip install or packaging overhead—just copy and run

---

## File Overview

| File | Description |
|------|-------------|
| `main.py` | Entry point. Parses CLI args and wires together model, dataset, trainer |
| `trainer.py` | Handles training loop, validation, and checkpointing |
| `dataloaders.py` | Abstracts over torchvision datasets (CIFAR-10/100, Food101) |
| `mininet.py` | Lightweight ResNet-style model using SE blocks and depthwise convs |

---

## Example Usage

```bash
python main.py \
  --dataset cifar100 \
  --epochs 30 \
  --batch_size 256 \
  --activation gelu \
  --initial_lr 0.01
```

---

## Supported Datasets

| Dataset   | Native Size | # Classes |
|-----------|-------------|-----------|
| CIFAR-10  | 32×32×3     | 10        |
| CIFAR-100 | 32×32×3     | 100       |
| Food-101  | 224×224×3   | 101       |

Planned:  
- HuggingFace and Kaggle-hosted datasets with download logic

---

## MiniNet Summary

ResNet-inspired shallow architecture.
- 5x5 conv -> 3x3 conv using ReLU for initial feature extraction
  - Hard coded ReLU in stem to introduce discontinuity. In deeper layers something that handles backprop better like mish is a better choice
- 6 residual blocks (each with 2× depthwise-separable convs)
- Skip connections w/ optional spatial downsampling
- SE modules between blocks (squeeze ratio configurable)
- AdaptiveAvgPool → Linear classification head
- Configurable activation: ReLU, SiLU, GELU

---
## Some Results

**75% accuracy on food101 @ 677k params** ain't half bad. Not going to declare victory until I sort out FLOPS/inference and get a better hyperparameter sweep. Check out "model-epoch299.pth" to verify, if you'd like. Please notify me if you spot an error in test methodology.

  | Dataset  | Epochs | Initial LR | Activation | Batch | Train time   | Accuracy | Params  |
  |----------|--------|------------|------------|-------|--------------|----------|---------|
  | CIFAR10  |        |            |            |       |              |          |         |
  | CIFAR100 |        |            |            |       |              |          |         |
  | FOOD101  |  300   | 0.001      |  mish      |  512  | 10h on gh200 | ~75%     | 677,573 |

---

## Training Details

The code is intended to be easy to modify. Over-parameterizing adds unnecessary maintenance burden.

| Feature         | Value                   |
|----------------|--------------------------|
| Optimizer       | `SGD`                   |
| LR Schedule     | `OneCycleLR` (per batch)|
| Loss Function   | `CrossEntropy` w/ label smoothing = 0.1 |
| Mixed Precision | AMP enabled when using CUDA |
| Gradient Clipping | `max_norm=1.0`        |
| Normalization   | `mean=0.5`, `std=0.5` per channel |

---
## Limitations

- Checkpointing logic saves model only; other state is not restored.
- DDP path was copied from an older project without testing. More there as a motivator to address multi-gpu.
---

## Roadmap

- Support more datasets selectable by CLI flags
- Add hyperparameter sweep utility  
- Multi-GPU support (initial DDP sketch copied from an older project)
- Extend support to PINNs and other approximators (likely via a second training loop reusing existing CLI interface)
---
