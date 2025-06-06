# CIFAR-10/100 Classifier

## Objectives
* Gain practical experience with deep learning by developing an image classification pipeline.
* Establish performance baselines using reference models (ResNet, MobileNet_v2, EfficientNet), w.r.t. hardware constraints, training effort, and dataset requirements.
* Practice documentation—it's been over a decade since I last updated paradigmjhc.com, and writing skills atrophy.

## Setup

### Dependencies
```
pip install torch torchinfo torchvision
```
# Files
`naivenet_128x128_cuda.py` is the latest iteration in this project. Other than some cosmetic improvements the project was cobbled together over a few days.
## naivenet_128x128_cuda.py
This is the most recent model (~320k parameters).
* ~70% accuracy on CIFAR-100  
* \>90% accuracy on CIFAR-10

### New Features Compared to 64x64 Version
The most impactful improvement came from adding Squeeze-and-Excitation (SE) blocks after hitting an accuracy ceiling. I'd still prefer to bring this back to a pure convolutional network by tuning hyperparameters more effectively. I don’t yet have intuition for using fully connected layers in this way.

Notable updates:
* Upscale 32×32 CIFAR images to 128×128 before applying affine and rotation-based augmentation. This led to significant improvements in performance.
* Discovered that sweeping across multiple dimensions in the hyperparameter space is standard practice—and now I understand why. Learning rate instability and early convergence were both markedly improved.
* Incorporated SE blocks after reviewing MobileNet_v2 and EfficientNet designs.
* Replaced 3×3 standard conv layers with depthwise separable convolutions followed by pointwise convolutions to reduce parameter count and training time.

## early_attempts/naivenet_64x64_cuda.py
This version served as a stepping stone toward the 128×128 implementation. Scaling up proved out the training benefits. Still very fast to train.

### New Features Since versioned_example.py
Mainly focused on image upscaling and more aggressive augmentation.
* IIRC, this was when the frontend bottleneck was added.

## early_attempts/versioned_example.py
My initial attempt at constructing a CNN architecture.

Progression:
* Started with two convolutional layers and a single fully connected layer.
* Added third and fourth conv layers—results improved, but returns diminished.
* Introduced skip connections and eventually implemented residual blocks.

### Remarks
* Two convolutional layers don’t perform well enough to be useful, but still exceeded my expectations.
* Some basic augmentations work surprisingly well when tuned to the data domain. For instance, horizontal flipping is useful since many object classes have symmetry around the Z-axis. Vertical flipping, however, generally reduces performance—these image classes aren’t gravity-invariant. Full rotational invariance isn’t a good match for CNNs anyway.