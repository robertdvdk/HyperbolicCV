# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the official PyTorch implementation of "Fully Hyperbolic Convolutional Neural Networks for Computer Vision" (ICLR 2024). The repository implements HCNN, a generalization of CNNs that learns feature representations in hyperbolic spaces using the Lorentz model.

## Key Architecture Concepts

### Manifold Types

The codebase supports multiple geometric spaces for neural network operations:

- **Euclidean**: Standard CNNs
- **Lorentz**: Hyperbolic geometry in the Lorentz model (primary contribution)
- **Poincaré**: Hyperbolic geometry in the Poincaré ball model
- **Hybrid**: Euclidean backbone with hyperbolic decoder (e.g., EL = Euclidean encoder + Lorentz decoder)

### Curvature Parameter

**IMPORTANT**: The curvature K is defined differently in the paper vs. Geoopt:
```
geoopt.K = -1/K
```

Where K is the curvature parameter used in config files (`encoder_k`, `decoder_k`).

### Core Components

The hyperbolic neural network library is in `code/lib/`:

**Lorentz Model** (`code/lib/lorentz/`):
- `manifold.py`: Extended Lorentz manifold with operations like flattening/reshaping, centroid, parallel transport addition
- `layers/LConv.py`: 1D and 2D (transposed) convolutional layers using Lorentz Direct Concatenation
- `layers/LBnorm.py`: Batch normalization on the Lorentz manifold
- `layers/LMLR.py`: Multinomial logistic regression classifier for hyperbolic spaces
- `layers/LFC.py`: **Novel custom fully-connected layer** (replaces the Chen et al. 2022 version, which is commented out)
- `layers/LModules.py`: Non-linear activations and global average pooling
- `blocks/resnet_blocks.py`: ResNet building blocks (BasicBlock, Bottleneck) adapted for Lorentz manifold
- `distributions/wrapped_normal.py`: Wrapped normal distribution (modified from Nagano et al. 2019)

**Poincaré Ball** (`code/lib/poincare/`):
- `layers/PMLR.py`: MLR from Shimizu et al. (2019)
- `distributions/wrapped_normal.py`: Wrapped normal from Mathieu et al. (2019)

**Models** (`code/lib/models/`):
- `resnet.py`: Unified ResNet implementation supporting Euclidean, Lorentz, and hybrid architectures

### Hyperbolic Operations

Key operations in the Lorentz model (`code/lib/lorentz/manifold.py`):
- **Lorentz Direct Concatenation**: Used in convolutional layers to combine features within patches while respecting hyperbolic geometry
- **Projection**: `add_time()` and `calc_time()` methods add/calculate time component from space component
- **Centroid**: Compute weighted mean on the manifold
- **Parallel Transport Addition**: `pt_addition()` implements Chami et al. (2019) method
- **Manifold Switching**: `switch_man()` projects between Lorentz manifolds with different curvatures
- **Flattening/Reshaping**: `lorentz_flatten()` and `lorentz_reshape_img()` based on Qu et al. (2022)

### Novel Fully Connected Layer

**IMPORTANT**: The `LorentzFullyConnected` layer in `code/lib/lorentz/layers/LFC.py` has been replaced with a novel custom implementation. The original Chen et al. (2022) version is commented out in the same file (lines 92-166).

The new implementation:
- Uses learnable parameters U, a, and V_auxiliary for spacelike vector construction
- Supports creating spacelike vectors via `create_spacelike_vector()`
- Can compute signed distances to hyperplanes in two ways:
  - `signed_dist2hyperplanes_scaled_angle()`: Scale by angle (implicitly)
  - `signed_dist2hyperplanes_scaled_dist()`: Scale by distance (explicitly)
- Includes an MLR mode (`do_mlr` flag) for classification
- Provides `forward_cache()` method using V_auxiliary for cached computations
- Projects output space orthogonally onto the manifold via `projection_space_orthogonal()`

This custom layer is used as the building block in `LorentzConv1d` and `LorentzConv2d`.

## Training Commands

### Classification

Train models using config files. Commands should be run from the repository root:

```bash
# Fully Lorentz ResNet-18
python code/classification/train.py -c classification/config/L-ResNet18.txt

# Hybrid Euclidean-Lorentz ResNet-18
python code/classification/train.py -c classification/config/EL-ResNet18.txt

# Override config parameters
python code/classification/train.py -c classification/config/L-ResNet18.txt \
  --output_dir classification/output --device cuda:1 --dataset CIFAR-10
```

Supported datasets: CIFAR-10, CIFAR-100, Tiny-ImageNet

### Image Generation (VAE)

```bash
# Fully Lorentz VAE
python code/generation/train.py -c generation/config/L-VAE/L-VAE-CIFAR.txt

# Hybrid VAE
python code/generation/train.py -c generation/config/EL-VAE/EL-VAE-CIFAR.txt

# Override config parameters
python code/generation/train.py -c generation/config/L-VAE/L-VAE-CIFAR.txt \
  --output_dir generation/output --device cuda:1 --dataset CIFAR-100
```

Supported datasets: CIFAR-10, CIFAR-100, CelebA

## Testing and Evaluation

### Classification

```bash
# Test accuracy
python code/classification/test.py -c classification/config/L-ResNet18.txt \
  --mode test_accuracy --load_checkpoint PATH/TO/WEIGHTS.pth

# Visualize embeddings
python code/classification/test.py -c classification/config/L-ResNet18.txt \
  --mode visualize_embeddings --load_checkpoint PATH/TO/WEIGHTS.pth \
  --output_dir classification/output

# Adversarial robustness (FGSM)
python code/classification/test.py -c classification/config/L-ResNet18.txt \
  --mode fgsm --load_checkpoint PATH/TO/WEIGHTS.pth

# Adversarial robustness (PGD)
python code/classification/test.py -c classification/config/L-ResNet18.txt \
  --mode pgd --load_checkpoint PATH/TO/WEIGHTS.pth
```

### Generation

```bash
# Test FID score
python code/generation/test.py -c generation/config/L-VAE/L-VAE-CIFAR.txt \
  --mode test_FID --load_checkpoint PATH/TO/WEIGHTS.pth

# Visualize latent embeddings
python code/generation/test.py -c generation/config/L-VAE/L-VAE-CIFAR.txt \
  --mode visualize_embeddings --load_checkpoint PATH/TO/WEIGHTS.pth \
  --output_dir generation/output

# Generate samples
python code/generation/test.py -c generation/config/L-VAE/L-VAE-CIFAR.txt \
  --mode generate --load_checkpoint PATH/TO/WEIGHTS.pth \
  --output_dir generation/output

# Reconstruct images
python code/generation/test.py -c generation/config/L-VAE/L-VAE-CIFAR.txt \
  --mode reconstruct --load_checkpoint PATH/TO/WEIGHTS.pth \
  --output_dir generation/output
```

## Dataset Setup

CIFAR-10/100 and CelebA are automatically downloaded via torchvision.

For Tiny-ImageNet:
```bash
cd code/classification
bash get_tinyimagenet.sh    # Download
python org_tinyimagenet.py   # Organize into proper structure
```

## Environment Setup

```bash
# Create environment
conda create -n HCNN python=3.8 pip
conda activate HCNN

# Install PyTorch (adjust for your CUDA version)
conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia

# Install dependencies
pip install -r requirements.txt
```

Required: Python>=3.8, PyTorch, torchvision

## Config File Structure

Config files use ConfigArgParse format (`.txt` files with key-value pairs). Key parameters:

**Model Architecture**:
- `encoder_manifold`: euclidean or lorentz
- `decoder_manifold`: euclidean, lorentz, or poincare
- `num_layers`: ResNet depth (18 or 50)
- `embedding_dim`: Dimensionality of feature space

**Hyperbolic Settings**:
- `encoder_k` / `decoder_k`: Curvature parameter (remember: geoopt.K = -1/K)
- `learn_k`: Make curvature learnable
- `clip_features`: Clipping for hybrid models (Guo et al. 2022)

**Optimization**:
- `optimizer`: RiemannianSGD, RiemannianAdam, SGD, or Adam
- `lr`: Learning rate
- `use_lr_scheduler`: Enable learning rate scheduling
- `lr_scheduler_milestones`: Epochs to reduce LR (classification)
- `lr_scheduler_step`: Step size for LR reduction (generation)

## Project Structure

```
code/
├── classification/          # Image classification experiments
│   ├── train.py            # Main training script
│   ├── test.py             # Evaluation and testing
│   ├── config/             # Config files for different models
│   └── utils/              # Dataset loaders, initialization
├── generation/             # Image generation (VAE) experiments
│   ├── train.py
│   ├── test.py
│   ├── config/             # Organized by model type (E-VAE, L-VAE, etc.)
│   ├── models/             # VAE architectures
│   └── utils/              # FID calculation, utilities
└── lib/                    # Reusable library components
    ├── lorentz/            # Lorentz model operations
    ├── poincare/           # Poincaré ball operations
    ├── Euclidean/          # Standard Euclidean blocks
    ├── models/             # General model architectures (ResNet)
    ├── geoopt/             # Extended Geoopt library
    └── utils/              # Visualization, metrics
```

## Working Directory Convention

Both `train.py` scripts change working directory to `HyperbolicCV/code/` at startup. All relative paths in commands and config files are relative to this directory.

## Optimizer Notes

- Use `RiemannianSGD` or `RiemannianAdam` for hyperbolic models
- Standard `SGD` or `Adam` can be used for Euclidean models
- The library extends Geoopt's optimizers for Riemannian manifolds
