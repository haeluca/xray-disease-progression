# DDPM X-Ray Disease Progression Model

**Repository:** [haeluca/xray-disease-progression](https://github.com/haeluca/xray-disease-progression)

## Project Overview

This project implements a **Conditional Denoising Diffusion Probabilistic Model (DDPM)** for simulating disease progression in hand X-rays, specifically for **Carpometacarpal (CMC) I Osteoarthritis (OA)** progression.

### Goal
Generate synthetic future X-ray images showing disease progression by conditioning on baseline X-rays, enabling medical research and patient outcome simulation.

## Architecture

### Core Components

**Models** (`/models/`)
- `UNet` (`unet.py`): Conditional U-Net backbone for denoising
  - Input: 2 channels (noisy X-ray + baseline condition)
  - Output: 1 channel (predicted noise)
  - Time embedding integration for diffusion timesteps
  
- `DDPM` (`diffusion.py`): Denoising Diffusion Probabilistic Model
  - **Training**: Predicts noise added at timestep `t` given x_t1 and conditioning x_t0
  - **Sampling**: Iteratively denoises from random noise using conditioning
  - 1000 diffusion timesteps (T=1000), beta schedule from 1e-4 to 0.02

**Data** (`/data/`)
- `PairedXrayDataset`: Loads paired X-rays (baseline/timepoint)
  - Expected structure: `{TRAIN_DATA_PATH}/{patient_id}/t0.png` (baseline) and `t1.png` (future)
- `transforms.py`: Image preprocessing pipeline
- Grayscale images (1 channel), 256×256 resolution

**Training** (`train.py`)
- AdamW optimizer, MSE loss on predicted noise
- Batch size: 8, Learning rate: 1e-4, 200 epochs
- Checkpoints saved every 10 epochs to `./checkpoints/`

**Config** (`config.py`)
- Centralized hyperparameter management
- Device auto-detection (CUDA/CPU)

## Getting Started

### Training
```bash
python train.py
```
Expects training data in `./data/train/` following the paired X-ray structure.

### Sampling
```bash
python sample.py
```
Generates future progression images conditioned on baseline X-rays.

## Key Design Decisions

1. **Conditional Architecture**: Concatenating baseline X-ray with noisy progression X-ray allows the model to learn disease-specific deformation patterns
2. **Grayscale Input**: X-rays are naturally grayscale; color information unnecessary
3. **Fixed Image Size (256×256)**: Balances detail preservation with computational efficiency
4. **1000 Timesteps**: Provides fine-grained diffusion schedule for stable training

## Common Tasks

| Task | Command/Location |
|------|------------------|
| Adjust hyperparameters | Edit `config.py` |
| Change model architecture | Modify `models/unet.py` or `models/diffusion.py` |
| Prepare new dataset | Update `data/dataset.py` and point `TRAIN_DATA_PATH` in config |
| Sample from trained model | Run `sample.py` with checkpoint path |
| Monitor training | Checkpoints logged to `./checkpoints/` |

## Development Workflow

1. **Create a feature branch** for changes
2. **Test locally** before pushing
3. **Commit with clear messages** (include dataset/config changes in description)
4. **Push to GitHub** when ready for review

## Dependencies

- PyTorch (GPU recommended)
- torchvision
- tqdm
- numpy
- PIL

See `requirements.txt` for pinned versions.

## Notes

- Ensure paired X-ray data (baseline + future) exists before training
- GPU training strongly recommended (1000+ timesteps per epoch)
- Monitor loss during training; may require learning rate adjustment for convergence
