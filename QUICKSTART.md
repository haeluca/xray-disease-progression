# Quick Start Guide

## Overview

This repository implements a **two-project modular architecture** for CMC I osteoarthritis X-ray synthesis:
- **Project A**: Feature-conditioned synthesis
- **Project B**: Contralateral pseudo-longitudinal modeling

Both projects share a common data pipeline, classifier, metrics, and training infrastructure.

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Prepare Data

```bash
# Build metadata CSV from raw images
python data/prepare_metadata.py --raw_dir data/raw --output data/metadata.csv

# Extract and crop thumb CMC ROI
python data/extract_roi.py --metadata data/metadata.csv --output data/processed/roi

# Normalize left/right orientation
python data/normalize_laterality.py --roi_dir data/processed/roi --output data/processed/normalized

# Create patient-level splits
python data/build_splits.py --metadata data/metadata.csv --output data/splits

# (Project B only) Build contralateral pairs
python data/pair_contralateral.py --metadata data/metadata.csv --output data/contralateral_pairs.csv
```

## 3. Train Baseline Classifier

```bash
python scripts/run_feature_conditioned.py --config configs/classifier.yaml --stage classifier
```

Saves best checkpoint to `checkpoints/classifier/best.pt`

## 4. Train Project A (Feature-Conditioned)

```bash
# Stage 1: Conditional VAE baseline (debugging)
python scripts/run_feature_conditioned.py --config configs/project_a.yaml --stage baseline

# Stage 2: Main conditional diffusion model
python scripts/run_feature_conditioned.py --config configs/project_a.yaml --stage main

# Stage 3: Held-out evaluation
python scripts/run_feature_conditioned.py --config configs/project_a.yaml --stage test
```

Checkpoints saved to `checkpoints/project_a/`

## 5. Train Project B (Contralateral)

```bash
# Stage 1: Pix2Pix baseline (debugging)
python scripts/run_contralateral.py --config configs/project_b.yaml --stage baseline

# Stage 2: Main image-conditioned diffusion
python scripts/run_contralateral.py --config configs/project_b.yaml --stage main

# Stage 3: Held-out evaluation
python scripts/run_contralateral.py --config configs/project_b.yaml --stage test
```

Checkpoints saved to `checkpoints/project_b/`

## 6. Generate Samples

**Project A** generates from pure noise conditioned on a feature vector (no source image needed):

```python
from models.diffusion_unet import DDPM, DiffusionUNet
from utils.feature_schema import DEFAULT_FEATURE_SCHEMA
import torch

# Load model (in_channels=1 — feature-only conditioning, no source image)
num_features = len(DEFAULT_FEATURE_SCHEMA)
unet = DiffusionUNet(in_channels=1, out_channels=1, condition_dim=num_features)
model = DDPM(unet, device="cuda")
model.load_state_dict(torch.load("checkpoints/project_a/best.pt")["model_state_dict"])
model.eval()

# Feature vector: one value per feature in DEFAULT_FEATURE_SCHEMA order
# e.g. JSN=2, osteophytes=1, cysts_relevant=0, cysts_irrelevant=0,
#      sclerosis=1, stt_involvement=0, subluxation_ratio=0.3
feature_vector = torch.tensor([[2, 1, 0, 0, 1, 0, 0.3]], dtype=torch.float32).to("cuda")

# Generate — shape is (batch, channels, height, width)
with torch.no_grad():
    generated = model.sample((1, 1, 64, 64), condition_vector=feature_vector)

# Save
from torchvision.transforms import ToPILImage
output_img = ToPILImage()((generated[0] * 0.5 + 0.5).clamp(0, 1).cpu())
output_img.save("generated.png")
```

**Project B** generates by translating a source image toward the more-affected state:

```python
from models.diffusion_unet import DDPM, DiffusionUNet
from datasets.transforms import get_val_transforms
from PIL import Image
import torch

num_features = 7
unet = DiffusionUNet(in_channels=2, out_channels=1, condition_dim=num_features)
model = DDPM(unet, device="cuda")
model.load_state_dict(torch.load("checkpoints/project_b/best.pt")["model_state_dict"])
model.eval()

# Load less-affected source image
tf = get_val_transforms(64)
source = tf(Image.open("source.png").convert("L")).unsqueeze(0).to("cuda")

# Feature delta: how much more affected the target should be
delta = torch.tensor([[1, 1, 0, 0, 0, 0, 0.1]], dtype=torch.float32).to("cuda")

with torch.no_grad():
    generated = model.sample(source.shape, condition_vector=delta, x_condition=source)

from torchvision.transforms import ToPILImage
ToPILImage()((generated[0] * 0.5 + 0.5).clamp(0, 1).cpu()).save("generated.png")
```

Or use the CLI:

```bash
# Project A
python engine/infer.py --config configs/project_a.yaml \
    --checkpoint checkpoints/project_a/best.pt \
    --project a --objective ddpm --output outputs/samples_a/

# Project B
python engine/infer.py --config configs/project_b.yaml \
    --checkpoint checkpoints/project_b/best.pt \
    --project b --objective ddpm --output outputs/samples_b/
```

## Key Directories

| Directory | Purpose |
|-----------|---------|
| `data/` | Data preparation scripts |
| `datasets/` | PyTorch dataset classes |
| `models/` | Neural network modules |
| `engine/` | Training & inference loops |
| `metrics/` | Evaluation metrics |
| `viz/` | Visualization utilities |
| `utils/` | Logging, checkpoints, reproducibility |
| `configs/` | YAML configuration files |
| `scripts/` | Entry points (`run_feature_conditioned.py`, `run_contralateral.py`) |
| `checkpoints/` | Saved model weights |
| `logs/` | TensorBoard logs |

## Config Files

- `configs/classifier.yaml` — Shared baseline classifier config
- `configs/project_a.yaml` — Project A (feature-conditioned) config
- `configs/project_b.yaml` — Project B (contralateral) config

Edit these YAML files to change:
- Model architecture
- Learning rate & optimizer
- Batch size & epochs
- Loss weights
- Data paths

## Monitoring Training

```bash
# View TensorBoard logs
tensorboard --logdir logs/
```

Opens at `http://localhost:6006`

## Next Steps

1. **Read CLAUDE.md** for detailed architecture overview
2. **Read README.md** for component descriptions
3. **Read architecture_plan_thumb_cmc_oa.pdf** for design rationale

## Common Issues

**No data found error:**
- Ensure raw images are in `data/raw/` with correct directory structure (one subdir per patient)

**CUDA out of memory:**
- Reduce `batch_size` in config YAML or use smaller `image_size`

**Model not converging:**
- Check learning rate in config
- Verify data pipeline by inspecting a batch manually
- Try baseline model first to isolate generator issues

## Questions?

See CLAUDE.md for architecture overview or README.md for detailed component explanations.
