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

```python
from models.diffusion_unet import DDPM, DiffusionUNet
from datasets.transforms import get_val_transforms
from PIL import Image
import torch

# Load model
unet = DiffusionUNet(in_channels=1, out_channels=1, condition_dim=5)
model = DDPM(unet, device="cuda")
model.load_state_dict(torch.load("checkpoints/project_a/best.pt")["model_state_dict"])
model.eval()

# Load condition image and feature vector
condition_img = Image.open("example.png").convert("L")
transform = get_val_transforms()
condition_tensor = transform(condition_img).unsqueeze(0).to("cuda")

# Feature condition: e.g., [OA_grade, JSW, osteophyte, ...]
feature_vector = torch.tensor([[2, 1.5, 0.8, 0.3, 0.2]], dtype=torch.float32).to("cuda")

# Generate
with torch.no_grad():
    generated = model.sample(condition_tensor, shape=condition_tensor.shape, condition_vector=feature_vector)

# Save
from torchvision.transforms import ToPILImage
output_img = ToPILImage()((generated[0] * 0.5 + 0.5).clamp(0, 1).cpu())
output_img.save("generated.png")
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
