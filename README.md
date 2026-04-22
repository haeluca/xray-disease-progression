# CMC I Osteoarthritis X-Ray Synthesis

Modular architecture for two related generative modeling projects on thumb CMC I osteoarthritis radiographs:

- **Project A**: Feature-conditioned synthesis across disease spectrum
- **Project B**: Contralateral pseudo-longitudinal modeling using left-right asymmetry

## Architecture Overview

```
Raw images + labels
    ↓
Data Preparation (metadata, ROI extraction, patient-level splits)
    ↓
Shared Infrastructure
├─ Classifier (baseline feature-level classification)
├─ Datasets (shared base + branch-specific)
├─ Models (reusable components)
├─ Engine (config-driven training)
└─ Metrics (evaluation framework)
    ↓
Project A: Feature-Conditioned
├─ Baseline: Conditional VAE
└─ Main: Compact conditional diffusion
    ↓
Project B: Contralateral Pseudo-Longitudinal
├─ Baseline: Pix2Pix-style paired translation
└─ Main: Image-to-image conditional diffusion
    ↓
Shared Evaluation & Visualization
```

## Quick Start

### 1. Data Preparation

```bash
# Build metadata CSV from raw images
python data/prepare_metadata.py --raw_dir data/raw --output data/metadata.csv

# Extract and crop thumb CMC ROI
python data/extract_roi.py --metadata data/metadata.csv --output data/processed/roi

# Normalize left/right hand orientations
python data/normalize_laterality.py --roi_dir data/processed/roi --output data/processed/normalized

# Create patient-level splits
python data/build_splits.py --metadata data/metadata.csv --output data/splits

# Build contralateral pairs (Project B only)
python data/pair_contralateral.py --metadata data/metadata.csv --output data/contralateral_pairs.csv
```

### 2. Train Baseline Classifier

```bash
python scripts/run_feature_conditioned.py --config configs/classifier.yaml --stage classifier
```

### 3. Train Project A (Feature-Conditioned)

```bash
# Stage 1: Conditional VAE baseline
python scripts/run_feature_conditioned.py --config configs/project_a.yaml --stage baseline

# Stage 2: Main conditional diffusion model
python scripts/run_feature_conditioned.py --config configs/project_a.yaml --stage main

# Stage 3: Held-out evaluation
python scripts/run_feature_conditioned.py --config configs/project_a.yaml --stage test
```

### 4. Train Project B (Contralateral)

```bash
# Stage 1: Pix2Pix baseline
python scripts/run_contralateral.py --config configs/project_b.yaml --stage baseline

# Stage 2: Main image-conditioned diffusion
python scripts/run_contralateral.py --config configs/project_b.yaml --stage main

# Stage 3: Held-out evaluation
python scripts/run_contralateral.py --config configs/project_b.yaml --stage test
```

## Directory Structure

```
data/
├── prepare_metadata.py        # build master metadata CSV
├── extract_roi.py             # crop thumb CMC region
├── normalize_laterality.py    # mirror hands to common orientation
├── build_splits.py            # patient-level train/val/test
└── pair_contralateral.py      # bilateral pairing + feature deltas

datasets/
├── base_dataset.py            # shared dataset base class
├── transforms.py              # transforms with paired-flip fix
├── feature_conditioned_dataset.py  # Project A: single image + features
└── contralateral_dataset.py        # Project B: bilateral pairs

models/
├── classifier_backbone.py     # feature classifier (ResNet-18)
├── condition_encoder.py       # feature → conditioning embedding
├── diffusion_unet.py          # conditional diffusion UNet + DDPM
├── vae_baseline.py            # conditional VAE (Project A baseline)
├── pix2pix_baseline.py        # Pix2Pix generator + discriminator (Project B)
└── losses.py                  # centralized loss functions

engine/
├── train_classifier.py        # classifier training loop
├── train_generator.py         # generic config-driven generator training
├── validate_generator.py      # validation metrics + grid
├── test_generator.py          # final held-out evaluation
└── infer.py                   # batch inference / sampling

metrics/
├── feature_metrics.py         # label fidelity, per-feature agreement
├── image_metrics.py           # SSIM, PSNR, L1 distance
└── augmentation_study.py      # downstream utility evaluation

viz/
├── make_validation_grid.py    # training monitoring grid
├── make_feature_traversals.py # single-feature edit panels
├── make_final_report.py       # final PDF/image report
└── embedding_plots.py         # real-vs-synthetic embeddings

utils/
├── reproducibility.py         # seed setting, env capture
├── checkpoint.py              # save/load/best-model logic
└── logger.py                  # TensorBoard + scalar logging

scripts/
├── run_feature_conditioned.py # Project A entry point (--stage classifier|baseline|main|test)
└── run_contralateral.py       # Project B entry point

configs/
├── classifier.yaml            # shared baseline classifier config
├── project_a.yaml             # Project A training config
└── project_b.yaml             # Project B training config
```

## Key Design Principles

1. **Patient-level separation first** — All splits defined before modeling; no patient leakage across train/val/test
2. **ROI-focused** — Models operate on 256×256 thumb CMC crops, not full-hand radiographs
3. **Config-driven** — Reusable training engine; branches differ only in config
4. **Baseline before complexity** — Discriminative baseline, lightweight generative baseline, then main diffusion
5. **Evaluation beyond realism** — Success measured on feature fidelity, anatomy preservation, downstream utility

## Model Recommendations

**Project A (Feature-Conditioned Synthesis)**
- Baseline: Conditional VAE
- Main: Compact conditional diffusion UNet

**Project B (Contralateral Pseudo-Longitudinal)**
- Baseline: Pix2Pix with U-Net generator + PatchGAN discriminator
- Main: Image-to-image conditional diffusion

## Config Structure

All training configs (YAML) specify:
- Model architecture and hyperparameters
- Optimizer and learning rate schedule
- Batch size, epochs, checkpoint interval
- Data paths and feature dimensionality
- Loss weights and training objectives

See `configs/*.yaml` for examples.

## Testing Strategy

**Primary Outcomes**
- Feature fidelity: Does generated image express requested feature state?
- Anatomy preservation: Does image preserve patient-specific anatomy?
- Downstream utility: Does synthetic augmentation improve real-image classifier?

**Secondary Outcomes**
- Image similarity metrics (SSIM, PSNR, LPIPS)
- Sample quality summaries
- Blinded expert ranking of realism and appropriateness

## Evaluation Metrics

- `metrics/feature_metrics.py` — Label fidelity, per-feature agreement, condition matching
- `metrics/image_metrics.py` — SSIM, PSNR, L1 distance
- `metrics/augmentation_study.py` — Retrain classifier with/without synthetic data

## References

- Denoising Diffusion Probabilistic Models (DDPM): [arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)
- Pix2Pix: Image-to-Image Translation with Conditional GANs: [arxiv.org/abs/1611.05957](https://arxiv.org/abs/1611.05957)
- U-Net: Convolutional Networks for Biomedical Image Segmentation: [arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597)
