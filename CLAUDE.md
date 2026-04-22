# CMC I Osteoarthritis X-Ray Synthesis — Modular Two-Project Architecture

**Repository:** [haeluca/xray-disease-progression](https://github.com/haeluca/xray-disease-progression)

## Overview

This project implements a **shared infrastructure architecture** supporting two related research projects on thumb CMC I osteoarthritis X-ray synthesis:

- **Project A**: Feature-conditioned synthesis — generate X-rays conditioned on a target disease feature vector
- **Project B**: Contralateral pseudo-longitudinal modeling — simulate progression by translating a less-affected hand toward the more-affected contralateral state

The architecture follows **one codebase, two branches** principle: shared data pipeline, classifier, models, metrics, and training infrastructure; project-specific differences confined to dataset construction, conditioning logic, and generator objectives.

---

## Design Principles (from `architecture_plan_thumb_cmc_oa.pdf`)

1. **Patient-level separation first** — All train/val/test splits defined before modeling; no patient leakage across sets
2. **ROI-focused modeling** — All models operate on 256×256 thumb CMC crops, not full-hand radiographs
3. **Config-driven experiments** — Reusable training engine; projects differ by swapping YAML config files
4. **Baseline before complexity** — Discriminative baseline classifier → lightweight generative baseline → main diffusion model
5. **Evaluation beyond realism** — Success measured on feature fidelity, anatomy preservation, downstream utility

---

## Staged Development Plan

| Stage | Task | Output |
|-------|------|--------|
| **0** | Data preparation & QC | `metadata.csv`, patient-level splits (CSV) |
| **1** | Baseline classifier | Real-image ResNet-18; frozen evaluation checkpoint |
| **2** | Lightweight generative baseline | Project A: Conditional VAE / Project B: Pix2Pix |
| **3** | Main generator | Project A/B: Conditional diffusion models |
| **4** | Held-out testing | Feature fidelity, anatomy preservation, downstream utility metrics |
| **5** | Clinical visualization | Final PDF report with galleries, traversals, failures |

---

## Repository Structure

### Data Pipeline

**Location:** `data/`

- `prepare_metadata.py` — Scan raw images, build `metadata.csv` (patient_id, side, features, QC flags)
- `extract_roi.py` — Crop thumb CMC region, save 256×256 PNGs to `data/processed/roi/`
- `normalize_laterality.py` — Mirror left-hand images to right orientation; save to `data/processed/normalized/`
- `build_splits.py` — Patient-level 70%/15%/15% train/val/test; frozen manifests (CSV)
- `pair_contralateral.py` — (Project B) Bilateral pairs + feature deltas → `data/contralateral_pairs.csv`

### Shared Models

**Location:** `models/`

- `classifier_backbone.py` — ResNet-18 adapted for 1-channel grayscale; used for feature evaluation
- `condition_encoder.py` — Encodes feature vector → conditioning embedding (shared API for both projects)
- `diffusion_unet.py` — Conditional U-Net + DDPM diffusion schedule; reused by Projects A & B
- `losses.py` — Reconstruction, KL, adversarial, condition-consistency, perceptual losses
- `vae_baseline.py` — Conditional VAE (Project A baseline)
- `pix2pix_baseline.py` — UNet generator + PatchGAN discriminator (Project B baseline)

### Datasets

**Location:** `datasets/`

- `base_dataset.py` — Shared metadata loading, image loading, path handling
- `transforms.py` — Normalization, resizing, **paired-flip fix** (consistent flip across image pairs)
- `feature_conditioned_dataset.py` — (Project A) Single image + source & target feature vectors
- `contralateral_dataset.py` — (Project B) Bilateral pairs (less/more affected) + side indicator + feature delta

### Training Engine

**Location:** `engine/`

- `train_classifier.py` — Trains classifier on real images; freezes best checkpoint for later use
- `train_generator.py` — Generic config-driven training loop (instantiates model/dataset by class name)
- `validate_generator.py` — Validation metrics + validation grid export
- `test_generator.py` — Final held-out evaluation → structured results
- `infer.py` — Batch inference / sampling from trained generator

### Metrics & Visualization

**Metrics** (`metrics/`):
- `feature_metrics.py` — Label fidelity, per-feature agreement, condition matching
- `image_metrics.py` — SSIM, PSNR, L1 distance
- `augmentation_study.py` — Retrain classifier with/without synthetic data to measure downstream utility

**Visualization** (`viz/`):
- `make_validation_grid.py` — Training monitoring: real vs generated images
- `make_feature_traversals.py` — Single-feature edit panels (vary one feature, hold others fixed)
- `make_final_report.py` — Final PDF: representative cases, failures, feature traversals
- `embedding_plots.py` — Real-vs-synthetic embedding plots (using frozen classifier)

### Utilities

**Location:** `utils/`

- `reproducibility.py` — Seed setting, deterministic flags, environment capture
- `checkpoint.py` — Save/load checkpoints, best-model selection, resume logic
- `logger.py` — TensorBoard scalar & image logging

### Configs

**Location:** `configs/`

- `classifier.yaml` — Shared classifier training config
- `project_a.yaml` — Project A (feature-conditioned) config
- `project_b.yaml` — Project B (contralateral) config

Each YAML specifies: model architecture, optimizer, scheduler, batch size, loss weights, data paths.

### Entry Points

**Location:** `scripts/`

- `run_feature_conditioned.py` — Project A training: `--stage [classifier|baseline|main|test]`
- `run_contralateral.py` — Project B training: `--stage [classifier|baseline|main|test]`

---

## Key Components Explained

### Diffusion UNet

**File:** `models/diffusion_unet.py`

- **DiffusionUNet**: Conditional U-Net denoising backbone
  - Input: 2-channel concatenation (noisy target + condition image) + timestep embedding
  - Output: Predicted noise
  - Sinusoidal positional embeddings for timestep
  - ResBlocks with time injection via MLP
  - 4-level encoder-decoder with skip connections
  - Optional condition embedding injection via `ConditionEncoder`

- **DDPM**: Diffusion model wrapper
  - Precomputes noise schedule (T=1000, linear beta schedule from 1e-4 to 0.02)
  - `forward(x_target, x_condition, t, condition_vector)` — adds noise, predicts, returns MSE loss
  - `sample(x_condition, condition_vector)` — reverse diffusion loop (1000 steps down to 0)

### Feature-Conditioned Synthesis (Project A)

**Baseline:** Conditional VAE (`models/vae_baseline.py`)
- Encoder: image → latent + condition → z_mean, z_logvar
- Decoder: z + condition → reconstructed image
- Loss: ELBO = reconstruction + KL

**Main:** Diffusion UNet conditioned on feature vector
- Input image: arbitrary or template
- Condition: feature vector (e.g., [OA grade, joint space width, osteophyte, …])
- Output: Generated X-ray with requested feature state

### Contralateral Pseudo-Longitudinal (Project B)

**Baseline:** Pix2Pix (`models/pix2pix_baseline.py`)
- Generator: UNet (less-affected image → more-affected image)
- Discriminator: 70×70 PatchGAN
- Losses: L1 reconstruction + adversarial

**Main:** Image-conditioned diffusion
- Input image: less-affected (contralateral) side
- Condition: feature delta (how much more-affected is the target)
- Output: Synthetic more-affected progression of input anatomy

---

## Data Flows

### Training (Project A: Feature-Conditioned)

```
Raw images
  ↓ [prepare_metadata.py]
metadata.csv (patient_id, side, features, qc_pass)
  ↓ [extract_roi.py]
data/processed/roi/ (256×256 crops)
  ↓ [normalize_laterality.py]
data/processed/normalized/ (left→right mirrored)
  ↓ [build_splits.py]
splits/{train,val,test}.csv (patient-level)
  ↓ [FeatureConditionedDataset]
Batches: {image, source_features, target_features}
  ↓ [DiffusionUNet + DDPM]
Generated images conditioned on target_features
```

### Training (Project B: Contralateral)

```
Raw images → metadata.csv → roi → normalized → splits
  ↓ [pair_contralateral.py]
contralateral_pairs.csv (less_affected, more_affected, deltas)
  ↓ [ContralateralDataset]
Batches: {source, target, side, feature_delta}
  ↓ [DiffusionUNet + DDPM]
Generated progression: source → more-affected state
```

---

## How to Use

### Step 1: Data Preparation

```bash
python data/prepare_metadata.py --raw_dir data/raw --output data/metadata.csv
python data/extract_roi.py --metadata data/metadata.csv --output data/processed/roi
python data/normalize_laterality.py --roi_dir data/processed/roi --output data/processed/normalized
python data/build_splits.py --metadata data/metadata.csv --output data/splits
python data/pair_contralateral.py --metadata data/metadata.csv  # Project B only
```

### Step 2: Train Classifier (Shared)

```bash
python scripts/run_feature_conditioned.py --config configs/classifier.yaml --stage classifier
```

### Step 3: Project A Training

```bash
python scripts/run_feature_conditioned.py --config configs/project_a.yaml --stage baseline
python scripts/run_feature_conditioned.py --config configs/project_a.yaml --stage main
python scripts/run_feature_conditioned.py --config configs/project_a.yaml --stage test
```

### Step 4: Project B Training

```bash
python scripts/run_contralateral.py --config configs/project_b.yaml --stage baseline
python scripts/run_contralateral.py --config configs/project_b.yaml --stage main
python scripts/run_contralateral.py --config configs/project_b.yaml --stage test
```

---

## File Mapping: Old → New

| Old Code | New Location | Purpose |
|----------|--------------|---------|
| `config.py` | `configs/*.yaml` | YAML-driven config per project |
| `train.py` | `engine/train_generator.py`, `scripts/run_*.py` | Generic training loop + entry points |
| `sample.py` | `engine/infer.py` | Inference / batch sampling |
| `models/unet.py` | `models/diffusion_unet.py` | UNet backbone (adapted, reused) |
| `models/diffusion.py` | `models/diffusion_unet.py` | DDPM wrapper (adapted, reused) |
| `data/dataset.py` | `datasets/base_dataset.py`, `datasets/feature_conditioned_dataset.py`, `datasets/contralateral_dataset.py` | Dataset classes (refactored, split by project) |
| `data/transforms.py` | `datasets/transforms.py` | Transforms with paired-flip bug fix |
| README/CLAUDE | This file | Architecture documentation |

---

## Architecture Benefits

✅ **Code Reuse** — Shared classifier, metrics, visualization, checkpoint logic
✅ **Maintainability** — One pipeline, two branches (not two independent codebases)
✅ **Flexibility** — Config-driven; easy to swap models, loss functions, optimizer settings
✅ **Reproducibility** — Frozen patient-level splits, deterministic seeding, environment capture
✅ **Stagewise Development** — Reduces risk; baseline baselines catch data bugs early
✅ **Evaluation** — Unified metrics framework; feature fidelity, anatomy, downstream utility

---

## Common Commands

**Train classifier only:**
```bash
python scripts/run_feature_conditioned.py --config configs/classifier.yaml --stage classifier
```

**Train Project A baseline (quick debug):**
```bash
python scripts/run_feature_conditioned.py --config configs/project_a.yaml --stage baseline
```

**Full Project A pipeline:**
```bash
for stage in classifier baseline main test; do
  python scripts/run_feature_conditioned.py --config configs/project_a.yaml --stage $stage
done
```

**Full Project B pipeline:**
```bash
for stage in classifier baseline main test; do
  python scripts/run_contralateral.py --config configs/project_b.yaml --stage $stage
done
```

---

## Dependencies

See `requirements.txt`:
- PyTorch, torchvision
- NumPy, Pandas, Pillow
- YAML config loading, scikit-learn splits
- TensorBoard logging
- LPIPS image metrics

Install:
```bash
pip install -r requirements.txt
```

---

## References

- Architecture Plan: `architecture_plan_thumb_cmc_oa.pdf`
- Denoising Diffusion Probabilistic Models (DDPM): [arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)
- Pix2Pix: [arxiv.org/abs/1611.05957](https://arxiv.org/abs/1611.05957)
- U-Net: [arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597)
