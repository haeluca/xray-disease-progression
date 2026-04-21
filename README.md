# DDPM X-Ray Disease Progression Model

A **Conditional Denoising Diffusion Probabilistic Model (DDPM)** that generates synthetic future X-ray images showing disease progression (CMC I OA) conditioned on baseline hand X-rays.

## 📋 Quick Overview

This project learns to simulate how osteoarthritis progresses in hand X-rays over time. Given a baseline X-ray, the model can generate realistic future progression images, useful for:
- Medical research and patient outcome prediction
- Training data augmentation for clinical ML models
- Understanding disease progression patterns

---

## 🏗️ Architecture & Model Components

### The Two Models: UNet + DDPM

#### **1. UNet** (`models/unet.py`)
The **denoising backbone** that learns to remove noise from X-ray images.

**Architecture:**
- **Input:** 2 channels (noisy progressed X-ray + baseline X-ray concatenated)
- **Output:** 1 channel (predicted noise to subtract)
- **Special feature:** Time embeddings that tell the network *which diffusion step* it's at

**Components:**
- **SinusoidalPositionEmbeddings:** Encodes the diffusion timestep into a 128-dim vector using sine/cosine waves (like positional encoding in transformers)
- **ResBlocks:** Residual blocks that process features while incorporating time information through an MLP
- **Encoder:** 4 downsampling stages with skip connections
- **Bottleneck:** 2 residual blocks at the lowest resolution
- **Decoder:** 4 upsampling stages with skip connections from encoder (U-Net style)
- **Final projection:** Outputs predicted noise

**Why this architecture?**
The U-Net with skip connections allows the model to preserve fine-grained details from the baseline X-ray while learning disease-specific deformations at multiple scales.

---

#### **2. DDPM** (`models/diffusion.py`)
The **diffusion model** that manages the noise schedule and orchestrates training/sampling.

**What it does:**

**Training phase:**
```
Input: baseline X-ray (x_t0), future X-ray (x_t1), random timestep (t)

1. Add noise to x_t1 according to timestep t:
   x_noisy = √(ᾱ_t) × x_t1 + √(1 - ᾱ_t) × random_noise

2. Concatenate: [x_noisy, x_t0] (noisy future + clean baseline)

3. UNet predicts what noise was added

4. Calculate MSE loss: ||predicted_noise - actual_noise||²

5. Backprop to improve UNet's denoising
```

**Sampling phase (generating new X-rays):**
```
Input: baseline X-ray (x_t0)

1. Start with pure random noise (x_T)

2. For t = 1000 down to 0:
   - Feed [x_t, x_t0] to UNet → get predicted noise
   - Remove predicted noise from x_t
   - If not final step, add small random noise back (controlled variance)
   - x_t = slightly_denoised_x_t + small_random_noise

3. Final x_0 = generated future X-ray
```

**Key parameters:**
- **T = 1000:** Number of diffusion steps (more steps = slower but smoother)
- **Beta schedule:** Linear schedule from 1e-4 to 0.02
  - Determines how much noise to add at each step
  - Small betas early (preserve signal), larger betas late (more noise)
- **Alphas:** Derived from betas, used to scale signal vs noise

---

## 🔄 The Diffusion Process Explained

### Why Diffusion?

Traditional generative models (VAE, GAN) directly transform noise → images.

**Diffusion models** do it step-by-step:
1. **Forward process (training):** Gradually add noise to real images
2. **Reverse process (sampling):** Gradually remove noise to generate images

This makes the learning problem **much easier**—the network only needs to predict one small denoising step at a time.

### Conditional Diffusion

Our model is **conditional**: it generates images *conditioned on* a baseline X-ray.

```
Traditional DDPM:
  noise → [denoise 1000 steps] → random image

Conditional DDPM (ours):
  noise + baseline_xray → [denoise 1000 steps] → realistic future progression
```

The baseline is concatenated to every denoising step, guiding the model to generate disease-realistic progressions on that specific patient's anatomy.

---

## 📁 Data Flow

### Input: Paired X-Rays

**Directory structure:**
```
data/train/
├── patient_001/
│   ├── t0.png  (baseline)
│   └── t1.png  (future/progression)
├── patient_002/
│   ├── t0.png
│   └── t1.png
└── ...
```

### Data Pipeline

**PairedXrayDataset** (`data/dataset.py`):
- Scans all patient directories
- Loads `t0.png` (baseline) and `t1.png` (future)
- Applies transforms

**Transforms** (`data/transforms.py`):
1. **Resize** to 256×256
2. **Random horizontal flip** (data augmentation)
3. **ToTensor** (convert PIL image to PyTorch tensor)
4. **Normalize** to [-1, 1] range (mean=0.5, std=0.5)

```python
# Normalization formula:
# output = (input / 255 - 0.5) / 0.5 → maps [0,1] to [-1,1]
```

---

## 🎯 Training Pipeline

### `train.py` Flow

```python
1. Load paired X-ray dataset
   └─ Returns batches of {x_t0, x_t1}

2. Initialize models:
   ├─ UNet (denoising network)
   └─ DDPM (diffusion orchestrator)

3. For each epoch (200 total):
   For each batch:
   
   a) Random timestep sampling
      └─ t = random integer in [0, 1000)
   
   b) Forward pass (DDPM.forward):
      ├─ Add noise to x_t1 at step t
      ├─ Concatenate with x_t0
      ├─ UNet predicts noise
      └─ Calculate MSE loss
   
   c) Backward pass
      ├─ optimizer.zero_grad()
      ├─ loss.backward()
      └─ optimizer.step()
   
   d) Log loss to progress bar

4. Every 10 epochs: Save checkpoint to `./checkpoints/`
```

**Hyperparameters:**
- Batch size: 8
- Learning rate: 1e-4
- Optimizer: AdamW
- Loss: MSE (mean squared error on predicted vs actual noise)

---

## 🎨 Sampling Pipeline

### `sample.py` Flow

```python
1. Load trained UNet checkpoint
   └─ Contains learned denoising weights

2. Initialize DDPM model
   └─ Loads diffusion schedule (same as training)

3. Load baseline X-ray
   └─ Resize, normalize to [-1, 1]

4. Call DDPM.sample(baseline_xray)
   └─ Runs iterative denoising (1000 steps)
   └─ Returns generated progression X-ray

5. Denormalize output
   └─ Maps [-1, 1] back to [0, 1] for image saving

6. Save as PNG
```

**Example usage:**
```bash
python sample.py \
  --baseline data/patient_001/t0.png \
  --checkpoint checkpoints/model_epoch_100.pt \
  --output generated_progression.png
```

---

## 🔑 Key Concepts

| Concept | Explanation |
|---------|-------------|
| **Noise schedule (betas)** | Controls how much noise is added at each diffusion step. Linear schedule balances stability and quality. |
| **Timestep embedding** | Sinusoidal encoding tells UNet which step it's denoising at. This is crucial—the network needs to know *when* it is. |
| **Conditioning** | Concatenating baseline X-ray ensures generated progressions respect the patient's baseline anatomy. |
| **MSE loss** | We only predict noise, not images directly. This makes the learning signal cleaner. |
| **Skip connections** | U-Net's skip connections preserve baseline details while learning progression-specific features. |
| **Grayscale (1 channel)** | X-rays are inherently grayscale; color information is unnecessary. Saves memory and computation. |

---

## 📊 Model Capacity & Parameters

**UNet layers:**
- Initial conv: 2 → 64 channels
- Encoder: 64 → 128 → 256 → 512 (with downsampling)
- Bottleneck: 512 channels
- Decoder: 512 → 256 → 128 → 64 (with upsampling + skip connections)
- Final conv: 64 → 1 channel

**Total parameters:** ~40M (typical for UNet this size)

---

## 💡 Why This Approach?

### Advantages of DDPM over alternatives:

| Method | Pro | Con |
|--------|-----|-----|
| **GANs** | Fast sampling | Unstable training, mode collapse |
| **VAE** | Stable, interpretable latent space | Blurry outputs |
| **Diffusion (DDPM)** | **High quality, stable, flexible conditioning** | **Slow sampling (1000 steps)** |

For medical imaging, **quality and stability** are critical. Diffusion models excel here.

---

## 🚀 Next Steps / Extension Ideas

1. **Accelerated sampling:** Use faster diffusion schedulers (DDIM, Karras) to reduce 1000 steps to ~50
2. **Better conditioning:** Add temporal information (disease stage classification) to guide progression
3. **Multi-timepoint:** Condition on multiple baseline timepoints for better anatomical consistency
4. **Quantitative evaluation:** Compare generated vs real progression images using radiological metrics
5. **Fine-tuning:** Start from pretrained diffusion models (ImageNet) instead of scratch

---

## 📝 File Structure Summary

```
├── config.py              # Hyperparameters (image size, beta schedule, learning rate)
├── train.py               # Training loop
├── sample.py              # Inference/sampling script
│
├── models/
│   ├── unet.py            # UNet denoising network
│   ├── diffusion.py       # DDPM diffusion model wrapper
│   └── __init__.py
│
├── data/
│   ├── dataset.py         # PairedXrayDataset loader
│   ├── transforms.py      # Image preprocessing (resize, normalize)
│   └── __init__.py
│
└── checkpoints/           # Saved model weights (created during training)
```

---

## ⚙️ Technical Details

### Noise Scheduling

The diffusion process uses a **linear beta schedule**:
```
β_t = β_start + (β_end - β_start) × t / T
    = 0.0001 + 0.0199 × t / 1000
```

Then α_t = 1 - β_t, and ᾱ_t = ∏(α_i) from i=0 to t.

These values control the signal-to-noise ratio at each step.

### Time Embeddings

Timesteps are encoded using sinusoids (like in Transformers):
```
emb[2k]   = sin(t / 10000^(2k/dim))
emb[2k+1] = cos(t / 10000^(2k/dim))
```

This allows the UNet to learn which diffusion step it's at, essential for multi-scale denoising.

---

## 📖 References

- Denoising Diffusion Probabilistic Models (DDPM): [arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)
- Classifier-Free Diffusion Guidance (for future improvements): [arxiv.org/abs/2207.12598](https://arxiv.org/abs/2207.12598)
