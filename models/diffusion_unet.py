"""
Conditional U-Net denoising backbone (DiffusionUNet) and DDPM diffusion wrapper.

Architecture:
  - DiffusionUNet: 4-level encoder-decoder with skip connections. Accepts a
    2-channel input (noisy target concatenated with conditioning image), a
    sinusoidal timestep embedding, and an optional feature-vector condition
    that is projected into the timestep embedding space.
  - DDPM: wraps DiffusionUNet with the standard linear noise schedule. Exposes
    forward() for training (adds noise, predicts, returns MSE) and sample() for
    reverse-diffusion inference (T steps from pure noise to generated image).
"""

import math
import torch
import torch.nn as nn


class SinusoidalPositionEmbeddings(nn.Module):
    """Maps scalar timestep t → fixed sinusoidal embedding of length `dim`."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResBlock(nn.Module):
    """
    Residual conv block with timestep conditioning.

    Injects the time (+ optional condition) embedding additively after the
    first conv, following the DDPM ResBlock design.
    """

    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.res_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, t_emb):
        h = self.block1(x)
        # broadcast time embedding over spatial dims
        h += self.mlp(t_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)


class Downsample(nn.Module):
    """Strided 4×4 conv that halves spatial resolution."""

    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Nearest-neighbour ×2 upsample followed by a 3×3 conv to reduce aliasing."""

    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class DiffusionUNet(nn.Module):
    """
    Conditional denoising U-Net backbone for DDPM.

    Input channels: 2 — noisy target image concatenated with conditioning image.
    The feature condition vector (if given) is projected to time_emb_dim and
    added to the timestep embedding so conditioning is injected at every ResBlock.

    Args:
        in_channels:    Number of input channels (2 = noisy + condition image).
        out_channels:   Number of output channels (predicted noise, 1 for grayscale).
        time_emb_dim:   Dimensionality of the sinusoidal timestep embedding.
        channels:       Feature map widths at each of the 4 encoder levels.
        condition_dim:  Dimension of the feature condition vector; 0 disables it.
    """

    def __init__(
        self,
        in_channels=2,
        out_channels=1,
        time_emb_dim=128,
        channels=(64, 128, 256, 512),
        condition_dim=0,
    ):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.time_embeddings = SinusoidalPositionEmbeddings(time_emb_dim)

        if condition_dim > 0:
            self.condition_proj = nn.Linear(condition_dim, time_emb_dim)
        else:
            self.condition_proj = None

        self.initial_conv = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)

        self.encoder_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        ResBlock(channels[0] if i == 0 else channels[i - 1], channels[i], time_emb_dim),
                        ResBlock(channels[i], channels[i], time_emb_dim),
                        Downsample(channels[i]) if i < len(channels) - 1 else nn.Identity(),
                    ]
                )
                for i in range(len(channels))
            ]
        )

        self.bottleneck = nn.ModuleList(
            [
                ResBlock(channels[-1], channels[-1], time_emb_dim),
                ResBlock(channels[-1], channels[-1], time_emb_dim),
            ]
        )

        decoder_list = []
        for order, i in enumerate(reversed(range(len(channels)))):
            incoming = channels[-1] if order == 0 else channels[i + 1]
            skip = channels[i]
            decoder_list.append(
                nn.ModuleList(
                    [
                        ResBlock(incoming + skip, channels[i], time_emb_dim),
                        ResBlock(channels[i], channels[i], time_emb_dim),
                        Upsample(channels[i]) if i > 0 else nn.Identity(),
                    ]
                )
            )
        self.decoder_blocks = nn.ModuleList(decoder_list)

        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], out_channels, kernel_size=1),
        )

    def forward(self, x, t, condition=None):
        """
        Args:
            x:         (B, in_channels, H, W) — noisy target concat with condition image.
            t:         (B,) integer timestep tensor.
            condition: (B, condition_dim) float feature vector, or None.
        Returns:
            Predicted noise tensor (B, out_channels, H, W).
        """
        t_emb = self.time_embeddings(t)

        if condition is not None and self.condition_proj is not None:
            condition_emb = self.condition_proj(condition)
            t_emb = t_emb + condition_emb

        h = self.initial_conv(x)
        encoder_outputs = []

        for block_list in self.encoder_blocks:
            h = block_list[0](h, t_emb)
            h = block_list[1](h, t_emb)
            encoder_outputs.append(h)
            h = block_list[2](h)

        for block in self.bottleneck:
            h = block(h, t_emb)

        for i, block_list in enumerate(self.decoder_blocks):
            # concatenate encoder skip connection before each decoder stage
            h = torch.cat([h, encoder_outputs[-(i + 1)]], dim=1)
            h = block_list[0](h, t_emb)
            h = block_list[1](h, t_emb)
            h = block_list[2](h)

        return self.final_conv(h)


class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model wrapper around DiffusionUNet.

    Precomputes the linear noise schedule and all derived quantities used in
    the forward (training) and reverse (sampling) passes.

    Args:
        unet:       DiffusionUNet backbone.
        T:          Total diffusion timesteps.
        beta_start: Starting value of the linear beta schedule.
        beta_end:   Ending value of the linear beta schedule.
        device:     Device on which schedule buffers are allocated.
    """

    def __init__(self, unet, T=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        super().__init__()
        self.unet = unet
        self.T = T
        self.device = device

        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

    def forward(self, x_target, x_condition=None, t=None, condition_vector=None):
        """
        Training step: corrupt x_target at timestep t, predict the noise, return MSE loss.

        Args:
            x_target:          (B, C, H, W) clean target image in [-1, 1].
            x_condition:       (B, C, H, W) conditioning image concatenated channel-wise, or
                               None for unconditional / feature-only conditioning (Project A).
                               Must be provided when the UNet was built with in_channels > 1.
            t:                 (B,) integer timestep sampled uniformly from [0, T).
            condition_vector:  (B, condition_dim) feature vector, or None.
        Returns:
            Scalar MSE loss between predicted and actual noise.

        Raises:
            ValueError: if x_condition is None but the UNet expects more than 1 input channel.
        """
        if t is None:
            raise ValueError("Timestep t must be provided to DDPM.forward().")

        in_channels = self.unet.initial_conv.in_channels
        if x_condition is None and in_channels > 1:
            raise ValueError(
                f"DDPM.forward: x_condition is None but DiffusionUNet was built with "
                f"in_channels={in_channels}. Either pass x_condition or rebuild with in_channels=1."
            )

        noise = torch.randn_like(x_target)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]

        x_noisy = (
            sqrt_alphas_cumprod_t[:, None, None, None] * x_target
            + sqrt_one_minus_alphas_cumprod_t[:, None, None, None] * noise
        )

        if x_condition is not None:
            x_combined = torch.cat([x_noisy, x_condition], dim=1)
        else:
            x_combined = x_noisy

        predicted_noise = self.unet(x_combined, t, condition=condition_vector)
        loss = torch.nn.functional.mse_loss(predicted_noise, noise)

        return loss

    @torch.no_grad()
    def sample(self, shape, condition_vector=None, x_condition=None):
        """
        Reverse diffusion: iteratively denoise pure Gaussian noise into a generated image.

        Args:
            shape:            Output shape tuple (B, C, H, W). B determines batch size.
            condition_vector: (B, condition_dim) feature vector, or None.
            x_condition:      (B, C, H, W) conditioning image concatenated at every step, or
                              None for unconditional / feature-only conditioning (Project A).
                              Must be provided when the UNet was built with in_channels > 1.
        Returns:
            Generated image tensor (B, C, H, W) in approximately [-1, 1].

        Raises:
            ValueError: if x_condition is None but the UNet expects more than 1 input channel.
        """
        in_channels = self.unet.initial_conv.in_channels
        if x_condition is None and in_channels > 1:
            raise ValueError(
                f"DDPM.sample: x_condition is None but DiffusionUNet was built with "
                f"in_channels={in_channels}. Either pass x_condition or rebuild with in_channels=1."
            )

        batch_size = shape[0]
        x_t = torch.randn(shape, device=self.device)

        for t in reversed(range(self.T)):
            t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=self.device)

            sqrt_recip_alphas_t = self.sqrt_recip_alphas[t]
            sqrt_recipm1_alphas_cumprod_t = self.sqrt_recipm1_alphas_cumprod[t]
            betas_t = self.betas[t]

            if x_condition is not None:
                x_combined = torch.cat([x_t, x_condition], dim=1)
            else:
                x_combined = x_t

            predicted_noise = self.unet(x_combined, t_tensor, condition=condition_vector)

            # DDPM reverse step (eq. 11 in Ho et al. 2020)
            x_t = sqrt_recip_alphas_t * (
                x_t - betas_t * sqrt_recipm1_alphas_cumprod_t * predicted_noise
            )

            if t > 0:
                noise = torch.randn_like(x_t)
                posterior_variance = (
                    betas_t * (1.0 - self.alphas_cumprod[t - 1]) / (1.0 - self.alphas_cumprod[t])
                )
                x_t = x_t + torch.sqrt(posterior_variance) * noise

        return x_t
