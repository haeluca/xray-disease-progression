"""
Conditional VAE baseline for Project A (feature-conditioned X-ray synthesis).

The feature condition vector is concatenated to the encoder output before the
reparameterisation step and again to the latent z before decoding, so the
generator can target arbitrary feature states without seeing a paired image.
"""

import torch
import torch.nn as nn


class ConditionalVAE(nn.Module):
    """
    Conditional Variational Autoencoder for single-channel X-ray synthesis.

    Condition vector is injected at both encode (before z_mean/z_logvar) and
    decode (before the FC projection back to spatial feature maps).

    Args:
        image_channels: Number of input/output image channels (1 for grayscale).
        latent_dim:     Dimensionality of the latent space z.
        condition_dim:  Dimensionality of the feature condition vector.
        image_size:     Spatial size of the square input image.
    """

    def __init__(self, image_channels=1, latent_dim=32, condition_dim=0, image_size=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        # spatial size after 4 stride-2 downsamples
        self.decoder_spatial = image_size // 16

        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.z_mean = nn.Linear(256 + condition_dim, latent_dim)
        self.z_logvar = nn.Linear(256 + condition_dim, latent_dim)

        self.condition_dim = condition_dim

        self.decoder_fc = nn.Linear(latent_dim + condition_dim, 256 * self.decoder_spatial * self.decoder_spatial)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def encode(self, x, condition=None):
        """Encode image (+ optional condition) to z_mean and z_logvar."""
        h = self.encoder(x)
        h = h.view(h.size(0), -1)

        if condition is not None:
            h = torch.cat([h, condition], dim=1)

        z_mean = self.z_mean(h)
        z_logvar = self.z_logvar(h)

        return z_mean, z_logvar

    def reparameterize(self, z_mean, z_logvar):
        """Sample z via the reparameterisation trick: z = μ + ε·σ, ε ~ N(0,1)."""
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        return z_mean + eps * std

    def decode(self, z, condition=None):
        """Decode latent z (+ optional condition) to a reconstructed image in [-1, 1]."""
        if condition is not None:
            z = torch.cat([z, condition], dim=1)

        h = self.decoder_fc(z)
        h = h.view(h.size(0), 256, self.decoder_spatial, self.decoder_spatial)
        x_recon = self.decoder(h)

        return x_recon

    def forward(self, x, condition=None):
        """
        Args:
            x:         (B, 1, H, W) input image in [-1, 1].
            condition: (B, condition_dim) feature vector, or None.
        Returns:
            Tuple of (reconstruction, z_mean, z_logvar) — z stats needed for KL loss.
        """
        z_mean, z_logvar = self.encode(x, condition)
        z = self.reparameterize(z_mean, z_logvar)
        x_recon = self.decode(z, condition)

        return x_recon, z_mean, z_logvar
