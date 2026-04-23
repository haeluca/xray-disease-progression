"""
Pix2Pix baseline for Project B (contralateral pseudo-longitudinal synthesis).

Pix2PixGenerator: UNet-style encoder-decoder with skip connections.
  Input: less-affected hand X-ray (1 channel).
  Output: predicted more-affected hand X-ray (1 channel) in [-1, 1].

PatchGANDiscriminator: 70×70 receptive field patch discriminator.
  Input: source-target image pair concatenated along the channel dimension (2 channels).
  Output: patch-level real/fake logits.
"""

import torch
import torch.nn as nn


class Pix2PixGenerator(nn.Module):
    """
    UNet generator for image-to-image translation.

    Skip connections concatenate encoder feature maps into decoder at matching
    spatial resolution. The final Tanh activation constrains outputs to [-1, 1].

    Args:
        input_channels:  Number of input image channels.
        output_channels: Number of output image channels.
    """

    def __init__(self, input_channels=1, output_channels=1):
        super().__init__()

        def conv_block(in_c, out_c, kernel_size=4, stride=2, padding=1, normalize=True):
            layers = [nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=not normalize)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2))
            return nn.Sequential(*layers)

        def deconv_block(in_c, out_c, kernel_size=4, stride=2, padding=1, dropout=0.0):
            layers = [nn.ConvTranspose2d(in_c, out_c, kernel_size, stride, padding)]
            layers.append(nn.InstanceNorm2d(out_c))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU())
            return nn.Sequential(*layers)

        self.encoder = nn.Sequential(
            conv_block(input_channels, 64, normalize=False),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512),
            conv_block(512, 512),
        )

        self.decoder = nn.Sequential(
            deconv_block(512, 512, dropout=0.5),
            deconv_block(1024, 256),
            deconv_block(512, 128),
            deconv_block(256, 64),
            deconv_block(128, output_channels, dropout=0.0),
            nn.Tanh(),
        )

    def forward(self, x):
        """
        Args:
            x: (B, input_channels, H, W) source image.
        Returns:
            Generated image (B, output_channels, H, W) in [-1, 1].
        """
        encoder_outputs = []
        h = x
        for layer in self.encoder:
            h = layer(h)
            encoder_outputs.append(h)

        h = encoder_outputs[-1]
        for i, layer in enumerate(self.decoder):
            # skip the first and last decoder steps — no skip for bottleneck or final Tanh
            if i > 0 and i < len(self.decoder) - 1:
                h = torch.cat([h, encoder_outputs[-(i + 1)]], dim=1)
            h = layer(h)

        return h


class PatchGANDiscriminator(nn.Module):
    """
    70×70 PatchGAN discriminator.

    Classifies overlapping 70×70 patches as real or fake rather than the whole
    image, encouraging high-frequency detail matching.

    Args:
        input_channels: Channels of the concatenated source+target pair (2 for grayscale pairs).
    """

    def __init__(self, input_channels=2):
        super().__init__()

        def conv_block(in_c, out_c, kernel_size=4, stride=2, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size, stride, padding),
                nn.InstanceNorm2d(out_c),
                nn.LeakyReLU(0.2),
            )

        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x):
        """
        Args:
            x: (B, 2, H, W) source-target image pair concatenated along channels.
        Returns:
            (B, 1, H', W') patch logits — positive = real, negative = fake.
        """
        return self.model(x)
