import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
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
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb):
        h = self.block1(x)
        h += self.mlp(t_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)

class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, time_emb_dim=128, channels=(64, 128, 256, 512)):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.time_embeddings = SinusoidalPositionEmbeddings(time_emb_dim)

        self.initial_conv = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)

        # Encoder
        self.encoder_blocks = nn.ModuleList([
            nn.ModuleList([
                ResBlock(channels[i], channels[i], time_emb_dim),
                ResBlock(channels[i], channels[i], time_emb_dim),
                Downsample(channels[i]) if i < len(channels) - 1 else nn.Identity()
            ]) for i in range(len(channels))
        ])

        # Bottleneck
        self.bottleneck = nn.ModuleList([
            ResBlock(channels[-1], channels[-1], time_emb_dim),
            ResBlock(channels[-1], channels[-1], time_emb_dim)
        ])

        # Decoder
        self.decoder_blocks = nn.ModuleList([
            nn.ModuleList([
                ResBlock(channels[i] * 2, channels[i], time_emb_dim),
                ResBlock(channels[i], channels[i], time_emb_dim),
                Upsample(channels[i]) if i > 0 else nn.Identity()
            ]) for i in reversed(range(len(channels)))
        ])

        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], out_channels, kernel_size=1)
        )

    def forward(self, x, t):
        t_emb = self.time_embeddings(t)
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
            h = torch.cat([h, encoder_outputs[-(i + 1)]], dim=1)
            h = block_list[0](h, t_emb)
            h = block_list[1](h, t_emb)
            h = block_list[2](h)

        return self.final_conv(h)
