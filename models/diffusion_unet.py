import math
import torch
import torch.nn as nn


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


class DiffusionUNet(nn.Module):
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
            h = torch.cat([h, encoder_outputs[-(i + 1)]], dim=1)
            h = block_list[0](h, t_emb)
            h = block_list[1](h, t_emb)
            h = block_list[2](h)

        return self.final_conv(h)


class DDPM(nn.Module):
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

    def forward(self, x_target, x_condition, t, condition_vector=None):
        batch_size = x_target.shape[0]
        noise = torch.randn_like(x_target)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]

        x_noisy = (
            sqrt_alphas_cumprod_t[:, None, None, None] * x_target
            + sqrt_one_minus_alphas_cumprod_t[:, None, None, None] * noise
        )

        x_combined = torch.cat([x_noisy, x_condition], dim=1)
        predicted_noise = self.unet(x_combined, t, condition=condition_vector)
        loss = torch.nn.functional.mse_loss(predicted_noise, noise)

        return loss

    @torch.no_grad()
    def sample(self, x_condition, shape, condition_vector=None):
        batch_size = x_condition.shape[0]
        x_t = torch.randn(shape, device=self.device)

        for t in reversed(range(self.T)):
            t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=self.device)

            sqrt_recip_alphas_t = self.sqrt_recip_alphas[t]
            sqrt_recipm1_alphas_cumprod_t = self.sqrt_recipm1_alphas_cumprod[t]
            betas_t = self.betas[t]

            x_combined = torch.cat([x_t, x_condition], dim=1)
            predicted_noise = self.unet(x_combined, t_tensor, condition=condition_vector)

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
