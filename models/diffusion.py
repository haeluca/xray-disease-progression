import torch
import torch.nn as nn
import numpy as np
from config import T, BETA_START, BETA_END, DEVICE

class DDPM(nn.Module):
    def __init__(self, unet_model):
        super().__init__()
        self.unet = unet_model
        self.T = T

        betas = torch.linspace(BETA_START, BETA_END, T)
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

    def forward(self, x_t1, x_t0, t):
        batch_size = x_t1.shape[0]
        noise = torch.randn_like(x_t1)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]

        x_noisy = sqrt_alphas_cumprod_t[:, None, None, None] * x_t1 + sqrt_one_minus_alphas_cumprod_t[:, None, None, None] * noise

        # Concatenate noisy x_t1 with conditioning x_t0
        x_combined = torch.cat([x_noisy, x_t0], dim=1)

        predicted_noise = self.unet(x_combined, t)
        loss = nn.functional.mse_loss(predicted_noise, noise)

        return loss

    @torch.no_grad()
    def sample(self, x_t0, shape):
        batch_size = x_t0.shape[0]
        x_t = torch.randn(shape, device=DEVICE)

        for t in reversed(range(self.T)):
            t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=DEVICE)

            sqrt_recip_alphas_t = self.sqrt_recip_alphas[t]
            sqrt_recipm1_alphas_cumprod_t = self.sqrt_recipm1_alphas_cumprod[t]
            betas_t = self.betas[t]

            x_combined = torch.cat([x_t, x_t0], dim=1)
            predicted_noise = self.unet(x_combined, t_tensor)

            x_t = sqrt_recip_alphas_t[None, None, None] * (x_t - betas_t[None, None, None] * sqrt_recipm1_alphas_cumprod_t[None, None, None] * predicted_noise)

            if t > 0:
                noise = torch.randn_like(x_t)
                posterior_variance = betas_t * (1.0 - self.alphas_cumprod[t - 1]) / (1.0 - self.alphas_cumprod[t])
                x_t = x_t + torch.sqrt(posterior_variance)[None, None, None] * noise

        return x_t
