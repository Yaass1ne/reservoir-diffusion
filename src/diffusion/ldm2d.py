"""Phase 2 -- latent diffusion prior (2D).

A DDPM that runs in the autoencoder's latent space (default 16 x 8 x 8). It learns
to denoise latent codes; sampling from noise and decoding gives a fresh, plausible
rock field. Repeat the sampling and a whole ensemble comes out -- that ensemble is
the prior the assimilation step (Phase 4) will condition on.

Contains:
    SmallUNet          tiny time-conditioned UNet on the 8x8 latent grid
    GaussianDiffusion  linear-beta forward noising + epsilon-prediction training
                       loss + ancestral (DDPM) sampling

Gate for this phase: decoded samples look like plausible fields and their summary
stats match the training distribution (checked in sample_prior.py).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    """Sinusoidal embedding of integer timesteps -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, cin, cout, temb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, cin), cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.temb = nn.Linear(temb_dim, cout)
        self.norm2 = nn.GroupNorm(min(8, cout), cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(temb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SmallUNet(nn.Module):
    """Two-resolution UNet (8x8 <-> 4x4) with time conditioning."""

    def __init__(self, latent_ch=16, base=64, temb_dim=128):
        super().__init__()
        self.temb_dim = temb_dim
        self.temb_mlp = nn.Sequential(
            nn.Linear(temb_dim, temb_dim), nn.SiLU(), nn.Linear(temb_dim, temb_dim))
        self.in_conv = nn.Conv2d(latent_ch, base, 3, padding=1)
        self.d1 = ResBlock(base, base, temb_dim)
        self.down = nn.Conv2d(base, base * 2, 3, stride=2, padding=1)   # 8 -> 4
        self.mid = ResBlock(base * 2, base * 2, temb_dim)
        self.up = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)  # 4 -> 8
        self.u1 = ResBlock(base * 2, base, temb_dim)                     # concat skip
        self.out_norm = nn.GroupNorm(min(8, base), base)
        self.out_conv = nn.Conv2d(base, latent_ch, 3, padding=1)

    def forward(self, x, t):
        temb = self.temb_mlp(timestep_embedding(t, self.temb_dim))
        h0 = self.in_conv(x)
        h1 = self.d1(h0, temb)
        h = self.down(h1)
        h = self.mid(h, temb)
        h = self.up(h)
        h = self.u1(torch.cat([h, h1], dim=1), temb)
        return self.out_conv(F.silu(self.out_norm(h)))


class GaussianDiffusion(nn.Module):
    """DDPM with a linear beta schedule and epsilon prediction."""

    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=2e-2):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("acp", acp)
        self.register_buffer("sqrt_acp", torch.sqrt(acp))
        self.register_buffer("sqrt_one_minus_acp", torch.sqrt(1.0 - acp))

    def q_sample(self, x0, t, noise):
        return (self.sqrt_acp[t][:, None, None, None] * x0
                + self.sqrt_one_minus_acp[t][:, None, None, None] * noise)

    def loss(self, x0):
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        pred = self.model(xt, t)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, n, shape, device):
        """Ancestral DDPM sampling. shape = (latent_ch, H, W)."""
        x = torch.randn(n, *shape, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((n,), i, device=device, dtype=torch.long)
            eps = self.model(x, t)
            alpha = self.alphas[i]
            acp = self.acp[i]
            coef = (1 - alpha) / torch.sqrt(1 - acp)
            mean = (x - coef * eps) / torch.sqrt(alpha)
            if i > 0:
                x = mean + torch.sqrt(self.betas[i]) * torch.randn_like(x)
            else:
                x = mean
        return x


def build_diffusion(cfg: dict) -> GaussianDiffusion:
    m = cfg["model"]
    unet = SmallUNet(latent_ch=m.get("latent_ch", 16),
                     base=m.get("base", 64),
                     temb_dim=m.get("temb_dim", 128))
    return GaussianDiffusion(unet, timesteps=m.get("timesteps", 1000),
                             beta_start=m.get("beta_start", 1e-4),
                             beta_end=m.get("beta_end", 2e-2))


if __name__ == "__main__":
    diff = build_diffusion({"model": {"latent_ch": 16, "base": 32, "timesteps": 50}})
    z = torch.randn(4, 16, 8, 8)
    print("loss:", float(diff.loss(z)))
    s = diff.sample(2, (16, 8, 8), torch.device("cpu"))
    print("sample:", tuple(s.shape))
    print("OK")
