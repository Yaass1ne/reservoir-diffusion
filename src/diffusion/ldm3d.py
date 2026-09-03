"""Phase 6 (2) -- 3D latent diffusion prior.

DDPM on the 3D autoencoder latents (default 8 x 6 x 16 x 16). A small 3D UNet
denoiser; sampling from noise + decoding gives a fresh Perm volume, so repeated
sampling yields an ensemble. Same machinery as ldm2d.py, in 3D.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], -1)
    return F.pad(emb, (0, 1)) if dim % 2 else emb


class ResBlock(nn.Module):
    def __init__(self, cin, cout, temb):
        super().__init__()
        self.n1 = nn.GroupNorm(min(8, cin), cin); self.c1 = nn.Conv3d(cin, cout, 3, padding=1)
        self.temb = nn.Linear(temb, cout)
        self.n2 = nn.GroupNorm(min(8, cout), cout); self.c2 = nn.Conv3d(cout, cout, 3, padding=1)
        self.skip = nn.Conv3d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.temb(temb)[:, :, None, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class SmallUNet3D(nn.Module):
    """One down/up level (stride 2 on all axes) with time conditioning."""

    def __init__(self, latent_ch=8, base=48, temb_dim=128):
        super().__init__()
        self.temb_dim = temb_dim
        self.temb = nn.Sequential(nn.Linear(temb_dim, temb_dim), nn.SiLU(),
                                  nn.Linear(temb_dim, temb_dim))
        self.inc = nn.Conv3d(latent_ch, base, 3, padding=1)
        self.d1 = ResBlock(base, base, temb_dim)
        # downsample X,Y only (Z is already small, e.g. 6); keeps it robust for any Z
        self.down = nn.Conv3d(base, base * 2, 3, stride=(1, 2, 2), padding=1)
        self.mid = ResBlock(base * 2, base * 2, temb_dim)
        self.up = nn.ConvTranspose3d(base * 2, base, (1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
        self.u1 = ResBlock(base * 2, base, temb_dim)
        self.outn = nn.GroupNorm(min(8, base), base)
        self.outc = nn.Conv3d(base, latent_ch, 3, padding=1)

    def forward(self, x, t):
        temb = self.temb(timestep_embedding(t, self.temb_dim))
        h0 = self.inc(x)
        h1 = self.d1(h0, temb)
        h = self.mid(self.down(h1), temb)
        h = self.up(h, output_size=h1.shape[-3:])
        h = self.u1(torch.cat([h, h1], 1), temb)
        return self.outc(F.silu(self.outn(h)))


class GaussianDiffusion3D(nn.Module):
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=2e-2):
        super().__init__()
        self.model = model; self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1 - betas; acp = torch.cumprod(alphas, 0)
        for k, v in dict(betas=betas, alphas=alphas, acp=acp,
                         sqrt_acp=torch.sqrt(acp),
                         sqrt_om=torch.sqrt(1 - acp)).items():
            self.register_buffer(k, v)

    def q_sample(self, x0, t, noise):
        b = lambda a: a[t][:, None, None, None, None]
        return b(self.sqrt_acp) * x0 + b(self.sqrt_om) * noise

    def loss(self, x0):
        t = torch.randint(0, self.timesteps, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        return F.mse_loss(self.model(self.q_sample(x0, t, noise), t), noise)

    @torch.no_grad()
    def sample(self, n, shape, device):
        x = torch.randn(n, *shape, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((n,), i, device=device, dtype=torch.long)
            eps = self.model(x, t)
            a, ac = self.alphas[i], self.acp[i]
            mean = (x - (1 - a) / torch.sqrt(1 - ac) * eps) / torch.sqrt(a)
            x = mean + (torch.sqrt(self.betas[i]) * torch.randn_like(x) if i > 0 else 0)
        return x


def build_diffusion3d(cfg):
    m = cfg["model"]
    unet = SmallUNet3D(latent_ch=m.get("latent_ch", 8), base=m.get("base", 48),
                       temb_dim=m.get("temb_dim", 128))
    return GaussianDiffusion3D(unet, timesteps=m.get("timesteps", 1000),
                               beta_start=m.get("beta_start", 1e-4),
                               beta_end=m.get("beta_end", 2e-2))


if __name__ == "__main__":
    diff = build_diffusion3d({"model": {"latent_ch": 8, "base": 16, "timesteps": 20}})
    z = torch.randn(2, 8, 6, 16, 16)
    print("loss", float(diff.loss(z)))
    print("sample", tuple(diff.sample(1, (8, 6, 16, 16), torch.device("cpu")).shape))
    print("OK")
