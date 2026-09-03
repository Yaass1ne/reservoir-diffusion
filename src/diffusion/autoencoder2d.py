"""Phase 1 -- 2D convolutional autoencoder.

Compresses a normalized field (C, 60, 60) to a small latent grid and reconstructs
it. The latent is what the diffusion model (Phase 2) generates, and what ESMDA
(Phase 4) adjusts -- adjusting ~1000 numbers instead of every cell is what keeps
the whole pipeline affordable.

The 60x60 grid is padded to 64x64 (a power of two) so the strided convs halve
cleanly, then cropped back on decode:
    64 -> 32 -> 16 -> 8   (three stride-2 downs)
    latent = latent_ch x 8 x 8  (default 16 -> 1024 numbers)

Gate for this phase: reconstruction R^2 per field on val is high (>~0.98) and the
slice plots look sharp, not blurred. If this is weak, nothing built on top of it
can be trusted.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

GRID = 60
PAD = 64  # padded working size (power of two)


def _pad_to_work(x):
    """(B, C, 60, 60) -> (B, C, 64, 64), symmetric-ish reflect padding."""
    p = PAD - GRID  # 4
    a = p // 2
    b = p - a
    return F.pad(x, (a, b, a, b), mode="reflect")


def _crop_to_grid(x):
    p = PAD - GRID
    a = p // 2
    return x[..., a:a + GRID, a:a + GRID]


class _ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.GroupNorm(min(8, cout), cout),
            nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.GroupNorm(min(8, cout), cout),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class AutoEncoder2D(nn.Module):
    """down_levels stride-2 stages set the latent grid: from the 64x64 working
    size, 3 downs -> 8x8, 2 downs -> 16x16. Fewer downs = larger latent = better
    reconstruction (less spatial squeeze), at a slightly bigger latent to diffuse.
    """

    def __init__(self, in_ch=2, base=32, latent_ch=16, down_levels=3):
        super().__init__()
        self.in_ch = in_ch
        self.latent_ch = latent_ch
        self.down_levels = down_levels
        enc_ch = [base * (2 ** i) for i in range(down_levels)]   # e.g. [32,64,128]
        # encoder: down_levels stride-2 stages
        self.enc = nn.ModuleList()
        cin = in_ch
        for c in enc_ch:
            self.enc.append(_ConvBlock(cin, c))
            cin = c
        self.down = nn.AvgPool2d(2)
        self.to_latent = nn.Conv2d(enc_ch[-1], latent_ch, 1)
        # decoder mirrors the encoder
        self.from_latent = nn.Conv2d(latent_ch, enc_ch[-1], 1)
        dec_ch = list(reversed(enc_ch[:-1])) + [base]           # e.g. [64,32,32]
        self.dec = nn.ModuleList()
        cin = enc_ch[-1]
        for c in dec_ch:
            self.dec.append(_ConvBlock(cin, c))
            cin = c
        self.out = nn.Conv2d(cin, in_ch, 1)

    def encode(self, x):
        x = _pad_to_work(x)
        for blk in self.enc:
            x = self.down(blk(x))
        return self.to_latent(x)          # (B, latent_ch, 8, 8)

    def decode(self, z):
        x = self.from_latent(z)
        for blk in self.dec:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = blk(x)
        x = self.out(x)                    # (B, in_ch, 64, 64)
        return _crop_to_grid(x)            # (B, in_ch, 60, 60)

    def forward(self, x):
        return self.decode(self.encode(x))

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_autoencoder(cfg: dict) -> AutoEncoder2D:
    m = cfg["model"]
    n_fields = len(cfg["data"].get("fields", ["Perm", "Por"])) \
        if "data" in cfg else m.get("in_ch", 2)
    return AutoEncoder2D(
        in_ch=m.get("in_ch", n_fields),
        base=m.get("base", 32),
        latent_ch=m.get("latent_ch", 16),
        down_levels=m.get("down_levels", 3),
    )


def recon_loss(pred, target, grad_weight=0.1):
    """MSE + light 2D gradient-L1 so edges near wells are not smoothed away."""
    mse = F.mse_loss(pred, target)
    g = pred.new_zeros(())
    for dim in (-2, -1):
        g = g + (torch.diff(pred, dim=dim) - torch.diff(target, dim=dim)).abs().mean()
    return mse + grad_weight * (g / 2.0)


if __name__ == "__main__":
    for dl in (3, 2):
        net = AutoEncoder2D(in_ch=1, base=16, latent_ch=16, down_levels=dl)
        x = torch.randn(4, 1, 60, 60)
        z = net.encode(x)
        y = net.decode(z)
        print(f"down_levels={dl} | z {tuple(z.shape)} = {z[0].numel()} numbers "
              f"| out {tuple(y.shape)} | params {net.num_params:,}")
        assert y.shape == x.shape, y.shape
    print("OK")
