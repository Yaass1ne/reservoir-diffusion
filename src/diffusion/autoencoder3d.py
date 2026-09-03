"""Phase 6 (1) -- 3D convolutional autoencoder.

Compresses a Perm volume (1, Z, X, Y) = (1, 24, 60, 60) to a compact 3D latent
and rebuilds it. down_levels stride-2 stages: from the padded working size
(Zp, 64, 64), 2 downs -> (Zp/4, 16, 16). For Z=24 that is a (6, 16, 16) latent.

X, Y are padded to 64 (reflect) and cropped back to 60 on decode; Z is padded up
to a multiple of 2**down_levels. Same recon-R2 gate idea as the 2D version.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

WORK_XY = 64


def _pad(x, down_levels):
    """(B,C,Z,X,Y) -> padded so each spatial dim divides 2**down_levels; XY->64."""
    f = 2 ** down_levels
    Z, X, Y = x.shape[-3:]
    Zp = ((Z + f - 1) // f) * f
    pz, px, py = Zp - Z, WORK_XY - X, WORK_XY - Y
    pad = (px // 2, px - px // 2, 0, 0, 0, 0)  # placeholder, build below
    # F.pad order for 5D: (Yl,Yr, Xl,Xr, Zl,Zr)
    pad = (py // 2, py - py // 2, px // 2, px - px // 2, pz // 2, pz - pz // 2)
    return F.pad(x, pad, mode="replicate"), (Z, X, Y), (pz, px, py)


def _crop(x, orig, pads):
    Z, X, Y = orig
    pz, px, py = pads
    zl, xl, yl = pz // 2, px // 2, py // 2
    return x[..., zl:zl + Z, xl:xl + X, yl:yl + Y]


class _Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1), nn.GroupNorm(min(8, cout), cout), nn.SiLU(),
            nn.Conv3d(cout, cout, 3, padding=1), nn.GroupNorm(min(8, cout), cout), nn.SiLU())

    def forward(self, x):
        return self.net(x)


class AutoEncoder3D(nn.Module):
    def __init__(self, in_ch=1, base=24, latent_ch=8, down_levels=2):
        super().__init__()
        self.in_ch, self.latent_ch, self.down_levels = in_ch, latent_ch, down_levels
        enc_ch = [base * (2 ** i) for i in range(down_levels)]
        self.enc = nn.ModuleList()
        c = in_ch
        for ch in enc_ch:
            self.enc.append(_Block(c, ch)); c = ch
        self.down = nn.AvgPool3d(2)
        self.to_latent = nn.Conv3d(enc_ch[-1], latent_ch, 1)
        self.from_latent = nn.Conv3d(latent_ch, enc_ch[-1], 1)
        dec_ch = list(reversed(enc_ch[:-1])) + [base]
        self.dec = nn.ModuleList()
        c = enc_ch[-1]
        for ch in dec_ch:
            self.dec.append(_Block(c, ch)); c = ch
        self.out = nn.Conv3d(c, in_ch, 1)

    def encode(self, x):
        xp, self._orig, self._pads = _pad(x, self.down_levels)
        for blk in self.enc:
            xp = self.down(blk(xp))
        return self.to_latent(xp)

    def pads_for(self, grid):
        """Padding (pz,px,py) implied by a target (Z,X,Y) grid + down_levels."""
        f = 2 ** self.down_levels
        Z, X, Y = grid
        Zp = ((Z + f - 1) // f) * f
        return (Zp - Z, WORK_XY - X, WORK_XY - Y)

    def decode(self, z, grid=None, pads=None):
        # grid=(Z,X,Y) lets us decode generated latents that never saw encode()
        orig = grid or getattr(self, "_orig", None)
        if pads is None:
            pads = self.pads_for(orig) if grid else getattr(self, "_pads", None)
        x = self.from_latent(z)
        for blk in self.dec:
            x = F.interpolate(x, scale_factor=2, mode="nearest"); x = blk(x)
        x = self.out(x)
        return _crop(x, orig, pads) if orig else x

    def forward(self, x):
        return self.decode(self.encode(x))

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_autoencoder3d(cfg):
    m = cfg["model"]
    return AutoEncoder3D(in_ch=m.get("in_ch", 1), base=m.get("base", 24),
                         latent_ch=m.get("latent_ch", 8), down_levels=m.get("down_levels", 2))


def recon_loss(pred, target, grad_weight=0.1):
    mse = F.mse_loss(pred, target)
    g = pred.new_zeros(())
    for dim in (-3, -2, -1):
        g = g + (torch.diff(pred, dim=dim) - torch.diff(target, dim=dim)).abs().mean()
    return mse + grad_weight * (g / 3.0)


if __name__ == "__main__":
    net = AutoEncoder3D(in_ch=1, base=8, latent_ch=8, down_levels=2)
    x = torch.randn(2, 1, 24, 60, 60)
    z = net.encode(x); y = net.decode(z)
    print("in", tuple(x.shape), "z", tuple(z.shape), "=", z[0].numel(),
          "out", tuple(y.shape), "params", f"{net.num_params:,}")
    assert y.shape == x.shape, y.shape
    print("OK")
