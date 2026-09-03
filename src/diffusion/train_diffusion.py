"""Phase 2 driver -- train the latent diffusion prior.

    python src/diffusion/train_diffusion.py --config configs/diffusion/ldm2d.yaml

Uses the frozen autoencoder from Phase 1 to encode every training slice into a
latent, standardizes the latents (stats saved with the checkpoint), and trains the
DDPM to denoise them. Saves best_ldm.pt / last_ldm.pt (with latent mean/std and the
autoencoder path baked in, so sampling is self-contained), history.json, config copy.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data2d import get_loader, Slices2D  # noqa: E402
from autoencoder2d import build_autoencoder  # noqa: E402
from ldm2d import build_diffusion  # noqa: E402


def load_autoencoder(ae_ckpt, device):
    ck = torch.load(ae_ckpt, map_location=device)
    ae = build_autoencoder(ck["config"]).to(device)
    ae.load_state_dict(ck["model_state"])
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae, ck["config"]


@torch.no_grad()
def encode_all(ae, loader, device):
    zs = []
    for x in loader:
        zs.append(ae.encode(x.to(device)).cpu())
    return torch.cat(zs, dim=0) if zs else torch.zeros(0)


def train_from_config(cfg: dict):
    d, m, t = cfg["data"], cfg["model"], cfg["training"]
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = d["cache_dir"]
    torch.manual_seed(t.get("seed", 42))
    np.random.seed(t.get("seed", 42))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ae, ae_cfg = load_autoencoder(d["ae_ckpt"], device)

    # encode train + val slices to latents (frozen AE), then standardize latents
    enc_loader = get_loader(cache, "train", t.get("batch_size", 32), shuffle=False,
                            num_workers=t.get("num_workers", 0))
    z_train = encode_all(ae, enc_loader, device)
    z_mean = z_train.mean().item()
    z_std = z_train.std().item() + 1e-8
    z_train = (z_train - z_mean) / z_std
    latent_ch, H, W = z_train.shape[1:]
    print(f"[ldm] encoded {z_train.shape[0]} latents of shape "
          f"({latent_ch},{H},{W}) | z_mean={z_mean:.4f} z_std={z_std:.4f} device={device}")

    diff = build_diffusion(cfg).to(device)
    opt = torch.optim.AdamW(diff.model.parameters(),
                            lr=t.get("learning_rate", 2e-4),
                            weight_decay=t.get("weight_decay", 0.0))
    print(f"[ldm] unet params: {sum(p.numel() for p in diff.model.parameters()):,}")

    shutil.copy2(_dump_cfg(cfg, out_dir), out_dir / "config_used.yaml")
    bs = t.get("batch_size", 32)
    n = z_train.shape[0]
    history, best = [], float("inf")
    epochs = t.get("epochs", 300)
    z_train = z_train.to(device)

    latent_meta = {"latent_ch": latent_ch, "H": H, "W": W,
                   "z_mean": z_mean, "z_std": z_std, "ae_ckpt": d["ae_ckpt"]}

    for ep in range(epochs):
        tic = time.time()
        perm = torch.randperm(n, device=device)
        diff.train()
        tot = 0.0
        for i in range(0, n, bs):
            batch = z_train[perm[i:i + bs]]
            opt.zero_grad()
            loss = diff.loss(batch)
            loss.backward()
            opt.step()
            tot += loss.item() * batch.shape[0]
        tot /= max(n, 1)
        dt = time.time() - tic
        history.append({"epoch": ep, "loss": tot, "sec": round(dt, 1)})
        if ep % max(1, epochs // 20) == 0 or ep == epochs - 1:
            print(f"[ldm] epoch {ep:4d} | loss {tot:.5f} | {dt:.1f}s")

        ckpt = {"epoch": ep, "model_state": diff.state_dict(), "config": cfg,
                "latent_meta": latent_meta, "ae_config": ae_cfg}
        torch.save(ckpt, out_dir / "last_ldm.pt")
        if tot < best:
            best = tot
            torch.save(ckpt, out_dir / "best_ldm.pt")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    print(f"[ldm] done. best loss {best:.5f}. artifacts in {out_dir}/")
    return {"best_loss": best, "out_dir": str(out_dir)}


def _dump_cfg(cfg, out_dir):
    p = out_dir / "_effective_config.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    train_from_config(yaml.safe_load(Path(args.config).read_text()))
