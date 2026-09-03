"""Phase 6 (2) driver -- train the 3D latent diffusion prior.
    python src/diffusion/train_diffusion3d.py --config configs/diffusion/ldm3d.yaml
"""
from __future__ import annotations

import argparse, json, shutil, sys, time
from pathlib import Path
import numpy as np, torch, yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data3d import get_loader  # noqa: E402
from autoencoder3d import build_autoencoder3d  # noqa: E402
from ldm3d import build_diffusion3d  # noqa: E402


def load_ae(ckpt, device):
    ck = torch.load(ckpt, map_location=device)
    ae = build_autoencoder3d(ck["config"]).to(device); ae.load_state_dict(ck["model_state"])
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae, ck["config"]


@torch.no_grad()
def encode_all(ae, loader, device):
    zs = [ae.encode(x.to(device)).cpu() for x in loader]
    return torch.cat(zs, 0) if zs else torch.zeros(0)


def train_from_config(cfg):
    d, t = cfg["data"], cfg["training"]
    out = Path(cfg["output"]["dir"]); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(t.get("seed", 42)); np.random.seed(t.get("seed", 42))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ae, ae_cfg = load_ae(d["ae_ckpt"], device)
    z = encode_all(ae, get_loader(d["cache_dir"], "train", t.get("batch_size", 8),
                                  shuffle=False, num_workers=t.get("num_workers", 0)), device)
    z_mean, z_std = z.mean().item(), z.std().item() + 1e-8
    z = ((z - z_mean) / z_std)
    lc, Z, X, Y = z.shape[1:]
    print(f"[ldm3d] encoded {z.shape[0]} latents ({lc},{Z},{X},{Y}) "
          f"z_mean={z_mean:.4f} z_std={z_std:.4f} device={device}")
    diff = build_diffusion3d(cfg).to(device)
    opt = torch.optim.AdamW(diff.model.parameters(), lr=t.get("learning_rate", 2e-4),
                            weight_decay=t.get("weight_decay", 0.0))
    print(f"[ldm3d] unet params {sum(p.numel() for p in diff.model.parameters()):,}")
    shutil.copy2(_dump(cfg, out), out / "config_used.yaml")
    lm = {"latent_ch": lc, "Z": Z, "X": X, "Y": Y, "z_mean": z_mean, "z_std": z_std,
          "ae_ckpt": d["ae_ckpt"], "grid": ae_cfg["data"].get("grid", [24, 60, 60]) if "data" in ae_cfg else [24, 60, 60]}
    z = z.to(device); n = z.shape[0]; bs = t.get("batch_size", 8)
    best, hist = float("inf"), []
    epochs = t.get("epochs", 400)
    for ep in range(epochs):
        tic = time.time(); perm = torch.randperm(n, device=device); tot = 0.0
        diff.train()
        for i in range(0, n, bs):
            b = z[perm[i:i + bs]]; opt.zero_grad()
            loss = diff.loss(b); loss.backward(); opt.step(); tot += loss.item() * b.shape[0]
        tot /= max(n, 1); hist.append({"epoch": ep, "loss": tot, "sec": round(time.time() - tic, 1)})
        if ep % max(1, epochs // 20) == 0 or ep == epochs - 1:
            print(f"[ldm3d] ep {ep:4d} | loss {tot:.5f} | {time.time()-tic:.1f}s")
        ck = {"epoch": ep, "model_state": diff.state_dict(), "config": cfg,
              "latent_meta": lm, "ae_config": ae_cfg}
        torch.save(ck, out / "last_ldm.pt")
        if tot < best:
            best = tot; torch.save(ck, out / "best_ldm.pt")
        (out / "history.json").write_text(json.dumps(hist, indent=2))
    print(f"[ldm3d] done. best {best:.5f}. in {out}/")
    return {"best_loss": best}


def _dump(cfg, out):
    p = out / "_effective_config.yaml"; p.write_text(yaml.safe_dump(cfg, sort_keys=False)); return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    a = ap.parse_args(); train_from_config(yaml.safe_load(Path(a.config).read_text()))
