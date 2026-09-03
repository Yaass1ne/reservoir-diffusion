"""Phase 2 gate -- sample the diffusion prior and check it against the data.

    python src/diffusion/sample_prior.py --ckpt outputs/diffusion/ldm2d/best_ldm.pt \
        --cache outputs/diffusion/cache2d --n 8

Loads the diffusion checkpoint (which references its autoencoder), draws N latent
samples, decodes them to fields, denormalizes for display, and writes:
    prior_samples.png     a grid of generated fields per output field
    prior_stats.json      generated vs real (train) per-field mean/std, to confirm
                          the prior matches the training distribution (the gate).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from autoencoder2d import build_autoencoder  # noqa: E402
from ldm2d import build_diffusion  # noqa: E402
from data2d import Slices2D  # noqa: E402
from normalization import FieldNormalizer  # noqa: E402


def load_stack(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    diff = build_diffusion(ck["config"]).to(device)
    diff.load_state_dict(ck["model_state"])
    diff.eval()
    lm = ck["latent_meta"]
    ae = build_autoencoder(ck["ae_config"]).to(device)
    ae.load_state_dict(torch.load(lm["ae_ckpt"], map_location=device)["model_state"])
    ae.eval()
    return diff, ae, lm, ck["config"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True, help="cache dir (for real-stat comparison + normalizer)")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    diff, ae, lm, cfg = load_stack(args.ckpt, device)
    out_dir = Path(args.out) if args.out else Path(args.ckpt).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    fields = Slices2D(args.cache, "train").fields
    normalizer = FieldNormalizer.load(Path(args.cache) / "normalizer.json")

    # sample latents -> denormalize latent std -> decode
    with torch.no_grad():
        z = diff.sample(args.n, (lm["latent_ch"], lm["H"], lm["W"]), device)
        z = z * lm["z_std"] + lm["z_mean"]
        gen = ae.decode(z).cpu().numpy()          # (n, C, 60, 60) normalized

    # real train slices for the distribution check
    real = Slices2D(args.cache, "train").X.numpy()  # (M, C, 60, 60) normalized

    # ---- stats gate (normalized space) ----
    stats = {}
    for ci, fld in enumerate(fields):
        g, r = gen[:, ci], real[:, ci]
        stats[fld] = {
            "gen_mean": float(g.mean()), "real_mean": float(r.mean()),
            "gen_std": float(g.std()), "real_std": float(r.std()),
        }
    (out_dir / "prior_stats.json").write_text(json.dumps(stats, indent=2))
    print("[prior] normalized-space stats (generated vs real train):")
    for fld, s in stats.items():
        print(f"  {fld}: mean {s['gen_mean']:+.3f} vs {s['real_mean']:+.3f} | "
              f"std {s['gen_std']:.3f} vs {s['real_std']:.3f}")

    # ---- figure: generated fields, denormalized to physical units ----
    ncol = args.n
    fig, axes = plt.subplots(len(fields), ncol, figsize=(1.7 * ncol, 1.9 * len(fields)),
                             squeeze=False)
    for ci, fld in enumerate(fields):
        phys = np.stack([normalizer.denormalize(fld, gen[k, ci]) for k in range(ncol)])
        vmin, vmax = np.percentile(phys, 2), np.percentile(phys, 98)
        for k in range(ncol):
            ax = axes[ci][k]
            ax.imshow(phys[k], cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
            ax.set_xticks([]); ax.set_yticks([])
            if k == 0:
                ax.set_ylabel(fld, fontsize=11)
            if ci == 0:
                ax.set_title(f"sample {k + 1}", fontsize=8)
    fig.suptitle("Diffusion prior -- unconditional samples (decoded)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "prior_samples.png", dpi=180)
    print(f"[prior] wrote {out_dir/'prior_samples.png'} and prior_stats.json")


if __name__ == "__main__":
    main()
