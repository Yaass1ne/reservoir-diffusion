"""Phase 6 (2) gate -- sample the 3D prior, show a mid-depth slice of each volume.
    python src/diffusion/sample_prior3d.py --ckpt outputs/diffusion3d/ldm3d/best_ldm.pt \
        --cache outputs/diffusion3d/cache3d --n 8
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from autoencoder3d import build_autoencoder3d  # noqa: E402
from ldm3d import build_diffusion3d  # noqa: E402
from data3d import Volumes3D  # noqa: E402
from normalization import FieldNormalizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--cache", required=True)
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location=device)
    diff = build_diffusion3d(ck["config"]).to(device); diff.load_state_dict(ck["model_state"]); diff.eval()
    lm = ck["latent_meta"]
    ae = build_autoencoder3d(ck["ae_config"]).to(device)
    ae.load_state_dict(torch.load(lm["ae_ckpt"], map_location=device)["model_state"]); ae.eval()
    norm = FieldNormalizer.load(Path(a.cache) / "normalizer.json")
    fld = Volumes3D(a.cache, "train").fields[0]
    with torch.no_grad():
        z = diff.sample(a.n, (lm["latent_ch"], lm["Z"], lm["X"], lm["Y"]), device)
        gen = ae.decode(z * lm["z_std"] + lm["z_mean"], grid=tuple(lm["grid"])).cpu().numpy()  # (n,1,Z,X,Y)
    real = Volumes3D(a.cache, "train").X.numpy()
    stats = {fld: {"gen_mean": float(gen.mean()), "real_mean": float(real.mean()),
                   "gen_std": float(gen.std()), "real_std": float(real.std())}}
    out = Path(a.ckpt).parent
    (out / "prior_stats.json").write_text(json.dumps(stats, indent=2))
    print("[prior3d]", stats)
    zc = gen.shape[2] // 2
    phys = np.stack([norm.denormalize(fld, gen[k, 0, zc]) for k in range(a.n)])
    vmin, vmax = np.percentile(phys, 2), np.percentile(phys, 98)
    fig, axes = plt.subplots(1, a.n, figsize=(1.7 * a.n, 2.0), squeeze=False)
    for k in range(a.n):
        ax = axes[0][k]; ax.imshow(phys[k], cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"sample {k+1}", fontsize=8)
    fig.suptitle(f"3D diffusion prior -- {fld} (mid-depth slice) of unconditional samples", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.9]); fig.savefig(out / "prior_samples.png", dpi=170)
    print(f"[prior3d] wrote {out/'prior_samples.png'}")


if __name__ == "__main__":
    main()
