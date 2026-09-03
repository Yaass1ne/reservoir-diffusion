"""Phase 6 (5) -- 3D validation figure (mid-depth slice of the assimilation result).
    python src/diffusion/validate_assim3d.py --run outputs/diffusion3d/assim3d \
        --cache outputs/diffusion3d/cache3d
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from normalization import FieldNormalizer  # noqa: E402

INK = "#1a2230"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--cache", required=True)
    a = ap.parse_args()
    run = Path(a.run); z = np.load(run / "assim_results.npz")
    norm = FieldNormalizer.load(Path(a.cache) / "normalizer.json")
    truth = norm.denormalize("Perm", z["truth_slice"])
    post = np.stack([norm.denormalize("Perm", s) for s in z["post_slice"]])
    post_mean = post.mean(0); post_std = z["post_slice"].std(0)
    mis = z["misfit_hist"]

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
    vmin, vmax = np.percentile(truth, 2), np.percentile(truth, 98)
    for a_, img, title, cmap, lo, hi in [
        (ax[0], truth, "True permeability (mid slice)", "viridis", vmin, vmax),
        (ax[1], post_mean, "Posterior mean", "viridis", vmin, vmax),
        (ax[2], post_std, "Posterior spread (uncertainty)", "magma", None, None)]:
        (a_.imshow(img, cmap=cmap, origin="lower", vmin=lo, vmax=hi) if lo is not None
         else a_.imshow(img, cmap=cmap, origin="lower"))
        a_.set_xticks([]); a_.set_yticks([]); a_.set_title(title, fontsize=10.5, color=INK, weight="bold")
    fig.suptitle(f"3D assimilation result -- z={int(z['z_center'])}. Obs misfit "
                 f"{mis[0]:.3f} to {mis[-1]:.3f} ({round(100*(1-mis[-1]/max(mis[0],1e-9)))}% lower).",
                 fontsize=11.5, color=INK, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(run / "assim_validation.png", dpi=180)
    print(f"[validate3d] wrote {run/'assim_validation.png'} | obs {mis[0]:.4f} -> {mis[-1]:.4f}")


if __name__ == "__main__":
    main()
