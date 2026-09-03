"""Phase 5 (2D) -- the validation figure from a REAL posterior.

This is the data-driven version of Figure 4 in the report: it shows that the
kept maps match the sensor data at every location and over time, produced from
the actual assimilation output rather than a synthetic stand-in.

    python src/diffusion/validate_assim.py --run outputs/diffusion/assim \
        --cache outputs/diffusion/cache2d

Reads assim_results.npz + assim_meta.json and writes:
    assim_validation.png   top: truth, posterior mean, posterior spread (uncertainty);
                           bottom: per observable/sensor, observed dots vs prior (grey)
                           and posterior (coloured) predictions over time.
    (also prints the obs-misfit reduction, the headline number.)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from normalization import FieldNormalizer  # noqa: E402

GREY = "#b9c1cc"
SEL = ["#1f4e79", "#c07e23", "#2f9e8f", "#7d3ac1"]
INK = "#1a2230"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run = Path(args.run)
    z = np.load(run / "assim_results.npz")
    meta = json.loads((run / "assim_meta.json").read_text())
    obs_fields, times, xy = meta["obs_fields"], meta["times"], meta["sensors_xy"]
    n_obs, n_times, n_sen = int(z["n_obs"]), int(z["n_times"]), int(z["n_sen"])
    normalizer = FieldNormalizer.load(Path(args.cache) / "normalizer.json")

    truth = normalizer.denormalize("Perm", z["truth_perm"])
    post = np.stack([normalizer.denormalize("Perm", p) for p in z["post_perm"]])
    post_mean = post.mean(0)
    post_std = z["post_perm"].std(0)  # spread in normalized space (uncertainty)

    prior_read = z["prior_read"].reshape(-1, n_obs, n_times, n_sen)
    post_read = z["post_read"].reshape(-1, n_obs, n_times, n_sen)
    d_obs = z["d_obs"].reshape(n_obs, n_times, n_sen)

    # choose up to 3 sensors to show as time-series columns
    show_sen = list(range(min(3, n_sen)))
    ncol = len(show_sen)

    fig = plt.figure(figsize=(3.6 * max(ncol, 3), 7.4))
    gs = GridSpec(2, max(ncol, 3), figure=fig, height_ratios=[1.05, 1.0],
                  hspace=0.42, wspace=0.22)

    # ---- top row: truth, posterior mean, spread ----
    vmin, vmax = np.percentile(truth, 2), np.percentile(truth, 98)
    top = [("True permeability", truth, "viridis", vmin, vmax),
           ("Posterior mean", post_mean, "viridis", vmin, vmax),
           ("Posterior spread (uncertainty)", post_std, "magma", None, None)]
    for j, (title, img, cmap, lo, hi) in enumerate(top):
        ax = fig.add_subplot(gs[0, j])
        im = ax.imshow(img, cmap=cmap, origin="lower",
                       vmin=lo, vmax=hi) if lo is not None else \
            ax.imshow(img, cmap=cmap, origin="lower")
        for (sx, sy) in [xy[s] for s in show_sen]:
            ax.scatter([sx], [sy], s=40, c="white", edgecolors="black", linewidths=0.8, zorder=5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=10.5, color=INK, weight="bold", pad=5)

    # ---- bottom row: observed vs predicted over time, per sensor ----
    obs_c = 0  # pressure usually carries the signal; obs_fields[0] by default
    # prefer 'pressure' if present (sat is often ~0 at sensors)
    if "pressure" in obs_fields:
        obs_c = obs_fields.index("pressure")
    fld = obs_fields[obs_c]
    for c, s in enumerate(show_sen):
        ax = fig.add_subplot(gs[1, c])
        for m in range(prior_read.shape[0]):
            ax.plot(times, prior_read[m, obs_c, :, s], color=GREY, lw=0.7, alpha=0.5, zorder=1)
        col = SEL[c % len(SEL)]
        for m in range(post_read.shape[0]):
            ax.plot(times, post_read[m, obs_c, :, s], color=col, lw=1.0, alpha=0.55, zorder=2)
        ax.plot(times, d_obs[obs_c, :, s], "o", color="black", ms=5.5, zorder=4)
        ax.set_title(f"Sensor {c+1} ({fld})", fontsize=10.5, color=INK, weight="bold", pad=4)
        ax.set_xlabel("time (timestep)", fontsize=8.8, color="#5d6b7a")
        if c == 0:
            ax.set_ylabel(f"{fld} (normalized)", fontsize=8.8, color="#5d6b7a")
        ax.tick_params(labelsize=7.5, colors="#5d6b7a")
        for sp in ax.spines.values():
            sp.set_edgecolor("#cfd8e2")

    handles = [Line2D([0], [0], color=GREY, lw=1.6, label="prior ensemble"),
               Line2D([0], [0], color=SEL[0], lw=2.0, label="posterior (conditioned)"),
               Line2D([0], [0], marker="o", color="black", lw=0, ms=6, label="observed data")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, 0.005))

    mis = z["misfit_hist"]
    fig.suptitle("Assimilation result: the posterior maps match the sensor data",
                 fontsize=14, color=INK, weight="bold", y=0.985)
    fig.text(0.5, 0.945,
             f"Observation misfit {mis[0]:.3f} to {mis[-1]:.3f} "
             f"({round(100*(1-mis[-1]/max(mis[0],1e-9)))}% lower after conditioning). "
             f"Posterior (coloured) moves onto the observed readings (black); prior (grey) misses.",
             ha="center", va="top", fontsize=8.8, color="#5d6b7a")
    fig.subplots_adjust(left=0.06, right=0.98, top=0.9, bottom=0.11)

    out = Path(args.out) if args.out else run / "assim_validation.png"
    fig.savefig(out, dpi=190, facecolor="white")
    print(f"[validate] wrote {out}")
    print(f"[validate] obs misfit {mis[0]:.4f} -> {mis[-1]:.4f}")


if __name__ == "__main__":
    main()
