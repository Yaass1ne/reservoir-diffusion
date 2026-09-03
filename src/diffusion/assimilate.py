"""Phase 4b -- ensemble assimilation (ES-MDA) in latent space.

Conditions the diffusion prior on the sensor readings so the kept permeability
maps reproduce the data at the sensor locations and times. This is a twin
experiment: a held-out real field is the truth, its sensor readings (through the
forward surrogate + noise) are the observations, and ES-MDA updates the ensemble
of latent codes until their predicted readings match.

Pipeline per ensemble member:
    latent z  --AE.decode-->  Perm field  --forward surrogate-->  sat/pressure
    --H-->  readings at the sensors.  ES-MDA nudges z toward the observed readings.

Working in latent space (a few thousand numbers) instead of every grid cell is
what keeps the Kalman update cheap. Direct latent update is a first, standard
choice; it can be refined later.

    python src/diffusion/assimilate.py --config configs/diffusion/assim.yaml

Saves outputs/diffusion/assim/ : assim_results.npz (truth, prior & posterior perm
ensembles, prior/posterior predicted readings, observations), assim_metrics.json
(data misfit + perm error, prior vs posterior).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from autoencoder2d import build_autoencoder  # noqa: E402
from ldm2d import build_diffusion  # noqa: E402
from forward2d import build_forward  # noqa: E402
from data2d import Slices2D  # noqa: E402
from measurement import load_sensors  # noqa: E402
from normalization import FieldNormalizer  # noqa: E402


def unique_xy(sensors):
    """The 2D proof uses the unique (x, y) columns of the 3D sensor layout."""
    seen, xy = set(), []
    for (z, x, y) in sensors.locations:
        if (x, y) not in seen:
            seen.add((x, y)); xy.append((x, y))
    return xy


def build_loc_mask(xy, H, W, grid, L, device):
    """Localization taper on the latent grid: ~1 near a sensor, ->0 far away.

    Sensor (x=col, y=row) on the `grid`-sized field maps to the H x W latent grid.
    Multiplying the ES-MDA update by this mask keeps each correction local, so the
    far field stays on the realistic prior instead of collapsing to a blurry mean.
    """
    yy, xx = np.mgrid[0:H, 0:W]
    m = np.zeros((H, W))
    for (sx, sy) in xy:
        li = sy * H / grid          # row centre in latent
        lj = sx * W / grid          # col centre in latent
        m = np.maximum(m, np.exp(-(((xx - lj) ** 2 + (yy - li) ** 2) / (2 * L * L))))
    return torch.tensor(m, dtype=torch.float32, device=device)


def load_prior(ldm_ckpt, device):
    ck = torch.load(ldm_ckpt, map_location=device)
    diff = build_diffusion(ck["config"]).to(device); diff.load_state_dict(ck["model_state"]); diff.eval()
    lm = ck["latent_meta"]
    ae = build_autoencoder(ck["ae_config"]).to(device)
    ae.load_state_dict(torch.load(lm["ae_ckpt"], map_location=device)["model_state"]); ae.eval()
    return diff, ae, lm


def load_forward(fwd_ckpt, device):
    ck = torch.load(fwd_ckpt, map_location=device)
    meta = ck["meta"]
    net = build_forward(ck["config"], len(meta["obs_fields"]), len(meta["times"])).to(device)
    net.load_state_dict(ck["model_state"]); net.eval()
    return net, meta


def predict_readings(net, ae, lm, z_std_latent, xy, device):
    """Standardized latents (Ne, C, H, W) -> readings (Ne, n_obs, n_times, n_sensors)."""
    with torch.no_grad():
        z_phys = z_std_latent * lm["z_std"] + lm["z_mean"]
        perm = ae.decode(z_phys)                       # (Ne, 1, 60, 60) normalized Perm
        obs = net(perm)                                # (Ne, n_obs, n_times, 60, 60)
        rows = torch.tensor([p[1] for p in xy], device=device)  # y index -> row
        cols = torch.tensor([p[0] for p in xy], device=device)  # x index -> col
        read = obs[..., rows, cols]                    # (Ne, n_obs, n_times, n_sensors)
    return read, perm


def esmda(cfg):
    d = cfg["assim"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(d.get("seed", 0)); np.random.seed(d.get("seed", 0))

    diff, ae, lm = load_prior(d["ldm_ckpt"], device)
    net, meta = load_forward(d["forward_ckpt"], device)
    sensors = load_sensors(d["sensors"])
    xy = unique_xy(sensors)
    n_obs, n_times, n_sen = len(meta["obs_fields"]), len(meta["times"]), len(xy)
    Ne = d.get("ensemble", 60)
    Na = d.get("iterations", 4)
    obs_std = float(d.get("obs_noise_std", 0.05))
    localize = bool(d.get("localize", False))
    loc_len = float(d.get("localize_len", 3.0))
    print(f"[esmda] device={device} Ne={Ne} iters={Na} sensors(xy)={n_sen} "
          f"localize={localize} obs={meta['obs_fields']} times={n_times} "
          f"-> Nd={n_obs*n_times*n_sen}")

    # --- truth (a held-out real slice) + its observations ---
    truth_pool = Slices2D(d["cache_dir"], d.get("truth_split", "test"))
    if len(truth_pool) == 0:
        truth_pool = Slices2D(d["cache_dir"], "val")
    ti = d.get("truth_index", 0) % len(truth_pool)
    truth_perm = truth_pool.X[ti:ti + 1].to(device)             # (1,1,60,60) normalized
    gsize0 = int(truth_perm.shape[-1])                          # field grid size (60)
    with torch.no_grad():
        obs_full = net(truth_perm)
        rows = torch.tensor([p[1] for p in xy], device=device)
        cols = torch.tensor([p[0] for p in xy], device=device)
        d_true = obs_full[..., rows, cols]                      # (1,n_obs,n_times,n_sen)
    d_obs = d_true.reshape(-1)                                   # (Nd,)
    d_obs = d_obs + obs_std * torch.randn_like(d_obs)           # measurement noise
    Nd = d_obs.numel()
    Cd = (obs_std ** 2) * torch.ones(Nd, device=device)         # diagonal obs cov

    # --- prior ensemble of latents (standardized) ---
    with torch.no_grad():
        z = diff.sample(Ne, (lm["latent_ch"], lm["H"], lm["W"]), device)   # (Ne,C,H,W)
    read0, perm0 = predict_readings(net, ae, lm, z, xy, device)
    prior_perm = perm0.detach().cpu().numpy()
    prior_read = read0.reshape(Ne, -1).detach().cpu().numpy()

    zdim = z[0].numel()
    misfit_hist = []

    def rmse_to_obs(read):
        return float(np.sqrt(((read.reshape(Ne, -1) - d_obs.cpu().numpy()) ** 2).mean()))

    misfit_hist.append(rmse_to_obs(prior_read))

    # --- ES-MDA iterations (equal inflation: alpha = Na for each step) ---
    alpha = float(Na)
    zf = z.reshape(Ne, -1)                                       # (Ne, D)
    loc_mask = build_loc_mask(xy, lm["H"], lm["W"], gsize0, loc_len, device) \
        if localize else None
    for it in range(Na):
        read, _ = predict_readings(net, ae, lm, zf.reshape(Ne, lm["latent_ch"], lm["H"], lm["W"]),
                                   xy, device)
        dmat = read.reshape(Ne, -1)                              # (Ne, Nd)
        # perturbed observations
        d_pert = d_obs[None, :] + torch.sqrt(torch.tensor(alpha, device=device)) \
            * torch.sqrt(Cd)[None, :] * torch.randn(Ne, Nd, device=device)
        # anomalies
        zmean = zf.mean(0, keepdim=True); dmean = dmat.mean(0, keepdim=True)
        Za = zf - zmean; Da = dmat - dmean
        C_zd = (Za.T @ Da) / (Ne - 1)                            # (D, Nd)
        C_dd = (Da.T @ Da) / (Ne - 1)                            # (Nd, Nd)
        A = C_dd + alpha * torch.diag(Cd)
        K = C_zd @ torch.linalg.pinv(A)                          # (D, Nd)
        delta = (K @ (d_pert - dmat).T).T                        # (Ne, D)
        if loc_mask is not None:
            delta = (delta.reshape(Ne, lm["latent_ch"], lm["H"], lm["W"])
                     * loc_mask).reshape(Ne, -1)
        zf = zf + delta
        r = rmse_to_obs(dmat.detach().cpu().numpy())
        misfit_hist.append(r)
        print(f"[esmda] iter {it+1}/{Na} | obs RMSE {r:.4f}")

    # --- posterior ---
    z_post = zf.reshape(Ne, lm["latent_ch"], lm["H"], lm["W"])
    read_post, perm_post = predict_readings(net, ae, lm, z_post, xy, device)
    post_read = read_post.reshape(Ne, -1).detach().cpu().numpy()
    post_perm = perm_post.detach().cpu().numpy()
    misfit_hist.append(rmse_to_obs(post_read))

    # --- metrics ---
    tp = truth_perm.cpu().numpy()[0, 0]                          # (60,60) normalized
    prior_mean = prior_perm.mean(0)[0]                           # (60,60)
    post_mean = post_perm.mean(0)[0]

    def perm_rmse(mean_field):
        return float(np.sqrt(((mean_field - tp) ** 2).mean()))

    sst = float(((tp - tp.mean()) ** 2).sum())

    def perm_r2(mean_field):
        return float(1.0 - ((mean_field - tp) ** 2).sum() / max(sst, 1e-9))

    # near-sensor region: cells within near_radius of any sensor (row=y, col=x)
    R = int(d.get("near_radius", 8))
    yy, xx = np.mgrid[0:gsize0, 0:gsize0]
    near = np.zeros((gsize0, gsize0), bool)
    for (sx, sy) in xy:
        near |= ((xx - sx) ** 2 + (yy - sy) ** 2) <= R * R

    def near_rmse(mean_field):
        return float(np.sqrt(((mean_field - tp) ** 2)[near].mean()))

    # ensemble coverage: fraction of cells whose truth falls in the posterior band
    lo = post_perm[:, 0].min(0); hi = post_perm[:, 0].max(0)
    coverage = float(((tp >= lo) & (tp <= hi)).mean())

    metrics = {
        "obs_rmse_prior": misfit_hist[0],
        "obs_rmse_posterior": misfit_hist[-1],
        "obs_rmse_reduction_pct": round(100 * (1 - misfit_hist[-1] / max(misfit_hist[0], 1e-9)), 1),
        "perm_rmse_prior_mean": perm_rmse(prior_mean),
        "perm_rmse_posterior_mean": perm_rmse(post_mean),
        "perm_r2_prior_mean": perm_r2(prior_mean),
        "perm_r2_posterior_mean": perm_r2(post_mean),
        "perm_near_rmse_prior": near_rmse(prior_mean),
        "perm_near_rmse_posterior": near_rmse(post_mean),
        "coverage_fraction": coverage,
        "n_sensors": n_sen, "localized": localize,
        "ensemble": Ne, "iterations": Na, "Nd": int(Nd),
        "obs_fields": meta["obs_fields"], "times": meta["times"], "sensors_xy": xy,
    }
    out_dir = Path(cfg["output"]["dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "assim_results.npz",
        truth_perm=tp, prior_perm=prior_perm[:, 0], post_perm=post_perm[:, 0],
        prior_read=prior_read, post_read=post_read, d_obs=d_obs.cpu().numpy(),
        d_true=d_true.reshape(-1).cpu().numpy(),
        misfit_hist=np.array(misfit_hist),
        n_obs=n_obs, n_times=n_times, n_sen=n_sen,
    )
    (out_dir / "assim_metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "assim_meta.json").write_text(json.dumps(
        {"obs_fields": meta["obs_fields"], "times": meta["times"], "sensors_xy": xy,
         "cache_dir": d["cache_dir"]}, indent=2))
    print(f"[esmda] obs RMSE {metrics['obs_rmse_prior']:.4f} -> "
          f"{metrics['obs_rmse_posterior']:.4f} ({metrics['obs_rmse_reduction_pct']}% lower)")
    print(f"[esmda] perm R2 (mean): {metrics['perm_r2_prior_mean']:.3f} -> "
          f"{metrics['perm_r2_posterior_mean']:.3f} | near-sensor RMSE "
          f"{metrics['perm_near_rmse_prior']:.3f} -> {metrics['perm_near_rmse_posterior']:.3f} "
          f"| coverage {metrics['coverage_fraction']:.2f} | sensors={n_sen} localize={localize}")
    print(f"[esmda] wrote {out_dir/'assim_results.npz'} + assim_metrics.json")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    a = ap.parse_args(); esmda(yaml.safe_load(Path(a.config).read_text()))
