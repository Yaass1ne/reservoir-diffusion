"""Phase 6 (4b) -- 3D ES-MDA assimilation.

Same twin experiment as the 2D version, on full Perm volumes. Now the sensors are
real 3D points (z, x, y) at two depths, and the forward map is 3D, so matching the
readings should carry real information about the permeability volume.

    python src/diffusion/assimilate3d.py --config configs/diffusion/assim3d.yaml
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import h5py, numpy as np, torch, yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from autoencoder3d import build_autoencoder3d  # noqa: E402
from ldm3d import build_diffusion3d  # noqa: E402
from forward3d import build_forward3d  # noqa: E402
from data3d import Volumes3D  # noqa: E402
from measurement import load_sensors  # noqa: E402
from normalization import FieldNormalizer  # noqa: E402
from field_groups import FIELD_GROUP  # noqa: E402


def load_prior(ckpt, device):
    ck = torch.load(ckpt, map_location=device)
    diff = build_diffusion3d(ck["config"]).to(device); diff.load_state_dict(ck["model_state"]); diff.eval()
    lm = ck["latent_meta"]
    ae = build_autoencoder3d(ck["ae_config"]).to(device)
    ae.load_state_dict(torch.load(lm["ae_ckpt"], map_location=device)["model_state"]); ae.eval()
    return diff, ae, lm


def load_forward(ckpt, device):
    ck = torch.load(ckpt, map_location=device); meta = ck["meta"]
    net = build_forward3d(ck["config"], len(meta["obs_fields"]), len(meta["times"]),
                          meta.get("in_ch", 1 + len(meta.get("control_fields", [])))).to(device)
    net.load_state_dict(ck["model_state"]); net.eval()
    return net, meta


def load_controls(data_path, ctrl_fields, ctrl_norm_path, sample_idx, device):
    """The known injection controls for one sample -> (1, n_ctrl, Z, X, Y) normalized."""
    cn = FieldNormalizer.load(ctrl_norm_path)
    chans = []
    with h5py.File(data_path, "r") as f:
        for cf in ctrl_fields:
            cv = np.asarray(f[FIELD_GROUP[cf]][cf][int(sample_idx), 0, 0], np.float64)  # (Z,X,Y)
            chans.append(cn.normalize(cf, cv[None])[0])
    arr = np.stack(chans, 0)[None].astype(np.float32)                                    # (1,n_ctrl,Z,X,Y)
    return torch.from_numpy(arr).to(device)


def sensor_idx(sensors, device):
    zs = torch.tensor([z for (z, x, y) in sensors.locations], device=device)
    xs = torch.tensor([x for (z, x, y) in sensors.locations], device=device)
    ys = torch.tensor([y for (z, x, y) in sensors.locations], device=device)
    return zs, xs, ys


def readings(net, ae, lm, zstd, idx, ctrl, device):
    zs, xs, ys = idx
    with torch.no_grad():
        perm = ae.decode(zstd * lm["z_std"] + lm["z_mean"], grid=tuple(lm["grid"]))  # (Ne,1,Z,X,Y)
        Ne = perm.shape[0]
        inp = torch.cat([perm, ctrl.expand(Ne, -1, -1, -1, -1)], dim=1)              # (Ne,1+n_ctrl,Z,X,Y)
        obs = net(inp)                                     # (Ne,n_obs,n_times,Z,X,Y)
        read = obs[..., zs, xs, ys]                        # (Ne,n_obs,n_times,n_sen)
    return read, perm


def loc_mask(sensors, Zl, Xl, Yl, grid, L, device):
    gz, gx, gy = grid
    zz, xx, yy = np.mgrid[0:Zl, 0:Xl, 0:Yl]
    m = np.zeros((Zl, Xl, Yl))
    for (z, x, y) in sensors.locations:
        cz, cx, cy = z * Zl / gz, x * Xl / gx, y * Yl / gy
        m = np.maximum(m, np.exp(-(((zz - cz) ** 2 + (xx - cx) ** 2 + (yy - cy) ** 2) / (2 * L * L))))
    return torch.tensor(m, dtype=torch.float32, device=device)


def esmda(cfg):
    d = cfg["assim"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(d.get("seed", 0)); np.random.seed(d.get("seed", 0))
    diff, ae, lm = load_prior(d["ldm_ckpt"], device)
    net, meta = load_forward(d["forward_ckpt"], device)
    sensors = load_sensors(d["sensors"]); idx = sensor_idx(sensors, device)
    n_obs, n_times, n_sen = len(meta["obs_fields"]), len(meta["times"]), len(sensors.locations)
    Ne, Na = d.get("ensemble", 60), d.get("iterations", 4)
    obs_std = float(d.get("obs_noise_std", 0.05)); localize = bool(d.get("localize", False))
    grid = tuple(lm["grid"])
    print(f"[esmda3d] device={device} Ne={Ne} iters={Na} sensors={n_sen} localize={localize} "
          f"grid={grid} -> Nd={n_obs*n_times*n_sen}")

    pool = Volumes3D(d["cache_dir"], d.get("truth_split", "test"))
    if len(pool) == 0:
        pool = Volumes3D(d["cache_dir"], "val")
    ti = d.get("truth_index", 0) % len(pool)
    truth = pool.X[ti:ti + 1].to(device)                   # (1,1,Z,X,Y)
    truth_sid = int(pool.sid[ti])                          # original sample index -> its controls

    # the KNOWN injection controls for the truth case (fixed for every ensemble member)
    ctrl_fields = meta.get("control_fields", [])
    ctrl_norm_path = Path(d["forward_ckpt"]).parent / "ctrl_normalizer.json"
    ctrl = load_controls(d["data_path"], ctrl_fields, ctrl_norm_path, truth_sid, device) \
        if ctrl_fields else torch.zeros(1, 0, *truth.shape[-3:], device=device)
    print(f"[esmda3d] truth sample={truth_sid} | controls={ctrl_fields}")

    zs, xs, ys = idx
    with torch.no_grad():
        obs_full = net(torch.cat([truth, ctrl], dim=1))
        d_true = obs_full[..., zs, xs, ys]
    d_obs = d_true.reshape(-1) + obs_std * torch.randn(d_true.numel(), device=device)
    Nd = d_obs.numel(); Cd = (obs_std ** 2) * torch.ones(Nd, device=device)

    with torch.no_grad():
        z = diff.sample(Ne, (lm["latent_ch"], lm["Z"], lm["X"], lm["Y"]), device)
    read0, perm0 = readings(net, ae, lm, z, idx, ctrl, device)
    prior_perm = perm0.detach().cpu().numpy(); prior_read = read0.reshape(Ne, -1).cpu().numpy()

    def rmse_obs(r):
        return float(np.sqrt(((r.reshape(Ne, -1) - d_obs.cpu().numpy()) ** 2).mean()))
    mis = [rmse_obs(prior_read)]

    alpha = float(Na); zf = z.reshape(Ne, -1)
    lmask = loc_mask(sensors, lm["Z"], lm["X"], lm["Y"], grid, d.get("localize_len", 2.0), device) \
        if localize else None
    for it in range(Na):
        read, _ = readings(net, ae, lm, zf.reshape(Ne, lm["latent_ch"], lm["Z"], lm["X"], lm["Y"]), idx, ctrl, device)
        dmat = read.reshape(Ne, -1)
        d_pert = d_obs[None] + torch.sqrt(torch.tensor(alpha, device=device)) * torch.sqrt(Cd)[None] \
            * torch.randn(Ne, Nd, device=device)
        Za = zf - zf.mean(0, keepdim=True); Da = dmat - dmat.mean(0, keepdim=True)
        C_zd = (Za.T @ Da) / (Ne - 1); C_dd = (Da.T @ Da) / (Ne - 1)
        K = C_zd @ torch.linalg.pinv(C_dd + alpha * torch.diag(Cd))
        delta = (K @ (d_pert - dmat).T).T
        if lmask is not None:
            delta = (delta.reshape(Ne, lm["latent_ch"], lm["Z"], lm["X"], lm["Y"]) * lmask).reshape(Ne, -1)
        zf = zf + delta
        r = rmse_obs(dmat.detach().cpu().numpy()); mis.append(r)
        print(f"[esmda3d] iter {it+1}/{Na} | obs RMSE {r:.4f}")

    z_post = zf.reshape(Ne, lm["latent_ch"], lm["Z"], lm["X"], lm["Y"])
    read_post, perm_post = readings(net, ae, lm, z_post, idx, ctrl, device)
    post_perm = perm_post.detach().cpu().numpy(); post_read = read_post.reshape(Ne, -1).cpu().numpy()
    mis.append(rmse_obs(post_read))

    tp = truth.cpu().numpy()[0, 0]                          # (Z,X,Y)
    prior_mean = prior_perm.mean(0)[0]; post_mean = post_perm.mean(0)[0]
    sst = float(((tp - tp.mean()) ** 2).sum())
    r2 = lambda mf: float(1 - ((mf - tp) ** 2).sum() / max(sst, 1e-9))
    R = int(d.get("near_radius", 6))
    zz, xx, yy = np.mgrid[0:tp.shape[0], 0:tp.shape[1], 0:tp.shape[2]]
    near = np.zeros(tp.shape, bool)
    for (z0, x0, y0) in sensors.locations:
        near |= ((zz - z0) ** 2 + (xx - x0) ** 2 + (yy - y0) ** 2) <= R * R
    nrmse = lambda mf: float(np.sqrt(((mf - tp) ** 2)[near].mean()))
    lo = post_perm[:, 0].min(0); hi = post_perm[:, 0].max(0)
    coverage = float(((tp >= lo) & (tp <= hi)).mean())

    metrics = {
        "obs_rmse_prior": mis[0], "obs_rmse_posterior": mis[-1],
        "obs_rmse_reduction_pct": round(100 * (1 - mis[-1] / max(mis[0], 1e-9)), 1),
        "perm_rmse_prior_mean": float(np.sqrt(((prior_mean - tp) ** 2).mean())),
        "perm_rmse_posterior_mean": float(np.sqrt(((post_mean - tp) ** 2).mean())),
        "perm_r2_prior_mean": r2(prior_mean), "perm_r2_posterior_mean": r2(post_mean),
        "perm_near_rmse_prior": nrmse(prior_mean), "perm_near_rmse_posterior": nrmse(post_mean),
        "coverage_fraction": coverage, "n_sensors": n_sen, "localized": localize,
        "ensemble": Ne, "iterations": Na, "Nd": int(Nd), "dim": "3D",
        "obs_fields": meta["obs_fields"], "times": meta["times"],
    }
    out = Path(cfg["output"]["dir"]); out.mkdir(parents=True, exist_ok=True)
    # save mid-depth slices for the figure (keeps the npz small)
    zc = tp.shape[0] // 2
    np.savez_compressed(out / "assim_results.npz",
                        truth_slice=tp[zc], prior_slice=prior_perm[:, 0, zc], post_slice=post_perm[:, 0, zc],
                        prior_read=prior_read, post_read=post_read, d_obs=d_obs.cpu().numpy(),
                        misfit_hist=np.array(mis), n_obs=n_obs, n_times=n_times, n_sen=n_sen, z_center=zc)
    (out / "assim_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[esmda3d] obs RMSE {mis[0]:.4f} -> {mis[-1]:.4f} ({metrics['obs_rmse_reduction_pct']}% lower)")
    print(f"[esmda3d] perm R2 (mean): {metrics['perm_r2_prior_mean']:.3f} -> "
          f"{metrics['perm_r2_posterior_mean']:.3f} | near-sensor RMSE "
          f"{metrics['perm_near_rmse_prior']:.3f} -> {metrics['perm_near_rmse_posterior']:.3f} "
          f"| coverage {coverage:.2f}")
    print(f"[esmda3d] wrote {out/'assim_metrics.json'}")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    a = ap.parse_args(); esmda(yaml.safe_load(Path(a.config).read_text()))
