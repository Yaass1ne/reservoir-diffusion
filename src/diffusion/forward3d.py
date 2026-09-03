"""Phase 6 (4a) -- 3D forward surrogate, CONTROLS-AWARE.

Perm alone cannot predict the fluids: pressure especially is driven by the
injection program (Rate, Bhp, Boundary), and the dataset holds 39 different
injection scenarios. A Perm-only forward therefore fails on pressure (R2 < 0).
So the input is Perm PLUS the injection controls:

    (Perm, Rate, Bhp, Boundary)  ->  sat & pressure at the recording times

This is legitimate for inversion: in a real case the injection program is known;
we solve for the unknown rock. In assimilation the controls are fixed (the known
truth-case controls) and only Perm varies across the ensemble.

Perm is normalized with the diffusion CACHE normalizer (so decoded latents match);
the controls get their own train-set standardization; sat/pressure their own.
Controls are read at timestep 0 (they carry the injection pattern / scenario id).

    python src/diffusion/forward3d.py --config configs/diffusion/forward3d.yaml
"""
from __future__ import annotations

import argparse, json, shutil, sys, time
from pathlib import Path
import h5py, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from normalization import FieldNormalizer  # noqa: E402
from field_groups import FIELD_GROUP  # noqa: E402
from data2d import make_splits  # noqa: E402


def _field_stats(f, fields, idxs, times=None):
    acc = {fl: {"s": 0.0, "ss": 0.0, "n": 0} for fl in fields}
    for s in idxs:
        for fl in fields:
            if times is None:
                arr = np.asarray(f[FIELD_GROUP[fl]][fl][int(s), 0, 0], np.float64)          # (Z,X,Y)
            else:
                arr = np.asarray(f[FIELD_GROUP[fl]][fl][int(s), times, 0], np.float64)       # (nt,Z,X,Y)
            acc[fl]["s"] += arr.sum(); acc[fl]["ss"] += (arr ** 2).sum(); acc[fl]["n"] += arr.size
    out = {}
    for fl in fields:
        n = max(acc[fl]["n"], 1); m = acc[fl]["s"] / n
        out[fl] = {"log": False, "mean": float(m), "std": float(np.sqrt(max(acc[fl]["ss"] / n - m * m, 0.0)))}
    return out


def build_cache(cfg):
    d = cfg["data"]
    perm_norm = FieldNormalizer.load(Path(d["perm_normalizer"]))
    ctrl_fields = list(d.get("control_fields", ["Rate", "Bhp", "Boundary"]))
    obs_fields = list(d.get("obs_fields", ["sat", "pressure"]))
    times = list(d["times"]); dp = d["data_path"]
    out = Path(cfg["output"]["dir"]); out.mkdir(parents=True, exist_ok=True)
    with h5py.File(dp, "r") as f:
        n_file = f["inputs"]["Perm"].shape[0]
    n_total = min(n_file, d.get("max_samples") or n_file)
    splits = make_splits(n_total, d.get("val_ratio", 0.1), d.get("test_ratio", 0.1), d.get("seed", 42))

    with h5py.File(dp, "r") as f:
        cstats = _field_stats(f, ctrl_fields, splits["train"])            # controls (timestep 0)
        ostats = _field_stats(f, obs_fields, splits["train"], times)      # sat/pressure at times
    ctrl_norm = FieldNormalizer(cstats); ctrl_norm.save(out / "ctrl_normalizer.json")
    onorm = FieldNormalizer(ostats); onorm.save(out / "obs_normalizer.json")

    def collect(split):
        Xin, Yo = [], []
        with h5py.File(dp, "r") as f:
            for s in splits[split]:
                perm = np.asarray(f["inputs"]["Perm"][int(s), 0, 0], np.float64)
                chans = [perm_norm.normalize("Perm", perm[None])[0]]
                for cf in ctrl_fields:
                    cv = np.asarray(f[FIELD_GROUP[cf]][cf][int(s), 0, 0], np.float64)
                    chans.append(ctrl_norm.normalize(cf, cv[None])[0])
                Xin.append(np.stack(chans, 0).astype(np.float16))         # (1+n_ctrl,Z,X,Y)
                obs = [onorm.normalize(fl, np.asarray(f[FIELD_GROUP[fl]][fl][int(s), times, 0], np.float64))
                       for fl in obs_fields]
                Yo.append(np.stack(obs, 0).astype(np.float16))            # (n_obs,nt,Z,X,Y)
        if not Xin:
            return (np.zeros((0, 1, 1, 1, 1), np.float16), np.zeros((0, 1, 1, 1, 1, 1), np.float16))
        return np.asarray(Xin, np.float16), np.asarray(Yo, np.float16)

    arrays = {}
    for split in ("train", "val"):
        Xin, Yo = collect(split); arrays[f"X_{split}"], arrays[f"Y_{split}"] = Xin, Yo
        print(f"[fwd3d-cache] {split}: input {Xin.shape} -> obs {Yo.shape}")
    np.savez_compressed(out / "forward_slices.npz", **arrays)
    (out / "forward_meta.json").write_text(json.dumps(
        {"obs_fields": obs_fields, "times": times, "control_fields": ctrl_fields,
         "in_ch": 1 + len(ctrl_fields)}, indent=2))
    print(f"[fwd3d-cache] wrote forward_slices.npz + ctrl/obs normalizers (in_ch={1+len(ctrl_fields)})")
    return out


class ForwardNet3D(nn.Module):
    def __init__(self, in_ch=4, n_obs=2, n_times=7, base=16):
        super().__init__()
        self.n_obs, self.n_times = n_obs, n_times
        out_ch = n_obs * n_times
        self.e1 = self._b(in_ch, base); self.e2 = self._b(base, base * 2); self.e3 = self._b(base * 2, base * 4)
        self.pool = nn.AvgPool3d(2)
        self.d2 = self._b(base * 4 + base * 2, base * 2); self.d1 = self._b(base * 2 + base, base)
        self.out = nn.Conv3d(base, out_ch, 1)

    @staticmethod
    def _b(ci, co):
        return nn.Sequential(nn.Conv3d(ci, co, 3, padding=1), nn.GroupNorm(min(8, co), co), nn.SiLU(),
                             nn.Conv3d(co, co, 3, padding=1), nn.GroupNorm(min(8, co), co), nn.SiLU())

    def forward(self, x):
        h1 = self.e1(x); h2 = self.e2(self.pool(h1)); h3 = self.e3(self.pool(h2))
        u2 = F.interpolate(h3, size=h2.shape[-3:], mode="nearest"); u2 = self.d2(torch.cat([u2, h2], 1))
        u1 = F.interpolate(u2, size=h1.shape[-3:], mode="nearest"); u1 = self.d1(torch.cat([u1, h1], 1))
        y = self.out(u1)
        return y.view(y.shape[0], self.n_obs, self.n_times, *y.shape[-3:])


def build_forward3d(cfg, n_obs, n_times, in_ch=4):
    return ForwardNet3D(in_ch, n_obs, n_times, cfg["model"].get("base", 16))


def train_from_config(cfg):
    t = cfg["training"]; cache = build_cache(cfg)
    z = np.load(cache / "forward_slices.npz")
    meta = json.loads((cache / "forward_meta.json").read_text())
    n_obs, n_times, in_ch = len(meta["obs_fields"]), len(meta["times"]), meta["in_ch"]
    Xtr = torch.from_numpy(z["X_train"].astype(np.float32)); Ytr = torch.from_numpy(z["Y_train"].astype(np.float32))
    Xva = torch.from_numpy(z["X_val"].astype(np.float32)); Yva = torch.from_numpy(z["Y_val"].astype(np.float32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = build_forward3d(cfg, n_obs, n_times, in_ch).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=t.get("learning_rate", 1e-3), weight_decay=t.get("weight_decay", 1e-5))
    print(f"[fwd3d] device={device} in_ch={in_ch} (Perm+{meta['control_fields']}) obs={meta['obs_fields']} "
          f"times={n_times} train={Xtr.shape[0]} val={Xva.shape[0]} params={sum(p.numel() for p in net.parameters()):,}")
    shutil.copy2(_dump(cfg, cache), cache / "config_used.yaml")
    bs = t.get("batch_size", 2); n = Xtr.shape[0]; best, hist = float("inf"), []
    for ep in range(t.get("epochs", 40)):
        tic = time.time(); net.train(); perm = torch.randperm(n); tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]; xb, yb = Xtr[idx].to(device), Ytr[idx].to(device)
            opt.zero_grad(); loss = F.mse_loss(net(xb), yb); loss.backward(); opt.step(); tot += loss.item() * len(idx)
        tot /= max(n, 1)
        net.eval(); vm = 0.0
        with torch.no_grad():
            sse = {c: 0.0 for c in range(n_obs)}; sst = {c: 0.0 for c in range(n_obs)}
            for i in range(0, Xva.shape[0], bs):
                xb, yb = Xva[i:i+bs].to(device), Yva[i:i+bs].to(device)
                pv = net(xb); vm += F.mse_loss(pv, yb).item() * xb.shape[0]
                for c in range(n_obs):
                    sse[c] += ((pv[:, c] - yb[:, c]) ** 2).sum().item()
                    sst[c] += ((yb[:, c] - yb[:, c].mean()) ** 2).sum().item()
            vm /= max(Xva.shape[0], 1)
            r2 = {meta["obs_fields"][c]: 1 - sse[c] / max(sst[c], 1e-9) for c in range(n_obs)}
        hist.append({"epoch": ep, "train": tot, "val": vm, "val_r2": r2})
        print(f"[fwd3d] ep {ep:3d} | train {tot:.4f} | val {vm:.4f} | "
              f"R2 {{{', '.join(f'{k}:{v:.3f}' for k, v in r2.items())}}} | {time.time()-tic:.1f}s")
        ck = {"model_state": net.state_dict(), "config": cfg, "meta": meta, "val_r2": r2}
        torch.save(ck, cache / "last_forward.pt")
        if vm < best:
            best = vm; torch.save(ck, cache / "best_forward.pt")
        (cache / "forward_history.json").write_text(json.dumps(hist, indent=2))
    print(f"[fwd3d] done. best val {best:.4f} | final R2 {hist[-1]['val_r2']}")
    return {"best_val": best, "val_r2": hist[-1]["val_r2"]}


def _dump(cfg, out):
    p = out / "_effective_config.yaml"; p.write_text(yaml.safe_dump(cfg, sort_keys=False)); return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    a = ap.parse_args(); train_from_config(yaml.safe_load(Path(a.config).read_text()))
