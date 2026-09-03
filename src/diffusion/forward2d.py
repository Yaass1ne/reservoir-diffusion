"""Phase 4a -- the forward surrogate: Perm slice -> sat/pressure at sensor times.

The assimilation loop (Phase 4b) needs the FORWARD map (rock -> fluids): given a
candidate permeability field, what would the sensors read? Our GRU-FNO runs the
other way (fluids -> rock), so we train a small dedicated forward surrogate here.
This is the GRU-FNO repo's native direction (it was built perm -> sat).

2D proof: input a Perm slice (1, 60, 60); predict saturation and pressure on that
slice at the recording times (2, n_times, 60, 60). The measurement operator then
samples the sensor (x, y) points. sat is a sparse plume so it fits worse than
pressure -- expected, and consistent with the Phase 3 finding. For the twin
experiment in Phase 4b the surrogate only has to be a fixed function; its physical
accuracy is reported but does not affect whether the assimilation machinery works.

Builds its own train/val cache (Perm normalized with the diffusion cache's stats;
sat/pressure standardized with their own train stats) then trains the net.

    python src/diffusion/forward2d.py --config configs/diffusion/forward2d.yaml
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from normalization import FieldNormalizer, safe_log10  # noqa: E402
from field_groups import FIELD_GROUP  # noqa: E402
from data2d import make_splits, resolve_z_slices  # noqa: E402


# ----------------------------- data -----------------------------
def build_forward_cache(cfg: dict) -> Path:
    d = cfg["data"]
    data_path = d["data_path"]
    perm_norm = FieldNormalizer.load(Path(d["perm_normalizer"]))  # reuse Perm stats
    obs_fields = list(d.get("obs_fields", ["sat", "pressure"]))
    times = list(d["times"])
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(data_path, "r") as f:
        n_file, n_t, _, n_z, n_x, n_y = f["inputs"]["Perm"].shape
    n_total = min(n_file, d.get("max_samples") or n_file)
    z_slices = resolve_z_slices(d.get("z_slices", "all"), n_z)
    splits = make_splits(n_total, d.get("val_ratio", 0.1), d.get("test_ratio", 0.1),
                         d.get("seed", 42))

    # obs stats from train slices (standardize sat & pressure)
    acc = {fl: {"s": 0.0, "ss": 0.0, "n": 0} for fl in obs_fields}
    with h5py.File(data_path, "r") as f:
        for s in splits["train"]:
            for fl in obs_fields:
                arr = np.asarray(f[FIELD_GROUP[fl]][fl][int(s), times, 0][:, z_slices],
                                 dtype=np.float64)  # (n_times, len(z), X, Y)
                acc[fl]["s"] += arr.sum(); acc[fl]["ss"] += (arr ** 2).sum()
                acc[fl]["n"] += arr.size
    obs_stats = {}
    for fl in obs_fields:
        n = max(acc[fl]["n"], 1); mean = acc[fl]["s"] / n
        obs_stats[fl] = {"log": False, "mean": float(mean),
                         "std": float(np.sqrt(max(acc[fl]["ss"] / n - mean ** 2, 0.0)))}
    obs_norm = FieldNormalizer(obs_stats)
    obs_norm.save(out_dir / "obs_normalizer.json")

    def collect(split):
        Xp, Yo = [], []
        with h5py.File(data_path, "r") as f:
            for s in splits[split]:
                perm = np.asarray(f["inputs"]["Perm"][int(s), 0, 0], dtype=np.float64)  # (Z,X,Y)
                obs = {fl: np.asarray(f[FIELD_GROUP[fl]][fl][int(s), times, 0], dtype=np.float64)
                       for fl in obs_fields}  # (n_times, Z, X, Y)
                for z in z_slices:
                    Xp.append(perm_norm.normalize("Perm", perm[z][None])[0][None])  # (1,X,Y)
                    chans = [obs_norm.normalize(fl, obs[fl][:, z]) for fl in obs_fields]  # each (nt,X,Y)
                    Yo.append(np.stack(chans, axis=0))  # (n_obs, nt, X, Y)
        if not Xp:
            return np.zeros((0, 1, 1, 1), np.float32), np.zeros((0, 1, 1, 1, 1), np.float32)
        return np.asarray(Xp, np.float32), np.asarray(Yo, np.float32)

    arrays = {}
    for split in ("train", "val"):
        Xp, Yo = collect(split)
        arrays[f"X_{split}"], arrays[f"Y_{split}"] = Xp, Yo
        print(f"[fwd-cache] {split}: perm {Xp.shape} -> obs {Yo.shape}")
    np.savez_compressed(out_dir / "forward_slices.npz", **arrays)
    (out_dir / "forward_meta.json").write_text(json.dumps(
        {"obs_fields": obs_fields, "times": times, "z_slices": z_slices}, indent=2))
    print(f"[fwd-cache] wrote {out_dir/'forward_slices.npz'} + obs_normalizer.json")
    return out_dir


# ----------------------------- model -----------------------------
class ForwardNet(nn.Module):
    """Perm (1,60,60) -> observables (n_obs, n_times, 60,60). Small U-Net."""

    def __init__(self, n_obs=2, n_times=7, base=32):
        super().__init__()
        self.n_obs, self.n_times = n_obs, n_times
        out_ch = n_obs * n_times
        self.e1 = self._blk(1, base)
        self.e2 = self._blk(base, base * 2)
        self.e3 = self._blk(base * 2, base * 4)
        self.pool = nn.AvgPool2d(2)
        self.d2 = self._blk(base * 4 + base * 2, base * 2)
        self.d1 = self._blk(base * 2 + base, base)
        self.out = nn.Conv2d(base, out_ch, 1)

    @staticmethod
    def _blk(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(min(8, cout), cout), nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(min(8, cout), cout), nn.SiLU())

    def forward(self, x):
        h1 = self.e1(x)                       # 60
        h2 = self.e2(self.pool(h1))           # 30
        h3 = self.e3(self.pool(h2))           # 15
        u2 = F.interpolate(h3, size=h2.shape[-2:], mode="nearest")
        u2 = self.d2(torch.cat([u2, h2], 1))
        u1 = F.interpolate(u2, size=h1.shape[-2:], mode="nearest")
        u1 = self.d1(torch.cat([u1, h1], 1))
        y = self.out(u1)                      # (B, n_obs*n_times, 60,60)
        return y.view(y.shape[0], self.n_obs, self.n_times, *y.shape[-2:])


def build_forward(cfg, n_obs, n_times):
    m = cfg["model"]
    return ForwardNet(n_obs=n_obs, n_times=n_times, base=m.get("base", 32))


# ----------------------------- train -----------------------------
def train_from_config(cfg: dict):
    d, t = cfg["data"], cfg["training"]
    cache = build_forward_cache(cfg)
    z = np.load(cache / "forward_slices.npz")
    meta = json.loads((cache / "forward_meta.json").read_text())
    n_obs, n_times = len(meta["obs_fields"]), len(meta["times"])
    Xtr = torch.from_numpy(z["X_train"]); Ytr = torch.from_numpy(z["Y_train"])
    Xva = torch.from_numpy(z["X_val"]); Yva = torch.from_numpy(z["Y_val"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = build_forward(cfg, n_obs, n_times).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=t.get("learning_rate", 1e-3),
                            weight_decay=t.get("weight_decay", 1e-5))
    out_dir = cache
    shutil.copy2(_dump_cfg(cfg, out_dir), out_dir / "config_used.yaml")
    print(f"[fwd] device={device} obs={meta['obs_fields']} times={n_times} "
          f"train={Xtr.shape[0]} val={Xva.shape[0]} params={sum(p.numel() for p in net.parameters()):,}")

    bs = t.get("batch_size", 32); n = Xtr.shape[0]
    Xtr, Ytr = Xtr.to(device), Ytr.to(device)
    Xva, Yva = Xva.to(device), Yva.to(device)
    best, history = float("inf"), []
    for ep in range(t.get("epochs", 60)):
        net.train(); perm = torch.randperm(n, device=device); tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = F.mse_loss(net(Xtr[idx]), Ytr[idx])
            loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        tot /= max(n, 1)
        net.eval()
        with torch.no_grad():
            pv = net(Xva)
            va = F.mse_loss(pv, Yva).item()
            # per-observable R2 in normalized space
            r2 = {}
            for c, fl in enumerate(meta["obs_fields"]):
                p, g = pv[:, c], Yva[:, c]
                sse = ((p - g) ** 2).sum().item()
                sst = ((g - g.mean()) ** 2).sum().item()
                r2[fl] = 1.0 - sse / max(sst, 1e-12)
        history.append({"epoch": ep, "train": tot, "val": va, "val_r2": r2})
        if ep % max(1, t.get("epochs", 60) // 12) == 0 or ep == t.get("epochs", 60) - 1:
            print(f"[fwd] epoch {ep:3d} | train {tot:.4f} | val {va:.4f} | "
                  f"R2 {{{', '.join(f'{k}:{v:.3f}' for k, v in r2.items())}}}")
        ck = {"model_state": net.state_dict(), "config": cfg, "meta": meta, "val_r2": r2}
        torch.save(ck, out_dir / "last_forward.pt")
        if va < best:
            best = va; torch.save(ck, out_dir / "best_forward.pt")
        (out_dir / "forward_history.json").write_text(json.dumps(history, indent=2))
    print(f"[fwd] done. best val {best:.4f}. artifacts in {out_dir}/")
    return {"best_val": best, "val_r2": history[-1]["val_r2"]}


def _dump_cfg(cfg, out_dir):
    p = out_dir / "_effective_config.yaml"; p.write_text(yaml.safe_dump(cfg, sort_keys=False)); return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    a = ap.parse_args(); train_from_config(yaml.safe_load(Path(a.config).read_text()))
