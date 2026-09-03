"""Phase 6 (0) -- 3D data cache for the diffusion prior.

Same idea as data2d.py but the training unit is the FULL permeability VOLUME
(Z, X, Y) = (24, 60, 60), one per sample, instead of 2D slices. 780 volumes.

Reuses the project FieldNormalizer (log10 Perm then standardize), train-split
stats only, and the self-contained split from data2d.

    python src/diffusion/data3d.py --config configs/diffusion/cache3d.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalization import FieldNormalizer, safe_log10  # noqa: E402
from field_groups import FIELD_GROUP  # noqa: E402
from data2d import make_splits  # noqa: E402  (self-contained split)


def build_cache(cfg: dict) -> Path:
    d = cfg["data"]
    data_path = d["data_path"]
    fields = list(d.get("fields", ["Perm"]))
    log_fields = set(d.get("log_fields", ["Perm"]))
    group_of = {f: FIELD_GROUP[f] for f in fields}
    out_dir = Path(cfg["output"]["dir"]); out_dir.mkdir(parents=True, exist_ok=True)

    a = fields[0]
    with h5py.File(data_path, "r") as f:
        n_file = f[group_of[a]][a].shape[0]
    n_total = min(n_file, d.get("max_samples") or n_file)
    splits = make_splits(n_total, d.get("val_ratio", 0.1), d.get("test_ratio", 0.1),
                         d.get("seed", 42))
    print(f"[cache3d] samples total={n_total} train={len(splits['train'])} "
          f"val={len(splits['val'])} test={len(splits['test'])} fields={fields}")

    # stats from train volumes (timestep 0, full Z,X,Y)
    acc = {fl: {"s": 0.0, "ss": 0.0, "n": 0} for fl in fields}
    with h5py.File(data_path, "r") as f:
        for s in splits["train"]:
            for fl in fields:
                arr = np.asarray(f[group_of[fl]][fl][int(s), 0, 0], dtype=np.float64)  # (Z,X,Y)
                if fl in log_fields:
                    arr = safe_log10(arr)
                acc[fl]["s"] += arr.sum(); acc[fl]["ss"] += (arr ** 2).sum(); acc[fl]["n"] += arr.size
    stats = {}
    for fl in fields:
        n = max(acc[fl]["n"], 1); mean = acc[fl]["s"] / n
        stats[fl] = {"log": fl in log_fields, "mean": float(mean),
                     "std": float(np.sqrt(max(acc[fl]["ss"] / n - mean ** 2, 0.0)))}
    norm = FieldNormalizer(stats); norm.save(out_dir / "normalizer.json")
    print("[cache3d] stats: " + " | ".join(
        f"{k}: mean={v['mean']:.3f} std={v['std']:.3f}" for k, v in stats.items()))

    def collect(split):
        rows, sids = [], []
        with h5py.File(data_path, "r") as f:
            for s in splits[split]:
                chans = [norm.normalize(fl, np.asarray(f[group_of[fl]][fl][int(s), 0, 0])[None])[0]
                         for fl in fields]  # each (Z,X,Y)
                rows.append(np.stack(chans, 0)); sids.append(int(s))
        if not rows:
            return np.zeros((0, len(fields), 1, 1, 1), np.float32), np.zeros((0,), np.int64)
        return np.asarray(rows, np.float32), np.asarray(sids, np.int64)

    arrays = {}
    for split in ("train", "val", "test"):
        X, sid = collect(split)
        arrays[f"X_{split}"] = X; arrays[f"sid_{split}"] = sid
        print(f"[cache3d] {split}: {X.shape[0]} volumes, tensor {X.shape}")
    (out_dir / "cache_meta.json").write_text(json.dumps(
        {"fields": fields, "grid": list(arrays["X_train"].shape[-3:])}, indent=2))
    np.savez_compressed(out_dir / "volumes3d.npz", **arrays)
    print(f"[cache3d] wrote {out_dir/'volumes3d.npz'} + normalizer.json")
    return out_dir


class Volumes3D(Dataset):
    def __init__(self, cache_dir, split="train"):
        z = np.load(Path(cache_dir) / "volumes3d.npz")
        self.X = torch.from_numpy(z[f"X_{split}"].astype(np.float32))  # (N,C,Z,X,Y)
        self.sid = z[f"sid_{split}"]                                    # original sample indices
        meta = json.loads((Path(cache_dir) / "cache_meta.json").read_text())
        self.fields = meta["fields"]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i]


def get_loader(cache_dir, split="train", batch_size=2, shuffle=None, num_workers=0):
    ds = Volumes3D(cache_dir, split)
    if shuffle is None:
        shuffle = (split == "train")
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    a = ap.parse_args(); build_cache(yaml.safe_load(Path(a.config).read_text()))
