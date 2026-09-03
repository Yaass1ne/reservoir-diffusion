"""Phase 0 -- 2D data cache for the diffusion prior.

The diffusion prior is trained on the STATIC rock fields (Perm, Por). Those are
constant across the 56 timesteps, so we take timestep 0 and pull one or more
horizontal Z-slices per sample, giving many 2D fields to train on cheaply.

    780 samples x len(z_slices) layers  ->  M 2D fields of shape (C, 60, 60)
    C = len(fields) channels (default 2: Perm, Por)

Normalization reuses the project convention (FieldNormalizer): log10(Perm) then
standardize per field; standardize Por. Stats are computed from the TRAIN split
only and saved to normalizer.json so the autoencoder, diffusion, and any later
decoding all share the exact same transform as the GRU-FNO model.

A whole 3D sample lands entirely in one split (train/val/test); its Z-slices are
never scattered across splits, so there is no leakage between a sample's own
layers. Scenario-held-out splitting is supported too (samples_per_scenario).

Usage:
    python src/diffusion/data2d.py --config configs/diffusion/cache2d.yaml
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

# src/ on path so we reuse the shared normalizer + field-group map.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from normalization import FieldNormalizer, safe_log10  # noqa: E402
from field_groups import FIELD_GROUP  # noqa: E402


def make_splits(total, val_ratio, test_ratio, seed=42, group_size=None):
    """Train/val/test split over sample indices (fixed seed).

    Self-contained here so the diffusion track does not depend on which version
    of src/inverse_dataset.make_splits happens to be installed on a machine.
    group_size=None -> random split; group_size=k -> whole-scenario held out
    (scenario = index // k), so no scenario appears in two splits.
    """
    rng = np.random.RandomState(seed)
    if group_size is None:
        perm = rng.permutation(total)
        n_test = int(total * test_ratio)
        n_val = int(total * val_ratio)
        return {"test": perm[:n_test],
                "val": perm[n_test:n_test + n_val],
                "train": perm[n_test + n_val:]}
    if group_size <= 0 or total % group_size != 0:
        raise ValueError(f"samples_per_scenario ({group_size}) must divide total ({total}).")
    n_groups = total // group_size
    gperm = rng.permutation(n_groups)
    n_test_g = max(1, int(round(n_groups * test_ratio)))
    n_val_g = max(1, int(round(n_groups * val_ratio)))
    if n_test_g + n_val_g >= n_groups:
        raise ValueError(f"too few scenarios ({n_groups}) for the requested val/test ratios.")

    def expand(groups):
        idx = []
        for g in groups:
            base = int(g) * group_size
            idx.extend(range(base, base + group_size))
        return np.array(sorted(idx), dtype=int)

    return {"test": expand(gperm[:n_test_g]),
            "val": expand(gperm[n_test_g:n_test_g + n_val_g]),
            "train": expand(gperm[n_test_g + n_val_g:])}


def resolve_z_slices(spec, n_z: int):
    """z_slices in config may be a list, an int (that single layer), or 'all'."""
    if isinstance(spec, str) and spec.lower() == "all":
        return list(range(n_z))
    if isinstance(spec, int):
        return [spec]
    return list(spec)


def _extract_slices(f, group_of, fields, sample_idx, z_slices, t_index=0):
    """Return {field: (len(z), 60, 60)} for one sample at timestep t_index."""
    out = {}
    for fld in fields:
        ds = f[group_of[fld]][fld]
        vol = np.asarray(ds[sample_idx, t_index, 0], dtype=np.float64)  # (Z, X, Y)
        out[fld] = vol[z_slices]  # (len(z), X, Y)
    return out


def build_cache(cfg: dict) -> Path:
    d = cfg["data"]
    data_path = d["data_path"]
    fields = list(d.get("fields", ["Perm", "Por"]))
    log_fields = set(d.get("log_fields", ["Perm"]))
    group_of = {fld: FIELD_GROUP[fld] for fld in fields}
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    a_field = fields[0]
    a_group = group_of[a_field]
    with h5py.File(data_path, "r") as f:
        n_total_file, _, _, n_z, _, _ = f[a_group][a_field].shape
    n_total = min(n_total_file, d.get("max_samples") or n_total_file)
    z_slices = resolve_z_slices(d.get("z_slices", "all"), n_z)

    group_size = d.get("samples_per_scenario") if d.get("split_by_scenario") else None
    splits = make_splits(n_total, d.get("val_ratio", 0.1), d.get("test_ratio", 0.1),
                         d.get("seed", 42), group_size=group_size)

    print(f"[cache] samples total={n_total} train={len(splits['train'])} "
          f"val={len(splits['val'])} test={len(splits['test'])} | z_slices={z_slices} "
          f"| fields={fields}")

    # --- pass 1: stats from TRAIN slices only (log then mean/std) ---
    acc = {fld: {"sum": 0.0, "sumsq": 0.0, "n": 0} for fld in fields}
    with h5py.File(data_path, "r") as f:
        for s in splits["train"]:
            sl = _extract_slices(f, group_of, fields, int(s), z_slices)
            for fld in fields:
                arr = sl[fld]
                if fld in log_fields:
                    arr = safe_log10(arr)
                acc[fld]["sum"] += float(arr.sum())
                acc[fld]["sumsq"] += float((arr ** 2).sum())
                acc[fld]["n"] += arr.size
    stats = {}
    for fld in fields:
        n = max(acc[fld]["n"], 1)
        mean = acc[fld]["sum"] / n
        var = max(acc[fld]["sumsq"] / n - mean ** 2, 0.0)
        stats[fld] = {"log": fld in log_fields, "mean": float(mean),
                      "std": float(np.sqrt(var))}
    normalizer = FieldNormalizer(stats)
    normalizer.save(out_dir / "normalizer.json")
    print(f"[cache] normalizer stats: "
          + " | ".join(f"{k}: mean={v['mean']:.3f} std={v['std']:.3f} log={v['log']}"
                       for k, v in stats.items()))

    # --- pass 2: extract + normalize every slice, tagged by split ---
    def collect(split_name):
        idxs = splits[split_name]
        rows, sample_ids = [], []
        with h5py.File(data_path, "r") as f:
            for s in idxs:
                sl = _extract_slices(f, group_of, fields, int(s), z_slices)
                for zi, z in enumerate(z_slices):
                    chans = [normalizer.normalize(fld, sl[fld][zi][None])[0]
                             for fld in fields]  # each (X, Y)
                    rows.append(np.stack(chans, axis=0))       # (C, X, Y)
                    sample_ids.append(int(s))
        if not rows:
            return (np.zeros((0, len(fields), 1, 1), np.float32),
                    np.zeros((0,), np.int64))
        return (np.asarray(rows, dtype=np.float32),
                np.asarray(sample_ids, dtype=np.int64))

    arrays = {}
    for split in ("train", "val", "test"):
        X, sid = collect(split)
        arrays[f"X_{split}"] = X
        arrays[f"sid_{split}"] = sid
        print(f"[cache] {split}: {X.shape[0]} slices, tensor {X.shape}")

    meta = {"fields": fields, "log_fields": sorted(log_fields), "z_slices": z_slices,
            "grid": list(arrays["X_train"].shape[-2:]), "n_samples": n_total,
            "data_path": data_path}
    (out_dir / "cache_meta.json").write_text(json.dumps(meta, indent=2))
    np.savez_compressed(out_dir / "slices2d.npz", **arrays)
    print(f"[cache] wrote {out_dir/'slices2d.npz'} and normalizer.json")
    return out_dir


class Slices2D(Dataset):
    """Loads one split from the cached npz. Returns (C, X, Y) float32 tensors."""

    def __init__(self, cache_dir: str | Path, split: str = "train"):
        z = np.load(Path(cache_dir) / "slices2d.npz")
        self.X = torch.from_numpy(z[f"X_{split}"].astype(np.float32))
        self.sid = torch.from_numpy(z[f"sid_{split}"])
        meta = json.loads((Path(cache_dir) / "cache_meta.json").read_text())
        self.fields = meta["fields"]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i]


def get_loader(cache_dir, split="train", batch_size=16, shuffle=None,
               num_workers=0):
    ds = Slices2D(cache_dir, split)
    if shuffle is None:
        shuffle = (split == "train")
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=torch.cuda.is_available())


def _load_cfg(path):
    return yaml.safe_load(Path(path).read_text())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    build_cache(_load_cfg(args.config))
