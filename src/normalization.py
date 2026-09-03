"""Per-field normalization for the inverse task.

Why per-field: permeability spans ~5 orders of magnitude (0.0003-51.8) while
porosity is tiny (0.018-0.14) and pressure is ~3e4. If we standardize them
together or feed raw values into MSE, permeability/pressure dominate and the
other fields barely train. So each field gets its own transform:

    * optional log10 (used for permeability, whose range is huge)
    * then standardize:  (x - mean) / std

Statistics are computed ONLY from the training split (never val/test) by
streaming one sample at a time from the HDF5, so it stays memory-safe on the
60 GB file. Stats are saved to JSON so evaluation/inference reuse the exact
same transform.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Sequence

import h5py
import numpy as np


EPS = 1e-8


def safe_log10(x: np.ndarray, eps: float = EPS) -> np.ndarray:
    return np.log10(np.clip(x, eps, None))


class FieldNormalizer:
    """Holds {field: {log, mean, std}} and applies / inverts the transform."""

    def __init__(self, stats: Dict[str, dict] | None = None):
        self.stats: Dict[str, dict] = stats or {}

    # ----- apply -----
    def normalize(self, field: str, arr: np.ndarray) -> np.ndarray:
        s = self.stats[field]
        x = arr.astype(np.float32)
        if s["log"]:
            x = safe_log10(x)
        return (x - s["mean"]) / (s["std"] + EPS)

    def denormalize(self, field: str, arr: np.ndarray) -> np.ndarray:
        s = self.stats[field]
        x = arr.astype(np.float32) * (s["std"] + EPS) + s["mean"]
        if s["log"]:
            x = np.power(10.0, x)
        return x

    # ----- persistence -----
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.stats, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "FieldNormalizer":
        return cls(json.loads(Path(path).read_text()))


def compute_stats(
    file_path: str | Path,
    group_of: Dict[str, str],
    log_fields: Iterable[str] = (),
    indices: Sequence[int] | None = None,
    time_dim: Sequence[int] | None = None,
    max_samples: int | None = None,
) -> FieldNormalizer:
    """Stream over training samples and compute per-field mean/std.

    Args:
        file_path: path to the HDF5 file.
        group_of: {field_name: hdf5_group}  e.g. {"sat": "outputs", "Perm": "inputs"}.
        log_fields: fields to log10-transform before computing stats.
        indices: sample indices to use (the TRAIN split). Defaults to all.
        time_dim: timesteps to include. Defaults to all.
        max_samples: cap the number of samples used (handy for a quick pass).
    """
    file_path = Path(file_path)
    log_fields = set(log_fields)
    fields = list(group_of.keys())

    with h5py.File(file_path, "r") as f:
        if indices is None:
            first = group_of[fields[0]]
            indices = list(range(f[first][fields[0]].shape[0]))
        indices = list(indices)
        if max_samples is not None:
            indices = indices[:max_samples]

        # Welford-style running accumulation per field (sum, sumsq, count).
        acc = {fld: {"sum": 0.0, "sumsq": 0.0, "n": 0} for fld in fields}

        for idx in indices:
            for fld in fields:
                ds = f[group_of[fld]][fld]
                arr = ds[idx] if time_dim is None else ds[idx, list(time_dim)]
                arr = np.asarray(arr, dtype=np.float64)
                if fld in log_fields:
                    arr = safe_log10(arr)
                acc[fld]["sum"] += float(arr.sum())
                acc[fld]["sumsq"] += float((arr ** 2).sum())
                acc[fld]["n"] += arr.size

    stats: Dict[str, dict] = {}
    for fld in fields:
        n = acc[fld]["n"]
        mean = acc[fld]["sum"] / n
        var = max(acc[fld]["sumsq"] / n - mean ** 2, 0.0)
        stats[fld] = {
            "log": fld in log_fields,
            "mean": float(mean),
            "std": float(np.sqrt(var)),
        }
    return FieldNormalizer(stats)
