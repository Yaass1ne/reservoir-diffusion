"""Phase 3 -- the measurement operator H.

A real site does not report saturation and pressure everywhere. It reports them at
a few sensor locations, at a few recording times. H is that restriction:

    H(field_timeseries) -> readings at (location, time)

field_timeseries is (T, Z, X, Y) for one observable (sat or pressure). The sensor
set is a list of (z, x, y) locations plus a list of recording timesteps, held in a
YAML config so the same sensors are used everywhere in the pipeline. This is the
bridge Phase 4 (ESMDA) compares against: a kept realization must reproduce these
readings at the right places AND times -- exactly what Walid asked to see.

Because sat/pressure already exist in 780_c5.h5, this operator is testable NOW,
before any model is trained:

    python src/diffusion/measurement.py --config configs/diffusion/sensors.yaml \
        --data data/raw/780_c5.h5 --sample 0
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import h5py
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from field_groups import FIELD_GROUP  # noqa: E402


@dataclass
class SensorConfig:
    locations: List[tuple]      # list of (z, x, y)
    times: List[int]            # recording timesteps
    observables: List[str]      # e.g. ["sat", "pressure"]

    @property
    def n_readings(self) -> int:
        return len(self.observables) * len(self.locations) * len(self.times)


def load_sensors(path: str | Path) -> SensorConfig:
    cfg = yaml.safe_load(Path(path).read_text())
    s = cfg["sensors"]
    locs = [tuple(int(v) for v in loc) for loc in s["locations"]]
    return SensorConfig(locations=locs,
                        times=[int(t) for t in s["times"]],
                        observables=list(s.get("observables", ["sat", "pressure"])))


def apply_H(field_ts: np.ndarray, sensors: SensorConfig) -> np.ndarray:
    """Restrict one observable's (T, Z, X, Y) volume to sensor (location, time).

    Returns a (len(times), len(locations)) array of readings, in the same order as
    sensors.times x sensors.locations.
    """
    out = np.empty((len(sensors.times), len(sensors.locations)), dtype=np.float32)
    for ti, t in enumerate(sensors.times):
        for li, (z, x, y) in enumerate(sensors.locations):
            out[ti, li] = float(field_ts[t, z, x, y])
    return out


def readings_from_h5(data_path, sensors: SensorConfig, sample_idx: int):
    """Read real sensor readings for one sample straight from the dataset.

    Returns {observable: (len(times), len(locations))}. Validates that requested
    sensor coordinates and times are inside the grid.
    """
    result = {}
    with h5py.File(data_path, "r") as f:
        for obs in sensors.observables:
            ds = f[FIELD_GROUP[obs]][obs]           # (N, T, 1, Z, X, Y)
            _, n_t, _, n_z, n_x, n_y = ds.shape
            _validate(sensors, n_t, n_z, n_x, n_y)
            vol = np.asarray(ds[sample_idx, :, 0])  # (T, Z, X, Y)
            result[obs] = apply_H(vol, sensors)
    return result


def _validate(sensors: SensorConfig, n_t, n_z, n_x, n_y):
    for t in sensors.times:
        if not 0 <= t < n_t:
            raise ValueError(f"sensor time {t} outside 0..{n_t - 1}")
    for (z, x, y) in sensors.locations:
        if not (0 <= z < n_z and 0 <= x < n_x and 0 <= y < n_y):
            raise ValueError(f"sensor loc {(z, x, y)} outside grid "
                             f"(Z<{n_z}, X<{n_x}, Y<{n_y})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--sample", type=int, default=0)
    args = ap.parse_args()

    sensors = load_sensors(args.config)
    print(f"[H] {len(sensors.locations)} locations x {len(sensors.times)} times "
          f"x {len(sensors.observables)} observables = {sensors.n_readings} readings")
    r = readings_from_h5(args.data, sensors, args.sample)
    for obs, mat in r.items():
        print(f"\n[{obs}] readings (rows=times {sensors.times}, cols=locations):")
        print(np.array2string(mat, precision=4, suppress_small=True))
        print(f"  range: {mat.min():.4g} .. {mat.max():.4g}")
    print("\n[H] operator OK -- readings extracted at the requested locations and times.")


if __name__ == "__main__":
    main()
