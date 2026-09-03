# Reservoir Permeability Inversion by Latent Diffusion

Estimating **permeability** fields from **sparse sensor data** (saturation and
pressure at a few locations and times) using a **latent-diffusion prior** plus
**ensemble data assimilation (ES-MDA)**. Instead of a single map, the method
returns an **ensemble** of maps consistent with the readings, so the spread across
the ensemble is a calibrated **uncertainty**.

Data: a CO2/aquifer-storage simulation (`780_c5.h5`, 780 samples, fields shaped
`(56, 1, 24, 60, 60)`).

## Result (see `reports/diffusion_results.pdf`)

The 3D pipeline recovers the permeability field where the data constrains it, with
honest uncertainty elsewhere:

| Setting | Perm R² (prior → posterior) | Near-sensor RMSE | Coverage |
|---|---|---|---|
| 2D | −0.02 → **−0.17** (fails) | 1.17 → 1.40 | 0.57 |
| 3D + injection controls, 8 sensors | 0.16 → **0.37** | 0.90 → 0.66 | 0.89 |
| 3D + injection controls, 48 sensors | 0.22 → **0.79** | 0.87 → 0.43 | 0.84 |

The key finding: the forward model must see the **injection program**
(`Rate`, `Bhp`, `Boundary`), not just the rock. Pressure is driven by injection,
and the dataset spans many injection scenarios, so a rock-only forward cannot tell
them apart. With the controls fed in, the forward becomes accurate (pressure R²
−0.22 → 0.89) and the inverse recovers the field.

Full write-up, tables, and figures are in **`reports/diffusion_results.pdf`**; the
original plan is in **`reports/diffusion_plan.pdf`**.

## How it works

1. **Autoencoder** compresses a permeability field to a small latent grid.
2. **Latent diffusion** learns a prior over those latents; sampling it gives an
   ensemble of plausible permeability fields.
3. **Forward surrogate** maps a candidate field (+ known injection controls) to the
   saturation/pressure the sensors would read.
4. **ES-MDA** in latent space nudges the ensemble so its predicted readings match
   the observed ones; only the rock varies, the controls are known.
5. **Validation** reports recovery near the sensors and ensemble coverage.

The pipeline is built in **2D first** (fast to iterate) then in **full 3D**.

## Layout

```
src/diffusion/        the pipeline (2D modules + their *3d.py counterparts)
  data{2,3}d.py         cache fields + normalization + splits
  autoencoder{2,3}d.py  the compressor  (+ train_autoencoder*.py)
  ldm{2,3}d.py          latent diffusion (+ train_diffusion*.py, sample_prior*.py)
  forward{2,3}d.py      forward surrogate: (Perm[, controls]) -> sat/pressure
  measurement.py        sensor operator H(field) -> readings at (z,x,y) x time
  assimilate{2,3}d.py   ES-MDA assimilation
  validate_assim*.py    the result figure
src/normalization.py  per-field log/standardize (train-set statistics)
src/field_groups.py   which HDF5 group each field lives in
configs/diffusion/    one YAML per step (2D and 3D)
reports/              the two PDFs + the report builder
```

## Running (3D pipeline)

Put the dataset at `data/raw/780_c5.h5`, then run from the repo root:

```bash
python src/diffusion/data3d.py              --config configs/diffusion/cache3d.yaml
python src/diffusion/train_autoencoder3d.py --config configs/diffusion/ae3d.yaml
python src/diffusion/train_diffusion3d.py   --config configs/diffusion/ldm3d.yaml
python src/diffusion/forward3d.py           --config configs/diffusion/forward3d.yaml
python src/diffusion/assimilate3d.py        --config configs/diffusion/assim3d.yaml
python src/diffusion/validate_assim3d.py    --run outputs/diffusion3d/assim3d --cache outputs/diffusion3d/cache3d
python reports/build_diffusion_results_pdf.py
```

The 2D pipeline is the same with the non-`3d` scripts and `cache2d/ae2d/ldm2d/
forward2d/assim*` configs. Runs on GPU if available, else CPU. Sensors are defined
in `configs/diffusion/sensors.yaml` (the base 8-point layout) or
`sensors3d_dense.yaml` (48 points).

## Notes

- Recovering a full field from a handful of sensors is under-determined for any
  method; success here is recovery **near the data** plus a **calibrated
  uncertainty band** elsewhere, not a global R² near 1.
- Target is permeability; porosity is a straightforward extension (single vs. two
  output channels) left for later.
- The dataset and trained checkpoints are not in this repo (size); paths for them
  are relative and created under `outputs/` and `data/` when you run.

## Requirements

See `requirements.txt` (PyTorch, NumPy, h5py, PyYAML, matplotlib; reportlab for the
PDF).
