"""Phase 1 driver -- train the 2D autoencoder and report the recon gate.

    python src/diffusion/train_autoencoder.py --config configs/diffusion/ae2d.yaml

Saves into output.dir: best_ae.pt (lowest val recon MSE), last_ae.pt, history.json,
config_used.yaml, and recon_report.json with per-field R^2 on val (the gate). Runs
on GPU if available, else CPU. Reuses the cache built by data2d.py.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data2d import get_loader, Slices2D  # noqa: E402
from autoencoder2d import build_autoencoder, recon_loss  # noqa: E402


@torch.no_grad()
def per_field_r2(model, loader, device, n_fields):
    """R^2 per channel in normalized space: 1 - SSE/SST (SST vs field mean)."""
    model.eval()
    sse = torch.zeros(n_fields, dtype=torch.float64)
    # accumulate SST against the global (per-field) mean of the target
    sums = torch.zeros(n_fields, dtype=torch.float64)
    sumsq = torch.zeros(n_fields, dtype=torch.float64)
    count = 0
    for x in loader:
        x = x.to(device)
        y = model(x)
        err = (y - x) ** 2
        sse += err.sum(dim=(0, 2, 3)).double().cpu()
        sums += x.sum(dim=(0, 2, 3)).double().cpu()
        sumsq += (x ** 2).sum(dim=(0, 2, 3)).double().cpu()
        count += x.shape[0] * x.shape[2] * x.shape[3]
    mean = sums / max(count, 1)
    sst = sumsq - count * mean ** 2
    r2 = 1.0 - sse / torch.clamp(sst, min=1e-12)
    return r2.tolist()


def train_from_config(cfg: dict):
    d, m, t = cfg["data"], cfg["model"], cfg["training"]
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = d["cache_dir"]

    torch.manual_seed(t.get("seed", 42))
    np.random.seed(t.get("seed", 42))

    fields = Slices2D(cache, "train").fields
    n_fields = len(fields)
    bs = t.get("batch_size", 16)
    train_loader = get_loader(cache, "train", bs, num_workers=t.get("num_workers", 0))
    val_loader = get_loader(cache, "val", bs, shuffle=False,
                            num_workers=t.get("num_workers", 0))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_autoencoder(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=t.get("learning_rate", 1e-3),
                            weight_decay=t.get("weight_decay", 1e-5))
    gw = t.get("grad_weight", 0.1)
    print(f"[ae] device={device} fields={fields} params={model.num_params:,} "
          f"train={len(train_loader.dataset)} val={len(val_loader.dataset)}")

    shutil.copy2(_dump_cfg(cfg, out_dir), out_dir / "config_used.yaml")
    history, best = [], float("inf")
    epochs = t.get("epochs", 60)
    for ep in range(epochs):
        tic = time.time()
        model.train()
        tr = 0.0
        for x in train_loader:
            x = x.to(device)
            opt.zero_grad()
            loss = recon_loss(model(x), x, gw)
            loss.backward()
            opt.step()
            tr += loss.item() * x.shape[0]
        tr /= max(len(train_loader.dataset), 1)

        model.eval()
        va = 0.0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                va += torch.nn.functional.mse_loss(model(x), x).item() * x.shape[0]
        va /= max(len(val_loader.dataset), 1)
        r2 = per_field_r2(model, val_loader, device, n_fields)
        dt = time.time() - tic
        rec = {"epoch": ep, "train_loss": tr, "val_mse": va,
               "val_r2": {f: r2[i] for i, f in enumerate(fields)}, "sec": round(dt, 1)}
        history.append(rec)
        print(f"[ae] epoch {ep:3d} | train {tr:.5f} | val_mse {va:.5f} | "
              f"R2 {['%.4f' % v for v in r2]} | {dt:.1f}s")

        ckpt = {"epoch": ep, "model_state": model.state_dict(), "config": cfg,
                "fields": fields, "val_mse": va, "val_r2": r2}
        torch.save(ckpt, out_dir / "last_ae.pt")
        if va < best:
            best = va
            torch.save(ckpt, out_dir / "best_ae.pt")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    # gate report from the best checkpoint
    best_ck = torch.load(out_dir / "best_ae.pt", map_location=device)
    model.load_state_dict(best_ck["model_state"])
    r2 = per_field_r2(model, val_loader, device, n_fields)
    report = {"val_r2": {f: r2[i] for i, f in enumerate(fields)},
              "val_mse_best": best,
              "gate_pass": bool(all(v >= t.get("gate_r2", 0.98) for v in r2)),
              "gate_r2": t.get("gate_r2", 0.98)}
    (out_dir / "recon_report.json").write_text(json.dumps(report, indent=2))
    print(f"[ae] GATE {'PASS' if report['gate_pass'] else 'CHECK'} "
          f"| val R2 {report['val_r2']} (threshold {report['gate_r2']})")
    print(f"[ae] done. artifacts in {out_dir}/")
    return report


def _dump_cfg(cfg, out_dir):
    p = out_dir / "_effective_config.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    train_from_config(yaml.safe_load(Path(args.config).read_text()))
