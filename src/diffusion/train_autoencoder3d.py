"""Phase 6 (1) driver -- train the 3D autoencoder + recon-R2 gate.
    python src/diffusion/train_autoencoder3d.py --config configs/diffusion/ae3d.yaml
"""
from __future__ import annotations

import argparse, json, shutil, sys, time
from pathlib import Path
import numpy as np, torch, yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data3d import get_loader, Volumes3D  # noqa: E402
from autoencoder3d import build_autoencoder3d, recon_loss  # noqa: E402


@torch.no_grad()
def per_field_r2(model, loader, device, n):
    model.eval()
    sse = torch.zeros(n, dtype=torch.float64); sums = torch.zeros(n, dtype=torch.float64)
    sumsq = torch.zeros(n, dtype=torch.float64); count = 0
    for x in loader:
        x = x.to(device); y = model(x)
        red = (0, 2, 3, 4)
        sse += ((y - x) ** 2).sum(red).double().cpu()
        sums += x.sum(red).double().cpu(); sumsq += (x ** 2).sum(red).double().cpu()
        count += x.shape[0] * x.shape[2] * x.shape[3] * x.shape[4]
    mean = sums / max(count, 1); sst = sumsq - count * mean ** 2
    return (1 - sse / torch.clamp(sst, min=1e-12)).tolist()


def train_from_config(cfg):
    d, t = cfg["data"], cfg["training"]
    out = Path(cfg["output"]["dir"]); out.mkdir(parents=True, exist_ok=True)
    cache = d["cache_dir"]
    torch.manual_seed(t.get("seed", 42)); np.random.seed(t.get("seed", 42))
    fields = Volumes3D(cache, "train").fields
    bs = t.get("batch_size", 2)
    tr_l = get_loader(cache, "train", bs, num_workers=t.get("num_workers", 0))
    va_l = get_loader(cache, "val", bs, shuffle=False, num_workers=t.get("num_workers", 0))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_autoencoder3d(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=t.get("learning_rate", 1e-3),
                            weight_decay=t.get("weight_decay", 1e-5))
    gw = t.get("grad_weight", 0.1)
    print(f"[ae3d] device={device} fields={fields} params={model.num_params:,} "
          f"train={len(tr_l.dataset)} val={len(va_l.dataset)}")
    shutil.copy2(_dump(cfg, out), out / "config_used.yaml")
    best, hist = float("inf"), []
    for ep in range(t.get("epochs", 80)):
        tic = time.time(); model.train(); tl = 0.0
        for x in tr_l:
            x = x.to(device); opt.zero_grad()
            loss = recon_loss(model(x), x, gw); loss.backward(); opt.step()
            tl += loss.item() * x.shape[0]
        tl /= max(len(tr_l.dataset), 1)
        model.eval(); vm = 0.0
        with torch.no_grad():
            for x in va_l:
                x = x.to(device); vm += torch.nn.functional.mse_loss(model(x), x).item() * x.shape[0]
        vm /= max(len(va_l.dataset), 1)
        r2 = per_field_r2(model, va_l, device, len(fields))
        dt = time.time() - tic
        hist.append({"epoch": ep, "train_loss": tl, "val_mse": vm,
                     "val_r2": {f: r2[i] for i, f in enumerate(fields)}, "sec": round(dt, 1)})
        print(f"[ae3d] ep {ep:3d} | train {tl:.5f} | val_mse {vm:.5f} | "
              f"R2 {['%.4f' % v for v in r2]} | {dt:.1f}s")
        ck = {"epoch": ep, "model_state": model.state_dict(), "config": cfg, "fields": fields, "val_r2": r2}
        torch.save(ck, out / "last_ae.pt")
        if vm < best:
            best = vm; torch.save(ck, out / "best_ae.pt")
        (out / "history.json").write_text(json.dumps(hist, indent=2))
    r2 = per_field_r2(model, va_l, device, len(fields))
    rep = {"val_r2": {f: r2[i] for i, f in enumerate(fields)}, "val_mse_best": best,
           "gate_pass": bool(all(v >= t.get("gate_r2", 0.97) for v in r2)), "gate_r2": t.get("gate_r2", 0.97)}
    (out / "recon_report.json").write_text(json.dumps(rep, indent=2))
    print(f"[ae3d] GATE {'PASS' if rep['gate_pass'] else 'CHECK'} | val R2 {rep['val_r2']}")
    return rep


def _dump(cfg, out):
    p = out / "_effective_config.yaml"; p.write_text(yaml.safe_dump(cfg, sort_keys=False)); return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    a = ap.parse_args(); train_from_config(yaml.safe_load(Path(a.config).read_text()))
