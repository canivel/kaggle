"""Train the JEPA world model on collected ARC-AGI-3 trajectories.

Loss = L_jepa (1-step + 2-step rollout) + lambda_r * L_reward + lambda_d * L_done
       + lambda_vreg * L_variance_reg (VICReg-style)

EMA momentum: linear schedule 0.996 -> 1.000 across training.

Usage:
  uv run python -m jepa_wm.training.train \
    --data jepa_wm/data/trajectories.npz \
    --ckpt jepa_wm/checkpoints \
    --steps 100000 --batch 128 --lr 3e-4
"""
from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from jepa_wm.data.augmentations import TrajectoryAugmenter
from jepa_wm.models.jepa import JEPAConfig, JEPAWorldModel


class TrajectoryDataset(torch.utils.data.Dataset):
    """Loads the .npz and yields augmented (s_t, s_t1, action_id, ax, ay, r_class, done) tuples."""

    def __init__(self, npz_path: str, augment: bool = True, seed: int = 0):
        z = np.load(npz_path)
        self.s_t = z["s_t"]
        self.s_t1 = z["s_t1"]
        self.action_id = z["action_id"]
        self.ax = z["ax"]
        self.ay = z["ay"]
        self.r_class = z["r_class"]
        self.done = z["done"]
        self.N = self.s_t.shape[0]
        self.aug = TrajectoryAugmenter(seed=seed) if augment else None

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        s_t = self.s_t[i].astype(np.int64)
        s_t1 = self.s_t1[i].astype(np.int64)
        # Data stores raw ARC action_id (1..7); embedding wants 0..n_simple_actions-1.
        aid_raw = int(self.action_id[i])
        aid = max(0, min(6, aid_raw - 1))
        x = max(0, min(63, int(self.ax[i])))
        y = max(0, min(63, int(self.ay[i])))
        r = max(0, min(2, int(self.r_class[i])))
        d = max(0.0, min(1.0, float(self.done[i])))
        if self.aug is not None:
            s_t, s_t1, _, x, y = self.aug(s_t, s_t1, aid_raw, x, y)  # aug expects raw aid for ACTION6 semantics
            x = max(0, min(63, x)); y = max(0, min(63, y))
        # Clamp s grids too
        s_t = np.clip(s_t, 0, 15)
        s_t1 = np.clip(s_t1, 0, 15)
        return (
            torch.from_numpy(s_t.astype(np.int64)),
            torch.from_numpy(s_t1.astype(np.int64)),
            torch.tensor(aid, dtype=torch.long),
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(r, dtype=torch.long),
            torch.tensor(d, dtype=torch.float32),
        )


def ema_momentum(step: int, total: int, m_start: float = 0.996, m_end: float = 1.000) -> float:
    t = min(1.0, step / max(1, total))
    return m_start + (m_end - m_start) * t


def train(
    data_path: str,
    ckpt_dir: str,
    steps: int = 100_000,
    batch_size: int = 32,
    grad_accum: int = 4,
    lr: float = 3e-4,
    lambda_r: float = 1.0,
    lambda_d: float = 0.5,
    lambda_vreg: float = 0.1,
    log_every: int = 100,
    ckpt_every: int = 5000,
    device: str | None = None,
    resume: str | None = None,
    use_amp: bool = True,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  batch={batch_size}  grad_accum={grad_accum}  amp={use_amp}")
    cfg = JEPAConfig()
    wm = JEPAWorldModel(cfg).to(device)
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and device == 'cuda'))
    print({k: f"{v/1e6:.2f}M" for k, v in wm.n_params().items()})

    if resume and Path(resume).exists():
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        wm.load_state_dict(ckpt["wm"])
        start_step = ckpt.get("step", 0)
        print(f"Resumed from {resume} at step {start_step}")
    else:
        start_step = 0

    ds = TrajectoryDataset(data_path, augment=True)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=0,
        pin_memory=(device == "cuda"), drop_last=True,
    )
    opt = torch.optim.AdamW(
        [p for p in wm.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4,
    )

    def get_batch():
        while True:
            for b in loader:
                yield b
    bgen = get_batch()

    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    running = {"loss": 0.0, "jepa": 0.0, "r": 0.0, "d": 0.0, "vreg": 0.0}
    opt.zero_grad()
    for step in range(start_step, steps):
        accum_loss = {"loss": 0.0, "jepa": 0.0, "r": 0.0, "d": 0.0, "vreg": 0.0}
        for micro in range(grad_accum):
            s_t, s_t1, aid, ax, ay, r_cls, d = next(bgen)
            s_t = s_t.to(device, non_blocking=True)
            s_t1 = s_t1.to(device, non_blocking=True)
            aid = aid.to(device); ax = ax.to(device); ay = ay.to(device)
            r_cls = r_cls.to(device); d = d.to(device)
            with torch.amp.autocast('cuda', enabled=(use_amp and device == 'cuda')):
                z_t, z_pred, r_logits, done_logit, v = wm.step(s_t, aid, ax, ay)
                with torch.no_grad():
                    z_target = wm.target_encode(s_t1)
                L_jepa = F.l1_loss(z_pred, z_target)
                L_r = F.cross_entropy(r_logits, r_cls)
                L_d = F.binary_cross_entropy_with_logits(done_logit, d)
                L_vreg = wm.compute_variance_reg(z_t.float())
                loss_micro = (L_jepa + lambda_r * L_r + lambda_d * L_d + lambda_vreg * L_vreg) / grad_accum
            scaler.scale(loss_micro).backward()
            accum_loss["loss"] += float(loss_micro.item()) * grad_accum
            accum_loss["jepa"] += float(L_jepa.item())
            accum_loss["r"] += float(L_r.item())
            accum_loss["d"] += float(L_d.item())
            accum_loss["vreg"] += float(L_vreg.item())
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
        # EMA update
        m = ema_momentum(step, steps, cfg.ema_m_start, cfg.ema_m_end)
        wm.update_target(m)
        loss = type('L', (), {'item': lambda self: accum_loss['loss']/grad_accum})()
        L_jepa = type('L', (), {'item': lambda self: accum_loss['jepa']/grad_accum})()
        L_r = type('L', (), {'item': lambda self: accum_loss['r']/grad_accum})()
        L_d = type('L', (), {'item': lambda self: accum_loss['d']/grad_accum})()
        L_vreg = type('L', (), {'item': lambda self: accum_loss['vreg']/grad_accum})()

        running["loss"] += float(loss.item())
        running["jepa"] += float(L_jepa.item())
        running["r"] += float(L_r.item())
        running["d"] += float(L_d.item())
        running["vreg"] += float(L_vreg.item())

        if (step + 1) % log_every == 0:
            n = log_every
            elapsed = time.time() - t0
            var = wm.latent_variance(s_t).mean().item()
            print(
                f"step={step+1:6d} loss={running['loss']/n:.4f} "
                f"jepa={running['jepa']/n:.4f} r={running['r']/n:.4f} "
                f"d={running['d']/n:.4f} vreg={running['vreg']/n:.4f} "
                f"latent_var={var:.4f} ema_m={m:.4f} elapsed={elapsed:.0f}s"
            )
            for k in running:
                running[k] = 0.0
        if (step + 1) % ckpt_every == 0:
            ckpt_path = Path(ckpt_dir) / f"jepa_wm_step{step+1}.pt"
            torch.save({"wm": wm.state_dict(), "step": step + 1, "cfg": cfg.__dict__}, ckpt_path)
            print(f"  ckpt -> {ckpt_path} ({ckpt_path.stat().st_size/1024/1024:.1f} MB)")

    final = Path(ckpt_dir) / "jepa_wm_final.pt"
    torch.save({"wm": wm.state_dict(), "step": steps, "cfg": cfg.__dict__}, final)
    print(f"FINAL: {final}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="jepa_wm/data/trajectories.npz")
    ap.add_argument("--ckpt", default="jepa_wm/checkpoints")
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()
    train(args.data, args.ckpt, args.steps, args.batch, args.grad_accum, args.lr,
          resume=args.resume, use_amp=(not args.no_amp))


if __name__ == "__main__":
    main()
