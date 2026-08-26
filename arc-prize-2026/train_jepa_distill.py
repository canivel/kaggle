"""Distill the 25M JEPAWorldModel teacher into a ViT-XXS student
(~3-4M params) for CPU-friendly Kaggle inference.

Student architecture:
- d_model 256 → 128
- enc_layers 12 → 4
- pred_layers 8 → 3
- Same tokenizer, action encoder, head shapes (preserves output API)

Loss:
- L1(student_z_t, teacher_z_t) at encoder output (after EMA target)
- L1(student_z_pred, teacher_z_pred) for action-conditioned prediction
- L1 on reward/done logits (small weight)

Usage:
  uv run python train_jepa_distill.py --steps 15000 --batch 64
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from jepa_wm.models.jepa import JEPAConfig, JEPAWorldModel
from jepa_wm.training.train import TrajectoryDataset


def make_xxs_config() -> JEPAConfig:
    cfg = JEPAConfig()
    cfg.d_model = 128
    cfg.enc_layers = 4
    cfg.pred_layers = 3
    cfg.enc_heads = 4
    cfg.pred_heads = 4
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="jepa_wm/checkpoints/jepa_wm_final.pt")
    ap.add_argument("--data", default="jepa_wm/data/trajectories.npz")
    ap.add_argument("--out", default="jepa_wm/checkpoints/jepa_wm_xxs.pt")
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--log-every", type=int, default=200)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load teacher
    teacher_cfg = JEPAConfig()
    teacher = JEPAWorldModel(teacher_cfg).to(device).eval()
    ck = torch.load(args.teacher, map_location=device, weights_only=False)
    teacher.load_state_dict(ck["wm"])
    for p in teacher.parameters():
        p.requires_grad = False
    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"Teacher params: {teacher_params/1e6:.2f}M")

    # Build student
    student_cfg = make_xxs_config()
    student = JEPAWorldModel(student_cfg).to(device)
    student_params = sum(p.numel() for p in student.parameters())
    print(f"Student (ViT-XXS) params: {student_params/1e6:.2f}M (target 3-4M)")

    # Data
    ds = TrajectoryDataset(args.data, augment=True)
    print(f"Dataset: {ds.N} samples")
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=True, num_workers=0,
        pin_memory=(device.type == "cuda"), drop_last=True,
    )
    opt = optim.AdamW(
        [p for p in student.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-5
    )

    def gen():
        while True:
            for b in loader:
                yield b
    bgen = gen()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    running = {"enc": 0.0, "pred": 0.0, "r": 0.0, "d": 0.0, "total": 0.0}

    for step in range(args.steps):
        s_t, s_t1, aid, ax, ay, r_cls, d = next(bgen)
        s_t = s_t.to(device); s_t1 = s_t1.to(device)
        aid = aid.to(device); ax = ax.to(device); ay = ay.to(device)
        r_cls = r_cls.to(device); d = d.to(device)

        # Teacher forward (no grad)
        with torch.no_grad():
            t_z_t = teacher.target_encode(s_t)  # (B, n_tokens, d_model_teacher)
            t_z_t1, t_z_pred, t_r_logits, t_done_logit, t_v = teacher.step(s_t, aid, ax, ay)

        # Student forward
        z_t, z_pred, r_logits, done_logit, v = student.step(s_t, aid, ax, ay)

        # Distillation losses
        # Encoder: project teacher d_model down to student d_model via mean-pool
        # along the last dim. Simpler: take linear projection on the fly.
        # For now: project teacher to student by truncating channel dim (since
        # 256 -> 128 is a clean half, take first 128 dims).
        t_z_t_proj = t_z_t[..., :student_cfg.d_model]
        t_z_pred_proj = t_z_pred[..., :student_cfg.d_model]
        L_enc = F.l1_loss(z_t, t_z_t_proj)
        L_pred = F.l1_loss(z_pred, t_z_pred_proj)
        # Reward / done heads: train against teacher logits
        L_r = F.l1_loss(r_logits, t_r_logits)
        L_d = F.l1_loss(done_logit, t_done_logit)
        total = L_enc + L_pred + 0.1 * L_r + 0.1 * L_d

        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()

        running["enc"] += float(L_enc.item())
        running["pred"] += float(L_pred.item())
        running["r"] += float(L_r.item())
        running["d"] += float(L_d.item())
        running["total"] += float(total.item())

        if (step + 1) % args.log_every == 0:
            n = args.log_every
            elapsed = time.time() - t0
            print(f"step={step+1:6d} total={running['total']/n:.4f} "
                  f"enc={running['enc']/n:.4f} pred={running['pred']/n:.4f} "
                  f"r={running['r']/n:.4f} d={running['d']/n:.4f} elapsed={elapsed:.0f}s")
            for k in running:
                running[k] = 0.0

    torch.save({
        "wm": student.state_dict(),
        "cfg": student_cfg.__dict__,
        "step": args.steps,
        "teacher_ckpt": args.teacher,
    }, args.out)
    sz = Path(args.out).stat().st_size / 1024 / 1024
    print(f"Saved {args.out} ({sz:.1f} MB, {student_params/1e6:.2f}M params)")


if __name__ == "__main__":
    main()
