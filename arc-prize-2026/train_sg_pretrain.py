"""Pretrain the StochasticGoose ActionModel on bug-fixed BFS trajectories.

Data: jepa_wm/data/trajectories.npz (10k tuples, ACTION6 coords correct).
Target: ActionModel from v47_agent.py — pretrained weights for v34 / v47 ports.

Architecture: 16-ch one-hot 64x64 → 4-conv backbone (32-64-128-256) → action
head (5) + 64x64 coord head (4096). Total 4101 logits per sample.

Loss: BCE on selected logit only (action_idx → frame_changed). Following SG's
_train_action_model.

Usage:
  uv run python train_sg_pretrain.py --steps 20000 --batch 64 \
    --out runs/sg_pretrain/sg_action_model.pt
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


# Import ActionModel from v47 (which is our SG port)
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "notebooks" / "forge_agent"))
# v47 needs `agents.agent` and arcengine; avoid the import chain by re-defining
# the model inline. Single source of truth: copy/paste from v47.

class ActionModel(nn.Module):
    """SG architecture (matches notebooks/forge_agent/v47_agent.py)."""
    def __init__(self, input_channels=16, grid_size=64):
        super().__init__()
        self.grid_size = grid_size
        self.num_action_types = 5
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.action_pool = nn.MaxPool2d(4, 4)
        self.action_fc = nn.Linear(256 * 16 * 16, 512)
        self.action_head = nn.Linear(512, self.num_action_types)
        self.coord_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.coord_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.coord_conv3 = nn.Conv2d(64, 32, kernel_size=1)
        self.coord_conv4 = nn.Conv2d(32, 1, kernel_size=1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        f = F.relu(self.conv4(x))
        a = self.action_pool(f).view(f.size(0), -1)
        a = self.dropout(F.relu(self.action_fc(a)))
        a = self.action_head(a)
        c = F.relu(self.coord_conv1(f))
        c = F.relu(self.coord_conv2(c))
        c = F.relu(self.coord_conv3(c))
        c = self.coord_conv4(c).view(f.size(0), -1)
        return torch.cat([a, c], dim=1)


def frame_to_one_hot(frame: np.ndarray, n_colours: int = 16) -> np.ndarray:
    """Convert (H, W) color-indexed frame → (n_colours, H, W) one-hot float32."""
    H, W = frame.shape
    out = np.zeros((n_colours, H, W), dtype=np.float32)
    f = np.clip(frame, 0, n_colours - 1).astype(np.int64)
    np.put_along_axis(out, f[None], 1, axis=0)
    return out


class SGDataset(torch.utils.data.Dataset):
    def __init__(self, npz_path: str):
        z = np.load(npz_path)
        self.s_t = z["s_t"]
        self.s_t1 = z["s_t1"]
        self.action_id = z["action_id"]
        self.ax = z["ax"]
        self.ay = z["ay"]
        self.N = self.s_t.shape[0]
        # Pre-compute labels (frame_changed)
        diffs = (self.s_t != self.s_t1).reshape(self.N, -1).any(axis=1)
        self.labels = diffs.astype(np.float32)
        # Pre-compute unified action indices
        self.action_idx = np.zeros(self.N, dtype=np.int64)
        for i in range(self.N):
            aid = int(self.action_id[i])
            if 1 <= aid <= 5:
                self.action_idx[i] = aid - 1  # 0-4
            elif aid == 6:
                x = max(0, min(63, int(self.ax[i])))
                y = max(0, min(63, int(self.ay[i])))
                self.action_idx[i] = 5 + (y * 64 + x)  # 5-4100
            else:
                self.action_idx[i] = 0  # ACTION7 / unknown → treat as ACTION1 (won't dominate)

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        state = frame_to_one_hot(self.s_t[i].astype(np.int64))
        return (
            torch.from_numpy(state),
            torch.tensor(int(self.action_idx[i]), dtype=torch.long),
            torch.tensor(float(self.labels[i]), dtype=torch.float32),
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="jepa_wm/data/trajectories.npz")
    ap.add_argument("--out", default="runs/sg_pretrain/sg_action_model.pt")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--log-every", type=int, default=200)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds = SGDataset(args.data)
    print(f"Dataset: {ds.N} samples, label balance: {ds.labels.mean():.3f}")
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=True, num_workers=0,
        pin_memory=(device.type == "cuda"), drop_last=True,
    )

    model = ActionModel(input_channels=16, grid_size=64).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params / 1e6:.2f}M")
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    def get_batch():
        while True:
            for b in loader:
                yield b
    bgen = get_batch()

    t0 = time.time()
    running = {"loss": 0.0, "acc": 0.0}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # Class-balanced positive weight: counters the 78.8% positive bias in BFS-replay
    # data. Without this the CNN learns to predict "1 for everything" → catastrophic
    # on Kaggle (v36 = 0.01).
    pos_frac = float(ds.labels.mean())
    pos_weight = torch.tensor([(1.0 - pos_frac) / max(pos_frac, 1e-6)], device=device)
    print(f"pos_frac={pos_frac:.3f}  pos_weight={pos_weight.item():.3f}")

    for step in range(args.steps):
        states, action_idx, labels = next(bgen)
        states = states.to(device, non_blocking=True)
        action_idx = action_idx.to(device)
        labels = labels.to(device)
        logits = model(states)  # (B, 4101)
        selected = logits.gather(1, action_idx.unsqueeze(1)).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(selected, labels, pos_weight=pos_weight)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            acc = ((torch.sigmoid(selected) > 0.5) == labels.bool()).float().mean()
        running["loss"] += float(loss.item())
        running["acc"] += float(acc.item())
        if (step + 1) % args.log_every == 0:
            n = args.log_every
            print(f"step={step+1:6d} loss={running['loss']/n:.4f} "
                  f"acc={running['acc']/n:.4f} elapsed={time.time()-t0:.0f}s")
            running = {"loss": 0.0, "acc": 0.0}

    torch.save({"model": model.state_dict(), "cfg": {"input_channels": 16, "grid_size": 64}}, args.out)
    sz = Path(args.out).stat().st_size / 1024 / 1024
    print(f"Saved {args.out} ({sz:.1f} MB)")


if __name__ == "__main__":
    main()
