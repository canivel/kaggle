#!/usr/bin/env bash
# Full JEPA-WM training pipeline: trajectory gen -> training -> final ckpt.
# Run after 8pm when GPU is free. Outputs go to jepa_wm/_pipeline.log.

set -e
cd "$(dirname "$0")/.."  # arc-prize-2026/

LOG="jepa_wm/_pipeline.log"
echo "==== JEPA-WM pipeline START $(date -u +%FT%TZ) ====" | tee -a "$LOG"

# Step 1: trajectory generation (CPU, BFS + random exploration)
if [ ! -f jepa_wm/data/trajectories.npz ]; then
  echo "Step 1: generating trajectories (BFS 180s + 400 random actions/game)..." | tee -a "$LOG"
  uv run python -m jepa_wm.data.gen_trajectories \
    --out jepa_wm/data/trajectories.npz --bfs-timeout 180 --random-actions 400 2>&1 | tee -a "$LOG"
else
  echo "Step 1: trajectories.npz already exists, skipping gen" | tee -a "$LOG"
fi

# Step 2: training (GPU, OOM-safe: batch 32 x grad_accum 4 = effective 128, AMP)
echo "Step 2: training JEPA world model (batch=32 ga=4 AMP)..." | tee -a "$LOG"
RESUME_FLAG=""
if [ -f jepa_wm/checkpoints/jepa_wm_final.pt ]; then
  echo "  final ckpt exists, resuming for additional steps" | tee -a "$LOG"
  RESUME_FLAG="--resume jepa_wm/checkpoints/jepa_wm_final.pt"
fi
uv run python -m jepa_wm.training.train \
  --data jepa_wm/data/trajectories.npz \
  --ckpt jepa_wm/checkpoints \
  --steps 20000 --batch 32 --grad-accum 4 --lr 3e-4 $RESUME_FLAG 2>&1 | tee -a "$LOG"

echo "==== JEPA-WM pipeline END $(date -u +%FT%TZ) ====" | tee -a "$LOG"
