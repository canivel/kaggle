# JEPA World Model — Design Spec (locked 2026-05-24)

Research agent's full recommendation. We build to this spec. Updates as we learn.

## Architecture (~35M params)

### Tokenizer
- 64×64 grid of 16 colors → per-cell `Embedding(16, d=256)` → 4×4 patch ViT → 256 tokens
- 2D RoPE positional encoding (V-JEPA-2 found rotary > sincos)
- Special tokens: `[ACT]`, `[REW]`, `[CLS]`
- **Action embedding**: 7 action types; for ACTION6, sequence is `[ACT6, x_emb, y_emb]` where x,y are 64-way categorical embeddings (NOT continuous coords)

### Encoder `E_θ`
- ViT-S: 12 layers, d=256, 8 heads, MLP×4 → ~22M params

### Target encoder `Ē`
- Identical to `E_θ`, EMA-updated (no grad)
- Momentum schedule: `m: 0.996 → 1.000` linear over training (I-JEPA pattern, arXiv:2301.08243)

### Predictor `P_φ` (the dynamics core)
- 8 layers, d=256, 8 heads → ~8M params (V-JEPA pattern: predictor << encoder)
- Sequence: `[state_tokens(256)] [ACT, action_emb, x_emb, y_emb] [mask_tokens(256)]`
- Causal mask between segments, bidirectional within. A-JEPA / V-JEPA-2-AC pattern (arXiv:2506.09985)

### Heads
- `r_head`: 3-way classifier {0, +1 step, +level-complete} (or symexp two-hot regression à la DreamerV3)
- `done_head`: binary
- `v_head`: scalar value (for MCTS bootstrap)

## Training

### Data
1. Self-play rollouts from current FORGE agent on training games — target 1M transitions
2. Random + ε-greedy exploration (esp. ACTION6 coords)
3. ARC-AGI-1/2 static puzzles as auxiliary state-only pretraining

### Augmentations
- D4 symmetries (8 rotations/reflections) consistently applied to `(s, a, s')`; ACTION6 coords transformed too
- Color permutations (random shuffle of 16-color palette) identical across trajectory
- NO cutout/blur (destroys ARC semantics)

### Loss
```
L = L_jepa + λ_r · L_reward + λ_d · L_done + λ_tf · L_teacher_forcing
L_jepa = || P_φ(s_t, a_t) - sg(Ē(s_{t+1})) ||₁   # L1 in latent space (V-JEPA: L1 > L2)
```
- Add 2-step autoregressive rollout loss (V-JEPA-2-AC) to prevent compounding error
- Teacher-force 80%, free-run 20%

### Schedule
- AdamW, lr=3e-4 cosine, 200K steps, batch 256, seq length 4
- ~6-8 hours on a single A40; pretrain once, ship as Kaggle dataset

## Inference — MCTS in latent space (MuZero/UniZero pattern)

1. Encode current real frame: `z_0 = E_θ(s_real)` — once per real step
2. **64 MCTS simulations** at root `z_0`:
   - Selection: PUCT
   - Expansion: `P_φ(z, a) → z', r̂, v̂, done` — **never decode pixels**
   - Backup with `v̂` (or 0 if done)
3. Batch all leaves at each depth (one forward pass)
4. Rollout depth **12** (matches ARC-AGI-3 typical level length); beyond → expand width
5. Pick action by visit count, execute, **re-encode true s_{t+1}** (don't trust latent dynamics for ground-truth state). Closed-loop MPC.

### Hybrid with BFS
- BFS first 3 levels (fast, deterministic)
- JEPA-MCTS when BFS budget exhausted (>20 steps) — per `feedback_arc_long_bfs_mcts.md`

## Top 3 risks + mitigations

1. **Latent collapse** (classic JEPA failure). Mitigations: EMA m≥0.996, predictor << encoder, VICReg-style variance regularizer on encoder outputs, monitor latent norm + per-dim variance, stop on rank collapse.

2. **Reward signal too sparse** (RHAE fires only on level-complete). Mitigations: dense pseudo-rewards from `||z_{t+1} - z_t||`; pretrain `r_head` on BFS-winning trajectories; RND-style frame-prediction-confidence intrinsic reward.

3. **ACTION6 coord space too large** (64×64 = 4096). Mitigations: (a) factored (x, y) head → 64+64 branching; (b) coordinate-attention prior using encoder attention map → top-K click candidates (top-8 covers >90%); (c) `sel.set_data({"x":x,"y":y})` (per `feedback_arc_set_data_bug.md`).

## Build order

- **Day 1 (today)**: `jepa_wm/models/jepa.py` — tokenizer + encoder + target-EMA + predictor + heads. Synthetic train on 10K transitions. Verify no collapse (variance > 0.01 per dim).
- **Day 2**: `jepa_wm/inference/mcts.py` — latent MCTS with batched expansion. Plug into agent behind flag. A/B vs BFS on levels 4-8.
- **Day 3-4**: Trajectory generation + augmentation pipeline. Bulk training data.
- **Day 5-7**: Full 200K-step training run (RunPod A40 if needed).
- **Day 8-10**: Agent integration, hybrid policy, local sweep.
- **Day 11-14**: Kaggle ship, sweep, iterate.

## Key papers
- I-JEPA: arXiv:2301.08243
- V-JEPA: arXiv:2404.08471
- V-JEPA-2 + V-JEPA-2-AC: arXiv:2506.09985 (PRIMARY template)
- DINO-WM (latent dynamics + planning): arXiv:2411.04983
- Discrete JEPA (VQ tokenization for grids): arXiv:2506.14373
- IRIS (transformer WM + Atari): arXiv:2209.00588
- UniZero (transformer WM + MCTS, KV-cached): arXiv:2406.10667
- DreamerV3 (symexp two-hot reward head, Nature 2025)
- LeCun "Path Toward AMI" (2022) — conceptual JEPA cookbook
