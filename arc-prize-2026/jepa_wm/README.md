# JEPA World Model for ARC-AGI-3

Goal: a learned latent-space world model trained on ARC-AGI-3 trajectory data,
queried by MCTS rollouts for action selection. Targets the public-FORGE ceiling
(~0.25) by replacing reactive CNN with model-based planning — Tufa-class machinery.

## Directory layout

```
jepa_wm/
├── data/         Trajectory generation, augmentation, sharding
├── models/       JEPA encoder, target encoder (EMA), action-conditioned predictor, reward/value heads
├── training/     Offline pretrain loop, EMA scheduler, masking strategies
├── inference/    Per-game online fine-tune, MCTS-over-rollouts planner
├── configs/      Architecture/training/inference hyperparameter files
└── README.md     This file
```

## Architecture sketch (refined after research lands)

- **Encoder f_θ(frame) → z**: tokenize 64×64 categorical grid; small transformer (≤50M params for fast rollout).
- **Target encoder f_ξ**: EMA of f_θ (no gradient).
- **Action embedding**: simple actions ACTION1-5 + click(x,y) discretized.
- **Predictor g_φ(z, a) → ẑ_next**: action-conditioned head; predicts in LATENT space, never decodes pixels.
- **Reward head r_ψ(z, a) → r**: predicts immediate reward / level-up.
- **Value head v_ψ(z) → V**: predicts return-to-win for MCTS leaf evaluation.

## Loss

Smooth-L1 between predictor output and target-encoder embedding of next frame
+ reward prediction loss + value bootstrap (Bellman) loss.

## Training data

- BFS-solved trajectories from local sweeps across all 25 public games.
- Augmentations: color permutation (16-color group), 4× rotations, flips.
- Multi-step prediction (1-step + 4-step rollouts) for stable dynamics.

## Inference (per-game online)

1. Load pretrained weights from Kaggle dataset.
2. Per game: continue training f_θ + predictor on live (s, a, s', r) tuples (TTT).
3. MCTS: from current state, expand tree by querying predictor only (no env interaction);
   ucb1 with value-head leaf estimates.
4. Take best root action; observe; repeat. ~50-200 simulations per real action.

## Status

- [x] Project scaffolded (2026-05-24)
- [ ] Research agent JEPA design recommendation — IN FLIGHT
- [ ] Encoder + predictor first cut
- [ ] Offline training pipeline
- [ ] MCTS planner
- [ ] Agent integration + Kaggle ship
