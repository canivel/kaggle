# ARC Prize 2026 (ARC-AGI-3) - Autonomous Research Program

## Goal
Maximize RHAE (Relative Human Action Efficiency) on the ARC-AGI-3 interactive reasoning benchmark.
Current frontier AI best: 0.37% (Gemini 3.1 Pro). Humans: 100%.
Target: Top 5 on leaderboard. Grand prize: $700K for 100%.

## Key Difference from Tabular Competitions
This is NOT a prediction task. It's an **interactive agent** problem:
- 64x64 grid environments with 16 colors
- 7-action space (5 simple + 1 coordinate + 1 undo)
- No instructions, no rules, no stated goals
- Agent must explore, model, set goals, and plan
- RHAE scoring: efficiency relative to human baseline

## Setup
1. Install `arc-agi` SDK: `uv add arc-agi`
2. Ensure kaggle-agent framework: `uv pip install -e ../kaggle-agent`
3. GPU available (RTX 3080 local, A40 on RunPod)
4. All models must be self-contained (no internet during Kaggle eval)

## Research Loop

### NEVER STOP. The human might be asleep. You are autonomous.

```
LOOP FOREVER:
  1. Read experiments/results.tsv to see current best RHAE
  2. Choose next experiment:
     a. If < 3 experiments: run baselines (random, simple heuristic, CNN)
     b. If < 10: train CNN policy variants (architecture, reward shaping)
     c. If < 20: curriculum learning, environment clustering
     d. If >= 20: world models, planning, meta-learning
  3. Run the experiment:
     - Interact with public environments via arc_agi SDK
     - Train agent, evaluate on held-out envs
     - Record: RHAE, levels_completed, actions, training time
  4. Compare to best:
     - If improved: save checkpoint to models/
     - If not: analyze why and log learnings
  5. Log to experiments/results.tsv
  6. If RHAE improves significantly: prepare submission notebook
  7. Continue to next experiment
```

## What to Try (Research Roadmap)

### Phase 1: Foundation (Weeks 1-4)
1. Random baseline (establish floor)
2. CNN policy trained with frame-change detection (preview winner approach)
3. PPO/A2C on frame-change intrinsic reward
4. Environment-specific vs shared policy comparison
5. Action space analysis (which envs use arrows vs clicks)

### Phase 2: Intelligence (Weeks 5-10)
6. World model: predict next frame given action (learn dynamics)
7. Model-based planning with learned world model
8. Curiosity-driven exploration (intrinsic motivation)
9. Goal discovery: identify "winning" states from trajectory patterns
10. Curriculum learning: easy levels first, transfer to harder ones

### Phase 3: Generalization (Weeks 11-16)
11. Meta-learning across environments (MAML, Reptile)
12. Transfer learning: pretrain on many envs, fine-tune on new ones
13. Attention-based architectures (transformer over frame sequences)
14. Hierarchical policies (high-level strategy + low-level actions)
15. Multi-environment ensembles

### Phase 4: Optimization (Weeks 17-20)
16. Action efficiency optimization (minimize actions per level)
17. Undo strategy (when to use action 7)
18. Submission notebook optimization (12-hour budget)
19. Per-environment specialization within runtime budget
20. Final ensemble of best approaches

## Architecture Options
- **CNN-PPO** (baseline): 4-layer ConvNet, works for simple envs
- **CNN-LSTM-PPO**: Add memory for environments with hidden state
- **Vision Transformer + RL**: Better at spatial reasoning
- **World Model (Dreamer-style)**: Learn dynamics, plan in latent space
- **DQN variants**: For environments with small effective action spaces

## Scoring Formula
```
Per-level: S(l,e) = min(1.0, h(l,e) / a(l,e))^2
Per-env: weighted sum with w_l = l (later levels count more)
Overall: average across all environments
```

## Constraints
- 12-hour Kaggle notebook runtime
- No internet during evaluation
- Models must be bundled in submission
- Max 5x human baseline actions per level (capped)
- NEVER use litellm

## Results Format
Tab-separated file: `experiments/results.tsv`
```
experiment_id	timestamp	agent_type	description	rhae_score	levels_completed	total_actions	status	duration_seconds	params	notes
0001	2026-03-29T...	random	random_baseline	N/A	0	1500	completed	120	{}	baseline
```

## Key Insights from Preview Winner
1. Frame change detection is the crucial first step (most random actions are no-ops)
2. CNN architecture: 4 layers, 32->64->128->256 channels
3. Simple RL (PPO) works better than complex approaches for initial progress
4. Environment-specific tuning helps but shared training gives better generalization
5. Undo (action 7) is underutilized but powerful for course-correction
