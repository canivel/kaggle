# v8 Restart: Techniques to Scale ARC-AGI-3 from 0.27 -> 0.40+

**Date:** 2026-04-16
**Context:** Current approach is BFS + MCTS + CNN hybrid at 0.27 ceiling, ~1140s/game budget.
**Constraint:** Kaggle P100 (16 GB VRAM), offline, no API calls, ~12h runtime total.
**Scoring:** RHAE = min(1, human_actions / AI_actions)^2, level-weighted (level 5 = 5x level 1).
**Goal:** 10 concrete, game-agnostic, non-hardcoded directions to brainstorm with KAOS.

---

## Benchmark Designer Intent (ARC-AGI-3)

From arcprize.org/arc-agi/3, the 30-day learnings post, and technical report (arXiv 2603.24621):

- **Observation:** 64x64 grid, 16 colors, optionally multi-frame animations between turns.
- **Action space:** 5 directional keys + Undo + 1 cell-selection (ACTION6, x,y on 64x64).
- **Scoring:** RHAE power-law penalizes inefficient exploration; level-5 weighted 5x.
- **Designers explicitly reward:** (1) environmental learning — perceive what matters, (2) goal discovery on the fly, (3) strategic adaptation across levels, (4) long-horizon planning with sparse feedback. "Intelligence as efficiency" — NOT final correctness alone.
- **What won the preview:** StochasticGoose (12.58%, CNN action-prediction RL, Tufa Labs), Blind Squirrel (6.71%, state graph + ResNet18 value model), Fluxonian (8.04%, DSL+LLM). LLM-only agents crashed and burned (<4%).
- **What WILL NOT work:** pure random search, LLM-heavy prompting per frame (too expensive per action), any game-specific hardcoding.

Sources: [ARC-AGI-3](https://arcprize.org/arc-agi/3) · [30-Day Learnings](https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings) · [Tech Report PDF](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf) · [arXiv 2603.24621](https://arxiv.org/abs/2603.24621)

---

## 10 Concrete Directions

### 1. DreamerV3-style Latent World Model + Imagination Rollouts

**Paper:** Hafner et al., "Mastering Diverse Domains through World Models" (arXiv 2301.04104, DreamerV3).
**Idea:** Replace the CNN value model with a Recurrent State Space Model (RSSM) that learns latent dynamics from agent interaction. Train actor-critic inside imagined latent rollouts (not real env rollouts — 100x faster than MCTS sim).
**ARC-AGI-3 fit:** RSSM compresses the 64x64x16 grid to ~200-dim latent. Symlog reward transform already handles sparse-reward regime (DreamerV3 solved Minecraft diamond from sparse reward, 17 days on V100 from scratch). Single hyperparameter set across 150+ tasks — truly game-agnostic.
**Implementability:** 2 weeks tight but feasible. Reference: `danijar/dreamerv3` repo. Needs tuning for 100k-step budget (ARC-AGI-3 per-game budget). Memory: ~4 GB VRAM for RSSM+actor+critic at small width.
**Expected lift:** DreamerV3 outperforms CNN+RL baselines by 2-5x on sample-efficient control. On ARC-AGI-3, could replace both BFS (use imagination for search) AND CNN value (use critic). Est +0.05-0.10.
**Risk:** RSSM training requires ~50k env steps before behavior emerges. Budget tight.

### 2. IRIS / DART Transformer World Model

**Papers:** Micheli et al., "Transformers are Sample-Efficient World Models" (arXiv 2209.00588, IRIS); "Accurate and Efficient World Modeling with Masked Latent Transformers" (arXiv 2507.04075, DART).
**Idea:** VQ-VAE tokenizer (discrete autoencoder) + autoregressive transformer world model. Agent trained inside imagined tokenized trajectories. Achieved mean human-normalized 1.046 on Atari 100k with ~2h gameplay equivalent — exactly ARC-AGI-3 regime.
**ARC-AGI-3 fit:** 64x64 grid -> ~16x16 token grid via VQ. Transformer predicts next tokens given action. Perfect for discrete pixel grids (ARC is MORE tokenizable than Atari). DART does same with masked latent objective, SOTA without lookahead search.
**Implementability:** 2 weeks moderate. Reference: `eloialonso/iris`. Needs ~8 GB VRAM for small transformer (6 layers, 256 dim). Per-game training ~30min on P100.
**Expected lift:** IRIS beats DreamerV2 on 17/26 Atari 100k games. On ARC-AGI-3 est +0.05-0.08 vs current CNN.
**Risk:** Transformer world model drift on long horizons; mitigate with periodic re-grounding.

### 3. AXIOM Object-Centric Bayesian World Model

**Paper:** Heins et al., "AXIOM: Learning to Play Games in Minutes with Expanding Object-Centric Models" (arXiv 2505.24784, May 2025).
**Idea:** Bayesian model that grows mixture components online from single events; models scenes as compositions of objects with piecewise-linear dynamics. **Masters games in 10k steps, beating DreamerV3 and BBF with a small parameter count and NO gradient descent.** Bayesian model reduction induces generalization.
**ARC-AGI-3 fit:** ARC is LITERALLY object-centric (grid cells, colored shapes). AXIOM's piecewise-linear object dynamics directly model "this sprite moved 1 cell right when I pressed LEFT." Game-agnostic by construction.
**Implementability:** 2 weeks feasible. Reference: `VersesTech/axiom`. No GPU needed for Bayesian updates; CNN segmenter for object extraction can run on CPU or tiny GPU slice.
**Expected lift:** AXIOM converges in first 5k steps on Gameworld-10k benchmark vs ~10k for DreamerV3. On ARC-AGI-3 est +0.08-0.12 — this might be THE killer technique if objectness assumption holds.
**Risk:** Object segmentation on raw 64x64 grids needs tuning; fallback is connected-component labeling (fully symbolic, zero training).

### 4. Go-Explore + Hash-Based Pseudo-Counts

**Papers:** Ecoffet et al., "Go-Explore: a New Approach for Hard-Exploration Problems" (arXiv 1901.10995); Bellemare et al., "Unifying Count-Based Exploration and Intrinsic Motivation" (arXiv 1606.01868); DRND (arXiv 2401.09750, ICML 2024).
**Idea:** Maintain archive of visited cell representations (downsampled grid hash). Sample a cell, "Go" (replay deterministic actions), then "Explore" with random/noisy policy from there. Use hash-based pseudo-counts for intrinsic reward.
**ARC-AGI-3 fit:** ARC-AGI-3 is DETERMINISTIC per level — Go-Explore's determinism assumption holds exactly. Blind Squirrel's 2nd-place preview solution is essentially graph-based Go-Explore. The recent arXiv 2512.24156 "Graph-Based Exploration for ARC-AGI-3" paper ranks 3rd on private leaderboard with pure training-free graph exploration.
**Implementability:** 1 week. Pure Python, no GPU for the archive; optional CNN for cell downsampling.
**Expected lift:** Go-Explore beat Montezuma SOTA by 4x (43k points). arXiv 2512.24156 solved 30/52 levels on preview with NO ML. Est +0.05-0.10 over pure BFS since it avoids loop-backs.
**Risk:** Memory for archive on long games; compress via perceptual hashing.

### 5. Test-Time Training with LoRA Adapters (per-game)

**Papers:** Akyurek et al., "The Surprising Effectiveness of Test-Time Training for Few-Shot Learning" (arXiv 2411.07279); ARC-AGI-2 Tech Report notes TTT with LoRA + task-specific memory; TRM test-time adaptation (arXiv 2511.02886). NVARC (2025 ARC-AGI-2 winner, 24%) used synthetic data + TTT.
**Idea:** Pretrain a small policy/value transformer on synthetic ARC-like games, then at test time: for each novel game, apply per-game LoRA updates using collected (frame, action, delta-frame) trajectories as self-supervised signal. Localized specialization without catastrophic forgetting.
**ARC-AGI-3 fit:** Directly addresses "novelty that prevents memorization." Each game gets its own adapter; 6+ levels per game means enough data to adapt. TTT gave 6x improvement on ARC-AGI-1.
**Implementability:** 2 weeks — hardest on this list. Need synthetic ARC-AGI-3-like game generator (see direction 9). LoRA updates ~100 steps per game fits in 1140s budget on P100 with small base model (<100M params).
**Expected lift:** +0.08-0.15 if synthetic distribution matches. NVARC's 2025 win came from this exact recipe. Highest upside on the list.
**Risk:** Requires synthetic game curriculum; without it, TTT overfits to trivial patterns.

### 6. Latent Action Model from Unsupervised Trajectory Data

**Papers:** Bruce et al., "Genie: Generative Interactive Environments" (arXiv 2402.15391); "Latent Action Pretraining from Videos" LAPA (arXiv 2410.11758, ICLR 2025); Dreamer 4 (arXiv 2509.24527).
**Idea:** Train VQ-VAE latent action model to infer *what action caused* each frame transition from unlabeled trajectories. Then a latent policy picks latent actions; a decoder maps latent -> real action space. Works without ground-truth action labels.
**ARC-AGI-3 fit:** Random exploration early in each game produces (f_t, f_{t+1}) pairs. LAM learns "the true action effect vocabulary" for this game (e.g., "this click teleports sprite," "this key increments counter") — often smaller than the 6-action nominal space. Compresses policy search.
**Implementability:** 2 weeks moderate. Small VQ-VAE (~2M params) + MLP policy. Budget: train LAM first 30% of per-game budget, then use it for planning.
**Expected lift:** +0.04-0.08. More speculative but unifies direction 2/3 approaches.
**Risk:** LAM quality bottlenecks everything downstream.

### 7. Neural-Guided MCTS with Learned Prior + Frame-Delta Value

**Papers:** Schrittwieser et al., MuZero; "Demystifying MuZero Planning" (arXiv 2411.04580); UniZero (2024 planning with latent world models); EfficientZero V2 (arXiv 2403.00564, March 2024, SOTA on Atari 100k).
**Idea:** Replace current hand-tuned MCTS with MuZero-style planning: learned dynamics f(s,a)->s', learned reward r(s,a), policy prior pi(a|s), value v(s). All planning in latent space — 100x faster than re-simulating the env. EfficientZero V2 achieves super-human Atari in 2h real-time.
**ARC-AGI-3 fit:** You already have MCTS — swap hand-written sim for learned dynamics model. Current CNN becomes the value head; add policy head for prior. Tree depth can increase 10x because latent rollouts are cheap.
**Implementability:** 1.5 weeks — you already own the MCTS half. Reference: `opendilab/LightZero` has clean MuZero + UniZero impls. Memory: ~6 GB VRAM.
**Expected lift:** +0.05-0.08. EfficientZero V2 beats BBF/DreamerV3 on 50/66 benchmarks under limited data.
**Risk:** Learned dynamics error compounds; mitigate with periodic env-grounding.

### 8. RND + Ensemble Disagreement for Intrinsic Reward

**Papers:** Burda et al., "Exploration by Random Network Distillation" (arXiv 1810.12894); DRND (ICML 2024, arXiv 2401.09750).
**Idea:** Fixed random target network f*(s); learned predictor f_theta(s); intrinsic reward = ||f_theta(s) - f*(s)||^2. Novel states -> high prediction error -> high reward. DRND extends to distributional RND, which implicitly counts and beats RND on hard-exploration Atari.
**ARC-AGI-3 fit:** Sparse extrinsic reward (only at level complete / progress) + dense intrinsic from RND = much more learning signal per step. Zero hardcoding — RND is game-agnostic.
**Implementability:** 3 days. Two small CNNs (~1M params each). Can be bolted onto current hybrid as an exploration bonus in your MCTS UCB formula.
**Expected lift:** +0.03-0.06 standalone; more when combined with direction 1/7. RND was the first method to exceed human mean on Montezuma.
**Risk:** Reward noise; scale carefully vs extrinsic.

### 9. Synthetic Game Generator + Offline Pre-Training

**Papers:** NVARC 2025 report (synthetic data + TTT on 4B model, 24% on ARC-AGI-2); "Assessing Adaptive World Models with Novel Games" (arXiv 2507.12821); Procgen (arXiv 1912.01588) + improvements (arXiv 2410.10905).
**Idea:** Procedurally generate thousands of ARC-AGI-3-style games (random object types, random action->effect rules, random goals). Pretrain a world model + policy on this distribution OFFLINE. Then at test time, TTT adapts per-game. This is the recipe that won ARC Prize 2025.
**ARC-AGI-3 fit:** The 3 public games give templates; synthesize variations (color permutations, shape changes, rule tweaks). Augmentation-heavy pretraining + TTT was NVARC's winning recipe. Game-agnostic because the generator enforces diversity.
**Implementability:** 3 weeks is tight; start with a simple generator (5 game templates). P100 pretraining in 8-24h.
**Expected lift:** +0.10-0.15 if generator covers test distribution. This is the highest-ceiling direction but highest-effort.
**Risk:** Distribution mismatch — synthetic games too easy / too different.

### 10. Neurosymbolic: Object-Centric Perception + Program Synthesis over Action Sequences

**Papers:** DreamCoder (PLDI 2021, wake-sleep library learning); HYSYNTH (NeurIPS 2024, LLM-guided program synthesis); "Neuro-Symbolic AI in 2024: A Systematic Review" (arXiv 2501.05435); "Modeling Open-World Cognition as On-Demand Synthesis of Probabilistic Models" (arXiv 2507.12547).
**Idea:** (a) Perception: connected-components over colors -> object list with (shape, color, pos, id). (b) Symbolic DSL over actions: sequences like `WHILE player.x < goal.x: RIGHT`. (c) Synthesize programs that explain observed (state, action, next_state) triples. HYSYNTH uses LLM completions as a CFG prior to guide search; DreamCoder grows a library of reusable subroutines across games.
**ARC-AGI-3 fit:** The state graph from Blind Squirrel + a DSL of primitive actions is already half a neurosymbolic agent. Fluxonian's DSL+LLM combo scored 8% in preview. A DreamCoder-style growing library transfers across games (levels N+1 reuses subroutines from level N). Fully game-agnostic if DSL is primitive-only.
**Implementability:** 2-3 weeks. DSL is simple; synthesis via beam search with learned prior. No heavy GPU needed.
**Expected lift:** +0.06-0.10. DreamCoder's library learning is THE mechanism for multi-level compositional reasoning ARC-AGI-3 explicitly rewards.
**Risk:** DSL design bias; keep it minimal and action-primitive-only to stay non-hardcoded.

---

## Ranking for KAOS Brainstorm

**Tier S (highest expected lift, high confidence):**
- **Direction 3 (AXIOM)** — object-centric priors match ARC structure natively, 10k-step convergence
- **Direction 5 (TTT+LoRA)** — this recipe won ARC Prize 2025

**Tier A (high lift, moderate risk):**
- **Direction 1 (DreamerV3)** — proven across 150+ tasks, strong baseline
- **Direction 9 (synthetic pretrain)** — effort-heavy but ceiling is huge
- **Direction 10 (neurosymbolic+DreamCoder)** — unique fit for multi-level compositionality

**Tier B (solid, low-risk additions):**
- **Direction 4 (Go-Explore)** — easy win, provably beats BFS-only (3rd place training-free)
- **Direction 7 (MuZero/EfficientZero V2)** — natural upgrade of current MCTS
- **Direction 8 (RND)** — cheap bolt-on exploration bonus

**Tier C (specialized, combine with others):**
- **Direction 2 (IRIS/DART)** — strong but partly subsumed by 1/7
- **Direction 6 (LAM)** — needs enough data, high-variance

## Hybrid Proposal (KAOS seed)

The v8 restart could combine: **AXIOM-style object-centric world model** (D3) for dynamics, **Go-Explore archive** (D4) for exploration, **RND bonus** (D8) for intrinsic reward, **MuZero-style planning** (D7) in the learned latent, and **TTT LoRA** (D5) for per-game adaptation. All five are game-agnostic and fit Kaggle P100. Start with D3+D4 (lowest risk, fastest path to measuring improvement).

---

## Sources

- [ARC-AGI-3 Benchmark](https://arcprize.org/arc-agi/3)
- [ARC-AGI-3 Technical Report (PDF)](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf) / [arXiv 2603.24621](https://arxiv.org/abs/2603.24621)
- [ARC-AGI-3 30-Day Learnings](https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings)
- [ARC Prize 2025 Results Analysis](https://arcprize.org/blog/arc-prize-2025-results-analysis)
- [ARC Prize 2025 Technical Report arXiv 2601.10904](https://arxiv.org/abs/2601.10904)
- [Graph-Based Exploration for ARC-AGI-3 arXiv 2512.24156](https://arxiv.org/abs/2512.24156)
- [StochasticGoose 1st place writeup](https://medium.com/@dries.epos/1st-place-in-the-arc-agi-3-agent-preview-competition-49263f6287db)
- [StochasticGoose GitHub](https://github.com/DriesSmit/ARC3-solution)
- [DreamerV3 arXiv 2301.04104](https://arxiv.org/abs/2301.04104)
- [Dreamer 4 arXiv 2509.24527](https://arxiv.org/abs/2509.24527)
- [IRIS arXiv 2209.00588](https://arxiv.org/abs/2209.00588)
- [DART arXiv 2507.04075](https://arxiv.org/html/2507.04075)
- [AXIOM arXiv 2505.24784](https://arxiv.org/abs/2505.24784)
- [Go-Explore arXiv 1901.10995](https://arxiv.org/abs/1901.10995)
- [Random Network Distillation arXiv 1810.12894](https://arxiv.org/abs/1810.12894)
- [DRND ICML 2024 arXiv 2401.09750](https://arxiv.org/abs/2401.09750)
- [TTT for Few-Shot Learning arXiv 2411.07279](https://arxiv.org/abs/2411.07279)
- [TRM Test-Time Adaptation arXiv 2511.02886](https://arxiv.org/html/2511.02886)
- [EfficientZero V2 arXiv 2403.00564](https://arxiv.org/html/2403.00564v2)
- [Demystifying MuZero arXiv 2411.04580](https://arxiv.org/abs/2411.04580)
- [V-JEPA 2 arXiv 2506.09985](https://arxiv.org/abs/2506.09985)
- [Genie arXiv 2402.15391](https://arxiv.org/html/2402.15391v1)
- [LAPA arXiv 2410.11758](https://arxiv.org/abs/2410.11758)
- [DreamCoder PLDI 2021](https://dl.acm.org/doi/10.1145/3453483.3454080)
- [HYSYNTH NeurIPS 2024](https://openreview.net/forum?id=5jt0ZSA6Co)
- [Neuro-Symbolic AI 2024 Review arXiv 2501.05435](https://arxiv.org/pdf/2501.05435)
- [Adaptive World Models Novel Games arXiv 2507.12821](https://arxiv.org/abs/2507.12821)
- [WorldLLM arXiv 2506.06725](https://arxiv.org/html/2506.06725)
- [NVARC solution](https://github.com/1ytic/NVARC)
- [Procgen improvements arXiv 2410.10905](https://arxiv.org/html/2410.10905v1)
