# ARC-AGI-3 Literature Review & Applicable Approaches

**Date**: 2026-03-29
**Purpose**: Comprehensive survey of academic papers and approaches relevant to ARC-AGI-3

---

## Table of Contents

1. [ARC-AGI-3 Technical Report](#1-arc-agi-3-technical-report)
2. [Graph-Based Exploration (3rd Place)](#2-graph-based-exploration-3rd-place)
3. [Preview Winner Approach (1st Place)](#3-preview-winner-approach-1st-place)
4. [RL for Interactive Grid Environments](#4-rl-for-interactive-grid-environments)
5. [World Model Approaches](#5-world-model-approaches)
6. [Meta-Learning for Rapid Adaptation](#6-meta-learning-for-rapid-adaptation)
7. [Intrinsic Motivation & Exploration](#7-intrinsic-motivation--exploration)
8. [Object-Centric Representations](#8-object-centric-representations)
9. [Program Synthesis & Inductive Logic](#9-program-synthesis--inductive-logic)
10. [2026 Papers Specifically About ARC-AGI-3](#10-2026-papers-specifically-about-arc-agi-3)
11. [Synthesis: Recommended Architecture](#11-synthesis-recommended-architecture)

---

## 1. ARC-AGI-3 Technical Report

**Paper**: "ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence"
**ArXiv**: 2603.24621 | **Authors**: Chollet et al. (ARC Prize Foundation) | **Venue**: ICLR 2026

### What Makes ARC-AGI-3 Hard

ARC-AGI-3 is the first **interactive** reasoning benchmark. Unlike ARC-AGI-1/2 (static grid puzzles), agents must:
- **Explore** environments with no instructions, no rules, no stated goals
- **Infer goals** from sparse feedback (only level completion signals)
- **Build world models** of novel environment dynamics on the fly
- **Plan** effective action sequences and adapt as complexity increases across levels

### Technical Specifications

| Spec | Value |
|------|-------|
| Grid size | 64x64 pixels |
| Color palette | 16 colors |
| Action space | 5 directional keys + Undo + Grid cell click (coordinate on 64x64) |
| Environments | 25 public + 55 semi-private + 55 private = 135 total |
| Levels per env | Minimum 6, increasing difficulty |
| Time budget | 8 hours per game in competition |
| Step budget | ~100K steps per game |

### RHAE Metric (Relative Human Action Efficiency)

```
Per-level score:  S(l,e) = min(1.0, h(l,e) / a(l,e))^2
Environment score: E(e) = sum(l * S(l,e)) / (n*(n+1)/2)   [linear level weighting]
Total score:       T = (1/|D|) * sum(E(e))
```
- h(l,e) = second-best human action count (baseline)
- a(l,e) = agent's action count
- Power-law (squared) heavily penalizes inefficiency
- Later levels weighted more heavily (level 5 = 5x weight of level 1)

### Core Knowledge Priors (ONLY these are used)

- **Objectness**: Coherent entities with persistence and collision
- **Basic geometry/topology**: Symmetries, rotations, connectedness
- **Basic physics**: Gravity, momentum, bouncing
- **Agentness**: Recognition of intentional goal-pursuing objects
- **Exclusion**: No language, numbers, letters, cultural symbols

### Frontier Model Performance (March 2026)

| Model | RHAE Score |
|-------|-----------|
| Gemini 3.1 Pro Preview | 0.37% |
| GPT 5.4 (High) | 0.26% |
| Opus 4.6 (Max) | 0.25% |
| Grok-4.20 (Beta) | 0.00% |
| StochasticGoose (CNN-RL, preview) | 12.58% |
| Humans | 100% |

### Key Insight for Our Approach

The paper identifies **context management** as the central challenge: "Environment frames are 64x64 grids, and maintaining a naive rolling window exhausts a model's context budget." Random policies validated to win less than 1 in 10,000 times on non-tutorial levels. The 8x larger human-AI gap vs ARC-AGI-1 confirms this requires fundamentally new approaches.

### Generational Comparison

| | ARC-AGI-1 | ARC-AGI-2 | ARC-AGI-3 |
|--|-----------|-----------|-----------|
| Format | Static grid pairs | Static grids (complex) | Interactive turn-based |
| Human time | ~30 sec/task | ~300 sec/task | ~8 min median |
| Grid | Up to 30x30 | Up to 30x30 | Fixed 64x64 |
| Colors | 10 | 10 | 16 |
| Metric | Accuracy (%) | Accuracy (%) | RHAE (action efficiency) |
| Frontier perf | <1% (2024) | 24% (2025) | <1% (2026) |

---

## 2. Graph-Based Exploration (3rd Place)

**Paper**: "Graph-Based Exploration for ARC-AGI-3 Interactive Reasoning Tasks"
**ArXiv**: 2512.24156 | **Authors**: Rudakov, Shock, Cowley
**Code**: https://github.com/dolphin-in-a-coma/arc-agi-3-just-explore

### Core Idea

Training-free, systematic state-space exploration using a directed graph of observed states and transitions. No neural networks, no learning -- pure algorithmic exploration.

### Architecture

**Frame Processor:**
- Segments frames into single-color connected components
- Detects and masks status bars (non-game UI elements)
- Assigns priority tiers to segments (5 tiers based on button likelihood: size, color salience)
- Hashes frames for fast state comparison

**Level Graph Explorer:**
- Maintains directed graph: nodes = unique frame hashes, edges = actions
- Tracks which actions have been tested from each state
- Frontier management: identifies unexplored transitions
- Priority-based action selection: high-priority actions first, only descends to lower tiers when upper exhausted
- Distance computation: preferentially navigates to states nearest unexplored frontiers

### Results

- Median 30/52 levels across 6 games (preview)
- 12/25 private levels in official eval; post-bugfix median 17 levels (range 14-19)
- 3rd place, substantially outperforming all frontier LLM-based agents

### Applicability to ARC-AGI-3

**High relevance** - this is a direct competitor approach. Key insight: systematic state tracking matters more than model size. The approach degrades with large state spaces and non-deterministic environments.

### Implementation Complexity

**Low** (1-2 weeks). Pure Python, no ML training required. Already open-source. Could serve as a strong baseline and complementary module for hybrid approaches.

### Limitations Noted by Authors

- Degrades when status bars integrate into scenes
- Struggles with large state spaces (e.g., FT09 levels 3-4)
- Cannot handle non-deterministic environments
- Authors suggest world model learning as natural extension

---

## 3. Preview Winner Approach (1st Place)

**Author**: Dries Smit (StochasticGoose / Tufa Labs)
**Blog**: https://medium.com/@dries.epos/1st-place-in-the-arc-agi-3-agent-preview-competition-49263f6287db

### Core Idea

CNN-based frame change prediction: learn which actions change the game state, then explore efficiently by preferring state-changing actions.

### Architecture

- **4-layer CNN** encoding 64x64 frames (one-hot 16 channels)
- **Simple action head**: predicts P(frame changes) for 5 simple actions (RESET, ACTION1-4)
- **Spatial coordinate decoder**: for ACTION6 (click), spatially-aware decoding predicting click coordinates
- **Training**: BCE loss on frame change prediction (supervised, NOT RL)
- **Memory**: all frame transitions stored, hash-based deduplication
- **Buffer resets** between levels to prevent confusion

### Results

- Score: 12.58% RHAE
- Highest levels completed and most games completed in preview
- 18 levels total across previously unseen games

### Key Insights

- **Supervised BCE >> RL/PPO** for this problem domain
- This is "informed random search" -- not truly solving, but exploring smarter
- Did not implement frame segmentation (2nd place used it successfully)
- Click-only games benefit most from CNN spatial reasoning
- 8-hour budget gives ~200K+ actions; our 3-min budget only solves tutorial levels

### Applicability

**Very high** -- we already reproduced this (Exp 0003, 10/183 levels in 75 min). The approach is our current baseline. Need longer time budgets and potential improvements.

### Implementation Complexity

**Already done** (run_iter2_winner.py). Improvements: frame segmentation (~1 week), persistent model across levels (~days), spatial decoder refinements (~1 week).

---

## 4. RL for Interactive Grid Environments

### 4a. Craftax (ICML 2024 Spotlight)

**Paper**: "Craftax: A Lightning-Fast Benchmark for Open-Ended Reinforcement Learning"
**ArXiv**: 2402.16801

**Core idea**: JAX-based rewrite of Crafter (Crafter + NetHack) running 250x faster than Python-native. PPO with 1B steps finishes in under 1 hour on single GPU.

**Applicability**: Craftax is the closest existing benchmark to ARC-AGI-3 in spirit (open-ended, discovery-based). Its JAX environment architecture could inspire faster ARC-AGI-3 simulation. However, Craftax has fixed rules -- ARC-AGI-3 requires learning novel rules per environment.

**Complexity**: Medium (2-3 weeks to adapt Craftax-style architecture).

### 4b. XLand-MiniGrid (NeurIPS 2024)

**Paper**: "XLand-MiniGrid: Scalable Meta-Reinforcement Learning Environments in JAX"
**ArXiv**: 2312.12044

**Core idea**: JAX-accelerated meta-RL environments inspired by DeepMind's XLand. Millions of unique tasks, varying difficulty. Reaches millions of steps/second during training.

**Applicability**: **High** -- XLand-MiniGrid is designed for exactly the kind of meta-RL that ARC-AGI-3 requires: training an agent that can adapt to novel grid-based tasks. The procedural task generation is directly analogous to ARC-AGI-3's diverse environments. Pre-training on XLand-MiniGrid tasks could give strong initialization.

**Complexity**: Medium-High (3-4 weeks). Need to adapt observation space (64x64x16 vs MiniGrid's 7x7) and action space.

### 4c. MiniHack / NetHack Learning Environment

**Core idea**: Rich procedural grid environments based on NetHack. Complex dynamics requiring exploration, item interaction, and strategic planning.

**Applicability**: Medium. Demonstrates that RL agents can learn complex grid dynamics, but NetHack environments are orders of magnitude more complex than ARC-AGI-3.

**Complexity**: High (4+ weeks). Heavy engineering overhead.

---

## 5. World Model Approaches

### 5a. DreamerV3 (Nature 2025)

**Paper**: "Mastering Diverse Domains through World Models"
**ArXiv**: 2301.04104

**Core idea**: Learn a latent dynamics model (RSSM), then train a policy entirely in "imagination" (rollouts in the learned model). Single set of hyperparameters works across 150+ tasks including Minecraft, Atari, continuous control.

**Key components**:
- Encoder: CNN or MLP maps observations to latent states
- Dynamics model: Recurrent State-Space Model (RSSM) predicts next latent + reward
- Policy: Actor-critic trained purely on imagined rollouts
- Normalization tricks for stable cross-domain learning

**Applicability to ARC-AGI-3**: **High**. DreamerV3 could learn environment dynamics from the 64x64 grid observations and train a policy in imagination. The challenge is that each ARC-AGI-3 environment has DIFFERENT dynamics, so the world model must be learned online from scratch per environment (or meta-learned).

**Implementation complexity**: Medium-High (3-4 weeks). Well-documented codebase exists. Main challenge: adapting to 64x64 discrete grid with 16-color palette, handling the per-environment novelty.

### 5b. IRIS / Delta-IRIS (ICLR 2023 / ICML 2024)

**Paper**: "Transformers are Sample-Efficient World Models" / "Efficient World Models with Context-Aware Tokenization"
**ArXiv**: 2209.00588 / 2406.19320

**Core idea (IRIS)**: Discrete autoencoder (VQ-VAE style) tokenizes frames into discrete tokens. Autoregressive Transformer predicts next frame tokens + reward. Policy trained on imagined rollouts.

**Core idea (Delta-IRIS)**: Instead of encoding full frames, encodes **deltas between consecutive frames**. Order of magnitude faster training. State-of-the-art on Crafter.

**Applicability to ARC-AGI-3**: **Very High**. Delta-IRIS is particularly relevant because:
1. ARC-AGI-3 environments often have sparse changes between frames (most of the grid stays the same)
2. The discrete tokenization naturally maps to the 16-color grid
3. Frame-change detection (winner's approach) is essentially a simplified version of delta prediction
4. The autoregressive Transformer can capture complex multi-step dynamics

**Implementation complexity**: Medium (2-3 weeks). IRIS code is open-source. Delta-IRIS adaptation for 64x64 discrete grids is natural -- the 16-color palette could even BE the codebook directly.

### 5c. Improved Transformer World Models (ICML 2025)

**Paper**: "Improving Transformer World Models for Data-Efficient RL"
**ArXiv**: 2502.01591

**Core idea**: Three improvements to IRIS-style models:
1. **Dyna with warmup**: delay imaginary rollouts until world model is sufficiently trained
2. **Nearest neighbor tokenizer**: static codebooks after initialization
3. **Block teacher forcing**: predict multiple future tokens jointly

**Results**: 69.66% on Craftax-classic after only 1M steps (vs DreamerV3's 53.2%, vs human 65.0%). Superhuman with dramatically fewer samples.

**Applicability to ARC-AGI-3**: **Very High**. The Dyna-with-warmup idea directly addresses ARC-AGI-3's challenge: first explore to build a good world model, THEN exploit it for efficient solving. The static nearest-neighbor tokenizer maps perfectly to a 16-color grid.

**Implementation complexity**: Medium (2-3 weeks on top of IRIS implementation).

---

## 6. Meta-Learning for Rapid Adaptation

### 6a. MAML / Reptile (Foundational)

**MAML** (Finn et al., 2017): Learn initialization parameters such that a few gradient steps on a new task produce good performance. Requires second-order gradients (expensive).

**Reptile** (Nichol et al., 2018): First-order approximation of MAML. Repeatedly: sample task, run SGD, move initialization toward final parameters. Much cheaper.

**Applicability to ARC-AGI-3**: **Medium-High**. Could meta-learn a world model or policy initialization across diverse training environments, then fine-tune quickly on each ARC-AGI-3 environment. Challenge: we need diverse training environments that share structure with ARC-AGI-3 tasks.

**Implementation complexity**: Low-Medium (1-2 weeks). Both are well-understood algorithms with many implementations.

### 6b. PEARL (Rakelly et al., 2019)

**Core idea**: Off-policy meta-RL with probabilistic context variables. A permutation-invariant encoder takes transitions (s, a, r, s') and produces a Gaussian posterior over latent task variables. Policy conditions on sampled task variable for structured exploration.

**Key advantage**: 20-100x fewer samples during meta-training than prior meta-RL. Posterior sampling enables hypothesis-driven exploration.

**Applicability to ARC-AGI-3**: **High**. PEARL's context encoder is exactly the mechanism needed to infer environment dynamics from a few interactions. The posterior sampling exploration strategy maps to ARC-AGI-3's need for hypothesis testing.

**Implementation complexity**: Medium (2-3 weeks). Established codebase. Challenge: scaling context encoder to 64x64 visual observations.

### 6c. AMAGO-2 (NeurIPS 2024)

**Paper**: "AMAGO-2: Breaking the Multi-Task Barrier in Meta-Reinforcement Learning with Transformers"
**ArXiv**: 2411.11188

**Core idea**: Transformer-based meta-RL that converts actor AND critic objectives into classification (decoupling from return scales). Off-policy RL^2 on long sequences. Memory-based task inference.

**Key innovation**: By classifying returns instead of regressing them, the agent handles diverse reward scales across tasks without normalization.

**Results**: State-of-the-art across 5 diverse benchmarks: Meta-World ML45, Procgen, POPGym, Atari, BabyAI.

**Applicability to ARC-AGI-3**: **Very High**. AMAGO-2 is arguably the closest existing architecture to what ARC-AGI-3 needs:
- Transformer memory for in-context task inference
- Multi-task training across diverse environments
- Classification-based objectives handle ARC-AGI-3's varying reward structures
- Off-policy training enables sample-efficient exploration
- Tested on BabyAI (grid-based!)

**Implementation complexity**: High (4-6 weeks). Complex architecture, but code is open-source (github.com/UT-Austin-RPL/amago).

### 6d. DICP - Distillation for In-Context Planning (ICLR 2025)

**Paper**: "Distilling Reinforcement Learning Algorithms for In-Context Model-Based Planning"
**ArXiv**: 2502.19009

**Core idea**: Transformer simultaneously learns environment dynamics AND improves policy in-context (no parameter updates at test time). Model-based planning in-context overcomes suboptimal behavior inherited from imitated algorithms.

**Applicability to ARC-AGI-3**: **High**. The in-context learning paradigm directly matches ARC-AGI-3's requirement: adapt to novel environments without updating parameters. The model-based planning component enables strategic action rather than reactive behavior.

**Implementation complexity**: Medium-High (3-4 weeks).

---

## 7. Intrinsic Motivation & Exploration

### 7a. Random Network Distillation (RND)

**Core idea**: Fixed random target network + trained predictor. Intrinsic reward = prediction error (high for novel states, low for familiar ones). Extremely simple to implement.

**Applicability**: **Medium**. Could complement the frame-change prediction approach. Novel states get explored more. However, in ARC-AGI-3, "novel" doesn't always mean "useful."

**Complexity**: **Very Low** (days). Drop-in module.

### 7b. Change-Based Exploration Transfer (CBET)

**Paper**: "World Model Agents with Change-Based Intrinsic Motivation"

**Core idea**: Intrinsic reward based on whether actions cause observable changes in the environment state. Adapted for world model algorithms like DreamerV3.

**Applicability to ARC-AGI-3**: **Very High**. This is essentially the theoretical framework behind the preview winner's approach (frame change prediction = CBET). Integrating CBET with a proper world model (DreamerV3 or IRIS) could combine the winner's exploration strategy with model-based planning.

**Results**: CBET improves DreamerV3 returns in Crafter but achieves suboptimal policies in MiniGrid (suggesting environment-dependent).

**Complexity**: Low-Medium (1-2 weeks on top of world model).

### 7c. SENSEI - Semantic Exploration (ICML 2025)

**Core idea**: Uses Vision-Language Model to generate semantic reward signals, which are distilled into a world model for exploration policy training.

**Applicability**: **Low** for ARC-AGI-3 (no internet access allowed in submission, can't use VLM at inference). However, the principle of semantic-guided exploration could be implemented with a smaller learned module.

### 7d. Random Distribution Distillation (RDD, 2025)

**Core idea**: Unifies count-based and prediction-error exploration by sampling target network outputs from a normal distribution. More extensive exploration than RND.

**Applicability**: **Medium**. Improvement over RND with similar implementation ease.

**Complexity**: **Very Low** (days).

---

## 8. Object-Centric Representations

### 8a. OC-STORM (2025)

**Paper**: "Object-Centric World Models from Few-Shot Annotations"
**ArXiv**: 2501.16443

**Core idea**: Enhance world model (STORM) with object representations from pretrained segmentation network. Few-shot annotated frames train object tracking. Significantly outperforms baselines on Atari 100k and Hollow Knight.

**Applicability to ARC-AGI-3**: **High**. ARC-AGI-3 environments are inherently object-centric (colored blocks, buttons, agents). The 3rd-place approach already does connected-component segmentation. A learned object-centric world model could:
1. Segment the 64x64 grid into objects (connected components of same color)
2. Model object-level dynamics (object X moves when action Y is taken)
3. Plan at the object level (move object to target position)

**Complexity**: Medium-High (3-4 weeks). Need segmentation adapted for 16-color discrete grids.

### 8b. FIOC-WM (2025)

**Paper**: "Learning Interactive World Model for Object-Centric RL"
**ArXiv**: 2511.02225

**Core idea**: Factored world model that explicitly models object interactions as composable primitives. Hierarchical policy selects interaction type, then executes.

**Applicability**: **Medium**. The interaction primitive decomposition maps to ARC-AGI-3's action types, but the framework is designed for robotic manipulation, not grid worlds.

**Complexity**: High (4+ weeks adaptation).

### 8c. STICA - Slot Transformer (2025)

**Paper**: "Object-Centric World Models for Causality-Aware RL"
**ArXiv**: 2511.14262

**Core idea**: Observations decomposed into object-centric "slot" tokens. Transformer world model predicts token-level dynamics. Causality-aware.

**Applicability to ARC-AGI-3**: **High**. Slot attention maps naturally to ARC-AGI-3's color-based objects. Causality-aware dynamics learning aligns with ARC-AGI-3's requirement to infer environment rules.

**Complexity**: High (4-6 weeks).

---

## 9. Program Synthesis & Inductive Logic

### 9a. ILP for ARC (Rocha et al., 2025)

**Paper**: "Program Synthesis Using Inductive Logic Programming for the Abstraction and Reasoning Corpus"

**Core idea**: Solve ARC with Inductive Logic Programming (POPPER system) using object-centric domain-specific language.

**Applicability to ARC-AGI-3**: **Low-Medium**. ARC-AGI-3 is interactive, not static. However, the idea of learning rules as programs could apply to modeling environment dynamics symbolically.

**Complexity**: High (4+ weeks). Fundamentally different paradigm.

---

## 10. 2026 Papers Specifically About ARC-AGI-3

### Currently Published (as of 2026-03-29)

1. **ARC-AGI-3 Technical Report** (2603.24621) - Chollet et al., ICLR 2026. The benchmark paper itself. Published March 24, 2026.

2. **Graph-Based Exploration** (2512.24156) - Rudakov et al. 3rd place preview approach. Published December 2025.

3. **The ARC of Progress towards AGI: A Living Survey** (2603.13372) - Comprehensive survey covering ARC-AGI-1 through 3. Published March 2026. Key finding: 8x larger human-AI gap on ARC-AGI-3 vs ARC-AGI-1.

### Not Yet Published (Competition Just Launched March 25, 2026)

The main competition launched only 4 days ago (March 25, 2026). No new agent-specific papers have been published yet for the main competition. The preview competition papers (StochasticGoose 1st place, Blind Squirrel 2nd, graph-based 3rd) remain the only documented approaches.

**Expected wave of papers**: Likely after Milestone 1 (June 30, 2026) when first competition results are in.

---

## 11. Synthesis: Recommended Architecture

### The Landscape

The research clearly points to a **hybrid approach** combining multiple techniques:

| Approach | Solves What Problem | Priority |
|----------|-------------------|----------|
| Frame change prediction (winner) | Efficient exploration | Already implemented |
| Graph-based exploration (3rd place) | Systematic state tracking | High - 1-2 weeks |
| Delta-IRIS world model | Learn environment dynamics | High - 2-3 weeks |
| CBET intrinsic motivation | Direct exploration toward changes | Medium - 1-2 weeks |
| Object-centric segmentation | Structured state representation | Medium - 2-3 weeks |
| AMAGO-2 meta-RL | Cross-environment transfer | Lower - 4-6 weeks |
| MAML/Reptile initialization | Fast adaptation | Medium - 1-2 weeks |

### Proposed Architecture (Phased)

**Phase 1: Improve Baseline (Now - April)**
- Increase time budget (RunPod A40, 30+ min per game)
- Add connected-component segmentation (from 3rd place approach)
- Integrate graph-based state tracking alongside CNN prediction
- Estimated improvement: 10-20% RHAE (from current ~1%)

**Phase 2: World Model (April - May)**
- Implement Delta-IRIS style world model for 64x64 grids
  - Natural fit: 16-color palette = 16 codebook entries (no VQ-VAE needed)
  - Delta encoding captures sparse frame changes
  - Transformer predicts dynamics
- Add CBET intrinsic motivation on top of world model
- Train policy in imagination (Dyna with warmup)
- Estimated improvement: 20-40% RHAE

**Phase 3: Meta-Learning (May - June, for Milestone 1)**
- Pre-train on public ARC-AGI-3 environments
- Reptile-style meta-initialization for world model
- PEARL-style context encoder for rapid task inference
- Estimated improvement: 40-60% RHAE

**Phase 4: Scaling (June - September, for Milestone 2)**
- AMAGO-2 style full meta-RL architecture
- Object-centric slot attention world model
- Hierarchical planning (object-level + action-level)
- Target: 60-85% RHAE

### Key Technical Decisions

1. **World model type**: Delta-IRIS > DreamerV3 for this domain (discrete grid, sparse changes, natural tokenization)
2. **Exploration**: CBET + graph-based hybrid (change detection for online learning, graph for systematic coverage)
3. **Representation**: Object-centric via connected components (simple, fast, fits domain perfectly)
4. **Meta-learning**: Start with Reptile (simple), upgrade to AMAGO-2 if meta-training data is sufficient
5. **No LLMs at inference**: All top approaches confirm LLMs fail on this task. Pure visual RL/search wins.

---

## Sources

### Primary Papers
- [ARC-AGI-3 Technical Report (2603.24621)](https://arxiv.org/abs/2603.24621)
- [Graph-Based Exploration (2512.24156)](https://arxiv.org/abs/2512.24156)
- [3rd Place Code](https://github.com/dolphin-in-a-coma/arc-agi-3-just-explore)
- [1st Place Blog Post](https://medium.com/@dries.epos/1st-place-in-the-arc-agi-3-agent-preview-competition-49263f6287db)
- [ARC-AGI-3 Preview 30-Day Learnings](https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings)
- [The ARC of Progress: Living Survey (2603.13372)](https://arxiv.org/abs/2603.13372)

### World Models
- [DreamerV3 (Nature 2025)](https://arxiv.org/abs/2301.04104)
- [IRIS: Transformers as World Models (ICLR 2023)](https://arxiv.org/abs/2209.00588)
- [Delta-IRIS: Context-Aware Tokenization (ICML 2024)](https://arxiv.org/abs/2406.19320)
- [Improved Transformer World Models (ICML 2025)](https://arxiv.org/abs/2502.01591)

### Grid RL Environments
- [Craftax (ICML 2024 Spotlight)](https://arxiv.org/abs/2402.16801)
- [XLand-MiniGrid (NeurIPS 2024)](https://arxiv.org/abs/2312.12044)
- [MiniHack](https://github.com/facebookresearch/minihack)

### Meta-Learning
- [AMAGO-2 (NeurIPS 2024)](https://arxiv.org/abs/2411.11188)
- [DICP: In-Context Planning (ICLR 2025)](https://arxiv.org/abs/2502.19009)
- [PEARL](https://arxiv.org/abs/1903.08254)
- [CRAFT: Context Representation for Meta-RL](https://arxiv.org/abs/2512.14057)

### Exploration & Intrinsic Motivation
- [Random Distribution Distillation (2025)](https://arxiv.org/abs/2505.11044)
- [GNN-based Intrinsic Rewards (Nature Scientific Reports 2025)](https://www.nature.com/articles/s41598-025-23769-3)

### Object-Centric RL
- [OC-STORM (2025)](https://arxiv.org/abs/2501.16443)
- [FIOC-WM (2025)](https://arxiv.org/abs/2511.02225)
- [STICA (2025)](https://arxiv.org/abs/2511.14262)

### Competition & Leaderboard
- [ARC Prize 2026 Competition](https://arcprize.org/competitions/2026/arc-agi-3)
- [ARC-AGI-3 Benchmark](https://arcprize.org/arc-agi/3)
- [Kaggle Leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard)
- [ARC-AGI-3 Docs](https://docs.arcprize.org/)
