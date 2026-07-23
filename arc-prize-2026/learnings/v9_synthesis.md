# v9 Restart — Synthesis & Direction Proposal

**Date:** 2026-04-16
**Context:** forge_v35 v7 hit 0.27 (best), v6/v8 regressed to 0.18 when capping long-BFS MCTS. Ceiling reached on current BFS+MCTS+CNN hybrid. Starting fresh.
**Constraints:** Kaggle P100 16GB, offline, ~1140s/game (25 games × ~8hr total).

---

## Key Intel

**Preview ARC-AGI-3 leaderboard (validated, non-hardcoded):**
| Approach | Score | Author | Mechanism |
|---|---|---|---|
| CNN action-prediction RL | **12.58%** | StochasticGoose | Learn action→frame-delta mapping |
| Graph exploration + priority tiers | **3rd private** | dolphin-in-a-coma (arXiv 2512.24156) | **NO ML, 30/52 levels solved** |
| DSL + LLM | 8.04% | Fluxonian | Hybrid symbolic |
| State graph + ResNet value | 6.71% | Blind Squirrel | Graph + learned value |
| Our forge_v35 v7 | **27%** | us | BFS + MCTS + CNN |

**⚠️ Critical surprise:** We're already 2x the preview leaders on the public LB, but we've hit a ceiling the proven approaches broke through on private LB. Two signals:
1. **arXiv 2512.24156 (ZERO ML)** — pure frontier-aware graph exploration with priority tiers beat most approaches on private LB. Proves the floor is much higher with a smarter explorer, not a bigger model.
2. **NVARC (ARC-AGI-2 winner, 24%)** — synthetic data + TTT + LoRA. Different comp but same insight: test-time learning > compute.

## 10 Directions (Tier S → C)

### Tier S — Highest EV, game-agnostic, locally validatable

1. **[FLOOR] Graph exploration w/ priority tiers** — Replace BFS with arXiv 2512.24156 frontier-aware explorer. Priority tiers based on segment size, color salience, button likelihood, status-bar masking. Error threshold (close node after 3 failures) eliminates infinite loops. **PROVEN 3rd-place private LB.** Expected +0.05-0.10 over our BFS. Zero ML needed.

2. **[ADD] AXIOM-style object-centric priors** (arXiv 2505.24784) — Segment grid into colored components; track per-object dynamics (did this sprite move when I pressed X?). Prioritize actions that affect new/changing objects. Bayesian updates, no gradient descent. **Nobody in preview tried it. Matches ARC structure natively.** Expected +0.05-0.08.

### Tier A — Solid additions with novel twist

3. **RND novelty bonus** (arXiv 1810.12894) — Tiny random target net + learned predictor. Novel states get higher exploration priority. 3 days of work, +0.02-0.05 additive.

4. **Test-Time-Training with LoRA adapters** (arXiv 2411.07279) — Per-game LoRA on a small policy transformer. Needs synthetic pretrain data. High risk, high ceiling (+0.08-0.15). **NVARC's winning ARC-AGI-2 recipe.**

5. **Neurosymbolic DreamCoder** (PLDI 2021) — Growing DSL library across levels; subroutines reusable. Compositionality is explicit in ARC-AGI-3 scoring (level 5 = 5× level 1 weight). Expected +0.06-0.10.

### Tier B — Strong individual merits, maybe future work

6. **DreamerV3 latent world model** (arXiv 2301.04104) — Replace MCTS with imagination rollouts. 2 weeks, +0.05-0.10.
7. **EfficientZero V2 / MuZero** (arXiv 2403.00564) — Upgrade MCTS to learned dynamics. Natural path. +0.05-0.08.
8. **IRIS transformer world model** (arXiv 2209.00588) — Tokenize grid with VQ-VAE. +0.05-0.08.

### Tier C — Speculative, combine later

9. **Latent Action Model / Genie** (arXiv 2402.15391) — Infer "true" action vocabulary from frames.
10. **Synthetic game generator + pretrain** — Ceiling is huge (+0.10-0.15) but 3-week effort.

---

## Recommended v9 Architecture: "ForgeExplore"

Combine **Tier S-1 (proven floor) + S-2 (novel priors) + A-3 (cheap bonus)** as an **incremental but NOVEL** upgrade to forge_v35:

### Architecture
```
┌─────────────────────────────────────────────┐
│ v9 ForgeExplore: Graph-first exploration    │
├─────────────────────────────────────────────┤
│ Stage 1: Object segmentation                │
│   → connected components by color           │
│   → track (shape, color, pos, id) per frame │
│                                             │
│ Stage 2: Action candidate generation        │
│   → directional keys: always                │
│   → clicks: one per object centroid         │
│   → status bar / UI regions: deprioritized  │
│                                             │
│ Stage 3: Graph exploration (replaces BFS)   │
│   → frame hash = node                       │
│   → priority tiers per candidate:           │
│     T1: novel object changes (AXIOM signal) │
│     T2: high-salience clicks (button-like)  │
│     T3: directional + generic clicks        │
│     T4: UI / status regions                 │
│   → RND bonus: novelty weights ties          │
│   → close node after 3 failures             │
│   → when frontier untestable, advance group │
│                                             │
│ Stage 4: MCTS shortening (keep v7 behavior) │
│   → if graph found solution ≤20 steps:      │
│     90s MCTS cap (v7 proven)                │
│   → if >20 steps: FULL MCTS (v7 validated)  │
│                                             │
│ Stage 5: Level-level skill transfer         │
│   → primitive sequence library across levels│
│     (DreamCoder-lite; purely behavioral)   │
└─────────────────────────────────────────────┘
```

### Why this wins
- **S-1 alone gives +0.05-0.10** (proven). Even if S-2, A-3 don't help, we likely clear 0.30+.
- **S-2 (object-centric priors)** is the novel part — nobody tried object-aware priority tiers on ARC-AGI-3. High ceiling if objects segment cleanly.
- **Keeps v7's MCTS shortening** — preserves our 0.27 baseline capability; only adds upside.
- **Zero training required** — no synthetic data, no LoRA. Fully local-validatable before Kaggle.
- **Tight 2-week implementation** — feasible given deadline 2026-11-02.

### Validation plan (local)
1. Port graph_explorer from arXiv 2512.24156 repo into forge framework (1 day).
2. Add object segmentation + priority tiers (2 days).
3. Run on all 25 preview games locally with 1140s budget — target: ≥0.30 mean RHAE.
4. If ≥0.30: push to Kaggle.
5. If <0.30: iterate priority heuristics locally until ≥0.30, only then Kaggle.

---

## Directions NOT to pursue (yet)

- **Hardcoded per-game rules** — what the user explicitly rejected. Also violates comp spirit.
- **LLM-per-frame** — too expensive, <4% in preview.
- **Pure learned world models** (DreamerV3, IRIS) — would need training data we don't have; 2-week full timeline.
- **TTT+LoRA on 4B model** — NVARC's recipe but needs synthetic pretrain; 3-week effort for uncertain lift.

## Open questions for user approval
1. Proceed with **v9 = ForgeExplore (S-1 + S-2 + A-3)**? Or prefer different mix?
2. Do we replace BFS entirely, or keep BFS as fallback when graph explorer stalls?
3. Any appetite for Tier A-4 (TTT+LoRA) if we can prep synthetic data offline in a week?
