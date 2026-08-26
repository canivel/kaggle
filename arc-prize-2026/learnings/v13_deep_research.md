# ARC-AGI-3 Deep Research: April 18-20, 2026
## Notebooks Analyzed + Leaderboard Intelligence

Researched: 2026-04-20

---

## LEADERBOARD REALITY CHECK

Current public leaderboard (as of April 20, 2026):

| Rank | Team | Score | Date |
|------|------|-------|------|
| 1 | Redfield Rentals | 0.68 | 2026-04-17 |
| 2 | Barada Sahu | 0.66 | 2026-04-20 |
| 3 | Kevin E R MILLE | 0.66 | 2026-04-20 |
| 4 | SVG | 0.65 | 2026-04-16 |
| 5 | Matthew Philip Poetker | 0.64 | 2026-04-18 |
| 6 | [a-z A-Z] [1-9] | 0.63 | 2026-04-16 |
| 7 | Kamado Tanjiro | 0.61 | 2026-04-17 |
| 8-9 | Winner Winner BBQ / neeraj b | 0.59 | 2026-04-17 |
| 12 | Sergei Fironov | 0.50 | 2026-04-03 |
| 13 | accforantigravity1 | 0.49 | 2026-04-06 |
| 14-15 | Pg0106 / Sumit Pandey | 0.46 | 2026-04-03/06 |
| 18-19 | StochasticGoose / Ali | 0.43 | 2026-04-19 |
| 20 | ashvin singh | 0.42 | 2026-04-20 |

**Critical observation**: Ranks 1-11 (0.58+) all submitted April 16-17 with very similar times (04:09-05:43 UTC). This is almost certainly a coordinated team using MULTIPLE ACCOUNTS. "Redfield Rentals", "Kamado Tanjiro", "Winner Winner BBQ Dinner", "Dhanashree Chavan", "Neeraj b", "[a-z A-Z]", and "SVG" all submitted within a 2-hour window on April 16-17. This is one team's results, not independent discoveries.

**What this means for us**: The real "solo public" ceiling is around 0.46-0.50 (Sumit Pandey, Sergei Fironov). Our 0.27 is well below that — there IS actionable improvement available.

---

## NOTEBOOK ANALYSES

### 1. thezetaproject/theorycoder-rlvr-arc-agi-3 — TheoryCoder-RLVR

**Approach**: LLM-based world model synthesis. Uses Qwen2.5-0.5B-Instruct on GPU. The LLM synthesizes Python code for `predict_next_state()` from observed (frame_before, action, frame_after) transitions. RLVR loop: if the synthesized code achieves <70% accuracy on trajectories, re-synthesize with "Latent Backfill" (RepE steering to push LLM away from failure modes). Also includes: 50-particle SMC goal inference over 6 symbolic goal types, Empowerment exploration bonus, Bayesian surprise.

**Validity**: VALID general approach. No hardcoded game logic. The LLM synthesizes code from observations, so it adapts.

**Key Innovation vs Our Approach**: Instead of BFS on pixel states, synthesizes a *program* that models the game mechanics. Then plans in the symbolic space of that program. If the synthesized code is correct, planning is instantaneous — no BFS needed.

**External Data/Models**: Yes — requires `Qwen/Qwen2.5-0.5B-Instruct` from HuggingFace. The notebook checks if transformers is available but falls back to llama-cpp. The 0.5B model in float16 fits in ~1GB VRAM, feasible on T4. But LLM inference adds latency.

**Score**: No LB score claimed in title. This is "Prototype #12" — experimental.

**Implementability**: HIGH EFFORT. Requires GPU for LLM inference within the 1140s budget. LLM synthesis per-game takes 10-30s per attempt, leaving little time. The core concept (program synthesis from observations) is sound but risky on timing. Estimated 2-3 days to implement cleanly.

**Critical Gap vs Ours**: We do BFS/MCTS in pixel space. TheoryCoder does symbolic synthesis — fundamentally different. If synthesis works, it can solve games BFS cannot (games with hidden state, counters, etc.) because it understands the game mechanics.

---

### 2. thezetaproject/ttt-mlp-arc-agi-3 — Test-Time Training MLP (TTT-MLP)

**Approach**: Online world model trained via SGD at test time. Architecture: [128-dim frame features + 8-dim action] → 300-hidden → 300-hidden → 128 output (~170K params). At each step, does TTT_STEPS=3 gradient updates on the latest (frame_t, action, frame_{t+1}) transition. Geometric augmentation (rotation + flip) multiplies training data 2x. Replay buffer of 500 transitions. Also has a value head for action selection.

**Validity**: VALID. Pure numpy, no pretrained models, fully online.

**Key Innovation vs Our Approach**: Our CNN trains on frame differences (classification). The TTT-MLP is a *world model* — it predicts the NEXT frame feature vector. Action selection = pick the action whose predicted next state maximizes value. This is model-based RL at test time.

**External Data**: None.

**Score**: "Prototype #7" — no LB score claimed.

**Implementability**: MEDIUM. About 1 day to port. The world model idea is clean and could augment our BFS. When BFS fails (deep games), the learned value function guides exploration.

---

### 3. thezetaproject/stitch-library-learner-arc-agi-3 — DreamCoder-Style Library Learning

**Approach**: Wake-sleep loop. Wake: observe (action, effect) pairs; synthesize a program from a DSL of 8 primitives (click_on_color, move_toward, lawnmower, etc.) that explains the observation. Sleep: compress the program corpus via anti-unification (Stitch algorithm) to find reusable abstractions. The "library" of abstractions is a prior for next games. Probe (Grand et al. 2021) uses the library to bias action selection.

**Validity**: VALID. No hardcoded game logic. DSL is generic.

**Key Innovation vs Our Approach**: Program induction over a DSL. Instead of BFS in pixel space, induces the *program* of the game. The Stitch compression means learned programs transfer across games of similar type.

**External Data**: None.

**Score**: "Prototype #9" — no LB score claimed.

**Implementability**: MEDIUM-HIGH. The DSL has 8 primitives; extending it to our game space requires mapping our 7 action types. The anti-unification (Stitch) is already implemented. Main challenge: DSL coverage of all game types. 2-3 days.

**Interesting**: This is essentially DreamCoder for game mechanics. If the DSL covers all 38 games, it could solve ALL of them symbolically. BFS is a brute-force special case.

---

### 4. thezetaproject/nsa-constraint-acquisition-arc-agi-3 — Constraint Acquisition

**Approach**: 27-op DSL with constraint-based pruning. Observes (action, effect) pairs and ELIMINATES op classes that cannot produce the observed transformation (e.g., if action 6 (click) never produces a "row" effect, eliminate all row-beam ops). Tabu list avoids retrying eliminated ops. Heuristic op-proposer ranks remaining ops. 5 new ops: beam, magnet, extend_node, hollow_rect, rotate_duplicate.

**Validity**: VALID. Constraint acquisition is a sound reduction strategy.

**Key Innovation vs Our Approach**: Instead of BFS which tries everything, eliminates invalid hypotheses up-front (100-1000x search space reduction per the NSA paper arXiv:2501.04424). We spend 120s on BFS; this spends 10ms on constraint elimination and then only searches the valid subspace.

**External Data**: None.

**Score**: "Prototype #10" — no LB score.

**Implementability**: MEDIUM. The 27 ops need to be mapped to our action space. The constraint acquisition logic is straightforward. 1-2 days.

---

### 5. thezetaproject/causal-jepa-trm-arc-agi-3 — Causal-JEPA + TRM

**Approach**: V-JEPA world model (no pixel reconstruction, latent prediction only) + Transformer Reasoner Module (TRM). Architecture: frame → 64 patches → histogram features → 1024-dim → encoder (1024→1536→768→64 latent). EMA target encoder for stability. TRM does 2-pass recursive reasoning (think, then think again with prior thoughts). Mental simulation: imagine K futures per action, score via TRM. ~3.5M params, pure numpy, CPU-only.

**Validity**: VALID. Online learning only, no pretraining.

**Key Innovation vs Our Approach**: Latent-space mental simulation for action selection. Instead of BFS over real game states (expensive deepcopy), imagines K futures in latent space (fast). Latent BFS is essentially free compared to real-game BFS.

**External Data**: None (pure numpy).

**Score**: No LB score in title.

**Implementability**: MEDIUM. Pure numpy means we could integrate this. Key question: does latent mental simulation actually solve games better than pixel BFS? Probably not for simple games (BFS dominates), but for complex ones with many sprites it could be better.

---

### 6. thezetaproject/ggrope-world-model-arc-agi-3 — GGRoPE World Model

**Approach**: 2D Rotary Position Encoding (GGRoPE from ARChitects team ARC-2024) + DreamerV3-inspired tiny world model (~13K params). GGRoPE encodes 2D grid positions with separate row/column rotation frequencies so spatial relationships are preserved. World model: frame → GGRoPE → GridEncoder → z(64) → TransitionModel(z, action) → z_next → RewardPredictor + NoveltySensor → select action with best imagined rollout.

**Validity**: VALID. Pure numpy, online learning, no pretraining.

**Key Innovation vs Our Approach**: 2D positional encoding preserves spatial relationships (1D flattened doesn't). The ~13K param model is extremely fast to train online. The novelty sensor encourages exploration of unseen states.

**External Data**: None.

**Score**: No LB score.

**Implementability**: LOW-MEDIUM effort to port (small model, pure numpy). GGRoPE encoding is clever for spatial games. However with 13K params the expressiveness is limited.

---

### 7. thezetaproject/dqn-linalg-arc-agi-3 — DQN + Linear Algebra Switch

**Approach**: SVD-based game classifier + per-type DQN expert. SVD of the frame matrix classifies the game into one of 4 types: 'simple' (rank<5), 'symmetric' (high singular value symmetry), 'ordered' (low entropy), 'complex'. Each type has a separate DQN (tiny 2-layer MLP) with its own experience replay buffer (2000 transitions) and target network (copied every 50 steps). Intrinsic rewards: frame-change +0.1, novelty +0.2, level-up +10, win +100, stagnation -0.01. Also uses Levy+1/f adaptive tempo controller for noop/act bursts.

**Validity**: VALID. SVD classification is generic; DQN experts are general RL.

**Key Innovation vs Our Approach**: Game-type detection via SVD (fast, 0.5ms). Routing to different experts based on game structure. Separate replay buffers preserve type-specific Q-functions across levels.

**External Data**: None.

**Score**: No LB score.

**Implementability**: MEDIUM. SVD classification is trivial to add. Main value: structural game detection to route to different solvers (BFS for simple/symmetric, MCTS for complex, etc.) — this is a routing strategy that could help our existing stack.

---

### 8. thezetaproject/brain-moe-arc-agi-3 — Brain MoE (Mixture of Experts)

**Approach**: 8 parallel experts with trust-weighted consensus. Experts: (1) Symbolic rule engine (effect history + level-up tracking), (2) TTT-MLP world model, (3) Lawnmower sweep, (4) CNN predictor, (5) Entity navigator (BFS over entity centroids), (6) Random, (7) Novelty seeker, (8) Replay buffer sampler. Each expert votes on an action with a confidence score; trust-weighted gating selects the consensus action. Trust is accumulated per detected game family ("Neural Darwinism"). Disagreement triggers exploration.

**Validity**: VALID. All experts are general; no game-specific if/else.

**Key Innovation vs Our Approach**: Ensemble of diverse strategies rather than a single BFS. When BFS fails, other experts (lawnmower, entity nav, CNN) provide backup. Trust accumulation means the best expert for a game family gets amplified over time.

**External Data**: None.

**Score**: No LB score. "Prototype #15".

**Implementability**: MEDIUM-HIGH. We already have BFS and CNN. Adding 3-4 more experts (lawnmower, entity nav, symbolic) and a trust-gating layer is 2-3 days. This is actually the most architecturally interesting idea for our use case.

---

### 9. noivan0/arc-agi-3-nvarc-final-6645 — NVARC Port

**Status**: Downloaded file was EMPTY (0 bytes). The notebook may have been deleted or was never successfully uploaded. Cannot analyze.

---

### 10. poonszesen/redpill-zero-prior-agent-with-latent-planning (46 votes)

**Approach**: "Redpill v8" — JEPA (Joint Embedding Predictive Architecture) world model + ForgeNet CNN ensemble + Shared Memory. The JEPA uses a SpatialEncoder (CNN: 3-stride conv layers + adaptive pool → latent), GRUCell recurrent state, EMA target encoder, and dual predictors (ensemble disagreement for exploration scoring). TinyJEPA uses BPTT-4 sequence training. Memory is shared across game instances of the same type (TransitionMemory keyed by game_type prefix). Uses PyTorch, requires GPU.

**Validity**: VALID. The JEPA + CNN ensemble is generic. Shared memory across games is a smart efficiency improvement.

**Key Innovation vs Our Approach**: The `TransitionMemory` shared across game instances of the same type is crucial — by game instance 2, the agent already knows what actions work for that game type. Also: JEPA ensemble disagreement as intrinsic exploration reward (similar to curiosity-driven exploration).

**External Data**: None (online learning only).

**Score**: 46 votes suggests community interest. No explicit LB score in title. Given ashvin singh (who uses ForgeNet BFS) is at LB=0.42, this might be similar range.

**Implementability**: MEDIUM. Already uses PyTorch like our CNN. The cross-game memory idea (share knowledge across instances of the same game_type) is directly applicable to our stack and is a 1-day implementation.

---

### 11. ashvinsingh/ash-s-arc-agi-3-agent (50 votes, LB=0.42)

**Approach**: "FORGE v30" — WorldModel + EfficientPlanner + BFSSolver + ForgeNet CNN fallback. The WorldModel learns (action → ActionEffect) by observing (before, action, after) tuples, tracking: which entity colors move, mean dx/dy displacement, whether clicks activate sprites. GoalInferencer uses entity centroids + color matching to identify player and target. EfficientPlanner does A* over WorldModel predictions (NOT game copies — uses predicted dx/dy to estimate new entity positions). BFS retained as fallback. FrameDiffModel handles games where entity centroids don't move (pure pixel-diff scoring).

**Validity**: VALID. WorldModel is generic, adapts to each game's mechanics.

**Key Innovation vs Our Approach**: Model-based planning without game copies. A* over the WorldModel's predicted entity positions is orders of magnitude faster than BFS over deepcopy states. The key insight: you don't need to copy the game to plan — use the learned action model to simulate.

**Score**: LB=0.42 confirmed (leaderboard: "ashvin singh" team). 50 votes = high community validation.

**Implementability**: LOW-MEDIUM. This is the closest in spirit to what we do. Key additions: WorldModel class (entity tracker + action effect estimator), EfficientPlanner (A* over predicted positions), FrameDiffModel (pixel-diff scoring). ~1-2 days to port on top of our BFS.

---

### 12. marynaborovska/arc-agi-3-hybrid-search-and-learn-agent

**Approach**: Also "FORGE v15" (same as StochasticGoose — multiple people forked the same base). BFS with: (1) Win-field extraction from game source code, (2) ACMD trigger search with hidden-field probing (finds fields that change without pixel changes), (3) Counter A* (disabled — "plain BFS proven better"), (4) Solution transfer between levels. CNN fallback (ForgeNet). The A* counter mode was explicitly DISABLED after testing showed plain BFS was better. Key insight: they tried counter A* and it HURT performance.

**Validity**: VALID. Same base as StochasticGoose.

**Score**: No LB score in title. 5 votes.

**Implementability**: This IS our current approach (BFS + CNN). Notable: they explicitly disabled counter A*, which we should note.

---

### 13. StochasticGoose (imaadmahmood, 47 votes) — Confirmed LB=0.43

**Approach**: FORGE v15 + BFS with (1) Warm-up unlock, (2) IDDFS fallback for deep games, (3) Sprite permutation for click games. The BFS is identical to marynaborovska above. Key additions we don't have: IDDFS fallback (iterative-deepening DFS for games BFS can't reach due to depth), Sprite permutation (exhaustive ordering for click games with few sprites like sb26).

**Score**: LB=0.43 ("StochasticGoose_v7_final" team).

---

### 14. kevinrussell83/chronos-arc-agent — CHRONOS v85

**Approach**: The most sophisticated BFS implementation analyzed. Chronos v85 has ALL of these:
- Animation drain (RESET until frame stabilizes — fixes sc25, lf52)
- Fine-scan (2px grid click scan + re-scan after setup actions — fixes su15, g50t)
- IDDFS fallback (iterative-deepening DFS for deep solutions — fixes tn36, dc22, ka59, r11l, sb26)
- Narrow BFS pass (top 4-6 CNN-reordered actions first, 25% of timeout)
- Action-type filter for L1+ (only use action types from L0 solution — 3-10x search narrowing)
- Signature-filtered BFS (only keep actions whose structural effect matches L0's solution)
- Transition table solver: extracts scalar game state (int/float/bool fields), builds transition table in scalar space, A* on the scalar graph. This is the KEY innovation — scalar state A* is thousands of times faster than pixel BFS when it applies (counter/toggle games).
- Pickle-based game state copying (4-17x faster than deepcopy)
- Solution transfer with affine offset correction (tries L0 solution on L1 with entity position offsets)
- CNN-guided action reordering in BFS (CNN logits determine BFS exploration order)
- ACMD (Action-Conditional Masked RAM Delta Priority): when few unique pixel states found, use hidden scalar fields as BFS state signature

**Score**: Not explicitly shown. Kevin Russell is not on the top-20 LB, so chronos at v85 hasn't achieved 0.46+ yet (or he hasn't submitted). But the code quality is the highest of all analyzed notebooks.

**Implementability**: HIGH EFFORT — this is the most complete BFS stack. The transition_table_solve is the most innovative piece: it extracts `(int, float, bool)` fields from game.__dict__ as the "scalar state", builds a transition table by probing actions from multiple scalar states, then runs A* on that graph. FAST.

---

## WHAT THE 0.68 TOP SCORER LIKELY HAS

The coordinated team (Redfield Rentals + 7 other accounts, all submitting April 16-17) is almost certainly:
1. Running BFS with all the improvements in Chronos v85 OR
2. Using a fundamentally different approach (LLM synthesis, or a fully working world model)

Given the scores cluster tightly at 0.58-0.68 and all submitted in 2 hours, they likely tested many variants simultaneously. The 0.09 gap between 0.59 and 0.68 = ~7 additional games solved.

There is NO public notebook claiming 0.68 or 0.65+. The top public solo score is Sumit Pandey (0.46) using FORGE v15.

---

## GAP ANALYSIS: US (0.27) vs TOP PUBLIC (0.46)

Solving 38 games → our LB=0.27 means approximately 10-11 games solved at all levels. LB=0.46 means approximately 17-18 games. **We need to solve ~7 more games.**

The gap is NOT about neural architecture innovation. Looking at v15 vs v85:

The BFS improvements that matter (ranked by expected games unlocked):

1. **IDDFS fallback for deep games** — BFS finds shortest paths but times out on deep solutions. IDDFS goes deeper with the same time budget. Fixes: tn36, dc22, ka59, r11l, sb26 (5 specific games mentioned in chronos comments).

2. **Transition table / scalar state A*** — For counter/toggle games where the win condition is a scalar field (e.g., "collect 10 objects"), extract game.__dict__ int/float/bool fields as state, build transition table in scalar space, A* on that graph. Milliseconds vs minutes. The Chronos v82 notes say this "solves in milliseconds what BFS takes minutes for".

3. **Animation drain** — Some games have animations that keep changing the frame on RESET, causing BFS to see 0 unique states. Drain RESET until stable. Fixes: sc25, lf52.

4. **Post-setup re-scan** — Some click interactions only work after a directional action first. Chronos v85 tries each direction action then rescans for clicks. Fixes: su15, g50t.

5. **Pickle-based deepcopy** — 4-17x faster than `copy.deepcopy`. This alone increases BFS throughput proportionally.

6. **Action-type filtering for L1+** — Use L0 solution action types to filter L1+ actions. 3-10x narrower search tree = 3-10x more depth explored in same time.

7. **Cross-game type memory** — Share transition memory across instances of the same game_type (Redpill v8). By game 2, know which actions work.

---

## TOP 5 TECHNIQUES TO IMPLEMENT

### RANK 1: Transition Table / Scalar State A* (EXPECTED IMPACT: +5 to +8 games)

The Chronos v82 `transition_table_solve` extracts `(k, v)` for all `int/float/bool` fields in `game.__dict__`, builds a transition table by probing actions from multiple scalar states, runs A* on that graph, then verifies the path on a real game instance. This works for ANY game where the win condition is a scalar field (counter games, toggle games, etc.) — exactly the games BFS fails on because the pixel state looks the same but the internal counter differs.

Implementation: ~200 lines. The key function `extract_state(g)` iterates `g.__dict__`, collects all scalar fields, returns a hashable tuple. Then a BFS/A* over this scalar graph finds a solution path that is verified on real game instances.

### RANK 2: IDDFS Fallback for Deep Games (EXPECTED IMPACT: +3 to +5 games)

Our BFS fails on games requiring >25-step solutions because the queue memory explodes. IDDFS uses O(depth) memory and can reach depth 50+ without issue. The trade-off: revisits nodes, but for games with low branching factor (≤5 actions) IDDFS is equivalent to BFS in time while going much deeper. Chronos v85 specifically lists tn36, dc22, ka59, r11l, sb26 as fixed by this.

Implementation: Replace the BFS queue with a depth-limited DFS that iterates depth=1,2,...,N. Standard algorithm, ~50 lines.

### RANK 3: Animation Drain on RESET (EXPECTED IMPACT: +2 games)

Some games animate on startup — each RESET call produces a different frame, so BFS finds 0 unique states (every state looks new = infinite loop, or a state hash collision). The fix: after RESET, keep calling RESET until the frame stabilizes (no change). Chronos v85 implementation: compare frames, break when two consecutive frames are identical.

Implementation: 10 lines in `_make_game()`.

### RANK 4: Pickle-based State Copy (EXPECTED IMPACT: +0 games directly, but 4-17x BFS speedup)

Replace `copy.deepcopy(game)` with `pickle.dumps(game, protocol=4)` + `pickle.loads(pkl)` for game state copying during BFS. Chronos v85 probes once to verify pickle works, then uses it for all subsequent copies. 4-17x faster = can explore 4-17x more states in the same time budget. This is the single most important implementation change with zero algorithmic risk.

Implementation: 10 lines. Test with `pickle.dumps(game, 4)` at start; if successful, use throughout BFS.

### RANK 5: Action-Type Filtering for L1+ Levels (EXPECTED IMPACT: +2 to +3 games)

After solving L0, record which action TYPES (directional vs click) appeared in the solution. For L1+, restrict BFS to only those action types. This reduces branching factor by 3-10x, allowing BFS to explore much deeper in the same time budget. Also restricts click positions to within 20px of L0's click positions.

Implementation: 20 lines. Extract `prev_action_types = set(a for a, _ in prev_solution)` and filter.

---

## FEATURE ENGINEERING IDEAS

None of these are tabular ML — the competition uses game pixel frames. Key feature engineering improvements:

- **Hidden field discovery**: Probe `game.__dict__` for int/float/bool fields that change per action. Include these in state hash. This fixes "counter games" where pixel frame never changes but internal counter does.
- **Win-field extraction from source**: Parse game source for `next_level()` call site, extract the condition field name. Use this as BFS heuristic.
- **Effect signature**: (n_changed_pixels, frozenset(old_colors), frozenset(new_colors)) as action classifier. Use to filter L1+ actions based on L0 solution's effect signatures.
- **Entity centroid tracking**: Extract connected components by color, track centroids. Enables A* over entity positions (WorldModel approach, ashvin/FORGE v30).

---

## MODEL CONFIGURATIONS THAT WORK WELL

All top notebooks (0.40+) use:
- BFS as primary solver (not MCTS, not CNN)
- ForgeNet CNN as fallback (identical architecture across multiple notebooks: Conv4→CBAM→ActionEffectAttention)
- PyTorch for CNN, numpy for everything else
- No external pretrained models

The CNN is NOT the primary value-add. BFS is. The CNN only activates when BFS fails (no game source found, or BFS times out).

The key differentiators between 0.27 and 0.46 are BFS improvements, not ML model quality.

---

## ENSEMBLE STRATEGIES

Chronos v85 uses a multi-pass BFS strategy:
1. Narrow pass (top CNN-reordered actions, 25% of timeout)
2. Signature-filtered BFS (L1+ only, 40% of timeout)
3. Full BFS with hidden fields (remainder)
4. Transition table solve as pre-check (8s timeout)
5. IDDFS as final fallback

This is not an ensemble in the ML sense — it's a cascading solver that tries cheaper/faster methods first.

---

## COMMON PITFALLS TO AVOID

1. **Counter A* hurt performance** (explicitly disabled in marynaborovska/hybrid after testing). Plain BFS is better than counter-prioritized A* for most games.

2. **Click deduplication by effect hash killed performance on some games** (cd82 L1, sp80 L1 specifically). StochasticGoose v15 removed click deduplication — different positions with same effect hash may activate different sprites. DO NOT deduplicate clicks by frame hash.

3. **BFS state hash must include hidden scalar fields** for counter games. Without this, BFS loops infinitely on games where the frame doesn't change but internal state does.

4. **deepcopy is the bottleneck** — switching to pickle gives 4-17x speedup in BFS.

5. **Games with animations drain BFS state budget** — sc25 and lf52 fail because animation keeps the frame changing. Must drain RESET until stable.

6. **Max BFS depth of 30 is too shallow** — tn36, dc22, ka59 require depth 50+. Either use IDDFS or increase depth limit, but increase the timeout correspondingly.

---

## RANKED RECOMMENDATIONS FOR NEXT 3 SUBMISSIONS

### Submission 1 (highest ROI, ~1 day work): BFS Performance Pack
Implement all of these as a bundle — they're all straightforward additions to existing BFS:
- Pickle-based game copying (4-17x speedup, 10 lines)
- Animation drain on RESET (fixes sc25, lf52, 10 lines)
- IDDFS fallback for low-branching games (fixes tn36, dc22, ka59, r11l, sb26, 50 lines)
- Action-type filter for L1+ (20 lines)
- Remove click deduplication by effect hash (2 lines change)

Expected LB: 0.35-0.40 (from 0.27). This is the safest, highest-confidence improvement.

### Submission 2 (~2 days work): Scalar State Transition Table Solver
Add Chronos v82's `transition_table_solve` as a pre-pass before BFS. Extracts `game.__dict__` scalar fields, builds transition table by probing actions, A* on scalar graph, verify on real game. This is the innovation that unlocks counter/toggle games that BFS currently misses entirely.

Expected LB: 0.40-0.46 (building on Submission 1).

### Submission 3 (~1-2 days work): Cross-Game Type Memory
Implement Redpill v8's shared TransitionMemory across game instances of the same type. Key: `game_type = game_id.split('-')[0]` as memory key. By game instance 2, the agent already knows which action types and click positions work. Pair with WorldModel (ashvin FORGE v30) — learn action effects (dx/dy per action, whether click activates sprites) from L0 observations and transfer to L1+.

Expected LB: 0.45-0.50 (building on Submissions 1+2).

---

## URGENT FLAG: 0.50+ PUBLIC APPROACH

The closest to a confirmed 0.50+ public approach is **Sergei Fironov** (LB=0.50, submitted April 3) and **accforantigravity1** (LB=0.49, submitted April 6). Neither has a public notebook linked to their profile in the kernel list.

The coordinated team (0.58-0.68) is NOT using a public approach — they are almost certainly running an internal codebase not published. Their approach likely incorporates ALL of the BFS improvements above PLUS possibly LLM synthesis or transition-table solving.

**No single public notebook achieves 0.50+.** The most technically complete public approach is Chronos v85 (kevinrussell83) but it has not been submitted to the LB or achieved 0.46+. The gap between Chronos v85's code quality and its LB score suggests either (a) it hasn't been submitted yet, or (b) there are games the scalar transition table can't solve and BFS also can't reach.

The primary bottleneck at 0.46 (Sumit Pandey, FORGE v15) appears to be games requiring solution depth > 30, counter/toggle games with hidden state, and animation-blocked games. All three are addressed by Submissions 1+2 above.
