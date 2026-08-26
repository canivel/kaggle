# v18 Dream / Neuroplasticity Analysis

**Date:** 2026-04-24 (night of v17 submission)
**Analyst:** Claude (neuroplasticity replay agent)
**Budget:** 600-800 words

---

## Replay patterns (v5 → v17)

1. **Score variance is ~0.08 on identical code.** v9 @ 0.26 Apr 17 → v15 @ 0.17 Apr 22 using the SAME code proves eval drift dominates small algorithmic deltas. Any v17 "improvement" under +0.05 is unresolvable from noise.

2. **Env > algorithm.** The 2x jump came NOT from a new algorithm but from environment correctness: `kaggle-images + CPU + no dataset`. v11/v13 put Chronos/FORGE upstream directly and scored 0.17-0.19. v16 put ashvin's EXACT code on a CLEAN env and got 0.34. This means our 0.18 era was masking 0.30+ code. Likely v8, v10, v14 would all score higher on v16's env.

3. **Complex additions consistently regress.** v6 (+ MCTS cap), v8 (+ tiered cap), v10 (+ GraphExplorer), v14 (+ pickle + anim drain) all lost to the simpler v7 baseline. The winning v16 is a pure upstream copy with ZERO original additions. Our "synaptic signature" for regression: multi-feature bundles, deep interactions with MCTS, novel search topology.

4. **Score jumps correlate with *subtraction*, not addition.** v7 (cap only short BFS = less work on hard games) beat v6 (cap everything). v16 (minimal env, no dataset path) beat v14/v15 (full local-dev env). Simplicity is a first-class signal.

5. **Ceiling math.** Ashvin same code = 0.42 public; we get 0.34. The 0.08 gap matches the eval-drift variance observed in (1). That suggests v16 is likely `0.34 ± 0.06` on any given day. We haven't truly underperformed ashvin — we're inside his stochastic band.

---

## v17 critique — probability it beats 0.34

**Prediction: 40% probability v17 ≥ 0.34; 35% regression to 0.25-0.30; 25% flat 0.30-0.33.**

**Why it probably won't help much:**
- Chronos v85 with scalar-TT has never confirmed a public LB score — marynaborovska explicitly **disabled** counter A* after testing proved "plain BFS better." That's a direct data point against our v17 assumption.
- Ashvin's WorldModel A* already covers counter-like games via dx/dy effects. TT pre-pass overlaps existing capability.
- 8s x 25 games = 200s stolen from ashvin's BFS. On levels where ashvin would have BFS-solved in 10-40s, the 8s loss is 20-80%. Losses on easy games may exceed gains on counter games.
- `deepcopy` in TT probing on some games (heavy game state) can spike to 15-30s and blow the budget.

**Falsifiable prediction:** If v17 < 0.30, the scalar-TT interaction cost dominated. If 0.30-0.34, it was neutral (as marynaborovska predicted). If ≥ 0.38, we've confirmed counter games matter and can build on it.

---

## v18 primary proposal — "Dual-slug statistical collapse"

**Move:** Submit v16 (EXACT ashvin, no additions) to a NEW kernel slug `canivel/arc3-final` simultaneously with v17 on the existing slug, on the SAME day.

**Rationale:** We have 8 days left and variance ~0.08. Running v16 twice on different slugs tests the "kernel history effect" hypothesis AND gives us 2 data points for the same code. If fresh-slug v16 scores 0.40+, we've unlocked a structural bias in our current kernel and can commit future submissions to the fresh slug.

**EV: +0.04 expected** (probability 35% fresh slug gains 0.06+, 50% neutral, 15% loses 0.02). **Risk:** costs one submission slot; no code regression risk.

**Falsifiable:** If fresh-slug v16 ≤ v16 old-slug + 0.02, kernel history is NOT the issue and we lock focus on algorithmic moves.

---

## v18 plan B (if v17 < 0.30)

**Revert to v16 EXACT + add ONE isolated feature: cross-game type memory (Redpill v8).** ~100 lines. Key: `game_type = game_id.split('-')[0]` caches (action_type_signature, successful_click_positions) across game instances. By instance 2+ of same type, ashvin's BFS searches 3-5x narrower space.

This is the ONE addition the top 0.46+ solo scorers (Sergei, Sumit) likely have that ashvin v30 public does not. Expected +0.03-0.06, low regression risk (pure cache; if memory empty, behaves exactly like v16).

**Falsifiable:** if v18 ≥ 0.36 we've proven cross-game memory is the missing piece and v19 doubles down (WorldModel persistence). If ≤ 0.34, memory isn't the gap and we pivot to IDDFS fallback as v19.

---

## Meta-strategy

- **New slug `canivel/arc3-final` — YES, today.** Eliminates kernel-history hypothesis at cost of 1 submission. The v9-@-0.26 vs v15-@-0.17 drop is unexplained otherwise.
- **Submission cadence: go to 2/day.** We're not rate-limited. With 8 days × 2 = 16 submissions and variance 0.08, we can run each variant twice and actually attribute lifts.
- **Stop copying full upstreams.** Every upstream copy scored 0.17-0.19 except v16 on clean env. Rule: ONE feature per submission, layered on v16 exact.
- **What the 0.42 ashvin does on day 11:** likely adds cross-game memory OR IDDFS (two public improvements he lacks). If we add either first, we can briefly exceed him. He won't add scalar-TT because marynaborovska's public writeup warns against it.
- **Overfit check:** we ARE iteration-speed anchored. The dual-slug trick is the first structurally-different move in 13 days. Do it.

---

## TL;DR

v17 is 40/35/25 win/regress/flat. The ONE highest-EV move is **submit v16 EXACT to a fresh slug TODAY** to test the kernel-history hypothesis while v17 runs. If v17 fails, v18 = v16 + cross-game memory only. Stop bundling features.
