# OPINE-World deep read — arXiv:2607.01531 v2 (Jul 15, 2026)

Filed 2026-07-18. Follow-up to `research_2026-07-18.md` §"ARC-AGI-3 direct".
Sources: arxiv.org/abs/2607.01531 + full HTML v1/v2 (multiple targeted reads,
results table transcribed twice independently and cross-checked against our own
game-id list — all 25 ids match `determinism_audit_25.md` exactly, including
`s5i5`). **No code release exists** — nothing in the abstract-page comments, no
github link in the paper, web search finds none. `runs/opine_world_src/` was NOT
created; everything below is from the paper text only, and every "NOT specified"
is flagged.

Paper: *OPINE-World: Programmatic World Modeling with Ontology-error-Prioritized
Interactive Exploration* — Courtis, Li, Sanner (U. Toronto). v1 Jul 1, v2 Jul 15.

---

## 1. Architecture summary (implementation depth)

### 1.1 State representation (spriteless, synthesized perception)
- Input is the raw 64×64 grid ("the engine exposes each frame as a raw 64×64
  grid", §2). **Main results use the spriteless regime** (Appendix H.2 Remark 1,
  verbatim): "object records are produced by the synthesized `extract_objects`
  (α_LLM), and all pairing, effect-count, and ontology-error diagnostics below
  are computed over those inferred records."
- The LLM *writes* `extract_objects(frame)` as Python. Object record: tuple
  (κ name, τ type, **v** attributes) with Attr = {x, y, visible, rotation,
  pixels, …} (App. H.2). **NOT specified:** the segmentation algorithm inside α
  (no mention of connected components or color regions anywhere), whether
  objects may be disconnected, pixel encoding. **α is NOT replay-verified** —
  only the transition model is. Perception is the unverified layer.
- Cross-frame tracking: partial map μ_t via greedy heuristic (App. H.3):
  (1) exact (name,x,y) matches first, (2) greedy same-name pairing of the rest,
  (3) unmatched → ⊥ (disappeared). Authors admit ambiguity when same-name
  sprites move simultaneously.
- Vs our stack: our `current_frame.segmentation` is deterministic
  connected-components — *more* principled than their unverified α_LLM; nothing
  to import here except the (name,x,y)-first pairing heuristic.

### 1.2 World model + CEGIS loop
- Model = one Python file `game_engine.py` exposing
  `transition_function(state, action)` → predicted next state and
  `reward_function(state)` → (reward, goal flag) (App. A.1). Internally
  object-centric factored: per-type rule f_τ: O_τ × A × X → O_τ, where X is a
  read-only *local context* over the pre-action state (only concrete example
  given: "a hash of the object's own pixels"; full feature list **NOT
  specified**). Agent free to use functions/classes/lookup tables.
- Counterexample source = **live mispredictions only**: synthesis fires "when
  the live model mispredicts an observed transition" (§3.3, App. A.4), with a
  short deferral window (accumulate a few errors before rewriting) and a stall
  guard (stop repeated non-improving rewrites). Window length **NOT specified**.
- **Admission contract (the core of the paper), §3.3:**
  φ_t(T̂, D_t): T̂(s_ℓ, a_ℓ) = s_ℓ+1 for ALL recorded transitions,
  "attribute for attribute and object for object" — exact replay over the whole
  buffer, not a loss. Plus a **double-run rejection**: every candidate is run
  twice per transition; if the two runs differ it is rejected — removes any
  model depending on hidden state or nondeterminism in its own code.
- Candidates per trigger, prompts, LLM calls, tokens: **all NOT specified.**

### 1.3 Determinism + settled-state contract
- Assumption 1 (§2, Observable-Markov determinism): T is a deterministic
  function of (state, action) under some representation. Matches our N5 audit
  exactly (0/25 divergent).
- Multi-tick handling (App. A.5, verbatim): "A single action can animate over
  several internal engine ticks before the state settles… The model predicts
  the **settled after-state**, and the replay check compares against that
  settled state." Intermediate per-tick frames are surfaced to the synthesis
  agent as evidence (mechanics that play out and revert leave no before/after
  signature). Settlement *detection* mechanism: **NOT specified.**

### 1.4 Action selection / planning
- Two Opus agents, communicating "only through the buffer and a short
  structured handoff" (§3.2): a **goal-directed action agent** (reads the
  monotonic interaction log with file tools + short scripts; never edits the
  model) and a **synthesis agent** (writes game_engine.py, submits to the
  verifier). Uniform compaction discipline: on crossing a per-game token
  threshold the agent emits a structured handoff and its session resets
  (App. A.6; threshold value **NOT specified**).
- Exploration steering = **ontology error** (§3.4), all deterministic math:
  per-object η_i = 1 − (1−U_type)(1−U_row) (noisy-OR), where
  U_type = H[Pr(τ_i|D_t)]/log K and U_row = H(q̂_j)/log m with a symmetric
  Dirichlet posterior over *effect signatures* per (type,action,context) row:
  q̂_j(e) = (α₀ + C(j,e)) / (mα₀ + Σ C(j,e′)). High-η objects are what the
  action agent probes. α₀, K enumeration, and the Pr(τ|D) inference mechanism:
  **NOT specified.**
- **Planning (§3.5):** only after a model is admitted AND a level has been
  cleared, the synthesis agent *writes a planner* — "a bounded forward search"
  (algorithm class and depth bound **NOT specified**) — over T̂ and Ĝ. The plan
  is first checked *offline* by replanning from entry states of already-cleared
  levels, then executed against the live game **one step at a time; each
  executed step is compared with the model's prediction. A match continues the
  plan; a mismatch ends it, records the offending transition as a
  counterexample, and returns control to the goal-directed agent.**

### 1.5 Results and protocol
- Score 78.4 under the official arcprize action-efficiency formula (per-level
  (human/agent)² capped 1.15, index-weighted, per-game capped 100) — same
  family as our rail scorer. **20/25 games, 160/183 levels, single run per
  game, no variance estimate** (their limitation iv, verbatim: "headline counts
  could shift by a few games under resampling").
- Baselines: baseline1 = **Rodionov 2026 single-agent object-centric coding
  agent** (the exact family our exec_wm v64/v65 line copies) on GPT-5.5 high
  reasoning: 63.8, 14/25. Vision (pre-trains on eval set, excluded from fair
  comparison): 63.2, 12/25. WorldCoder and Dreamer/MuZero-family: 0/25 each.
- **No ablations at all** — authors state component interdependency prevents
  naive removal; left to future work. So the 78.4-vs-63.8 delta conflates
  (Opus 4.8 vs GPT-5.5) with (two-agent + CEGIS + ontology-error + planner).
  The paper cannot attribute its own gain; neither can we.
- No mention of Kaggle or the competition LB anywhere. Whether the 1.86 leader
  is this method family is **unverifiable from the paper** — plausible (it is
  the strongest published result and the timing fits) but not evidence.

### Per-game table (transcribed; wins then fails)
Solved 20: tu93 9/9 (272 acts vs human 462), sb26 8/8 (214/213), lp85 8/8
(110/388), ar25 8/8 (381/748), tr87 6/6 (212/414), r11l 6/6 (128/233),
**ft09 6/6 (111/208)**, cd82 6/6 (161/171), cn04 6/6 (263/789),
**su15 9/9 (334/361)**, re86 8/8 (850/1255), tn36 7/7 (417/317),
vc33 7/7 (427/447), m0r0 6/6 (259/1107), sc25 6/6 (256/350), sp80 6/6
(369/518), wa30 9/9 (1465/1843), g50t 7/7 (757/879), ls20 7/7 (959/776),
dc22 6/6 (1479/1228).
Failed 5: **ka59 6+/7** (1076 acts, GAME_OVER), **sk48 4+/8**, **lf52 3+/10**,
**bp35 2+/9**, **s5i5 4+/8**. Only stated failure mechanism (limitation iii):
"the synthesized planner is a bounded forward search, and on games with a high
branching factor a naive search reaches its bound without a plan" + budget
exhaustion. No per-game post-mortems.

---

## 2. Per-component transfer table

| OPINE component | our status | what we have | cost to adopt | verdict |
|---|---|---|---|---|
| Frame-determinism assumption (Assumption 1) | **HAVE** | N5: 0/25 divergent, lc reproduces on replay (`determinism_audit_25.md`) | 0 | done — OPINE independently ratifies |
| Exact-replay admission + counterexample-on-mispredict | **ADAPT** | = (d)+(c) mechanical-refutation flag already scheduled W1 (`grinder_cracking_design.md` §3) | 0 extra days | adopt their 3 refinements (§3 below) |
| Settled-state contract (predict settled frame; board_changed=False actions are real identity-effect transitions) | **HAVE (as a bug fix)** | this is exactly the N5 `prune_trace` diagnosis — leading no-visible-change actions mutate hidden state; unpruned replay survives 25/25 | 0 | already understood; OPINE names it a first-class contract |
| Double-run hidden-state rejection | **BUILD (trivial)** | nothing | ~10 lines in any verifier | adopt wherever we verify predictions/sims |
| Executable per-game transition models | **HAVE (offline)** | `exec_wm/` sims, 22/24 Class-A, 12 games ≥99.5% state-exact incl. ft09/su15/lf52/s5i5 (`scale_summary.md`) — literally baseline1's (Rodionov) family, which OPINE builds *online* | 0 (built) | ours are pre-built assets, not online synthesis; see legality note §3 |
| Plan → execute stepwise → per-step predict/compare → abort on mismatch → record counterexample (§3.5) | **BUILD (cheap, highest value)** | BFS solvers exist (warpack/long-BFS lore); sims exist; the *verified-execution contract* does not | ~2–3 build days + A10 trigger bench | **the single highest-value adaptation** (§4 Stage 1) |
| Object extractor α (LLM-synthesized, unverified) | **HAVE (better)** | deterministic `current_frame.segmentation` | 0 | skip their α; optionally import (name,x,y)-first pairing heuristic |
| Ontology-error exploration (Dirichlet effect rows, noisy-OR entropy) | **ADAPT-LATER** | ledger FACT store could host effect-signature counts; the math is pure Python, zero LLM cost | ~1–2 days | value unproven (no ablation); park behind Stage 1; surfaces as a "most-unexplained object" scripted hint — but `feedback_prompt_is_noise` cautions |
| Two-agent actor/synthesizer split + compaction handoffs | **SKIP** | our single-agent duck loop, ~60k tokens/game | doubles token budget minimum | infeasible at our 1.5M-token/25-game envelope; unattributed benefit |
| Online CEGIS model synthesis by the LLM | **SKIP at 27B / park to war-v4** | probe decomposition: Qwen never writes BFS unprompted; model-fitting is on the NOT-distillable list | war-v4 gate | even GPT-5.5 single-agent = 14/25; at 72B expect a fraction; the deterministic scaffold is what makes weak synthesis safe |
| Synthesized planner written by the LLM | **SKIP** | harness-side deterministic BFS is strictly more reliable for us | — | write the planner in the harness, not by the model |

## 3. Failure-game cross-check vs our grinder taxonomy

- **su15: OPINE solves 9/9 at 334 acts (under human 361).** Our A13 verdict
  ("mechanics near-unobservable in budget", wall suspended) is **challenged at
  frontier tier**: an Opus-4.8 model-based agent clears it fully and
  efficiently. This upgrades the A13 re-probe from disambiguation to expected-
  retraction: su15 is not a wall, it is a model+world-model gap. Keep the A12
  exclusion for Qwen-tier windows regardless.
- **ft09: solved 6/6 at 111 acts** (human 208). Our "ARC-1 transformation
  prior = NOT-distillable concept gap" stands at Qwen tier but the game itself
  yields to programmatic WM + frontier model — consistent with GPT-5.6-sol's
  5/6 in our probe. ft09 stays the top banking-variance carrier.
- **Canonical grinders sb26/lp85: both fully solved** (sb26 at human parity;
  lp85 at 3.5× human efficiency, 110 vs 388 — the aliased-cell crossing model
  our distill flagged as the un-teachable representation is exactly what an
  admitted `game_engine.py` captures). Confirms §1 of
  `grinder_cracking_design.md`: the grinder gap is model tier + executable WM,
  not scaffolding.
- **OPINE's 5 failures {ka59, sk48, lf52, bp35, s5i5} are all deep-level games
  (7–10 levels)** and all are games where we score ≤1 level (war_v2 s1: ka59 1,
  lf52 1, sk48/bp35/s5i5 0). Failure mode is *planner branching + budget*, not
  perception or determinism. Sharp irony for us: our offline sims are
  SATURATED on lf52 (100% state-exact) and s5i5 (99.5%) — the world model is
  not the bottleneck on those two; search depth is. Even the frontier version
  of this method dies where bounded forward search dies — consistent with our
  `feedback_arc_long_bfs_mcts` (never cap search on long-horizon games).
- Taxonomy mapping: OPINE failures ≈ our "budget-death + deep-level" bucket,
  NOT our CONCEPT bucket (their CONCEPT games all fell). GSME-style
  pathology-bucket archiving survives contact with this paper.

## 4. Compute-envelope verdict + staged adoption

**Verdict: OPINE-World as published CANNOT run in our Kaggle kernel.** It is
two Claude Opus 4.8 agents behind a filesystem sandbox — closed weights, paid
API, and the scored kernel is offline; token/call/wall-clock budgets are not
even reported (single run per game, no cost accounting). The only Kaggle-legal
instantiation of the *loop* is an open-weights coder under vLLM — i.e. our
already-registered war-v4 72B line — and the paper's own model ladder
(Opus 4.8 → 20/25; GPT-5.5 single-agent → 14/25; our Qwen3.6-27B ≈ 6 levels
total) says the headline number is mostly model tier. What DOES fit the kernel
at zero token cost is every deterministic contract in the paper: exact-replay
admission, double-run rejection, settled-state comparison, per-step
plan-execute-verify, Dirichlet effect counts. Those are pure Python.

**Legality note (per-game assets):** OPINE's "no per-game training" is a
paper-framing claim, not a rule. Our exec_wm sims are pre-built per-game
assets fitted on the public local engines — Kaggle-legal (public data), but
15/25 local engine versions differ from the Kaggle build (N5 caveat). OPINE's
mismatch-abort contract is precisely what makes shipping them safe: a
version-drifted sim fails closed on its first wrong prediction and costs ~0.

### Staged adoption path (counting-bound Δ, rail units, per grinder-doc method)

**Stage 0 — replay-check contract into W1 (cost: 0 extra days; scheduled).**
The research sweep's prong-0 recommendation is **validated and upgraded**.
Wire (d)+(c) with OPINE's three refinements: (i) compare against the *settled*
after-state and treat board_changed=False actions as identity-effect
transitions (kills the prune_trace bug class permanently); (ii) run any
predictive check twice, reject on self-disagreement; (iii) make a mismatch a
*control-flow event* (abort current plan/hypothesis, write refutation FACT),
not a passive digest — this is the missing activation trigger behind the
1552-digest/0-escalation ledger failure, and it satisfies the GSME activation
gate by construction. Δ: unchanged from grinder doc, +0.02–0.05 expectation,
ceiling +0.10.

**Stage 1 — sim-guided plan-execute-verify (war-v3.5 window; cost ~2–3 build
days + A10 canary; NEW).** Harness-side bounded BFS over the 12 Class-A
≥99.5% exec_wm sims proposes an action sequence toward the sim's reward flag;
execute one action per turn; per-step hash-compare predicted vs actual settled
frame; first mismatch → abort to normal duck loop + refutation FACT. Zero
LLM tokens; game-agnostic code path (sims are data). Counting bound: target
games where the sim is saturated but Qwen clears 0 levels — ls20 (sim 100%),
tn36 (100%), tr87 (100%), vc33 (99.5%), s5i5 (99.5%). An L1 clear at ~base
actions is worth ≈3.6–4.8 pts each (scorer arithmetic, grinder doc §2);
3 conversions ≈ +12 pts ≈ **+0.5/draw rail ceiling — larger than the entire
v3 conversion stack ceiling (+0.31)**. Honest discounts: sims validated on
held-out tuples from shallow trajectories, reward_acc 77–100%, goal predicates
partly unverified, engine-version drift on 15/25, search may hit the same
branching wall OPINE reports. **Honest expectation +0.10–0.30 rail** — still
the largest single registered item after the model swap. Gate: standard
3-seed compound rule + mechanism prong = plans executed ≥1/run on ≥5 games,
mismatch-aborts logged, 0 post-abort deadlocks. Requires full-panel sign-off
(new asset class in kernel), one flag, its own window after W2.

**Stage 2 — ontology-error-lite (park until Stage 1 verdict).** Deterministic
Dirichlet effect-row counts per (object-type, action) in the ledger FACT
store; surface top-η objects as a capped scripted hint. Cost ~1–2 days;
expectation ≤ +0.02 (hint-injection is historically noise); only worth a
window if Stage 1 shows the duck consumes structured facts.

**Stage 3 — war-v4 online synthesis (existing line, unchanged).** OPINE is
the strongest evidence yet FOR the 72B swap (its entire gain lives at frontier
tier) and simultaneously evidence the swap alone is insufficient without the
deterministic scaffold (WorldCoder 0/25; baseline1 loses 6 games to budget
exhaustion). war-v4 should ship Stages 0–1 scaffolding *before* the model
swap gate so the swap is evaluated with the verified-execution rail in place.

---

## Final statements

**Compute-envelope verdict:** Not runnable as published — two Claude Opus 4.8
API agents, closed weights, unreported (plainly large) token budgets, offline
kernel; only its deterministic contracts (exact replay, double-run, settled
state, plan-step verify, effect counts) fit our kernel, and they fit at zero
token cost; the full loop is war-v4-tier and model-tier-dependent by the
paper's own baseline ladder.

**Architecture essence (3 sentences):** Two frontier-LLM agents share a
filesystem: one acts, one maintains `game_engine.py`, an object-centric Python
transition+reward model over objects extracted by synthesized perception from
raw 64×64 frames. A model is admitted only if it exactly reproduces every
recorded transition attribute-for-attribute and twice-identically
(deterministic by construction); any live misprediction is a counterexample
that triggers a bounded rewrite, and exploration is steered toward objects
whose type/effect Dirichlet entropy (ontology error) is highest. Once a model
is admitted, a bounded forward search plans to the goal predicate and executes
one verified step at a time, aborting to exploration on the first predicted-vs-
settled-frame mismatch — determinism plus exact replay is the load-bearing
pair; everything else is unablated.

**Single highest-value adaptation for war-v3/v4:** OPINE's §3.5
plan-execute-verify contract grafted onto our existing exec_wm sims —
harness-side BFS over the 12 saturated sims, one live action per step,
hash-compare against the settled frame, fail-closed on first mismatch with a
refutation FACT (Stage 1 above: ceiling ≈ +0.5/draw rail, honest expectation
+0.10–0.30, ~2–3 build days, zero LLM tokens, panel sign-off required) — with
the settled-state + double-run + mismatch-as-control-flow refinements folded
into the already-scheduled W1 refutation flag at zero extra cost.
