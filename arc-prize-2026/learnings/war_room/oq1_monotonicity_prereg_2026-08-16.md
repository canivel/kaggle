# OQ-1 MONOTONICITY PROBE — MINI-PREREGISTRATION (SEALED)

**Date sealed:** 2026-08-16
**Directive:** panel round 26, directive #3 (`learnings/panel/round26/DIRECTIVES.md` §3 row 3; ruling §4 OQ-1).
**Cost class:** FREE / CPU-only. 0 GPU-hours, 0 kernel slots, 0 submissions, 0 dollars.
**Data:** `runs/a22_v2_seed1/` (one arm, one seed, 25 games, 2026-08-05) — observational, already on disk.

**Discharges the four blocking conditions of C5 / OQ-1 (RL, LA, PS, ME):** (a) estimand, (b) statistic,
(c) permutation null + pre-registered threshold, (d) trajectory set + selection rule + anti-HARKing
held-out split, (e) ★ named intervention branch (§7).

---

## 0. WHAT WAS INSPECTED BEFORE THIS FILE WAS SEALED (full disclosure)

An honest prereg must say what the author already knew. Everything below was looked at **before** sealing,
because the design is not specifiable without it. **No association, correlation, credit vector, τ, z, or
p-value of any kind was computed before this file was written.** What was inspected:

1. `runs/a22_v2_seed1/summary.txt` — **already in the record** (published run artifact): per-game
   `levels`, `actions`, `score`. This is where the binding sample-size fact comes from: **12 of 25 games
   completed ≥1 level; 13 completed none.** Level counts are 1 or 2. This fact alone caps the design.
2. Pickle **schema only**: field names on `taaf.game.GameState`, the action-object layout
   (`previous_action.id`, `.data{x,y}`), the pickle's game ordering (which is **run order, not
   alphabetical** — it matches `benchmark.json:game_runs[]`, verified by `(game_id, len(history))`),
   the per-game **step indices at which `levels_completed` increments**, and the **alphabet sizes** under
   two candidate action-symbol definitions.
3. Item 2's alphabet-size table is the sole reason the primary action alphabet is the **coarsened** one
   (§3.1): under action-**id**-only symbols, 6 of the 12 scoring games have |A| ≤ 2 and the statistic is
   undefined for them. That choice was made on **eligibility grounds, not on any observed association.**

---

## 1. ESTIMAND (condition a)

**Plain statement.** *Does the agent spend its steps on the actions that its own trajectory shows are
productive of score?*

**Precise statement.** Fix a game `g`. Let `σ(·)` map an executed action to an action **symbol** (§3.1),
and let `A_g` be the symbols the agent actually executed at least once in `g`.

- **Revealed selection weight** of a symbol: `n_g(s)` = number of steps in `g` at which the criterion
  selected `s`. This is the criterion's *revealed preference* — the only observable proxy for "how much
  the criterion likes `s`", since this harness emits a structured tool call and **no logits, no logprobs,
  and no candidate ranking are recorded anywhere in the run artifacts**. (Stated up front: the estimand is
  about *revealed* preference, not about the decoder's internal scores. We cannot observe the latter.)
- **Score productivity** of a step: `c_t` = the discounted proximity of step `t` to a future scoring event
  (§3.2). `levels_completed` is the score signal — the competition scores levels.

> **ESTIMAND.** The population value of a rank association `τ` between the criterion's revealed selection
> weight `n_g(σ(a_t))` and the score productivity `c_t`, taken over steps and aggregated over games.
> The criterion is **MONOTONE in what scores** iff `τ > 0` beyond every null in §4 — i.e. the criterion
> allocates *more of its steps* to the action symbols that lie on the trajectory's path to a level win.
> It is **ANTI-MONOTONE** iff `τ < 0` beyond every null. It is **SCORE-BLIND** iff `τ` is inside the
> pre-registered indistinguishability band of §5.

**What this estimand is NOT.** It is not causal. It is not "the agent would score more if it reallocated."
It is a statement about whether one measurable feature of the realized policy co-varies with the realized
score signal, on one seed of one arm. §8 (POWER HONESTY) states what it cannot carry.

---

## 2. TRAJECTORY SET AND SELECTION RULE (condition d, part 1)

**Universe.** All 25 games of `runs/a22_v2_seed1/intermediate_states.pkl`, one pass each.

**Unit of analysis.** A *step* `t ∈ [1, N_g]` = one `GameState` whose `previous_action` is not `None`
(state 0 is the initial observation and is dropped: no action produced it). `N_g` therefore equals
`len(game_runs[g].history)`. `RESET` is retained as a legitimate executed action.

**Eligibility (fixed now, applied identically to observed and to every null replicate):**
- **E1** — the game has **≥1 scoring event** (at least one step where `levels_completed` increases).
  Without one, `c_t ≡ 0` and the statistic is undefined.
- **E2** — the game's symbol alphabet has **|A_g| ≥ 3** (τ over fewer than 3 distinct selection weights
  is degenerate).
- **E3** — `N_g ≥ 30` eligible steps.

**E1 is a selection on the outcome and we say so.** RL's condition-2 objection ("successful only? the
selection is confounded") is **conceded, not dodged**: this probe can only speak about *trajectories in
which the score signal fired at least once.* It says nothing about the 13 games that scored zero. That
restriction is inherent — a probe of monotonicity in a signal requires the signal to have varied — and it
is recorded here as a **scope limit of the verdict**, not as a nuisance.

---

## 3. STATISTIC (condition b)

### 3.1 Action symbol `σ(·)` — PRIMARY: coarsened
- `ACTION1..ACTION5`, `ACTION7`, `RESET` → their own name.
- `ACTION6` (the click action, which carries `data={'x','y'}` on a 64×64 grid) →
  **`ACTION6@(x//8, y//8)`** — a fixed dyadic 8×8 coarsening of the grid into 64 cells.

Rationale, fixed in advance: `ACTION6`'s coordinates make the raw alphabet almost all singletons
(a click-heavy game emits >100 distinct raw actions in 228 steps), while id-only collapses click games to
`{ACTION6, RESET}` and makes 6 of the 12 eligible games degenerate. The 8-cell block is the natural dyadic
coarsening of a 64-wide grid and was **not** selected by trying alternatives against the outcome.

**ROBUSTNESS (descriptive, non-verdict-carrying):** the id-only alphabet, on whichever games satisfy E2
under it.

### 3.2 Score productivity `c_t` — discounted forward credit
Let `S_g = {t : levels_completed(t) > levels_completed(t−1)}` be the scoring steps of game `g`
(the action at a scoring step is the action that won the level; verified against `just_won_level`).

  c_t = max{ γ^(e − t) : e ∈ S_g, e ≥ t },  and  c_t = 0 if no scoring step is at or after t.

**γ = 0.98** (half-life ≈ 34 steps), fixed a priori. Forward-only: an action may be credited only for a
level win that comes *after* it. The winning step itself gets `c = 1`.
**ROBUSTNESS (descriptive):** γ ∈ {0.95, 0.99}.

### 3.3 The test statistic
For each eligible game `g`, over its `N_g` steps:

  x_t = n_g(σ(a_t))          (revealed selection weight of the symbol executed at t)
  y_t = c_t                  (score productivity of step t)
  τ_g = Kendall τ_b(x, y)    (τ_b, i.e. tie-corrected — both vectors are heavily tied by construction)

Aggregate over the eligible games of the split, weighted by step count:

  **T = Σ_g N_g · τ_g / Σ_g N_g**

`T` is the single verdict-carrying number. Per-game `τ_g` is reported **descriptively only** (directive 6 /
ME's multiplicity condition: per-game reads carry no verdict and cannot contaminate the primary).

---

## 4. NULL (condition c, part 1) — THREE NULLS, CONJUNCTIVE

A single "shuffle actions within game" null is not sufficient here, and we say why before running: `c_t` is
a **function of time** (it is large near a level win and exactly 0 after the last one), and the action
sequence has strong temporal structure (explore-then-repeat). A uniform shuffle destroys that structure and
would therefore convert *ordinary temporal autocorrelation* into apparent monotonicity. So the null is
conjunctive over three nulls that fail in different directions.

- **N1 — uniform within-game shuffle** (*the directive's literal null*). Independently in each game,
  uniformly permute the executed-action sequence. Preserves each game's action marginals `n_g(·)` exactly
  and the credit vector `c` exactly; destroys the step↔symbol pairing.
- **N2 — circular rotation.** Independently in each game, rotate the executed-action sequence by a uniform
  random offset. Preserves marginals **and the sequence's own autocorrelation**; destroys phase alignment
  with `c`. Strictly more conservative than N1 against the temporal confound.
- **N3 — score-anchor placebo.** The action sequence is left **completely untouched**; instead `S_g` is
  replaced by `|S_g|` step indices drawn uniformly without replacement from `[1, N_g]`, and `c` is
  recomputed. This isolates *the score signal specifically*: it asks whether the association is to the real
  level wins or to any arbitrarily-placed anchor in the same trajectory.

**B = 10,000** replicates per null. Master seed **20260816**; each null draws from an independent
`numpy.random.default_rng` stream. Two-sided permutation p-value with the standard +1 correction:
`p = (1 + #{|T_b − mean(T_null)| ≥ |T_obs − mean(T_null)|}) / (B + 1)`, and `z = (T_obs − mean(T_null)) / sd(T_null)`.

---

## 5. DECISION THRESHOLD (condition c, part 2) — PRE-REGISTERED, TWO-SIDED

Evaluated on the **held-out** split (§6). α = 0.05, two-sided, **conjunctive over N1, N2, N3**.

| verdict | rule |
|---|---|
| **MONOTONE** | `T_obs > mean(T_null)` **and** `p ≤ 0.05` under **all three** nulls |
| **ANTI-MONOTONE** | `T_obs < mean(T_null)` **and** `p ≤ 0.05` under **all three** nulls |
| **SCORE-BLIND** (indistinguishability band) | `|z| < 1.0` under **all three** nulls |
| **INDETERMINATE** | anything else — including `1.0 ≤ |z| < 1.96`, and including **any disagreement between the three nulls**, in which case the disagreeing null is named and the result is reported as CONFOUNDED, not as a finding |

**Pre-committed sentence for the SCORE-BLIND branch, written before the result is known** (the campaign's
standing sin is claiming more than the design carries):

> A SCORE-BLIND reading means the criterion's revealed allocation is **statistically indistinguishable, at
> this design's resolution, from an allocation that ignores the score signal.** It does **not** establish
> that the criterion is independent of score. With 6 held-out games it is an *upper bound on the size of
> any monotone association*, not a demonstration of its absence. §8 states that bound.

**Pre-committed sentence for the MONOTONE branch:**

> A MONOTONE reading **closes** OQ-1 in the negative: the routing/consumer story of 2608.12959 and
> 2608.12321 would then not be the operative defect at the level of action allocation, and the licensed
> next question is the *magnitude and efficiency* of the tracking, not a reranker. **No artifact is built.**

---

## 6. HELD-OUT SPLIT (condition d, part 2) — ANTI-HARKing

**Split rule, fixed before any statistic was computed and derivable from the already-published
`summary.txt` alone:** sort the 25 `game_id` strings ascending (byte order); rank 0-based.

- **DEV** (development read, **non-verdict-carrying**) = **odd** ranks — 12 games.
- **HELD-OUT** (**PRIMARY**) = **even** ranks — 13 games.

Resulting membership (from `summary.txt`, alphabetical; ✓ = passes E1, i.e. levels ≥ 1):

- **HELD-OUT (even ranks, 13):** `ar25`✓, `cd82`, `dc22`, `g50t`✓, `lf52`, `ls20`✓, `r11l`✓, `s5i5`,
  `sc25`, `sp80`✓, `tn36`, `tu93`✓, `wa30` → **6 eligible games**.
- **DEV (odd ranks, 12):** `bp35`, `cn04`, `ft09`✓, `ka59`✓, `lp85`✓, `m0r0`, `re86`, `sb26`✓, `sk48`,
  `su15`✓, `tr87`, `vc33`✓ → **6 eligible games**.

The parity rule was chosen because it is the simplest deterministic rule available and it was written down
before the eligibility counts were checked; that it lands **6 eligible games and 7 scoring events on each
side** is a property of the data, not a tuning.

**Order of operations, binding:** the statistic and all three nulls are **frozen by this document**. DEV is
run first purely as a **calibration and self-test** (does the pipeline produce a well-formed null, are the
nulls' means where theory says, does `--selftest` pass). **DEV cannot change one constant of §3–§5.** The
HELD-OUT number is the **primary and only verdict-carrying result.** If the DEV and HELD-OUT reads
disagree, both are reported and the verdict is taken from HELD-OUT, with the disagreement flagged as
evidence about *stability across games*, i.e. as further power information.

---

## 7. ★ NAMED INTERVENTION BRANCH (condition e — the blocking one)

PS: *"a confirmed non-monotonicity must name the intervention it licenses or it is a diagnostic with no
consumer, the very pathology the papers describe."* LA: prompt-side is **excluded by our own adopted 1c-2**
(2608.12321: *no prompted intervention helps — all inflate conservative bias*). LA's two available lanes are
**decode-side action reranking** and **scorer-in-the-loop selection**. We choose the second and specify it.

### 7.1 The artifact: **YIELD-RERANK** (scorer-in-the-loop selection)

*Why this one and not decode-side reranking:* our agent emits the action as a **structured tool call**
(`{"action":"ACTION6","x":3,"y":7}`, `parser=qwen3_coder`, verified live in the Q38 v1 log at t≈420 s).
Per-action logprobs are not recoverable from a parsed tool call without changing the serving path;
scorer-in-the-loop is a wrapper around the *selection* step only and requires no serving, weight, or prompt
change. It is buildable in the duck harness today.

**Specification (concrete enough to build):**
1. **Site.** The single point in the duck harness where the model's tool call is parsed into a
   `GameAction`. Nothing upstream of it moves. **Not one token of the prompt changes** (1c-2 compliance).
2. **Candidates.** Ask the same chat-completions call for `n = 4` samples instead of 1 — same prompt, same
   temperature, same everything. Parse each into a candidate `GameAction`; drop unparseable ones; drop
   candidates not in `available_actions`.
3. **Online yield table.** Maintain, per episode, an exponentially-decayed credit vector over the **same**
   symbol alphabet `σ(·)` and the **same** kernel `γ = 0.98` this probe uses. On each step: decay all
   accumulators by γ, add 1 to `attempts(σ(a_t))`'s decayed pool; when `levels_completed` increments at
   step `e`, add the decayed pool's mass to `credit(s)` for every symbol in proportion to its decayed
   presence. Posterior yield `Ŝ(s) = (credit(s) + 1) / (attempts(s) + 2)`. O(|A|) per step, no replay.
4. **Selection.** Choose `argmax_s Ŝ(σ(candidate))` over the ≤4 candidates; ties broken by **multiplicity
   among the 4** (i.e. the model's own modal proposal wins ties). **Fallback:** if every candidate symbol
   has `attempts < 3`, take the modal proposal — so the wrapper is a **no-op early in an episode** and only
   becomes score-driven once the episode has evidence.
5. **Cost.** 4× decode on the action call only; no extra prompt tokens (the `n=4` samples share one
   prefill). No serving change, no weight change, no dataset change.
6. **Validation.** One free Kaggle **build** eval on the same 25-game local set, against the frozen
   `a22_v2_seed1` baseline; read as Δ`levels_completed` under `duck_eval/SCREEN_PROTOCOL.md`, and against
   the standing promotion bar in `runs/ledger.json` (re-read at build time — it drifts) before any
   submission slot is spent. This consumes a slot only at the *submission* stage and therefore enters the
   OQ-5 portfolio question (directive 2) rather than bypassing it.

### 7.2 Trigger conditions — stated before the result is known

| held-out verdict | action |
|---|---|
| **ANTI-MONOTONE** | **BUILD YIELD-RERANK.** The criterion is putting its mass where score is not. |
| **SCORE-BLIND** (`\|z\| < 1.0` on all three nulls) | **BUILD YIELD-RERANK.** The criterion's allocation is indistinguishable from score-blind at this resolution; an explicit score consumer is exactly the missing consumer the two papers describe. |
| **MONOTONE** | **DO NOT BUILD.** The criterion already tracks the score signal; a reranker on the same signal is redundant and would be motivated reasoning. §5's pre-committed sentence applies. |
| **INDETERMINATE** (incl. null disagreement) | **DO NOT BUILD.** The licensed action is **power, not an artifact**: re-run this exact frozen probe on the `intermediate_states.pkl` that **every future 25-game arm banks for free**, accumulating independent seeds until the design reaches 80% power at the observed effect size (§8). Cost: 0. |

Two of the four branches build nothing. That is deliberate.

---

## 8. POWER HONESTY (modelled on `q38_engine_swap_prereg_2026-08-15.md` §4.3)

**Declared before the run:**

1. **n = 6 held-out games, 7 scoring events, ~1,076 steps.** The effective replication unit for N2 and N3 is
   the **game**, not the step, because both nulls are applied independently per game. **Six is a small
   number and this design is underpowered against small effects by construction.** No result here can be
   reported as "well powered".
2. **One seed, one arm, observational.** `a22_v2_seed1` is a single pass of a single configuration. There is
   **no second seed on disk.** Between-seed variance is therefore **unestimated**, and every number in §9 is
   conditional on this one draw.
3. **Half the corpus is invisible to the probe.** The 13 games that completed zero levels are excluded by
   E1. If the criterion's failure mode lives *there* — in trajectories that never touch the score signal —
   this probe cannot see it. This is the single largest scope limit.
4. **Pre-registered power computation** (run as part of the analysis, results in §9): a **credit-tilt
   simulation**. Holding each held-out game's credit vector and action multiset fixed, resample the action
   sequence with `P(s) ∝ n_g(s) · exp(λ · c̄_g(s))` for a grid of tilt strengths `λ ≥ 0`, and report the
   smallest `λ` — and the `T` it induces — at which the §5 conjunctive rule fires at ≥80%. That number is
   the **MDE of this design**, and it is reported whatever the verdict is.
5. **Pre-committed:** if the verdict is SCORE-BLIND or INDETERMINATE, the write-up **must** state the MDE
   from (4) in the same paragraph as the verdict, and **must not** describe the null as evidence of absence.

---

## 9. RESULTS AND VERDICT

*(APPEND-ONLY. Written after the run. Nothing in §0–§8 may be edited once results exist; the sha256 of this
file as of sealing is recorded in `runs/oq1_monotonicity/results.json` under `prereg_sha256_at_seal`.)*
