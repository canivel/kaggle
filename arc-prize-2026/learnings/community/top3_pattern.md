# TOP-3 PATTERN — living document (created 2026-08-24; update daily)

**Question (principal's standing order):** what is the PATTERN getting teams to top 3?
**Standing hypothesis (from the 08-20 rethink / 08-21 attribution):** private machinery + architecture-over-model + capability steps (not draw-grinding). This document sharpens or refutes it daily with evidence tags [V]/[V-doc]/[INF]/[UNK].

## Current top 5 (board-verified 2026-08-26 10:00 UTC, full archive 2546 rows)

| # | team | score | lifetime subs | cadence (from daily archives) | public artifacts | step history [V] |
|---|---|---|---|---|---|---|
| 1 | cstl | **5.99** | 36 | 1/day, unbroken | none, all campaign | at 2.70 BEFORE Q38 existed [V-doc, Ravindra 08-15]; 3.57 from 08-19, **flat 5 days while drawing daily** → **+2.42 (08-24)** ← largest single-draw step of the campaign → **DREW-NO-GAIN (08-25)** |
| 2 | Lord Han Solo | **4.99** | 43 | 1/day, unbroken | none (kernels+datasets checked) | 1.65 → **+1.11 (08-16)** → flat 6 days at 2.76 → **+0.60 (08-22)** → flat 2 → **+1.63 (08-24)** → **DREW-NO-GAIN (08-25)** |
| 3 | Tufa Labs | **4.67** | 118 | exactly 1/day, unbroken | June-30 harness (frozen), 08-07/08-15 bundles via Brüggen; **`taaf-kaggle-source` republished 08-25 17:01 w/ new `polyphony/` pkg**; NOTHING for the 2.0+ tier | 1.62 flat (08-14..18) → +0.45 → +0.90 → +0.07 → flat ×2 → **+1.54 (08-23)** → flat → **+0.09 (08-25)** |
| **4** | **Tong Hui Kang** | **3.39** | 54 | ~1/day | none | 2.24 → **+1.15 on 2 draws (08-25)** ← NEW top-5 entrant |
| 5 | Tony G (junvalue) | 3.17 | 13 | sparse | none | 0.30 → **+1.35 (08-21)** → flat → **+1.52/2 draws (08-24)** → DREW-NO-GAIN ×2 (08-25) |
| 6 | Daniel Franzen | 2.88 | 52 | ~1/day | none for -3 | steady climber, no single step >0.5 observed; DREW-NO-GAIN 08-24, 08-25 |

Reference points: FOYSAL 2.23/96 = the PUBLIC ceiling, **frozen since 08-18, 8th day** [V]. The 1.55–1.65 duck-floor band: **flat 10 consecutive days** on median (1.60) while growing to 52 teams [V, lb_diff]. Field-wide 08-25→26: **320 teams submitted, 73 (22.8%) gained anything**, median gainer 0.20/draw. Our own row: field-floor config draws 1.59/1.58/1.63/1.16/**1.92** — **n=5, mean 1.5760, sd 0.2713** — a fixed config redrawing around its mean, exactly as the top-tier flats show. **Our EXEC-WM v1 arm drew 1.05 on 08-26** (z = −1.94 on the floor; z = +0.67 on the frozen-fork null, i.e. inside it) — artifact read pending, and see the 08-26 sharpening for a candidate instrument defect.

## The pattern, stated plainly (2026-08-25)

**Top-3 teams run a private iteration ladder on agent/harness code, submit exactly one draw per day as a measurement, sit flat while the next change is built, and step +0.6..+2.4 the night a change lands.** The board is their eval harness, not their lever. Three independent lines of evidence:

1. **The flats are the control arm [V].** Every top team's history contains multi-day flats WHILE submitting daily (cstl 5 days at 3.57 then +2.42; LHS 6 days at 2.76 then +0.60, 2 days at 3.36 then +1.63; Tufa 5 days at 1.62, 7 at ~3.0). If draws-at-fixed-config paid, their own flats would drift up. They don't. Steps arrive on single draws with no sub-count spike — the opposite of best-of-N grinding.
2. **The machinery is private [V].** Nothing above 2.23 has EVER had a public artifact; the three teams above 4.0 have **zero public artifacts between them**; the forum produced zero mechanism topics through the entire 3.0→6.0 breakout. The recipe channel (kernel/dataset attach graph) distributes the ~1.5–2.2 FLOOR only.
3. **Architecture-over-model holds at both ends of the scale [V].** Ceiling: AVO (Opus 5, text-only + supervisor, 100.00 in 6,624 actions), VISTA (Opus 5 + lossless visual memory, 100% in 7,542), and now **Schema (~99% claimed, Opus 4.8 / Fable 5; 95.35% GPT-5.6 Sol)** all hit ~100% where the bare model gets ~30% — the model is constant, the harness is the variable. Floor: the Q38 engine wave moved the duck band's median by ZERO over 9 days.

### ★ SHARPENING 2026-08-25 — the mechanism class is REFUTATION AGAINST RECORDED HISTORY

Yesterday's class was "surface a hidden affordance." Today it resolves into something narrower, better evidenced, and **measured on our own rail**. Four independent instances:

| system | scale | the mechanism, in its own words | tag |
|---|---|---|---|
| **Schema** (Impossible Research, Zeng/Zanette) | ~99% claimed, frontier | models "act like physicists: write each game's mechanics as an executable program, **verify it against recorded history**, and plan inside it using search" | [V-doc], self-reported, NOT ARC-verified |
| **Executable World Models** (arXiv 2605.05138) | frontier | agent maintains an executable Python world model, **verifies it against previous observations**, refactors toward simpler abstractions (MDL proxy) | [V-doc] |
| **AVO supervisor** (NVIDIA) | 100.00, 6,624 actions | fires on stalls and **"unproductive cycles: edits that repeatedly fail to improve scores"** — no thresholds disclosed | [V-doc] |
| **`lawbook`** (tennant, duck v24, 08-24) | OUR rail, 27B | "the stock prompt demands a revised world model every turn… **nothing in the harness ever checks that model against the board** — so a law induced on turn 4 stays in the carried world model long after the game stopped obeying it" | [V, source-read] |

**The class is NOT "predict better."** Every system in it either predicts nothing new (lawbook is retrospective-only by design, explicitly to avoid `deadend`'s failure) or predicts only inside a model it has already falsified against history. **What they share is a falsification step the stock duck harness does not have at all.**

`lawbook` also supplies the NEGATIVE half, prequentially scored on **8 real commit-run game-passes / 647 actions under the real 27B analyzer** [V-doc] — these are prior-killers for our own design space, free:
- **Board-keyed prediction: DEAD.** `(board, action)` key recurs on 2.6% of actions; on recurrence the environment agrees with itself 47%. Reproduces `deadend`'s 0.486 and retires `schema_lite`'s `MemoWorldModel` (its 99.47% came from a random policy).
- **Global effect signatures: DEAD.** Changed-cell count, palette swaps, bbox, whole-board translation all at-or-below base rate; whole-board translation never fires once in 647 actions.
- **Object-level laws: ALIVE.** run≥2 → 32.3%/83.7%; run≥3 → 25.8%/87.4%; **run≥5 → 18.2% coverage at 92.4% precision.** Self-selects by game (m0r0 102/148 at 95.1%; tn36 near-silent).

**Priced counter-evidence, kept in front:** the distributor of every graft in this class sits at **1.54 on 34 subs, #332** [V] — below our own 1.92 — and **not one graft has any board validation.** An independent implementation of the same idea (`darkmatternet/arc3-duck-causal-guard-v1`, 08-25) sits at **1.02 on 2 subs, #680** — and it *prescribes* ("avoid lists", "causal control rules"), which is the failure mode lawbook was designed to avoid. **The mechanism class is well-evidenced as a description of what the ~100% systems do; it is NOT evidenced as a thing that pays on the 27B rail.**

### ★ SHARPENING 2026-08-26 — before you can refute a belief against history, the history must be READABLE

A fifth instance of the class arrived, and for the first time it is a claim we could **verify on our own data rather than take on report**. `thtennant/arc3-duck-v25` (08-25 12:47) ships `winframe`, whose thesis is not about better inference at all — it is that the lineage's *record* is corrupt at the one instant that carries the only ground truth the game ever volunteers about its objective.

**The claim** [V-doc]: `GameState.frame` returns `Frame(data=self.raw.frame[-1])` — "the final visible frame." On an ordinary step `raw.frame` holds one layer, the board. **On the step that COMPLETES a level it holds two: layer 0 is the terminal board of the level just cleared; layer 1 is the opening board of the NEXT one.** `.ascii`, `.segmentation` and every graft read the last layer. So at the win, the agent is handed the *next* level's board and told it is the current one, and the board that actually satisfied the win condition is discarded unread.

**Our replication, 2026-08-26, offline, on `data/human_replays/*.jsonl`** [V, ours] — 6 games with recorded level wins (ar25, cd82, lp85, sb26, sc25, tu93); every level-win step in all six carries extra layers:

| game | layers, ordinary steps | layers at win steps | prev vs **layer 0** | prev vs **layer[-1]** | layer[-1] vs NEXT step's board |
|---|---|---|---|---|---|
| lp85 | **1** (416/424) | **2** (7/8 wins) | 288 cells | **1505** | 93 |
| ar25 | **1** (570/579) | 3 (7/8 wins) | 105 cells | **692** | **0 — exact identity** |
| cd82 | **1** (110/138) | 15–16 (6/6 wins) | **1 cell** | **410** | 51 |

Layer 0 is one small edit from the previous board — it *is* the winning action's result. Layer[-1] is a different board entirely and on ar25 is bit-identical to the next step's board. **Confirmed.**

**Two consequences, both first-class:**
1. **A conclusion in our own 08-24 brief is retired.** It answered busyaprime's film-strip finding with *"our lineage already reads `raw.frame[-1]` — audited, no defect."* That audit covered animation strips **mid-level**; at **level transitions** reading `[-1]` is precisely the defect. This is the `feedback_audit_the_instrument` lesson recurring: the audit was real, its scope was not what the conclusion claimed.
2. **The one-level wall may not be a budget wall** [V-doc, tennant's archive, NOT replicated on ours]: on the two archived passes that cleared a level, the agent won and kept playing **34 and 120 more actions**, clearing nothing, spending them on a goal induced off layer 1 — which then compounded into a false transfer law ("the generator sets the next level's start = previous level's goal position"). This **qualifies** rather than refutes `feedback_decision_budget_binding`: that finding is about ACTING TURNS dying on the 7920 s clock; this is about ENVIRONMENT ACTIONS remaining after a win. Both can hold at once.

**So the class re-states as:** the top tier is not predicting better — it is (a) keeping an honest record and (b) checking beliefs against it. Our lineage has been failing **(a)**, at the single most informative frame in the game. That is a cheaper defect to fix than a capability gap, and it is ours to fix independent of whether any graft is ever imported.

**Counter-evidence ledger moved:** the "graft distributor sits below us" argument is **RETIRED** — Beyond Good and Eval went 1.54 → **1.93** (#332 → #154) on one draw, now one hundredth above our 1.92. But +0.39 on a single draw is inside this board's noise, so this validates *nothing* about lawbook or winframe. **"Zero board validation of any individual graft" survives untouched.**

## What would FALSIFY the standing hypothesis

1. **Private machinery** dies if a top-3 score is ever attributed to a public artifact run unchanged (a 3.0+ kernel with a matching board row, FOYSAL-style). Watch: kernels census daily. **Not observed** — public ceiling 2.23, frozen 7 days; three teams above 4.0, zero artifacts.
2. **Capability steps (vs draw-grinding)** dies if any top-3 rise coincides with a sub-count spike at flat config, or a top team's flat shows steady upward drift. Watch: per-team d/draw in daily lb_diff. **Not observed** — every recorded step is on exactly 1–2 new draws; 72.7% of submitting teams gained nothing this window.
3. **Architecture-over-model** dies if a pure engine swap moves a population, or a top team discloses a bigger model as their edge. Watch: band median. **Not observed** — band flat 9 days through the Q38 wave. (Honest caveat retained: our own 1.33→1.6 step was engine×effort×harness COMPOUND; a strict engine-null is band-level only.)
4. **The whole frame weakens if the rerun environment systematically favours some teams.** *Strengthened as a concern 08-25:* topic 737230 reports **the exact same submission scoring 2.11 then 0.89** [V-doc], and 736578 reports **local 5.0–5.4 → LB 1.4 while duck's local 2.1 → LB 1.4** [V-doc]. So (a) the instrument's same-config spread is large, and (b) local→LB transfer is broken and NOT constant across harnesses. Current reading: env noise is real, large, and roughly symmetric — **it can manufacture a +1.2 one-off, but it cannot manufacture a +2.42 that then HOLDS as a team's best across subsequent days**, and cstl/LHS/Tufa steps have all held. The falsifier is not discharged; it is bounded.
5. **NEW (08-25):** the refutation-against-history class dies as a *transferable* claim if a graft in it (lawbook/clockwatch/winframe/causal-guard) is ever run on our rail against the certified floor and returns null. **Untested — no board validation exists for any of them.** *Update 08-26:* the distributor's own board row rose 1.54 → 1.93 on one draw, which removes a counter-argument but supplies no validation (single draw, inside noise).
6. **NEW (08-26):** "local→LB transfer is broken and rail-side, not capability-side" would die if a team ever reported a local score that tracked its LB score proportionally. **Not observed — and the counter-evidence is now three-deep:** duck local 2.1 → LB ~1.4; Pellegrin's own harness local 5.0–5.4 → LB ~1.4; daoviet local **6.8 → public 1.19** [V-doc, topic 732854, 08-26]. Local scores span 3× and LB scores do not move. This is the shape of a rail-side constraint, and it prices every screening-rail gain.

## Ledger of tier-mechanism candidates (updated 08-26)

1. Newer harness generation / agent-code iteration — **STRONGEST** (cstl +2.42, LHS +1.63, Tufa +1.54, Tong Hui Kang +1.15, all on 1–2 draws; the ladder pattern, now 4 days unbroken).
2. **Refutation against recorded history** — **#2, promoted 08-25 from "hidden-affordance surfacing"; RE-SCOPED 08-26 to include record INTEGRITY as its precondition**: 5 independent instances, two measured on our exact rail, one (`winframe`'s frame-layer fact) **independently replicated by us on our own archive**. Still zero board validation for any implementation.
3. Private tuning on the Q38-xhigh floor — STRONG, dark (Diya Sharma 2.69 on ONE lifetime sub, 08-25).
4. Grafts/score-mechanics compound — public, measurable, still board-unvalidated.
5. More draws — DEAD (refuted by every top-team trajectory; **22.8%** gainer rate field-wide on 08-26, and on 08-26 the entire top 10 except Tufa drew and gained exactly nothing).
6. Banking-transfer — dead as tier explanation (trigger = a won game; 0/25 field-wide).
7. **RETIRED 08-25 by measurement, do not spend a slot:** board-keyed memoization (`MemoWorldModel` class) and global-effect-signature world models — both refuted on 647 real-27B actions.

**Verdict 2026-08-25: hypothesis CONFIRMED on every testable falsifier, and the mechanism class is now NAMED and NARROWED — the ~100% systems all add a falsification step (verify beliefs against recorded history) that the stock duck harness lacks entirely, and the one public implementation of that idea on OUR rail arrives with its own negative results attached. The class describes what the ceiling does; it is not yet evidence that it pays at the floor.**

**Verdict 2026-08-26: hypothesis CONFIRMED again — the cadence signature held for a fourth straight day (top 10 all flat-while-drawing except Tufa +0.09; two +1.0-class steps from outside it, each on 1–2 draws). The class is RE-SCOPED one step earlier: refutation against recorded history presupposes a readable record, and we verified on our own archive that our lineage's record is corrupt at the level-win frame — the one frame carrying the only ground truth about the objective. That is the first item in this class that is (a) measured on our own data, (b) not a graft we would have to import, and (c) cheap. Two standing arguments were retired today by evidence rather than by preference: "our lineage reads `frame[-1]`, audited, no defect" (scope error — mid-level only) and "the graft distributor sits below us" (1.54 → 1.93). Neither retirement supplies validation for any graft; the board-validation gap is unchanged.**
