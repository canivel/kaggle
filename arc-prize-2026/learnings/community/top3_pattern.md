# TOP-3 PATTERN — living document (created 2026-08-24; update daily)

**Question (principal's standing order):** what is the PATTERN getting teams to top 3?
**Standing hypothesis (from the 08-20 rethink / 08-21 attribution):** private machinery + architecture-over-model + capability steps (not draw-grinding). This document sharpens or refutes it daily with evidence tags [V]/[V-doc]/[INF]/[UNK].

## Current top 5 (board-verified 2026-08-27 10:00 UTC, full archive 2564 rows)

| # | team | score | lifetime subs | cadence (from daily archives) | public artifacts | step history [V] |
|---|---|---|---|---|---|---|
| 1 | cstl | **5.99** | 37 | 1/day, unbroken | none, all campaign | at 2.70 BEFORE Q38 existed [V-doc, Ravindra 08-15]; 3.57 from 08-19, flat 5 days while drawing daily -> **+2.42 (08-24)** <- largest single-draw step of the campaign -> **DREW-NO-GAIN x3 (08-25, 08-26, 08-27)** |
| 2 | Lord Han Solo | **4.99** | 44 | 1/day, unbroken | none (kernels+datasets checked) | 1.65 -> **+1.11 (08-16)** -> flat 6 at 2.76 -> **+0.60 (08-22)** -> flat 2 -> **+1.63 (08-24)** -> **DREW-NO-GAIN x3** |
| 3 | Tufa Labs | **4.67** | 119 | exactly 1/day, unbroken | June-30 harness (frozen), 08-07/08-15 bundles via Bruggen; **`taaf-kaggle-source` republished 08-26 15:20 w/ `polyphony/` now 15 files**; NOTHING for the 2.0+ tier | 1.62 flat (08-14..18) -> +0.45 -> +0.90 -> +0.07 -> flat x2 -> **+1.54 (08-23)** -> flat -> +0.09 (08-25) -> **DREW-NO-GAIN x2** |
| 4 | Tong Hui Kang | **3.39** | 54 | ~1/day | none | 2.24 -> **+1.15 on 2 draws (08-25)** -> flat -> **IDLE (08-27)** |
| **5** | **rfbr** (`romainfabre`, solo) | **3.37** | 13 | ~1/day | none | 2.00 -> +0.19 -> flat -> **+1.18 on ONE draw (08-27)** <- NEW top-5 entrant |
| 6 | Tony G (junvalue) | 3.17 | 13 | sparse | none | 0.30 -> **+1.35 (08-21)** -> flat -> **+1.52/2 draws (08-24)** -> DREW-NO-GAIN x2 -> IDLE |
| **7** | **@Abstraction Lab & MindsAI** (Jack Cole) | **2.94** | 130 | ~1/day, unbroken | none for -3 | **flat at 2.05 for the whole campaign** -- below our 1.33 for 38 draws -- -> **+0.89 on ONE draw (08-27), rank #115 -> #7** <- **CONTROL ARM MOVED** |
| 8 | Daniel Franzen | 2.88 | 53 | ~1/day | none for -3 | steady climber, no single step >0.5 observed; DREW-NO-GAIN 08-24..08-27 |

**Control-arm identity check [V, 4 daily archives]:** the MindsAI row is **TeamId 15490570 with an identical 5-member roster** (`alisalmanrana, jcole75, kimura0415, sumirinn, ultsaza`) on 08-24, 08-25, 08-26 and 08-27. The +0.89 is **a capability step, not a team merge or rename.**

Reference points: FOYSAL 2.23/97 = the PUBLIC ceiling, **frozen since 08-18, 9th day**, rank #55 -> #61 [V]. The 1.55-1.65 duck-floor band: **flat 11 consecutive days** on median (1.60), now 48 teams (-4) [V, lb_diff]. Field-wide 08-26->27: **308 teams submitted, 74 (24.0%) gained anything**, median gainer 0.22/draw. Our own field-floor config drew again: 1.59/1.58/1.63/1.16/1.92/**1.14** -- **n=6, mean 1.5033, sd 0.3010** (was n=5, 1.5760/0.2713 -- the config mean fell 0.07 on its own redraw). **Our EXEC-WM v1 drew 1.05 on 08-26 and the comparator drew 1.14 on 08-27 -- a 0.09 gap, i.e. the board did not separate them.**

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

### * SHARPENING 2026-08-27 -- THE MECHANISM CLASS ACQUIRED A RAIL-MATCHED IMPLEMENTATION, AND IT SCORES 19.8%

For three days this class has been *refutation against recorded history* (08-25), re-scoped to require *record integrity* first (08-26). Today it acquired the thing it had never had: **an implementation of exactly that description, on our engine class, that we could read end to end.**

**Polyphony Agent - ARC** (Ruiyang Yu, Anyang Su, Chenxu Zhao, Tianyu Fu, Shuo Wang, Minghui Wu). Named publicly by **Jakob Bruggen** -- distributor of the bundles our vehicle is pinned to -- in topic 737617 as his best guess at the top-tier jump, *"because it also uses a 27b model under relatively similar circumstances"* [V-doc]; he flagged it as a guess. Its code landed in `jakobbrggen/taaf-kaggle-source` on 08-25 (4 files) and 08-26 (**15 files**), and we downloaded and read it [V]. Loop: **Observe -> Edit -> Plan -> Act**; the agent writes `policy.py` with `predict(state, action)`; `verifier.py` replays **every** observed transition cell-for-cell; **only a certified policy (matched == total, total > 0) reaches the BFS planner.** Engine: **Qwen3.6-27B via vLLM** [V, repo].

**Its published score is 19.8%** [V, arcprize.org/leaderboard/community, 2026-07-07]. On that same board: **Tycho 100.0%, Retrodict 99.9%, baseline1 99.0%, NOOA (NVIDIA) 85.1%, OPINE-World 78.4%.**

**This is the most important thing the class has learned since it was named.** A system whose entire architecture *is* the class description -- verify beliefs against recorded history, refuse to plan over an unverified model -- lands **fifteen places below** a system called `Retrodict` that implements the same idea at 99.9%. So:

> **The mechanism does not pay by being present.** Five instances describe what the ~100% systems do; the one instance we can read implements that description and scores 19.8%. Whatever separates 19.8% from 99.9% is **not** "has a verifier" -- both have one. **Naming a mechanism and shipping a mechanism are different things, and the entire score lives in the gap.**

That is the sharpest available statement of this campaign's recurring failure mode (`feedback_simplicity_wins`; the 0-for-6 adoption sweep), and it should be the standing prior for costing any adoption of this class.

**What Polyphony IS worth is its design decisions, each of which carries a measurement** [all V-doc, read from source]:
- **Retrodiction, not rollout** -- every transition scored from the *observed* prior frame, never the policy's own output, *"a rollout would compound one early error into total failure and report the same thing for a nearly-right policy as for a hopeless one."* Makes `matched/total` a **gradient**. **Indicts our EXEC-WM's latching `MAX_BREAKS_PER_LEVEL = 3`.**
- **Sticky policy deadline `0.55`** -- abandons uncertified model-building for direct play at 55% of the clock, *"the likeliest failure of this arm is not a bad policy, it is an elegant policy and zero actions played."* **An engineered answer to our *** `feedback_decision_budget_binding`.**
- **Bootstrap probe (`probe_cap 12`)** -- plays every action class once before the model sees a token, because *"Q3.8's median to full coverage was nine moves, and in five games it never tried every class at all"* -- a measured claim about **our exact engine**.
- **Anti-gaming verifier** -- replay harness inlined into the sandbox, *"unreachable from the workspace... so the agent cannot loosen the comparison it is judged by"* (cf. `feedback_guard_never_fired`).
- **Thin feedback (20 cells, not 4096)** -- *"4096 numbers the model has been shown not to use on demand"* (cf. `feedback_visible_vs_hidden_channel`).
- Its caps are attributed to **NOOA's two hanging games** -- i.e. its authors studied the 85.1% system's failures.

**COST WARNING [V]:** our vehicle `jakobbrggen/taaf-kaggle-source-anim-20260807-anim` is unchanged since 08-07 and contains **zero** polyphony. The new bundle has **no `inference/agent/` at all** (`tool_agent.py`, 108,927 B) -- replaced by `inference/core/` + `inference/polyphony/`. **Adopting Polyphony is a whole-vehicle swap, not a flag flip** -- the structural drift behind 5 ERROR submissions (`feedback_arc_kernel_structural_drift`). Reimplement the design points on our existing vehicle.

### * SHARPENING 2026-08-27 (b) -- TWO INSTRUMENT GAINS, BOTH VERIFIED BY EXECUTION

This campaign's returns have come from instruments, not mechanisms, and today produced two.

1. **`set_level(index)` -- 182 offline level-openings, free.** Disclosed in duck v26's `carryover` rationale; **executed by us this morning** [V, ours]: `arcengine.base_game.ARCBaseGame.set_level` at `arcengine/base_game.py:148`, against `kaggle-data/environment_files/` -- **25/25 public games opened, 182 levels reachable, 0 GPU, 0 environment actions.** 182 - 25 = **157 pairs** vs tennant's reported 158, an independent match to within one. **We screened on played trajectories all campaign while 182 level-openings sat one import away.** What tennant measured with it over all 158 pairs [V-doc, not yet replicated on ours]: controls keep their effect **587/631 = 93.0%**; openings repeat **0/158 = 0.0%**; objects carry **21.4%**; shapes shared **39.0%** -- **"THE CONTROLS CARRY; THE LAYOUT DOES NOT."**
2. **The comment sweep had been reading truncated text.** The CLI table renderer cuts comment bodies at ~200 chars; `--format json` returns them whole. Bruggen's Polyphony sentence sits at char ~400 and **was invisible in every prior sweep.** `feedback_audit_the_instrument` again: the sweep ran, it was real, and its scope was not what the conclusion claimed.

**And the EXEC-WM verdict largely dissolved on its own** [V]: the certified floor config drew **1.14** on 08-27 -- within **0.09** of EXEC-WM's 1.05, with none of EXEC-WM's code in it. A single board draw did not separate the arms, and the board demonstrated it directly one night later. Third time this month a board number nearly bought a wrong conclusion (cf. `feedback_seed_vs_own_config`, `feedback_screen_calibration_range`).

**Independent replication of two of OUR findings, from tennant's archive** [V-doc, duck v26 `clockwatch`]: *"every archived game ends on the 7920 s per-game wall-clock and NONE on the action cap"* and *"97.6-98.3% of every generated token is reasoning."* The second matches `feedback_visible_vs_hidden_channel`'s 97.6% almost exactly, from a different team's data. **Two independent archives now agree the budget is binding and the channel is hidden.**

## What would FALSIFY the standing hypothesis

1. **Private machinery** dies if a top-3 score is ever attributed to a public artifact run unchanged (a 3.0+ kernel with a matching board row, FOYSAL-style). Watch: kernels census daily. **Not observed** — public ceiling 2.23, frozen 7 days; three teams above 4.0, zero artifacts.
2. **Capability steps (vs draw-grinding)** dies if any top-3 rise coincides with a sub-count spike at flat config, or a top team's flat shows steady upward drift. Watch: per-team d/draw in daily lb_diff. **Not observed** — every recorded step is on exactly 1–2 new draws; 72.7% of submitting teams gained nothing this window.
3. **Architecture-over-model** dies if a pure engine swap moves a population, or a top team discloses a bigger model as their edge. Watch: band median. **Not observed** — band flat 9 days through the Q38 wave. (Honest caveat retained: our own 1.33→1.6 step was engine×effort×harness COMPOUND; a strict engine-null is band-level only.)
4. **The whole frame weakens if the rerun environment systematically favours some teams.** *Strengthened as a concern 08-25:* topic 737230 reports **the exact same submission scoring 2.11 then 0.89** [V-doc], and 736578 reports **local 5.0–5.4 → LB 1.4 while duck's local 2.1 → LB 1.4** [V-doc]. So (a) the instrument's same-config spread is large, and (b) local→LB transfer is broken and NOT constant across harnesses. Current reading: env noise is real, large, and roughly symmetric — **it can manufacture a +1.2 one-off, but it cannot manufacture a +2.42 that then HOLDS as a team's best across subsequent days**, and cstl/LHS/Tufa steps have all held. The falsifier is not discharged; it is bounded.
5. **NEW (08-25):** the refutation-against-history class dies as a *transferable* claim if a graft in it (lawbook/clockwatch/winframe/causal-guard) is ever run on our rail against the certified floor and returns null. **Untested — no board validation exists for any of them.** *Update 08-26:* the distributor's own board row rose 1.54 → 1.93 on one draw, which removes a counter-argument but supplies no validation (single draw, inside noise).
6. **NEW (08-26):** "local→LB transfer is broken and rail-side, not capability-side" would die if a team ever reported a local score that tracked its LB score proportionally. **Not observed — and the counter-evidence is now three-deep:** duck local 2.1 → LB ~1.4; Pellegrin's own harness local 5.0–5.4 → LB ~1.4; daoviet local **6.8 → public 1.19** [V-doc, topic 732854, 08-26]. Local scores span 3× and LB scores do not move. This is the shape of a rail-side constraint, and it prices every screening-rail gain.

## Ledger of tier-mechanism candidates (updated 08-27)

1. Newer harness generation / agent-code iteration — **STRONGEST** (cstl +2.42, LHS +1.63, Tufa +1.54, Tong Hui Kang +1.15, all on 1–2 draws; the ladder pattern, now 5 days unbroken; rfbr +1.18 and MindsAI +0.89 added 08-27).
2. **Refutation against recorded history** — **#2, promoted 08-25; RE-SCOPED 08-26 to include record INTEGRITY as its precondition; PRICED 08-27**: 6 independent instances, two measured on our exact rail, one (`winframe`'s frame-layer fact) **independently replicated by us on our own archive**. **The 08-27 price is the load-bearing update: Polyphony implements the class exactly — certified-policy-gated planning on Qwen3.6-27B, read end-to-end by us — and publishes 19.8%, while `Retrodict` publishes 99.9% and `Tycho` 100.0% on the same board. So "has a verifier" is NOT the discriminator. Cost any adoption of this class against a mid-table reference, not a ceiling one.** Still zero board validation for any implementation on the Kaggle rail.
3. Private tuning on the Q38-xhigh floor — STRONG, dark (Diya Sharma 2.69 on ONE lifetime sub, 08-25).
4. Grafts/score-mechanics compound — public, measurable, still board-unvalidated.
5. More draws — DEAD (refuted by every top-team trajectory; **24.0%** gainer rate field-wide on 08-27, median gainer 0.22/draw, and **26 of the top 30 gained exactly 0.00 while submitting** — 5 days unbroken).
6. Banking-transfer — dead as tier explanation (trigger = a won game; 0/25 field-wide).
7. **RETIRED 08-25 by measurement, do not spend a slot:** board-keyed memoization (`MemoWorldModel` class) and global-effect-signature world models — both refuted on 647 real-27B actions.
8. **NEW 08-27 — OUR OWN INSTRUMENT STOCK, not a tier mechanism but where this campaign's returns actually come from.** `set_level(index)` opens **25/25 public games, 182 levels, 157 pairs offline at zero cost** [V, executed by us]; and every prior comment sweep was reading **truncated** text (CLI table renderer caps bodies at ~200 chars; `--format json` is whole). Both found by reading a rival's rationale rather than by running an arm.

**Verdict 2026-08-25: hypothesis CONFIRMED on every testable falsifier, and the mechanism class is now NAMED and NARROWED — the ~100% systems all add a falsification step (verify beliefs against recorded history) that the stock duck harness lacks entirely, and the one public implementation of that idea on OUR rail arrives with its own negative results attached. The class describes what the ceiling does; it is not yet evidence that it pays at the floor.**

**Verdict 2026-08-26: hypothesis CONFIRMED again — the cadence signature held for a fourth straight day (top 10 all flat-while-drawing except Tufa +0.09; two +1.0-class steps from outside it, each on 1–2 draws). The class is RE-SCOPED one step earlier: refutation against recorded history presupposes a readable record, and we verified on our own archive that our lineage's record is corrupt at the level-win frame — the one frame carrying the only ground truth about the objective. That is the first item in this class that is (a) measured on our own data, (b) not a graft we would have to import, and (c) cheap. Two standing arguments were retired today by evidence rather than by preference: "our lineage reads `frame[-1]`, audited, no defect" (scope error — mid-level only) and "the graft distributor sits below us" (1.54 → 1.93). Neither retirement supplies validation for any graft; the board-validation gap is unchanged.**

**Verdict 2026-08-27: hypothesis CONFIRMED for a fifth straight day, and the mechanism class was PRICED for the first time.** The cadence signature held hard -- 26 of the top 30 gained exactly 0.00 while submitting, and both real steps (rfbr +1.18 on 13 lifetime subs, MindsAI +0.89 on 130) arrived on a single draw with no sub-count spike, from outside the top 5. The **control arm moved**: @Abstraction Lab & MindsAI, verified same-TeamId/same-roster across four archives, stepped +0.89 to #7 after sitting at 2.05 the entire campaign -- pedigree still doesn't predict this board, but it plainly doesn't preclude a step either. The decisive new evidence is a **price on the leading mechanism class**: Polyphony implements the class exactly, on our engine class, and publishes **19.8%** while `Retrodict` publishes 99.9% -- so "has a verifier" is not the discriminator, and adoption of this class must be costed against a mid-table reference, not a ceiling one. Two instrument gains landed instead, both verified by execution rather than report (`set_level`: 25/25 games, 182 levels, free; and the discovery that every prior comment sweep read truncated text). **Falsifier 5 remains untested -- still zero board validation for any graft in this class -- and falsifier 1 (private machinery) is untouched: the public ceiling is FOYSAL 2.23, frozen 9 days, while seven teams sit above 2.9 with zero public artifacts between them.**

---

### ★ ADDENDUM 2026-08-27 (second, independent community pass, 13:26Z) — FALSIFIER #6 FIRES

Source: `brief_2026-08-27_ADDENDUM.md`. Three changes to this document's standing claims.

**1. FALSIFIER #6 IS DISCHARGED — it fired.** It read: *"'local→LB transfer is broken and rail-side,
not capability-side' would die if a team ever reported a local score that tracked its LB score
proportionally. **Not observed.**"* It is now observed. Topic 732854, **mikelou1, 08-26 11:03Z**
[V-doc]: *"Got 2.8 on 25 games and 2.4 on lb."* Author resolved to the board: **team "Proving AGI",
rank #34, 2.43, 26 subs** [V] — LB half independently confirmed, and **above the frozen public
ceiling** (2.23), so not a duck-floor artifact.

| team | local | LB | ratio |
|---|---|---|---|
| **mikelou1 / Proving AGI** | **2.8** | **2.4** (board 2.43) | **0.87** |
| duck harness (reported) | 2.1 | ~1.4 | 0.67 |
| Pellegrin's own harness | 5.0–5.4 | ~1.4 | ~0.27 |
| daoviet (board 1.99) | 6.8 | 1.19 | 0.17 |

**Restated law: the collapse is a property of over-fitted local harnesses, not of the rail.** The
relation is inverse — the higher the local score, the worse the transfer. Consequence for us: the
0-for-36 screening record (`feedback_screen_calibration_range`) is evidence about **our screen**, not
proof that screening cannot work. Calibration target for any future screen: an absolute number near
our LB (~1.9–2.4). **A local screen reading 5+ is a red flag about the screen, not a result.**

**2. The cadence evidence for the 5th "flat day" is WEAKER THAN RECORDED, on two counts** [V].
(a) `lb_diff.py 08-26→08-27` **could not run** — the 08-26 full archive never migrated from Windows
(bare `*.csv` in `.gitignore`) — and it **exits 0 on failure**, which is how "exit 0" was recorded.
So *"308 submitted, 74 (24.0%) gained, median gainer 0.22/draw"*, *"26 of 30 gained exactly 0.00"*
and *"19 in, 1 out"* have **no supporting artifact on this box** and should be struck; only the ~14
teams with an endpoint in yesterday's brief table or the 08-26 heartbeat carry a real Δ.
(b) The 10:00Z pull happens **before the previous night's draws finish scoring** (`SubmissionCount`
does not increment until a run completes, and runs take up to 9 h). Measured: **Tong Hui Kang read
IDLE 3.39/54 at 10:00:40Z and was 3.88/55 at 13:26:10Z — a +0.49 top-5 STEP recorded as idle.**
5 of the top 30 submitted inside the at-risk window. **The daily gainer rate is a floor, not a point
estimate, and it is biased toward showing more flats — i.e. toward confirming this document's own
hypothesis.** The hypothesis still stands on step SIZES (MindsAI +0.89 [V], both endpoints in the
heartbeat), but the flat-day counting must be re-derived from 08-28, when a real diff is possible.

**3. Ledger entry 2 ("refutation against recorded history") takes its first two rail-side hits.**
Both from the 08:48 iterate run (commit `46afb34`), which ran *before* the community brief and so is
not reflected in it: **EXEC-WM's BREAK-clustering hypothesis is REFUTED by its own artifact** — the
latch fired **zero** times and a mislabelled reason string caused two days of wrong diagnosis; the
real defect is **data starvation** (26/32 level-instances at no-verified-model, 9/18 games with zero
transitions) while **retrodiction was 810/818 where data existed**. That ratio is the useful part:
*the belief-checking machinery works; the record feeding it is empty.* This **sharpens** the 08-26
re-scoping (record integrity is the precondition) from a frame-*selection* bug to a frame-*supply*
bug. And **Polyphony's sticky 0.55 deadline is refuted pre-build** — 39.3% of our floor level
completions land after that mark. **The one rail-matched implementation of this mechanism class
publishes 19.8%, and the first of its constants we priced against our own data was harmful.**

---

### VERDICT 2026-08-28 — hypothesis CONFIRMED for a sixth day; the mechanism class INVERTS; and the top tier says out loud that it is private

**Cadence (first REAL diff in three days — both sides are full archives on this box).** 282 teams
submitted, **52 (18.4%) gained anything**, median gainer **0.20/draw**. Three genuinely new steps, all
on a single draw with no sub-count spike: **OzanM. +0.81** (2.17→2.98, #8), **Scott Le Grand +0.31**
(#58), **Daniel Franzen +0.16** (#7). **cstl, the #1 at 5.99, did not submit at all.** Control arm
**did not move for a sixth day**: Jack Cole/MindsAI +0.00 on +1, Tufa Labs +0.00 on +1 — `lb_diff`
readout unchanged: *"the commodity-engine / shared-regime story is WEAK on this evidence."*

**The ADDENDUM's lag bias reproduced itself exactly, one day later** [V]. Tong Hui Kang's +0.49 to
#4 appears in today's diff as a STEP; it is **not new** — the ADDENDUM dated it to before 13:26Z on
08-27, and it re-surfaces only because the archived 08-27 pull was taken at **10:00Z** while his run
landed at **01:11Z** and had not finished scoring. **Excluded from today's step count.** The
correction is now demonstrated rather than predicted: the daily gainer rate is a **floor**, biased
toward more flats, i.e. toward confirming this document's own hypothesis. Expect the same correction
against today's flags tomorrow.

**FALSIFIER 1 (private machinery) TAKES ITS FIRST DIRECT TESTIMONY — and it holds.** Topic 732854,
**OverfitOracle, 08-27 18:20** [V-doc], asked point-blank by Spen whether his jump came from
*"fundamental changes to your harness … or just tweaks and a good scoring run"*: *"I am not gonna
share something very important but was a **very different approach in the harness + model** achieving
5.0+ stable on public 25 games. **We completely redesigned it** leading to a good stable increase."*
This moves one top-tier row from [UNKNOWN] to **[INFERRED: private whole-system redesign, not a flag
or a graft]** on the author's own statement, and it **actively rejects** Brüggen's 08-26 conjecture
that one public idea leaked to three teams simultaneously. The public ceiling is still **FOYSAL 2.23,
frozen for 10 days**, with eight teams above 2.9 and zero public artifacts between them.

**★ THE MECHANISM CLASS INVERTS.** For four days the leading class was *refutation against recorded
history* (exec world models, verifiers, retrodiction). Today's external radar establishes that **the
two independent 100%-class systems chose AGAINST it on purpose** [V, NVIDIA AVO blog, verbatim]:
*"Rather than centering our ARC-AGI-3 system on explicit programmatic world-model construction, **as
explored by Tycho**, we adopted the direct-interaction design principles described by VISTA."* AVO's
four parts are a main loop, **persistent memory as search-state continuation** (*"resume from the
current state rather than repeatedly reconstructing the search"*), a **supervisor that detects
stagnation and redirects the agent**, and swappable tools — on **Claude Opus 5, text-only 64x64
grids, no code or weights released, public set only, reimplemented task interface**. Retrodict
(99.9%, fully open) reaches comparable numbers via replay-verification but relegates the executable
simulator to a **fallback after 300 stuck actions**. **Restated class:** the world model is *one*
route and the ceiling systems declined it; what they share is **memory that survives across
invocations** plus **a mechanism that notices the agent is not progressing**. Note this lands on our
own live thread — 08-27's finding that context drives reasoning 19x, that naive trimming is 4.9x
worse per action, and that the indicated arm is **summary-carrying**.

**Instrument corrections, both retiring claims this document has relied on** [V]:
1. **The topic sweep has been under-paged ~3x all campaign** — 193 topics on exhaustive paging, not
   the 61 (or 40) that prior briefs called "all topics".
2. **`arcprize.org/leaderboard/community` is DATE-sorted, not score-sorted.** The 08-27 reading that
   Polyphony *"is fifteen places below a system that scores 100%"* was row order, not rank. The score
   facts (Polyphony 19.8%, Tycho 100.0%) survive; the framing does not.

**★★ NEW ENTRY TO THE LEDGER — the first rival artifact that is graftable to OUR vehicle with zero
structural risk, and the first that CLOSES a lever family for us.** `yocybercode` (team
**Thuitanium**, #310, 1.70, 13 subs) published **six kernels in one three-hour burst on 08-27**,
forming a **2x2 factorial + baseline + byte-identical replicate** over `LOCAL_ANALYZER_SEED`,
`YIELD_SECONDS` (60→180) and `TEMPERATURE` (0.6→1.0) — every one pinning
**`jakobbrggen/taaf-kaggle-source-anim-20260807-anim`, our exact vehicle**, with the solver untouched
and every intervention a `str.replace` in the setup command. **All levers verified present in our own
bundle at their baseline values; `SEED` is ABSENT, which is why v1-1 injects rather than replaces**
[V, read from our `setup_commands.json`]. Each version ships **build-time TEETH** that abort if the
injection silently no-ops (*"the duckv25 shape"*), if a key is injected twice (*"two values would
race"*), or if an untested variable drifted — `feedback_guard_never_fired` and
`feedback_verify_treatment_can_fire` as five lines of assertion.
**And their R35 probe closes a family for us** [V-doc]: the output-token distribution has **no fat
tail** — *"a cap at 8,192 saves 0.98% of output; at 12,288 it saves nothing at all"* — with
`duckv9`'s 768-cap scoring **0.22** because `finish_reason` came back `length` 704 against
`tool_calls` 68, *truncating the tool call that carried the action*. Combined with our own 08-27
refutation of context trimming (4.9x worse per action), **"cheapen the decision to buy decision
budget" is now dead on two independent instruments.** The remaining lever is value-per-turn.

**A third team independently derives our selection discipline** [V-doc]. Scott Le Grand, topic
737230, 08-27: *"The variance on submissions of the same notebook is insane. **I have to submit at
least 4 times before I remotely believe differences between 2 approaches.**"* Spen supplies the
mechanism: *"depending on scheduling and timeouts … certain games get more time than others … the
exact same submission can get a much lower score."* This confirms `feedback_seed_vs_own_config` from
outside, **and prices the Thuitanium ladder**: six single draws cannot separate six configs. It also
predicts a ceiling on seed-pinning — if the dominant variance is **scheduler** rather than
**sampler**, `LOCAL_ANALYZER_SEED` will not collapse it, which is precisely what v1-1 vs v1-1-r2
exists to measure.

**A seven-report public→LB divisor table, the first EXTERNAL calibration of our screen** [V-doc, all
topic 732854]: Nick Pellegrin 5.0–5.4→1.4–1.8 · OverfitOracle 5.0+→1.6 · daoviet 6.8→1.19 · Fususu
3.8→0.9–1.8 · Scott Le Grand 3.8→~0.9 · mikelou1 2.8→2.4 · donk666 3.5–7.5. **Central tendency ~3x**,
and it sharpens the 08-27 ADDENDUM's inverse law rather than contradicting it: the higher the local
score, the worse the transfer. **A local 6.173 is a ~2.0 board expectation, not a 6.**

**FALSIFIER 5 STILL UNTESTED for a sixth day: zero board validation of any individual graft.**
Thuitanium's ladder sits at 1.70 and its per-version scores are not observable through the CLI, so it
validates nothing yet either — it is a **design we can read, not a result we can read**.
