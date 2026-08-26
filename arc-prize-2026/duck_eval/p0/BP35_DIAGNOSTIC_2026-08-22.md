# BP35 / R11L / SP80 DIAGNOSTIC — why the field floor scored 0 where a weaker config clears ≥65%
2026-08-22 · P0 instrument week · zero GPU, benchmark.json forensics only
Sources: `runs/tufa_example_run/benchmark.json` (reference, label `0-history-turns`, 20 passes × 25 games, 2026-06-02) vs `runs/kernel_pulls/{q38_field_v1, private_base_v1, budget_t05_v1, budget_t3_v1}/benchmark.json` (ours, n_passes=1 each).

## VERDICT IN THREE LINES

1. **[V] The agent NEVER REACHES the state the reference reaches — it is not failing at a mechanic.** Per-turn behaviour (action classes, burst size, actions-to-clear when it does clear) is indistinguishable from the reference; the field floor simply ran out of clock after 7–15 acting turns where the reference gets ~45–50.
2. **[V] The failure IS behavioural and IS shared — but the shared behaviour is upstream of all three games:** ~6–8× more generated tokens per acting turn (6.4k–9.6k vs ref ~1.0–1.3k), which starves every run of actions. The per-game "0" lines are the downstream shadow of one cadence defect.
3. **[V] This licenses ONE targeted arm (turn-cadence / deliberation budget), not three game arms.** At the field floor's realised action counts, the reference policy itself would have gone 0/0/0 with probability ≈ 24% — the coordinator's 65%-vs-0 contrast dissolves once you condition on actions taken.

## 1. The 0s do not replicate as 0s — they replicate as a budget dose-response **[V]**

L1 cleared? (all four runs end `state=gave_up`, i.e. clock-killed):

| run | clock | bp35 | r11l | sp80 | L1 clears |
|---|---|---|---|---|---|
| budget_t05_v1 | 3,960 s | 0 | **1** | 0 | 1/3 |
| **q38_field_v1** (certified floor) | 7,920 s | 0 | 0 | 0 | 0/3 |
| private_base_v1 | 7,920 s | 0 | **1** | **1** | 2/3 |
| budget_t3_v1 | 23,760 s | **1** | **1** | **1** | **3/3** |
| reference (20 passes) | up to ~21.8k s, ~120–300 actions/pass | 18/20 | 18/20 | 13/20 | — |

Monotone in budget; t3 clears all three with the same policy and same game builds (`hint`, `number_of_levels`, `base_actions_per_level` identical across files **[V]**). A stuck mechanic or wrong action class would not yield to 3× clock.

## 2. Same actions, same bursts, ~7× the tokens per turn **[V]**

Acting turn = history entry with `generated_tokens > 0`; the following `generated_tokens == 0` entries are the rest of that turn's action batch (entry token sums reconcile to `solver_note` totals, e.g. bp35 field 55 entries summing to 100,782 = solver_note).

| game | metric | reference (median of 20) | q38_field_v1 |
|---|---|---|---|
| bp35 | acting turns / actions | 45 / 188.5 | **13 / 55** |
| bp35 | actions per turn | 4.1 | 4.2 |
| bp35 | tokens per turn | 1,009 | **7,752** |
| r11l | acting turns / actions | 49.5 / 54.5 | **7 / 7** |
| r11l | actions per turn | 1.0 | 1.0 |
| r11l | tokens per turn | 1,265 | **9,577** |
| sp80 | acting turns / actions | 46 / 182.5 | **15 / 62** |
| sp80 | actions per turn | 3.6 | 4.1 |
| sp80 | tokens per turn | 1,177 | **6,384** |

Action-class mix matches the reference's *clearing* passes game-for-game **[V]**:
- **r11l**: 100% `ACTION6` clicks on both sides (ref clears: 43–82 clicks, ours: 7–16 clicks).
- **bp35**: `ACTION3/ACTION4/ACTION6` mix on both sides (ref clear example: 14×A4+18×A3+158×A6; ours: 27×A4+10×A3+18×A6).
- **sp80**: movement-dominated `ACTION1–5` with ≤6 clicks on both sides.

No wrong action class, no missing `ACTION6.set_data` (our r11l clicks carry real x/y), no format failures.

Token totals per run are actually HIGHER than the reference's (bp35 100,782 vs ref mean 46,538; r11l 80,069 vs 61,563; sp80 98,596 vs ~58k) **[V]** — we spend a comparable-or-larger generation budget and convert it into 3–8× fewer environment actions. The extreme: the r11l field pass generated **41,513 tokens between action 6 and action 7** (a single inter-action deliberation block exceeding the 31,744 context budget — inspection-loop churn, cf. §1.1 B1's 62% inspection-only generations) and then the clock fired at 7,920 s with 7 clicks on the board.

## 3. Conditioned on actions taken, the reference would ALSO have failed **[V]**

Reference empirical P(clear L1 within N actions), evaluated at the field floor's action count at death:

| game | field actions at death | ref P(clear ≤ N) | ref median actions-to-clear L1 |
|---|---|---|---|
| bp35 | 55 | 0.55 (11/20) | 51.5 |
| r11l | 7 | 0.10 (2/20) | 14.5 |
| sp80 | 62 | 0.40 (8/20) | 55 |

Expected clears for a reference-quality policy truncated at our action counts: 0.55+0.10+0.40 = **1.05**; P(0/3) = 0.45×0.90×0.60 ≈ **0.24**. The certified floor's 0/0/0 is an unremarkable draw *given the truncation*; the anomaly is entirely in the action count, not in what the agent does with its actions.

And when our agent does get enough turns, it clears with FEWER actions than the reference median **[V]**: r11l apl0 = 4 (t3) / 7 (private_base) / 12 (t05) vs ref median 14.5; sp80 apl0 = 9 (private_base) vs ref median 55; bp35 apl0 = 38 (t3) vs ref median 51.5. The policy is action-EFFICIENT; it is action-STARVED.

## 4. Degeneration signatures (§1.1 / §1.4 of the per-turn program)

- **C2 immediate-repeat share**: bp35 field 55.6% with a 16-long `ACTION4` streak — but the reference's own *failing* bp35 passes show 62.5–65.7% repeats with streaks of 34 and 54, while its clearing passes show 8–37% **[V]**. High repeat share is what a bp35 failure looks like under ANY policy (a movement game flailing), not a signature unique to ours. sp80 repeat shares are high even in reference clears (48–68%) — burst-movement by design, not diagnostic.
- **§1.4 no-tool-call generations**: not measurable from `benchmark.json` (generation-level, lives in transcripts); the `generated_tokens==0` history entries are batched burst actions and appear equally in reference clears (e.g. 150 of 196 entries in a clearing bp35 pass) — do NOT read them as "no-action generations" **[V for the artifact; the §1.4 signature itself is untestable here]**.
- **C3 click-coordinate reuse**: the field floor is CLEANER than the reference (bp35 2/18 duplicate clicks vs ref 65–113 of ~100–160; r11l 2/7 vs ref 9–46) **[V]** — mostly because it never lived long enough to loop. The signature does appear at t3 on r11l's L2 dead phase (94/160 duplicate clicks, 10 RESETs after clearing L1 at action 4) **[V]**, consistent with §1.4 long-run degeneration — but that is post-L1 and post-budget-extension, irrelevant to the field-floor 0s.

## 5. Corrections this forces on existing docs

- **§1.3's GRINDER classification of bp35 (55 actions) and sp80 (62) is wrong at level 1** **[V]**: both died at/just past the reference's median actions-to-clear (51.5, 55), and t3 cleared both (bp35 at 38 actions — under its field action count). They were TRUNCATED, same as r11l; "budget is pure waste on them" does not hold for their L1. (The grinder label may still hold for their later levels.)
- **§1.5's "lost to draw noise"** is imprecise: the loss is not clone-lottery noise on an equal footing — it is a systematic ~4–7× turn-cadence deficit that pushes every pass into the far-left tail of the reference's actions distribution, where failure is the mode. Consistency work aimed at *per-game decisions* will not recover these three; cadence work will.

## 6. Answer to the coordinator, verbatim terms

- **FAILS DIFFERENTLY or NEVER REACHES?** → **NEVER REACHES.** At death: bp35 55 actions / 13 turns (ref clears use ~120–300 actions), r11l 7 actions / 7 turns (ref median-to-clear alone is 14.5 actions), sp80 62 actions / 15 turns. No wrong action class, no stuck mechanic (t3 clears all three), no earlier per-game timeout — the runs used their full shared clock (last actions at 7,817 / 6,060 / 7,427 s of 7,920) but bought ~10× less board interaction per second with it.
- **BEHAVIOURAL AND SHARED?** → **YES, shared — one behaviour, upstream of all three games**: ~6.4k–9.6k generated tokens per acting turn vs the reference's ~1.0–1.3k. What it licenses is a single **turn-cadence / deliberation-budget arm** (cap inter-action deliberation, force an action every K tokens, or the §2.5 "convert surplus tokens into extra attempts" inversion) — NOT three per-game mechanic arms, and NOT more prompt content (which raises tokens/turn further, the exact wrong direction).
- **Residual heterogeneity [INF]**: r11l is pure truncation (unambiguous — 7 actions vs 14.5 needed at median). bp35/sp80 field passes sat near the ref median-clear point and still failed with repeat shares at the failing-pass end (0.56/0.66); at n=1 this is indistinguishable from the 45%/60% baseline failure probability at those action counts. Nothing here supports a game-specific defect.

## Provenance
- Reference: `F:\kaggle\arc-prize-2026\runs\tufa_example_run\benchmark.json` (+ `run_config.json`, `summary.txt`; slurm 2×GPU, 32 concurrent jobs/GPU — per-run wallclock includes serving contention, so seconds/turn is not directly comparable across rails; turns/tokens/actions are).
- Ours: `F:\kaggle\arc-prize-2026\runs\kernel_pulls\{q38_field_v1,private_base_v1,budget_t05_v1,budget_t3_v1}\benchmark.json`.
- Taxonomy referenced: `learnings/war_room/perturn_program_2026-08-22.md` §1.1, §1.3, §1.4, §1.5, §2.5.
- All counts recomputed from raw `history` streams (action id + data, `generated_tokens`, `wallclock_seconds`); acting-turn definition validated by reconciling entry token sums to `solver_note` totals.
