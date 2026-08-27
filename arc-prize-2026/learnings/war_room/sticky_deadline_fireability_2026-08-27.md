# STICKY POLICY DEADLINE — REFUTED ON FIREABILITY, PRE-BUILD, FOR FREE

**Proposed arm:** Polyphony's `sticky policy deadline = 0.55` re-implemented on our pinned vehicle.
**Source:** `learnings/community/brief_2026-08-27.md` handoff #3 — *"if only one point is built today,
build the sticky policy deadline"*, ranked highest of the five Polyphony design points.
**Status: NOT BUILT. Killed by its own pre-registered fireability check. Zero GPU, zero slots spent.**

## THE PROPOSAL AND WHY IT LOOKED STRONG

Polyphony abandons uncertified model-building for direct play at **55 % of the game clock**. Its
stated rationale: *"the likeliest failure of this arm is not a bad policy, it is an elegant policy
and zero actions played"* — without it, *"the likeliest outcome is '25 games, 0 actions, score 0',
which teaches nothing."*

This read as an **engineered answer to our own ★★★ finding** (`feedback_decision_budget_binding`):
675/675 game-runs die on the 7920 s wall at ~17 turns against a designed 132. ~20 lines of code
aimed squarely at our best-evidenced constraint. It deserved to be built — **if** it could fire.

## THE CHECK (standing rule `feedback_verify_treatment_can_fire`)

Measured on **retained real artifacts** before any build: for every level completion in
`runs/kernel_pulls/*/benchmark.json`, the cumulative `actions_per_level` indexes into `history[]`
to give the `wallclock_seconds` at which that level completed, normalised by that run's **own**
final wallclock.

**Question:** what share of our actual level completions arrive in the back 45 % of the clock —
i.e. in the region a 0.55 deadline would change?

| corpus | game-runs | completions | **after 0.55 of clock** | last-level after 0.55 |
|---|---|---|---|---|
| **certified field floor alone** | 25 | 28 | **11/28 = 39.3 %** | 8/17 = **47.1 %** |
| standard-budget arms (floor + P1 + P2 + exec-WM) | 100 | 104 | **33/104 = 31.7 %** | 27/67 = 40.3 % |
| all incl. the 3× T3 budget arm | 125 | 139 | 42/139 = 30.2 % | 35/89 = 39.3 % |

Distribution of completion times (per-run normalised, n=139):
p10 = 0.051 · p25 = 0.134 · **p50 = 0.310** · p75 = 0.609 · p90 = 0.850.

## VERDICT: REFUTED. THE BACK HALF OF OUR CLOCK IS NOT DEAD TIME.

**On the certified field floor — the vehicle any such arm would be built on — 39.3 % of all level
completions and 47.1 % of last-level completions land after the 0.55 mark.** Nearly half the floor's
scoring events happen in the region the deadline governs. A mechanism that curtails behaviour there
is not reclaiming waste; it is **cutting into live scoring**.

And the failure mode the deadline exists to prevent **does not describe us**. Polyphony guards against
*"an elegant policy and zero actions played."* Our agent plays **2081 actions** and completes **30 %
of its levels in the back half**. We do not have a paralysis problem. We have a throughput problem —
and those call for opposite interventions.

There is also a **structural mismatch**, which the fireability number makes moot but which is worth
recording: Polyphony's deadline switches between two *phases* (uncertified model-building → direct
play). **Our vehicle has no such phase split** — the agent plays directly throughout. There is no
model-building phase to abandon, so the mechanism has no distinct thing to cut. Porting it would have
meant inventing a phase boundary and then cutting it, which is a different arm than the one whose
rationale we were borrowing.

## INSTRUMENT AUDIT (run before the numbers were trusted)
- `wallclock_seconds` verified **monotonic in all 125 game-runs** (0 non-monotonic series).
- An apparent 2.24× outlier resolved cleanly: it is the **T3 3×-budget arm** (final wallclock 23 796 s
  = 3 × 7920), correct by construction. Re-normalising per-run rather than against a global 7920 s
  changed the headline by **1.5 pp** (30.2 % vs 29.5 %) — the finding is not an artefact of the
  denominator.
- The 7920 s wall is **soft**: 25/125 runs exceeded 8400 s. Recorded; does not affect this read.
- Result reported separately for the floor alone, standard-budget arms, and the full pool. **The
  floor-alone number is the strongest against the arm**, which is the honest way round — the vehicle
  the arm would ship on is the one where the deadline would do the most damage.

## WHAT THIS COST AND WHAT IT BOUGHT
**Cost:** one measurement, ~10 minutes CPU. **No slot, no GPU, no build, no submission.**

**Bought:** an arm killed before it consumed a slot — and, more usefully, a **reusable prior**: our
level completions are distributed with a median at **0.31** of the clock and a long tail to 0.85+.
Any future arm that proposes to *truncate, re-phase, or re-budget the back half of the game clock*
must now clear this table first. Efficiency-family arms remain arithmetically capped
(`+0.06 LB max`, strategy-0822 cap theorem); this adds the empirical complement — **the back half is
not slack, it is where 30–40 % of our score is made.**

## PROCESS NOTE — THIS IS TODAY'S P2 LESSON, APPLIED THE SAME DAY
P2 died today because its D1 (*will the trigger FIRE?*) was pre-measured on retained artifacts and
passed, while its D2 (*will the model USE it?*) was never pre-measured and failed. The post-mortem
demanded: **pre-measure the thing that can kill the arm, before building.** This arm's killer was
fireability, it was pre-measurable from artifacts already on disk, and it was measured first.
**One arm cost a slot and 2h13m of GPU to learn its lesson; the next cost ten minutes.**
