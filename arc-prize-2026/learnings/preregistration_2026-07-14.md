# Pre-registrations & instrument amendments — 2026-07-14

Written in response to panel round 10 (3× MAJOR-REVISION, 0 fatal; unanimous directives).
Evidence base: `runs/variance_reconcile/report.md` + `raw.json` (computed today, validated
RHAE scorer, max err 0e+00 over 1000 cross-checks vs Tufa's 500 runs).

## 1. Variance reconciliation — RESULT (closes R9 ME-NEW-11 / P / systems-Q3)

The 0.52-vs-0.074 "incoherence" is resolved: the two numbers describe **different
instruments**, and the build rail is even noisier than the bootstrap claimed.

| instrument | statistic | value |
|---|---|---|
| build/pod rail (25 public games, null10, direct) | 1-seed run-mean sd | **0.572** |
| build/pod rail (45 pairwise seed deltas, direct) | 1-seed paired-Δ sd | **0.780** |
| Kaggle LB (5 frozen-fork draws) | per-draw sd | **0.074** (χ² 95% CI on σ: [0.044, 0.213]) |
| ratio | build-rail / LB paired-Δ sd | **7.5×** |

Mechanism: build-rail variance is a single-game tail phenomenon — **ft09 carries 54.3%**
of run-mean variance (null mean 10.2, across-seed var 83.1), ft09+vc33 carry **71.5%**.
The LB hidden set/aggregate does not exhibit this tail (σ 0.074 over 5 draws). The
bootstrap's 0.52 was directionally right for the build rail and irrelevant to the LB.
Neither number was wrong; they were cross-applied.

## 2. H4 AMENDMENT — build-rail gate statistic (supersedes "sd 0.52 ≈ 7× σ̂" clause)

RHAE-mean is **retired as a build-rail gate statistic**: a 3-seed gate at α = 0.0125 has
power **0.02** vs +0.10 (0.04 vs +0.20). Measured alternatives (same data, same seeds):

| statistic | sd_1seed | se_3seed | power +0.10 | power +0.20 |
|---|---|---|---|---|
| RHAE mean (old) | 0.572 | 0.467 | 0.02 | 0.03 |
| RHAE mean, ft09+vc33 excluded | 0.328 | 0.268 | 0.03 | 0.07 |
| log1p(RHAE) mean | 0.116 | 0.095 | 0.12 | 0.45 |
| **levels_completed mean (NEW PRIMARY)** | **0.086** | **0.070** | 0.21 | **0.73** |

- **Primary build-rail gate statistic: paired per-game Δ levels_completed** (exact
  sign-flip test unchanged). Units change: targets are stated in Δlc, not ΔRHAE.
- **Secondary: paired Δ log1p(RHAE)** (retains efficiency signal that lc misses).
- RHAE mean is still *reported* (it is the official currency) but never gates.
- LB-rail inference: t with df = n−1 (never z), per methodology R10.

## 3. Warpack (war-v1) control ledger — PRE-REGISTERED

- war-v1 is its own arm; frozen-fork control {0.82, 0.89, 0.93, 1.02, 0.95} is NOT its
  null. The z ≈ −0.16 quoted in daily_brief_2026-07-14 §1a is retracted as a standardized
  claim (draw is *descriptively* inside the frozen band; nothing more).
- Ledger: draw #1 = **0.91** (2026-07-14 00:12Z). Draw #2 queued tonight (byte-identical
  kernel v1, queue head set 2026-07-14).
- **n ≥ 3 war-v1 draws before any war-v2 (ledger-ON) contrast is interpreted; n ≥ 5
  before any standardized war-v1-vs-frozen-fork effect is quoted (t, df = n−1).**
- Split EV claim (methodology R10 obj. 4):
  (i) *mechanism EV* (order stats over k draws) is arithmetic, not LB-inferable — see §6;
  (ii) *per-draw distributional effect of banking* — two-sample t vs frozen control,
  MDE at n=5/5 with sd 0.074: ≈ 0.14 at 80% power, α=0.05. Anything smaller is
  undetectable this way and will not be claimed from LB draws.

## 4. R2 (ledger + escalation) A/B — LAUNCH CONDITIONS (all must hold)

1. Variance reconciliation published — **DONE today** (§1).
2. war-v1 ledger n ≥ 3 (ETA: draw #3 lands 2026-07-16 morning).
3. P1–P5 inline with observables + thresholds — **DONE below** (§5).
4. Design lock: **war-v2 = war-v1 + flags {ledger, escalation} ONLY; banking, recovery,
   retry_guard, shortcircuit, soft_end identical in both arms** (answers rl-planning
   R10 obj. 4: yes, banking is held constant; the A/B isolates the ledger stack).
5. Endpoints: primary = P1–P5 on pulled build-rail transcripts (observable; scored-rerun
   transcripts do not exist); scored-LB windows are logged to the war-v2 ledger and
   interpreted only under §3's n-minimums with a stopping rule: alternate arms nightly,
   continue until the war-v2 − war-v1 CI half-width < 0.10 or 6 windows/arm, whichever
   first. No mid-course threshold changes.

## 5. P1–P5 — VERBATIM (from `learnings/war_room/intervention_plan.md` §R2 step 3), with observables

> P1: sb26 leaves the fill-in-order family before action 80 and states ≥3 distinct goal
> families. P2: su15 states a third goal family within 30 actions of refuting the second.
> P3: verbatim-paragraph recurrence drops >70%. P4: SPACE/no-op re-probes ≤2 per run.
> P5: sb26 post-restart does NOT re-execute a refuted plan. ≥4/5 = concept validated
> even if L2 doesn't fall.

Observables: pulled build-rail transcripts (`runs/kernel_pulls/<kernel>/transcripts/`),
GOAL:/RESULT: fields regex-extracted; P3 measured as SequenceMatcher ≥ 0.9 paragraph-pair
rate vs the 13-run historical baseline; P4 counted from the FACT-ledger no-op table.
Effect-size target for the scored windows (stated in the NEW gate currency): **Δlc ≥ +0.08**
(2 grinder L2s / 25 games); build-rail 3-seed power vs +0.08 lc ≈ 0.15 → the A/B leans on
P1–P5 (mechanism) + accumulating LB windows (outcome), not on a single 3-seed gate.

## 6. Order-statistics ceiling — published (closes the Q4 arithmetic dispute)

E[max of k LB draws], control mean 0.922: k=30 → **1.07** / k=60 → 1.09 / k=110 → **1.11**
(σ = 0.074). Under the χ² CI-upper σ = 0.213: k=110 → **1.46**.

Honest narrative replacing brief §Q4: order stats from OUR current distribution cannot
reach the 1.44 wall unless our true per-draw σ is ≥ ~0.2 (3× the point estimate, but
inside the n=5 CI — not excludable). The wall teams either have a higher per-draw mean
(Cottaar's own fork drew 1.21 on its first submission) or fatter draw tails. Either way:
**only per-draw mean improvements (R2–R5 grinder cracking) are budgeted to close the
0.42 gap; order stats are a floor-raiser worth ≈ +0.15 total by Nov 2.**

## 7. ME-NEW-12 / Q disposition — scored-run identity & wheel reconstruction

- Identity: **the scored rerun is a separate hidden execution.** Fast-submit build log
  proves the pulled kernel output is a 0.6 s dummy write (`RUN_HEAVY=False`). No
  per-game artifacts of any scored run exist or will exist.
- Aggregate reconstruction as an *identity* check is not feasible: the hidden game set ≠
  the public 25 (build-rail null mean 1.64 vs LB control mean 0.922 — the rails differ in
  population, not only in noise). What is pre-registered instead: **cross-rail
  monitoring** — each build's rail lc-mean is logged next to its LB draws; a build whose
  rail lc improves ≥ +0.20 while its LB ledger stays flat over ≥3 windows triggers a
  "rail divergence" review. No pass/fail beyond that; this is a tripwire, not a test.
- Banking canary (llm-agents R10 obj. 3): adopted as a build task — fold
  `replay_attempted/replay_succeeded` counts into a side-channel observable (e.g., an
  actions-count residue) so replay integrity in the hidden rerun is binarily checkable.
  Scheduled for the next war kernel version (NOT tonight; tonight is byte-identical v1).

## 8. Sched-v1 adjudication — KILLED by its own pre-registered gate

Pre-registration (2026-07-12): promote ≥ baseline+0.12; kill < baseline. Draw #1 = 0.90 <
baseline 0.922 → **kill**. Offline concurs (Δ −0.32, recovered L1s RHAE-crushed under
pooled-single-run semantics). Draw #2 removed from queue today. Ledgered per H3-style
discipline; the surviving idea (P(recover) × clean-L2+ value) lives on inside warpack's
`recovery` module, which is subject to §3's accounting.
