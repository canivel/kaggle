## Summary (2 sentences)

The brief's centerpiece — the code-forensic resolution of the scoring semantics (pooled-single-run with first-clear tax, cited to `api.py:417`, `scorecard.py:192/430/655/673`, `game_api.py:221`) — is exactly the method I demanded in N7 and it correctly kills the authors' own headline lever, which earns real credit. However, a scored submission window was spent at 02:19Z on that same lever under a plan the panel has *not approved*, the plan's least-assumption-laden leg (R0 fork audit) remains unstarted for another round, N6 is still open, and the Q2 redesign quietly re-imports the exchangeability assumption plus an unstated stall-detector FP rate — so this is progress under broken sequencing, not an approvable state.

## Objections

**Resolution of prior-round objections:**

**[LA-M1] Winning notebook / fork-delta audit — PARTIALLY-RESOLVED (carried, now with a prioritization complaint).** §R0's design was accepted conditionally on the filed body; v4 is not yet filed, and per Q4 the leg itself — the *only* leg with no unverified assumptions — is still unstarted while two panel rounds and one scored window were consumed by a scheduler the code read has now killed. Answer to Q4: yes, start today; this leg should have started before sched draw #1 was ever queued.

**[LA-M2] R2 shortlist / exec-WM metric — PARTIALLY-RESOLVED.** Substance stands from R7; artifact verification still pending the checksummed v4 filing. No regression, no progress.

**[LA-M3] / [LA-M4] / [LA-M5] / [LA-m1] — RESOLVED** (prior rounds; dispositions stand).

**[N1] Vanilla gap — RESOLVED (design), execution unstarted.** Same as LA-M1: the experiment is correct on paper and has now sat idle for a round in which resources went to a dead lever.

**[N2] RESTART semantics / exchangeability pre-check — PARTIALLY-RESOLVED, with a sequencing violation.** Credit: the Kaggle build-rail run (1b) supplies the behavioral half of my fix (ii) — 18 restarts, 4 recovered L1 clears, mechanism fires as designed. Debit: my mandate was that this evidence land *before* any scored window, and sched draw #1 was submitted at 02:19Z regardless. Further, the exchangeability assumption is still untested and Q2 proposes to *reuse* it (cross-seed fresh-start clears as proxy for attempt-2 outcomes) — when the 1b data now contains 18 *actual* second attempts. Fix: compute the empirical attempt-2 good-mode/clear rate from the 1b restarts and compare to the cross-seed p before the null10 replay's output is trusted; n is small but it is the direct measurement.

**[N3] Tokens/action kill; 1.56 anchor — RESOLVED** per change-log; §R0/v4 verification caveat carries.

**[N4] Incomplete document — UNRESOLVED.** Q3 promises a checksummed v4; a promise is not a filing. This brief is itself complete, which shows the distribution channel *can* work — so the third-truncation clock on the binding plan document still runs, and the "Accepted" dispositions in v3 still carry no credit until v4 arrives verifiable.

**[N5] Wall-cost column — OVERTAKEN / PARTIALLY-RESOLVED.** The best-across EV table is now moot (authors' own 1c kills both columns), and 1c's survival formula does name "− wall cost." But naming the term is not computing it: the requirement carries verbatim into Q2 — the null10 replay must simulate the *full 25-game draw* under per-game and session wall caps, including tokens/action inflation from fresh-context attempt-2 exploration, and print the wall-cost column.

**[N6] Porting-gate track for prompt-class deltas — UNRESOLVED.** Q3 acknowledges it and defers to v4. Acceptable only if v4 files with the explicit choice — bundled Track A with per-delta local ablations, or a Track B mechanism-statistic template with named false-attribution estimator — before the first port is tested. Since Q4 proposes starting the fork audit *today*, this can no longer wait a round.

**[N7] Scoring-semantics fact-check method — RESOLVED.** Reading the shipped wheel and taaf source, with line-level citations, discovering a *third* semantics neither panel branch predicted, and finding empirical confirmation in tu93 is precisely the "from code, not documentation" standard I set. One residual caveat, filed as N11 below, because the wheel is client-side.

**New objections:**

**[MAJOR] N8: A scored window was spent on an unapproved plan's lever, and a second is queued — pull it.** The header states v3 is NOT approved (3× MAJOR-REVISION) and `winning_solution_FINAL.md` is the fallback authority; nothing in the fallback authorizes sched draws. Draw #1 went out manually at 02:19Z "post-panel," and draw #2 sits in tonight's queue for a lever that 1c kills *as implemented* — v3's own pre-registered language for a killed-by-fact-check lever is "zero windows spent," and the promote threshold (draw2 ≥ 1.184, above every control draw ever observed) makes the draw's information value ≈ zero. Answer to Q1: replace with frozen-duck σ-draw #6; honoring a pre-registration whose underlying hypothesis has been refuted by code is cargo-cult pre-registration, and the fact-check kill path *is* the pre-registered exit. Fix: log the 02:19Z authorization chain, and add a hard rule to v4 that no scored window fires on a lever without an approved plan section covering it.

**[MAJOR] N9: The local↔Kaggle rail sign disagreement (1d) is an instrument-validity problem, not just a phase-1 footnote.** Local 3-seed says −0.54; the Kaggle rail says +0.19 mean / 8W-11L on the banner-verified identical config. On the phase-1 adjudication I concur with the author — closed; 1 seed cannot overturn a 3-seed pre-registered FAIL and the sign pattern is the pre-registered variance-domination mode. But every gate in this plan uses the local rail as its screening instrument, and this is the first paired same-config observation across rails — and it disagrees in sign. Fix: before the next local gate is treated as binding, quantify cross-rail agreement on existing paired configs (frozen duck exists on both rails); if paired deltas are uncorrelated or sign-flipped beyond seed noise, local gates require one Kaggle-rail confirmation seed before any KILL/PROMOTE is final.

**[MAJOR] N10: The Q2 park-only redesign has an unstated load-bearing parameter — the stall detector's false-positive rate at 90 actions.** "Parks cost nothing under pooled scoring" is only true for games that were never going to clear; my prior-round corpus stat says 7.9% of clears arrive after action 120, and a park-at-90 policy deletes every clear that would have arrived post-park on a false-positive stall call. The v1 evidence itself is warning you: 4 parks at exactly 272 actions in one 20-game run is a high trigger rate for a detector whose precision has never been measured. Fix: the null10 replay must report, for runs flagged "dead at 90" by the actual detector logic, the fraction that subsequently cleared (at any action count), and the park-only EV must be net of those deleted clears plus the N5 wall simulation on the benefit side.

**[MINOR] N11: Wheel-vs-server identity is assumed, and it is checkable for free.** The 1c forensics reads `arc_agi-0.9.6` as shipped in the kernel; if the official LB computes scores server-side, the wheel is a proxy that could drift or diverge. Fix: cross-check the scorecard actually returned for draw #1 (per-game, per-level action buckets — especially bp35/ls20/tu93/ft09) against the wheel formula's predictions; agreement on the four recovered games confirms 1c on live data at zero cost.

**[MINOR] N12: The overnight session performed an unlogged manual submission and skipped ITERATION_LOG.** For an autonomous loop this is the canonical audit-trail failure: the one action taken outside the log is the one action taken outside the approved plan (N8). Backfilling is fine; the fix is mechanical — make ARCDailySubmit refuse to fire without a same-day log entry naming the authorizing plan section.

## Questions for the authors (numbered)

1. (Q1) Confirm draw #2 is pulled tonight and replaced with control σ-draw #6, per v3's "zero windows spent" language for fact-check-killed levers — or state which authority overrides it.
2. Who or what authorized the 02:19Z manual submission of sched draw #1, given v3 is unapproved and the fallback plan does not include the scheduler? Where is that decision recorded?
3. Is LB scoring computed server-side, and if so from what version? Report the draw-#1 scorecard cross-check (N11) for the four recovered games.
4. From the 1b build-rail run: of the 18 restarts, what was the attempt-2 good-mode/clear rate, and how does it compare to the cross-seed p the null10 replay will assume (exchangeability, N2 fix (ii))?
5. For the park-only variant: what is the exact detector predicate at 90 actions, and what is its measured FP rate on null10 (fraction of "dead-at-90" runs that later clear)?
6. When does v4 file with the checksum, and does it include the N6 porting-gate decision *before* the Q4 fork audit tests its first port?

## What I cannot judge

The statistical machinery (σ̂/df bookkeeping, era-based baselines, the 1.042/0.922 gate thresholds' Type-I/II calibration) — statistics reviewer's lane. The RHAE formula's exact arithmetic and the offline-score validity of null10 as a baseline population — methodology/scoring reviewer. Leaderboard trajectory projections (1.35–1.5 top-100 by Sep-30). I can and do judge harness semantics, code forensics, rail fidelity, agent-loop governance, and the scheduler/porting mechanism designs.

## Verdict: MAJOR-REVISION

## Score: 6/10