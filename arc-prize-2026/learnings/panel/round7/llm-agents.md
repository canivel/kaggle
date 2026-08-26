## Summary (2 sentences)
v3 does the two things I demanded most loudly — gates the vanilla duck as the first R0 candidate before any porting arithmetic, and stress-tests the restart lever against its own corpus to the point of retracting the +0.10 EV headline — and the §E2 retraction-with-measurement is exemplary evidentiary conduct. However, the reviewed artifact *again* truncates mid-sentence (this time inside §Instruments at "excluded from the rol"), so §R0/§R1/§R2/§R3/§Windows/§Risks remain change-log-verified only, and two new load-bearing gaps have opened: the restart EV's "FP restarts cost nothing" claim ignores wall-clock displacement, and the two-track gate has no viable track for the diffuse fork-port deltas that the plan now ranks first.

## Objections

**Resolution of prior-round objections:**

**[LA-M1] Winning notebook / fork-delta audit — PARTIALLY-RESOLVED.** The R0 design (vanilla-first gate, delta audit of the 1.28–1.56 band, tokens/action kill on ports, 1.56-anchor verification) now answers everything I asked, on paper. But §R0's body is not in the reviewed artifact; I am verifying the change-log's summary of itself, which I explicitly said carries no credit. Resolution is conditional on the section being filed and matching the change-log.

**[LA-M2] R2 shortlist / exec-WM metric / decision-table timing — RESOLVED (in substance).** Object-level transition accuracy on state-changing transitions only, ≥+15 points over a copy-last-frame identity baseline, seed-held-out split, n≥200, plus the pre-transcript commit of the decision table with a recorded hash — this is exactly the specification I demanded, and the identity-baseline margin closes the trivial-predictor hole. Verification caveat as above: §R2's body is absent from the artifact.

**[LA-M3] Reallocator EV — RESOLVED** (prior round; disposition stands).

**[LA-M4] L1/L2 thrash — RESOLVED** (prior round; stands).

**[LA-M5] Generalization gate — RESOLVED** (prior round; stands).

**[LA-m1] Baseline staleness — RESOLVED** (prior round; the era-based df refinement is the statistics reviewer's lane).

**[N1] Vanilla gap — RESOLVED (design).** Vanilla-first gating, with the drift-vs-fork-regression decomposition (vanilla-now vs 1.21 measures drift; vanilla-now vs substrate-now measures fork effect on matched games/windows), is the correct experiment and costs what I said it would. Subject to §R0 existing in the filed document.

**[N2] RESTART context semantics and discount provenance — PARTIALLY-RESOLVED.** The discount is now measured (and the measurement killed the headline — credit for that), RESET is specified as fresh episode + fresh per-game analyzer context enforced in harness code and checked in transcripts, and the version confound is refuted. But my fix (ii) — one **free local** seed with the scheduler active on 3–4 flip games, reporting the empirical second-attempt good-mode rate vs the cross-seed p *before* any scored window — was replaced by "exchangeability check pre-registered on first live runs," i.e., you will learn the answer while spending scored windows. Note also that the measured disc(t) curve itself *inherits* the exchangeability assumption: it is computed from cross-seed good runs, so if restart attempts are correlated draws, both the +0.24 and the ≈0 numbers move. The free local pre-check remains mandatory before the lever consumes any ledger resource.

**[N3] Tokens/action kill on ports; 1.56 anchor — RESOLVED** per change-log; same §R0 verification caveat.

**[N4] Incomplete document — UNRESOLVED, and now escalated.** The v3 header asserts "filed complete" with checkable line-counts, yet the artifact in front of this panel truncates mid-sentence inside §Instruments ("excluded from the rol") — for the second consecutive round. Whether this is a distribution pipeline bug or a filing bug, the consequence is the same: every §R0/§R1/§R2/§R3/§Windows/§Risks disposition marked "Accepted" is verified only against the change-log's own summary, and per my prior-round statement those dispositions carry no credit. Fix: file the document through a channel where the panel can verify the stated line-counts, and treat a third truncation as a process failure attributable to the authors regardless of cause.

**New objections:**

**[MAJOR] N5: "FP restarts cost nothing" is false under a wall-clock budget, and the +0.24 best-across EV is computed without a wall-cost column.** The §E2 EV table charges FP restarts zero under best-across scoring because the game-local score can't regress — but restart actions are not free: a (90, cap 2) policy adds up to ~180 actions on every triggered game (138/250 runs trigger at 90), all of which consume the shared session wall under the 10.5-h watchdog and per-game caps, displacing actions on *other* games in the same commit. Your own corpus says 7.9% of clears arrive after action 120 — those are exactly the clears a wall squeeze deletes. Fix: the per-transcript simulation must replay the *full 25-game draw* under the actual per-game and session wall caps and report EV net of displaced-clear loss and the tokens/action inflation from fresh-context re-exploration on attempt 2; if the sim already does this, print the wall-cost column, because the table as published does not show it.

**[MAJOR] N6: The two-track gate has no viable track for the plan's own #1 leg.** Fork ports are budgeted at +0.10–0.175 *in aggregate*, but individual ports (a prompt edit, prev-frame handling) will plausibly land at +0.03–0.08 each — below Track A's +0.12, where P(promote|works) ≈ 14% and the pre-registered gate-consistency check *bars submission*. Track B requires a pre-registered event-log mechanism statistic with a per-event false-attribution rate, which is well-defined for a scheduler (discrete restart-recovered clears) but has no obvious analogue for a diffuse prompt delta — what is the countable "event" for "better system prompt"? As written, the plan's first and least-assumption-laden leg is unpromotable under its own instruments unless ports are gated as a *bundle* (which forfeits per-delta attribution and violates the one-delta-one-gate structure of the R0 audit). Fix: pre-register the porting gate explicitly — either (a) bundled Track A submission of the port stack with per-delta local ablation evidence, or (b) a defined Track B mechanism-statistic template for prompt-class deltas (e.g., stall-segment escape rate, actions-to-first-level on the affected phase) with its false-attribution estimator named before the first port is tested.

**[MAJOR] N7: The scoring-semantics fact-check — the single fact that decides a +0.13 official lever — has no specified method.** "A free, pre-registered binary fact-check of the harness/API scoring semantics" could mean reading documentation (which for this competition has been ambiguous before), reading the public winning notebook's scorecard handling, or an empirical probe; these have very different error rates, and a wrong answer either wastes the lever or ships a net-zero build. Fix: specify the observable and its failure mode — e.g., (i) inspect the scorecard API response schema for per-attempt vs max fields in the public harness code, and (ii) confirm empirically on one already-scheduled window by comparing a game's scorecard value against its known per-attempt levels in the transcript (piggybacked, costing zero extra windows). Pre-register what result maps to "best-across" and commit that an ambiguous result defaults to the last-attempt (kill) branch.

**[MINOR] N8: RESET freshness needs a machine-checkable transcript invariant, not "verified in transcript logs."** Specify the check: e.g., prompt-token count and context hash at attempt-2/action-1 must equal attempt-1/action-1 modulo frame content, with zero carried scratchpad keys — an assertion in the harness, not a human read of logs. Your ar25/su15 pollution results are precisely why a silent summary-carry bug would convert the lever into a measured harm while passing casual inspection.

## Questions for the authors (numbered)
1. What exactly is the scoring-semantics fact-check procedure, what observable decides it, and what happens on an ambiguous result? (N7)
2. Which track does each individual fork port enter, and what is the Track B mechanism event for a prompt-class delta? (N6)
3. Does the §E2 EV simulation enforce the per-game and 10.5-h session wall caps when replaying a full 25-game draw with the scheduler active, and what is the displaced-clear loss at (90, cap 2)? (N5)
4. Does the unmodified vanilla duck artifact still run unmodified against the current API (dependency/version rot since Milestone-1), and if not, what is the minimal-patch policy so "vanilla" stays vanilla?
5. What is the machine-checkable transcript invariant for RESET context freshness? (N8)
6. Will the free local scheduler pre-check on 3–4 flip games (N2 fix ii) be run before any scored window, yes or no?

## What I cannot judge
The era-based σ̂/df accounting, the rule-of-three game-level bounds, the binomial lower-bound construction for the segmentation gate, and the Track B P(≥3 events all-spurious) calculation are the statistics reviewer's lane. vLLM in-kernel serving reliability, the watchdog/checkpoint mechanics, and the RunPod cost model are the systems reviewer's lane. I cannot independently verify the projected Sep-30 top-100 cutoff (1.35–1.5) or the ~55-private-game structure of the final evaluation.

## Verdict: MAJOR-REVISION

## Score: 7/10