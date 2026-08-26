## Summary (2 sentences)
The revision does real work: the counterfactual replay I demanded was run, the indefensible L1 EV was retracted rather than defended, the scheduler semantics are now fully specified, and the R2 gate was restructured to game-level units — this is what a responsive revision looks like. However, the replacement lever's EV rests on an unvalidated free parameter (the 0.4 depth discount) and an unexamined independence assumption, and the plan's own arithmetic now puts its primary free lever (+0.055 official) *below* its own promotion threshold (+0.12), an incoherence that must be fixed before R1 ships.

## Objections

**[RESOLVED] RL-F1 (was FATAL): L1 EV contradicted the STUCK finding —** The replay was run on the corpus I named, the numbers are specific (12/152 clears after action 120; truncation saves 7.9% of tokens, not 20.2%), and the +0.10–0.30 EV is explicitly retracted. The pivot to restarts is legitimately grounded in the flip-game data (16/25 games flip across seeds), which correctly narrows my "deterministic per game" clause to the dead 5. The contradiction is gone. The *replacement* derivation has its own problem — see new objection [A].

**[RESOLVED] RL-M1: L1/L2 interaction incoherent —** Counter semantics are now explicit (per-attempt counter resets, cumulative attempt counter never resets), cap 2 restarts, park dominates restart, and dead-game behavior is bounded at ≤270 actions and simulated on the null10 transcripts as I asked. No thrash loop is possible under these semantics as written.

**[PARTIALLY-RESOLVED] RL-M2: R2 gate independence —** The structural fix is exactly right (≥2 distinct games, ≥2/3 seeds each, plus the r11l holdout). But the claimed p<0.01 does not follow from the cited data: with only 10 null seeds per game, the rule-of-three per-game upper bound is q ≈ 3/10 = 0.3, not 0.1 — the "0/30" pooling re-commits the clustering error the fix was meant to remove. Worst-case P(≥2 of 3 games flip) ≤ 3q² ≈ 0.2, not 0.01; claiming p<0.01 requires q ≤ 0.058, which 10 seeds cannot establish. State the honest bound and rely explicitly on the r11l holdout + mechanism prediction for the remaining selectivity; this is a one-line fix (MINOR residual).

**[PARTIALLY-RESOLVED] RL-M3: BFS under-specified —** "Stall-scoped *executed* exploration over the existing segmentation graph with a hard action/token budget" plus the exec-WM ≥70% next-state fidelity gate is the right shape, and the one-page designs are scheduled for the Aug 3 deliverable as I demanded. But the specification does not yet exist — in particular the predicted tokens/action delta against the >10% kill criterion is still unwritten, and "executed exploration" is precisely the RHAE-penalty-paying variant. Resolution is contingent on the Aug 3 artifact; I will re-review it there. No new action needed now beyond delivering what is promised.

**[RESOLVED] RL-M4: window gate uninterpretable at σ̂ upper endpoint —** Error rates are computed and printed at both endpoints, the sign-flip rule is pre-registered with a defined provisional-promote path, and df grows via the control class (df≈8 at first candidate gate — close enough to my df≥9 demand given the mechanism). The honest admission of 24.5% false-promote at the upper endpoint plus mandatory re-confirmation is an acceptable decision rule.

**[PARTIALLY-RESOLVED] RL-M5: Nov-2 compression thesis unevidenced —** Downgrading to a pre-registered hypothesis with the fork-delta audit as its test is the correct move, and the LB forensics (82% of the ≥1.0 cohort landing within 2 days of the June-30 open-source) is genuine supporting evidence I did not have last round. However, the "pre-registered failure consequence" lives in §Risks, which is not present in the document as provided to me, so I cannot verify it exists or has teeth. Unresolvable until I see §Risks; the audit itself is not yet run.

**[RESOLVED] RL-m1: exploration closed as a family —** The closure is narrowed to always-on context injection, stall-scoped exploration stays on the shortlist, and the state-coverage metric (distinct segmentation-graph states per 100 actions, null vs stuck) is added to R2 forensics. Exactly the requested fix.

---

**[MAJOR — NEW] (A) The restart EV's depth discount 0.4 is a free parameter, and the within-run independence assumption is untested —** The +0.10 ± 0.05 is not a counterfactual replay result; it is a model with an assumed discount and an assumed equivalence between a fresh *seed* (the flip evidence) and a mid-run *restart*. Both are measurable from the corpus you already have: (i) compute the discount empirically as the budget-truncated value curve of good-mode runs (value achieved in the first T−90 actions as a fraction of full-run value), replacing 0.4 with a measured number; (ii) specify whether RESET clears the analyzer's LLM context — your own replicated context-pollution finding says a restart with retained context may sample the good mode at p′ ≪ p, which would gut the EV. If (i) yields a discount near 0.2 rather than 0.4, the net EV drops to ≈ +0.03 and R1 is not worth its windows. Both analyses are free and must ship with the build.

**[MAJOR — NEW] (B) The plan's primary free lever cannot pass the plan's own gate —** The attempt scheduler's expected effect is +0.10 local → **+0.055 official**, but the window gate promotes at Δ ≥ +0.12 official. At σ̂ = 0.074 (SE ≈ 0.060 for the 2-vs-rolling-6 design), a scheduler working *exactly as derived* promotes with probability ≈ 14% and is killed outright (Δ̂ < 0) with probability ≈ 18%. The instrument is thus expected to reject the one lever the replay data supports — this is the same threshold-vs-effect-size incoherence I flagged last round, now strictly worse. Fix before R1: pre-register either (a) a component-specific promote threshold derived from the component's EV and a paired per-game statistic (you already built this machinery for ME-m2 — use the ~20 version-matched games as exchangeable units, which has far more power than the aggregate Δ), or (b) an explicit bundling policy (scheduler + fork deltas gated together) with a pre-registered attribution rule.

**[MINOR — NEW] (C) The 90-action trigger is unswept when the sweep is free —** You have per-action event logs for 250 runs; sweeping trigger ∈ {90, 120, 150} × cap ∈ {1, 2} on the replay corpus and publishing the FP/EV frontier costs nothing and either confirms 90 or improves it. Also: the simulated behavior published with the build must cover the 16 flip games' transcripts (where the FP cost actually lives), not only the 5 dead games.

## Questions for the authors (numbered)
1. Does RESET clear the analyzer's context window, or does the failed attempt's context persist into the fresh episode? Given your ar25/su15 pollution results, what does retained context predict for p′ on the restarted attempt?
2. What is the provenance of the 0.4 depth discount? Provide the budget-truncated value curve from null10 good-mode runs.
3. How do you reconcile gating the attempt scheduler (+0.055 official expected) against a +0.12 promote threshold — what is the pre-registered path by which a correctly-working scheduler gets promoted?
4. For the R2 two-game gate: how do you justify a per-game null rate below 0.058 from 10 null seeds per game, as required for the claimed p<0.01?
5. What is the exact pre-registered failure consequence for the compression hypothesis in §Risks (not present in the reviewed document), and what audit result triggers it?
6. Will the Aug 3 one-pager for stall-scoped BFS include a *measured* dry-run tokens/action estimate on logged stall segments (not just a prediction) against the >10% kill criterion?

## What I cannot judge
- Dollar and Kaggle-quota accounting ($14–28 provenance, 30 h/wk ledger, commit-hour arithmetic) — systems reviewer's domain.
- Accuracy of the leaderboard forensics and fork-band claims (1.28–1.56 band composition, the 82%-within-2-days statistic) — I take these as asserted.
- Calibration of the 0.55 local→official difficulty ratio, which scales every EV in the target arithmetic — metrics reviewer's domain.
- Sections §R2 (tail), §R3, §Windows, and §Risks were truncated in the copy provided; several change-log dispositions pointing there are unverifiable by me.

## Verdict: MAJOR-REVISION

## Score: 7/10