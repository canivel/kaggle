## Summary (2 sentences)
The revision is a genuinely responsive document: the null10 counterfactual replay was actually run and the L1 EV retracted, the R2 gate is now game-level with consistent local/confirmation arithmetic, and the r11l holdout I demanded is adopted essentially verbatim. However, the gate/EV incoherence I flagged in R2 last round has silently migrated to R1 (the attempt scheduler's own claimed EV of +0.055 official sits below the +0.12 promote threshold, giving it roughly a 1-in-7 chance of promotion *if it works exactly as claimed*), and the pre-registered component fidelity gates are ill-posed as verification instruments — no null-model baselines, no transition-level object-persistence metric, no defined held-out split — so they can be passed by trivial predictors.

## Objections

**Prior-round resolution audit (required before new comments):**

**[PS-M1] Component gates before GPU spend — PARTIALLY-RESOLVED.** Numbers now exist (segmentation ≥90% on 20 hand-labeled frames; exec-WM ≥70% next-state accuracy on held-out transitions) and the "gates before GPU" ordering is adopted. But the metrics as stated do not verify the load-bearing properties — see new objection N2 below. Static-frame segmentation fidelity is not object-identity/transition consistency, which is what BFS over a graph requires, and 70% next-state accuracy is meaningless without a baseline.

**[PS-M2] R2 primary gate statistically incoherent — RESOLVED.** The gate is now game-level (≥2 distinct games, ≥2/3 seeds each), robust to perfect seed clustering, and the arithmetic is now internally consistent: a 2-game crack ≈ +0.19 official > +0.12 promote threshold. The 0/30 null and rule-of-three reasoning is acceptable.

**[PS-M3] L1 EV contradicts STUCK finding — RESOLVED.** The exact counterfactual replay I specified was run, the +0.10–0.30 EV was retracted in print, and the lever was redesigned around the flip-game bimodality the replay actually supports. This is what a resolution should look like.

**[PS-M4] Forensics overfit loop, no out-of-sample test — RESOLVED (provisionally).** r11l is adopted as a pre-registered directional holdout with a published falsifiable prediction before GPU spend and blocking authority over confirmation — my fix verbatim. Provisional because the full §R2 text is truncated from the document I received; I verified only the change-log entry (see What I cannot judge).

**[PS-M5] Frozen null10 in a drifting environment — RESOLVED.** Version-pinning via version-suffix hashes, per-game refresh triggers, version-matched-only paired scoring, and a >5-game invalidation rule are all adequate. One residual: how bumps are "observed" between weekly sentinels is unspecified (Question 5).

**[PS-M6] Window-gate instability across σ̂ CI — PARTIALLY-RESOLVED.** Error rates are now computed and printed (good), and the sign-flip rule with provisional-promote is a coherent alternative to my "hold at upper endpoint" fix. But it has a hole: the instruments section states "after a promotion the promoted build becomes the control class." If a *provisional* promote enters the control class before stack-gate re-confirmation, a false promote (24.5% at the σ̂ upper endpoint) contaminates the rolling control for every subsequent gate — see N3.

**[PS-m1] Nov-2 compression thesis unfalsifiable — RESOLVED.** The fork-delta audit is pre-registered as an R0 deliverable with a stated failure consequence, exactly the cheap proxy I asked for.

**New objections:**

**[MAJOR] N1: The R1 gate/EV incoherence — the defect I flagged in R2 last round has migrated to R1.** The attempt scheduler's own derived EV is +0.10 local → +0.055 official (their 0.55 ratio), which is *below* the +0.12 promote threshold. With SE = σ̂·√(1/2+1/6) ≈ 0.060 at the point estimate, P(Δ̂ ≥ 0.12 | true Δ = 0.055) ≈ 14%: the plan's second strategic leg, working exactly as designed, is killed or left in limbo by its own gate ~86% of the time. Fix: pre-register a scheduler-specific decision rule consistent with its EV — e.g., primary evidence from a free local paired statistic (flip-game clear-rate across version-matched games on the pinned harness), with the 2-window test demoted to a regression guard (kill only at Δ < 0); or bundle scheduler + top fork delta into one stack candidate whose combined expected delta exceeds +0.12. The same check should be run on every fork-delta candidate whose target range starts at +0.1.

**[MAJOR] N2: The component fidelity gates are pass-able by trivial predictors and measure the wrong invariants.** (a) The exec-WM gate ("≥70% next-state accuracy on held-out logged transitions") has no null baseline: stuck transcripts are, by the plan's own STUCK finding, dominated by state-preserving actions, so an identity/copy-last-frame predictor plausibly exceeds 70% while predicting nothing. (b) "Held-out" is undefined — if held out within-game at random, memorization passes; it must be held-out *seeds*, plus a transfer report on r11l transcripts. (c) Static segmentation fidelity does not verify object-identity persistence across transitions, which is the property BFS-over-graph actually consumes; at n=20 frames the 90% point estimate carries a 95% binomial CI of roughly [68%, 99%]. Fix: redefine the exec-WM gate as accuracy *on state-changing transitions only, with a required margin over the identity baseline*; add an object-track consistency metric over consecutive frames; hold out by seed; label ≥50 frames.

**[MAJOR] N3: Provisional promotes must not enter the control class.** Under the sign-flip rule a provisional promote is decided at the σ̂ point estimate with up to 24.5% false-promote probability at the CI upper endpoint; if such a build becomes the rolling control before stack-gate re-confirmation, every subsequent candidate is gated against a possibly-null "improvement," and the ratchet compounds across the ~6 candidate families (their own multiplicity estimate: up to 1.5 expected false promotions at the upper endpoint). Fix: pre-register that only *confirmed* (non-provisional) promotions update the control class, and that a provisional build's redraws are logged in a separate class excluded from the rolling 6-draw mean.

**[MINOR] N4: The restart EV carries two unsourced free parameters.** The depth discount 0.4 and the exchangeability assumption (a within-run restart draws from the same good-mode rate p as an independent seed) have no provenance; the +0.10 ± 0.05 figure is sensitive to both. Fix: publish a sensitivity table over discount ∈ {0.2, 0.4, 0.6}, and pre-register the first live-run restart outcomes as a check on exchangeability (restart-attempt clear rate vs cross-seed p on the same games).

## Questions for the authors
1. What is the identity-predictor (copy-last-frame) accuracy on the same held-out transitions the exec-WM gate uses, and will you commit to a required margin over it?
2. How exactly is the exec-WM held-out split constructed — by seed, by game, or random within-game?
3. Does a *provisional* promote become the control class before its stack-gate re-confirmation? If yes, why is that not a ratchet contaminating all downstream gates?
4. What is P(promote) for the attempt scheduler under its own +0.055 official EV, and given the answer, why is the 2-window gate the right instrument for it as designed?
5. How are game version bumps "observed" between weekly sentinels — is there any detection channel faster than the sentinel, and what happens to a gate decision taken in the blind interval?
6. Where does the 0.4 depth discount in the restart EV come from — measured, or assumed?
7. Is the entire 1.28–1.56 band actually obliged to open-source (Milestone-eligible)? What is the audit's fallback if the 1.56 leader's deltas are not public?
8. Please supply the full §R2, §R3, §Windows, and §Risks text: several of my prior objections are marked resolved only in the change-log, and pre-registration lives in section text, not in a disposition table.

## What I cannot judge
The document truncates mid-§R1; I verified PS-M2/M4/M6 dispositions only via the change-log, not the governing section text. Outside my expertise: Kaggle quota/ledger accounting and GPU pricing provenance (systems reviewer's domain); leaderboard sociology, cutoff projections (top-100 at 1.35–1.5), and the difficulty ratio 0.55 (leaderboard-analysis reviewer); the internal optimality of the restart/park scheduling policy beyond its EV arithmetic (RL-planning reviewer). I also cannot independently verify the null10 replay numbers or that the substrate is truly the Milestone-1 winner's fork.

## Verdict: MAJOR-REVISION

## Score: 6/10