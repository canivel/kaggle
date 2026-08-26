## Summary (2 sentences)
The revision materially improves statistical hygiene: the invalid z-score against the frozen-fork band is gone, the undefined "+0.10–0.15 EV" claim is withdrawn in favor of a pre-registered paired offline screen, and R2 window 1 is now gated behind a war-v1 control ledger of n≥3 — so most of my round-1 objections are resolved or partially resolved. What remains unaddressed is the *power* side of the ledger gate (n≥3 is a floor, not a design), a banking-divergence "check" performed with an instrument that has never once fired, single-game leverage in the p=0.0074 screen, and the still-unverified headline-number discrepancy between the panel briefing and the document.

## Objections

**Prior-objection resolution status (required first):**

**[Prior #1 — R2 must not start before variance work] PARTIALLY-RESOLVED.** The sequencing violation is fixed: Q3 now enforces prereg §4 (war-v1 ledger n≥3 before ledger-ON window 1, earliest Jul 17), and tonight's/today's submissions are control draws — exactly the ordering I demanded. However, the second half of the fix — recompute MDE/power and window count under the reconciled variance, amending thresholds if needed — is nowhere in this brief. n≥3 is a necessary gate, not a powered design (see new objection below).

**[Prior #2 — no valid warpack control band; wrong-null z] RESOLVED.** No standardized effect is quoted anywhere in the brief; the team is accumulating a dedicated war-v1 ledger (currently n=1) before any inference, which is precisely the remedy requested. One residual bad habit is flagged as a new MINOR ("Consistent with LB draw #1 = 0.91" is an n=1 inference).

**[Prior #3 — order-statistics arithmetic refuting the wall claim] PARTIALLY-RESOLVED.** The quantitatively false "wall mechanism is adopted" justification has been withdrawn — §1c now reports the wall descriptively (12 teams ≥ 1.44, gap 0.42) with no order-statistics claim. But the requested corrected artifact (expected-max-vs-k curve in path_forward_v5, making explicit that only per-draw mean improvements can close the 0.42 gap) is not attached or referenced; withdrawal-by-silence is not the same as publishing the correct arithmetic the strategy rests on.

**[Prior #4 — "+0.10–0.15 EV" has no estimand] RESOLVED.** The claim is gone. Warpack's effect is now decomposed correctly: a defined offline estimand (paired Δlc per game, exact sign-flip test, pre-registered as primary in §2) plus LB per-draw evidence deferred to the accumulating ledger. This is the split I asked for.

**[Prior #5 — briefing/document headline discrepancy] UNRESOLVED.** The panel briefing still says best 0.43 / leader 1.56; the document says best 1.02 / leader 1.86. No submissions CSV or screenshot was attached. Every gap number in §1c depends on which is true; this was a one-line fix and it was not done.

**New objections:**

**[MAJOR] The n≥3 ledger gate is a floor, not a powered design — publish the MDE before Jul 17 or window 1 is theater.** With the frozen-fork σ̂ ≈ 0.074 (itself estimated on n=5, 95% CI on σ roughly [0.044, 0.21]), a 3-vs-3 alternate-nightly comparison has SE(Δ) ≈ 0.074·√(2/3) ≈ 0.060, giving an 80%-power MDE of ≈ 2.8·SE ≈ 0.17 — larger than any plausible ledger effect and 40% of the entire gap to the wall. Worse, warpack's own per-draw σ is unknown (n=1) and banking is expected to compress the left tail, so even the SE is a guess. Actionable fix: before window 1, pre-register (i) the assumed σ with its provenance, (ii) the MDE at the planned n per arm, (iii) the number of windows required to reach a stated power for a stated effect, and (iv) the stopping/decision rule — otherwise the A/B will terminate in an uninterpretable "no significant difference" that gets misread as "ledger does nothing."

**[MAJOR] "No observed banking divergence" was certified by an instrument that has never fired — the handoff condition in §1b is vacuously satisfied, not satisfied.** Zero replay events across the entire war-eval run (the brief concedes banking has never executed its core mechanism outside local smoke) means the divergence check has zero sensitivity: an instrument that cannot trigger cannot detect. Yet this vacuous pass is used to queue draws #2–#3 and to assert "banking identical in both arms" for the Jul 17 A/B — if the LB environment does trigger replays, a latent banking bug contaminates the *control* ledger itself, and no offline evidence exists either way. Actionable fix: Q4's canary build is necessary but insufficient — before Jul 17, run one engineered validation (multi-pass or seeded-win configuration guaranteed to produce ≥1 replay) and confirm replay_attempted/succeeded counts and score invariance; a canary that has only been smoked on runs with nothing to replay proves nothing.

**[MINOR] The p = 0.0074 screen is leveraged by a few large paired differences and is single-seed; report leave-one-out sensitivity and frame the mechanism story as hypothesis.** A plain sign test on 12W/5L (17 non-ties) gives p ≈ 0.14 two-sided; the 0.0074 comes from the magnitude-weighted sign-flip permutation, where sc25 (+1.8) alone contributes ~26% of the total Δlc = 6.8. Report the statistic with sc25 (and m0r0) removed, and label the "recovery buys stuck-game L1s at full action cost" narrative as a one-seed hypothesis pending the 3-seed gate — the author's Q2(a) choice (more seeds) is the right call and I endorse it, but the brief's causal prose outruns its n.

**[MINOR] "Consistent with LB draw #1 = 0.91 (no lift in official currency)" is an n=1 inference.** 0.91 sits comfortably inside the frozen-fork band (0.82–1.02); a single draw is consistent with lift, no lift, or regression. Strike the phrase or replace with "uninformative at n=1."

## Questions for the authors (numbered)
1. What σ, n per arm, MDE, and number of alternate-nightly windows will the R2 A/B use, and where is this pre-registered before window 1 fires on Jul 17? (P1–P5-equivalent thresholds under the current variance model, please.)
2. Can you produce, before Jul 17, one run in which banking demonstrably replays at least once (canary counts > 0) with verified score invariance? If not, what is the fallback for the "banking identical in both arms" assumption?
3. What is the paired Δlc statistic and sign-flip p with sc25 excluded (leave-one-out)?
4. Which headline numbers are correct — 0.43/1.56 (briefing) or 1.02/1.86 (document)? Attach the exported submissions CSV.
5. Does the LB evaluation environment run multi-pass (i.e., can replays occur on the leaderboard at all)? If banking is provably inert on LB, say so — it simplifies both the control ledger and the A/B.

## What I cannot judge
The daemon UTC-window bug diagnosis and fix (Q1/Q5 engineering), whether the Kaggle quota semantics are as described, the internal correctness of the warpack code (recovery/retry_guard/shortcircuit/banking implementations), the fidelity of the offline lc/RHAE scorer to official LB scoring, and whether the ≤2-push/zero-spend constraints are optimally allocated — these belong to the systems/infrastructure reviewers. My review covers only the experimental design, inference, and power claims.

## Verdict: MAJOR-REVISION

## Score: 6/10