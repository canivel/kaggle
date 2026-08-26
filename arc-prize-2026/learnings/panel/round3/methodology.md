## Summary (2 sentences)
v3 fixes every specification error I flagged in round 2 — the exchangeable unit is now the game with an exact sign-flip test, the resampling recipe is written down and correctly demoted to a calibration role, 8 looks are enumerated with per-look α=0.0125, MDE with a formula-executed seed-raising rule is pre-registered, confirmation seeds equal selection seeds, and the extreme-value selection statistic is replaced by holdout mean with a never-touched vault — the instrument's skeleton is now sound. Two new gaps remain that must be written into the pre-registration before Phase 0 runs: the calibration check has no pre-registered failure action (reopening the discretion the plan exists to eliminate), and the gate+confirmation conjunction means realized end-to-end power at the stated MDE is ~64%, not 80%, so the seed-raising rule and the "underpowered kill" label are miscalibrated against their own target.

## Objections

**Resolution of prior-round objections:**

**[Prior MAJOR] Game×seed pseudoreplication ("~75 paired observations") — RESOLVED.** Exchangeable unit = game; gate statistic = mean of per-game mean deltas; exact sign-flip at n=18 with 2¹⁸ patterns; the v2 claim is explicitly retracted and power is redone at n=18. This is exactly the fix demanded, stated without hedging.

**[Prior MAJOR] Identical-build null undefined for the 3-vs-3 statistic — RESOLVED.** The disjoint-splits recipe is specified, its dependence acknowledged, and — the correct move — it is demoted from primary threshold to a calibration check on the sign-flip test, with a bootstrap CI on the realized rejection rate. One arithmetic note: 8 seeds give C(8,3)·C(5,3) = 560 ordered (280 unordered) disjoint splits; state which is meant so the recipe is executable as written. A new gap in this machinery is objection N1 below.

**[Prior MAJOR] No family-wise error control / no MDE — RESOLVED.** Looks enumerated (8), per-look α=0.0125 for family ≤0.10, MDE at 80% power published from Phase-0 variance components, seed counts made a Phase-0 output via a no-discretion escalation rule, and false kills made interpretable via the underpowered-kill label. The 1-seed hyperparameter screen feeding one gate look is valid for error control since the gate data is fresh. Residual power miscalibration is objection N2.

**[Prior MAJOR] Extreme-value selection statistic / burned holdout — RESOLVED.** Min-over-games dropped; selection = holdout-7 mean at 3 fresh seeds with CVaR tiebreak; holdout consumption acknowledged; vault-5 hash-frozen Aug 1 as the untouched generality report. Residuals on the vault's evidential value and final-selection multiplicity are objections N4/N5.

**[Prior MAJOR, round 1, carried] Winner's curse — RESOLVED.** Confirmation N = selection N ≥ 3; confirmation number is the reported number; contradiction reverts the merge. (The interaction of this rule with power is N2.)

**[Prior MAJOR, round 1, carried] Adaptive overfitting to public games — RESOLVED.** dev-18 / holdout-7 / vault-5 three-tier structure with the vault never used for any decision is the design I asked for.

**[Prior MINOR] σ̂ precision — RESOLVED.** χ² CI reported, criterion evaluated at both endpoints, severity restated as 2σ̂/√n_draws and labeled intended. A residual asymmetry in that criterion is N3.

**[Prior MINOR] Attenuation-blind transfer check — RESOLVED.** Disattenuated ρ with pre-registered threshold 0.5 and a defined consequence.

**New objections:**

**[MAJOR] N1: The calibration check has no pre-registered failure action.** The split-null exists to validate that the sign-flip test's realized α ≈ nominal, but the plan never says what happens if it doesn't — e.g., if seed-level nondeterminism or asymmetric per-game delta distributions make the realized rejection rate land outside the bootstrap CI around 0.0125. Without a pre-registered branch, a calibration failure in mid-July gets resolved by on-the-spot judgment, which is precisely the discretion this document is engineered to remove. Fix in one paragraph: if realized α exceeds the CI upper bound, gates run at the empirical percentile of the split-null (accepting its dependence-induced imprecision, reported); if below, proceed with the sign-flip as conservative. Also state the number of null splits evaluated and the α of the calibration test itself.

**[MAJOR] N2: End-to-end power at MDE is ~64%, not 80%, because merges require passing the gate AND the confirmation at the same threshold.** The seed-raising rule sizes N so the *gate* has 80% power at MDE, but the confirmation run (same N, same threshold: "delta below the gate threshold reverts") is a second independent test with the same ~80% power, so a true effect exactly at MDE survives both with probability ≈0.64 — and every kill/underpowered-kill label is calibrated to the wrong number. This is conservative for false merges but strategically expensive: it false-kills the thesis components at ~36% at MDE. Fix (pick one, pre-register it): (a) size N for joint 80% power; or (b) set the confirmation criterion below the gate threshold — its job is winner's-curse shrinkage detection, not independent re-testing — e.g., confirmation delta ≥ 50% of the gate critical value, with the operating characteristics of the pair stated.

**[MINOR] N3: The Phase-4 success criterion ignores the frozen-fork mean's own sampling error.** "Mean of draws ≥ frozen-fork mean + 2σ̂/√n_draws" treats the baseline mean as known, but it comes from 4–6 draws with the same σ; the SE of the *difference* is σ̂·√(1/n_new + 1/n_frozen) ≈ √2 larger, so the criterion as written is ~√2 anti-conservative relative to the stated 2-SE severity. One-line fix to the formula; keep the both-endpoints evaluation.

**[MINOR] N4: Vault-5 is a weak instrument for the load-bearing role it is assigned.** Five self-generated procedural variants, executed "exactly once" (seeds unspecified, presumably 1), generated by the team from mechanics their pipeline already targets: its SE will be enormous and its sampling frame is biased toward the pipeline's strengths, so it can detect only gross overfitting, not estimate generality. Fix: run vault at ≥3 seeds (extra seeds consume no decisions), report mean ± CI, and state in the Nov-2 report that the vault bounds catastrophic overfitting rather than estimating private-set performance.

**[MINOR] N5: Final-selection multiplicity on holdout-7 is unquantified.** The number of candidate configs entering the Phase-4/Nov-2 selection is never stated; comparing K configs on a 7-game mean at 3 seeds re-introduces a winner's curse at the one decision that can never be confirmed before the deadline. Pre-register K (it should be ≤3 given the ablation matrix is estimation-only), and publish, from Phase-0 variance components, P(select the truly better config | true Δ = 0.05, 0.10) — the CVaR tiebreak helps only inside the 1-SE band.

**[MINOR] N6: Count-threshold gates (≥6/18, zone 3–5/18, kill <3/18) rest on noisy per-game binaries with no aggregation rule.** Whether a game is "Class-A" varies by seed; the plan never states how the per-game binary is computed across 3 seeds (any seed? majority? pooled transitions?), and a permanent synthesis kill can hinge on 2/18 vs 3/18 flipping on one seed. Pre-register the aggregation rule and report the count's seed-to-seed sampling variability from the pilot alongside the zone decision.

## Questions for the authors
1. What is the pre-registered action if the split-null calibration shows realized rejection rate outside the bootstrap CI around α=0.0125 — and how many null splits will be evaluated (280 unordered or 560 ordered)?
2. Do you intend merges to require gate-pass AND confirmation-pass at the same critical value? If yes, confirm you accept ~64% end-to-end power at MDE, or state which fix from N2 you adopt.
3. How many candidate configurations will enter the Phase-4/Nov-2 holdout selection, and what is P(correct selection) at true Δ=0.1 RHAE under your Phase-0 variance components?
4. How many seeds will the vault-5 run use, and who generates/validates the procedural variants so that generation is blind to the pipeline's specific strengths?
5. What is the per-game Class-A aggregation rule across seeds for the 6/18, 3–5/18, and <3/18 thresholds, and what is the sampling SD of the count?
6. Confirm the Phase-4 success criterion will be restated as 2σ̂·√(1/n_new + 1/n_frozen), or justify treating the frozen-fork mean as noiseless.

## What I cannot judge
Kaggle kernel/vLLM/quantization engineering (Blackwell SKU claims, FP8-dequant, A40 concurrency, quota branch mechanics); whether Qwen 3.6 27B can pass the pilot (the pilot's design is now sound; the capability question is outside my remit); the plausible-effect bands' domain realism (substrate +0.10–0.25 etc. — I can only verify they are pre-registered, not that they are right); ARC-AGI-3 domain specifics (segmentation, aliasing, RESET semantics, planner abstraction adequacy); the veracity of cited papers and leaderboard forensics; competition rules.

## Verdict: MAJOR-REVISION

## Score: 8/10

Every load-bearing statistical fix I demanded in round 2 was implemented correctly and without evasion — the exchangeable unit, the resampling recipe's demotion to calibration, the enumerated look budget, the MDE machinery, the confirmation seed parity, and the selection statistic are now a defensible instrument, and I would trust the campaign it gates. The MAJOR-REVISION verdict rests on two narrow, genuinely new gaps that this plan's own no-discretion standard makes mandatory: a calibration check with no pre-registered failure branch, and a gate+confirmation conjunction whose realized power at MDE (~64%) contradicts the pre-registered 80% that the seed-escalation rule and kill labels are calibrated to. Both are a day of drafting; with those two paragraphs and the four MINOR fixes, this passes review from my seat.