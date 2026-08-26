## Summary (2 sentences)
This is the strongest brief of the campaign from a design-discipline standpoint: the seed-2 null screen is reported without spin, the order-statistics curve I demanded is finally published (and I verified its arithmetic independently), and the n=3 ledger is handled with no standardized claims. However, three load-bearing statistical instruments — the MDE quoted for the R2 A/B, the A3 variance stopping rule, and the ledgers themselves under the pending seed audit — are each currently miscalibrated or at risk of invalidation, so the plan cannot proceed to a ledger-ON window unchanged.

## Objections

**Prior-objection resolution status (required first):**

**[Prior #1 — recompute MDE/power under reconciled variance] PARTIALLY-RESOLVED.** An MDE now exists ("≈ 0.14 at n=5/5") and a variance stopping rule (A3) is on the record and was actually applied — this is the machinery I demanded. But the MDE's σ provenance is undisclosed, and back-solving shows it must use σ ≈ 0.074–0.077 (the frozen fork), not the war ledger's own σ̂ = 0.108, under which the MDE is 0.19, not 0.14. See new MAJOR #1.

**[Prior #3 — publish the corrected order-statistics artifact] RESOLVED.** The E[max]-vs-k curve is in the brief, and I verified it: E[max of k N(0,1)] constants (1.16 at k=5, 1.54 at k=10, 2.16 at k=40, ≈2.50 at k=110) are correct, all table cells reproduce, and the k=40 kill-shot on YUTO (0.922 + 0.213×2.16 ≈ 1.38 ≪ 1.86) is sound arithmetic supporting the right conclusion: only per-draw mean matters. The reach table makes this load-bearing explicitly. This item is closed.

**[Prior #5 — briefing/document headline discrepancy] PARTIALLY-RESOLVED.** The authors did their part: a screenshot artifact is now cited by path and the numbers (best 1.02, leader 1.86, 40 entries) are transcribed inline, consistent across rounds. But the panel briefing STILL says 0.43/1.56, and I cannot open the PNG from the review context to independently verify. Residual action sits with the organizers: reconcile the briefing header, or the panel is formally reviewing gaps computed on disputed numbers.

**[Prior new-MAJOR 1 — publish MDE/power/window-count/stopping rule before Jul 17] PARTIALLY-RESOLVED.** Of my four required items: (ii) an MDE is quoted and (iv) a stopping rule (A3) exists and was applied; but (i) σ provenance is absent and (iii) windows-to-power for the predicted effect is nowhere. Two of four, and the two missing ones are the ones that determine whether the A/B is interpretable. Escalated in new MAJOR #1.

**[Prior new-MAJOR 2 — engineered replay validation before Jul 17] PARTIALLY-RESOLVED.** `bank_fire_validation.json` (fired + score-invariant on ar25 & s5i5; frame_divergence aborts on sc25/m0r0 by design) gives the instrument nonzero sensitivity offline — a genuine improvement over the vacuous canary. But on-kernel sensitivity remains exactly zero (seed-2 canary: 0 wins → replay structurally unreachable, correctly labeled vacuous), and the flagship mechanism game sc25 is one where replay *aborts*. Answering Q1's embedded question: no, this does not fully satisfy A2 for licensing window 1 tonight — option (c) is, as the author senses, one day early; the attempts-counting war-v2 canary must fire on-kernel first.

**[Prior MINOR 1 — LOO sensitivity on the p=0.0074 screen] RESOLVED-IN-SUBSTANCE.** Seed 2 answered the fragility question more decisively than the LOO I asked for: the seed-1 primary collapsed (−0.008, p=0.539), the brief says so plainly, and the mechanism story is now correctly narrowed to the replicating games (sc25 +1.8 both seeds). LOO remains scheduled per the disposition table; acceptable.

**[Prior MINOR 2 — n=1 "consistent with" inference] RESOLVED.** No n=1 claims appear; the n=3 comparison is labeled descriptive, and I verified the Welch t (SE = √(0.108²/3 + 0.074²/5) = 0.071, t = 0.035/0.071 = 0.49 ✓).

**New objections:**

**[MAJOR] The quoted MDE uses the wrong σ, and the design is unpowered for its own predicted effect — publish the window budget or the Jul 17+ A/B is uninterpretable by construction.** "MDE ≈ 0.14 at n=5/5" back-solves to σ ≈ 0.074 (frozen fork); using the war arm's own σ̂ = 0.108, MDE = 2.8×0.108×√(2/5) ≈ 0.19. Meanwhile the reach table predicts R2 delivers +0.05–0.10 LB — *below the MDE at either σ*. Power for Δ = 0.08 at n=5/5 is ~30–40%; reaching 80% power requires n ≈ 14/arm (σ=0.074) to 29/arm (σ=0.108), i.e., 28–58 nightly draws — a material fraction of the remaining calendar to Nov 2. Actionable fix: before window 1, pre-register the σ used with provenance, the terminal n, the power at that n for Δ ∈ {0.05, 0.08, 0.10}, and — if the answer is "infeasible" — the fallback decision currency (e.g., offline Δlc as primary with LB as a non-inferiority harm-check with a stated margin), so a null A/B is read as "unpowered" and not "ledger does nothing."

**[MAJOR] The A3 gate (σ̂ < 0.15) evaluated at df=2 has essentially no discriminating power — "LB windows remain live" was certified by an instrument that passes bad states almost half the time.** If true σ = 0.20 (well inside the brief's own reported CI of [0.056, 0.678]), P(σ̂ < 0.15 | df=2) = P(χ²₂ < 1.125) ≈ 0.43 — a coin flip. This is the same pathology as the vacuous banking canary, now in the variance gate: a check that cannot fail informatively cannot certify. Actionable fix: restate A3 on the χ² CI *upper bound* (e.g., "windows live iff CI-hi(σ) < X at df ≥ 4"), or defer the check to n≥5 as the brief's own "recompute at n=5" hints, and publish the gate's operating characteristics (P(pass | σ) at σ = 0.10, 0.15, 0.20).

**[MAJOR] The seed audit (#726552 ADAPT) can invalidate every existing ledger — pre-register the segmentation rule BEFORE applying any fix, and prove kernel v1/v2/v3 are seed-only diffs.** If the audit finds and fixes unseeded `random`/`numpy` fallback paths, the post-fix agent is a different stochastic process: the control σ̂ = 0.074 (n=5) and war ledger n=3 no longer estimate the variance of future draws, and every MDE, A3 check, and band in this brief silently dies. Decide now, in writing: fix found → ledgers reset (or changepoint-modeled), or fix deferred until after the R2 A/B under the old build. Relatedly, "seed 3 (kernel v3)" must be certified as a seed-only diff from v1/v2 (attach the diff or build hashes) — if any code changed across seed pulls, tomorrow's 3-seed pooled gate look mixes versions and the A1 test is invalid.

**[MINOR] The 0.00-censoring rule (Q3) needs a corroboration requirement or it biases the ledger mean upward.** Pre-registering before contact with data is exactly right and I endorse adopting it tonight — but a blanket "exact 0.00 = infra" rule censors legitimate zeros (own-code crash, genuine wipeout), inflating the mean. Fix: censor only with corroborating infra evidence (timeout marker, log signature), and report ledger stats both with and without the censored draw as a standing sensitivity line.

**[MINOR] "1.08 is the highest single draw of the campaign" is selection-on-the-maximum and carries no information.** With 8 total campaign draws (frozen n=5 + war n=3), E[max] ≈ 0.922 + 1.42×0.074 ≈ 1.03 at σ̂ — observing a 1.08 maximum is unremarkable under the null of no difference. Strike the bolding or annotate it as expected-max arithmetic; the brief elsewhere earned the right not to backslide into max-hunting.

## Questions for the authors (numbered)
1. Which σ, from which ledger, produced "MDE ≈ 0.14 at n=5/5"? Publish the formula and provenance line in the prereg.
2. What is the planned terminal n per arm for the R2 A/B, and the power at that n for Δ = +0.05, +0.08, +0.10 LB? If <50% for the predicted effect, what is the pre-registered fallback decision currency?
3. Are war-eval kernel v1/v2/v3 byte-identical except for seed? Attach build hashes or a diff summary before tomorrow's pooled gate look.
4. If the seed audit finds unseeded fallback paths, do the existing control (n=5) and war (n=3) ledgers reset? State the rule before running the fix.
5. What conversion factor underlies "Δlc ≥ +0.08 ≈ +0.05–0.10 LB," given your own pooled data shows *negative* RHAE conversion (−0.098)? Is it fitted or assumed?
6. On Q1: my vote is (a) war draw #4 — the war arm's σ CI ([0.056, 0.678], df=2) is the binding uncertainty and df 2→3 shrinks it most; (c) is not licensed per A2 (see above). On Q4: confirmed, no peeking adjustment is required for a fixed-n=3 rule filed before seed 2 — provided the interim look could not have triggered early stopping and Q3's answer to my question 3 is "seed-only."

## What I cannot judge
The engineering internals of banking/replay and whether frame_divergence aborts on sc25/m0r0 are truly "by design"; the substantive merit of AutoMem/ECHO as mechanisms (I can only judge their evaluation plans); Kaggle infrastructure causes of 0.00 scores; GPU/push budget realism; and whether the P1–P5 behavioral observables (regex/SequenceMatcher extraction) faithfully measure the intended constructs — I can judge only their pre-registration form, which is adequate.

## Verdict: MAJOR-REVISION

## Score: 6/10