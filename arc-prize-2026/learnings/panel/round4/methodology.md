## Summary (2 sentences)
v4 resolves all six of my round-3 objections exactly as demanded — the calibration failure branch is pre-registered with the correct split count, the confirmation is redefined as c/2 shrinkage detection with correct operating characteristics (I verified the arithmetic: gate 0.80 × confirmation Φ(3.08−1.12) ≈ 0.78 joint, per-look false-merge ≈ 0.0016, family ≈ 0.013), the Phase-4 criterion uses the SE of the difference, the vault is re-scoped honestly, K ≤ 3 with published selection-error probabilities, and the count-threshold binaries have a majority rule with reported sampling SD. However, the v4-redefined quantization anchor ([llm-NN2]) reintroduces the exact pseudoreplication pattern I killed in round 2 — its SE is computed from 15 attempt pairs while the decision it licenses generalizes across games from a between-game sample of df = 2 — and its acceptance rule is a point-estimate threshold, not an equivalence test.

## Objections

**Resolution of prior-round objections:**

**[Prior MAJOR] N1: Calibration check had no pre-registered failure action — RESOLVED.** All three branches are pre-registered (empirical percentile if anti-conservative, proceed-as-conservative if under-rejecting, proceed if CI covers nominal), the split count is fixed at 280 unordered (correcting v3's arithmetic as I noted), and the calibration test's own α = 0.05 is stated. Residual gaps in this machinery are new objections N9 and N10 below — but the discretion hole I flagged is closed.

**[Prior MAJOR] N2: End-to-end power ~64% vs stated 80% — RESOLVED.** Option (b) adopted precisely as specified: confirmation criterion = c/2, operating characteristics published, and — critically — the seed-raising rule and underpowered-kill label are recalibrated to the joint 0.78 number rather than the gate-only 0.80. The stated numbers check out under the normal approximation. One specification residue is N11.

**[Prior MINOR] N3: Phase-4 baseline-mean sampling error — RESOLVED.** Criterion restated as 2σ̂·√(1/n_draws + 1/n_frozen), both-endpoints evaluation retained.

**[Prior MINOR] N4: Vault-5 evidential weakness — RESOLVED.** ≥3 seeds, mean ± CI, and the claim is explicitly demoted to "bounds catastrophic overfitting, not private-set estimate" with the sampling-frame bias acknowledged in the pre-registration itself. This is claim-scoping in lieu of blinded generation — acceptable.

**[Prior MINOR] N5: Final-selection multiplicity — RESOLVED.** K ≤ 3 pre-registered, P(select truly better | Δ = 0.05, 0.10) published from Phase-0 variance components, with a no-discretion rule cutting K to 2 if P(correct | 0.10) < 0.7.

**[Prior MINOR] N6: Per-game binary aggregation — RESOLVED.** Majority over 3 gate seeds, seed-to-seed SD of the count published and reported beside every zone/kill decision. (Trivial residue: the ≥2/3 rule is stated for 3 seeds only; see Q5.)

All round-1/round-2 fixes (exchangeable unit, look budget, MDE machinery, confirmation seed parity, selection statistic, three-tier splits) remain intact in v4; nothing was silently regressed.

**New objections:**

**[MAJOR] N7: The redefined quantization anchor's SE is computed at the wrong level — attempt pairs, not games — reintroducing the pseudoreplication error this document was purged of in round 2.** The anchor statistic is a mean over 3 games, its SE is taken "from the 15 attempt pairs," and the validity condition 10 pp ≥ 2·SE uses that SE — but the 5 attempts within a game share that game's true quantization sensitivity, so the 15 pairs are clustered with between-game df = 2, and the anchor's *purpose* is to certify A40→Blackwell transfer for the *other* 7 pilot games and the Phase-2 runtime, which is between-game generalization the 15-pair SE cannot support. As written, the anchor will look "measurable" (small SE) exactly when it shouldn't, and a passing anchor licenses local numbers to decide Phase-2 entry. Fix, pre-registered before Phase 0c runs: (a) compute the anchor as the mean of the 3 per-game mean paired diffs with SE = SD(per-game means)/√3 and a t-critical value at df = 2, or (b) scope the acceptance to the worst-case per-game |gap| ≤ 10 pp with per-game SEs — and if neither is satisfiable at 3 games (likely, at df = 2), the pre-registration should say so now and default to kernel-anchored entry rather than discovering it in mid-July.

**[MINOR] N8: The anchor acceptance rule is a point-estimate threshold, not an equivalence test.** "|mean paired diff| ≤ 10 pp, valid if 10 ≥ 2·SE" passes a true 10 pp gap ~50% of the time and a true 15 pp gap ~16% of the time at SE = 5 — anti-conservative for a criterion whose job is to certify equivalence. Fix in one line: require the 90% CI of the (game-level, per N7) mean paired difference to lie entirely within ±10 pp (TOST at α = 0.05); the existing 5→10 attempt expansion branch already handles the precision failure mode.

**[MINOR] N9: Calibration branch (ii) leaves the power machinery miscalibrated in the conservative direction.** If the realized rejection rate is materially below 0.0125, the sign-flip test's true power at the published MDE is below the 0.78 joint target, so the seed-raising rule stops raising too early and "underpowered kill" labels are attached against an MDE that is too optimistic. Pre-register the symmetric action: in branch (ii), either recompute MDE at the realized rejection rate (or run gates at the split-null empirical percentile, as in branch (i)), or at minimum print the realized-α-adjusted MDE beside every kill decision.

**[MINOR] N10: The calibration check's own power is unstated and may be near zero, making branch (iii) vacuous.** At α = 0.0125 the expected rejection count over 280 mutually dependent splits built from only 8 seeds is ≈ 3.5, so the hierarchical-bootstrap CI on the realized rate will be wide; if it nearly always covers nominal, the check cannot distinguish realized α = 0.0125 from, say, 0.03, and "calibration passed" is theater. Pre-register the check's minimum detectable miscalibration (e.g., the smallest realized α whose CI excludes nominal with 80% probability, computed from the same bootstrap), and report it beside the calibration verdict so the panel and the authors know what the pass actually certifies.

**[MINOR] N11: The confirmation criterion c/2 is underspecified in two states of the world.** (a) Which SE defines c — the gate run's realized SE, the confirmation run's own SE, or a pooled value? These differ, and the choice changes the operating characteristics quoted. (b) Under calibration branch (i), gates run at the split-null empirical percentile: state explicitly that c (and hence c/2) is then the empirical critical value, so the confirmation stage inherits the recalibration rather than silently reverting to 2.24·SE.

## Questions for the authors
1. For the quantization anchor: will you recompute the SE at the game level (df = 2) or adopt worst-case per-game acceptance — and given df = 2, what is the pre-computed probability that the anchor is "measurable" at all under your expected within-game attempt variance? If near zero, do you pre-commit now to kernel-anchored entry?
2. Will the anchor acceptance be restated as a TOST/CI-within-±10pp rule (N8), or do you accept ~50% pass probability at a true 10 pp gap?
3. In calibration branch (ii), is the MDE recomputed at the realized rejection rate, and if not, how is the underpowered-kill label protected from an optimistic MDE?
4. What is the minimum detectable miscalibration of the 280-split check (N10)? Publish it with the calibration verdict.
5. Which SE defines the gate critical value c used in the c/2 confirmation criterion, and does c become the empirical split-null critical value under branch (i)? Also: when the seed-raising rule moves N to 5 or 8, what does the per-game "majority" binary become (≥3/5, ≥5/8)?
6. The confirmation runs fresh seeds on the same 18 games; your joint-power arithmetic treats gate and confirmation as independent. Confirm this independence holds under your variance decomposition (i.e., that game×treatment interaction is treated as fixed across the two runs) and that the calibration check covers the asymmetric-τ case.

## What I cannot judge
Kaggle kernel/vLLM/quantization engineering substance (whether FP8-dequant→BF16 *should* produce ≤10 pp differences — I judge only the anchor's statistics); whether Qwen 3.6 27B can pass the pilot; the plausible-effect bands' domain realism; ARC-AGI-3 domain specifics (segmentation, aliasing, RESET economics, planner abstraction, the 6k-token transition-window adequacy); the LOGO scaffold-selection content (its counting rules are statistically sound; whether the scaffolds are the right three is outside my remit); cited papers, leaderboard forensics, and competition rules.

## Verdict: MAJOR-REVISION

## Score: 9/10

All six prior objections were resolved without evasion and with correct arithmetic — the gate/confirmation/calibration instrument is now something I would sign my name to. The single new MAJOR is not a carried failure but a regression introduced by v4's own anchor redesign: an SE computed over 15 clustered attempt pairs presented as if it supported between-game generalization from df = 2, guarding a decision (local-vs-kernel anchoring of Phase-2 entry) that runs in the first two weeks. That is one paragraph to fix — game-level SE or worst-case per-game acceptance, plus the CI-within-band form — and with N9–N11 written in, this passes from my seat.