## Summary (2 sentences)

The revision adopts, nearly verbatim, the three fixes my round-14 FATAL demanded (pooling unit defined, P(pass) published, cumulative binding endpoint with per-window demotion to mechanism + non-inferiority), fixes the truncation, retires the winner's-curse banking list, and — creditably — killed component (d) via its own sealed pre-registered threshold. However, the gate that "seals on this round" is powered against a stack that no longer exists (the alternative still includes the dead (d) and un-shrunk banking), its α retains a multiplicity correction whose rationale the recalibration itself removed, its sole remaining per-window kill (the −0.10 guard) has no published operating characteristics, and the A17 72B screen is single-noisy-sample inference — all of which must be fixed *before* sealing, not after.

## Objections

**Prior-objection resolution status (required first):**

**[Prior FATAL] Sign-flip prong arithmetically unpassable — PARTIALLY-RESOLVED.** A14 adopts all three demanded fixes: pooling unit = game (n=24 per-game means, the passable reading), P(pass | expectations) published, and the binding decision moved to one cumulative stack-vs-W0 look with per-window decisions demoted to mechanism + non-inferiority; the FAIL consequence is now non-destructive (ship with honest label). The architecture is fixed. But the calibration is not: the published sketch is internally inconsistent — with *exactly* 4 nonzero clean positives the minimum sign-flip p is 2⁻⁴ = 0.0625, so P(pass | 4 positives, clean) is exactly **0**, not "≈ 0.05"; the "≈ 0.05" must come from an undocumented distributional assumption over the realized positive count, which is nowhere shown. And note 6-0 still fails (2⁻⁶ = 0.0156 > 0.0125); the pass region is {7-0, 8-0, 9-0, 9-1, …}. See new objections N1/N2.

**[Prior MAJOR] Truncated circulation — RESOLVED.** Single-part delivery, all five parts carry END lines, the document ends with the literal END OF PROPOSAL line, per-part lengths sum consistently with the declared total. (I cannot verify the sha256 values themselves; I accept the structural integrity as presented.)

**[Prior MAJOR] Banking winner's curse — PARTIALLY-RESOLVED.** A16 retires the frozen ft09/sc25/re86 list, pre-registers an online game-agnostic retry policy, and requires a permutation-calibrated shrinkage recomputation before the window opens — the right fix, correctly specified. But the recomputation is promised, not delivered, and the un-shrunk +0.03–0.08 band still sits inside the +0.07–0.19 stack expectation that A14.2 declares as the cumulative look's alternative (see N1).

**[Prior MAJOR] Compressed-bench regime transfer — RESOLVED** (A15 adopts exactly the demanded rule: provisional inclusion, full-budget confirmation replicate, relabeled compressed-regime quantities, published full-budget trigger frequencies), with one residual ambiguity promoted to new objection N5.

**[Prior MINOR] Interim-look contradiction — RESOLVED.** A14.3: guard evaluated only at the window's sealed look after seed 3; no per-seed evaluation.

**[Prior MINOR] 0.56× conversion / uncalibrated thresholds — PARTIALLY-RESOLVED.** A20 declares the conversion an assumption with a 0.4–0.8× band (adequate). But (a)'s mechanism-prong threshold "unseen budget deaths halved vs the 3 control seeds" remains a same-transcript-calibrated criterion with an n=3 denominator and no stated noise band; (d)'s equivalent is moot only because (d) is dead.

**New objections:**

**[MAJOR] N1: The cumulative gate seals THIS ROUND against a stale alternative.** A14.2 powers the binding look against +0.07–0.19 rail, but that sum includes (d) (+0.02–0.04, killed by A18 on the same day) and un-shrunk banking (+0.03–0.08, pending A16 shrinkage and full-panel sign-off it may never receive). The realistic retained stack — (a) + (b) + possibly (c)-as-Reki-signature — has a summed expectation of roughly +0.02–0.06 rail, and the published P(pass) ≈ 0.2–0.4 is correspondingly an overestimate; under the shrunken alternative the look may drift back toward the false-negative machine A14 was written to abolish. Fix before sealing: recompute the stack expectation and P(pass) with (d) excluded and banking either excluded or shrinkage-haircut, and publish the binomial sketch's actual assumptions (including why "4 positives, clean" yields 0.05 rather than 0).

**[MAJOR] N2: α = 0.0125 has lost its multiplicity rationale and now needlessly suppresses power at the single binding look.** The 0.0125 (= 0.05/4) made sense when four per-window score tests were binding; A14 demotes all per-window score statistics to descriptive, leaving ONE binding score decision — yet keeps the Bonferroni-sized α, which is why 6-0 still fails. Either state explicitly what family of binding tests α = 0.0125 now corrects for (if the secondary RHAE prong and the cumulative prong form the family, say so and size it as 2, not 4), or re-set α = 0.05 one-sided for the single cumulative look, which moves 5-0 and 6-0 into the pass region and roughly doubles P(pass) under the honest alternative. This is a one-line change with first-order power consequences and it must happen before the seal.

**[MAJOR] N3: The non-inferiority guard (pooled Δlc ≤ −0.10 → flag OFF) is now the only per-window score kill, and its false-kill probability is unpublished.** W0's own baseline shows run-level lc totals of {13, 15, 22} across 3 seeds (0.52–0.88 lc/draw — a 0.36/draw spread), so run-level noise dwarfs the −0.10 threshold; paired same-seed ON/OFF diffs will be tighter, but trajectory divergence after the first differing action means the paired-diff noise is unknown and nowhere estimated. If P(pooled Δlc ≤ −0.10 | true Δ = 0) is even ~0.15 per window, then across 3–4 windows there is a ~40–50% chance of falsely killing at least one neutral component — the A5 pattern reborn as a guard instead of a gate. Fix: calibrate the guard using the existing war_eval cross-seed diffs (an upper bound on paired noise) plus the W1 control seeds, publish P(false kill) per window and familywise, and if it exceeds ~0.10, replace the point threshold with an SE-based boundary or require the −0.10 breach on ≥2 of 3 seeds.

**[MAJOR] N4: The A17 72B go/no-go is single-noisy-sample inference on a margin smaller than the comparator's own seed noise.** "GO iff ≥2 levels beyond the 27B baseline summed across 4 games" — but the 27B baseline on ft09 alone spans 0–2 levels across the 3 certified seeds, so the entire GO margin fits inside one game's seed variance; as written, one lucky/unlucky 72B run (seed count unspecified) against an unspecified 27B comparator statistic can flip the campaign's only registered wall-closer either way. Fix before the screen runs: define the comparator as the 3-seed per-game mean (Σ = stated number), require the 72B sum to exceed comparator + a margin derived from the observed 3-seed spread (or run 2 seeds of 72B and gate on the min), and pre-register both.

**[MINOR] N5: Budget regime of the binding cumulative look is unspecified.** A15 requires a full-budget confirmation replicate "before the cumulative look can claim score credit," which implies the 3-seed cumulative look itself may run compressed; if so, a PASS is a compressed-regime conclusion and the n=1 full-budget replicate cannot upgrade it to "confirmed floor/mid raiser" at scored-regime budgets. State the regime in the sealed text; if compressed, the PASS label must carry the compressed-regime qualifier.

**[MINOR] N6: A18 demonstrates post-filing edits to a sealed amendment.** The document filed ~13:30 EDT contains a RESULT block observed ~14:30 EDT — i.e., a sealed pre-registration file was modified after its seal, and only the document's own narrative attests the threshold preceded the observation. The decision itself was self-adverse (a kill), so no harm here, but the same channel would launder a favorable post-hoc threshold. Fix: sealed thresholds go in their own file with hash committed (e.g., in the panel record) before the measurement script runs; results go in separate append-only artifacts.

## Questions for the authors (numbered)

1. What is the retained-stack composition assumed for the cumulative look's alternative now that (d) is dead — and what are the recomputed stack expectation and P(pass)? (Blocks the seal.)
2. Publish the A14.5 binomial sketch: what distribution over positive/negative game counts produces "P(pass) ≈ 0.05" at 4 expected positives, given P(pass | exactly 4 clean positives) = 0?
3. What family of binding tests does α = 0.0125 now correct for, given per-window score tests are demoted? Why not 0.05 for the single look?
4. What are the units of the −0.10 guard (Δlc per draw, pooled over 24 games × 3 seeds?) and its estimated false-kill probability under Δ = 0?
5. A17: how many seeds per model tier, and is the 27B comparator the per-game 3-seed mean, max, or a specific seed?
6. Does the binding cumulative look run at compressed or full budgets?
7. Does (c)-as-Reki-dead-signature (Part D priority 2) carry its own counting bound and A19 prong, or does it inherit the dead (d)+(c) flag's registration? Its disposition is explicitly punted to R15 and must be sealed if it enters W1/W2.

## What I cannot judge

The vLLM/AWQ 72B throughput and quantization-quality engineering; Kaggle rules legality of banking/replay and the harness-side BFS executor; the correctness of the transcript forensics and sim implementations underlying the counting bounds (I take the counted events as given and audit only the inference built on them); the technical merit of the OCM/resync arXiv adaptations; and the sha256 values themselves, which I cannot recompute.

## Verdict: MAJOR-REVISION

The revision is a genuine and well-executed response — every structural fix I demanded was adopted, and the A18 self-kill is exactly what a functioning pre-registration regime looks like. But the recalibrated gate proposes to seal *this round* with a stale power analysis, an orphaned α, an uncalibrated sole kill-criterion, and a single-sample v4 screen; sealing in that state would re-create the A5/round-13 failure mode at the cumulative level. Fix N1–N4 (each is a bounded, one-day computation on existing data) and the gate can seal next round.

## Score: 6/10