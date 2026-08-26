## Summary (2 sentences)
v2 is a genuinely responsive revision: the fatal dependency collapse is now stated honestly, the 27B synthesis pilot is moved to week 1 under a leakage-audited, degenerate-baseline-scored metric, scripted routing is primary, and a context budget exists — this is what a fixed proposal looks like, not a re-lawyered one. However, three execution-blocking gaps remain: the Phase-0 pilot will run a *different numeric model* than the Kaggle kernel (FP8-dequant/AWQ on Ampere vs. FP8 on the Kaggle SKU), the pilot's offline recorded-history setting doesn't test the joint in-kernel budget question (self-collected data + synthesis + planning inside 2.2h/game), and the context budget table is a Phase-1 artifact that demonstrably does not close for Phase 2's payloads.

## Objections

**Resolution audit of my round-1 objections:**

**[RESOLVED] Prior [FATAL] — differentiators reduce to one untested dependency.** The "14 games" claim is retracted, `bfs_solve()` is explicitly a generic planner over a supplied transition function, the dependency graph is redrawn in the Thesis, the pilot is Phase 0, and the fallback ("honestly a well-run fork; we accept that floor") is stated without euphemism. The dependency still exists — it always will — but it is now tested first and priced correctly. This is the correct resolution.

**[RESOLVED] Prior [MAJOR] — Rodionov de-risking category error.** The Phase-0(c) pilot is exactly the experiment I specified, plus improvements I didn't ask for: leave-one-game-out template audit, plan-transfer re-scoring of the Opus sims, three pre-registered scaffold variants so a kill indicts the model rather than one prompt, and a Class-A definition with held-out 5-step rollouts against identity/lookup baselines under a mechanical MDL penalty. The "prompt it better" hand-waving is gone. (But see new objection N1 — the pilot as specified tests the wrong *model instance*.)

**[RESOLVED] Prior [MAJOR] — LLM self-routing as default.** Scripted no-progress trigger is now the primary architecture; LLM routing is the A/B variant. The prompt-distribution-shift risk is now covered by the null-calibrated permutation gate rather than the underpowered 2/3-seed guard.

**[PARTIALLY-RESOLVED] Prior [MAJOR] — no context/token accounting.** A worst-case table with an eviction order exists and sums to ~29.5k/32k — for **Phase 1**. It does not close for Phase 2: (i) "templates" appear in the eviction order but have no line in the budget; sim-template scaffolds are code and plausibly 2–5k tokens each; (ii) "world-model blocks ~1.5k, never evicted" — the 22 Opus sims that define Class-A are almost certainly longer than 1.5k tokens of code, so either the runtime models must be far smaller than the validated exemplars (unvalidated assumption) or the never-evict guarantee breaks; (iii) 2.5k slack plus 5k "thinking headroom" must also cover *generation* of model code and refactor diffs in a 60s turn. Fix: a second, Phase-2-entry budget table for the synthesize/verify/refactor turn, with a measured token length distribution of the pilot's accepted models.

**[PARTIALLY-RESOLVED] Prior [MAJOR] — criteria pre-committed before variance known.** The structure is now right (null first, gates instantiated from the null, permutation test, fresh-seed confirmation, FDR). But the explicit power calculation I asked for is still absent: variant arms are fixed at 3 seeds *in the pre-registered gate definition* before Phase 0 reveals whether 3 seeds give acceptable power against effect sizes worth shipping, and the null (8–10 seeds) vs. variant (3 seeds) asymmetry is unexplained. Fix: pre-register now that variant seed count is a Phase-0 *output* (chosen to hit, say, 80% power at the minimum effect of interest), not the constant 3.

**[RESOLVED] Prior [MINOR] — notes-carried confabulation.** Clean-restart vs. notes-carried arms, RHAE-denominated, variance claim de-gated.

**[RESOLVED] Prior [MINOR] — local ≠ Kaggle environment.** Token parity, cross-environment rank correlation, duck concurrency extracted. (The quantization aspect turned out to be worse than I flagged — promoted to N1 below.)

**New objections:**

**[MAJOR] N1 — The Phase-0 pilot tests a different numeric model than the one that must perform at runtime.** The A40 is Ampere: no FP8 hardware. The proposal's own parenthetical — "weight-dequant; AWQ-int8 fallback if unusable" — means the pilot's Class-A rate is measured on FP8-dequantized-to-BF16 or AWQ-int8 Qwen 27B, while the kernel runs the FP8 checkpoint on the Kaggle SKU. Quantization deltas of this kind are known to hit code generation and long-horizon tool-use disproportionately; a pilot pass on dequant weights (effectively higher precision) could overstate kernel capability, and an AWQ fail could kill a viable Phase 2. Fix: run at least 2–3 pilot games *on the Kaggle SKU itself* (this fits the existing smoke-test hours) as a calibration anchor, or rent one FP8-capable GPU for the pilot; pre-register the acceptable local-vs-kernel Class-A discrepancy.

**[MAJOR] N2 — The pilot validates synthesis from curated recorded histories; the Phase-2 gate requires synthesis from self-collected data inside a shared 2.2h/game budget, and nothing bridges the gap.** Recorded duck trajectories are a favorable data distribution (whatever duck happened to see); the runtime loop must fund its own data collection via explore(), synthesis, verification rollouts, and BFS planning out of the *same* token/wall budget that currently produces baseline behavior — and the pilot measures none of that joint accounting. A pilot pass followed by an in-kernel flop on Aug 24 is fully consistent with this design, which recreates the exact "discovery too late" failure mode Phase 0 was restructured to prevent. Fix: (a) pilot arm where transition histories are truncated/subsampled to what N explore()-turns realistically yield; (b) a pre-registered per-game token allocation (explore/synthesize/plan/act) as a Phase-2 entry artifact, with the pilot reporting tokens-to-first-Class-A-model per game against it.

**[MINOR] N3 — Dev/holdout protocol is internally inconsistent with the Phase-2 gate.** The gate reads "Class-A on ≥8/25 and plan-transfer ≥40%," but the holdout-7 is supposed to be touched only at phase boundaries with one sweep; a Class-A count over 25 either burns holdout mid-phase or is a boundary-only number that can't drive the 4–7 zone decision in time. Restate all Phase-1–3 gates over dev-18 (e.g., ≥6/18) and reserve the 25-game number for reporting.

**[MINOR] N4 — The scripted trigger's core parameters are unspecified.** "N no-progress turns" leaves both N and the definition of progress (new level? new archive state? new segmentation hash?) open; under the archive, "no new deduped state in N turns" and "no level in N turns" are very different policies with different failure modes on high-aliasing games. Pre-register the definition and the N sweep as part of the Phase-1 spec, not an improvised constant.

## Questions for the authors (numbered)
1. What quantization/precision will the Phase-0 pilot actually run, and what evidence bounds the Class-A rate gap between that configuration and the FP8 kernel model? (See N1 — can 2–3 pilot games run on the Kaggle SKU as an anchor?)
2. What per-game token allocation (explore / synthesize / verify / plan / act) does the Phase-2 loop assume, and what fraction of the pilot's recorded histories would N turns of runtime exploration actually reproduce?
3. What is the measured token length of the accepted Opus sims, and how does it reconcile with "world-model blocks ~1.5k, never evicted"?
4. Is the variant-arm seed count (3) a constant or a Phase-0 output? If constant, what minimum detectable effect does 3 seeds give at α=0.1 against your measured null?
5. The Phase-2 gate counts over 25 games; the holdout-7 is boundary-only. Which is it? Restate the gate over dev-18.
6. Define "no-progress" for the scripted explore() trigger, and give the pre-registered N (or the sweep that sets it).

## What I cannot judge
Kaggle quota/rerun mechanics and whether "2 scored runs/wk + 6h smoke" and the sigma-experiment scheduling are feasible (ops reviewer); the RunPod/A40 cost arithmetic beyond the quantization-capability point in N1; the statistical fine print of the permutation/FDR machinery beyond the power-analysis gap I flagged (methods reviewer should confirm the 90th-percentile-of-null construction); the cited arXiv results and Preview-LB claims, taken at face value; October competitive dynamics beyond their scaffolding implications.

## Verdict: MAJOR-REVISION

## Score: 6/10