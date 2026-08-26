## Summary (2 sentences)
The revision is a genuine response: the reallocator EV is retracted on real replay data, the scheduler semantics are fully specified, the R2 gate now tests cross-game generalization with a holdout, and the SOTA-differencing gap is addressed via the fork-delta audit. However, the R1 primary lever's EV silently assumes that a mid-run RESTART re-samples the cross-seed attempt distribution — a harness-semantics claim about context state that is nowhere specified — and the revision never explains why the team's own fork sits 0.29 *below* the parent it claims is the Milestone-1 winner, which undermines the base on which every ported delta and lever EV is stacked.

## Objections

**Resolution of prior-round objections:**

**[LA-M1] Winning notebook absent — PARTIALLY-RESOLVED.** The rebuttal ("the duck *is* the winner; we are its fork") is fair, and the R0 fork-delta audit of the 1.28–1.56 band with a delta table, game-agnostic classification, and 2-window gates per ported delta is exactly the deliverable I asked for. But it opens a hole the revision does not address: the parent scored 1.21 and your substrate draws mean 0.922 (best 1.02). A −0.2 to −0.3 gap to your own upstream — the same magnitude as your bounded v2 cost [−0.54, −0.2] — means your fork may be net-harmful relative to vanilla, and every lever EV and porting target is computed on the wrong base. See new objection N1.

**[LA-M2] R2 shortlist hand-waving — PARTIALLY-RESOLVED.** The commitments are correct (decision table, one-page designs with token cost/node, component fidelity gates before GPU, BFS specified as stall-scoped *executed* exploration, the state-coverage metric answering RL-m1). But the decision table is "due with the Aug 3 deliverable," i.e., it may be authored contemporaneously with — or after — reading the forensic transcripts, which is precisely the post-hoc mapping the objection banned. Fix: commit the observable-signature → intervention-class → falsifiable-prediction table to `ITERATION_LOG.md` **before the first forensic transcript is read**, not by Aug 3. Also, the exec-WM gate metric ("next-state accuracy ≥70%") is undefined: exact-frame match, per-object, or per-changed-cell? Per-frame exact match at 70% and per-cell accuracy at 70% are wildly different bars; specify.

**[LA-M3] Reallocator EV contradiction — RESOLVED.** The counterfactual replay was run, the +0.10–0.30 EV was retracted in print, L1 was demoted, and the replacement lever is derived from measured quantities (flip-game bimodality, 11.1% FP rate). This is how a retraction should look. Residual concern about the replacement's own assumptions is filed as N2.

**[LA-M4] L1/L2 thrash — RESOLVED.** Per-attempt counter resets on restart, cumulative attempt counter never resets, cap 2, park-dominates-restart precedence, dead games bounded at ≤270 actions with the simulation published with the build. Complete and correct as specified.

**[LA-M5] Generalization gate — RESOLVED.** ≥2 distinct games × ≥2/3 seeds each, plus the r11l directional holdout with a pre-published falsifiable prediction that can block confirmation, is strictly stronger than either of my proposed fixes.

**[LA-m1] Baseline staleness — RESOLVED.** Rolling 6-draw control, per-game drift statistic, freeze-confirm-recenter rule, version-pinned null10 with an explicit refresh trigger. The power question at the σ̂ upper endpoint remains the statistics reviewer's lane.

**New objections:**

**[MAJOR] N1: The −0.29 gap to your own parent is unexplained and contaminates the entire arithmetic.** If the vanilla Tufa duck scored 1.21 and your fork draws 0.922 mean, either (a) the environment drifted/hardened since Milestone-1, (b) your fork's modifications are net-harmful, or (c) the model/config differs — and the plan cannot distinguish these, yet the answer changes everything: under (b), the single cheapest +0.2–0.3 intervention is *reverting to vanilla*, ahead of any fork-band porting or R2 spend. Fix: R0 must include the unmodified vanilla duck as a gated candidate (2 windows through the standard gate) before any fork-band delta is ported; if vanilla ≥ substrate, vanilla becomes the porting base and all lever EVs are re-derived against it. The "stack gate vs vanilla-duck fork" mention in ME-m1 suggests the artifact exists — so this costs 2 windows and zero dollars.

**[MAJOR] N2: RESTART's context semantics are unspecified, and the +0.10 EV is load-bearing on them.** The bimodality evidence (16/25 flip games) is measured *across seeds* — i.e., across independent agent contexts. A mid-run RESET re-samples that distribution only if it spawns a fresh analyzer context for that game (clean scratchpad, no carried summaries, no polluted history); if any per-game memory survives the reset, the second attempt is a correlated draw and the EV derivation collapses — your own context-pollution results (ar25, su15) say carried context is exactly what kills attempts. Additionally, the 0.4 depth discount is asserted with no provenance and cannot be measured from null10 (which contains no restarts). Fix: (i) specify RESET = fresh episode + fresh per-game agent context, verified in the transcript log; (ii) before the window gate, run one free local seed with the scheduler active on 3–4 flip games and report the empirical second-attempt good-mode rate vs the cross-seed p — that single number validates or kills the 0.4 discount.

**[MINOR] N3: Fork-band deltas must pass the same token-overhead kill rule, and the audit's premise needs one verification.** Ported deltas (prompt edits, prev-frame handling) change tokens/action; the plan applies the >10% tokens/action kill rule to R2 items but does not state it applies to R0 ports — it must, since a +0.1 delta that inflates tokens/action 15% degrades every other game under the wall-clock cap. Also verify that the 1.56 leader's kernel is actually Milestone-eligible and public; if the top of the band is closed-source, the "existence proof that +0.1–0.35 sits in cheap game-agnostic deltas" loses its upper anchor and the porting target should be restated against the highest *audited* fork.

**[MINOR] N4: The submitted document is incomplete.** The text truncates mid-sentence in §R1; §R2's body (decision table skeleton, shortlist one-pagers, the r11l prediction template), §R3, the window/quota ledgers, and §Risks are referenced in the change-log but absent from the reviewed artifact. Several "Accepted" dispositions are therefore verified only against the change-log's own summary of itself. Fix: file the complete document; unverifiable dispositions carry no credit at the next round.

## Questions for the authors (numbered)
1. Does RESET clear the analyzer's per-game context/scratchpad entirely, or does any summary/memory carry across attempts? Where in the harness is this enforced, and how will it be verified in transcripts?
2. What is the provenance of the 0.4 depth discount in the restart EV, and what measured number would cause you to revise it?
3. Why does your substrate draw 0.922 when its claimed parent scored 1.21 — drift, fork regressions, or config? What experiment in R0 distinguishes these?
4. Will the forensics→intervention decision table be committed before the first forensic transcript is read (not merely "by Aug 3")? Commit hash, please.
5. For the exec-WM ≥70% gate: accuracy of *what* — exact next frame, changed cells, or object-level transitions — and on how many held-out transitions?
6. Is the 1.56 leader's kernel confirmed public and Milestone-eligible, or is the fork-band audit's upper anchor an inference?

## What I cannot judge
The statistical machinery: the χ² CI on σ̂, the printed false-promote/false-kill rates, the rule-of-three p-values behind the two-game gate, the multiplicity accounting (0.14–1.5 expected false promotions), and the paired sign statistic over ~20 games — I defer all of these to the statistics reviewer. I also cannot independently verify the null10 replay numbers, the leaderboard forensics (82% within 2 days of open-source), or the Kaggle quota/commit-hour accounting; I take the RunPod cost provenance ($14–28 for 3 seeds) at face value pending the pre-registered 1-seed calibration.

## Verdict: MAJOR-REVISION

## Score: 6/10