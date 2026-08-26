## Summary (2 sentences)
v3 resolves or credibly operationalizes every objection I raised in round 2 — the SKU is named and anchored, the pilot has closed-loop arms with a pre-registered token allocation, all gates are dev-18-denominated, the trigger is specified, the Phase-2 context table exists with a measured-p75 re-close rule, and seed count is now a formula-executed Phase-0 output — this is disciplined convergence, not lawyering. Two execution-blocking defects remain, both in the Phase-0(c) pilot that everything downstream leans on: the deciding LOGO numbers are not constrained to the Phase-2 runtime input regime (recorded histories vs. the 6k transition window), and the 3-game on-SKU quantization anchor's ≤15 pp acceptance criterion is arithmetically unmeasurable on binary per-game Class-A at n = 3.

## Objections

**Resolution audit of my round-2 objections:**

**[PARTIALLY-RESOLVED] Prior [MAJOR] N1 — pilot tests a different numeric model than runtime.** The mechanism I asked for is present and better than asked: SKU named (RTX PRO 6000 Blackwell, FP8-native, confirmed from logs hour 1), 3 pilot games re-run on-SKU inside the smoke budget, ≤15 pp pre-registered discrepancy, kernel-anchored entry if tripped, all turn caps token-denominated. But the acceptance test as written is ill-posed — see NN2 below. The architecture of the resolution is correct; the arithmetic is not.

**[RESOLVED] Prior [MAJOR] N2 — offline pilot doesn't test the joint in-kernel budget.** Both fixes delivered: ≥2 closed-loop pilot games where the agent self-collects transitions via scripted `explore()`, tokens-to-first-Class-A reported against a pre-registered 30/40/5/25 allocation of measured T_game, and Phase-2 entry requires ≥1 closed-loop pass. Two closed-loop games is thin (see NN5), but the bridging experiment I demanded exists and gates entry. Resolved.

**[RESOLVED] Prior [MINOR] N3 — dev/holdout inconsistency.** All gates restated on dev-18 (≥6/18, zone 3–5/18, pilot bar 4/10 consistent at ~40% vs 33%); 25-game numbers demoted to reporting. Exactly the fix requested.

**[RESOLVED] Prior [MINOR] N4 — trigger parameters unspecified.** Progress ≐ new deduped archive state OR score/level increment; N ∈ {5, 10, 20} screened at 1 seed with only the selected N facing the gate (correctly costed as one look). Resolved.

**[RESOLVED] Prior carryover [MAJOR] — Phase-2 context budget.** The Phase-2-entry table now exists with a template line-item (2.5k), model source pinned to the measured p75 of pilot-accepted lengths with a mandatory re-close if exceeded, a >4k post-refactor rejection rule consistent with the MDL penalty, and generation counted inside the 32k (30.2k total). This answers all three sub-points I raised. (Residual concern on whether 8k generation headroom is sufficient is NN4, minor.)

**[RESOLVED] Prior carryover [MAJOR] — power pre-committed before variance known.** MDE at 80% power published from the Phase-0 variance decomposition; seeds/arm raised 3→5→8 by formula until MDE ≤ the component's pre-registered band lower bound; residual underpowered kills labeled and revisit-eligible. This is the pre-registration I asked for.

**New objections:**

**[MAJOR] NN1 — The deciding LOGO pilot numbers are not constrained to the Phase-2 runtime input regime.** The LOGO arms synthesize from "recorded histories" (full duck trajectories, bounded only by the 32k window), but the runtime synthesize/verify/refactor turn sees a 6k transition window plus a 1k verify report per its own entry table — the deciding numbers may therefore measure synthesis from ~5× the evidence the runtime loop will ever hold in context. A LOGO pass at 32k-history followed by a 6k-window failure in-kernel is fully consistent with this design, and it is precisely the pilot-passes/kernel-flops failure mode v3 was restructured to exclude; the 2 closed-loop games mitigate but do not carry the entry decision (only ≥1 must pass). Fix: pre-register that the deciding LOGO prompts conform to the Phase-2 context table — transition evidence capped at 6k tokens (with the same eviction/selection policy the runtime will use) — and report the unconstrained-history number separately as an upper bound.

**[MAJOR] NN2 — The quantization anchor's acceptance criterion cannot resolve 15 pp on 3 games of binary Class-A.** Per-game Class-A is binary; a rate over 3 games takes values in {0, 33, 67, 100} pp, so "|kernel − local Class-A rate| ≤ 15 pp" is satisfiable only at exactly 0 and is otherwise auto-tripped — the sole bound on the plan's quantization risk (Risk 2 says "bounded by the 3-game kernel anchor") is an ill-posed test. If "synthesis battery" means multiple attempts per game, say so and define the statistic. Fix: define the anchor discrepancy on the continuous, paired per-game held-out exact-match (pp) and/or acceptance score, averaged over the 3 games with a pre-registered threshold sized to its noise; keep binary Class-A flips as a reported side flag, not the criterion.

**[MINOR] NN3 — Entry-gate scaffold multiplicity is uncounted.** "≥4/10 games Class-A across the 3 scaffolds" reads as best-of-three-scaffolds per game — three shots per game inflates the per-game pass rate relative to the single scaffold (or fixed router) that actually ships in the kernel, where running all three costs ~3× the 40% synthesis token allocation. Pre-register the counting rule: either the entry count is under the single scaffold selected on disjoint games, or the runtime loop genuinely runs multi-scaffold and the token allocation is re-closed accordingly. The Phase-2 dev gate would eventually catch this, but the entry decision commits three weeks.

**[MINOR] NN4 — 8k generation headroom is assumed, not measured.** Free-form synthesis of a ~3.5k-token model plus reasoning tokens (Qwen thinking-mode traces routinely exceed 4–8k on code tasks) can overflow 8k, silently truncating model source — the worst failure mode for executable-model synthesis. The pilot already logs everything needed: pin the headroom line to the p90 of measured generation lengths per scaffold, and pre-register truncation-rate as a scaffold-selection criterion (diff-refactor should win this dimension; verify it does).

**[MINOR] NN5 — Closed-loop pilot arms use a pre-Phase-1 `explore()` on 2 games.** The scripted `explore()` used in the week-1–2 pilot predates the Phase-1 trigger-N screening and gating, so tokens-to-first-Class-A is measured under a throwaway exploration policy on the minimum defensible sample; a marginal closed-loop result (1 pass / 1 fail) is nearly uninformative. Pre-register how a 1/2 closed-loop outcome is interpreted at the entry gate, and re-report tokens-to-first-Class-A under the gated Phase-1 `explore()` before the Phase-2 mid-phase zone decision.

## Questions for the authors (numbered)
1. Do the deciding LOGO synthesis prompts conform to the Phase-2 context table (6k transition window, 1k verify report), or do they use full recorded histories up to 32k? If the latter, why is that number decision-grade? (NN1)
2. Define the on-SKU anchor statistic precisely: how many synthesis attempts per game in the "battery," and what continuous quantity does the ≤15 pp threshold apply to? (NN2)
3. At the Phase-2 entry gate, does a game count as Class-A if *any* of the three scaffolds passes, and if so, which single configuration ships at runtime under the 40% synthesis allocation? (NN3)
4. What is the measured generation-length distribution (per scaffold) in the pilot, and what is the truncation rate at 8k headroom? (NN4)
5. If closed-loop results come back 1-pass/1-fail on the 2 games, what does the entry gate conclude? (NN5)
6. The Phase-2 table's transition window is 6k tokens — under duck's segmentation object-graph encoding, how many transitions is that on a median dev game, and is that consistent with the ≥30-transition prospective verification requirement plus enough training evidence for synthesis?

## What I cannot judge
Kaggle quota/rerun branch mechanics and the class-B weekly ledger arithmetic (ops reviewer); RunPod A40 pricing, concurrency, and preemption accounting; the fine print of the sign-flip permutation construction, the disjoint-split null's dependence structure, χ² CIs on σ̂, and the disattenuation procedure (methods reviewer); the AERA/Rodionov/Rudakov citations and Preview-LB claims, taken at face value; whether the vault-5 procedural-variant games are a valid generality proxy for the private set; October competitive dynamics beyond their scaffolding implications.

## Verdict: MAJOR-REVISION

## Score: 7/10

The revision is near-accept: every prior objection is resolved or correctly mechanized, and both remaining MAJORs are localized pre-registration fixes to Phase 0(c) — but they sit directly under the pilot whose numbers decide Phase-2 entry, so they must be fixed before Jul 7, not after.