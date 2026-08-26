## Summary (2 sentences)
v3 delivers essentially every systems fix I demanded — the slot ledger is now denominated in GPU-hours with the pessimistic quota branch as the default schedule, TTT is committed to the Nov-2 build only (my proposed fix, verbatim), the Kaggle SKU is named (RTX PRO 6000 Blackwell) with hour-1 log confirmation and token-denominated caps throughout, the null is costed with a fallback, and the infra/score failure taxonomy plus resume-and-include close the minors. One real residual survives: the ledger's own tail weeks (wks 11–12.5) are oversubscribed — 1 sigma + 5–6 draws cannot fit under the 2-scored-runs/week cap — which directly changes n_draws in the milestone success criterion; this is a one-paragraph errata that must land before Phase 4 (Sep 8), not before Jul 7, so I do not hold the campaign start hostage to it.

## Objections

**Resolution of prior objections:**

**[Prior MAJOR-borderline-FATAL N1 — slot ledger contradiction] PARTIALLY-RESOLVED.** Slot classes are defined in GPU-hours (A = 20–40 A40-h local, B = 12 GPU-h scored, C = ≤1 h smoke), the pessimistic branch is the default schedule, the weekly class-B ledger is published (17–18 ≤ 24), and the triage order is pre-registered with sigma/draws protected — this is the fix. Residual arithmetic error: wk 11 (Sep 15–21) carries sigma #5 plus draws, wk 12 carries draws, and "wk 12.5" is Sep 29–30; at 2/wk the tail capacity is 1+2+2 = 5 runs maximum, with the final two 12-h scored draws landing on the deadline days themselves with zero infra margin — so "5–6 draws" is 6-impossible and 5-only-with-zero-slack. Since the success criterion is mean ≥ baseline + 2σ̂/√n_draws, n = 4 vs 6 moves the severity threshold by √1.5 — this is not cosmetic. Cheapest fix: the wk-1–2 duck scored runs are byte-identical frozen-fork executions and are therefore valid σ samples; count them, cut scheduled sigma runs to 3 (wks 3, 5, 7), and free wks 9/11 slots for draws.

**[Prior MAJOR N2 — Phase-3/Phase-4 freeze unschedulability] RESOLVED.** TTT ships only in the Nov-2 build, the Sep-30 freeze covers Phase-0–2 only, and Phase-3 ablation + confirmation moves to Sep 15–Oct 15 on October quota — this is option 2 of my proposed fixes, adopted exactly.

**[Prior MAJOR N3 — unnamed Kaggle SKU / wall-denominated pilot caps] RESOLVED.** RTX PRO 6000 (Blackwell, 96 GB, native FP8) is named with source (duck's `GPU_NAME_PATTERNS`), confirmation scheduled for Phase-0 hour 1, all turn caps re-denominated in tokens (duck p90 tokens/turn), and a 3-game on-SKU synthesis anchor with a pre-registered ≤15 pp discrepancy rule bounds the A40↔Blackwell numeric gap for the Phase-2 entry decision. Caveat folded into a new minor: the anchor's hour cost inside the smoke budget is unquantified.

**[Prior MINOR N4 — zero week-1 quota margin] RESOLVED.** One attempt reserved for week 2; infra-failure (free retry) vs score-failure (counts) taxonomy pre-registered.

**[Prior MINOR N5 — preemption-exclusion bias] RESOLVED.** Resume-and-include is the default; exclusion counts reported per sweep.

**[Carried residuals from round-1 partials] RESOLVED.** The null is costed (45–95 A40-h, 2–3 concurrent pods) with a pre-registered fallback if parity doubles it; max sustainable A40 vLLM concurrency at 32k is a Phase-0 measurement with per-turn yield re-denomination; the co-residency tax gets an in-kernel co-resident tokens/s smoke on the actual SKU before any Phase-3 merge.

**New objections:**

**[MAJOR] N-A: Tail-week class-B oversubscription (the surviving fragment of prior N1).** As detailed above: 1 sigma + 5–6 draws in wks 11–12.5 violates the 2/wk cap the plan itself declares hard, and the milestone success criterion's denominator (n_draws) hangs on it. Required before Phase 4: a day-level Sep 8–30 class-B schedule that fits the cap, with the success criterion restated at the achievable n and the duck-runs-as-σ-samples accounting decided now (it changes both the sigma schedule and σ̂'s df).

**[MINOR] N-B: On-SKU anchor cost inside the smoke budget is unquantified.** The 3-game synthesis battery on the RTX PRO 6000 must fit in ≤6 h/wk smoke during the same Phase-0 weeks that also consume smoke for preflight of the two scored duck attempts. State the battery's expected kernel-hours (games × synthesis calls × tokens ÷ measured on-SKU tokens/s) and which week's smoke budget carries it; if it exceeds ~4 h, it collides with preflight and needs to split across weeks.

**[MINOR] N-C: Phase-0 internal sequencing of the token caps.** The pilot's token-denominated turn caps derive from duck's p90 tokens/turn and T_game "extracted from logs" — state whether these are extractable from the *public* duck bundle today (no dependency) or only from your own week-1 scored rerun; in the latter case an infra-failed week-1 run leaves the pilot capless until week 2 against a Jul 20 exit gate, so pre-register an interim cap.

**[MINOR] N-D: No cost-blowout rule for gate sweeps (only for the null).** Back-of-envelope on the A40: ~48 GB − ~27 GB dequantized weights ≈ ~21 GB KV headroom, i.e., plausibly only ~4–7 sustainable concurrent 32k contexts, and aggregate tokens/s at batch ~5 vs batch ~28 can differ 2–3×; the 20–40 A40-h/sweep bracket may hold, but nothing says what happens if the measured number lands above 40. Pre-register the sweep analogue of the null's fallback (e.g., seed cap + MDE-printed decision) so a throughput surprise doesn't silently eat the $150/wk cap.

**[MINOR] N-E: The class-B ledger stops at Sep 30.** Oct 1–Nov 2 (~4.5 wks ≈ 9 scored runs) must carry Phase-3 Kaggle confirmation, the absorption drill's validation runs (potentially unbounded), and any final-submission verification; publish the October ledger with the same triage discipline. Also confirm the Phase-4 ablation matrix's A40 bill at worst-case (8-seed) counts fits Sep 8–12 within 2–3 pods and the ≤$150/wk + $300 reserve — at 4 arms × ~53–107 A40-h it brushes both.

## Questions for the authors
1. Show the day-level Sep 8–30 class-B schedule fitting 2/wk: which days carry sigma #5 and each draw, and what is the final n_draws in the success criterion? Do the wk-1–2 duck runs count as σ samples (they are byte-identical frozen-fork executions)?
2. What is the on-SKU 3-game synthesis anchor's estimated kernel-hours, and which weeks' 6-h smoke budgets absorb it alongside scored-run preflights?
3. Are T_game and the p90 tokens/turn extractable from the public duck bundle today, or only from your own scored rerun — and if the latter, what interim token cap governs the pilot if the week-1 run infra-fails?
4. What is your KV-headroom estimate for max concurrent 32k contexts on the A40 (show the per-context KV GB), and what is the pre-registered rule if a measured sweep exceeds 40 A40-h?
5. Publish the Oct 1–Nov 2 class-B ledger: Phase-3 confirmation runs, absorption-drill validation runs, and final-submission verification, under the pessimistic quota branch.

## What I cannot judge
The validity of the cited literature claims (Rodionov 58.12% and its leakage audit, Rudakov's Preview result, AERA's compression claim); the algorithmic merit of the exploration substrate, segmentation stack, Class-A metric design, MDL acceptance rule, and planner abstraction; the statistical fine print of the sign-flip/FDR/disattenuation machinery beyond its compute cost (methodology reviewer's lane); the quality of the 25 opus sims; and competition rules/eligibility beyond their quota implications — including whether the weekly quota actually resets in time for Sep 29–30 runs, which the authors should verify against Kaggle's reset schedule, not my assumption.

## Verdict: ACCEPT
(Conditional: the N-A ledger errata — day-level Sep 8–30 schedule and restated n_draws — must be filed before Phase-4 execution begins Sep 8; nothing in N-A blocks the Jul 7 start.)

## Score: 8/10