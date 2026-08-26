## Summary (2 sentences)
v2 genuinely closes most of the round-1 systems holes — the Phase-0/Phase-4 quota arithmetic, the 8–10-seed null, token-parity budgeting, co-residency measurement, and duck's concurrency are all now addressed as commitments — but the compute envelope still contains one internal contradiction and one scheduling impossibility: the phase "slot" counts (81 slots Jul 7–Sep 30) cannot fit the plan's own stated worst-case cap of ~2 scored runs/week, and the Phase-3/Phase-4 overlap means the TTT component gates on Sep 14 while the ablation-matrix-plus-freeze it must feed closes Sep 12–15. These are fixable on paper before Jul 7, which is why this is a revision and not a rejection.

## Objections

**Resolution of prior objections (all seven, in order):**

**[Prior FATAL — quota arithmetic, Phases 0/4] PARTIALLY-RESOLVED.** Phase 0 is cut to 2 attempts (24 GPU-h) and Phase 4 moves the freeze to Sep 12–15 with 5–6 draws at ≤2/wk — this is the fix I asked for, and the rerun-quota question is scheduled as a Phase-0 empirical check. Residual: the fix holds only at the top line; see new objection N1 — the per-phase slot counts re-open the same hole under a different name, and the sigma experiment (4–6 identical-fork runs during Phases 1–2 "at ≤2/wk") consumes the *entire* 2-runs/wk scored cap in those weeks, leaving zero Kaggle validation runs for the Phase-1/2 components being built.

**[Prior MAJOR — A40 operating-point / FP8 parity] PARTIALLY-RESOLVED.** Token-parity denomination, the tokens/s measurement, the AWQ fallback, and the re-cost to 20–40 A40-h/sweep and ~$1,900 all directly answer the objection. Two residuals: (1) the 8–10-seed × 25-game null itself is never costed at token-parity budgets — at the plan's own 20–40 A40-h per 3-seed sweep, the null is ~60–130 A40-h inside Phase 0's two weeks, requiring 2–3 concurrent pods; state that explicitly and the decision rule if measured parity blows the Phase-0 window. (2) Token parity fixes *total* tokens but not *batch/turn* parity: duck runs 28 concurrent threads on one vLLM server, and a 48 GB A40 carrying ~27 GB of dequantized weights has KV headroom that may not sustain 28 × 32k contexts, so per-60s-turn token yield differs from Kaggle even at aggregate parity. The 60 s turn caps must also be re-denominated in tokens, not wall (this contaminates the Phase-0 pilot — see N3).

**[Prior MAJOR — 3 seeds / unpowered gates] RESOLVED.** The 8–10-seed frozen-baseline null, paired per-game permutation tests against the null 90th percentile, BH-FDR regressions, de-gating the F(2,2) variance claim, and fresh-seed confirmation are exactly the requested repair. Conditional caveat only: the null must be run at gate-identical (token-parity) budgets, which folds into the previous item.

**[Prior MAJOR — vLLM co-residency / TTT VRAM] PARTIALLY-RESOLVED.** The 1-day 0.90-vs-0.75 utilization measurement with the tax entering the 15% kill rule ex ante is what I asked for. Residual: the measurement is on the A40 while the tax is paid on the still-unnamed Kaggle SKU, whose VRAM and KV headroom may differ qualitatively; add a single in-kernel smoke measurement of co-resident tokens/s on Kaggle before Phase 3 merges anything.

**[Prior MAJOR — per-run wall-clock accounting] RESOLVED.** Concurrency is now stated (28 threads, one vLLM server, token-bound, 25 games × 2.2 h in ~12 h wall), extraction of realized per-game wall/tokens from the source bundle is a Phase-0 task, and budget-denominated criteria are re-denominated in (tokens, wall).

**[Prior MINOR — spot preemption] RESOLVED.** Checkpoint/resume mandatory, instance type logged. Nit: *excluding* preempted runs from paired comparisons biases the sample (preemption probability correlates with run length, i.e., with configs that run long); prefer resume-and-include, exclude only non-resumable runs.

**[Prior MINOR — BFS caps] RESOLVED.** Numeric caps (10 s / 2×10⁵ nodes, anytime best-partial) are in the Phase-1 spec.

**New objections:**

**[MAJOR — borderline FATAL] N1: The slot ledger contradicts the plan's own worst-case quota claim.** Phases 0–4 list 7+14+21+21+18 = 81 "slots" over ~12 weeks, while the resource budget declares a hard worst-case cap of 2 scored runs/wk + 6 h smoke — i.e., at most ~30–36 scored runs in the same window — and claims "all scheduled inside this worst case." Both cannot be true unless a "slot" is something materially cheaper than a scored 12 h run, and the proposal never defines a slot's GPU cost. If slots presume scored reruns are quota-free, then the default schedule is built on the optimistic branch of the very question Phase 0 is supposed to test — the inversion of the plan's stated logic. Fix: define slot cost in GPU-hours, publish a per-week ledger under both quota assumptions, and give a pre-registered slot-triage rule for the pessimistic branch (which components lose their Kaggle validation runs first).

**[MAJOR] N2: The Phase-3/Phase-4 overlap makes the freeze unschedulable as written.** Phase 3 runs Aug 25–Sep 14; Phase 4 starts Sep 8 and freezes Sep 12–15. The ablation matrix "at Phase-0-derived seed counts" plus fresh-seed confirmation must therefore complete in ≤7 days *including* a TTT arm whose own gate does not resolve until Sep 14 — one day before the latest freeze. At 20–40 A40-h per 3-seed sweep, a joint ablation over even 4 configs at 8-seed counts is several hundred A40-hours; it cannot start after Sep 14 and finish by Sep 15. Fix: either move the Phase-3 gate to Sep 7, pre-declare that TTT ships only in the Nov 2 build (never the Sep 30 milestone), or shift the freeze to Sep 18–20 and accept 4 draws.

**[MAJOR] N3: The Kaggle GPU SKU is still unnamed, and the Phase-0 pilot's "runtime-realistic caps" are wall-denominated on the wrong hardware.** Two rounds in, the single most load-bearing systems fact — what silicon the scored run executes on — remains unstated, despite being extractable in an hour from the duck notebook's logs (`nvidia-smi`) or comp docs; the FP8-vs-dequant ratio, co-residency headroom, and every parity number hang off it. Concretely damaging now: pilot (c) tests the riskiest premise (27B synthesis) under "60 s turns" on the A40, where 60 s buys a different number of generated tokens than 60 s on the Kaggle SKU — the kill/continue decision for Phase 2 is being made at an uncontrolled token budget. Fix: name the SKU in the revision (not in Phase 0), and denominate pilot turn caps in tokens using the measured throughput ratio.

**[MINOR] N4: Phase-0 week 1 has zero quota margin.** 2 × 12 h + 6 h smoke = 30 h exactly; one OOM, kernel timeout, or metadata error consumes an attempt with no retry, and the "≥0.9 within 2 attempts" gate then collides with the Jul 24 kill date on infrastructure noise rather than signal. Reserve one attempt for week 2 or pre-commit a rule distinguishing infra-failure (free retry) from score-failure (counts).

**[MINOR] N5: Preemption-exclusion bias** — as noted above under the resolved spot objection: make resume-and-include the default; log and report the exclusion count per sweep so the paired-comparison denominator is auditable.

## Questions for the authors
1. Define a "slot" in GPU-hours and show the per-week Kaggle ledger for all 81 slots under both branches (reruns quota-counted vs. quota-free). Which slots are cut first in the pessimistic branch?
2. What is the Kaggle GPU SKU (from the duck notebook's logs — available today)? Given it, what is the expected A40:Kaggle tokens/s ratio and the token-denominated pilot turn cap?
3. What does the 8–10-seed × 25-game null cost in A40-hours at token-parity budgets, how many concurrent pods does it require, and what happens to the Jul 20 exit gate if the measured parity budget doubles the estimate?
4. During Phases 1–2, the sigma experiment claims ≤2 runs/wk — the full scored cap. Which weeks get sigma runs vs. dev-validation runs, explicitly?
5. How does a component gated on Sep 14 (TTT) enter an ablation matrix + fresh-seed confirmation that must finish before a Sep 12–15 freeze? Show the day-level Sep 8–15 schedule.
6. What is the maximum sustainable vLLM concurrency (KV-cache-limited) for the dequantized 27B at 32k context on the 48 GB A40, and how are gates affected if it is ≪28?

## What I cannot judge
The validity of the cited literature claims (Rodionov 58.12% and its leakage audit, Rudakov's Preview result, AERA's compression claim); the algorithmic merit of the exploration substrate, segmentation stack, Class-A metric design, and MDL acceptance rule; the statistical fine print of the permutation/FDR machinery beyond its compute cost (that is the methodology reviewer's lane); the quality of the 25 opus sims; and competition rules/eligibility beyond their quota implications.

## Verdict: MAJOR-REVISION

## Score: 6/10