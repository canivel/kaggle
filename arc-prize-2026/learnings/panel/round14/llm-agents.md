## Summary (2 sentences)

This revision is a substantial, unusually honest discharge of the prior round's demands: it names the mechanism class truthfully ("grinder cracking without a model change does not exist"), supplies counted per-component Δ bounds against the real scorer, unbundles (f), drops (g), builds the budget-faithful bench I demanded, and suspends the confounded su15 verdict. However, the pre-registered gate it seals is arithmetically unpassable at the proposal's own expected event counts — the primary prong will kill every window by construction — and the sole registered wall-closer (v4 72B swap) rests entirely on frontier-tier evidence with zero measurement at the tier actually proposed.

## Objections

**Resolution of prior-round objections (all seven, in order):**

1. **[MAJOR] Build rail cannot evaluate budget-regime mechanisms — RESOLVED.** §3's A10 bench does exactly what I demanded: compressed per-game caps (~40% of observed median) scaled so triggers fire ≥1/run on ≥5 games, a canary verifying trigger counts *before* the gate seals, and — critically — the concession that a non-firing trigger yields LB-accumulation status, never a build-rail kill. The residual regime-transfer risk is filed below as a new objection, but the false-negative-by-construction failure I flagged is fixed.

2. **[MAJOR] (a)+(f) bundling violates one-flag rule — RESOLVED for (a)+(f); the new (c)+(d) bundle is PARTIALLY-RESOLVED.** (f) now ships first, standalone, unflagged, with its own quick screen — exactly the demanded remedy. The new (c)+(d) single-flag exception is at least *declared* and instrumented with separate counters, but (i) their own rule says "no exceptions without full-panel sign-off," and registration-in-the-doc is not sign-off — this panel round is where that sign-off happens or doesn't; (ii) "a pass decomposes mechanically" overstates what firing counters give you: they establish which mechanism *fired*, not which caused the score Δ, and on a FAIL you still cannot tell which half to kill. I accept the bundle only because (c) is below MDE/2 alone (their own R13 rule) and both live in one code path; state explicitly that a FAIL parks both.

3. **[MAJOR] (g) targeting category error — RESOLVED by withdrawal.** (g) per-game budget re-allocation is absent from the backlog entirely. I cannot confirm it isn't resurrected in the unseen Part 2; if it is, my prior objection stands verbatim.

4. **[MAJOR] Wrong backlog / no grinder design content / no Δ estimates — RESOLVED.** This document *is* the grinder design doc, filed before Jul 20, with per-candidate counting bounds against the exact scorer and the honest per-game table (§4) conceding Δclears = 0 on the two canonical grinders. The residual — that the named wall-closer still has near-zero evidentiary content — is filed as new objection N2.

5. **[MAJOR] su15 wall verdict confounded by (f) — RESOLVED.** Verdict suspended per A13, su15 excluded from all post-(f) evaluations per A12, re-probe registered as a separate frontier-tier experiment after the (f) fix. Note: the re-probe has no timeline slot in §5 — give it a date or it will drift.

6. **[MINOR] Ledger content never tested — PARTIALLY-RESOLVED.** The (d) redesign counts firing events from transcripts (dead-target re-probes 8–20×/run, ~30 offline-refutable hypotheses on ft09), which supplies the firing-frequency half of the counterfactual replay I asked for. The "would the injected FACT have altered the next action" half remains untested; the honest ceiling (+0.08) makes this tolerable, but the mechanism prong should capture it (see new objection N4).

7. **[MINOR] Research sweep grades papers by agreement — CANNOT VERIFY.** Not addressed in the visible Part 1; presumably Part 2. Carried forward unresolved.

**New objections:**

**[MAJOR] N1: The sealed gate's primary prong is arithmetically unpassable at the proposal's own expected effect sizes — the v3 stack will end empty by construction.** A one-sided exact sign-flip test at α = 0.0125 with zeros dropped requires ≥7 nonzero same-sign pairs to reach significance (2⁻⁷ ≈ 0.008; n=6 all-wins gives p ≈ 0.016 > α). The proposal's own §4 arithmetic expects **1–4 extra clears per run panel-wide** — i.e., roughly 1–4 nonzero Δlc pairs per window across 3 pooled seeds. Therefore every window fails prong 1 regardless of whether the mechanism truly pays, the "FAIL on score prongs with mechanism firing → flag OFF" rule fires, and the cumulative stack ends with zero flags — despite §2 explicitly acknowledging components sit below the ~0.2 MDE and invoking "cumulative-stack gate design" as the remedy. The cumulative design changes the *baseline*, not the *power*; the fix must land before the W1 look (Jul 21): either (i) keep-on-{mechanism prong PASS + non-inferiority PASS} with a single end-of-stack pooled score gate on the full v3 stack vs. null (where the summed +0.07–0.19 rail expectation is at least near the 3-seed MDE), or (ii) re-derive α/n to match realistic nonzero-pair counts and re-seal with panel sign-off. As sealed, this is a false-negative machine — my Round-13 objection reincarnated in power terms rather than trigger terms.

**[MAJOR] N2: The only registered wall-closer (war-v4, 72B swap) has zero evidence at the proposed tier, and its throughput guard may be unpassable as written.** The +150-game-point basis is GPT-5.6; nothing measures whether Qwen3.6-72B possesses *any* of the NOT-distillable capabilities (one-shot correspondence induction, representation invention, BFS-on-command) that separate GPT from 27B — the 27B→72B step may recover 0% of a gap that is frontier-specific, and "capturing even 10%" is exactly the hand-waving this panel exists to stop. The cheap bridging experiment exists inside the free 30 GPU-h/wk: run 72B-4bit on ft09/sb26/lp85/vc33 under the identical harness *before* Aug 1 scoping, and report levels cleared plus measured actions/hour. Separately, the throughput guard ("total actions within 10% of 27B baseline") at 2.5–3× slower decode is likely unsatisfiable if wall-clock binds — specify which budget (actions vs. 8h wall-clock) actually binds on the scored rail and compute the expected 72B action count before registering a gate that cannot pass.

**[MAJOR] N3: Gate decisions made at the 40%-compressed regime are deployed at the 100% regime with no pre-registered transfer check.** The compressed bench fixes my false-negative objection but creates the mirror risk: (a)'s sentinel calibrated and validated under 40% caps fires far more often than at full budget, and can pay at 40% (early abandonment is cheap when everything is budget-starved) while being neutral or negative at 100% (premature abandonment of levels that were actually convertible). Pre-register one full-budget 3-seed confirmation run of the final accepted stack vs. null before any LB push, and report per-component trigger frequencies at full budget from existing war_eval transcripts alongside the compressed-bench counts so the compression factor is an explicit, checkable assumption rather than an ambient one.

**[MINOR] N4: The (c) mechanism prong is partially vacuous and misses the value-bearing observable.** "Verbatim-resubmit count = 0" is guaranteed by the hard-block's presence — it verifies the code shipped, not that it mattered. §2's own analysis says reclaimed actions pay *only if a new hypothesis family follows the block*, and the CONCEPT-lock finding (never leaves family #1 in 13 runs) predicts it won't. Add a post-block novel-family-rate counter to the prong; that is the actual mechanism-of-value observable, and it doubles as the free test of prior-objection 6's untested half.

**[MINOR] N5: Internal inconsistencies in the gate spec.** (i) "Sealed look after seed 3; no interim looks" contradicts the non-inferiority guard "pooled Δlc ≤ −0.10 **at any look**" — if the guard is evaluated per-seed, those are interim looks and affect the error accounting; specify. (ii) The 0.56× rail→LB conversion assumes multiplicative effect transfer from a ratio of two population means (1.636/0.922 inverted); this is a modeling convenience, not a measurement — label the LB-unit numbers as such. (iii) The document is truncated mid-sentence in its Verdict paragraph ("and the Ju") and is "Part 1 of 2" with Part 2 unseen; per panel rules I file this formally and have reviewed only the visible text.

## Questions for the authors (numbered)

1. Show the arithmetic under which any single window passes prong 1: how many nonzero Δlc pairs do you expect per window (per §4's own 1–4 clears panel-wide), and what is the minimum n for the sign-flip test at α = 0.0125? If the answer is "it can't pass," what is the actual keep/kill rule?
2. Which budget binds on the scored Kaggle rail — action count or 8h wall-clock — and what is the projected total-action count for 72B-4bit at measured decode throughput? Is the "within 10% of 27B" guard satisfiable at all?
3. Has Qwen3.6-72B (any quant) been run on even one grinder transcript-length episode? If not, why is v4 scoping deferred to Aug 1 when the bridging probe fits in this week's 30 GPU-h allowance?
4. For (a): what are the sentinel's firing frequencies at *full* budget, counted from existing war_eval v1–v3 transcripts, versus the projected 40%-compressed frequencies?
5. Does the (c)+(d) single-flag exception have full-panel sign-off as of this round, per your own "no exceptions without full-panel sign-off" rule, and what is the pre-registered disposition on a window FAIL — both parked?
6. When, by date, does the A13 su15 re-probe run? It appears in no timeline row.
7. Where is Part 2, and does it contain the research-sweep re-grading and confirmation that (g) is withdrawn rather than relocated?

## What I cannot judge

- The exact-scorer arithmetic's numerical correctness (validated "0e+00 vs Tufa's 500 runs" per the authors; I take the scorer implementation on trust — the statistics/evaluation reviewer should audit `rhae_score` and the marginal-value table).
- The 96 GB VRAM / AWQ 4-bit feasibility claims and vLLM KV-headroom arithmetic for 72B (systems reviewer's domain; I judge only the *evidential* status of the swap, which is my N2).
- Kaggle competition-rules legality of the model swap and of level banking via deterministic replay (compliance question, outside my remit).
- Anything in the unseen Part 2, including the research-sweep re-grading (prior objection 7).

## Verdict: MAJOR-REVISION

The revision discharges five of my seven prior objections — including both load-bearing MAJORs about gate regime and backlog honesty — and the intellectual honesty of §1 and §4 is exactly what this panel demanded. But the gate it "seals on filing" cannot pass its own primary prong at its own predicted event counts (N1), and the sole wall-closer is unevidenced at the proposed model tier (N2); both are fixable within the stated timeline and neither may be discovered post-hoc at a gate look.

## Score: 6/10