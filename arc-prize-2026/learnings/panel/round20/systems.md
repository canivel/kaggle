## Summary (2 sentences)

The v3 canary bought real information at the cost of a scored window: the 72B-AWQ envelope question is now empirically settled (2h12m healthy serve, 0 stalls, 4-way, gen_tps 34.3 — this is the smoke test I demanded, executed the expensive way), the fenced-code root cause is forensically credible and community-corroborated, and the $0 offline recovery replay is exactly the right systems practice. But the load-bearing cadence claim (ratio ≈ 1.1x) is measured in a zero-action regime and is not a valid ρ_action estimate, the in-run liveness gate whose absence just cost a full window is *still* not in the v4 spec, and the quota-accounting and budget-table asks remain untouched for another round.

## Objections

**[Prior FATAL 1 — quota is not free] PARTIALLY-RESOLVED.** The SKU/fit half is now settled the empirical way: the canary mounted 48/48 shards, passed GPU asserts, and served the 72B-VL for 2h12m — the envelope question ("FITS") is closed and I withdraw that half. But the quota-exemption half has again received zero evidence: Q4 (is a boristown fork submission "filler, no window cost"?) is carried forward verbatim as an open question rather than answered from the submission ledger, and the exploration-draw accounting (Q5) still rides on the unevidenced footnote. One `kaggle competitions submissions` pull cross-referenced against the window ledger settles this; it has now been free and undone for three rounds.

**[Prior FATAL 2 — A17 envelope] PARTIALLY-RESOLVED, and the new cadence datapoint is invalid as offered.** The physical envelope (fit, serve health, throughput ceiling) is cleared — genuine progress. But the "436 turns vs 480 actions ≈ 1.1x, far inside 3.5x" figure cannot stand in for ρ_action: it was measured in a run where **zero actions executed and all four games gave up**, so the environment never advanced — no new frames were encoded, context growth followed a degenerate static-frame pattern, and give-up loops are plausibly cheaper per turn than real play with vision-token frame updates. The bias direction is again toward false GO. Additionally, my prior concurrency confound (27B numerator at 25-game vs 72B denominator at 4-game) is unaddressed. Answer to open question 3's rider: **turn≈action does NOT suffice** — v4 must recompute cadence from a run with `step_executed=True` turns, and the 27B matched-concurrency control leg (or a pre-registered correction) is still owed before any GO is interpreted.

**[Prior MAJOR — watchdog kills the bench] RESOLVED.** No regression.

**[Prior FATAL — headline numbers contradict the briefing] RESOLVED.** The ground-truth file now contains the reconciliation I demanded: the 0.43/1.56 figures were a stale hardcoded May-era template in `panel_round.py`, root-caused and fixed 2026-07-24, with the 0.43→0.82–1.33 lineage explained. This brief's numbers (1.33 best, KOJIMA 1.86, gold ≈1.49, n=12 ledger) are consistent with the canonical ledger, including the new 0.84 draw. Closed.

**[Prior MAJOR — RC4/R5 pricing contradiction] RESOLVED.** No regression.

**[Prior MINOR — no tail model] PARTIALLY-RESOLVED.** A fit family now exists ("t ≈ −0.9 under the declared t-predictive") — that is the statement I asked for. But it lives in an amendment that is still DRAFT because R20 never ran, so the brief is using unratified machinery to declare the 0.84 draw in-band. Not load-bearing (0.84 is visibly inside the empirical band), but ratify before the next boundary case.

**[Prior MAJOR — single-point hardware claim] RESOLVED in substance.** The canary self-certified the rail exactly as predicted; the pre-registration ordering was violated but the question is now empirically moot. I still want the device line from `a17_canary_v3/analysis.md` quoted in the next brief (one line), but I no longer block on it.

**[Prior MINOR — contingency line] UNRESOLVED, now with a concrete arithmetic anomaly.** Still no budget table, third round. New and specific: the brief says vLLM "served healthily for 2h12m" — 2h12m is exactly 7920s, the full declared window. If serving alone consumed the whole window, where did the 43GB download/load/warmup go — outside the window, or is "2h12m" the total kernel wall-clock with serving a subset? This is not pedantry: the ρ_action denominator and the v4 "~2.5 GPU-h" estimate both depend on which it is. Itemize setup vs. serve in the v4 pre-registration.

**[Prior MAJOR — hang-risk mitigation diagnostic not preventive] PARTIALLY-RESOLVED, and v3 is the proof of the residual cost.** The hang itself didn't occur (2h12m, 0 stalls at 4-way — the transferability worry is empirically retired for this config). But note what DID happen: a run with **zero completed actions for the entire window** proceeded silently to completion. The in-run liveness gate I specified ("no completed action in N minutes → loud-fail/restart-with-fallback") would have flagged this at minute ~15; instead diagnosis was post-hoc from logs and the window burned. v4 as specified still has no such gate.

**[MAJOR — new] v4 must carry an in-game zero-action assert and an in-window fallback, or a single residual format defect burns a fourth window.** Concretely: after the first K analysis turns (K≈5–10), assert `step_executed ≥ 1`; on failure, emit a loud banner and either (a) restart the serving loop with layer (iii) few-shot exemplars enabled as fallback, or (b) fail fast so the window's forensic value is captured in minutes not hours. The recovery adapter's 99.5% was validated on traffic from a model that never saw an executed action; once actions execute and frames update, the output distribution shifts and the recovery rate is not guaranteed to transfer — the gate is the insurance for exactly that gap. This is cheap (harness-side, $0) and should be a condition of v4 authorization, which I otherwise support in the author's (i)+(iv)+(v) composition — the isolation argument for deferring xgrammar to v5 is sound engineering (one behavioral change per push), and (iv)/(v) are correct $0 confounder kills, with the caveat that if (iv) changes observed output format, the adapter's replay validation must be re-run against non-streaming transcripts before interpreting hits.

**[MINOR — new] The fingerprint-store exclusion of A17 zero-action runs is a classification choice that should be pre-registered, not asserted.** "The A17 canary ERRORs/zero-action runs are eval kernels, tracked in the lane's own forensic chain" conveniently keeps `class:COMPLETE:0.00` at n=3 (last 06-10) when the family's natural reading now includes v3. Fine to keep lanes separate, but write the rule down before the next incident, or the weekly table's "no NEW incidents" headline is doing quiet work.

## Questions for the authors (numbered)

1. Does the 7920s scored window include the 43GB weight download/load/warmup, or was "2h12m serving" measured after setup inside a longer kernel wall-clock? Give the kernel's total runtime and the setup/serve split from the v3 log.
2. What was the wall-clock distribution of the 436 analysis turns — uniform across 2h12m, or front-loaded before give-up? What triggered `gave_up` (turn cap, time cap, harness patience)?
3. For v4: will you add the zero-action assert (K turns, loud-fail or fallback-to-(iii))? If not, state the expected cost of a fourth silent 0.00.
4. When will the 27B 4-game matched-concurrency control leg run (bench rail acceptable), and what is the pre-registered ρ_action GO threshold from C3 in this brief's units?
5. Quota ledger: produce the submissions-API cross-reference that classifies fork pushes as filler or arm. Third ask.
6. Was the served chat template already dumped in v3's forensics (layer (v) is listed as a v4 addition) — i.e., can the template-strip hypothesis (#1093) be excluded from recorded v3 artifacts today at $0, before the v4 push?

## What I cannot judge

The C3 capability-gate semantics and whether "no capability claim either way" is the correct epistemic posture for v3 (methodology reviewer); the RL/planning value of the depth-event thesis and whether A17 remains the right lane given gold at +0.16 (rl-planning reviewer); the governance question of ratifying the 07-24 amendment and the /1→/2 pin deviation (I note only that shard-count+size identity without a hash is weak provenance, but artifact identity policy is not my chair); the boristown adoption-timing tradeoff beyond its quota-accounting component.

## Verdict: MAJOR-REVISION

## Score: 6/10

The trajectory is right — v3 converted a burned window into a closed envelope question and a credible root cause, and the offline replay discipline is exemplary. Authorization of v4 in the (i)+(iv)+(v) composition is warranted **conditional on**: the zero-action in-game assert, the setup/serve window itemization, a valid (action-executing, concurrency-controlled) ρ_action plan, and — for the third time — the quota-classification evidence and budget table.