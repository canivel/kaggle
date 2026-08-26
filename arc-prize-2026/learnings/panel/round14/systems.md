## Summary (2 sentences)
The revision is genuinely responsive: it retracts the "grinder cracking = wall-closer" claim honestly, unbundles (a)/(f), names the v4 model swap (Qwen3.6-72B W4A16 on the 96 GB rail) as the sole registered wall-closer, and supplies counting-bound arithmetic per component. However, my core demand — a compute-envelope feasibility calc for the wall-closer — is deferred to an Aug 1 "scoping" milestone rather than supplied, and the v4 line as sketched benches throughput on hardware (RTX PRO 6000) that is not the scored environment, which is exactly the class of instrument/regime mismatch this campaign keeps re-committing.

## Objections

**Prior-objection resolution audit (required before new comments):**

1. **[MAJOR→FATAL-adjacent] Wall-closer compute envelope — PARTIALLY-RESOLVED.** The revision names the model (72B-tier 4-bit, ~40 GB weights, KV headroom on 96 GB), acknowledges 2.5–3× decode slowdown, and adds a throughput guard to the v4 gate. But the actual envelope numbers I demanded — tokens/sec on the *scored* hardware, actions/hour, actions/game under the scored wall-clock, GPU-h/draw, draws affordable before Nov 2 — are still absent; §5 defers them to Aug 1. Naming the mechanism is progress; the arithmetic remains an IOU.

2. **[MAJOR] Build-rail/LB regime mismatch for budget-conditional mechanisms — PARTIALLY-RESOLVED.** The A10 canary-before-seal design, "trigger-can't-fire → LB-accumulation-only, never a build-rail kill," and banking's re-entry only via a trigger-firing bench directly implement the structural fix. But the specific retrospective data I asked for — trigger-fire counts instrumented on the LB draws already run — is not published in the visible text (possibly in truncated Part 2; see objection 8). Publish it or state it doesn't exist.

3. **[MAJOR] (a)+(f) bundling — PARTIALLY-RESOLVED (mostly resolved).** (f) now ships first as standalone unflagged hygiene with a smoke screen, and (a) gates alone in W2 — exactly my fix. However, the pre-registered fire-rate cap and token-overhead budget for the sentinel (expected fires/game, tokens/fire, max % of context) is still missing; the mechanism prong ("budget deaths halved") measures efficacy, not context tax. Add the tripwire before W2 opens.

4. **[MAJOR] Token/wall-clock cost accounting — PARTIALLY-RESOLVED, quantification still missing.** (b)'s ~120 tok/action cost is now acknowledged and it gates last under a non-inferiority guard — good ordering. But no candidate has the numbers I asked for: Δtokens/action, Δlatency/action, projected Δactions per scored window. (a)/(c)/(d) are implicitly treated as free, yet FACT/RESULT records also occupy context; and the non-inferiority guard (Δlc ≤ −0.10) sits *below* the rail's ~0.2 3-seed MDE, so a throughput-induced −0.1 drag is undetectable by the very tripwire meant to catch it. This remains a one-afternoon measurement on existing logs; do it.

5. **[MINOR] Ledger refutation magnitude claim — RESOLVED.** The revision cites only the deterministic mechanistic finding (1552 digests / 0 escalations); no unpowered magnitude claim appears.

6. **[MINOR] Q3/Q5 draw economics — UNVERIFIABLE.** This lived in the daily brief, which is truncated in my copy. Cannot score resolution.

**New objections:**

7. **[FATAL unless answered before v4 registration] Scored-hardware parity: the wall-closer is benched on a GPU the scored harness does not have.** The v4 throughput bench runs on the RTX PRO 6000 (96 GB, high-bandwidth single card). The Kaggle scored environment is not that card — if it is L4×4, a 72B-4bit model runs tensor-parallel over PCIe at a large additional decode penalty beyond the quoted 2.5–3×; if it is T4×2 (32 GB) the weights do not fit at all. The proposal is entirely silent on what GPU the scored harness provides. Required before Aug 1: (i) the scored-environment GPU spec, (ii) a vLLM decode-throughput measurement on that hardware or a faithful proxy (a Kaggle-notebook probe run costs nothing), (iii) the resulting actions/game projection under the scored wall-clock. Without this, war-v4 can pass its rail gate and be dead on arrival at scoring — the exact failure mode my R13 objection was about, relocated one instrument to the left.

8. **[MAJOR] v4 gate does not fit the 30 GPU-h/wk quota as scheduled, and the throughput guard's fallback is undefined.** Arithmetic the proposal owes: a full-budget compound gate is 3 ON seeds at ~8 h wall-clock ≈ 24 GPU-h, before the throughput bench, canary, control seeds (if not reused), or any 17/17-sha re-certification failure — one bad seed burns the week, and the gate realistically spans 2–3 weeks against a Nov 2 deadline with no registered fallback wall-closer if v4 FAILs. Separately, the guard "total actions within 10% of 27B baseline" is near-certain to fail given 2.5–3× slower decode under fixed wall-clock, which routes every outcome to the fallback clause "Δlc must beat the throughput-adjusted null" — a quantity nowhere defined. Define the throughput-adjusted null now (formula, inputs, who computes it), and publish the week-by-week GPU-h ledger for the v4 gate.

9. **[MAJOR] The v3 gates measure a different operating point than the Δ claims.** §2's counting bounds are computed at full budgets, but §3's bench compresses per-game action caps to ~40% of observed median specifically to force triggers to fire. Under the (base/actions)² scorer, compression changes clear rates, clear costs, and death frequencies nonlinearly — a PASS/FAIL at 40% caps is evidence about a regime the scored run mostly doesn't occupy, the mirror image of the banking regime-mismatch I flagged last round. Fix: for any window that PASSes at 40%, run one full-budget confirmation seed before the flag enters the scored config; report both operating points.

10. **[MINOR] Copy truncation.** Part 2 (daily brief) is marked [TRUNCATED] mid-sentence; the shown text is well short of the declared 30,355 chars even though the end-sentinel line appears. My prior objections 2 and 6 may be addressed in the missing text; I reviewed only Part 1 and the resolution states above are conditional on that.

## Questions for the authors
1. What GPU(s) does the Kaggle scored harness actually provide, and have you confirmed 40 GB of 72B-4bit weights + KV fit on it?
2. What is the measured Qwen3.6-27B decode throughput (tok/s) and actions/hour on the scored harness today, so the 72B 2.5–3× penalty can be converted into a concrete actions/game number?
3. What is the "throughput-adjusted null" in the v4 gate, exactly — formula and inputs?
4. What are Δtokens/action for (a), (c), (d) FACT/RESULT records (not just (b)'s 120), and do they displace or add to raw observations in context?
5. Do LB trigger-fire counts for banking/recovery exist in the already-run LB draw logs, per my R13 request — yes or no?
6. If war-v4 FAILs its gate in mid-August, what is the registered wall-closer for the remaining ~11 weeks?

## What I cannot judge
The transcript-forensics failure taxonomy (CONCEPT/MEMORY/PERCEPTION), the NOT-distillable cognitive-capability claims, the exactness of the sign-flip test construction and α choice, and everything in the truncated Part 2 (variance flip, sweeps, Q-A..Q-E). I take the scorer validation ("0e+00 vs Tufa's 500 runs") on faith.

## Verdict: MAJOR-REVISION

## Score: 6/10