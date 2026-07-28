## Summary (2 sentences)
The revision fixes the worst provenance failure (the ledger is now API-refreshed before any statistic is cited) and finally states the power fact that filler cannot reach gold, but the two highest-value statistical actions — the readiness-gate A/B and a defensible exploration-arm design — remain unscheduled open questions rather than pre-registered plans, and the weekday-gate operating characteristics I demanded were never published even as the restructure was used to fire v6 mid-week. Additionally, a new instance of the single-sample failure mode has appeared: the 72B route's kill/continue decision is being set up to be read off one seed-1 bench run with no variance estimate.

## Objections

**Prior-round resolution review:**

**[MAJOR] Ledger provenance gap — RESOLVED.** The 07-27 and 07-28 refreshes ran the stated live-API path, cross-checked against `runs/submission_log.jsonl`, appended the 07-26/07-27/07-28 draws to the canonical file, and recomputed n=14 / 0.9686 / 0.1384 numerically (I verified: the 14 listed draws give mean 0.9686, s ≈ 0.138). The brief now states "all numbers below come from that file." This is exactly the fix I asked for.

**[MAJOR] Power analysis / readiness-gate A/B — PARTIALLY-RESOLVED.** The brief now quotes P(single frozen draw ≥ 1.49) ≈ 2×10⁻⁴ and concedes "filler holds rank, it does not climb," which discharges the acknowledgment half. But the A/B itself — a 5/5 R21 directive and the only causal hypothesis for the 1.39/1.47 anchor tail — is *still unscheduled*, parked as open question 3 with the panel asked to rank it. There is also no draws-to-cutoff calculation for a hypothetical gated arm. Acknowledging the arithmetic without scheduling the experiment does not move the plan toward gold.

**[MAJOR] Exploration arm decided from n=1 — PARTIALLY-RESOLVED.** The disposition is now honestly marked OPEN, option (a) proposes n ≥ 4–5 with a sequential stop, and no new single-draw claim is made (the p ≈ 0.07 framing is correctly hedged). But the proposed stopping rule ("2 consecutive < 0.80 or mean of first 3 < 0.80") ships without the error rates I required — which are computable: under the frozen null, P(draw < 0.80) ≈ Φ(−1.22) ≈ 0.11, mean-of-3 < 0.80 has p ≈ 0.017, and the 2-consecutive rule over 5 draws fires falsely with probability ≈ 0.05, giving a combined false-kill rate of roughly 6–7%. Publish those numbers in the pre-registration; "Panel to pick one, with error rates" delegates my objection instead of answering it.

**[MAJOR] Sealed-gate operating characteristics — UNRESOLVED.** Nothing in this brief publishes (a) per-draw false-trigger probability, (b) family-wise weekly trigger rate, or (c) escalation rules for the weekday gates — and meanwhile the restructure was exercised in anger: v6 fired mid-week "under an explicit user order" with NC-4 and NC-5 explicitly undischarged. The restructure is now operating with exactly the uncontrolled error rates I warned about. This must be delivered before the next unreviewed weekday promotion, not after.

**[MINOR] Band/MK-CUSUM evidential weight — UNRESOLVED.** The brief again reports "all interior to band 0.82–1.33" and "MK/CUSUM no-trend verdict stands" without power caveats, CUSUM reference value, or control limits; the band endpoints remain the sample's own min–max. It additionally adds "s has tightened 3 refreshes in a row" as if it were a stability signal — sample s shrinking as interior draws accumulate is near-deterministic arithmetic, not evidence.

**[MINOR] Trigger-family multiplicity — UNRESOLVED.** No trigger-family specification appeared. This can be folded into the same one-page document as the gate operating characteristics.

**New objections:**

**[MAJOR] The 72B route's life-or-death decision is being staged on a k=1 bench run with no variance model.** v6 is a single full-window run at seed 1, yet open question 1 asks the panel to ratify a kill threshold Y such that ρ_action < Y kills the route. A threshold applied to one draw of a quantity with unknown between-run variance has unknown false-kill and false-continue rates — the identical structural error as the sentinel n=1 shelving, now on the build rail. Fix: either pre-register k ≥ 2–3 seeds before any kill/continue verdict, or state a variance bound for ρ_action (e.g., from the per-game N counts' dispersion, treating actions as approximately Poisson across the four games) and set Y with a stated one-sided error rate against that bound. The v5 slice already shows per-game counts of 0–2 — that heterogeneity is your first variance estimate; use it.

**[MINOR] In-sample z-scores.** The week's draws are z-scored against statistics that include those same draws (1.05 on 07-25 is inside the n=13 stats it is compared to). Use leave-one-out or prior-refresh statistics for each draw's z; at n=14 the correction is small but the discipline prevents the shrinking-s artifact from flattering interior-ness.

**[MINOR] The proposed A/B design wastes control draws.** With frozen n=14 already banked and an anchor-implied effect of Δ ≈ 0.5 (d ≈ 3.6), a gated-arm-only design of n=3 tested against the existing frozen ledger via a two-sample t (or predictive interval) has power ≈ 1 at that effect size and near-adequate power even if the gate explains only half the anchor gap (d ≈ 1.8, power ≈ 0.7 at n=4, one-sided α=0.05). Pre-register gated-only draws; do not burn slots re-sampling the control.

## Questions for the authors (numbered)
1. Will you commit, in this round, a calendar date and pre-registration for the readiness-gate A/B (n, test, α, anchor-implied effect size), rather than leaving it as a panel-ranking question for a third consecutive round?
2. For question 2, my ranking within my remit: the readiness-gate A/B strictly dominates sentinel draw #2 on information value (5/5 directive, single-variable causal test, d ≈ 3.6 testable at n=3–4 against the banked control). Do you accept option (b) with the sentinel un-shelve rule pre-registered but queued *behind* the A/B?
3. For ρ_action: what is your between-run variance estimate (or bound), and will you run k ≥ 2 seeds before any kill decision? If k=1 is forced by GPU budget, state the widened threshold Y that compensates.
4. When will the one-page trigger-family specification (per-trigger and 6-day family-wise false-alarm rates under the frozen null, plus escalation map) be delivered — and will weekday autonomous promotions be suspended until it exists?
5. Which prior-refresh statistics will future briefs use for out-of-sample z-scores of new draws?

## What I cannot judge
The engineering validity of the A17 canary evidence (vLLM serve grafts, hermes parser, fenced-recovery adapter, dataset-mount mechanics), the Kaggle resource/budget accounting (GPU-hours, push slots, zero-budget-rule compliance), the doctrinal/eval-rail content of the sentinel disposition memo, and the EWM Stage-1 latent-state audit substance (question 4) — I can only insist that whatever measurement it produces not be interpreted at k=1.

## Verdict: MAJOR-REVISION

## Score: 5/10