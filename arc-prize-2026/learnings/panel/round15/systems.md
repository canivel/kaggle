## Summary (2 sentences)
The revision delivers real structure on my two central systems complaints — A17 makes the 72B tokens/s bench a blocking pre-Aug-1 screen with a sealed go/no-go and finally defines the throughput-adjusted null, and A15 fixes the compressed-bench operating-point problem — but the actual envelope numbers remain IOUs, and the document now contradicts itself on which GPU the wall-closer bench runs on (Part A: RTX PRO 6000 96 GB single card; A17: "free Kaggle GPU build rail"). The A17 GO condition as written in Part C is a conjunction whose throughput prong (≥90% of 27B action count) is arithmetically near-impossible under the team's own 2.5–3× decode slowdown, and the recalibrated gate's binding runs (cumulative look + A15 full-budget replicate) appear in no timeline and no GPU-h ledger.

## Objections

**Prior-objection resolution audit:**

1. **[MAJOR] Wall-closer compute envelope — PARTIALLY-RESOLVED.** A17 converts the IOU into a blocking, sealed pre-registration: measured tokens/s, N₇₂B computed before registration, NO-GO consequence escalated to the panel in July. That is the right structure. But the numbers still do not exist, and the bench hardware is now self-contradictory (see new objection 11) — the envelope arithmetic cannot be scored until the rail is named and the probe run.

2. **[MAJOR] Build-rail/LB regime mismatch retrospective data — PARTIALLY-RESOLVED.** A15 commits to publishing per-component trigger frequencies at full budget from the war_eval seeds 1–3 transcripts (which are Kaggle pulls, i.e., scored-regime data — this is the retrospective instrumentation I asked for). The commitment is sealed; the table is not yet published. Publish it alongside the W1 canary, not after.

3. **[MAJOR] Sentinel fire-rate cap / token-overhead budget — UNRESOLVED, urgency increased.** (d) was killed by A18, so W1 becomes (a)'s window effective immediately (Jul 20), and there is still no pre-registered fires/game cap, tokens/fire estimate, or max-%-of-context tripwire anywhere in Parts A–E. The mechanism prong ("budget deaths halved") still measures efficacy, not context tax. This must be added to W1's prong before the window opens — which is tomorrow.

4. **[MAJOR] Token/wall-clock cost accounting — UNRESOLVED (partially mooted).** (d)'s death removes the FACT-record context tax, which helps. But Δtokens/action, Δlatency/action, and projected Δactions per scored window for (a) and (b) remain unmeasured despite being a one-afternoon job on existing logs, and the non-inferiority guard (Δlc ≤ −0.10) is now *enshrined in A14 §3* while still sitting below the rail's ~0.2 3-seed MDE — a throughput-induced −0.1 drag remains undetectable by the tripwire built to catch it. A14 should seal with this insensitivity acknowledged and the guard either widened in seeds or backed by a direct latency counter.

5. **[MINOR] Ledger magnitude claim — RESOLVED** (A11 label maintained; no unpowered point estimates cited).

6. **[MINOR] Q3/Q5 draw economics — RESOLVED.** Now visible in Parts D/E: filler E[max@~106] ≈ 1.39, window break-even +0.06–0.12, pooled n=11 σ̂ validation. Coherent and priced.

7. **[was FATAL] Scored-hardware parity — PARTIALLY-RESOLVED.** A17 relocates the bench to "the free Kaggle GPU build rail" — the faithful proxy I demanded — and defines the binding budget as scored-rail wall-clock. But the scored-environment GPU spec is *still never named*, and Part A simultaneously asserts the 72B bench runs on the RTX PRO 6000. Downgraded from FATAL only because A17's structure, correctly executed on the actual Kaggle SKU, answers it; new objection 11 carries the remainder.

8. **[MAJOR] v4 quota fit + undefined null — PARTIALLY-RESOLVED.** The throughput-adjusted null is now defined with a formula (null_adj = 27B levels completed by action N₇₂B; 72B must beat Σ null_adj) — that closes half the objection. The week-by-week GPU-h ledger is still absent, and the guard's boolean structure is now *contradictory* between §5 and A17 (new objection 13).

9. **[MAJOR] 40%-cap operating point — RESOLVED.** A15 is exactly my fix: provisional inclusion only, one full-budget certified confirmation replicate before score credit, full-budget trigger frequencies published, compressed-regime relabeling of §2 ceilings.

10. **[MINOR] Copy truncation — RESOLVED.** Single untruncated delivery, all five END-OF-PART lines present, END OF PROPOSAL present.

**New objections:**

11. **[MAJOR] Bench-hardware contradiction: Part A and A17 name different GPUs for the same bench.** Part A §1 says the 72B line is "evaluated inside the free 30 GPU-h/wk" on "the RTX PRO 6000 (96 GB) build rail"; A17 says the screen runs on "the free Kaggle GPU build rail." A 96 GB single high-bandwidth workstation card is not Kaggle free-quota hardware; if the Kaggle free tier is T4×2 (32 GB) the 40 GB AWQ weights do not load at all, and if it is L4×4 the model runs tensor-parallel with an interconnect decode penalty *beyond* the quoted 2.5–3×, invalidating any tokens/s measured on the PRO 6000. Required before A17 seals: (i) name the accelerator SKU/count/memory on both the bench rail and the scored harness, (ii) run the tokens/s probe on that exact SKU, (iii) recompute N₇₂B from it.

12. **[MAJOR] The recalibrated gate's binding runs are unscheduled and unbudgeted.** A14's cumulative sealed look (3 ON seeds vs the W0 baseline) and A15's full-budget confirmation replicate appear nowhere in the §5 timeline, which is already packed through Jul 30 with W1–W4 plus the pre-Aug-1 A17 screen in the same quota week. Worse, the author recommends no W0 seed-2 (Part E Q3), leaving the cumulative look's control arm at n=1 against 3 ON replicates — a paired design without its pairs. Publish the ledger (runs × hours × calendar week vs 30 GPU-h/wk; W0 eval = 2h12m/seed at 27B is the anchor), schedule the cumulative look and confirmation replicate explicitly, and state the W0 control seed count the look will actually use.

13. **[MAJOR] A17's GO condition, read as written, auto-kills v4 regardless of capability.** A17 states "GO iff ≥2 levels beyond 27B AND measured throughput sustains ≥90% of the 27B action count" — but under the team's own 2.5–3× decode slowdown, a decode-bound action loop yields ~33–40% of the 27B action count, so the second conjunct fails by construction and the null formula ("closing the else-branch") has no branch to close. §5's version is disjunctive ("within 10%... *else* Δlc must beat the throughput-adjusted null"); A17's is conjunctive. Seal one boolean — presumably GO iff (≥2 levels AND ≥90% actions) OR (72B beats Σ null_adj, with a registered margin) — this round, since A17 is blocking and a mis-sealed conjunction hands the campaign a guaranteed NO-GO on its only wall-closer.

14. **[MINOR] The two proposed step-0 fixes carry unpriced action/wall-clock costs.** Resync-before-abort spends an extra observation per mismatched plan, and OCM-style pre-execution validation spends CPU wall-clock inside the scored window; under the (base/actions)² scorer and the dry-run's abort counts (100–250 step-0 aborts/run on some games), the resync tax is material, not free. Price both from the dry-run plan counts before R15 rules on Q2.

## Questions for the authors (numbered)
1. What GPU does the *scored* Kaggle harness provide (SKU, count, per-card memory), and is the free-rail bench kernel guaranteed the same SKU?
2. Which machine actually runs the A17 tokens/s bench — the RTX PRO 6000 or a Kaggle kernel — and if the former, what is the transfer function to the scored rail?
3. Does the 30 GPU-h/wk quota count session-hours or GPU-hours (×N for multi-GPU sessions)? The entire ledger changes by 4× depending on the answer.
4. Where in the calendar do the A14 cumulative look and the A15 full-budget confirmation replicate run, and what is their GPU-h cost?
5. How many W0-baseline control seeds will the cumulative look use if seed-2 is dropped per the Part E recommendation?
6. For (a): pre-registered expected fires/game, tokens/fire, and max % of context — numbers, before W1 opens Jul 20?
7. Where is the Δtokens/action and Δlatency/action measurement for (a) and (b) from existing war_eval logs (my standing one-afternoon ask, round 3 of asking)?
8. A17's GO boolean: conjunction or disjunction with the throughput-adjusted null? State it as a formula.

## What I cannot judge
The statistical calibration of the sign-flip test, α choice, and the P(pass) binomial sketch (methodology reviewer's lane); the cognitive-science validity of the CONCEPT/MEMORY/PERCEPTION taxonomy and NOT-distillable lists (llm-agents/prog-synthesis); the game-level transcript forensics underlying §4's Δclears; whether the A19 novel-family prong is the right falsifiable bet. My resolution audit of prior objections 2 and 6 is now unconditional, since this copy is complete.

## Verdict: MAJOR-REVISION

## Score: 6/10

The systems trajectory is genuinely improving — A15, the defined null, the EWMEVT log-volume analysis against the 10 MB cap, and the front-loaded-penalty pricing of mixed-tier routing all show the team now thinks in envelopes. But the campaign's sole registered wall-closer currently has a bench whose hardware is named two different ways, a GO condition that may be arithmetically unpassable as sealed, and a gate whose binding runs have no slot in the quota ledger. Objections 11 and 13 must be fixed *this round* (A17 is pre-Aug-1 blocking); objection 3 must be fixed before W1 opens tomorrow. On Part E Q1: I do not block the A14 seal on systems grounds, conditional on the guard-insensitivity acknowledgment (audit item 4) being written into the seal text.