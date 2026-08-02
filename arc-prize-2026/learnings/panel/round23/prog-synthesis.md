## Summary (2 sentences)

The brief is well-instrumented (API-verified ledger, pre-registered watch-rule that actually fired, discharged entry gates), but the centerpiece — the "stationarity guard" for the boristown A/B — is statistically incoherent: the corrected promote bar 1.1701 is derived from an external variance (yw8837's σ≈0.24) that is rejected at p≈0.007 by our own 15-draw sealed control, the stated direction-of-bias argument is backwards for the step-down hypothesis, and the mean-over-4-draws test statistic does not match the causal mechanism of the intervention (a rare-stall readiness rescue). Ratification of option (a) as written would either burn 4 gated draws on a pre-determined-to-fail test (if the step-down is real) or inflate the bar on evidence our own data contradicts; only the interleaved-control design (b) is defensible, and it needs specification before sealing.

## Objections

**[MAJOR] The corrected bar 1.1701 rests on σ=0.24, which our own control data reject.** If the true per-draw σ were 0.24, the sealed control's observed s=0.1343 over n=15 would satisfy χ²₁₄ = 14·(0.1343/0.24)² ≈ 4.38, i.e. P(s ≤ 0.1343 | σ=0.24) ≈ 0.007. The σ≈0.24 "duck-family regime" is an external population figure from a different team's fork and is inconsistent with 15 API-verified draws of our own process at p<0.01. Worse, σ=0.24 is used twice in opposite rhetorical directions: to downgrade the 0.65/0.68 pair to "ordinary" (18.5%) and simultaneously to raise the promote bar. Commit to one variance model with a stated selection rule (e.g., posterior over σ given all 19 draws), or drop the external figure entirely.

**[MAJOR] The direction-of-bias argument for the guard is wrong, making option (a) incoherent with its own motivating hypothesis.** A step-down to ~0.665 (the brief's own change-point level) biases a sealed-control test toward spurious **non**-promotion, not promotion: under mean 0.665, σ=0.24, the 4-draw mean must clear 1.1701, a +4.2σ event (P≈10⁻⁵) — the A/B becomes a guaranteed fail that burns 4 scored draws and yields zero information. Under-dispersion relative to the seal likewise reduces bar-exceedance probability. Only over-dispersion (σ true > s sealed) biases toward spurious promotion, and that is precisely the σ=0.24 assumption objection #1 shows is untenable. R23 should be presented options (b)/(c) only; (a) as derived should be struck.

**[MAJOR] Mechanism/metric mismatch: a mean-shift test over n=4 cannot detect what the boristown diff plausibly does.** The only functional diff is a vLLM readiness gate; its causal mechanism is rescuing occasional slow-start/stall runs — a rare-event, left-tail-truncation effect, not a mean shift. The gate-eval's own evidence confirms this: `vllm_ready_latency_s=0.0` means the gate fired vacuously and changed nothing in the only direct test. Expected effect on a 4-draw mean is near zero unless a stall happens to occur in-window; the promote bar (1.0970 or 1.1701) then measures luck, not the intervention. Specify a mechanism-matched statistic (stall/failure incidence, min-draw, or left-tail mass over a larger K) or pre-register that a null mean result is uninformative about the gate.

**[MAJOR] The harm-pause (<0.80 on gated draws) is miscalibrated under the very regimes motivating the guard — "unaffected" is false.** Under the sealed control N(0.9727, 0.1343), P(gated draw < 0.80 | null) = 9.9%, so P(≥1 spurious harm-pause in 4 draws) ≈ 34%; under σ=0.24 it is ≈ 66%; under the step-down level 0.665 it fires almost surely on a null draw. A harm rule that trips at coin-flip-or-better rates under the null will confound the A/B and cannot distinguish harm from baseline drift. Re-derive it as a paired/relative criterion (gated minus contemporaneous filler) consistent with design (b).

**[MINOR] Non-harm screen results are laundered into directional support.** Δlc +0.152, sd 0.537, n=16 gives t ≈ 1.13, p ≈ 0.28 (seed-1 similar); this is fine as a harm *screen* but "both seeds PASS with positive direction" is noise presented as signal. State the screen's job (rule out large harm) and its power, nothing more.

**[MINOR] Change-point statistics and option (c) power are under-specified.** Welch |t|=8.64 with n₂=2 is a degenerate statistic (variance of a 2-point segment); the permutation p=0.0032 is only valid if the permutation null was computed on the *max-over-splits* scan statistic — state this explicitly in `stationarity_2026-08-02.md`. Option (c)'s two extra fillers move the post-break segment from n=2 to n=4, which has almost no power to separate step-down from tail — quantify before offering it as a serious alternative.

**[MINOR] The Living-Harness A22 amendment has no falsifiable acceptance metric.** "Reframes payload as graph-state" adopts a representation from arXiv:2607.26598 with no stated criterion for when the graph-state payload beats the plan-blob payload. Pre-register a measurable comparison (e.g., retained-index hit-rate or recovery-success delta on the smoked 41-case set) before amending the sealed prereg.

## Questions for the authors (numbered)

1. How exactly was 1.0970 derived, and does the 1.1701 recomputation drop the control-mean uncertainty term (s·√(1/4+1/15) → σ·√(1/4))? If so, the two bars are not even the same test family — reconcile.
2. Was the permutation p=0.0032 computed against the null distribution of the *maximum* Welch |t| over all split points, or against a single fixed split chosen post hoc?
3. What is the pre-registered test statistic, pairing/blocking scheme, and alternation order for interleaved design (b)? A paired t on 4 pairs has essentially no power — is the intent estimation-with-guard rather than hypothesis testing, and if so what is the promote rule?
4. Given the gate fired with latency 0.0 in both seed evals, what evidence exists that vLLM slow-starts occur at all in the Kaggle scoring environment (e.g., stall incidence in the host's 500-submission error post, or in our own 19-draw logs)?
5. Under the promoted regime you hope for, what is P(single draw ≥ 1.54) and the expected number of remaining scored draws before Nov 2 — i.e., does even a successful A/B make gold reachable without the compaction lane?
6. Does the sealed prereg contain an escape clause permitting the bar change from 1.0970 → 1.1701, or does R23 ratifying option (a) constitute an unsealed-amendment precedent?

## What I cannot judge

- Kaggle platform mechanics (submission quota accounting, scoring-window timing, whether "frozen draw" scheduling at 00:07Z is contractually reliable).
- The provenance and trustworthiness of yw8837's published σ≈0.24 figure and the boristown fork-diff audit — I take `fork_diff_boristown_2026-07-24.md` at its word.
- GPU/infra specifics (RTX PRO 6000 NC-12 marker semantics, vLLM deployment details beyond the cited issue).
- Team-operational items (agenda item 5, session/monitor process hygiene).

## Verdict: MAJOR-REVISION

## Score: 5/10