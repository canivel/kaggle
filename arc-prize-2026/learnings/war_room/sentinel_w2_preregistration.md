# Sentinel W2 (seed 2) pre-registration — filed 2026-07-23, BEFORE any W2 run

Status: DRAFT pending R17 ratification (filed before the R17 panel returned;
this file is immutable once the push happens — amendments require a new dated
section, never edits).

## What W2 is

A $0 free Kaggle build of the identical sentinel-v2 configuration that ran as
W1 (`canivel/arc3-duck-sentinel-eval`, SENTINEL_BUDGET=150, unit=game-envelope,
v2 patch byte-identical — no code changes of any kind), differing only in the
harness seed (seed 2). It is a **confirmatory null**, not a lift experiment.

## Pre-registered expectations (sealed before observation)

1. **Score:** arm mean falls INSIDE the certified baseline seed spread
   [1.16, 1.73] or below it; no lift claim will be made at any value. W1's
   0.855 was carried by three high-variance non-target games; W2 tests whether
   that was seed noise.
2. **Mechanism:** sidecars + stdout `SENTINEL v=2` events consistent; ≤3
   events/game; cumulative envelope keying (no re-arm across attempts);
   missing sidecar ⇔ game ended with zero crossings.
3. **Behavior:** warnings again change nothing (≤3/22 fired games advance a
   level after first warning). This is the expected FAILURE of the behavioral
   channel, being confirmed so the line can be closed cleanly.
4. **Envelope (condition 4):** tokens/game within 63k ±15% for ≥20/25 games.

## Decision rule (sealed)

- All four expectations met → sentinel line CLOSED as "certified observable,
  no lift channel"; no W3; the v2 patch remains available as a passive
  telemetry component for future arms (it costs ~0 tokens when not crossing).
- Score arm mean ABOVE 1.73 (surprisingly positive) → W3 permitted as a
  replication, question reopens.
- Mechanism or envelope failure → bug investigation before any further
  sentinel claim; the W1 mechanism seal is then re-examined at the next panel.

## Cost

1 kernel push (of 2/day), ~2.2h free Kaggle GPU build time, $0.

---

## AMENDMENT 2026-07-23 — calibrated decision rule (supersedes the §"Decision rule (sealed)" band above)

Filed 2026-07-23 in response to R17 methodology N2 (`learnings/panel/round17/methodology.md`
line 29) and the R-W2 ruling in `learnings/panel/round17/_directives.md` (Part 2).
This is an **append**, not an edit: the original draft (the [1.16, 1.73] min–max band,
"positive"-undefined rule) is retained above and is hereby **VOID / SUPERSEDED** — it is
rejected as filed for three stated reasons (methodology N2): (i) [1.16, 1.73] is the min–max
of the n=3 prior draws, so a true-null draw lands inside with probability (n−1)/(n+1) = 0.5
(a coin-flip acceptance region, no power); (ii) "positive" carried no number; (iii) the band
was chosen *after* observing W1's 0.855 (post-hoc). None of the four amendments below required
any compute — they are all writing over already-certified seed data.

### A. The legal control (fixes methodology N1)

W1 and W2 are compared **against the legal single-seed control `w0_s1` = 1.731**
(`runs/kernel_pulls/w0_eval_s1/`, benchmark label `duck-harness-kaggle-continuation-v1`),
**NOT** against the war_eval 3-seed warpack baseline (1.454), which §5.4 ruled ILLEGAL as a
control (config-diff = warpack exceeds the sealed {(f)} envelope). The warpack comparison is
published as **diagnostic only**. Once the two fallback W0 seeds land the control widens to
{w0_s1, w0_s2, w0_s3} (n=3) and the SE below is recomputed with the pooled control mean.

### B. W1 z under the frozen σ̂ (published, not a new measurement)

Frozen noise model σ̂ = 0.189 (sealed in `runs/sealed/r17_thresholds.json` → guard, z=1.834).
W1 sentinel score = **0.855** (`runs/sentinel_eval_analysis/summary.json`, prong1).

- **vs the legal control `w0_s1` (single-seed vs single-seed):** SE = σ̂·√2 = **0.267**;
  Δ = 0.855 − 1.731 = **−0.876**; **z = −3.28**. A ~0.05% event under the frozen null.
- **vs the war 3-seed mean (diagnostic only):** SE = σ̂·√(1 + 1/3) = **0.218**;
  Δ = 0.855 − 1.454 = −0.599; **z = −2.75** (this reproduces methodology N2's own −2.7 figure,
  confirming the arithmetic; it is the *diagnostic* anchor and is NOT the registered statistic).

This resolves the internal inconsistency methodology N2 flags: we do **not** get to dismiss the
−0.60 draw by appealing to a σ ≈ 0.29 seed spread while calibrating every guard at σ̂ = 0.189.
Under the frozen σ̂ the W1 deficit is a genuine, large-magnitude signal, not noise.

### C. The registered numeric W2 rule (a z-band on the score prong, fixes methodology N2)

W2 is one new seed (seed 2). Define Δ_W2 = score_W2 − w0_s1, SE = σ̂·√2 = 0.267 (updated to the
pooled-control SE once {w0_s1..s3} lands). **Sealed decision rule, before W2 runs:**

- **REPLICATED-DEFICIT ⇒ KILL (a):** if Δ_W2 ≤ **−0.28** (the sealed §8R16 guard boundary,
  `r17_thresholds.json` → guard.boundary_lc_per_game; as a z on the score delta this is
  z = −0.28/0.267 = **−1.05**) **AND** W1 already sits at or beyond that boundary (it does:
  Δ_W1 = −0.876 ≪ −0.28), then **(a) is killed OFF via the existing §8 machinery** — NOT a new
  ad-hoc adjudication. Two seeds both at/beyond −0.28 under the true null has probability
  (Φ(−1.05))² ≈ 0.147² ≈ **0.022** (~2%), so this is a calibrated ~α=0.02 kill, not a coin flip.
- **NULL / no-lift (the expected outcome):** −0.28 < Δ_W2 < +0.28 (|z| < 1.05) → sentinel line
  CLOSED as "certified observable, no lift channel"; v2 patch retained as passive telemetry; no W3.
- **SURPRISE-POSITIVE:** Δ_W2 ≥ +0.28 (z ≥ +1.05) → W3 permitted as a replication; question reopens.
- **False-alarm / miss (stated):** for the KILL arm, single-seed false-alarm P(Δ_W2 ≤ −0.28 | true
  null) = Φ(−1.05) = **0.147**; the joint two-seed KILL false-alarm is **0.022**. Miss rate at the
  guard boundary (true effect = −0.28): P(a single seed lands ≤ −0.28 | true = −0.28) = **0.50** by
  symmetry, so the two-seed KILL has power 0.50 at the boundary and higher below it.

### D. The registered PRIMARY statistic is BEHAVIORAL, not the score mean (fixes llm-agents O1)

The uninformative score mean is demoted to a secondary/diagnostic readout. The **registered W2
statistic** is the **post-first-warning strategy-switch rate**: of the games in which the sentinel
fires ≥1 warning, the fraction that advance a level after the first warning. This is computable
from existing W1 transcripts at $0 and W1 already establishes the null expectation:
**1 / 22 fired games advanced post-warning (only tn36); +618 total actions vs the 3-seed baseline;
wa30 ground 560 actions through all three warnings** (`runs/sentinel_eval_analysis/summary.json`,
prong4). **Registered W2 behavioral null:** ≤ 2/22 fired games advance after first warning (i.e.
the "fires-doesn't-pay" label replicates). If W2 shows a materially higher switch rate (≥ 6/22,
> 25%) the behavioral channel is NOT dead and the line reopens for a behavioral (not score) reason.

### E. Consequence for the A14 binding look (couples to the (a)-guard-default seal)

This rule is the n=3-producing path referenced by the sealed (a)-guard-default sentence
(`r17_thresholds.json` → sentinel_guard_default, added 2026-07-23; and
`learnings/panel/r17_discharge_memo.md`). W2 gives the guard its second ON seed; if both W1 and W2
sit at/beyond −0.28 the guard fires KILL and (a) defaults OFF at the look. If the guard is still
unevaluable at look time and the (a)-arm 2-seed mean < baseline − 0.28, (a) defaults OFF and the
branch is re-labeled; otherwise the look postpones until n=3 exists.

W2 may push **today** once this amendment is committed — it is all writing, no compute.
