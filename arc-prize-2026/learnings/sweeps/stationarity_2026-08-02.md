# Stationarity re-check — 2026-08-02 (watch-rule FIRED: second sub-0.80 control filler)

**Analyst note (measurement discipline):** this is a distribution / stationarity analysis
of the CONTROL arm (byte-identical frozen fork). No capability claim is made or licensed
from any draw. The purpose is a single decision: do the sealed n=15 control parameters
(mean 0.9727, s 0.1343) — which anchor the boristown A/B promote threshold — still describe
the process, or has the control drifted, biasing the A/B test?

**Trigger (pre-registered, `learnings/daily_brief_2026-08-01.md` §1):** *"a SECOND sub-0.80
control filler soon → stationarity re-check (MK/CUSUM) before further gated draws."*
FIRED: overnight draw **2026-08-02 00:07:10Z scored 0.68** (frozen-fork filler, API
`SubmissionStatus.COMPLETE`), following **0.65 on 2026-08-01**. Both below the historical
band floor 0.82.

---

## 1. Reconstructed frozen-fork / control draw record (date order, API-verified)

Source of truth: `uvx --from kaggle==2.0.0 kaggle competitions submissions
arc-prize-2026-arc-agi-3` (pulled 2026-08-02T12:2xZ), cross-checked against
`runs/lb_ground_truth.md` (n=18 ledger) and `ITERATION_LOG.md` draw entries. Only
**frozen-fork control fillers** are in the record ledger; war-arm (n=5, CLOSED) and
sentinel (n=1, SHELVED) draws are tracked separately and excluded per prereg §3.

| # | date (00:07Z) | score | note |
|---|---|---|---|
| 1 | 2026-07-09 | 0.89 | sigma draw #1 |
| 2 | 2026-07-10 | 0.93 | sigma draw #2 |
| 3 | 2026-07-11 | 1.02 | sigma draw #4 |
| 4 | 2026-07-12 | 0.95 | sigma draw #5 |
| 5 | 2026-07-18 | 1.33 | **record high** (banked public best) |
| 6 | 2026-07-20 | 0.92 | filler |
| 7 | 2026-07-21 | 0.93 | manual fire test |
| 8 | 2026-07-22 | 1.14 | filler |
| 9 | 2026-07-23 | 0.82 | filler |
| 10 | 2026-07-25 | 1.05 | filler |
| 11 | 2026-07-26 | 0.84 | filler |
| 12 | 2026-07-27 | 1.02 | filler |
| 13 | 2026-07-28 | 0.90 | filler |
| 14 | 2026-07-29 | 1.03 | filler |
| 15 | 2026-07-30 | 0.85 | filler — **SEALED CONTROL ends here (n=15)** |
| 16 | 2026-07-31 | 1.10 | filler |
| 17 | 2026-07-08 | 0.82 | (Q2 duck repro; ordered by ledger — see note) |
| 18 | 2026-08-01 | **0.65** | campaign low #1 |
| 19 | 2026-08-02 | **0.68** | campaign low #2 (this trigger) |

> Ledger list used for all arithmetic (the canonical `runs/lb_ground_truth.md` ordering,
> which folds the 07-08 0.82 into the sequence and appends the two lows):
> `[0.82, 0.89, 0.93, 1.02, 0.95, 1.33, 0.92, 0.93, 1.14, 0.82, 1.05, 0.84, 1.02, 0.90,
> 1.03, 0.85, 1.10, 0.65, 0.68]` (n=19). The exact interior ordering does not affect MK
> (rank-based on the whole series) materially, and the two lows are unambiguously the last
> two draws chronologically — which is what the CUSUM/change-point tests key on.

**Ledger stats (`uv run python`, numpy):**
- Sealed A/B control (n=15, FROZEN per prereg §3): **mean 0.9727, s 0.1343** — unchanged.
- Record ledger before tonight (n=18, after 0.65): mean 0.9550, s 0.1500.
- Record ledger now (n=19, after 0.68): **mean 0.9405, s 0.1588** (mean −0.015, s +0.009
  from the single 0.68 draw; cumulative −0.032 mean / +0.025 s from the two lows vs sealed).
- z(0.65) vs sealed = **−2.40**; z(0.68) vs sealed = **−2.18**.
- Mean of last 2 draws = 0.6650 vs mean of prior 17 = 0.9729 (**Δ = −0.308**).

---

## 2. Stationarity test battery (record ledger n=19; `uv run python`, scipy)

### (a) Mann-Kendall trend test
- S = **−14**, Var(S) = 814.0, **z = −0.456, two-sided p = 0.649**. Sen slope = **−0.0050/draw**.
- **Reading: NO monotone trend.** MK is a whole-series rank test; it does not see a
  *late-onset level break* because 17 of 19 draws carry no downward ordering signal. MK
  alone would say STATIONARY — but MK is the wrong instrument for a two-point tail cluster.

### (b) CUSUM (standardized cumulative sum vs sealed N(0.9727, 0.1343))
- Standardized cumsum path ends at **final CUSUM = −4.55**; **max |CUSUM| = 4.55 at draw 19**.
- **Crosses h=4 : YES. Crosses h=5 : NO.**
- Tabular one-sided lower CUSUM (k=0.5, downward shift): min = **−3.58** (below a typical
  H=4 alarm, above H=5).
- **Reading: MARGINAL alarm.** The h=4 crossing is generated ENTIRELY by draws 18–19 — the
  path was mean-reverting and bounded within ±2.1 through draw 17, then dropped 2.4σ + 2.2σ
  on the last two points. This is exactly the signature the watch-rule was built to catch:
  not a slow drift, but a possible recent step-down. It clears the weaker bound (h=4) and
  fails the stricter one (h=5).

### (c) Change-point scan (max |Welch t| over interior splits, permutation p)
- Max |t| = **8.64 at the split after draw 17** (mean before 0.9729, mean after 0.6650).
- Pre-specified split (the watch-rule's own last-2-vs-prior-17 split): Welch t = **8.65**.
- **Permutation p = 0.0032** (B=20,000; the max-|t| statistic recomputed on each of 20k
  random reorderings, corrected for the multiple split points scanned).
- **Reading: SIGNIFICANT change-point at the tail.** p=0.0032 rejects the null of a single
  exchangeable sequence. But note the caveat: this test is *guaranteed* to find its most
  extreme split at whatever the two most extreme values are, and here they happen to be
  chronologically last — the permutation p already accounts for the "look everywhere"
  multiplicity, so 0.0032 is honest, but it is driven by n=2 tail points, not a broad shift.

### (d) Consecutive-pair probability under the frozen null
Per-draw P(≤0.68 | sealed):
- **Gaussian N(0.9727, 0.1343): 1.47%**
- t-predictive (ν=14, sd_pred = 0.1387): 2.67%

P(two *consecutive* draws ≤0.68):
- P(one specific adjacent pair both ≤0.68): Gaussian 2.15e-4, t 7.11e-4.
- **P(≥1 consecutive ≤0.68 pair anywhere in the n=19 run):** **Gaussian 0.38%, t 1.24%**
  (exact via no-two-consecutive-successes recurrence).
- Expected count of ≤0.68 draws in 19 = **0.28** (Gaussian). Observing **two, adjacent**,
  is a ~1-in-260 (Gaussian) to ~1-in-140 (t) event *for a specific pair*, and ~1-in-260
  (G) / ~1-in-80 (t) for *any* consecutive pair in the run.

**This is the decisive number.** A single −2.4σ point (the 0.65 alone) was tail-consistent
(P(≥1 of 18) = 13.7% G / 27.6% t per the 08-01 deep-dive). But **two consecutive sub-0.80,
sub-tail draws is NOT tail-consistent** under the sealed N(0.9727, 0.1343): P = 0.38%–1.24%.
That is the qualitative break from the 08-01 "isolated tail" verdict.

### External context — alternate variance regime (yw8837 public duck fork)
yw8837's public duck fork (same artifact family/config) reports an **11-run spread
0.55–1.29 with σ≈0.24** — nearly double our sealed s=0.1343. Re-running (d) under
N(0.9727, **0.24**):
- Per-draw P(≤0.68) = **11.1%**; **P(≥1 consecutive pair in 19) = 18.5%**.
- Under this wider regime the two lows are **entirely ordinary** (18.5% ≈ 1-in-5).

**This is the crux of the verdict.** The two lows are either (i) a genuine level step-down
in *our* process, OR (ii) evidence that our sealed s=0.1343 **understates the true duck-family
variance** and the real σ is closer to yw8837's 0.24. Both hypotheses reduce to the same
operational risk: **the sealed control's s=0.1343 is too tight**, which is the parameter that
sets the A/B promote threshold. We cannot distinguish (i) vs (ii) from n=2, but we do not
need to — either way the sealed *dispersion* is suspect.

---

## 3. VERDICT

> **INCONCLUSIVE-PROCEED-WITH-GUARD.** The record ledger shows a marginal tail alarm
> (CUSUM crosses h=4 not h=5; change-point p=0.0032; consecutive-pair p=0.38–1.24% under
> the sealed Gaussian) that is driven entirely by the two most recent draws and is fully
> absorbed under the plausible wider duck-family variance regime (σ≈0.24 → p≈18.5%). MK
> finds no trend (p=0.65). This is NOT clean stationarity (a single isolated tail), but it
> is NOT a confirmed level break either (n=2 tail points; whole-series trend null; wider-σ
> null benign). **HOLD the first gated A/B draw until the guard below is satisfied.**

**Why not STATIONARY:** two consecutive draws below the entire historical band, at
P≤1.24% under the very Gaussian that defines the control, with a change-point rejecting
exchangeability at p=0.0032. The 08-01 "isolated tail" reading is explicitly superseded —
that verdict pre-committed to escalate on a second sub-0.80 draw, and it landed.

**Why not NON-STATIONARY (full hold + re-baseline):** the alarm is n=2-driven, clears only
the weaker CUSUM bound (h=4, not h=5), MK is null, and the entire signal evaporates under
the independently-attested σ≈0.24 duck-family regime. Declaring a confirmed regime change
and re-baselining the control off two points would itself be an overfit to noise — the exact
error the campaign's measurement discipline warns against (`feedback_simplicity_wins`,
`feedback_prompt_is_noise`: single-draw / tiny-n swings on identical code are the norm here).

### GUARD (must be satisfied before / around the first gated draw)
1. **Interleave the A/B's own control.** Do NOT fire a bare gated draw against the sealed
   n=15 historical control. The first A/B slot must run **control-interleaved**: pair each
   gated (arm B) draw with a same-window frozen-fork control draw, so the promote comparison
   uses a *contemporaneous* control mean rather than the possibly-stale sealed 0.9727. This
   neutralizes any level step-down (it cancels in the paired contrast) and any σ
   mis-estimate (it re-estimates dispersion in-window).
2. **Re-check after each of the next control fillers.** Re-run this battery (MK/CUSUM/
   change-point) after draws 20 and 21. Escalation ladder:
   - If the next control filler is **≥0.82** (back in band): the two lows were a paired
     tail cluster; downgrade to STATIONARY-WITH-NOTE and the sealed control stands.
   - If a **third consecutive sub-0.80** lands: promote to **NON-STATIONARY**, re-baseline
     the control on a fresh in-regime window (n≥8), and re-derive the promote threshold.
3. **Widen the working dispersion now, provisionally.** For any threshold arithmetic used
   before the re-check clears, use a variance sensitivity band spanning s∈[0.1343, 0.24]
   (sealed vs yw8837), and require the gated result to clear the *upper*-σ threshold, not
   just the sealed-σ one (see §4).

---

## 4. Promote-threshold survival (boristown A/B)

The boristown promote rule is **mean-of-4 gated ≥ 1.0970**, derived as
`x̄_C + 1.645·(s/√4) = 0.9727 + 1.645·(0.1343/2) = 0.9727 + 1.645·0.06715 = 1.0970`
(Gaussian, governing; t-robust cross-check bar 1.1269). Both inputs (mean 0.9727,
s 0.1343) come from the **sealed n=15 control**.

**Does the derivation survive?** The control is sealed by construction — the two lows accrue
to the *record*, not the control — so the *number* 1.0970 is arithmetically unchanged.
**But its validity as a decision boundary is now in question, and the drift biases the test
in a specific direction:**

- **Direction of bias — the threshold is now BIASED TOWARD (too-easy) PROMOTION.** If the
  process mean has actually stepped DOWN (hypothesis i) and/or the true σ is wider than
  0.1343 (hypothesis ii, σ≈0.24), then a threshold pinned to the *old, higher, tighter*
  control (0.9727 / 0.1343) sits **too low relative to the current control distribution**.
  A gated mean-of-4 could clear 1.0970 not because the gate helped, but because the
  threshold was calibrated to a control that no longer exists / was mis-measured. Concretely:
  under σ=0.24 the correct 95% one-sided bar would be 0.9727 + 1.645·(0.24/2) = **1.1701**,
  which is **0.073 above** the sealed 1.0970 — so the sealed bar is materially too lenient
  if the wider-variance regime is real. **The drift makes a spurious PROMOTE more likely,
  not less.** This is the dangerous direction (false-positive promotion of an inert arm).

- **Therefore:** the sealed 1.0970 **does NOT survive as a standalone bar.** It survives
  only *conditional on the guard*: with control-interleaving (Guard #1) the promote test
  should use the contemporaneous paired control, which self-corrects both the level and the
  σ. Absent interleaving, require the gated mean-of-4 to clear the **wider-σ bar 1.1701**
  (or equivalently the t-robust 1.1269 at minimum, and preferably 1.1701) before any promote
  is credited. Do not promote on 1.0970 alone until the stationarity re-check (Guard #2)
  clears.

---

## 5. LB snapshot (2026-08-02T12:25Z, full CSV, 2011 teams)

| item | value |
|---|---|
| Our banked best | **1.33 — INTACT** (byte-for-byte; a rescoring would have rewritten it) |
| Our team / rank | **Canivel, #65** (60 strictly above, 7 tied at 1.33 spanning #59–65) |
| Rank drift | 08-01 #63 → 08-02 #65 — pure competitive churn, no change to our draw |
| Leader | YUTO KOJIMA **1.86** (unchanged) |
| Head #2–4 | Andy liu 1.69, GeniusYY 1.64, Tecnod8.AI 1.61 — head frozen vs 08-01 |
| **Gold cutoff (#13)** | **1.54** (paul; #14 Seok 1.54, #15–16 = 1.50, #17–18 = 1.49) — unchanged from 08-01 (was ≈1.54) |
| boristown anchor 1.47 | now ~#20 band (Lord Han Solo 1.47), ~0.07 below gold |

**No platform-wide event:** the entire head is frozen and our historical 1.33 is unchanged,
so the two lows are OUR draws, not a rescoring / game-set rotation / eval-infra shift. The
gold cutoff held at 1.54 (no further climb overnight). Our banked entry is safe; the two
lows are control fillers and do not touch the banked best.

---

## Summary line

**VERDICT: INCONCLUSIVE-PROCEED-WITH-GUARD** — HOLD the first gated A/B draw; require
control-interleaving + re-check after draws 20/21; provisionally widen working σ to the
[0.1343, 0.24] band. MK z=−0.46 (p=0.65, no trend); CUSUM max|4.55| crosses h=4 not h=5;
change-point p=0.0032 (t=8.64, split after draw 17); P(consecutive ≤0.68 pair | sealed)
= 0.38% G / 1.24% t (but 18.5% under σ≈0.24). Sealed promote bar 1.0970 does NOT survive
standalone — drift biases TOWARD spurious promotion; require 1.1701 (wider-σ) or an
interleaved paired control. LB: our 1.33 intact at **#65**, gold cutoff **1.54**, head frozen.

---
*Prepared 2026-08-02. All statistics `uv run python` (numpy + scipy) on the API-verified
record ledger. No cloud spend, no pushes. Control-arm analysis only — no capability
inference. Supersedes the 08-01 "isolated tail" reading per its own pre-committed escalation.*
