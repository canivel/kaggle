# NC-13 / NC-15 discharge — stationarity memo audit (2026-08-02)

**Scope.** Discharges panel R23 statistics deliverables **NC-15** (pair-probability
reproduction script + conditional false-alarm sim + pre-registered scan config) and **NC-13**
(yw8837 σ=0.24 variance-compatibility) against the memo
`learnings/sweeps/stationarity_2026-08-02.md`, plus the 4th vote-blocker (paired harm-pause
re-derivation). These govern how ALL control-ledger draws and harm-pause thresholds are
interpreted going forward, independent of the now-dormant boristown A/B (NC-14 observational
leg found all-zero gate latencies).

**All compute local, `uv run python`, deterministic seed 20260802.** Reproduction script:
`scripts/stationarity_repro.py`. Machine-readable artifact:
`runs/stationarity_repro_2026-08-02.json`. Companion: `nc15b_false_alarm_2026-08-02.md`.

Ledger (n=19, canonical `runs/lb_ground_truth.md` ordering):
`[0.82, 0.89, 0.93, 1.02, 0.95, 1.33, 0.92, 0.93, 1.14, 0.82, 1.05, 0.84, 1.02, 0.90, 1.03,
0.85, 1.10, 0.65, 0.68]`.

---

## 1. NC-15a — reproduce-or-not, every memo number

Tolerance: rel 5% (abs floor 0.003) unless noted. 16 of 17 checks reproduce.

| memo number | memo value | reproduced | status |
|---|---|---|---|
| MK S | −14 | −14 | ✅ |
| MK Var(S) | 814.0 | 814.0 | ✅ |
| MK z | −0.456 | −0.456 | ✅ |
| MK two-sided p | 0.649 | 0.649 | ✅ |
| Sen slope | −0.0050 | −0.0050 | ✅ |
| CUSUM final | −4.55 | −4.55 | ✅ |
| CUSUM max\|·\| | 4.55 | 4.55 | ✅ |
| CUSUM tabular lower min | −3.58 | −3.58 | ✅ |
| change-point max\|t\| | 8.64 | 8.64 (split after draw 17) | ✅ |
| change-point pre-reg last-2 t | 8.65 | 8.64 | ✅ (rounding) |
| **change-point permutation p** | **0.0032** | **0.0117** | ❌ **DOES NOT REPRODUCE** |
| per-draw P(≤0.68) Gaussian | 1.47% | 1.46% | ✅ |
| per-draw P(≤0.68) t (ν=14) | 2.67% | 2.67% | ✅ |
| **P(≥1 consec pair ≤0.68) Gaussian** | **0.38%** | **0.38%** | ✅ **reproduces** |
| P(≥1 consec pair ≤0.68) t | 1.24% | 1.24% | ✅ |
| per-draw P(≤0.68) σ=0.24 | 11.1% | 11.1% | ✅ |
| P(≥1 consec pair ≤0.68) σ=0.24 | 18.5% | 18.5% | ✅ |

### Corrections to the memo (accuracy over consistency)

**Correction 1 — the change-point permutation p is wrong.** Memo §2(c) reports
**p=0.0032** (B=20,000). The faithful reproduction of the same statistic (max-|Welch t| over
interior splits, permutation null on the max) is **p=0.0117** — confirmed twice (20k
permutations in-script, and against an independent 200k reference in the NC-15b sim). The
memo overstates the change-point significance by ~3.7×. **At the memo's own implied α=0.01
bar, the observed change-point is NOT significant.** The memo's summary line and the
`runs/lb_ground_truth.md` header ("permutation p=0.0032 → exchangeability rejected") should be
corrected to p=0.012, exchangeability NOT rejected at 0.01.

**Correction 2 — the pair-probability the PANEL suspected is actually CORRECT; the panel's
counter-numbers are the ones that are wrong.** Directive 4 says no reviewer could reproduce
the 0.38%/18.5% pair probabilities and offers ≈0.98% / ≈5.6% instead. The reviewers computed
the wrong event: they used **P(draw ≤ 0.80)²** (≈0.99%) and a per-pair joint. The memo's
stated event is **P(≥1 consecutive pair BOTH ≤ 0.68, scanned over all 18 adjacencies in the
n=19 run)**, evaluated exactly via the no-two-consecutive-successes recurrence. Under that
definition the numbers reproduce to the digit: **Gaussian 0.38%, t 1.24%, σ=0.24 18.5%.** The
memo's pair-probs stand; the reviewers' 0.98%/5.6% used threshold 0.80 (not 0.68) and a
specific-pair (not scanned) event. **NC-15's demand for the exact event definition + script is
now satisfied and vindicates the memo on this point.**

**Correction 3 — the change-point is an n₂=2 artifact and collapses under the pre-registered
min-segment≥3 constraint (NC-15(iii) / directive 9).** With MIN_SEGMENT=3 the max-|t| drops
from **8.64 → 1.40** (best split moves to after draw 15) and the permutation p goes from
0.012 → **0.72**. The entire change-point signal is the two-point right segment (0.65, 0.68,
sd≈0.021), exactly the degenerate |t| systems flagged. **Pre-registered config now stated in
the script header:** CUSUM h-state=4, h-strict=5, k=0.5; min-segment=3; permutation B=20,000.

**Net NC-15a verdict:** 16/17 memo numbers reproduce. The one failure (permutation
p=0.0032) is a material overstatement — the true p is 0.0117, non-significant at the memo's
own 0.01 bar, and 0.72 under the pre-registered min-segment rule. The pair-probs the panel
distrusted are correct.

---

## 2. NC-15b — conditional multiple-looks false-alarm rate

10,000 stationary-null ledgers (sealed N(0.9727, 0.1343), n=19), identical pipeline, fraction
with permutation p<0.01:

- **False-alarm rate = 1.04%** (≈ nominal 1% → the permutation test is calibrated; the
  split-multiplicity IS absorbed by the max-|t| construction, so the memo's claim that the p
  "honestly accounts for looking everywhere" is correct *as far as splits go*).
- Median null p = 0.497; 5th-percentile = 0.049.
- The single look we took returned p=0.0117 — **fully consistent with the stationary null**
  once computed correctly. See `nc15b_false_alarm_2026-08-02.md`.

---

## 3. NC-13 — yw8837 σ=0.24 compatibility

χ² variance test H₀: σ = 0.24, statistic (n−1)·s²/σ₀² ~ χ²_{n−1}. 95% CI for σ from our
own data.

| our data | s | χ² | P(s ≤ obs \| σ=0.24) | two-sided p | 95% CI for σ | 0.24 inside CI? |
|---|---|---|---|---|---|---|
| sealed n=15 | 0.1343 | 4.384 | **0.0073** | 0.0146 | [0.098, 0.212] | **NO** |
| record n=19 | 0.1588 | 7.884 | **0.0197** | 0.0394 | [0.120, 0.235] | **NO** |

**The panel's p≈0.007 is CONFIRMED exactly** (prog-synthesis: "χ²₁₄ ≈ 4.38, P(s≤0.1343 |
σ=0.24) ≈ 0.007" → we get 0.0073). It holds on the larger n=19 record too (P=0.0197; σ=0.24
sits just outside the upper 95% CI bound of 0.235).

**NC-13 VERDICT: "the duck family has σ≈0.24" is REJECTED by our fork's own draw record.**
σ=0.24 lies outside the 95% CI for σ on both the sealed n=15 (upper bound 0.212) and the
record n=19 (upper bound 0.235), one-sided p 0.007 (n=15) / 0.020 (n=19). The two data-sets
concur. Per NC-13's own discharge clause, **σ=0.24 is struck from any decision rule** (promote
bar, pair-probability exoneration). It may not license the "1.1701 corrected bar" nor the
"18.5% ⇒ two lows are ordinary" exoneration in the memo. σ for any forward threshold must be
estimated from our own contemporaneous draws (design (b)/(c)).

*Provenance note (the other NC-13 discharge route):* the yw8837 figure has no fork-diff/ledger
provenance in-repo — it is an 11-run public-fork spread (0.55–1.29) quoted second-hand. The
only in-repo boristown-family provenance memo, `fork_diff_boristown_2026-07-24.md`, covers a
DIFFERENT fork (boristown, whose sole functional diff is the vLLM readiness gate) and does not
attest yw8837's variance. So σ=0.24 fails BOTH discharge routes: rejected on our data AND
unprovenanced. It is inadmissible.

### Harm-pause fixed <0.80 fire rates under each regime (reproduce the ~34%/~66% claim)

Fixed rule: pause if a draw < 0.80. P(single) and P(≥1 in 4 draws):

| regime | per-draw P(<0.80) | P(≥1 in 4 draws) | panel claim | match |
|---|---|---|---|---|
| sealed N(0.9727, 0.1343) | 9.9% | **34.2%** | ~34% | ✅ exact |
| σ=0.24 N(0.9727, 0.24) | 23.6% | **65.9%** | ~66% | ✅ exact |
| step-down μ=0.665, s=0.1343 | 84.3% | **99.9%** | "fires almost surely" | ✅ |

**The panel's ~34%/~66% figures reproduce exactly** (prog-synthesis directive 5). The memo's
claim that the fixed <0.80 harm-pause is "unaffected" is **false**: it trips a spurious
harm-pause in a 4-draw arm at 34% under the sealed control and 66% under σ=0.24 — coin-flip or
worse — *regardless of the gate's true effect*. This is the direct motivation for the paired
re-derivation below.

---

## 4. Paired / relative harm-pause re-derivation (4th vote-blocker)

**Proposed drop-in replacement for the fixed <0.80 floor:**

> **Pause the arm iff the gated draw < mean(trailing-k contemporaneous frozen-fork fillers)
> − c·s(trailing-k),** with k=4 and c=1.5 (a ~1.5-sd relative floor against the *same-window*
> control, not an absolute number).

**Why it's the right instrument:** harm is defined as *gated worse than its contemporaneous
control*, so a level step-down or a σ mis-estimate in the frozen fork **cancels in the
contrast** — the criterion is scale- and location-relative to the fillers drawn in the same
window. This is exactly design (b)'s pairing and satisfies directive 5.

**False-fire rates under the null (gate == control; 200k sims each):**

| regime | c=1.5 | c=2.0 |
|---|---|---|
| sealed s=0.1343 | **13.7%** | 8.5% |
| σ=0.24 | 13.7% | 8.6% |
| intermediate s=0.19 | 13.6% | 8.6% |

**The false-fire rate is invariant across all three σ regimes** (≈13.7% at c=1.5, ≈8.5% at
c=2.0) — the whole point: unlike the fixed <0.80 rule (34% → 66% as σ widens), the paired
rule does not inflate when the dispersion is mis-specified or the level steps down. Sanity
check on power: when the gate is *truly* degraded by −0.15, the paired rule fires at **42%**
(vs 13.7% null) — it retains discriminating power for real harm.

**Recommendation:** replace the fixed <0.80 floor with the paired **c=1.5** rule (13.7%
per-draw false-fire, regime-invariant) for the harm-pause; **c=2.0** if a stricter ~8.5%
false-fire is wanted at some cost to sensitivity. The trailing-k control mean must come from
interleaved contemporaneous fillers (design (b)), not the stale sealed n=15.

Per-draw false-fire for the fixed rule was 9.9% (sealed) but that number is a mirage — it
rises to 23.6% under σ=0.24 and 84% under a step-down, whereas the paired rule holds at ~13.7%
across all of them. The paired rule trades a slightly higher *nominal* false-fire under the
best case for *robustness* to exactly the two failure modes (level shift, σ mis-estimate) the
guard exists to handle.

---

## 5. Bottom line

- **NC-15a: 16/17 memo numbers reproduce.** The one that does NOT is the headline change-point
  **permutation p (memo 0.0032, actual 0.0117)** — overstated ~3.7×, and non-significant at
  the memo's own 0.01 bar. Under the pre-registered min-segment≥3 rule the change-point
  collapses entirely (max|t| 8.64→1.40, p→0.72): it is an n₂=2 tail artifact.
- **The pair-probs the panel suspected are correct** (0.38% / 1.24% / 18.5%); the panel's
  0.98%/5.6% used the wrong threshold (0.80 not 0.68) and event (specific-pair not scanned).
- **NC-15b false-alarm rate = 1.04%** — the permutation test is calibrated; the observed look
  (correctly, p=0.012) is consistent with stationarity.
- **NC-13: σ=0.24 is REJECTED on our own data** (p=0.007 at n=15, 0.020 at n=19; outside the
  95% CI both ways) AND unprovenanced → struck from all decision rules. Estimate σ from
  contemporaneous draws. The fixed-<0.80 harm fire rates **34.2% / 65.9%** reproduce the
  panel's ~34%/~66% exactly.
- **Paired harm-pause** (gate < trailing-4 filler mean − 1.5s): **false-fire ≈13.7%,
  invariant across σ∈{0.13, 0.19, 0.24}**, vs the fixed rule's 34%→66% blow-up; retains 42%
  power against a true −0.15 harm. Adopt as the drop-in replacement.

**Interpretation guidance going forward (governs all control-ledger draws):** (i) do not cite
p=0.0032 — the stationarity alarm is *not* significant; the two lows are within stationary-null
expectation once the change-point is computed correctly and constrained to min-segment≥3;
(ii) never use σ=0.24 in a threshold or exoneration — it is rejected by our data; estimate
dispersion from contemporaneous fillers; (iii) replace the fixed <0.80 harm-pause with the
regime-invariant paired rule.

Artifacts: `scripts/stationarity_repro.py`, `runs/stationarity_repro_2026-08-02.json`,
`learnings/sweeps/nc15b_false_alarm_2026-08-02.md`.
*Prepared 2026-08-02. Local compute only, no pushes, no cloud spend, no queue changes.*
