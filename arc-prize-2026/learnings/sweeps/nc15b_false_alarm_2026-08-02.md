# NC-15b — conditional multiple-looks false-alarm rate (change-point scan)

**Question (methodology, directive 9 / NC-15(ii)):** the memo's headline change-point
permutation p was computed *as if the test were unconditional*, but the test was only run
because the pre-registered watch-rule fired on a second sub-0.80 draw. What is the
conditional (trigger→scan) false-alarm rate of the memo's "p<0.01" claim under a genuinely
stationary null?

**Method (deterministic, seed 20260802, `scripts/stationarity_repro.py::false_alarm_sim`):**
1. Build a permutation reference distribution of the max-over-splits |Welch t| statistic
   under the exchangeable null (200,000 replicates; scale/location invariant, so built from
   standard normals). This IS the sampling distribution the memo's permutation test uses.
2. Simulate 10,000 stationary ledgers of n=19 from the sealed control N(0.9727, 0.1343).
3. For each simulated ledger, run the *identical* pipeline the memo ran (max-|t| over all
   interior splits, min-segment=2 — i.e. the unconstrained scan that produced the |t|=8.64
   artifact) and compute its permutation p against the reference.
4. Count the fraction with permutation p < 0.01.

**Result:**

| quantity | value |
|---|---|
| n_sims | 10,000 |
| reference size | 200,000 |
| **false-alarm rate (perm p < 0.01)** | **1.04%** |
| median permutation p under the null | 0.497 |
| 5th-percentile permutation p | 0.049 |

**Reading.** The permutation p is *properly calibrated* — a genuinely stationary process
triggers "p<0.01" about 1% of the time, which is exactly what a nominal α=0.01 test should
do. The multiple-splits multiplicity is already absorbed by the max-|t| construction, so the
memo's claim that "0.0032 is honest" (the permutation already accounts for the look-everywhere
multiplicity) is CORRECT as far as the split-multiplicity goes.

**BUT two things the memo got wrong land here (see the main discharge memo):**
1. The memo reported the observed permutation p as **0.0032**; the faithful reproduction on
   the same (unconstrained) config is **0.0117** (20k perms; confirmed against a 200k
   reference). At the memo's own α=0.01 bar the observed change-point is **NOT significant**.
   The 0.0032 figure does not reproduce and overstates the evidence ~3.7×.
2. The relevant multiplicity is not only over splits but over *looks in time* — the scan was
   armed by a data-dependent trigger. The 1.04% here is the per-look false-alarm rate; the
   campaign has taken ~1 such look. The single look we took returned p=0.0117, i.e. a result
   fully consistent with the stationary null once the number is computed correctly.

**Verdict for NC-15b:** the change-point permutation test is calibrated (1.04% ≈ nominal
1%), but the memo's *observed* p=0.0032 does not reproduce (correct value 0.0117, not
significant at 0.01). The stationarity alarm should be downgraded from "SIGNIFICANT
change-point (p=0.0032)" to "change-point not significant at 0.01 (p=0.012), and entirely an
n₂=2 tail artifact — it vanishes (max|t| 8.64→1.40, p→0.72) under the pre-registered
minimum-segment≥3 constraint."

Artifact: `runs/stationarity_repro_2026-08-02.json` (`changepoint`, `nc15b_false_alarm`).
