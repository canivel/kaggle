# Stationarity re-check — 2026-08-07 (watch-rule fire: 0.77 → 0.78)

**Verdict: STATIONARY.** Ledger is safe to use for gate arithmetic today.

- Trigger: pre-registered two-consecutive-sub-0.80 watch-rule FIRED (08-06 draw
  0.77, 08-07 draw 0.78). Rule requires this re-check before any gate arithmetic.
- Series: canonical n=24 frozen-fork record ledger from `runs/lb_ground_truth.md`
  (mean 0.94125, s 0.1595) = sealed n=19 + 0.99, 0.97, 1.21, 0.77, 0.78.
- Method: adapted copy `scripts/stationarity_recheck_20260807.py` of the SEALED
  `scripts/stationarity_repro.py` (unmodified), same pre-registered constants
  (min-segment ≥ 3, B=20,000 permutations, tabular CUSUM k=0.5/h=4, α=0.01).
- Output artifact: `runs/stationarity_recheck_2026-08-07.json` (seed 20260807).

## Numbers

| Test | Result | Call |
|---|---|---|
| Mann-Kendall trend | S=−21, z=−0.50, **p=0.620**; Sen slope −0.0034/draw | no drift |
| Change-point, min-segment≥3 (primary) | max\|t\|=1.43 at split 3, **perm p=0.757** | NS |
| Change-point, unconstrained (diagnostic) | \|t\|=5.32 at split 22 (n₂=2: the 0.77/0.78 pair itself), perm p=0.029 | NS at 0.01; same n₂=2 tail artifact as 08-02 |
| Tabular lower CUSUM (k=0.5, h=4) | min −3.58, no h=4 breach (min still dates from the 07-31/08-02 dip; new draws only reach −1.94 before resetting) | no alarm |
| Raw standardized cumsum (diagnostic) | maxabs 5.62 — mechanical ~−0.23σ/draw accumulation from the frozen-control mean offset (0.9413 vs 0.9727), adjudicated at 08-02 (maxabs 4.55, verdict stationary) | not an alarm statistic |
| Firing-draw z-scores | 0.77: −1.51 (sealed) / −1.18 (prior-22 record); 0.78: −1.43 / −1.06. Last-2 mean 0.775: z=−2.08 vs sealed | interior-low, not tail |
| σ=0.24 (yw8837) | n=24 record χ² lower-tail p=0.0097 (two-sided 0.0194), σ₀ outside 95% CI [0.124, 0.224]; sealed n=15 lower-tail p=0.0073 | stays REJECTED |

## Pair-event surprise (multiple-looks aware)

P(at least one adjacent pair both < 0.80 somewhere in a 24-draw record):

- sealed Gaussian N(0.9727, 0.1343): per-draw P(<0.80)=0.099 → **pair-prob 0.189**
- sealed t-predictive (df=14): per-draw 0.117 → **pair-prob 0.249**
- n=24 record-fit N(0.9413, 0.1595): per-draw 0.188 → **pair-prob 0.512**

## Interpretation

The two firing draws are individually unremarkable (z ≈ −1.2 and −1.1 against the
record) and the fired pair-event is not surprising once the ~24-draw record length
is accounted for: under the strictest null (sealed Gaussian) a sub-0.80 adjacent
pair was expected somewhere in the record with probability ≈ 0.19, rising to ≈ 0.25
under the honest t-predictive and to a coin-flip under the record's own fit —
this is the multiple-looks correction the 08-02 discharge taught us to apply. The
only nominally strong statistic is the unconstrained change-point split isolating
the last two draws (|t|=5.32), which is the identical n₂=2 tiny-variance artifact
that NC-15 discharged on 08-02 (0.77 vs 0.78 are nearly equal, so the two-point
"segment" has near-zero variance); under the pre-registered min-segment≥3 scan it
collapses completely (p=0.757, even weaker than 08-02's 0.72). MK shows no trend,
and the pre-registered tabular CUSUM never approaches h=4 — its minimum still
dates from the already-adjudicated 07-31/08-02 dip. Verdict **STATIONARY**: the
watch-rule did its job and the check clears it; the n=24 control ledger (and the
frozen n=15 sealed control parameters) are safe for gate arithmetic today. Residual
note for the record: mean-of-last-2 sits at z=−2.08 vs the sealed control — if
tomorrow's draw is a THIRD consecutive sub-0.80 (an event with per-draw probability
≈ 0.10–0.19 under the stationary fits), re-fire this check before any gate use.
