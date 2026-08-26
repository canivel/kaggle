# Variance reconciliation — null10 build rail vs Kaggle LB
Date: 2026-07-14. Addresses panel R9 ME-NEW-11 (methodology), P (rl-planning), Q3 (systems).
Scorer: validated RHAE mirror (max err vs Tufa 500 runs: 0.0e+00, 1000 checks).

## Direct (non-bootstrap) replicate noise, 25-game build/pod rail
- 10 per-seed 25-game means: 1.706, 0.751, 1.585, 1.399, 1.801, 1.562, 1.549, 2.365, 2.674, 0.967
- **1-seed run-mean sd = 0.572**
- **1-seed paired-delta sd (45 pairwise) = 0.780** (sqrt2 x run-mean = 0.809)
- If games were independent, implied paired-delta sd = 0.700

## Bootstrap replication (what unit produced 0.52?)
- Game-resampling, pairing BROKEN across arms: sd = 1.212
- Game-resampling, pairing KEPT: sd = 1.068

## Per-game variance decomposition (top 8)
| game | null mean | across-seed var | share of run-mean var |
|---|---|---|---|
| ft09 | 10.20 | 83.087 | 54.3% |
| vc33 | 4.21 | 26.245 | 17.2% |
| tn36 | 2.05 | 11.700 | 7.6% |
| ar25 | 3.26 | 9.482 | 6.2% |
| cn04 | 1.62 | 5.464 | 3.6% |
| tu93 | 1.54 | 4.091 | 2.7% |
| re86 | 2.57 | 3.799 | 2.5% |
| r11l | 3.52 | 2.881 | 1.9% |

## LB rail
- Control draws sd = 0.074 -> paired-delta sd = 0.105
- Ratio build-rail/LB paired-delta sd = 7.5x

## 3-seed gate on the build rail (alpha = 0.0125, one-sided)
- se(mean paired delta, 3 seeds) = 0.450
- Power vs +0.10: 0.02; vs +0.12: 0.02; vs +0.20: 0.04

## Alternative gate statistics (1-seed sd -> 3-seed se -> power vs +0.10 / +0.20)
| statistic | sd_1seed | se_3seed | power +0.10 | power +0.20 |
|---|---|---|---|---|
| full | 0.572 | 0.467 | 0.02 | 0.03 |
| no_ft09 | 0.387 | 0.316 | 0.03 | 0.05 |
| no_top2 | 0.328 | 0.268 | 0.03 | 0.07 |
| log1p | 0.116 | 0.095 | 0.12 | 0.45 |
| levels_completed | 0.086 | 0.070 | 0.21 | 0.73 |

## Expected max over k daily LB draws (control mean 0.922)
| k | sigma=0.074 | sigma=0.213 (chi2 CI hi) | sigma=0.52 (bootstrap claim) |
|---|---|---|---|
| 5 | 1.01 | 1.17 | 1.53 |
| 10 | 1.04 | 1.25 | 1.72 |
| 30 | 1.07 | 1.36 | 1.98 |
| 60 | 1.09 | 1.42 | 2.13 |
| 110 | 1.11 | 1.46 | 2.24 |

Reading: the 1.44 wall is unreachable by order statistics from the current
per-draw distribution under any candidate sigma except the (falsified) 0.52
LB claim; only per-draw mean improvements close the gap.
