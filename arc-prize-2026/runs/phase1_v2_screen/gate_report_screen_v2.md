# Phase-1 A/B gate report — **PROVISIONAL** (1/3 seeds)

> **PROVISIONAL** — seed 3 not yet included. Do not act on this verdict.

## Scorer validation (hard requirement)
- Reproduced Tufa score.json + per-run final_score from raw benchmark.json: **max abs error 0.000e+00** over 1000 cross-checks (500 runs, 25 game means, overall). PASS (limit 1e-9).
- Recomputed overall null score 1.600204 vs published 1.600204.
- Formula: per level i (0-idx, weight i+1): min(115, (base/actions)^2*100) if completed & actions>0 else 0; score = weighted mean over ALL level weights, capped at (weights of scoring levels)/(all weights)*100.

## Arms
- Null: runs/tufa_example_run/benchmark.json — 25 games x 20 vanilla passes.
- Ours seed 1: runs/phase1_v2_screen/seed1/benchmark.json (detail: runs\phase1_v2_screen\seed1\benchmark.json, 25 games)
- Both arms scored OFFLINE with identical formula and the null arm's base_actions_per_level (joined on 4-char game prefix).

## Per-game paired deltas (RHAE, Kaggle-comparable units)

| game | null mean (20p) | ours mean | delta | our lc | null lc mean | flags |
|---|---|---|---|---|---|---|
| ar25 | 1.7470 | 1.0519 | -0.6951 | 1 | 0.95 |  |
| bp35 | 0.3137 | — | — | — | 0.90 | **EXCLUDED: pre-registered (flaky arcade seeds 2-3)** |
| cd82 | 0.7752 | 0.0000 | -0.7752 | 0 | 0.25 |  |
| cn04 | 0.2130 | 0.0000 | -0.2130 | 0 | 0.15 |  |
| dc22 | 0.2008 | 0.0000 | -0.2008 | 0 | 0.10 |  |
| ft09 | 10.2830 | 14.2857 | +4.0027 | 2 | 1.50 |  |
| g50t | 0.1341 | 0.0000 | -0.1341 | 0 | 0.05 |  |
| ka59 | 1.3382 | 1.3827 | +0.0445 | 1 | 0.55 |  |
| lf52 | 1.2346 | 1.8182 | +0.5836 | 1 | 0.75 |  |
| lp85 | 2.7242 | 2.7778 | +0.0535 | 1 | 1.00 |  |
| ls20 | 0.3537 | 0.0000 | -0.3537 | 0 | 0.20 |  |
| m0r0 | 0.0477 | 0.0597 | +0.0119 | 1 | 0.05 |  |
| r11l | 3.7076 | 3.1615 | -0.5461 | 1 | 0.90 |  |
| re86 | 1.9125 | 2.7778 | +0.8652 | 1 | 0.90 |  |
| s5i5 | 0.2583 | 0.0000 | -0.2583 | 0 | 0.30 |  |
| sb26 | 2.8339 | 2.7778 | -0.0561 | 1 | 1.10 |  |
| sc25 | 0.1492 | 3.8571 | +3.7079 | 1 | 0.15 |  |
| sk48 | 0.2778 | 0.0000 | -0.2778 | 0 | 0.10 |  |
| sp80 | 1.8289 | 0.1431 | -1.6858 | 1 | 0.65 |  |
| su15 | 2.0842 | 2.2222 | +0.1381 | 1 | 1.00 |  |
| tn36 | 2.0811 | 3.5714 | +1.4904 | 1 | 0.65 |  |
| tr87 | 0.0000 | 0.0000 | +0.0000 | 0 | 0.00 |  |
| tu93 | 2.1714 | 0.0093 | -2.1622 | 1 | 1.65 |  |
| vc33 | 3.3350 | 0.0122 | -3.3228 | 1 | 1.60 |  |
| wa30 | 0.0000 | 0.0000 | +0.0000 | 0 | 0.00 |  |

## Primary gate (RHAE)
- Included games: n = 24
- Mean RHAE, included games: ours 1.6628 vs null 1.6538
- Mean RHAE, ours all 25 games (their-1.6002-scale): 1.5998 (null all-25 reference: 1.6002)
- Mean paired delta: **+0.0090**
- One-sided sign-flip permutation p (improvement): **0.488878** [exact (16777216 sign assignments)]
- Alpha: 0.0125

## VERDICT: **FAIL** (PROVISIONAL)

## Secondary: levels completed (robustness)
- Mean paired lc delta: +0.0604; one-sided p: 0.232708 [exact (16777216 sign assignments)]

## Exclusion / asymmetry checks
- bp35 excluded (pre-registered; absent from our seeds >=2).
- wa30: null levels_completed all integer=True, ours all integer=True; scored offline with identical formula+baselines both arms -> no exclusion needed.
- tr87: null levels_completed all integer=True, ours all integer=True; scored offline with identical formula+baselines both arms -> no exclusion needed.

- Game-version note: 0 games served to us under a different version hash than the null run (); scored with null baselines by level index (pre-registered same-baselines-both-arms).
