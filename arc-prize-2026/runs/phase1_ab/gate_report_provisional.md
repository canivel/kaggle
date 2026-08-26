# Phase-1 A/B gate report — **PROVISIONAL** (2/3 seeds)

> **PROVISIONAL** — seed 3 not yet included. Do not act on this verdict.

## Scorer validation (hard requirement)
- Reproduced Tufa score.json + per-run final_score from raw benchmark.json: **max abs error 1.776e-15** over 1000 cross-checks (500 runs, 25 game means, overall). PASS (limit 1e-9).
- Recomputed overall null score 1.600204 vs published 1.600204.
- Formula: per level i (0-idx, weight i+1): min(115, (base/actions)^2*100) if completed & actions>0 else 0; score = weighted mean over ALL level weights, capped at (weights of scoring levels)/(all weights)*100.

## Arms
- Null: runs/tufa_example_run/benchmark.json — 25 games x 20 vanilla passes.
- Ours seed 1: runs/phase1_ab/phase1_seed1.json (detail: runs\phase1_ab\seed1\benchmark.json, 25 games)
- Ours seed 2: runs/phase1_ab/phase1_seed2.json (detail: runs\phase1_ab\seed2\benchmark.json, 24 games)
- Both arms scored OFFLINE with identical formula and the null arm's base_actions_per_level (joined on 4-char game prefix).

## Per-game paired deltas (RHAE, Kaggle-comparable units)

| game | null mean (20p) | ours mean | delta | our lc | null lc mean | flags |
|---|---|---|---|---|---|---|
| ar25 | 1.7470 | 0.1142 | -1.6328 | 1,1 | 0.95 | diff-version |
| bp35 | 0.3137 | — | — | — | 0.90 | **EXCLUDED: pre-registered (flaky arcade seeds 2-3)** |
| cd82 | 0.7752 | 2.3810 | +1.6057 | 0,1 | 0.25 |  |
| cn04 | 0.2130 | 0.0000 | -0.2130 | 0,0 | 0.15 | diff-version; nlev 5 vs 6 |
| dc22 | 0.2008 | 0.0000 | -0.2008 | 0,0 | 0.10 | diff-version |
| ft09 | 10.2830 | 7.1429 | -3.1401 | 0,2 | 1.50 |  |
| g50t | 0.1341 | 0.0000 | -0.1341 | 0,0 | 0.05 |  |
| ka59 | 1.3382 | 0.0000 | -1.3382 | 0,0 | 0.55 | diff-version |
| lf52 | 1.2346 | 1.8182 | +0.5836 | 1,1 | 0.75 |  |
| lp85 | 2.7242 | 2.7778 | +0.0535 | 1,1 | 1.00 |  |
| ls20 | 0.3537 | 1.1856 | +0.8319 | 1,0 | 0.20 |  |
| m0r0 | 0.0477 | 0.1804 | +0.1326 | 1,0 | 0.05 | diff-version |
| r11l | 3.7076 | 4.7619 | +1.0543 | 1,1 | 0.90 | diff-version |
| re86 | 1.9125 | 3.1532 | +1.2407 | 2,1 | 0.90 | diff-version |
| s5i5 | 0.2583 | 0.0000 | -0.2583 | 0,0 | 0.30 | diff-version |
| sb26 | 2.8339 | 2.7778 | -0.0561 | 1,1 | 1.10 |  |
| sc25 | 0.1492 | 4.6842 | +4.5350 | 2,1 | 0.15 | diff-version |
| sk48 | 0.2778 | 1.3889 | +1.1111 | 0,1 | 0.10 | diff-version |
| sp80 | 1.8289 | 0.1183 | -1.7107 | 0,1 | 0.65 | diff-version |
| su15 | 2.0842 | 1.8770 | -0.2071 | 1,1 | 1.00 | diff-version |
| tn36 | 2.0811 | 6.3937 | +4.3127 | 2,1 | 0.65 | diff-version |
| tr87 | 0.0000 | 0.0000 | +0.0000 | 0,0 | 0.00 |  |
| tu93 | 2.1714 | 1.4871 | -0.6843 | 2,2 | 1.65 | diff-version |
| vc33 | 3.3350 | 0.8606 | -2.4743 | 2,1 | 1.60 | diff-version |
| wa30 | 0.0000 | 0.0000 | +0.0000 | 0,0 | 0.00 |  |

## Primary gate (RHAE)
- Included games: n = 24
- Mean RHAE, included games: ours 1.7959 vs null 1.6538
- Mean RHAE, ours all 25 games (their-1.6002-scale): 1.7426 (null all-25 reference: 1.6002)
- Mean paired delta: **+0.1421**
- One-sided sign-flip permutation p (improvement): **0.349438** [exact (16777216 sign assignments)]
- Alpha: 0.0125

## VERDICT: **FAIL** (PROVISIONAL)

## Secondary: levels completed (robustness)
- Mean paired lc delta: +0.1229; one-sided p: 0.086444 [exact (16777216 sign assignments)]

## Exclusion / asymmetry checks
- bp35 excluded (pre-registered; absent from our seeds >=2).
- wa30: null levels_completed all integer=True, ours all integer=True; scored offline with identical formula+baselines both arms -> no exclusion needed.
- tr87: null levels_completed all integer=True, ours all integer=True; scored offline with identical formula+baselines both arms -> no exclusion needed.

- Game-version note: 15 games served to us under a different version hash than the null run (ar25, cn04, dc22, ka59, m0r0, r11l, re86, s5i5, sc25, sk48, sp80, su15, tn36, tu93, vc33); scored with null baselines by level index (pre-registered same-baselines-both-arms).
