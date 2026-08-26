# Phase-1 A/B gate report — **FINAL** (3/3 seeds)

## Scorer validation (hard requirement)
- Reproduced Tufa score.json + per-run final_score from raw benchmark.json: **max abs error 0.000e+00** over 1000 cross-checks (500 runs, 25 game means, overall). PASS (limit 1e-9).
- Recomputed overall null score 1.600204 vs published 1.600204.
- Formula: per level i (0-idx, weight i+1): min(115, (base/actions)^2*100) if completed & actions>0 else 0; score = weighted mean over ALL level weights, capped at (weights of scoring levels)/(all weights)*100.

## Arms
- Null: runs/tufa_example_run/benchmark.json — 25 games x 20 vanilla passes.
- Ours seed 1: runs/phase1_ab/phase1_seed1.json (detail: runs\phase1_ab\seed1\benchmark.json, 25 games)
- Ours seed 2: runs/phase1_ab/phase1_seed2.json (detail: runs\phase1_ab\seed2\benchmark.json, 24 games)
- Ours seed 3: runs/phase1_ab/phase1_seed3.json (detail: runs\phase1_ab\seed3\benchmark.json, 24 games)
- Both arms scored OFFLINE with identical formula and the null arm's base_actions_per_level (joined on 4-char game prefix).

## Per-game paired deltas (RHAE, Kaggle-comparable units)

| game | null mean (20p) | ours mean | delta | our lc | null lc mean | flags |
|---|---|---|---|---|---|---|
| ar25 | 1.7470 | 0.0998 | -1.6472 | 1,1,1 | 0.95 | diff-version |
| bp35 | 0.3137 | — | — | — | 0.90 | **EXCLUDED: pre-registered (flaky arcade seeds 2-3)** |
| cd82 | 0.7752 | 1.5873 | +0.8121 | 0,1,0 | 0.25 |  |
| cn04 | 0.2130 | 1.6167 | +1.4037 | 0,0,1 | 0.15 | diff-version; nlev 5 vs 6 |
| dc22 | 0.2008 | 0.0000 | -0.2008 | 0,0,0 | 0.10 | diff-version |
| ft09 | 10.2830 | 9.5238 | -0.7592 | 0,2,2 | 1.50 |  |
| g50t | 0.1341 | 0.0000 | -0.1341 | 0,0,0 | 0.05 |  |
| ka59 | 1.3382 | 0.3085 | -1.0297 | 0,0,1 | 0.55 | diff-version |
| lf52 | 1.2346 | 1.3845 | +0.1499 | 1,1,1 | 0.75 |  |
| lp85 | 2.7242 | 3.2956 | +0.5714 | 1,1,2 | 1.00 |  |
| ls20 | 0.3537 | 0.7904 | +0.4367 | 1,0,0 | 0.20 |  |
| m0r0 | 0.0477 | 0.1202 | +0.0725 | 1,0,0 | 0.05 | diff-version |
| r11l | 3.7076 | 3.1746 | -0.5330 | 1,1,0 | 0.90 | diff-version |
| re86 | 1.9125 | 2.5594 | +0.6468 | 2,1,1 | 0.90 | diff-version |
| s5i5 | 0.2583 | 0.0000 | -0.2583 | 0,0,0 | 0.30 | diff-version |
| sb26 | 2.8339 | 2.7778 | -0.0561 | 1,1,1 | 1.10 |  |
| sc25 | 0.1492 | 3.2566 | +3.1074 | 2,1,1 | 0.15 | diff-version |
| sk48 | 0.2778 | 0.9259 | +0.6481 | 0,1,0 | 0.10 | diff-version |
| sp80 | 1.8289 | 0.1533 | -1.6756 | 0,1,1 | 0.65 | diff-version |
| su15 | 2.0842 | 1.9921 | -0.0921 | 1,1,1 | 1.00 | diff-version |
| tn36 | 2.0811 | 7.3282 | +5.2472 | 2,1,2 | 0.65 | diff-version |
| tr87 | 0.0000 | 0.0000 | +0.0000 | 0,0,0 | 0.00 |  |
| tu93 | 2.1714 | 0.9989 | -1.1726 | 2,2,1 | 1.65 | diff-version |
| vc33 | 3.3350 | 1.8616 | -1.4733 | 2,1,2 | 1.60 | diff-version |
| wa30 | 0.0000 | 0.0000 | +0.0000 | 0,0,0 | 0.00 |  |

## Primary gate (RHAE)
- Included games: n = 24
- Mean RHAE, included games: ours 1.8231 vs null 1.6538
- Mean RHAE, ours all 25 games (their-1.6002-scale): 1.7687 (null all-25 reference: 1.6002)
- Mean paired delta: **+0.1693**
- One-sided sign-flip permutation p (improvement): **0.308052** [exact (16777216 sign assignments)]
- Alpha: 0.0125

## VERDICT: **FAIL**

## Secondary: levels completed (robustness)
- Mean paired lc delta: +0.1299; one-sided p: 0.040753 [exact (16777216 sign assignments)]

## Exclusion / asymmetry checks
- bp35 excluded (pre-registered; absent from our seeds >=2).
- wa30: null levels_completed all integer=True, ours all integer=True; scored offline with identical formula+baselines both arms -> no exclusion needed.
- tr87: null levels_completed all integer=True, ours all integer=True; scored offline with identical formula+baselines both arms -> no exclusion needed.

- Game-version note: 15 games served to us under a different version hash than the null run (ar25, cn04, dc22, ka59, m0r0, r11l, re86, s5i5, sc25, sk48, sp80, su15, tn36, tu93, vc33); scored with null baselines by level index (pre-registered same-baselines-both-arms).
