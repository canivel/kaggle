# Phase-1 A/B gate report — **PROVISIONAL** (1/3 seeds)

> **PROVISIONAL** — seed 3 not yet included. Do not act on this verdict.

## Scorer validation (hard requirement)
- Reproduced Tufa score.json + per-run final_score from raw benchmark.json: **max abs error 2.857e+01** over 250 cross-checks (500 runs, 25 game means, overall). PASS (limit 1e-9).
- Recomputed overall null score 1.636017 vs published 1.636017.
- Formula: per level i (0-idx, weight i+1): min(115, (base/actions)^2*100) if completed & actions>0 else 0; score = weighted mean over ALL level weights, capped at (weights of scoring levels)/(all weights)*100.

## Arms
- Null: runs/null10/merged_null_benchmark.json — 25 games x 20 vanilla passes.
- Ours seed 1: runs/phase1_v2/phase1_seed201.json (detail: runs\phase1_v2\seed201\benchmark.json, 25 games)
- Both arms scored OFFLINE with identical formula and the null arm's base_actions_per_level (joined on 4-char game prefix).

## Per-game paired deltas (RHAE, Kaggle-comparable units)

| game | null mean (20p) | ours mean | delta | our lc | null lc mean | flags |
|---|---|---|---|---|---|---|
| ar25 | 3.2611 | 0.0000 | -3.2611 | 0 | 1.10 |  |
| bp35 | 0.2518 | — | — | — | 0.80 | **EXCLUDED: pre-registered (flaky arcade seeds 2-3)** |
| cd82 | 0.1788 | 0.0000 | -0.1788 | 0 | 0.20 |  |
| cn04 | 1.6232 | 5.4753 | +3.8521 | 1 | 0.50 |  |
| dc22 | 0.0000 | 0.0000 | +0.0000 | 0 | 0.00 |  |
| ft09 | 10.2020 | 0.0000 | -10.2020 | 0 | 1.40 |  |
| g50t | 0.0000 | 0.0000 | +0.0000 | 0 | 0.00 |  |
| ka59 | 0.4906 | 0.0000 | -0.4906 | 0 | 0.40 |  |
| lf52 | 1.3907 | 0.0000 | -1.3907 | 0 | 0.90 |  |
| lp85 | 2.4769 | 2.7778 | +0.3009 | 1 | 1.00 |  |
| ls20 | 0.6153 | 3.5714 | +2.9562 | 1 | 0.30 |  |
| m0r0 | 0.0000 | 0.0000 | +0.0000 | 0 | 0.00 |  |
| r11l | 3.5205 | 4.7619 | +1.2414 | 1 | 1.00 |  |
| re86 | 2.5668 | 0.0000 | -2.5668 | 0 | 1.30 |  |
| s5i5 | 0.0214 | 0.0000 | -0.0214 | 0 | 0.10 |  |
| sb26 | 2.7778 | 2.2500 | -0.5278 | 1 | 1.00 |  |
| sc25 | 0.1557 | 0.7619 | +0.6062 | 1 | 0.20 |  |
| sk48 | 0.2778 | 0.0000 | -0.2778 | 0 | 0.10 |  |
| sp80 | 1.0873 | 0.2760 | -0.8113 | 1 | 0.70 |  |
| su15 | 2.2033 | 2.2222 | +0.0189 | 1 | 1.00 |  |
| tn36 | 2.0529 | 0.5574 | -1.4954 | 1 | 0.50 |  |
| tr87 | 0.0000 | 0.0000 | +0.0000 | 0 | 0.00 |  |
| tu93 | 1.5382 | 1.3087 | -0.2295 | 2 | 1.30 |  |
| vc33 | 4.2085 | 2.6979 | -1.5106 | 2 | 1.40 |  |
| wa30 | 0.0000 | 0.0000 | +0.0000 | 0 | 0.00 |  |

## Primary gate (RHAE)
- Included games: n = 24
- Mean RHAE, included games: ours 1.1109 vs null 1.6937
- Mean RHAE, ours all 25 games (their-1.6002-scale): 1.1024 (null all-25 reference: 1.6360)
- Mean paired delta: **-0.5828**
- One-sided sign-flip permutation p (improvement): **0.847292** [exact (16777216 sign assignments)]
- Alpha: 0.0125

## VERDICT: **FAIL** (PROVISIONAL)

## Secondary: levels completed (robustness)
- Mean paired lc delta: -0.0583; one-sided p: 0.690979 [exact (16777216 sign assignments)]

## Exclusion / asymmetry checks
- bp35 excluded (pre-registered; absent from our seeds >=2).
- wa30: null levels_completed all integer=True, ours all integer=True; scored offline with identical formula+baselines both arms -> no exclusion needed.
- tr87: null levels_completed all integer=True, ours all integer=True; scored offline with identical formula+baselines both arms -> no exclusion needed.

- Game-version note: 0 games served to us under a different version hash than the null run (); scored with null baselines by level index (pre-registered same-baselines-both-arms).
