# Phase-1 A/B gate report — **FINAL** (3/3 seeds)

## Scorer validation (hard requirement)
- Reproduced Tufa score.json + per-run final_score from raw benchmark.json: **max abs error 2.857e+01** over 250 cross-checks (500 runs, 25 game means, overall). PASS (limit 1e-9).
- Recomputed overall null score 1.636017 vs published 1.636017.
- Formula: per level i (0-idx, weight i+1): min(115, (base/actions)^2*100) if completed & actions>0 else 0; score = weighted mean over ALL level weights, capped at (weights of scoring levels)/(all weights)*100.

## Arms
- Null: runs/null10/merged_null_benchmark.json — 25 games x 20 vanilla passes.
- Ours seed 1: runs/phase1_v2/phase1_seed201.json (detail: runs\phase1_v2\seed201\benchmark.json, 25 games)
- Ours seed 2: runs/phase1_v2/phase1_seed202.json (detail: runs\phase1_v2\seed202\benchmark.json, 25 games)
- Ours seed 3: runs/phase1_v2/phase1_seed203.json (detail: runs\phase1_v2\seed203\benchmark.json, 25 games)
- Both arms scored OFFLINE with identical formula and the null arm's base_actions_per_level (joined on 4-char game prefix).

## Per-game paired deltas (RHAE, Kaggle-comparable units)

| game | null mean (20p) | ours mean | delta | our lc | null lc mean | flags |
|---|---|---|---|---|---|---|
| ar25 | 3.2611 | 0.2315 | -3.0296 | 0,0,1 | 1.10 |  |
| bp35 | 0.2518 | — | — | — | 0.80 | **EXCLUDED: pre-registered (flaky arcade seeds 2-3)** |
| cd82 | 0.1788 | 0.0000 | -0.1788 | 0,0,0 | 0.20 |  |
| cn04 | 1.6232 | 2.6939 | +1.0707 | 1,1,1 | 0.50 |  |
| dc22 | 0.0000 | 0.0000 | +0.0000 | 0,0,0 | 0.00 |  |
| ft09 | 10.2020 | 2.8620 | -7.3400 | 0,2,0 | 1.40 |  |
| g50t | 0.0000 | 0.0000 | +0.0000 | 0,0,0 | 0.00 |  |
| ka59 | 0.4906 | 1.1098 | +0.6192 | 0,1,0 | 0.40 |  |
| lf52 | 1.3907 | 0.8270 | -0.5637 | 0,1,1 | 0.90 |  |
| lp85 | 2.4769 | 2.7778 | +0.3009 | 1,1,1 | 1.00 |  |
| ls20 | 0.6153 | 1.8307 | +1.2154 | 1,0,1 | 0.30 |  |
| m0r0 | 0.0000 | 0.0000 | +0.0000 | 0,0,0 | 0.00 |  |
| r11l | 3.5205 | 4.7619 | +1.2414 | 1,1,1 | 1.00 |  |
| re86 | 2.5668 | 1.0402 | -1.5265 | 0,1,1 | 1.30 |  |
| s5i5 | 0.0214 | 0.0000 | -0.0214 | 0,0,0 | 0.10 |  |
| sb26 | 2.7778 | 1.8129 | -0.9649 | 1,1,1 | 1.00 |  |
| sc25 | 0.1557 | 0.6417 | +0.4859 | 1,0,2 | 0.20 |  |
| sk48 | 0.2778 | 0.0000 | -0.2778 | 0,0,0 | 0.10 |  |
| sp80 | 1.0873 | 1.7600 | +0.6727 | 1,1,1 | 0.70 |  |
| su15 | 2.2033 | 1.6193 | -0.5840 | 1,1,1 | 1.00 |  |
| tn36 | 2.0529 | 0.4084 | -1.6444 | 1,1,0 | 0.50 |  |
| tr87 | 0.0000 | 0.0000 | +0.0000 | 0,0,0 | 0.00 |  |
| tu93 | 1.5382 | 1.7632 | +0.2250 | 2,0,2 | 1.30 |  |
| vc33 | 4.2085 | 1.5037 | -2.7048 | 2,1,1 | 1.40 |  |
| wa30 | 0.0000 | 0.0000 | +0.0000 | 0,0,0 | 0.00 |  |

## Primary gate (RHAE)
- Included games: n = 24
- Mean RHAE, included games: ours 1.1518 vs null 1.6937
- Mean RHAE, ours all 25 games (their-1.6002-scale): 1.1218 (null all-25 reference: 1.6360)
- Mean paired delta: **-0.5419**
- One-sided sign-flip permutation p (improvement): **0.923170** [exact (16777216 sign assignments)]
- Alpha: 0.0125

## VERDICT: **FAIL**

## Secondary: levels completed (robustness)
- Mean paired lc delta: -0.0306; one-sided p: 0.670166 [exact (16777216 sign assignments)]

## Exclusion / asymmetry checks
- bp35 excluded (pre-registered; absent from our seeds >=2).
- wa30: null levels_completed all integer=True, ours all integer=True; scored offline with identical formula+baselines both arms -> no exclusion needed.
- tr87: null levels_completed all integer=True, ours all integer=True; scored offline with identical formula+baselines both arms -> no exclusion needed.

- Game-version note: 0 games served to us under a different version hash than the null run (); scored with null baselines by level index (pre-registered same-baselines-both-arms).
