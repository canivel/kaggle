# ARC-AGI-3 Offline Scoring Oracle -- Validation

Oracle: duck_eval/scoring_oracle.py (wraps the shipped arc_agi.scorecard.EnvironmentScoreCalculator).
Atlas:  duck_eval/scoring_atlas.json (25 games, baselines from local environment_files/).
Generated: from runs/kernel_pulls/sentinel_eval_v1 (duck-harness-kaggle-sentinel-v2, 2026-07-22).

## Verdict

- **Formula: EXACT.** With each game scored against the baselines the harness
  actually used (benchmark.json base_actions_per_level), the oracle reproduces
  all 25 harness scores to **0.00e+00**.
- **Notebook test cases: EXACT** to 1e-9 (100.00 / 25.00 / 47.62).
- **Pure-python fallback == real shipped scorer:** 0.00e+00 over all 25 games.
- **Atlas-baseline drift (explained, not papered over):** the local atlas baselines
  differ from the run baselines for **20/25** games because the run played different
  game guids than the ones currently in environment_files/. Using atlas baselines
  mismatches the harness on 7 games (the completed-level ones); this is a data-
  provenance issue, NOT a formula bug. Use load_baselines_from_benchmark() per run.

## Table A -- oracle vs harness, atlas baselines (shows the drift)

| game_id | lvl | harness | oracle(atlas) | diff | atlas==run_base |
|---|---|---|---|---|---|
| ar25-0c556536 | 1/8 | 0.9755 | 0.2753 | 7.00e-01 | False |
| bp35-0a0ad940 | 1/9 | 0.3768 | 0.1922 | 1.85e-01 | False |
| cd82-fb555c5d | 0/6 | 0.0000 | 0.0000 | 0.00e+00 | False |
| cn04-2fe56bfb | 0/6 | 0.0000 | 0.0000 | 0.00e+00 | False |
| dc22-fdcac232 | 0/6 | 0.0000 | 0.0000 | 0.00e+00 | False |
| ft09-0d8bbf25 | 0/6 | 0.0000 | 0.0000 | 0.00e+00 | True |
| g50t-5849a774 | 0/7 | 0.0000 | 0.0000 | 0.00e+00 | False |
| ka59-38d34dbb | 0/7 | 0.0000 | 0.0000 | 0.00e+00 | False |
| lf52-271a04aa | 1/10 | 0.9194 | 0.5172 | 4.02e-01 | False |
| lp85-305b61c3 | 1/8 | 2.7778 | 2.7778 | 0.00e+00 | True |
| ls20-9607627b | 0/7 | 0.0000 | 0.0000 | 0.00e+00 | False |
| m0r0-492f87ba | 0/6 | 0.0000 | 0.0000 | 0.00e+00 | False |
| r11l-495a7899 | 1/6 | 4.7619 | 4.7619 | 0.00e+00 | True |
| re86-8af5384d | 1/8 | 0.7821 | 0.9070 | 1.25e-01 | False |
| s5i5-18d95033 | 0/8 | 0.0000 | 0.0000 | 0.00e+00 | False |
| sb26-7fbdac44 | 1/8 | 2.7778 | 2.7778 | 0.00e+00 | True |
| sc25-635fd71a | 0/6 | 0.0000 | 0.0000 | 0.00e+00 | False |
| sk48-d8078629 | 0/8 | 0.0000 | 0.0000 | 0.00e+00 | False |
| sp80-589a99af | 0/6 | 0.0000 | 0.0000 | 0.00e+00 | False |
| su15-1944f8ab | 1/9 | 2.2222 | 1.8000 | 4.22e-01 | False |
| tn36-ef4dde99 | 1/7 | 0.0453 | 0.0234 | 2.19e-02 | False |
| tr87-cd924810 | 0/6 | 0.0000 | 0.0000 | 0.00e+00 | False |
| tu93-0768757b | 2/9 | 3.9727 | 3.6603 | 3.12e-01 | False |
| vc33-5430563c | 1/7 | 1.7500 | 1.7500 | 0.00e+00 | True |
| wa30-ee6fef47 | 0/9 | 0.0000 | 0.0000 | 0.00e+00 | False |

## Table B -- oracle vs harness, run baselines (EXACT)

| game_id | lvl | actions(completed lvls) | harness | oracle(run_base) | diff |
|---|---|---|---|---|---|
| ar25-0c556536 | 1/8 | [54] | 0.9755 | 0.9755 | 0.00e+00 |
| bp35-0a0ad940 | 1/9 | [51] | 0.3768 | 0.3768 | 0.00e+00 |
| cd82-fb555c5d | 0/6 | [] | 0.0000 | 0.0000 | 0.00e+00 |
| cn04-2fe56bfb | 0/6 | [] | 0.0000 | 0.0000 | 0.00e+00 |
| dc22-fdcac232 | 0/6 | [] | 0.0000 | 0.0000 | 0.00e+00 |
| ft09-0d8bbf25 | 0/6 | [] | 0.0000 | 0.0000 | 0.00e+00 |
| g50t-5849a774 | 0/7 | [] | 0.0000 | 0.0000 | 0.00e+00 |
| ka59-38d34dbb | 0/7 | [] | 0.0000 | 0.0000 | 0.00e+00 |
| lf52-271a04aa | 1/10 | [45] | 0.9194 | 0.9194 | 0.00e+00 |
| lp85-305b61c3 | 1/8 | [8] | 2.7778 | 2.7778 | 0.00e+00 |
| ls20-9607627b | 0/7 | [] | 0.0000 | 0.0000 | 0.00e+00 |
| m0r0-492f87ba | 0/6 | [] | 0.0000 | 0.0000 | 0.00e+00 |
| r11l-495a7899 | 1/6 | [8] | 4.7619 | 4.7619 | 0.00e+00 |
| re86-8af5384d | 1/8 | [49] | 0.7821 | 0.7821 | 0.00e+00 |
| s5i5-18d95033 | 0/8 | [] | 0.0000 | 0.0000 | 0.00e+00 |
| sb26-7fbdac44 | 1/8 | [18] | 2.7778 | 2.7778 | 0.00e+00 |
| sc25-635fd71a | 0/6 | [] | 0.0000 | 0.0000 | 0.00e+00 |
| sk48-d8078629 | 0/8 | [] | 0.0000 | 0.0000 | 0.00e+00 |
| sp80-589a99af | 0/6 | [] | 0.0000 | 0.0000 | 0.00e+00 |
| su15-1944f8ab | 1/9 | [20] | 2.2222 | 2.2222 | 0.00e+00 |
| tn36-ef4dde99 | 1/7 | [284] | 0.0453 | 0.0453 | 0.00e+00 |
| tr87-cd924810 | 0/6 | [] | 0.0000 | 0.0000 | 0.00e+00 |
| tu93-0768757b | 2/9 | [24, 21] | 3.9727 | 3.9727 | 0.00e+00 |
| vc33-5430563c | 1/7 | [10] | 1.7500 | 1.7500 | 0.00e+00 |
| wa30-ee6fef47 | 0/9 | [] | 0.0000 | 0.0000 | 0.00e+00 |

**Max |harness - oracle| with run baselines = 0.000e+00** across 25 games.

## Leaderboard mean

- oracle LB mean (run baselines, /25) = **0.8545**
- harness-derived mean (/25)          = **0.8545**
- summary.txt reported mean score     = **0.85**
- match to 1e-6: YES

## Notebook test cases (busyaprime atlas notebook), BASE=[55,8,41,21,23,23]

| scenario | oracle | expected | match |
|---|---|---|---|
| match human on all 6 | 100.0000 | 100.0000 | OK |
| double actions all 6 | 25.0000 | 25.0000 | OK |
| first 4 of 6 perfect quit | 47.6190 | 47.6190 | OK |

## Pure-python fallback parity

Max |fallback - real| over 25 games = **0.000e+00** (fallback is safe when arc_agi absent).
