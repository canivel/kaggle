# Determinism audit — all 25 official games (panel R12 N5)

Property tested: would `bank_strict` replay survive? Same env, consecutive
plays (new play opened exactly as `_bank` does), identical action sequence,
per-step strict comparison of final-frame grid hash, levels_completed and
engine state, including the initial post-reset frame. Probe A replays the
game's real war-eval recorded history (board-changing, level-completing);
probe B a fixed seeded 48-action script. DIVERGENT if either
probe mismatches anywhere.

- **Divergent fraction: 0/25 = 0.0**
- Deterministic (25): ar25, bp35, cd82, cn04, dc22, ft09, g50t, ka59, lf52, lp85, ls20, m0r0, r11l, re86, s5i5, sb26, sc25, sk48, sp80, su15, tn36, tr87, tu93, vc33, wa30
- Divergent (0): -
- Untestable (0): -

| game | verdict | first divergence | fields | hist actions | hist lc p1/p2 (recorded) |
|---|---|---|---|---|---|
| ar25 | DETERMINISTIC | - | - | 201 | 2/2 (2) |
| bp35 | DETERMINISTIC | - | - | 123 | 1/1 (1) |
| cd82 | DETERMINISTIC | - | - | 141 | 0/0 (0) |
| cn04 | DETERMINISTIC | - | - | 304 | 1/1 (0) |
| dc22 | DETERMINISTIC | - | - | 61 | 0/0 (0) |
| ft09 | DETERMINISTIC | - | - | 45 | 2/2 (2) |
| g50t | DETERMINISTIC | - | - | 65 | 0/0 (0) |
| ka59 | DETERMINISTIC | - | - | 122 | 1/1 (1) |
| lf52 | DETERMINISTIC | - | - | 131 | 1/1 (1) |
| lp85 | DETERMINISTIC | - | - | 72 | 1/1 (1) |
| ls20 | DETERMINISTIC | - | - | 91 | 1/1 (1) |
| m0r0 | DETERMINISTIC | - | - | 323 | 1/1 (1) |
| r11l | DETERMINISTIC | - | - | 81 | 1/1 (1) |
| re86 | DETERMINISTIC | - | - | 163 | 2/2 (2) |
| s5i5 | DETERMINISTIC | - | - | 69 | 1/1 (1) |
| sb26 | DETERMINISTIC | - | - | 256 | 1/1 (1) |
| sc25 | DETERMINISTIC | - | - | 235 | 2/2 (2) |
| sk48 | DETERMINISTIC | - | - | 266 | 0/0 (0) |
| sp80 | DETERMINISTIC | - | - | 240 | 1/1 (1) |
| su15 | DETERMINISTIC | - | - | 178 | 1/1 (1) |
| tn36 | DETERMINISTIC | - | - | 116 | 1/1 (1) |
| tr87 | DETERMINISTIC | - | - | 254 | 0/0 (0) |
| tu93 | DETERMINISTIC | - | - | 154 | 2/2 (2) |
| vc33 | DETERMINISTIC | - | - | 88 | 2/2 (2) |
| wa30 | DETERMINISTIC | - | - | 198 | 0/0 (0) |

## R2 reach re-base (banking rows restricted to the deterministic subset)

- warpack Δlc-positive games (both screens): ft09, ka59, re86, sc25, tu93, sb26, su15, lp85
- **bankable (deterministic)**: ft09, ka59, re86, sc25, tu93, sb26, su15, lp85
- **banking-inert (divergent)**: NONE

## Reconciliation with bank_fire_validation.json (sc25/m0r0 aborts)

The step-0 `frame_divergence` aborts observed on sc25/m0r0 were NOT
per-play randomization. `prune_replay_diag.py` reproduces them on these
same deterministic engines: `prune_trace` drops 1-2 leading recorded
actions whose visible frame did not change (board_changed=False) but
which mutate hidden state, so the pruned replay's first action lands on
a different frame. Replaying the FULL unpruned history survives on all
25 games (probe A above) and reproduces the recorded levels_completed
on the new play. Banking as implemented is inert on such games due to
its pruning, not the environment; an unpruned (or trailing-only-pruned)
replay would be viable panel-wide.
(Evidence: runs/war_eval_v1/prune_replay_diag.json)

## Caveats

- Local env versions differ from the war-eval Kaggle build for some
  games (e.g. sc25 local f9b21a2f vs kaggle 635fd71a; flagged per game
  as version_mismatch_vs_kaggle in the JSON). Determinism is verified
  on the local engines — the same engines bank_fire_validation ran on.
- Two consecutive plays per probe; probe A exercises real
  board-changing, level-completing dynamics.
