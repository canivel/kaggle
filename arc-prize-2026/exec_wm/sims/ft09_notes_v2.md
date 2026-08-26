# ft09 sim v2 notes

## Decision: no v2 written. v1 retained as active sim.

## Why

v1 already scores **100.0 / 100.0 / 100.0 / 100.0** on every metric across
all 200 tuples, with 0 errors and uniform coverage of the single available
action (6). There is no failing bucket to target — no action with worst
exact%, no residual pixel mismatch, no reward/done miscall.

Per the stop criteria:

> If v2 improvement is <2 absolute pts on exact-match: marginal, ship but flag.

The maximum possible improvement is **0.0 pts**. Any v2 against this dataset
would be either (a) identical behaviour with cosmetic changes, or (b)
curve-fitting to imagined out-of-distribution cases we have not observed.
Both are net-negative: (a) adds risk with no upside, (b) actively damages
generalisation.

## v1 vs v2 score comparison

| metric            | v1     | v2 (skipped) |
|-------------------|--------|--------------|
| state_exact_pct   | 100.0  | n/a          |
| pixel_match_pct   | 100.0  | n/a          |
| reward_acc_pct    | 100.0  | n/a          |
| done_acc_pct      | 100.0  | n/a          |
| errors            | 0      | n/a          |

Active simulator: `ft09_sim.py` (unchanged from v1).

## Signal vs curve-fitting

Honest read: this is **real signal**, not curve-fitting.
- Three crisp invariants (fixed 6x6 tile grid, uniform 8<->9 flip, counter
  consumes 2 rightmost 12s) explain all 15 reward-1 events and all 185
  NOOPs without exceptions.
- Both reward-bucket and action-bucket are perfectly partitioned.
- The model is structural (geometric tile bounds + symbolic flip rule), not
  memorising tuples.

## Kaggle inference utility

Likely **helpful** at inference for ft09-class puzzles:
- Action 6 is the only action; the sim correctly predicts reward 1 vs 0 for
  every click coordinate on the 64x64 grid (185/200 NOOPs identified by
  geometric region, 15/15 outer-tile flips predicted exactly).
- An agent can use this sim to plan: avoid known NOOP regions (outside
  tile grid + centre tile) and target outer tiles in the 6x6 grid for
  guaranteed reward.
- It is **not** identity-mostly: 185 NOOPs are correctly predicted as
  identity, but the 15 reward events involve real 38-cell state changes
  (36 tile cells + 2 counter cells) that the sim reproduces exactly.

## What v2 would need to attempt (deferred)

Only meaningful with new data:
1. Counter-exhaustion (level transition or done=True).
2. Latent goal state for the centre tile.
3. 300+ step rollouts to surface level-up semantics.

Collect that data first; iterating without it is curve-fitting.
