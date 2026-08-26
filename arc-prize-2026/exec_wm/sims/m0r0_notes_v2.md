# m0r0 sim v2 notes

## Summary

v2 explored three counter-tick prediction strategies. None beat v1 (one
**regressed by 9 absolute points**, the other two were identical-by-design).
v1 stays active.

## v1 vs v2 score comparison

| metric             | v1     | v2 (any strategy) |
|--------------------|--------|-------------------|
| state_exact_pct    | 57.5   | 48.5 (B), 57.5 (A/C) |
| pixel_match_pct    | 99.979 | 99.979            |
| reward_acc_pct     | 78.0   | 78.0              |
| done_acc_pct       | 100    | 100               |

Active sim: **v1** (`m0r0_sim.py` unchanged). `m0r0_sim_v2.py` is an
exploratory file that re-exports v1.

## What v2 investigated

### Discovery: mod-7 pattern in episode 1

The tick sequence within episode 1 satisfies: `tick at step_idx iff
step_idx mod 7 in {1, 3, 5}`. Verified on all 30+ ticks of episode 1.
Run-length pattern between ticks is `[2, 2, 3, 2, 2, 3, ...]` = 3 ticks
per 7 steps.

### Strategy A: per-counter-value majority vote

For c % 3 == 0: tick:no-tick = 2:4 -> predict no-tick (== v1 behavior).
For c % 3 in {1, 2}: tick:no-tick = exactly 2:2 -> tie -> predict no-tick.
=> Identical to v1.

### Strategy B: predict tick when c % 3 != 0

Tested directly: **state_exact 48.5%** vs v1 57.5%. Regression. The
conditional distribution is genuinely 2:2 not skewed, so any tick
prediction creates new wrong predictions on the no-tick half.

### Strategy C: (action, c % 3) buckets

Strongest skews observed: `(action=6, c%3=0)` -> tick_rate 8/9 = 89%;
`(action=5, c%3=1)` -> tick_rate 6/9 = 67%. Both **below the 90%
invariant threshold** on tiny (9-sample) buckets. Per workflow stop
criterion ("if invariants verify on <90% of target bucket: curve-fitting,
revert"), discarded.

## Why v2 cannot beat v1 statelessly

The mod-7 pattern requires `step_idx`, which is **not recoverable from
state alone**:

1. The counter value c is the number of ticks so far; c = `(3/7) * step`
   approximately. Multiple step indices map to the same c (each c is
   visited 2-3 times before advancing).
2. The 200 tuples span two episodes whose step counters reset, so even
   global step-from-tuple-position would not work for a deployed sim
   that only sees state.
3. Spatial movement (rows 1-62) and counter tick are independent
   (verified): both=41, only-spatial=67, only-tick=44, neither=48.
   So spatial move success doesn't predict ticks either.

## Honest signal vs curve-fitting

The mod-7 pattern IS real signal (verified on every observed tick of
episode 1). But it can only be exploited with a **stateful API** that
tracks step_idx across calls. In the stateless `simulate(state, action,
x, y)` contract, v1 is at the theoretical ceiling.

The 89% bucket on `(action=6, c%3=0)` is curve-fitting on 9 samples.
Deploying it would risk private-test regression.

## Kaggle inference relevance

The m0r0 sim is **not identity-mostly**: actions 1-4 produce real spatial
state changes that v1 predicts perfectly (131/131 movement transitions
correct). For agents that use it for planning (Go-Explore, MCTS rollout)
the spatial model is fully usable. The 2-pixel counter-tick noise is
benign for spatial planning since it never affects move legality and is
isolated to two rows the agent never needs to reason about.

For reward-shaping rollouts, v1's 78% reward accuracy is a slight
under-prediction (we say "no change" when counter actually ticked) but
is consistent (never claims false reward).

Net: **v1 is shippable for downstream planning.** Counter-tick is
genuine irreducible single-frame noise.
