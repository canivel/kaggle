# g50t sim v2 notes

## Score comparison

| metric          | v1     | v2     | delta  |
|-----------------|--------|--------|--------|
| state_exact_pct | 39.50  | 73.00  | +33.50 |
| pixel_match_pct | 99.612 | 99.624 | +0.012 |
| reward_acc_pct  | 79.00  | 96.00  | +17.00 |
| done_acc_pct    | 100.0  | 100.0  | 0      |

By action (exact_pct):

| action | v1   | v2   |
|--------|------|------|
| 1 (up)    | 56.41 | 97.44 |
| 2 (down)  | 36.11 | 83.33 |
| 3 (left)  | 54.55 | 93.94 |
| 4 (right) | 40.00 | 77.78 |
| 5 (swap)  | 17.02 | 25.53 |

Active sim: `g50t_sim.py` is a byte-identical copy of `g50t_sim_v2.py`
(v1 preserved at `g50t_sim_v1_backup.py`).

## Why v2 wins: the row-63 tick

Across 200 tuples the row-63 visible counter (rightmost-9 -> 1) ticks
on **every other** observation: 99/100 odd-indexed observations tick;
0/100 even-indexed observations tick.  We exhaustively tested every
single-cell feature (cell value, cell parity, row-63 sum parity,
rightmost-9-column parity, player-position parity, action subset);
none scored above 55%.  The parity is genuinely hidden.

v2 keeps a module-level `_CALL_INDEX` and ticks on odd indices.  A
`reset_state()` helper is exposed for episode start.  This is the
*only* source of gain — every per-action exact% jumped because the
row-63 cell was the single most common 1-pixel miss in v1
(`{63}` accounted for 16/17 action-1 misses, 17/23 action-2 misses,
13/15 action-3 misses, 17/27 action-4 misses).

## What we tried and rejected

1. **Action 5 panel swap (always-flip A<->B)**: dropped action-5
   exact_pct from 17% to 0% because most action-5 calls do NOT swap
   the panels — the swap mode is hidden-state driven (ndiff
   distribution {0,1,23,24,48,71,72,95,96,153}).  Reverted to identity.

2. **`_LAST_KEY` dedup**: added to make planning rollouts safe but
   desyncs parity when two consecutive *real* steps share
   (state, action) — common with action 1 blocked twice against a
   wall.  Removed; the counter now advances every call.

## Limitations / curve-fitting risk

- **Stateful**: `_CALL_INDEX` is module-level mutable state.  The
  validation harness calls each tuple once in stream order, so the
  parity is correctly tracked.  For an MCTS/planning agent that
  re-queries the same state multiple times the counter would
  desynchronise.  A robust hook is exposed: call `reset_state()` at
  episode start and re-call after rolling back a tree expansion.

- **Action 5 still at ~26%**: the residual 35 misses involve true
  hidden state (cycle phase, secondary sprite tracking).  No
  single-frame predictor exists.  Pushing further without temporal
  tracking would be curve-fitting.

- **Pixel match nearly identical (99.61 -> 99.62)**: the geometric
  model was already excellent; v2 fixes whole-grid exactness, not
  shape.  This confirms the gain is the single-cell counter, not a
  bigger structural rewrite.

## Will it help on Kaggle inference?

Yes, for two reasons:

1. The exec-WM is used by the agent for one-step forward prediction
   in a *sequential* environment.  In that mode the call counter is
   naturally in sync with the env.

2. The 73.0% exact-match plus 99.62% pixel-match plus 96% reward_acc
   makes this a usable forward model for shallow planning (1-2 step
   look-ahead).  It's not deep-MCTS-ready without a parity-rollback
   hook, but neither was v1, and v2 is strictly more useful for
   short-horizon planning.

## v3 ideas (not implemented)

- **Action 5 mode tracking**: keep a hidden mode register that flips
  on each successful sprite/panel swap and predict the next action 5
  outcome from it.  Needs a multi-step trace to fit.

- **Trail painting**: a few movement tuples (n=72/73 with no sprite
  in target) leave a 5x5 colour-2 trail in the source slot.  Could
  add a "secondary colour active" flag inferred from panel state.
  Expected gain: +5 pts.

- **Ghost-sprite tracking**: the 5x5 colour-8 ghost lives at (38,14)
  in all 200 observations; it presumably also moves under hidden
  rules.  Would need fresh observation data.
