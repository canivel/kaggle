"""Executable world model for ARC-AGI-3 game `cd82` -- v1.

Rodionov-style (arXiv:2605.05138). Hand-derived from 200 observed tuples.

Game description (from observations):
- 64x64 grid, palette {0,2,3,4,5,15} dominant.
- Two distinct objects sit in the playfield:
    * A small "5/15"-textured frame at fixed rows 8 / 25-32 area (cosmetic;
      never observed to move).
    * A larger "2"-bordered rectangle with a "15"-filled interior that can
      ROTATE between an axis-aligned orientation (rows 24-32 cols 25-39
      style) and a diagonal/rotated orientation.  Actions 1-4 perform
      large rotations affecting ~200 cells.
- A "0"-filled paintable region (rows 34-43 cols 27-36 style).  Action 5
  paints those 0s to 15s in a sub-region (target choice unclear from a
  single frame).
- Row 63 is a counter: it starts as all 4s and rightmost 4 flips to 5
  each time the engine "ticks".  Whether the engine ticks on a given
  action is governed by HIDDEN state and is not predictable from one
  frame.
- 6 available actions: {1,2,3,4,5,6}.  Reward distribution:
    reward_class = 1  iff the state actually changed (n_changed > 0)
    reward_class = 0  iff the action was a NOOP (n_changed == 0)
- done = False, level = 0 across all observed tuples.

Empirical breakdown of `n_changed` over 200 tuples:
    n=0     46     NOOP, reward 0, no tick
    n=1     74     counter tick only (rightmost row-63 `4` -> `5`)
    n=200   24     object rotation, NO counter tick
    n=201   49     object rotation + counter tick
    n in {11,16,21,26,50,55}  action-5 paint (0->15) on partial 0-region

Why v1 is a "tick + identity" baseline:
- The COUNTER tick is the single biggest deterministic signal we can
  always predict correctly.  Always-tick matches all 74 n=1 cases and
  the 49 n=201 row-63 component.
- Rotation (actions 1-4) cycles a 2-bordered rectangle between a
  canonical axis-aligned and a diagonal pose.  We do not yet have a
  unique anchor or template to encode the rotation deterministically
  from a single frame (the "1-step rotation" hypothesis is testable
  but risky); a wrong rotation hurts pixel match more than it helps.
- Action 5 (paint) requires hidden target-region state.
- Click (action 6) is a 50/50 tick-vs-noop with no observable signal.

So v1 invariants:
  1. reward_class = 1 ALWAYS (matches 154/200 = 77%).
  2. done = False ALWAYS (matches 200/200).
  3. Counter: rightmost cell on row 63 with value 4 flips to 5.
     If row 63 contains no 4, leave it alone.
  4. Playfield: untouched (identity outside row 63).

Expected metrics (estimated):
  state_exact_pct ~ 37% (the 74 n=1 cases) + a bit if a few n=0 cases
    happen to also have full row-63 (none observed though).
  pixel_match_pct ~ 99%+  (only the playfield rotation rows ever change
    in the truth and we conservatively leave them alone, so we miss
    ~200 cells on ~73 cases = 14600/(200*4096) ~ 1.8% loss).
  reward_acc_pct ~ 77%.
  done_acc_pct = 100%.
"""
from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np

GridLike = Union[List[List[int]], np.ndarray]

COUNTER_ROW = 63
COUNTER_FILLED = 4   # initial / "not yet ticked" cell value on the counter row
COUNTER_TICKED = 5   # value after the cell has ticked


def _to_np(state: GridLike) -> np.ndarray:
    if isinstance(state, np.ndarray):
        return state.astype(np.uint8, copy=True)
    return np.asarray(state, dtype=np.uint8)


def _tick_counter(grid: np.ndarray) -> bool:
    """Flip the RIGHTMOST cell on row 63 that equals 4 to 5.

    Returns True if a tick was applied, False if row 63 had no 4s
    (counter exhausted) and the grid is unchanged.
    """
    row = grid[COUNTER_ROW]
    fours = np.where(row == COUNTER_FILLED)[0]
    if fours.size == 0:
        return False
    # rightmost 4
    col = int(fours[-1])
    grid[COUNTER_ROW, col] = COUNTER_TICKED
    return True


def simulate(state: GridLike, action_id: int, x: int, y: int):
    """Predict next state, reward_class, done for game cd82.

    Parameters
    ----------
    state : 64x64 grid (list[list[int]] or np.ndarray), values 0-15.
    action_id : int in {1,2,3,4,5,6}.
    x, y : ints in [0, 64). Only relevant when action_id == 6 (click).
    """
    grid = _to_np(state)

    # Single deterministic rule we trust from one frame: counter tick.
    # All 6 actions tick the counter at roughly 60% of observations and
    # never produce any other counter-row change, so an unconditional
    # tick is the highest-EV prediction for row 63.
    _tick_counter(grid)

    # Playfield: we cannot reliably predict rotation / paint / click
    # consequences from a single frame.  Leave untouched.

    # Reward heuristic: predict 1 (the modal outcome, 154/200).
    reward_class = 1
    done = False
    return grid, reward_class, done
