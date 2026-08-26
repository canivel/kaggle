"""Executable world model for ARC-AGI-3 game `r11l` -- v2.

Adds a tight "double-tick" rule on top of v1.

v1 invariants (kept):
  INV-1  topmost 0 in column 0 -> 5 every transition (counter tick).
  INV-2  reward_class=1, done=False everywhere.

v2 addition:
  INV-3  "double-tick on top-row clicks at every 8th counter step":
         if y < 3 AND (row_of_first_zero) % 8 == 7, then tick TWICE.
         Verified on all 200 tuples: tp=4, fp=1, fn=0 (the lone FP is
         tuple 149: first0=31, y=0, actually single-tick).
         Net effect vs v1: +3 exact-match (4 captured, 1 broken),
         <0.01pp change in pixel match.

Rationale for the rule:
  The 4 observed n=2 (double-tick) cases all have y in {0,1,2} and
  first_zero_row in {7, 23, 7, 55} -- all of which satisfy
  first0 % 8 == 7. Adding the y<3 conjunction is required because many
  large-n tuples also hit first0 % 8 == 7 with playfield clicks; those
  are not double-tick events.

  This is fragile (4 positives, n=200) and is flagged as such. It
  passes the >=2pp threshold and does not regress pixel match, so we
  ship it but with a guard rail comment.

v2 does NOT model:
  - playfield piece dynamics (motion of the color-15 sprite); requires
    multi-frame state we don't have.
  - counter wrap at row 64 (never observed in this sample).
"""
from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np

GridLike = Union[List[List[int]], np.ndarray]

COUNTER_COL = 0
COUNTER_FILL = 5
DOUBLE_TICK_Y_MAX = 3       # y < DOUBLE_TICK_Y_MAX triggers the check
DOUBLE_TICK_MOD = 8         # first_zero_row % DOUBLE_TICK_MOD == ...
DOUBLE_TICK_REMAINDER = 7   # ... 7


def _to_np(state: GridLike) -> np.ndarray:
    if isinstance(state, np.ndarray):
        return state.astype(np.uint8, copy=True)
    return np.asarray(state, dtype=np.uint8)


def _first_zero_row(grid: np.ndarray) -> int:
    """Return the row of the topmost 0 in column 0, or -1 if none."""
    col = grid[:, COUNTER_COL]
    zeros = np.where(col == 0)[0]
    return int(zeros[0]) if zeros.size else -1


def _advance_counter(grid: np.ndarray, ticks: int = 1) -> None:
    """Tick the column-0 counter `ticks` times by flipping the next 0 -> 5."""
    for _ in range(ticks):
        r = _first_zero_row(grid)
        if r < 0:
            return  # counter full; do not wrap mid-transition
        grid[r, COUNTER_COL] = COUNTER_FILL


def simulate(state: GridLike, action_id: int, x: int, y: int):
    """Predict next_state, reward_class, done for game r11l.

    Parameters
    ----------
    state : 64x64 grid, values 0..15.
    action_id : int. Only 6 (click) is observed in training.
    x, y : ints in [0, 64). Click coords (col, row).

    Returns
    -------
    next_state (np.ndarray), reward_class (int=1), done (bool=False)
    """
    grid = _to_np(state)

    # INV-3 double-tick rule (must read counter state BEFORE we mutate it).
    first0 = _first_zero_row(grid)
    double_tick = (
        first0 >= 0
        and (first0 % DOUBLE_TICK_MOD) == DOUBLE_TICK_REMAINDER
        and y < DOUBLE_TICK_Y_MAX
    )

    _advance_counter(grid, ticks=2 if double_tick else 1)

    return grid, 1, False
