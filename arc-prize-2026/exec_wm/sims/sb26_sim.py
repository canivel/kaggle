"""Executable world model for ARC-AGI-3 game `sb26` — v1.

Rodionov-style (arXiv:2605.05138): a hand-derived, self-contained Python
function that maps (state, action_id, x, y) -> (next_state, reward_class, done).

Empirical observations (200 random-exploration tuples):
- Available actions: {5, 6, 7}.  Distribution: 5=70, 6=72, 7=58.
- `done` is False in every tuple. `level` stays at 0.
- reward_class is 1 iff action_id == 5 (70/70), else 0 (130/130).
- Action 5: ALWAYS flips exactly one cell on row 53 from value 2 -> 3.
  The flipped column is the RIGHTMOST cell with value 2 on row 53.
  Verified 70/70.  Row 53 starts as all-2s and fills right-to-left with 3s
  as a step counter.  (x, y) for action 5 is (0, 0) — ignored.
- Action 6: clicks at random (x, y); NEVER changes the grid (72/72). Identity.
- Action 7: NEVER changes the grid (58/58). Identity.

Three invariants used:
  (i)  reward_class = 1 iff action_id == 5, else 0.
  (ii) `done` = False always.
  (iii) Action 5 mutates only one pixel: row 53, rightmost-2 col, 2 -> 3.
"""
from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np

GridLike = Union[List[List[int]], np.ndarray]

COUNTER_ROW = 53
COUNTER_FROM = 2
COUNTER_TO = 3


def _to_np(state: GridLike) -> np.ndarray:
    if isinstance(state, np.ndarray):
        return state.astype(np.uint8, copy=True)
    return np.asarray(state, dtype=np.uint8)


def _advance_counter(grid: np.ndarray) -> bool:
    """Flip the rightmost cell on row 53 with value 2 to value 3.

    Returns True if a flip happened, False if row 53 has no remaining 2s
    (counter is full).  We treat the no-2 case as a no-op (we have no
    observation of the wrap behaviour, so identity is the safe default).
    """
    row = grid[COUNTER_ROW]
    twos = np.where(row == COUNTER_FROM)[0]
    if twos.size == 0:
        return False
    grid[COUNTER_ROW, int(twos[-1])] = COUNTER_TO
    return True


def simulate(state: GridLike, action_id: int, x: int, y: int):
    """Predict next state, reward_class, done for game sb26.

    Parameters
    ----------
    state : 64x64 grid (list[list[int]] or np.ndarray), values 0-15.
    action_id : int in {5, 6, 7}.
    x, y : ints in [0, 64).  Unused for sb26 (action 6 click does nothing).

    Returns
    -------
    (next_state: np.ndarray, reward_class: int, done: bool)
    """
    grid = _to_np(state)

    if action_id == 5:
        _advance_counter(grid)
        return grid, 1, False

    # action_id == 6 (click) or action_id == 7: identity.
    return grid, 0, False
