"""Executable world model for ARC-AGI-3 game `vc33` — v1.

Rodionov-style (arXiv:2605.05138): a hand-derived, self-contained Python
function mapping (state, action_id, x, y) -> (next_state, reward_class, done).

Empirical findings on 200 random-exploration tuples (only action_id == 6
available; all rewards == 1, never done):

1. Outside of row 0, the playfield is invariant under action 6 in 199/200
   tuples (the lone exception is step 146 which we leave as identity).
2. Row 0 is a left-to-right "fill" timer: the right half is color 7, the
   left half is color 4, and each click flips the rightmost run of 7s
   to 4. After 50 clicks the whole row is 4 and the game auto-resets
   row 0 to all-7s (we observe this between state_t1[t=49] and
   state_t[t=50]). The reset is the env's job, not ours.
3. The number of 7-cells that flip on a single click is deterministic
   given the rightmost-7 column R7 in state_t[0]:
     R7 in DOUBLE_FLIP_SET -> flip both R7 and R7-1 (two cells)
     otherwise              -> flip only R7 (one cell)
   DOUBLE_FLIP_SET = {2,7,11,16,21,25,30,34,39,43,48,53,57,62}
   This was extracted from 199/199 simple-bucket tuples; verified
   deterministic (every R7 value seen >= 4 times always gives the same
   delta).
4. Reward is always 1, done is always False, level stays 0.
5. x,y of the click do NOT affect the change pattern (verified: same R7
   produces same delta regardless of click coords).
"""
from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np

GridLike = Union[List[List[int]], np.ndarray]

# Set of rightmost-7 column positions for which a click flips TWO cells
# (R7 and R7-1) instead of one. Derived from 199/199 observed tuples.
DOUBLE_FLIP_R7 = frozenset(
    {2, 7, 11, 16, 21, 25, 30, 34, 39, 43, 48, 53, 57, 62}
)

BG_FILLED = 4  # left side of the row-0 timer
BG_UNFILLED = 7  # right side of the row-0 timer


def _to_np(state: GridLike) -> np.ndarray:
    if isinstance(state, np.ndarray):
        return state.astype(np.uint8, copy=True)
    return np.asarray(state, dtype=np.uint8)


def _tick_row0(grid: np.ndarray) -> None:
    """Advance the row-0 timer by 1 or 2 cells depending on R7."""
    row = grid[0]
    sevens = np.where(row == BG_UNFILLED)[0]
    if sevens.size == 0:
        # Row already full of 4s. Leave it alone — the env's auto-reset
        # will be visible in the next state_t, not in our state_t1.
        return
    R7 = int(sevens.max())
    if R7 in DOUBLE_FLIP_R7 and R7 - 1 >= 0:
        grid[0, R7] = BG_FILLED
        grid[0, R7 - 1] = BG_FILLED
    else:
        grid[0, R7] = BG_FILLED


def simulate(state: GridLike, action_id: int, x: int, y: int):
    """Predict next state, reward_class, done for game vc33.

    Parameters
    ----------
    state : 64x64 grid (list[list[int]] or np.ndarray), values 0-15.
    action_id : int — only 6 is meaningful for vc33.
    x, y : ints in [0, 64) — unused (verified empirically).
    """
    grid = _to_np(state)

    if action_id == 6:
        _tick_row0(grid)
    # All other action_ids: identity (vc33 only exposes action 6, but be
    # robust if validators ever query a different id).

    return grid.tolist(), 1, False
