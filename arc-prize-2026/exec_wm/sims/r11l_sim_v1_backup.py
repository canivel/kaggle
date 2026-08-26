"""Executable world model for ARC-AGI-3 game `r11l` -- v1.

Rodionov-style hand-derived simulator. Maps (state, action_id, x, y) ->
(next_state, reward_class, done).

Game `r11l` characteristics (200-tuple summary + raw inspection):
- Only one action available: action 6 (click).
- reward_class == 1 for every observed tuple. done == False, level stays at 0.
- Grid is 64x64, values 0..15.
- Layout: column 0 acts as a vertical step-counter (top->bottom). Cells start
  at 0 and become 5 as steps tick. The playfield occupies the rest of the
  grid; background is color 2, interior is color 5 with various small sprites
  and pieces.

Empirically verified invariants (raw inspection of all 200 tuples):

  INV-1 ("counter tick"): every transition advances column 0 by setting the
  first 0-valued cell (topmost) to 5. Verified on 43/43 n_changed==1 cases
  and 151/153 n_changed>=25 cases.

  INV-2 (rewards/done): reward_class==1, done==False for 200/200 tuples.

The "n_changed >= 25" cases additionally include a playfield-state change
caused by the click landing on an active sprite/piece (color 5 interior).
The exact playfield transition depends on hidden game state (which piece
is active, its trajectory) that cannot be recovered from a single frame.
v1 leaves the playfield untouched in those cases, preserving the pixel-match
floor (we are wrong on ~30..150 cells per such tuple but right on the
~4000 cells the playfield change does not touch).

We also do not model:
- The 4 n_changed==2 cases where the counter ticks twice (rare trigger,
  probably "click on border 2 cell" but the rule is not crisp enough to ship).

A handful of large-n cases additionally fail INV-1 (2/153). We accept that.
"""
from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np

GridLike = Union[List[List[int]], np.ndarray]

COUNTER_COL = 0
COUNTER_FILL = 5


def _to_np(state: GridLike) -> np.ndarray:
    if isinstance(state, np.ndarray):
        return state.astype(np.uint8, copy=True)
    return np.asarray(state, dtype=np.uint8)


def _advance_counter(grid: np.ndarray) -> None:
    """Set the topmost 0-valued cell in column 0 to 5.

    If column 0 has no 0s (full counter, 64 ticks), do nothing -- the wrap
    is observed to happen between transitions, not within them.
    """
    col = grid[:, COUNTER_COL]
    zeros = np.where(col == 0)[0]
    if zeros.size == 0:
        return
    grid[zeros[0], COUNTER_COL] = COUNTER_FILL


def simulate(state: GridLike, action_id: int, x: int, y: int):
    """Predict next_state, reward_class, done for game r11l.

    Parameters
    ----------
    state : 64x64 grid (list[list[int]] or np.ndarray), values 0..15.
    action_id : int. Only 6 (click) is observed; we apply counter tick
                regardless to stay safe.
    x, y : ints in [0, 64). Click coords (col, row) convention. Not used
           directly in v1 -- the modal effect of any click is just the
           counter tick.
    """
    grid = _to_np(state)

    # Apply the counter tick for any action. In the observed data, every
    # action 6 triggers a tick; modeling all actions identically is the
    # safe default in case unseen actions also tick.
    _advance_counter(grid)

    # Reward and done are constant across all 200 observed tuples.
    reward_class = 1
    done = False
    return grid, reward_class, done
