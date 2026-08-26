"""Executable world model for ARC-AGI-3 game `tn36` -- v2.

Rodionov-style (arXiv:2605.05138): a hand-derived, self-contained Python
function that maps (state, action_id, x, y) -> (next_state, reward_class, done).

v1 (97.0% state_exact, 99.9978% pixel): countdown-tick only.
v2 adds the button-toggle rule for the 6 click cases that v1 missed.

Empirical decoding of fails (idx 10, 18, 72, 79, 98, 184):
- UI has a single row of 5 indicator columns at x in {21, 26, 31, 36, 41}.
- Each indicator has TWO orientations:
    * horizontal at row 42, cols (c-1, c, c+1) -- visible when y_click is 42-43
    * vertical at rows 44-46, col c           -- visible when y_click is 44-46
- Action 6 with click (x, y) toggles the nearest indicator (between 5<->1)
  iff y in {42..46} AND |x - nearest_center| <= 2.
- Above threshold dx=2 is the gap between max-fail-dx (=2) and
  min-nonfail-dx (=4) over the 13 in-band clicks; safe by construction.
- The toggled cells always swap between palette values 5 and 1; we just
  read the current value at the target cell and flip it (5 -> 1 or 1 -> 5).
- The countdown tick on row 1 STILL happens for these 6 cases too.

This rule is fitted on 6 positive and 7 negative in-band observations.
Out-of-band clicks (y not in 42..46) are 187/187 identity outside row 1.
"""
from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np

GridLike = Union[List[List[int]], np.ndarray]

# --- Constants ----------------------------------------------------------
COUNTDOWN_ROW = 1
COUNTDOWN_COL_MIN = 1
COUNTDOWN_COL_MAX = 61  # inclusive
COUNTDOWN_FULL = 9
COUNTDOWN_USED = 3

# Button layout (verified on rows 42 and 44-46 of every observed state_t)
BUTTON_CENTERS = (21, 26, 31, 36, 41)
BUTTON_DX_MAX = 2          # max |x - center| for which a click registers
BUTTON_ROW_HORIZ = 42      # horizontal-indicator row
BUTTON_Y_HORIZ = (42, 43)  # click-y values that target the horizontal indicator
BUTTON_ROWS_VERT = (44, 45, 46)  # vertical-indicator rows
BUTTON_Y_VERT = (44, 45, 46)     # click-y values that target the vertical indicator
TOGGLE_A = 5
TOGGLE_B = 1


def _to_np(state: GridLike) -> np.ndarray:
    if isinstance(state, np.ndarray):
        return state.astype(np.uint8, copy=True)
    return np.asarray(state, dtype=np.uint8)


def _tick_countdown(grid: np.ndarray) -> None:
    """Flip the rightmost 9 on row 1 (cols 1..61) to 3.

    If no 9 remains (countdown exhausted), reset cols 1..61 to 9 and then
    flip col 61 to 3. This matches the env's between-step reset behaviour.
    """
    row_slice = grid[COUNTDOWN_ROW, COUNTDOWN_COL_MIN:COUNTDOWN_COL_MAX + 1]
    nines = np.where(row_slice == COUNTDOWN_FULL)[0]
    if nines.size == 0:
        # Defensive: env normally resets between steps, but if we are ever
        # called with an exhausted countdown, simulate reset+tick.
        row_slice[:] = COUNTDOWN_FULL
        row_slice[-1] = COUNTDOWN_USED
        return
    rightmost = int(nines[-1])
    row_slice[rightmost] = COUNTDOWN_USED


def _toggle_value(v: int) -> int:
    """Swap 5 <-> 1; pass-through otherwise (defensive)."""
    if v == TOGGLE_A:
        return TOGGLE_B
    if v == TOGGLE_B:
        return TOGGLE_A
    return v


def _maybe_toggle_button(grid: np.ndarray, x: int, y: int) -> None:
    """Toggle the indicator nearest to (x, y) if the click is in the
    button band (y in 42..46) and close enough horizontally (dx <= 2).

    Horizontal vs vertical orientation is selected by y:
        y in {42, 43} -> horizontal triple at (row 42, cols c-1..c+1)
        y in {44, 45, 46} -> vertical triple at (rows 44..46, col c)
    """
    if not (42 <= y <= 46):
        return
    # Pick nearest center.
    c = min(BUTTON_CENTERS, key=lambda cc: abs(x - cc))
    if abs(x - c) > BUTTON_DX_MAX:
        return

    if y in BUTTON_Y_HORIZ:
        # Sanity: the 3 cells must all currently hold the same toggle value.
        # (If not, we still toggle each cell individually -- safer than
        # bailing because the rule never fired on a mixed-cell case in obs.)
        for col in (c - 1, c, c + 1):
            grid[BUTTON_ROW_HORIZ, col] = _toggle_value(int(grid[BUTTON_ROW_HORIZ, col]))
    else:  # y in BUTTON_Y_VERT
        for r in BUTTON_ROWS_VERT:
            grid[r, c] = _toggle_value(int(grid[r, c]))


def simulate(state: GridLike, action_id: int, x: int, y: int):
    """Predict next state, reward_class, done for game tn36.

    Parameters
    ----------
    state : 64x64 grid (list[list[int]] or np.ndarray), values 0-11.
    action_id : int, expected to be 6 (only action observed).
    x, y : ints in [0, 64). Click coords; used by the button-toggle rule.

    Returns
    -------
    (next_state: np.ndarray, reward_class: int, done: bool)
    """
    grid = _to_np(state)

    if action_id == 6:
        _tick_countdown(grid)
        _maybe_toggle_button(grid, int(x), int(y))
    # No other actions observed; leave grid unchanged for safety on
    # unexpected action ids.

    return grid, 1, False
