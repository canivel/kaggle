"""Executable world model for ARC-AGI-3 game `sc25` — v2.

Adds to v1:
1. Lives-bar depletion: on every non-6 action whose bar is non-empty,
   zero the top-most 2 rows of cols 62-63.  Drops are truly on a hidden
   timer (no in-state clue), but they fire ~63% of the time and almost
   exclusively under actions 1-4.  Modelling them as deterministic on
   non-6 actions is the +EV bet: it matches ~63% of those tuples while
   missing the ~37% that would have matched under v1's "never drop"
   model. Net per-action gain: 22.2→80.6 (a4), 36.6→65.9 (a3), 41.9→
   53.5 (a1), 37.5→59.4 (a2). See validate_sim numbers in notes_v2.
2. Action 6 "stamp" rule: when the click (x,y) lands in the bottom
   guide panel (x in [22,38] and y in [47,61]), stamp a 3x3 block of
   14s centred on the nearest button (button centres at row {50,55,60}
   × col {25,30,35}).  Verified on 5/5 observed stamps.
3. reward_class: 1 iff the predicted next-state differs from current.

The drop-prediction is honest curve-fitting on the marginal: there is
no state-side signal for the hidden timer, so this is a 63% bet, not
a deterministic invariant.
"""
from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np

GridLike = Union[List[List[int]], np.ndarray]

# --- Constants ----------------------------------------------------------
SPRITE_ROW0 = 19
SPRITE_ROW1 = 23  # exclusive (rows 19..22)
SPRITE_H = 4
SPRITE_W = 4

# Tube background revealed when sprite vacates a cell (rows 19..22 only)
TUBE_BG = 2

# Valid sprite-left columns (slot grid of width 4 from 23 to 39)
SPRITE_COL_MIN = 23
SPRITE_COL_MAX = 39
SPRITE_STEP = 4

# Lives bar lives at cols 62..63
BAR_COL0 = 62
BAR_COL1 = 64  # exclusive

# Bottom guide panel — click region and stamp grid
PANEL_X_MIN = 22
PANEL_X_MAX = 38
PANEL_Y_MIN = 47
PANEL_Y_MAX = 61
# Button centres (rows then cols).  Each button is a 3x3 block.
BUTTON_ROW_CENTRES = (50, 55, 60)
BUTTON_COL_CENTRES = (25, 30, 35)
STAMP_VAL = 14

# --- Sprite templates (4x4) ----------------------------------------------
TPL_A = np.array([
    [9, 9, 10, 10],
    [9, 9, 10, 10],
    [9, 9, 10, 10],
    [9, 9, 10, 10],
], dtype=np.uint8)

TPL_B = np.array([
    [9, 9, 9, 9],
    [9, 9, 9, 9],
    [10, 10, 10, 10],
    [10, 10, 10, 10],
], dtype=np.uint8)

TPL_C = np.array([
    [10, 10, 9, 9],
    [10, 10, 9, 9],
    [10, 10, 9, 9],
    [10, 10, 9, 9],
], dtype=np.uint8)

TPL_D = np.array([
    [10, 10, 10, 10],
    [10, 10, 10, 10],
    [9, 9, 9, 9],
    [9, 9, 9, 9],
], dtype=np.uint8)


def _to_np(state: GridLike) -> np.ndarray:
    if isinstance(state, np.ndarray):
        return state.astype(np.uint8, copy=True)
    return np.asarray(state, dtype=np.uint8)


def _find_sprite_col(grid: np.ndarray) -> int | None:
    band = grid[SPRITE_ROW0:SPRITE_ROW1]
    mask4 = ((band == 9) | (band == 10)).all(axis=0)
    runs: list[Tuple[int, int]] = []
    start = None
    for c in range(64):
        if mask4[c]:
            if start is None:
                start = c
        else:
            if start is not None:
                runs.append((start, c))
                start = None
    if start is not None:
        runs.append((start, 64))
    candidates = [(s, e) for s, e in runs if (e - s) == 4
                  and SPRITE_COL_MIN <= s <= SPRITE_COL_MAX]
    if not candidates:
        return None
    return candidates[0][0]


def _erase_sprite(grid: np.ndarray, col: int) -> None:
    grid[SPRITE_ROW0:SPRITE_ROW1, col:col + SPRITE_W] = TUBE_BG


def _stamp_sprite(grid: np.ndarray, col: int, orient: str) -> None:
    tpl = {'A': TPL_A, 'B': TPL_B, 'C': TPL_C, 'D': TPL_D}[orient]
    grid[SPRITE_ROW0:SPRITE_ROW1, col:col + SPRITE_W] = tpl


def _apply_rotate(grid: np.ndarray, target_orient: str) -> None:
    col = _find_sprite_col(grid)
    if col is None:
        return
    _stamp_sprite(grid, col, target_orient)


def _apply_translate(grid: np.ndarray, dcol: int, target_orient: str) -> None:
    col = _find_sprite_col(grid)
    if col is None:
        return
    new_col = col + dcol
    if new_col < SPRITE_COL_MIN or new_col > SPRITE_COL_MAX:
        _stamp_sprite(grid, col, target_orient)
        return
    _erase_sprite(grid, col)
    _stamp_sprite(grid, new_col, target_orient)


def _drop_bar_top_pair(grid: np.ndarray) -> bool:
    """Zero the top-most 2 rows of cols 62-63 that still hold 14s.
    Returns True iff a drop happened (bar was non-empty)."""
    bar = grid[:, BAR_COL0:BAR_COL1]
    rows_with_14 = np.where((bar == 14).any(axis=1))[0]
    if rows_with_14.size == 0:
        return False
    r = int(rows_with_14[0])
    # Always zero 2 rows even if only 1 row left (matches mechanic — last
    # observed drop went from 8 cells to 0 in two pairs)
    grid[r:r + 2, BAR_COL0:BAR_COL1] = 0
    return True


def _panel_stamp(grid: np.ndarray, x: int, y: int) -> bool:
    """If (x,y) is in the bottom guide panel, stamp a 3x3 block of 14s at
    the nearest button.  Returns True iff a stamp was placed."""
    if not (PANEL_X_MIN <= x <= PANEL_X_MAX and PANEL_Y_MIN <= y <= PANEL_Y_MAX):
        return False
    rc = min(BUTTON_ROW_CENTRES, key=lambda v: abs(v - y))
    cc = min(BUTTON_COL_CENTRES, key=lambda v: abs(v - x))
    grid[rc - 1:rc + 2, cc - 1:cc + 2] = STAMP_VAL
    return True


def simulate(state: GridLike, action_id: int, x: int, y: int):
    """Predict (next_state, reward_class, done) for sc25."""
    grid = _to_np(state)
    original = grid.copy()

    if action_id == 1:
        _apply_rotate(grid, 'B')
        _drop_bar_top_pair(grid)
    elif action_id == 2:
        _apply_rotate(grid, 'D')
        _drop_bar_top_pair(grid)
    elif action_id == 3:
        _apply_translate(grid, -SPRITE_STEP, 'A')
        _drop_bar_top_pair(grid)
    elif action_id == 4:
        _apply_translate(grid, +SPRITE_STEP, 'C')
        _drop_bar_top_pair(grid)
    elif action_id == 6:
        _panel_stamp(grid, x, y)
    # any other action_id: identity (defensive)

    changed = not np.array_equal(grid, original)
    reward_class = 1 if changed else 0
    done = False
    return grid, reward_class, done
