"""Executable world model for ARC-AGI-3 game `dc22` -- v2.

Rodionov-style hand-derived simulator. v2 extends v1's lower-playfield model
to the upper corridor (rows <= 37 with path-colour 9), so UP from row 38 and
moves at row 36 are now deterministically predicted.

What v2 adds over v1
--------------------
1. **Upper-corridor movement**: when the player is at rows <= 37, the
   path colour is 9 (not 2). Movement is allowed iff the 2x2 target cells
   are colour 9. Vacated cells are filled with 9. (Lower playfield, row >= 38,
   still uses path 2.)
2. **UP from row 38** is now allowed iff cells (r-2, c) and (r-2, c+1)
   are both colour 9 (verified: 3/3 such cases moved, 8/8 4-wall cases
   NOOPed in the v1 data).
3. **DOWN from row 36** to row 38 is allowed iff target cells (r+2..r+3)
   are colour 2 (verified: 2/2 such cases moved).
4. **UP from row 36** to row 34 is allowed iff target cells are colour 9
   (1/1 verified).

Counter (unchanged from v1)
---------------------------
Tick parity is not derivable from observable state. v1's `count % 2 == 0`
rule achieves ~51% counter accuracy, which is the empirical ceiling.

Action 6 big-repaint (still unmodelled)
---------------------------------------
2/48 clicks (steps 56, 153) produced n_changed >= 97 UI repaints. The
trigger sprite footprint and exact resulting pattern are too data-sparse
to model from observations alone; v2 still treats action 6 as counter-only.
"""
from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np

GridLike = Union[List[List[int]], np.ndarray]

# --- Constants ---------------------------------------------------------
PLAYER_COLOR = 14
LOWER_PATH = 2          # path colour in rows >= 38
UPPER_PATH = 9          # path colour in upper corridor (rows <= 37)

# Lower playfield bounds
COL_MIN = 8
COL_MAX = 12
ROW_BOTTOM = 42
LOWER_TOP_ROW = 38      # threshold: row 38 is the boundary; UP from 38 enters upper corridor

COUNTER_ROW = 63
COUNTER_TICK_VALUE = 3
COUNTER_LEN = 64


# --- Helpers -----------------------------------------------------------

def _to_np(state: GridLike) -> np.ndarray:
    if isinstance(state, np.ndarray):
        return state.astype(np.uint8, copy=True)
    return np.asarray(state, dtype=np.uint8)


def _find_player(grid: np.ndarray) -> Tuple[int, int] | None:
    """Top-left (row, col) of the 2x2 player block, or None."""
    ys, xs = np.where(grid == PLAYER_COLOR)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(xs.min())


def _path_for_row(r: int) -> int:
    """Path colour used to fill cells vacated by the player at row r."""
    return LOWER_PATH if r >= LOWER_TOP_ROW else UPPER_PATH


def _target_is_path(grid: np.ndarray, nr: int, nc: int) -> bool:
    """Return True iff the 2x2 region at (nr, nc) is the path colour
    appropriate for row nr (and within bounds)."""
    if nr < 0 or nc < 0 or nr + 2 > grid.shape[0] or nc + 2 > grid.shape[1]:
        return False
    target = grid[nr:nr + 2, nc:nc + 2]
    expected = _path_for_row(nr)
    return bool(np.all(target == expected))


def _move_player(grid: np.ndarray, tl: Tuple[int, int], new_tl: Tuple[int, int]) -> None:
    """Move the 2x2 player block. Vacated cells take the path colour of
    the *source* row; the new position becomes PLAYER_COLOR."""
    r, c = tl
    nr, nc = new_tl
    fill = _path_for_row(r)
    grid[r:r + 2, c:c + 2] = fill
    grid[nr:nr + 2, nc:nc + 2] = PLAYER_COLOR


def _tick_counter_if_due(grid: np.ndarray) -> bool:
    """Tick row-63 counter iff count of TICK cells is even and < 64."""
    row = grid[COUNTER_ROW]
    n_filled = int((row == COUNTER_TICK_VALUE).sum())
    if n_filled >= COUNTER_LEN:
        return False
    if n_filled % 2 != 0:
        return False
    zeros = np.where(row != COUNTER_TICK_VALUE)[0]
    if zeros.size == 0:
        return False
    grid[COUNTER_ROW, zeros[0]] = COUNTER_TICK_VALUE
    return True


# --- Per-action logic --------------------------------------------------

def _apply_up(grid: np.ndarray) -> None:
    tl = _find_player(grid)
    if tl is None:
        return
    r, c = tl
    new_r = r - 2
    if new_r < 0:
        return
    # Allowed iff the target 2x2 cells are the path colour for the destination row.
    if not _target_is_path(grid, new_r, c):
        return
    _move_player(grid, tl, (new_r, c))


def _apply_down(grid: np.ndarray) -> None:
    tl = _find_player(grid)
    if tl is None:
        return
    r, c = tl
    if r >= ROW_BOTTOM:
        return
    new_r = r + 2
    if not _target_is_path(grid, new_r, c):
        return
    _move_player(grid, tl, (new_r, c))


def _apply_left(grid: np.ndarray) -> None:
    tl = _find_player(grid)
    if tl is None:
        return
    r, c = tl
    if c <= COL_MIN:
        return
    new_c = c - 2
    if not _target_is_path(grid, r, new_c):
        return
    _move_player(grid, tl, (r, new_c))


def _apply_right(grid: np.ndarray) -> None:
    tl = _find_player(grid)
    if tl is None:
        return
    r, c = tl
    if c >= COL_MAX:
        return
    new_c = c + 2
    if not _target_is_path(grid, r, new_c):
        return
    _move_player(grid, tl, (r, new_c))


# --- simulate ----------------------------------------------------------

def simulate(state: GridLike, action_id: int, x: int, y: int):
    """Predict (next_state, reward_class, done) for game dc22."""
    grid = _to_np(state)
    before = grid.copy()

    if action_id == 1:
        _apply_up(grid)
    elif action_id == 2:
        _apply_down(grid)
    elif action_id == 3:
        _apply_left(grid)
    elif action_id == 4:
        _apply_right(grid)
    elif action_id == 6:
        # No deterministic-from-state model for the rare big-repaint clicks.
        pass
    # other action ids: leave unchanged

    _tick_counter_if_due(grid)

    changed = not np.array_equal(grid, before)
    reward_class = 1 if changed else 0
    done = False
    return grid, reward_class, done
