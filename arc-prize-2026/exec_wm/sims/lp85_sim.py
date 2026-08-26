"""Executable world model for ARC-AGI-3 game `lp85` — v1.

Rodionov-style (arXiv:2605.05138): a hand-derived, self-contained Python
function mapping (state, action_id, x, y) -> (next_state, reward_class, done).

Game lp85 overview (200 random-exploration tuples observed):
- ONLY action id 6 (click) is available.
- 197/200 clicks are NOOP (reward_class=0). The state is unchanged.
- 3/200 clicks hit one of two "button" zones at the bottom of the playfield
  and trigger a non-trivial reward_class=1 transition.
- `done` is always False, `level` is always 0.

Layout (background colour = 4):
- Col 0 holds a vertical "lives" counter: rows 0..N-1 are 14 (remaining
  lives), rows N..N+k are 5 (used lives). A successful click consumes 5
  lives (top 5 remaining 14-cells flip to 5).
- The puzzle is a 20-tile cyclic conveyor laid out as a rectangle:
    * top row at row 19-22, 7 tiles at columns {12,18,24,30,36,42,48}
    * right column at col 48-51, 3 tiles at rows {25,31,37}
    * bottom row at row 43-46, 7 tiles at columns {12,18,24,30,36,42,48}
      (in physical L->R order; the conveyor traverses bottom R->L)
    * left column at col 12-15, 3 tiles at rows {25,31,37}
      (in physical T->B order; the conveyor traverses left B->T)
  Total = 7 + 3 + 7 + 3 = 20 tiles. Each tile is 4 cols wide x 4 rows tall.
- The two click buttons (rows 29..36) live just outside the left and right
  conveyor columns:
    * LEFT button — a 8-coloured shape near cols 2..7
    * RIGHT button — a 14-coloured shape near cols 56..61

Click rules (verified on all 3 reward=1 events):
- Click inside the RIGHT button bounding box -> rotate conveyor clockwise
  by 1 tile (top shifts right, right-col shifts down, bottom shifts left,
  left-col shifts up).
- Click inside the LEFT button bounding box -> rotate conveyor
  counter-clockwise by 1 tile (inverse).
- Both also consume 5 lives.
- Any other click -> identity transition, reward_class=0.

Verified on 3/3 reward=1 events (full grid exactly matches state_t1) and
197/197 reward=0 events (identity).
"""
from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np

GridLike = Union[List[List[int]], np.ndarray]

# --- Constants ----------------------------------------------------------
BG = 4
LIFE_REMAIN = 14
LIFE_USED = 5
LIVES_PER_CLICK = 5

# Tile geometry: tiles are 4x4 blocks of a single colour.
TILE_H = 4
TILE_W = 4

# Top row tiles: row range 19..22 inclusive (i.e. rows [19:23]).
TOP_ROW0 = 19
TOP_ROW1 = 23  # exclusive
# Bottom row tiles: rows 43..46 inclusive.
BOT_ROW0 = 43
BOT_ROW1 = 47  # exclusive
# Left column tiles: col range 12..15 inclusive (cols [12:16]).
LCOL0 = 12
LCOL1 = 16  # exclusive
# Right column tiles: col range 48..51 inclusive.
RCOL0 = 48
RCOL1 = 52  # exclusive

# Column anchors for the 7 top/bot tiles (left edge of each 4-wide tile).
TILE_COLS = (12, 18, 24, 30, 36, 42, 48)
# Row anchors for the 3 left-col / right-col tiles.
TILE_ROWS = (25, 31, 37)

# Click button bounding boxes (inclusive).
LEFT_BUTTON_BBOX = (29, 36, 2, 7)    # (y_min, y_max, x_min, x_max)
RIGHT_BUTTON_BBOX = (29, 36, 56, 61)


def _to_np(state: GridLike) -> np.ndarray:
    if isinstance(state, np.ndarray):
        return state.astype(np.uint8, copy=True)
    return np.asarray(state, dtype=np.uint8)


def _in_bbox(x: int, y: int, bbox: Tuple[int, int, int, int]) -> bool:
    y_min, y_max, x_min, x_max = bbox
    return (y_min <= y <= y_max) and (x_min <= x <= x_max)


def _read_conveyor(grid: np.ndarray) -> List[int]:
    """Return the 20-tile cycle in clockwise order starting from top[0].

    Order: top[0..6], right_col[0..2] (top->bottom),
           bot[6..0] (right->left), left_col[2..0] (bottom->top).
    """
    cycle = []
    # top L->R
    for c in TILE_COLS:
        cycle.append(int(grid[TOP_ROW0, c]))
    # right col T->B
    for r in TILE_ROWS:
        cycle.append(int(grid[r, RCOL0]))
    # bottom R->L
    for c in reversed(TILE_COLS):
        cycle.append(int(grid[BOT_ROW0, c]))
    # left col B->T
    for r in reversed(TILE_ROWS):
        cycle.append(int(grid[r, LCOL0]))
    return cycle


def _write_conveyor(grid: np.ndarray, cycle: List[int]) -> None:
    """Stamp 20 tiles back into the grid using the same order as _read_conveyor.

    Each tile is a 4x4 monochrome block; we paint every cell in the block.
    """
    assert len(cycle) == 20
    idx = 0
    # top L->R, rows 19..22, cols c..c+3
    for c in TILE_COLS:
        grid[TOP_ROW0:TOP_ROW1, c:c + TILE_W] = cycle[idx]
        idx += 1
    # right col T->B, rows r..r+3, cols 48..51
    for r in TILE_ROWS:
        grid[r:r + TILE_H, RCOL0:RCOL1] = cycle[idx]
        idx += 1
    # bottom R->L, rows 43..46
    for c in reversed(TILE_COLS):
        grid[BOT_ROW0:BOT_ROW1, c:c + TILE_W] = cycle[idx]
        idx += 1
    # left col B->T, cols 12..15
    for r in reversed(TILE_ROWS):
        grid[r:r + TILE_H, LCOL0:LCOL1] = cycle[idx]
        idx += 1


def _consume_lives(grid: np.ndarray, n: int = LIVES_PER_CLICK) -> None:
    """Flip the top `n` remaining-life (14) cells in col 0 to used (5)."""
    col = grid[:, 0]
    remaining = np.where(col == LIFE_REMAIN)[0]
    if remaining.size == 0:
        return
    to_flip = remaining[:n]
    grid[to_flip, 0] = LIFE_USED


def _rotate_clockwise(grid: np.ndarray) -> None:
    """Rotate the 20-tile conveyor clockwise by 1 (RIGHT-button effect)."""
    cyc = _read_conveyor(grid)
    # Clockwise rotate by 1: new[i] = old[(i-1) mod 20]
    rotated = [cyc[(i - 1) % 20] for i in range(20)]
    _write_conveyor(grid, rotated)


def _rotate_counter_clockwise(grid: np.ndarray) -> None:
    """Rotate the 20-tile conveyor counter-clockwise by 1 (LEFT-button effect)."""
    cyc = _read_conveyor(grid)
    rotated = [cyc[(i + 1) % 20] for i in range(20)]
    _write_conveyor(grid, rotated)


def simulate(state: GridLike, action_id: int, x: int, y: int):
    """Predict (next_state, reward_class, done) for game lp85.

    Parameters
    ----------
    state : 64x64 grid (list[list[int]] or np.ndarray), values 0-15.
    action_id : int; only 6 (click) is meaningful for lp85.
    x, y : ints in [0, 64). Click position (x = column, y = row).
    """
    grid = _to_np(state)

    # All non-click actions, and clicks outside the two button zones,
    # are pure NOOPs in the observed data.
    if action_id != 6:
        return grid, 0, False

    in_left = _in_bbox(x, y, LEFT_BUTTON_BBOX)
    in_right = _in_bbox(x, y, RIGHT_BUTTON_BBOX)

    if not (in_left or in_right):
        return grid, 0, False

    if in_right:
        _rotate_clockwise(grid)
    else:
        _rotate_counter_clockwise(grid)

    _consume_lives(grid, LIVES_PER_CLICK)

    # done remains False on all 200 observed tuples (no level changes).
    return grid, 1, False
