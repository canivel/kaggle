"""Executable world model for ARC-AGI-3 game `tu93` -- v1.

Game description (derived from 200 random-exploration tuples):
- 6x6 maze of 3x3 cells, drawn on a 64x64 grid.
- Sprite is a 3x3 block of color 9 with a single color-4 "arrow" cell
  indicating facing direction (relative position in the 3x3 block):
    (0,1)=UP  (2,1)=DOWN  (1,0)=LEFT  (1,2)=RIGHT
- Cells live at top-left positions (row, col) where
    row in {16, 22, 28, 34, 40, 46}, col in {15, 21, 27, 33, 39, 45}.
- Between adjacent cells there is a 3x3 "connector" strip:
    color 2  -> open corridor (move succeeds)
    color 5  -> outside the maze (move blocked, NOOP for playfield)
- Goal: 3x3 block of color 14 at (46-48, 45-47). Never reached in
  the 200 sample tuples, so we leave it unmodelled.
- Row 63 is a 64-cell countdown timer that starts full of 6s and
  decrements right-to-left to 0, then resets to all 6s. Every action
  decrements the timer by either 1 or 2 cells. The set of pre-action
  6-counts that trigger a "double tick" is fully deterministic over
  the 50-action cycle (see DOUBLE_TICK_N6 below; confirmed on
  200 tuples = 4 full cycles, each pre-count seen exactly 4 times).

Actions (verified across all 200 tuples):
- 1 = UP    : block.row -= 6   if connector cells above are color 2; facing becomes (0,1)
- 2 = DOWN  : block.row += 6   if connector cells below are color 2; facing becomes (2,1)
- 3 = LEFT  : block.col -= 6   if connector cells left  are color 2; facing becomes (1,0)
- 4 = RIGHT : block.col += 6   if connector cells right are color 2; facing becomes (1,2)
- NOOP path (connector cells are color 5): playfield untouched, facing unchanged.
- (x, y) is always (0, 0) in this game (no clicks).

Reward is always reward_class=1, done is always False, level stays 0.
"""
from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np

GridLike = Union[List[List[int]], np.ndarray]

# --- Constants ----------------------------------------------------------
BG_OPEN = 0      # passable cell colour (where the sprite walked / can stand)
BG_OUTSIDE = 5   # outside-the-maze background
WALL = 2         # wall / open-connector colour
SPRITE_BODY = 9
SPRITE_ARROW = 4
GOAL = 14

CELL_SIZE = 3
STEP = 6  # cells are 3 wide + 3 connector

# Facing direction => relative position of the "4" arrow within the 3x3 block
FACING_UP = (0, 1)
FACING_DOWN = (2, 1)
FACING_LEFT = (1, 0)
FACING_RIGHT = (1, 2)

ACTION_TO_FACING = {1: FACING_UP, 2: FACING_DOWN, 3: FACING_LEFT, 4: FACING_RIGHT}
# action -> (drow, dcol) for block top-left
ACTION_DELTA = {1: (-STEP, 0), 2: (STEP, 0), 3: (0, -STEP), 4: (0, STEP)}

# Pre-action n6 values (count of 6s on row 63) that cause the timer to
# decrement by 2 instead of 1. Derived from 200 tuples; each value
# below was seen exactly 4 times with delta=2, and never with delta=1.
DOUBLE_TICK_N6 = frozenset({3, 8, 12, 17, 22, 26, 31, 35, 40, 44, 49, 54, 58, 63})

ROW_TIMER = 63


def _to_np(state: GridLike) -> np.ndarray:
    if isinstance(state, np.ndarray):
        return state.astype(np.uint8, copy=True)
    return np.asarray(state, dtype=np.uint8)


def _find_sprite_block(grid: np.ndarray) -> Tuple[int, int] | None:
    """Return (top_row, left_col) of the 3x3 sprite block, or None.

    Anchor by colour 4 (unique to the sprite -- 1 cell in every state).
    The 3x3 window containing the 4 must have exactly 1 cell of colour 4
    and 8 cells of colour 9.
    """
    pos = np.where(grid == SPRITE_ARROW)
    if len(pos[0]) != 1:
        return None
    r, c = int(pos[0][0]), int(pos[1][0])
    for tr in (r - 2, r - 1, r):
        if tr < 0 or tr + CELL_SIZE > grid.shape[0]:
            continue
        for tc in (c - 2, c - 1, c):
            if tc < 0 or tc + CELL_SIZE > grid.shape[1]:
                continue
            blk = grid[tr:tr + CELL_SIZE, tc:tc + CELL_SIZE]
            if (blk == SPRITE_ARROW).sum() == 1 and (blk == SPRITE_BODY).sum() == 8:
                return tr, tc
    return None


def _connector_slice(grid: np.ndarray, br: int, bc: int, action: int) -> np.ndarray | None:
    """Return the 3x3 connector slice in the direction of `action`, or None
    if the slice falls outside the grid."""
    H, W = grid.shape
    if action == 1:  # up: 3 rows above the block, same columns
        r0, r1 = br - CELL_SIZE, br
        c0, c1 = bc, bc + CELL_SIZE
    elif action == 2:  # down
        r0, r1 = br + CELL_SIZE, br + 2 * CELL_SIZE
        c0, c1 = bc, bc + CELL_SIZE
    elif action == 3:  # left
        r0, r1 = br, br + CELL_SIZE
        c0, c1 = bc - CELL_SIZE, bc
    elif action == 4:  # right
        r0, r1 = br, br + CELL_SIZE
        c0, c1 = bc + CELL_SIZE, bc + 2 * CELL_SIZE
    else:
        return None
    if r0 < 0 or c0 < 0 or r1 > H or c1 > W:
        return None
    return grid[r0:r1, c0:c1]


def _erase_sprite(grid: np.ndarray, br: int, bc: int) -> None:
    """Replace the 3x3 sprite block with the open-passage colour 0."""
    grid[br:br + CELL_SIZE, bc:bc + CELL_SIZE] = BG_OPEN


def _stamp_sprite(grid: np.ndarray, br: int, bc: int, facing: Tuple[int, int]) -> None:
    """Stamp the 3x3 sprite block: 8 cells of colour 9 + the arrow at `facing`."""
    grid[br:br + CELL_SIZE, bc:bc + CELL_SIZE] = SPRITE_BODY
    fr, fc = facing
    grid[br + fr, bc + fc] = SPRITE_ARROW


def _advance_timer(grid: np.ndarray) -> None:
    """Decrement the row-63 countdown.

    The rightmost run of 6s shrinks. Decrement by 2 if the current 6-count
    is in DOUBLE_TICK_N6, else by 1. If the row reaches all-zeros it would
    reset to all-6s on the *next* observation (handled by the game, not
    here). If the row is fully reset (all 6s) before a tick, also decrement
    by 1 (n6==64 not in DOUBLE_TICK_N6).
    """
    row = grid[ROW_TIMER]
    sixes = np.where(row == 6)[0]
    n6 = sixes.size
    if n6 == 0:
        # Already empty -- the actual game resets to all 6s here. Match that.
        row[:] = 6
        # And the new tick of the fresh timer would be a single decrement.
        row[ROW_TIMER if False else (row.size - 1)] = 0  # set col 63 -> 0
        return
    delta = 2 if n6 in DOUBLE_TICK_N6 else 1
    # rightmost-6 indices: the largest `delta` values in `sixes` -> set them to 0
    # (Row decrements right-to-left.)
    to_clear = sixes[-delta:]
    row[to_clear] = 0


def _apply_move(grid: np.ndarray, action: int) -> None:
    block = _find_sprite_block(grid)
    if block is None:
        return
    br, bc = block
    conn = _connector_slice(grid, br, bc, action)
    if conn is None:
        return  # off-grid: nothing to do
    new_facing = ACTION_TO_FACING[action]
    # Connector colour determines move success.
    # Colour 2 -> corridor (move).  Colour 5 -> outside (blocked NOOP).
    if (conn == WALL).all():
        dr, dc = ACTION_DELTA[action]
        new_br, new_bc = br + dr, bc + dc
        # Sanity check new block fits in grid.
        H, W = grid.shape
        if 0 <= new_br <= H - CELL_SIZE and 0 <= new_bc <= W - CELL_SIZE:
            _erase_sprite(grid, br, bc)
            _stamp_sprite(grid, new_br, new_bc, new_facing)
            return
    # Blocked or ambiguous: leave playfield (incl. facing) untouched.
    return


def simulate(state: GridLike, action_id: int, x: int, y: int):
    """Predict (next_state, reward_class, done) for ARC-AGI-3 game tu93.

    Parameters
    ----------
    state : 64x64 grid (list[list[int]] or np.ndarray), values 0-15.
    action_id : int in {1, 2, 3, 4} (1=UP, 2=DOWN, 3=LEFT, 4=RIGHT).
    x, y : ints, always (0, 0) in this game (kept for the standard signature).
    """
    grid = _to_np(state)
    if action_id in (1, 2, 3, 4):
        _apply_move(grid, action_id)
    _advance_timer(grid)
    return grid.tolist(), 1, False
