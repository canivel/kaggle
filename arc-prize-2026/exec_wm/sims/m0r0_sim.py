"""Executable world model for ARC-AGI-3 game `m0r0` — v1.

Rodionov-style (arXiv:2605.05138).

Layout (deduced from 200 observation tuples):

- 64x64 grid with a STATIC background made of {5, 11, 12}: cells where the
  state value is 11 or 12 in any observation always have those values
  (constant across all 200 tuples) and act as walls / decoration.
- Two 5x5 blocks of colour 10 form the "cursor" (left block + right block).
  Their (top-row, left-col) positions are always on a 5-step lattice and
  always lie strictly inside the 5-cell playfield channel.
- A monotonic step counter occupies row 0 (filling from col 63 leftward)
  and row 63 (filling from col 0 rightward) with colour 0. The two rows
  are mirror-symmetric. The counter ticks on SOME actions but the trigger
  is not predictable from a single frame (hidden state).

Actions:
- 1 (UP):    each cursor block tries to move -5 rows.
- 2 (DOWN):  each cursor block tries to move +5 rows.
- 3 (X-OUT): left block -5 cols, right block +5 cols (expand horizontally).
- 4 (X-IN):  left block +5 cols, right block -5 cols (contract).
- 5/6:       only tick the counter (no spatial change).

A block can move only if its destination 5x5 region is entirely inside the
playfield (no overlap with the 11 / 12 wall cells).  Verified 131/131
(100%) on the movement actions across all four directions.

Reward & done:
- reward_class == 1 iff next_state != state (verified 200/200).
- done is always False.
- level is always 0.

What v1 does NOT model:
- The shared row-0 / row-63 counter ticks. We leave the counter rows
  unchanged. This costs us ~35% of action-1 / 47% of action-4 / etc.,
  exactly the fraction of tuples where the counter happened to tick.
- A5 / A6 only ever tick the counter, so the deterministic prediction
  is "no change", which is correct for the 47% of A5+A6 tuples that
  had no change but wrong for the others.
"""
from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np

GridLike = Union[List[List[int]], np.ndarray]

WALL_A = 11
WALL_B = 12
PLAYFIELD = 5
CURSOR = 10
COUNTER = 0
BLOCK = 5  # block size = 5x5
STEP = 5


def _to_np(state: GridLike) -> np.ndarray:
    if isinstance(state, np.ndarray):
        return state.astype(np.uint8, copy=True)
    return np.asarray(state, dtype=np.uint8)


def _is_wall(v: int) -> bool:
    return v == WALL_A or v == WALL_B


def _find_blocks(grid: np.ndarray) -> List[Tuple[int, int]]:
    """Return list of (r, c) topleft corners of each 5x5 block of CURSOR.

    The cursor consists of EXACTLY two 5x5 blocks of colour 10 in every
    observed state. Blocks are returned sorted by column (left first).
    """
    mask = (grid == CURSOR)
    if not mask.any():
        return []
    blocks: List[Tuple[int, int]] = []
    used = np.zeros_like(mask, dtype=bool)
    rows, cols = np.where(mask)
    for r, c in sorted(zip(rows.tolist(), cols.tolist())):
        if used[r, c]:
            continue
        if r + BLOCK <= 64 and c + BLOCK <= 64 and mask[r:r + BLOCK, c:c + BLOCK].all():
            blocks.append((int(r), int(c)))
            used[r:r + BLOCK, c:c + BLOCK] = True
    blocks.sort(key=lambda x: x[1])
    return blocks


def _can_place(grid: np.ndarray, r: int, c: int, ignore_r: int, ignore_c: int) -> bool:
    """Can a 5x5 cursor block be placed at top-left (r, c)?

    Requirements:
    - In bounds (0 <= r, c, r+5 <= 64, c+5 <= 64).
    - The destination 5x5 region contains NO wall cells (11 or 12).
    - Other cells (currently PLAYFIELD or the moving block itself) are OK.
      `ignore_r, ignore_c` is the block's CURRENT topleft (those CURSOR cells
      are treated as background since they will be vacated).
    """
    if r < 0 or c < 0 or r + BLOCK > 64 or c + BLOCK > 64:
        return False
    sub = grid[r:r + BLOCK, c:c + BLOCK]
    # walls anywhere?
    if np.any((sub == WALL_A) | (sub == WALL_B)):
        return False
    return True


def _move_block(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    """Move the 5x5 cursor block from (r0,c0) to (r1,c1).

    Restores the vacated cells to PLAYFIELD (5).  Verified that the cursor
    never sits over the central 11/12 strip, so the vacated background is
    always entirely 5.
    """
    grid[r0:r0 + BLOCK, c0:c0 + BLOCK] = PLAYFIELD
    grid[r1:r1 + BLOCK, c1:c1 + BLOCK] = CURSOR


def _apply_move(grid: np.ndarray, action_id: int) -> bool:
    """Apply one of actions 1..4. Returns True iff grid was modified."""
    blocks = _find_blocks(grid)
    if len(blocks) != 2:
        return False
    tl, br = blocks  # sorted by column

    # Compute desired delta per block
    if action_id == 1:        # UP: both -5 rows
        d_tl = (-STEP, 0)
        d_br = (-STEP, 0)
    elif action_id == 2:      # DOWN: both +5 rows
        d_tl = (STEP, 0)
        d_br = (STEP, 0)
    elif action_id == 3:      # expand horizontally
        d_tl = (0, -STEP)
        d_br = (0, STEP)
    elif action_id == 4:      # contract horizontally
        d_tl = (0, STEP)
        d_br = (0, -STEP)
    else:
        return False

    tl_new = (tl[0] + d_tl[0], tl[1] + d_tl[1])
    br_new = (br[0] + d_br[0], br[1] + d_br[1])

    moved = False

    # Check & move TL block (treat its old cells as vacatable)
    if _can_place(grid, tl_new[0], tl_new[1], tl[0], tl[1]):
        # also forbid overlap with BR block's CURRENT footprint
        # (BR may also move; we test against BR's final position later)
        # For action 4 (contract), make sure the moved TL doesn't overlap BR_new.
        # Check overlap with BR_new bbox
        if not _bbox_overlap(tl_new, br_new):
            _move_block(grid, tl[0], tl[1], tl_new[0], tl_new[1])
            moved = True
            tl = tl_new

    if _can_place(grid, br_new[0], br_new[1], br[0], br[1]):
        if not _bbox_overlap(tl, br_new):
            _move_block(grid, br[0], br[1], br_new[0], br_new[1])
            moved = True

    return moved


def _bbox_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    """Do two 5x5 blocks at top-left a and b share any cell?"""
    return (abs(a[0] - b[0]) < BLOCK) and (abs(a[1] - b[1]) < BLOCK)


def simulate(state: GridLike, action_id: int, x: int, y: int):
    """Predict next state, reward_class, done for game m0r0.

    Parameters
    ----------
    state : 64x64 grid (list[list[int]] or np.ndarray), values 0-12.
    action_id : int in {1, 2, 3, 4, 5, 6}.
    x, y : ints in [0, 64). Only relevant when action_id == 6 (click).
    """
    grid = _to_np(state)
    original = grid.copy()

    if action_id in (1, 2, 3, 4):
        _apply_move(grid, action_id)
    # actions 5 and 6: only tick the row-0 / row-63 counter, which is
    # non-deterministic from a single frame. Leave grid unchanged.

    changed = not np.array_equal(grid, original)
    reward_class = 1 if changed else 0
    done = False
    return grid, reward_class, done
