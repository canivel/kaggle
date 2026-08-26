"""Executable world model for ARC-AGI-3 game `ft09` — v1.

Rodionov-style (arXiv:2605.05138): a hand-derived, self-contained Python
function that maps (state, action_id, x, y) -> (next_state, reward_class, done).

Game ft09 description
---------------------
A click-only puzzle (only action 6 is available). The playfield contains a
3x3 grid of 6x6 colour tiles centered on rows {36..41, 44..49, 52..57} and
cols {36..41, 44..49, 52..57}. Each outer tile is uniformly colour 8 or 9.
The centre tile (1,1) holds a static cross-shaped marker (values {0,2,8})
and is inert. Row 63 is a horizontal counter of value 12s decreasing from
the right end; each successful click consumes the two rightmost 12s and
turns them into 11s.

Empirically observed invariants (200 tuples)
--------------------------------------------
- Only action 6 is ever available. action_distribution = {6: 200}.
- reward_class = 0 for 185 tuples (NOOPs), 1 for 15 tuples (tile flips).
- done is False in all 200 tuples; level stays at 0.
- 15/15 successful flips: click (y,x) landed inside one of the 8 outer
  6x6 tiles. The entire tile flipped colour (8<->9, uniform before/after).
- 2/2 inside-noops: click landed inside the inert centre tile (1,1).
- 0/200 outer-tile clicks resulted in a NOOP.
- 183/200 NOOPs: click landed outside the 3x3 tile grid.
- On every flip, row 63's two rightmost remaining 12-cells become 11.
- No flip ever produced a row-63 12->11 longer than 2 cells, and no row-63
  cells ever transitioned 11->12 (counter monotonically depletes).

Simulator behaviour
-------------------
1. If action_id != 6: return identity (no other actions exist in the game).
2. Compute (tile_row, tile_col) for the click (y, x) using the fixed
   6x6 tile grid.
3. If the click is outside every tile, or inside the centre tile (1,1):
   return identity (reward 0, done False, no change).
4. Otherwise flip the tile uniformly between 8 and 9 (use the current
   value at (y,x) to decide direction: 8->9 or 9->8), consume the two
   rightmost 12s on row 63 (12->11), and return reward 1, done False.
"""
from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np

GridLike = Union[List[List[int]], np.ndarray]

# --- Constants ----------------------------------------------------------
# Fixed 3x3 grid of 6x6 tiles. Each tuple is (row_start, row_end_inclusive).
TILE_ROWS: Tuple[Tuple[int, int], ...] = ((36, 41), (44, 49), (52, 57))
TILE_COLS: Tuple[Tuple[int, int], ...] = ((36, 41), (44, 49), (52, 57))

CENTER_TILE = (1, 1)  # inert "X" tile

COUNTER_ROW = 63
COUNTER_FULL = 12
COUNTER_USED = 11
COUNTER_CELLS_PER_FLIP = 2


def _to_np(state: GridLike) -> np.ndarray:
    if isinstance(state, np.ndarray):
        return state.astype(np.uint8, copy=True)
    return np.asarray(state, dtype=np.uint8)


def _locate_tile(y: int, x: int) -> Tuple[int, int] | None:
    """Return (tile_row, tile_col) if (y,x) lies inside the 3x3 tile grid."""
    if not (0 <= y < 64 and 0 <= x < 64):
        return None
    for ri, (r0, r1) in enumerate(TILE_ROWS):
        if r0 <= y <= r1:
            for ci, (c0, c1) in enumerate(TILE_COLS):
                if c0 <= x <= c1:
                    return ri, ci
            return None
    return None


def _consume_counter(grid: np.ndarray, n: int = COUNTER_CELLS_PER_FLIP) -> None:
    """Turn the n rightmost COUNTER_FULL cells on row 63 into COUNTER_USED."""
    row = grid[COUNTER_ROW]
    full_cols = np.where(row == COUNTER_FULL)[0]
    if full_cols.size == 0:
        return
    # rightmost n cells
    take = full_cols[-n:]
    grid[COUNTER_ROW, take] = COUNTER_USED


def simulate(state: GridLike, action_id: int, x: int, y: int):
    """Predict (next_state, reward_class, done) for ft09.

    Parameters
    ----------
    state : 64x64 grid (list[list[int]] or np.ndarray), values 0-15.
    action_id : int. Only 6 is meaningful for ft09.
    x : int in [0, 64) — column of the click.
    y : int in [0, 64) — row of the click.
    """
    grid = _to_np(state)
    reward_class = 0
    done = False

    if action_id != 6:
        # No other actions exist in ft09; return identity.
        return grid, reward_class, done

    tile = _locate_tile(y, x)
    if tile is None or tile == CENTER_TILE:
        # Click outside the tile grid, or on the inert centre tile.
        return grid, reward_class, done

    ri, ci = tile
    r0, r1 = TILE_ROWS[ri]
    c0, c1 = TILE_COLS[ci]

    # Flip the tile uniformly between 8 and 9. Use the value at the clicked
    # cell to decide the direction. If the tile is somehow not uniformly 8/9
    # we still proceed by copying the new value across the whole 6x6 patch,
    # matching the observed behaviour (tiles are always uniform pre-flip).
    cur_val = int(grid[y, x])
    if cur_val == 8:
        new_val = 9
    elif cur_val == 9:
        new_val = 8
    else:
        # Defensive: should not happen on outer tiles in observed data.
        # Fall back to identity to avoid corrupting pixel-match.
        return grid, 0, done

    grid[r0:r1 + 1, c0:c1 + 1] = new_val
    _consume_counter(grid, COUNTER_CELLS_PER_FLIP)

    reward_class = 1
    return grid, reward_class, done
