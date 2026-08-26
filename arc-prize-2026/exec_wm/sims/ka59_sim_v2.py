"""Executable world model for ARC-AGI-3 game `ka59` -- v2.

Same game model as v1, with two changes:

1. **Counter tick rule, decoupled from playfield change.**  v1 ticked the
   counter iff the move succeeded.  In the 200-tuple data the counter and
   playfield are decoupled: 24/152 move tuples have a NOOP on the playfield
   but still tick the counter; 39/152 have a successful move but no tick.
   The single-frame "always tick on actions 1..4" rule is the best
   majority predictor (97/152 = 64% counter-correct on move actions vs
   89/152 = 58% for v1's tied rule).  Action 6 is also always-tick (modal
   28/48 = 58%) -- unchanged from v1.

2. **Adjacent-to-goal merge animation.**  When the player sprite would
   land such that its 3x3 outline becomes contiguous with the static goal
   sprite (color 14 around a centre = 5), the next state is the merged
   4-zero L-shape rather than a clean 3x3 swap.  We detect this pre-move:
   if the goal centre is at (gr, gc) and the predicted new sprite centre
   is at (gr, gc - 3) (player immediately left of goal) then the next
   state is the special "stretched sprite" shape spanning cols
   nc-1..gc-1 with zeros at (nr-1, nc+1), (nr, nc), (nr, nc+1), (nr+1, nc+1).
   Symmetric rule for sprite immediately right of goal.  Currently only
   the "right of player, goal-on-right" direction was observed in data,
   so we encode only that branch.

Invariants 1, 3, 4 from v1 unchanged. The counter-tick invariant was
relaxed (now always-tick on any non-identity action_id).
"""
from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np

GridLike = Union[List[List[int]], np.ndarray]

SPRITE_COLOR = 14
SPRITE_HOLE = 0
GOAL_CENTER = 5
OUTER_BG = 2
PLAYFIELD_BG = 1
WALL = 15
COUNTER_CELL = 4
COUNTER_TICKED = 0
COUNTER_ROW = 63

ACTION_DELTAS = {
    1: (-3, 0),   # UP
    2: (3, 0),    # DOWN
    3: (0, -3),   # LEFT
    4: (0, 3),    # RIGHT
}


def _to_np(state: GridLike) -> np.ndarray:
    if isinstance(state, np.ndarray):
        return state.astype(np.uint8, copy=True)
    return np.asarray(state, dtype=np.uint8)


def _find_goal_center(grid: np.ndarray):
    """Return (row, col) of the goal centre = unique 5-cell, or None."""
    mask = (grid == GOAL_CENTER)
    if mask.sum() != 1:
        return None
    rows, cols = np.where(mask)
    return int(rows[0]), int(cols[0])


def _find_sprite_center(grid: np.ndarray):
    """Return (row, col) of the unique 0-cell above row 63, or None.

    For the canonical 3x3 sprite. If there are multiple 0s (e.g. 4-zero
    transient adjacent-to-goal state), we fall back to identity.
    """
    mask = (grid[:COUNTER_ROW] == SPRITE_HOLE)
    if mask.sum() != 1:
        return None
    rows, cols = np.where(mask)
    r, c = int(rows[0]), int(cols[0])
    if not (1 <= r <= 61 and 1 <= c <= 62):
        return None
    patch = grid[r - 1:r + 2, c - 1:c + 2]
    if patch.shape != (3, 3):
        return None
    if (patch == SPRITE_COLOR).sum() != 8:
        return None
    return r, c


def _tick_counter(grid: np.ndarray) -> None:
    row = grid[COUNTER_ROW]
    fours = np.where(row == COUNTER_CELL)[0]
    if fours.size == 0:
        return
    grid[COUNTER_ROW, int(fours.max())] = COUNTER_TICKED


def _is_adjacent_goal_merge(grid: np.ndarray, nr: int, nc: int):
    """Detect 'sprite lands immediately left of goal' adjacency.

    The merge happens iff the predicted new sprite centre would be at
    (gr, gc - 3) -- i.e. the sprite's right edge (col nc+1) is one cell
    left of the goal's left edge (col gc-1).  Returns the goal centre
    (gr, gc) on match, else None.
    """
    goal = _find_goal_center(grid)
    if goal is None:
        return None
    gr, gc = goal
    if nr == gr and nc == gc - 3:
        return gr, gc
    return None


def _apply_merge_to_left_of_goal(grid: np.ndarray, r: int, c: int,
                                 nr: int, nc: int, gr: int, gc: int) -> None:
    """Apply the observed merge sprite shape when sprite lands left-of-goal.

    Observed transition (idx=4, etc.):
      input:  sprite 3x3 centred at (r, c), goal 3x3 centred at (gr, gc).
      output: sprite area becomes background (1), and a wider sprite
              outline appears between cols (nc-1) and (gc-1) inclusive
              with zeros at (nr-1, nc+1), (nr, nc), (nr, nc+1), (nr+1, nc+1).
              The goal sprite is unchanged.
    """
    # Clear the original sprite area to playfield bg.
    grid[r - 1:r + 2, c - 1:c + 2] = PLAYFIELD_BG
    # Build the merged sprite-outline rectangle spanning cols nc-1..gc-1
    # (inclusive) at rows nr-1..nr+1.  Fill with 14 first.
    c_lo = nc - 1
    c_hi = gc - 1  # inclusive
    grid[nr - 1:nr + 2, c_lo:c_hi + 1] = SPRITE_COLOR
    # Then place the zero pattern: (nr-1, nc+1), (nr, nc), (nr, nc+1), (nr+1, nc+1).
    grid[nr - 1, nc + 1] = SPRITE_HOLE
    grid[nr, nc] = SPRITE_HOLE
    grid[nr, nc + 1] = SPRITE_HOLE
    grid[nr + 1, nc + 1] = SPRITE_HOLE


def _try_move(grid: np.ndarray, action_id: int) -> bool:
    """Try to perform a sprite move. Returns True if the playfield changed."""
    detection = _find_sprite_center(grid)
    if detection is None:
        return False
    r, c = detection
    dr, dc = ACTION_DELTAS[action_id]
    nr, nc = r + dr, c + dc
    if not (1 <= nr <= 61 and 1 <= nc <= 62):
        return False
    target = grid[nr - 1:nr + 2, nc - 1:nc + 2]
    if np.all(target == OUTER_BG) or np.all(target == WALL):
        return False

    # Adjacent-goal merge animation (sprite lands immediately left of goal).
    merge = _is_adjacent_goal_merge(grid, nr, nc)
    if merge is not None:
        gr, gc = merge
        _apply_merge_to_left_of_goal(grid, r, c, nr, nc, gr, gc)
        return True

    # Regular 3x3 swap.
    sprite_block = grid[r - 1:r + 2, c - 1:c + 2].copy()
    target_block = target.copy()
    grid[r - 1:r + 2, c - 1:c + 2] = target_block
    grid[nr - 1:nr + 2, nc - 1:nc + 2] = sprite_block
    return True


def simulate(state: GridLike, action_id: int, x: int, y: int):
    """Predict next state, reward_class, done for game ka59."""
    grid = _to_np(state)
    original = grid.copy()

    if action_id in ACTION_DELTAS:
        _try_move(grid, action_id)
        # Counter tick is decoupled from move success: always tick.
        _tick_counter(grid)
    elif action_id == 6:
        # Click: modal outcome is counter-tick only.
        _tick_counter(grid)

    reward_class = 0 if np.array_equal(grid, original) else 1
    done = False
    return grid, reward_class, done
