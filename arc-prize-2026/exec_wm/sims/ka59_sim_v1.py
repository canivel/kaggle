"""Executable world model for ARC-AGI-3 game `ka59` -- v1.

Rodionov-style (arXiv:2605.05138): a hand-derived, self-contained Python
function that maps (state, action_id, x, y) -> (next_state, reward_class, done).

Game summary (observed from 200 random-exploration tuples):
- 64x64 grid, values 0-15.
- Outer background = 2. Playfield background = 1. Wall/separator color = 15.
- A 3x3 "player" sprite made of color 14 with a single 0-cell at the centre
  occupies one of a discrete grid of slots (rows in {22,25,28,31,34,37},
  cols in {10,13,16,19,22,25,28,31}, step = 3).
- A second 3x3 "goal" sprite (color 14 with center=5) is static.
- Actions 1,2,3,4 move the player UP, DOWN, LEFT, RIGHT by 3 cells.
  Movement is a clean swap of two 3x3 blocks: the sprite block and the
  3x3 block immediately in the move direction.
- Move is blocked (NOOP, reward=0) iff the target 3x3 is entirely outer
  background (2) or entirely wall (15).
- Row 63 is a monotonic step counter that DECREMENTS: it starts as all 4s
  and the rightmost remaining 4 ticks to 0 on each "active" frame.
  Empirically the counter ticks ~64% of the time after any rewarded move,
  but the precise trigger depends on hidden state -- this is the dominant
  unmodelled noise. v1 always ticks the counter when the action produced a
  change (reward=1) and never ticks on NOOP.
- Action 6 (click): 28/48 ticked the counter only, 20/48 were full NOOPs.
  The click trigger is hidden; v1 always ticks the counter for action 6
  (the modal correct outcome).
- reward_class = 0 iff the state is exactly unchanged. reward_class = 1
  otherwise.

Invariants used by this simulator:
1. The unique 0-cell in rows 0..62 IS the sprite centre (~97% of states
   have exactly one such cell -- when it has 0 or >1 we fall back to a
   conservative counter-only tick).
2. The 4 cardinal moves swap the sprite 3x3 with the adjacent 3x3 in the
   move direction (verified on 100% of n_changed=18 transitions).
3. NOOP occurs iff the target 3x3 is uniformly outer-background (2) or
   uniformly wall (15) (verified on all 13 observed NOOP transitions).
4. Row 63 counter is monotonically depleting from the right: tick =
   replace the rightmost 4 with 0.
"""
from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np

GridLike = Union[List[List[int]], np.ndarray]

# --- Constants ----------------------------------------------------------
SPRITE_COLOR = 14
SPRITE_HOLE = 0  # player has 0 at centre
OUTER_BG = 2
PLAYFIELD_BG = 1
WALL = 15
COUNTER_CELL = 4
COUNTER_TICKED = 0
COUNTER_ROW = 63

# Action -> (drow, dcol) for the sprite centre
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


def _find_sprite_center(grid: np.ndarray):
    """Return (row, col) of the unique 0-cell above row 63, or None.

    The player sprite is a 3x3 of color 14 with a 0 at the centre. In 97%
    of observed states the only 0-valued cell in rows 0..62 is the sprite
    centre.  When multiple 0s appear (rare transient state next to the
    goal), we bail out and apply identity.
    """
    mask = (grid[:COUNTER_ROW] == SPRITE_HOLE)
    if mask.sum() != 1:
        return None
    rows, cols = np.where(mask)
    r, c = int(rows[0]), int(cols[0])
    # Sanity: 3x3 around it must lie in the grid and be all 14s except centre.
    if not (1 <= r <= 61 and 1 <= c <= 62):
        return None
    patch = grid[r - 1:r + 2, c - 1:c + 2]
    if patch.shape != (3, 3):
        return None
    if (patch == SPRITE_COLOR).sum() != 8:
        return None
    return r, c


def _tick_counter(grid: np.ndarray) -> None:
    """Replace the rightmost remaining 4 on row 63 with 0.

    If no 4 remains (counter exhausted), leave row 63 unchanged. The
    observed counter monotonically depletes from the right and we never
    saw it wrap in 200 tuples.
    """
    row = grid[COUNTER_ROW]
    fours = np.where(row == COUNTER_CELL)[0]
    if fours.size == 0:
        return
    grid[COUNTER_ROW, int(fours.max())] = COUNTER_TICKED


def _try_move(grid: np.ndarray, action_id: int) -> bool:
    """Try to perform a sprite move. Returns True if the playfield changed.

    Mutates `grid` in place.
    """
    detection = _find_sprite_center(grid)
    if detection is None:
        return False
    r, c = detection
    dr, dc = ACTION_DELTAS[action_id]
    nr, nc = r + dr, c + dc
    # target 3x3 must fit in rows [0, 62] and cols [0, 63]
    if not (1 <= nr <= 61 and 1 <= nc <= 62):
        return False
    target = grid[nr - 1:nr + 2, nc - 1:nc + 2]
    # NOOP if target is uniformly outer background OR uniformly wall.
    if np.all(target == OUTER_BG) or np.all(target == WALL):
        return False
    # Otherwise: swap the two 3x3 blocks.
    sprite_block = grid[r - 1:r + 2, c - 1:c + 2].copy()
    target_block = target.copy()
    grid[r - 1:r + 2, c - 1:c + 2] = target_block
    grid[nr - 1:nr + 2, nc - 1:nc + 2] = sprite_block
    return True


def simulate(state: GridLike, action_id: int, x: int, y: int):
    """Predict next state, reward_class, done for game ka59.

    Parameters
    ----------
    state : 64x64 grid (list[list[int]] or np.ndarray), values 0-15.
    action_id : int in {1, 2, 3, 4, 6}.
    x, y : ints in [0, 64). Only relevant when action_id == 6 (click).

    Returns
    -------
    (next_state, reward_class, done)
        next_state : np.ndarray uint8 (64, 64)
        reward_class : 0 if state unchanged, 1 otherwise.
        done : bool (always False -- no terminal state was observed).
    """
    grid = _to_np(state)
    original = grid.copy()

    if action_id in ACTION_DELTAS:
        moved = _try_move(grid, action_id)
        if moved:
            _tick_counter(grid)
        # if not moved -> identity (NOOP, reward 0)
    elif action_id == 6:
        # Click: the underlying trigger is hidden. The modal outcome
        # (28/48) is a counter-only tick; the next-most-common (20/48)
        # is a full NOOP. Predict the modal: tick the counter.
        _tick_counter(grid)
    # Other actions: leave as identity.

    reward_class = 0 if np.array_equal(grid, original) else 1
    done = False
    return grid, reward_class, done
