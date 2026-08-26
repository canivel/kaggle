"""Executable world model for ARC-AGI-3 game `cd82` -- v2.

Adds deterministic rotation rendering on top of v1's tick + identity baseline.

v1 invariants (kept):
  1. Counter row 63: rightmost `4` -> `5` on every action.
  2. reward_class = 1 unconditionally.
  3. done = False, level = 0.

v2 NEW invariant (from data, 73/73 rotations verified):
  4. ROTATION: for actions 1-4, when the current frame contains a known
     2-bordered shape (one of 8 canonical poses), the next-frame change
     pattern is fully determined by (action_id, current_shape_norm).

     We mined the lookup table by:
       - normalize the 2-cells to (r0, c0) origin -> shape_norm
       - for each (action, shape_norm) observed, record the rel-cell
         diff (200 cells, excluding row 63).
       - verified all 16 distinct (action, shape_norm) keys give a
         100%-consistent diff across all 73 rotation tuples.

     Apply: detect 2-shape in current frame, look up (aid, shape_norm),
     overlay the rel-cell diff onto the predicted grid.

What v2 still does NOT model:
  - Action 5 paint sub-region (7 cases): the painted band depends on
    hidden cursor / progress state.
  - Action 6 click: 50/50 tick-vs-noop with no observable trigger.
  - Per-action NOOP detection: the 60% tick rate is hidden.
  - The 24/73 rotation cases where the engine did NOT tick the counter
    in addition to rotating (we still tick, which costs 1 cell of pixel
    match on those 24 cases).

Expected metrics (estimated):
  state_exact: 74 (tick-only) + 49 (rotation+tick) = 123/200 = 61.5%
    (vs v1's 37%).
  pixel_match: rotation cases now pixel-exact (200 cells) instead of
    leaving the playfield untouched. Net gain ~73 * 200 / (200*4096)
    = 1.78pp. Expected ~99.9%.
  reward_acc, done_acc unchanged (77%, 100%).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

import numpy as np

GridLike = Union[List[List[int]], np.ndarray]

COUNTER_ROW = 63
COUNTER_FILLED = 4
COUNTER_TICKED = 5

# Load the 44 KB lookup table from a sidecar data file.
# Keyed by (action_id, normalized_2cells_tuple) ->
#   list of (dr, dc, value_after) cells (positions relative to the
#   current frame's 2-bbox top-left corner (r0, c0); excludes row 63).
_THIS_DIR = Path(__file__).resolve().parent
_DATA_PATH = _THIS_DIR / "cd82_rotation_table_data.py"
_ns: dict = {}
exec(_DATA_PATH.read_text(), _ns)
ROTATION_TABLE: dict = _ns["ROTATION_TABLE"]


def _to_np(state: GridLike) -> np.ndarray:
    if isinstance(state, np.ndarray):
        return state.astype(np.uint8, copy=True)
    return np.asarray(state, dtype=np.uint8)


def _tick_counter(grid: np.ndarray) -> bool:
    row = grid[COUNTER_ROW]
    fours = np.where(row == COUNTER_FILLED)[0]
    if fours.size == 0:
        return False
    grid[COUNTER_ROW, int(fours[-1])] = COUNTER_TICKED
    return True


def _shape_norm_and_origin(grid: np.ndarray):
    """Return (normalized_2cells_tuple, r0, c0) for the 2-bordered shape.

    Returns (None, 0, 0) if there are no 2-cells.
    Normalization: translate to top-left = (0, 0) and sort lexicographically.
    """
    twos = np.where(grid == 2)
    if twos[0].size == 0:
        return None, 0, 0
    r0 = int(twos[0].min())
    c0 = int(twos[1].min())
    norm = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in zip(twos[0], twos[1])))
    return norm, r0, c0


def _apply_rotation(grid: np.ndarray, aid: int) -> bool:
    """If the current frame's 2-shape matches a known (aid, shape_norm),
    apply the recorded diff.  Returns True if a rotation was applied."""
    if aid not in (1, 2, 3, 4):
        return False
    norm, r0, c0 = _shape_norm_and_origin(grid)
    if norm is None:
        return False
    cells = ROTATION_TABLE.get((aid, norm))
    if cells is None:
        return False
    H, W = grid.shape
    for dr, dc, val in cells:
        rr = r0 + dr
        cc = c0 + dc
        if 0 <= rr < H and 0 <= cc < W and rr != COUNTER_ROW:
            grid[rr, cc] = val
    return True


def simulate(state: GridLike, action_id: int, x: int, y: int):
    """Predict next state, reward_class, done for game cd82.

    Parameters
    ----------
    state : 64x64 grid (list[list[int]] or np.ndarray), values 0-15.
    action_id : int in {1,2,3,4,5,6}.
    x, y : ints in [0, 64). Only relevant when action_id == 6 (click).
    """
    grid = _to_np(state)

    # Deterministic counter tick (matches ~60% of all transitions; the
    # only single-frame-derivable change for non-rotation cases).
    _tick_counter(grid)

    # Deterministic rotation rendering (16-entry lookup, 100% verified
    # on the 73 rotation tuples in the training observations).
    _apply_rotation(grid, int(action_id))

    reward_class = 1
    done = False
    return grid, reward_class, done
