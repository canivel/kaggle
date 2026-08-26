"""Executable world model for ARC-AGI-3 game `bp35` — v2.

Rodionov-style (arXiv:2605.05138): a hand-derived, self-contained Python
function that maps (state, action_id, x, y) -> (next_state, reward_class, done).

v2 changes vs v1:
- Detect the 5x5 sprite at rows 37-41 by anchoring on color 11 (which appears
  ONLY on the sprite in every observed state — verified across all 200 tuples).
- Determine sprite_left and facing (L or R) from the 11-cell position.
- For action 3: move sprite -6 cols, force L-facing. NOOP if sprite_left == 13.
- For action 4: move sprite +6 cols, force R-facing. NOOP if sprite_left == 49.
- Vacate old sprite cells to background (10) and stamp the new sprite with
  the appropriate L or R template.
- Action 6 and 7 keep v1 behaviour (counter-only). Action 7 simple-swap
  direction depends on hidden state we cannot recover from a single frame.

Empirical observations (full diagnosis in bp35_notes_v2.md):
- Sprite anchor: color 11 occurs ONLY inside the 5x5 sprite at rows 38-39
  (2 cells per row when L-facing the 11s are at col offset 1; when R-facing
  the 11s are at col offset 3).
- Sprite slots: sprite_left in {13, 19, 25, 31, 37, 43, 49} (step = 6).
- Action 3 simple-swap: 33/33 cases land the sprite at sprite_left - 6 with
  L-facing template. Action 3 NOOP: 3/3 cases were at sprite_left == 13.
- Action 4 simple-swap: 37/37 cases land at sprite_left + 6 R-facing.
  Action 4 NOOP: 1/1 case was at sprite_left == 49.
- Counter row 63: monotonic step counter (every action ticks, leftmost 0
  becomes 15). Once row 63 is full of 15s, the row clears and restarts —
  we model the simple "leftmost 0 -> 15" rule and accept that wraps
  occasionally mis-predict.
- A "score display" lives at rows 57-62 and refreshes after certain
  counter ticks. The trigger is non-deterministic from a single frame
  (likely depends on hidden game state). v2 does NOT model this; this
  costs us ~8% of action-3/4 cases that have score-display updates.
"""
from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np

GridLike = Union[List[List[int]], np.ndarray]

# --- Constants ----------------------------------------------------------
BG_PLAYFIELD = 10
SPRITE_ROW0 = 37
SPRITE_ROW1 = 42  # exclusive (rows 37..41)
SPRITE_W = 5
SPRITE_H = 5

# Valid sprite_left positions (slots of width 6, from col 13 to col 49)
SPRITE_LEFT_MIN = 13
SPRITE_LEFT_MAX = 49
SPRITE_STEP = 6

# The canonical R-facing sprite template (verified against all observed
# R-facing positions in 200 tuples).
SPRITE_R_TEMPLATE = np.array([
    [5, 5, 9, 5, 5],
    [5, 9, 9, 11, 5],
    [5, 9, 9, 11, 5],
    [5, 5, 9, 5, 5],
    [10, 5, 5, 5, 10],
], dtype=np.uint8)

# L-facing template = R flipped horizontally
SPRITE_L_TEMPLATE = SPRITE_R_TEMPLATE[:, ::-1].copy()


def _to_np(state: GridLike) -> np.ndarray:
    if isinstance(state, np.ndarray):
        return state.astype(np.uint8, copy=True)
    return np.asarray(state, dtype=np.uint8)


def _advance_step_counter(grid: np.ndarray) -> None:
    """Set the leftmost 0-valued cell on row 63 to 15.

    If row 63 has no 0s (full counter), wrap: clear row 63 and set col 0
    to 15. (Matches the observed wrap behaviour at counter==64.)
    """
    row = grid[63]
    zeros = np.where(row == 0)[0]
    if zeros.size == 0:
        grid[63, :] = 0
        grid[63, 0] = 15
        return
    grid[63, zeros[0]] = 15


def _find_sprite(grid: np.ndarray) -> Tuple[int, str] | None:
    """Return (sprite_left, facing) or None if no sprite detected.

    Facing is 'R' if the rightmost 11 occurs at col offset 3 from
    sprite_left, 'L' if the leftmost 11 occurs at col offset 1.
    Returns None if the sprite cannot be uniquely identified.
    """
    sprite_band = grid[SPRITE_ROW0:SPRITE_ROW1]
    mask = (sprite_band == 11)
    if not mask.any():
        return None
    cols = np.where(mask)[1]
    # The 11s form a single 1-col vertical strip at either sprite_left+1
    # (L-facing) or sprite_left+3 (R-facing).  When the playfield is dense
    # and multiple sprites have been stamped (n_changed > 47 cases), there
    # may be additional 11-cells — but those land on rows other than 38/39
    # (verified by inspection).  We restrict to the unique col in rows 38-39.
    mid = (grid[38:40] == 11)
    if not mid.any():
        return None
    mid_cols = np.where(mid)[1]
    # Expect a single unique col with 2 hits (rows 38 and 39)
    unique = np.unique(mid_cols)
    if len(unique) != 1:
        # Multiple candidates — ambiguous, bail to avoid wrong prediction
        return None
    col_11 = int(unique[0])

    # Try R-facing: sprite_left = col_11 - 3
    # Try L-facing: sprite_left = col_11 - 1
    for facing, off in (('R', 3), ('L', 1)):
        sl = col_11 - off
        if not (SPRITE_LEFT_MIN <= sl <= SPRITE_LEFT_MAX):
            continue
        template = SPRITE_R_TEMPLATE if facing == 'R' else SPRITE_L_TEMPLATE
        patch = grid[SPRITE_ROW0:SPRITE_ROW1, sl:sl + SPRITE_W]
        if patch.shape != template.shape:
            continue
        if np.array_equal(patch, template):
            return sl, facing
    return None


def _erase_sprite(grid: np.ndarray, sl: int) -> None:
    """Erase the 5x5 sprite at (rows 37..41, cols sl..sl+4) to background 10."""
    grid[SPRITE_ROW0:SPRITE_ROW1, sl:sl + SPRITE_W] = BG_PLAYFIELD


def _stamp_sprite(grid: np.ndarray, sl: int, facing: str) -> None:
    """Stamp the sprite at (rows 37..41, cols sl..sl+4) with given facing."""
    template = SPRITE_R_TEMPLATE if facing == 'R' else SPRITE_L_TEMPLATE
    grid[SPRITE_ROW0:SPRITE_ROW1, sl:sl + SPRITE_W] = template


def _apply_action_3(grid: np.ndarray) -> None:
    """Action 3: move sprite -6 cols, force L-facing."""
    detection = _find_sprite(grid)
    if detection is None:
        return  # cannot identify sprite, skip the move
    sl, _facing = detection
    new_sl = sl - SPRITE_STEP
    if new_sl < SPRITE_LEFT_MIN:
        # NOOP: sprite cannot move further left. Empirically the sprite
        # also keeps its current facing in this case.
        return
    _erase_sprite(grid, sl)
    _stamp_sprite(grid, new_sl, 'L')


def _apply_action_4(grid: np.ndarray) -> None:
    """Action 4: move sprite +6 cols, force R-facing."""
    detection = _find_sprite(grid)
    if detection is None:
        return
    sl, _facing = detection
    new_sl = sl + SPRITE_STEP
    if new_sl > SPRITE_LEFT_MAX:
        return
    _erase_sprite(grid, sl)
    _stamp_sprite(grid, new_sl, 'R')


def simulate(state: GridLike, action_id: int, x: int, y: int):
    """Predict next state, reward_class, done for game bp35.

    Parameters
    ----------
    state : 64x64 grid (list[list[int]] or np.ndarray), values 0-15.
    action_id : int in {3, 4, 6, 7}.
    x, y : ints in [0, 64). Only relevant when action_id == 6 (click).
    """
    grid = _to_np(state)

    if action_id == 3:
        _apply_action_3(grid)
    elif action_id == 4:
        _apply_action_4(grid)
    elif action_id == 7:
        # Action 7 swap direction is non-deterministic from a single frame
        # (depends on hidden facing-flip state). v1 already gets 32% by
        # leaving the playfield untouched (= correct on the 16 NOOP cases).
        # We leave this unchanged in v2.
        pass
    elif action_id == 6:
        # 48/51 clicks only tick the counter. The remaining 3 stamp a 5x5
        # sprite-shaped block at the click site, but the trigger condition
        # is unclear. Leave playfield untouched.
        pass

    _advance_step_counter(grid)

    # Reward and done are constant across all 200 observed tuples.
    reward_class = 1
    done = False
    return grid, reward_class, done
