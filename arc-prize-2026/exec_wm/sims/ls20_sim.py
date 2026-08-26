"""Executable world model for ARC-AGI-3 game `ls20` -- v2.

Adds counter-refill / score-decrement / sprite-respawn handling on top of v1.

v1 covered:
  - 5-cell cardinal sprite move with wall NOOP and lane-line restoration.
  - Leftmost (61,c)=(62,c)=11 pair -> (3,3) tick.

v2 additions (4 remaining mismatches in v1 -- steps 42, 85, 128, 171):

  Counter band cols 13..54 of rows 61-62 holds the step counter
  (11 = pending, 3 = ticked). Right-margin cols 55..63 of rows 61-62
  holds 3 score pairs at cols 56-57, 59-60, 62-63 separated by color-5
  dividers (cols 55, 58, 61). Each score-pair starts as (8,8) and is
  consumed -> (3,3).

  Trigger (observable from input state alone):
    The counter is fully ticked iff all cells in
    (61, 13..54) and (62, 13..54) equal 3.

  When triggered, in addition to the v1 sprite/tick logic, we:
    1. Consume the RIGHTMOST remaining 8-pair in the score margin.
       Pair positions, right-to-left: (62-63), (59-60), (56-57).
       Both rows 61 and 62 of that pair flip 8 -> 3.
    2. If at least one 8-pair remains after step 1, REFILL the
       counter: rows 61-62 cols 13..54 -> 11.  AND respawn the
       sprite at fixed point (45, 34): erase old sprite (lane-line
       restore as in v1), stamp new sprite at (45, 34).
    3. Skip the v1 sprite-move logic (because respawn overrides it).
    4. Skip the v1 counter-tick logic (because we either refilled, or
       we're leaving the counter all-3 because no refill triggered).

  When triggered but NO pairs remain after consumption (final pair):
    - Consume rightmost pair as above.
    - Do NOT refill, do NOT respawn. Sprite behaves per v1 (will
      typically NOOP since the level is over / walled in).
    - Do NOT run v1 counter-tick (counter is already all-3).

Empirical support (200 observed tuples):
  - Step 42 (act 1, score 8,8,8 -> 8,8,3): refill+respawn -> (45,34).
  - Step 85 (act 4, score 8,8,3 -> 8,3,3): refill+respawn -> (45,34).
  - Step 171 (act 1, score 8,8,8 -> 8,8,3): refill+respawn -> (45,34).
  - Step 128 (act 1, score 8,3,3 -> 3,3,3): consume only, sprite NOOP.

Reward is constantly 1; done constantly False; level constant 0.
"""
from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np

GridLike = Union[List[List[int]], np.ndarray]

# --- Constants ----------------------------------------------------------
SPRITE_COLOR_TOP = 12
SPRITE_COLOR_BOT = 9
PLAYFIELD_BG = 3
LANE_LINE = 5
WALL = 4
COUNTER_FILL = 11
COUNTER_OFF = 3
SCORE_FULL = 8

# Sprite is 5x5: 2 rows of 12, 3 rows of 9.
SPRITE_TEMPLATE = np.array(
    [
        [SPRITE_COLOR_TOP] * 5,
        [SPRITE_COLOR_TOP] * 5,
        [SPRITE_COLOR_BOT] * 5,
        [SPRITE_COLOR_BOT] * 5,
        [SPRITE_COLOR_BOT] * 5,
    ],
    dtype=np.uint8,
)

# Action -> (dr, dc) per move.
DELTAS = {1: (-5, 0), 2: (5, 0), 3: (0, -5), 4: (0, 5)}

# Counter band lives on these rows.
COUNTER_ROW_A = 61
COUNTER_ROW_B = 62
COUNTER_C0 = 13  # inclusive
COUNTER_C1 = 55  # exclusive  -> cols 13..54

# Score-pair column positions (left index of each (c, c+1) pair),
# ordered RIGHT-TO-LEFT so the first remaining pair is consumed first.
SCORE_PAIRS = [62, 59, 56]

# Sprite respawn position after refill.
RESPAWN_RC = (45, 34)


def _to_np(state: GridLike) -> np.ndarray:
    if isinstance(state, np.ndarray):
        return state.astype(np.uint8, copy=True)
    return np.asarray(state, dtype=np.uint8)


def _find_sprite(grid: np.ndarray) -> Tuple[int, int] | None:
    """Return (sprite_row, sprite_col) top-left, or None if sprite missing."""
    mask = grid == SPRITE_COLOR_TOP
    if not mask.any():
        return None
    rs, cs = np.where(mask)
    return int(rs.min()), int(cs.min())


def _row_restore_color(grid: np.ndarray, r: int, sc: int) -> int:
    """Per-row background color when the sprite vacates (r, sc..sc+5)."""
    if 0 <= sc - 1 < 64:
        v = int(grid[r, sc - 1])
        if v in (PLAYFIELD_BG, LANE_LINE):
            return v
    if 0 <= sc + 5 < 64:
        v = int(grid[r, sc + 5])
        if v in (PLAYFIELD_BG, LANE_LINE):
            return v
    return PLAYFIELD_BG


def _erase_sprite(grid: np.ndarray, sr: int, sc: int) -> None:
    for i in range(5):
        bg = _row_restore_color(grid, sr + i, sc)
        grid[sr + i, sc : sc + 5] = bg


def _stamp_sprite(grid: np.ndarray, sr: int, sc: int) -> None:
    grid[sr : sr + 5, sc : sc + 5] = SPRITE_TEMPLATE


def _try_move_sprite(grid: np.ndarray, action_id: int) -> None:
    """Apply the sprite-move rule. NOOP if destination clips a wall (4)
    or falls outside the playfield."""
    if action_id not in DELTAS:
        return
    detection = _find_sprite(grid)
    if detection is None:
        return
    sr, sc = detection
    dr, dc = DELTAS[action_id]
    nr, nc = sr + dr, sc + dc
    if not (0 <= nr <= 59 and 0 <= nc <= 59):
        return
    dest_patch = grid[nr : nr + 5, nc : nc + 5]
    if dest_patch.shape != (5, 5):
        return
    if (dest_patch == WALL).any():
        return
    _erase_sprite(grid, sr, sc)
    _stamp_sprite(grid, nr, nc)


def _tick_counter(grid: np.ndarray) -> bool:
    """Tick leftmost (61,c)=(62,c)=11 -> (3,3). Return True if a tick happened."""
    row_a = grid[COUNTER_ROW_A]
    row_b = grid[COUNTER_ROW_B]
    both = (row_a == COUNTER_FILL) & (row_b == COUNTER_FILL)
    idx = np.where(both)[0]
    if idx.size == 0:
        return False
    c = int(idx[0])
    grid[COUNTER_ROW_A, c] = COUNTER_OFF
    grid[COUNTER_ROW_B, c] = COUNTER_OFF
    return True


def _counter_is_fully_ticked(grid: np.ndarray) -> bool:
    """All 84 counter cells == 3."""
    band = grid[COUNTER_ROW_A : COUNTER_ROW_B + 1, COUNTER_C0:COUNTER_C1]
    return bool((band == COUNTER_OFF).all())


def _refill_counter(grid: np.ndarray) -> None:
    grid[COUNTER_ROW_A : COUNTER_ROW_B + 1, COUNTER_C0:COUNTER_C1] = COUNTER_FILL


def _count_remaining_score_pairs(grid: np.ndarray) -> int:
    n = 0
    for c in SCORE_PAIRS:
        if (
            grid[COUNTER_ROW_A, c] == SCORE_FULL
            and grid[COUNTER_ROW_A, c + 1] == SCORE_FULL
            and grid[COUNTER_ROW_B, c] == SCORE_FULL
            and grid[COUNTER_ROW_B, c + 1] == SCORE_FULL
        ):
            n += 1
    return n


def _consume_rightmost_score_pair(grid: np.ndarray) -> bool:
    """Flip rightmost 8,8/8,8 score pair -> 3,3/3,3. Return True if consumed."""
    for c in SCORE_PAIRS:  # right-to-left
        if (
            grid[COUNTER_ROW_A, c] == SCORE_FULL
            and grid[COUNTER_ROW_A, c + 1] == SCORE_FULL
            and grid[COUNTER_ROW_B, c] == SCORE_FULL
            and grid[COUNTER_ROW_B, c + 1] == SCORE_FULL
        ):
            grid[COUNTER_ROW_A, c] = COUNTER_OFF
            grid[COUNTER_ROW_A, c + 1] = COUNTER_OFF
            grid[COUNTER_ROW_B, c] = COUNTER_OFF
            grid[COUNTER_ROW_B, c + 1] = COUNTER_OFF
            return True
    return False


def _respawn_sprite(grid: np.ndarray) -> None:
    """Erase any current sprite and stamp a fresh one at RESPAWN_RC."""
    detection = _find_sprite(grid)
    if detection is not None:
        sr, sc = detection
        _erase_sprite(grid, sr, sc)
    _stamp_sprite(grid, *RESPAWN_RC)


def simulate(state: GridLike, action_id: int, x: int, y: int):
    """Predict next state, reward_class, done for game ls20.

    Returns
    -------
    (next_state: np.ndarray (64,64) uint8, reward_class: int, done: bool)
    """
    grid = _to_np(state)
    action_id = int(action_id)

    if _counter_is_fully_ticked(grid):
        # Special: counter full. Consume rightmost remaining 8-pair.
        n_pairs_before = _count_remaining_score_pairs(grid)
        consumed = _consume_rightmost_score_pair(grid)
        # If >=1 pair remains after consumption, refill + respawn.
        if consumed and n_pairs_before >= 2:
            _refill_counter(grid)
            _respawn_sprite(grid)
            return grid, 1, False
        # Otherwise: no refill, no respawn. Run normal sprite-move
        # (often NOOP since the level is effectively over).
        _try_move_sprite(grid, action_id)
        # Do NOT tick: counter is already all-3, _tick_counter is a no-op.
        return grid, 1, False

    # Normal step: v1 logic.
    _try_move_sprite(grid, action_id)
    _tick_counter(grid)
    return grid, 1, False
