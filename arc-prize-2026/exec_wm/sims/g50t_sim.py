"""Executable world model for ARC-AGI-3 game `g50t` — v2.

Builds on v1 (sprite movement + path/wall encoding) and adds:

  A. Row-63 countdown tick:
     The visible counter on row 63 (rightmost-9 column flips to 1) advances
     on every other action.  Every single-frame parity feature we tested
     (n_ones in row 63, rightmost-9 col parity, player-position parity,
     panel-colour, action subset) scores 49-55% on a 200-tuple split — the
     parity is genuinely hidden state.  v2 therefore keeps a module-level
     call counter and ticks on odd-indexed calls (i=1,3,5,...).
     reset_state() should be called at episode start.

     Caveat: planning rollouts that re-query the same state twice will
     double-advance the visible counter.  For straight-line play
     (validation harness + real env stepping) this is correct.

  B. Action 5:
     Hidden-state driven.  Observed ndiff distribution is bimodal:
     {0,1} (12/47) -> noop or just a tick; {23,24} (4/47) -> panel
     A<->B swap only; {48,71,72,95,96,153} (31/47) -> panel swap PLUS
     player/ghost sprite swap.  No single-frame feature predicts which
     mode fires.  v2 keeps the identity default for action 5 (12/47
     baseline) and lets the row-63 tick fire as normal.

Combined v1 -> v2 on 200 tuples: state_exact 39.5% -> 73.0%,
pixel_match 99.61 -> 99.62, reward_acc 79 -> 96.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Union

import numpy as np

GridLike = Union[List[List[int]], np.ndarray]

# Sprite geometry
SPRITE_SIZE = 5
GRID_STEP = 6
BG_PATH = 5
SPRITE_COLOR = 9

ACTION_DELTAS = {
    1: (-GRID_STEP, 0),
    2: (+GRID_STEP, 0),
    3: (0, -GRID_STEP),
    4: (0, +GRID_STEP),
}

# Panel rectangles (rows, cols).
_PANEL_L = (slice(1, 8), slice(1, 4))
_PANEL_R = (slice(1, 8), slice(5, 8))

# Panel "shape" templates (binary mask) read off the data:
# both panels carry a 7-row x 3-col blob with the same shape; only colour
# changes between A <-> B.  We detect a panel's current colour by looking
# at the unique non-zero value inside the panel.

# ----------------------- module-level tick state -------------------------
# Counter for the row-63 visible-tick parity.  Incremented on EVERY call
# to simulate(); ticks fire on odd observation indices (i=1,3,5,...).
# Call reset_state() at the start of a new episode.
_CALL_INDEX = 0
_LAST_KEY: Tuple[int, int] | None = None  # exposed for debugging


def _to_np(state: GridLike) -> np.ndarray:
    if isinstance(state, np.ndarray):
        return state.astype(np.uint8, copy=True)
    return np.asarray(state, dtype=np.uint8)


def _find_player(grid: np.ndarray) -> Tuple[int, int] | None:
    nines = (grid == SPRITE_COLOR).astype(np.int32)
    H_, W_ = nines.shape
    cs = np.zeros((H_ + 1, W_ + 1), dtype=np.int32)
    cs[1:, 1:] = nines.cumsum(axis=0).cumsum(axis=1)
    for r in range(H_ - 4):
        for c in range(W_ - 4):
            ws = cs[r + 5, c + 5] - cs[r, c + 5] - cs[r + 5, c] + cs[r, c]
            if ws == 24 and grid[r + 2, c + 2] == BG_PATH:
                if (
                    grid[r, c] == SPRITE_COLOR
                    and grid[r, c + 4] == SPRITE_COLOR
                    and grid[r + 4, c] == SPRITE_COLOR
                    and grid[r + 4, c + 4] == SPRITE_COLOR
                ):
                    return r, c
    return None


def _target_blocked(grid: np.ndarray, tr: int, tc: int) -> bool:
    if tr < 0 or tc < 0 or tr + 5 > grid.shape[0] or tc + 5 > grid.shape[1]:
        return True
    patch = grid[tr:tr + 5, tc:tc + 5]
    return bool(np.all(patch == 0))


def _erase_sprite(grid: np.ndarray, r: int, c: int) -> None:
    grid[r:r + 5, c:c + 5] = BG_PATH


def _stamp_sprite(grid: np.ndarray, r: int, c: int) -> None:
    grid[r:r + 5, c:c + 5] = SPRITE_COLOR
    grid[r + 2, c + 2] = BG_PATH


def _advance_counter(grid: np.ndarray) -> bool:
    """Advance the row-63 visible counter by one tick.

    Returns True if a tick was actually applied.
    The rule: rightmost remaining 9 in row 63 becomes 1.
    If no 9 remains, no further tick is possible.
    """
    row = grid[63]
    nines = np.where(row == SPRITE_COLOR)[0]
    if len(nines) == 0:
        return False
    grid[63, int(nines[-1])] = 1
    return True


# Panel-colour helpers
def _panel_colour(grid: np.ndarray, panel) -> int:
    """Return the dominant non-zero colour in `panel`, or 0 if blank."""
    vals = grid[panel]
    nz = vals[vals != 0]
    if nz.size == 0:
        return 0
    # take mode
    u, c = np.unique(nz, return_counts=True)
    return int(u[int(np.argmax(c))])


def _swap_panels(grid: np.ndarray, new_left: int, new_right: int) -> None:
    """Recolour panels in-place, preserving their non-zero pattern."""
    for panel, col in ((_PANEL_L, new_left), (_PANEL_R, new_right)):
        patch = grid[panel]
        mask = patch != 0
        patch[mask] = col
        grid[panel] = patch


# Stateful key for tick gating.
def _fingerprint(grid: np.ndarray) -> int:
    # cheap content hash; xxhash would be nicer but numpy bytes are fine
    return hash(grid.tobytes())


def reset_state() -> None:
    """Reset module-level tick parity (call once at start of a new episode)."""
    global _CALL_INDEX, _LAST_KEY
    _CALL_INDEX = 0
    _LAST_KEY = None


def simulate(state: GridLike, action_id: int, x: int, y: int):
    """Predict (next_state, reward_class, done) for game g50t.

    See module docstring.  The simulator is *mostly* stateless — only the
    row-63 visible-counter parity is tracked across calls because no
    single-frame feature predicts it (verified across 200 tuples).
    """
    global _CALL_INDEX, _LAST_KEY

    grid = _to_np(state)
    original = grid.copy()

    # --- movement (actions 1..4) ---
    moved = False
    if action_id in ACTION_DELTAS:
        player = _find_player(grid)
        if player is not None:
            pr, pc = player
            dr, dc = ACTION_DELTAS[action_id]
            tr, tc = pr + dr, pc + dc
            if not _target_blocked(grid, tr, tc):
                _erase_sprite(grid, pr, pc)
                _stamp_sprite(grid, tr, tc)
                moved = True
    elif action_id == 5:
        # Action 5 is partly hidden-state driven (n_changed in observed
        # tuples ranges over {0,1,23,24,48,71,72,95,96,153}).  Roughly:
        #   * "no-op" steps (n_changed in {0,1}) leave the grid almost
        #     unchanged — these were the only ones v1 got right.
        #   * "panel-swap" steps (n_changed in {23,24}) alternate the two
        #     panel colourings (config A: L=9/R=1, B: L=2/R=9).
        #   * larger n_changed includes a player/ghost sprite swap that
        #     requires multi-step memory.
        # From a single frame we can't know which mode fires.  v1's
        # identity policy got 8/47 right; aggressively flipping panels in
        # v2 dropped that to 0.  So v2 keeps the identity default for
        # action 5 and only handles the row-63 tick (below).
        pass

    # --- row-63 visible counter tick ---
    # Advance unconditionally.  A `_LAST_KEY` dedup was tried and rejected:
    # it desyncs whenever two consecutive real steps share (state, action),
    # which is common in g50t (e.g. blocked-against-wall repeated moves).
    _CALL_INDEX += 1
    _LAST_KEY = (_fingerprint(original), int(action_id))
    # Observation-stream index `i` = _CALL_INDEX - 1; ticks on odd i.
    do_tick = ((_CALL_INDEX - 1) % 2 == 1)
    if do_tick:
        _advance_counter(grid)

    next_state = grid
    changed = not np.array_equal(next_state, original)

    # Reward: empirically, reward_class == 1 iff state changed.
    if action_id == 5:
        # action-5 reward is harder; v1 used `1` and got the bucket right
        # ~83%.  Keep that.
        reward_class = 1
    elif changed:
        reward_class = 1
    else:
        reward_class = 0
    done = False
    return next_state.tolist(), reward_class, done
