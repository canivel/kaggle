"""Executable world model for ARC-AGI-3 game `wa30` - v2 (conservative).

v2 adds ONLY:
- Target-tile detection (4x4 frames of color 3/4 with interior 2x2 of color 9).
- Tile-as-obstacle: when the sprite's 4x4 proposed move would overlap a target
  tile, treat it like a wall: sprite stays put, only facing rotates. (Same
  behaviour as the grid-edge fallback in v1.)

v2 does NOT attempt to:
- Predict the row-63 step-counter tick: empirically tied to a hidden clock with
  no single-frame proxy (verified by exhaustive c_t%k tests).
- Toggle tile colors on adjacency: aggressive toggle heuristics regressed
  actions 2 and 4 (-6 to -7 abs pts) in pilot runs, so we stay neutral.

Expected gain over v1: +1 to +3 exact-match cases (the 2 tile-collision tuples
plus any sprite-overshoot they cascade into). Reward and pixel-match should be
within noise of v1.
"""
from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np

GridLike = Union[List[List[int]], np.ndarray]

BG = 1
BODY = 14
HEAD = 0
SPRITE_SIZE = 4
TILE_SIZE = 4
TILE_INTERIOR = 9
TILE_COLORS = (3, 4)


def _to_np(state: GridLike) -> np.ndarray:
    if isinstance(state, np.ndarray):
        return state.astype(np.uint8, copy=True)
    return np.asarray(state, dtype=np.uint8)


def _find_sprite(grid: np.ndarray):
    """Return (r0, c0, facing) for the 4x4 sprite, or None if not found.

    r0,c0 is the top-left of the FULL 4x4 bbox; facing in {U,D,L,R}.
    """
    rs14, cs14 = np.where(grid == BODY)
    if len(rs14) == 0:
        return None
    r0_14, r1_14 = int(rs14.min()), int(rs14.max()) + 1
    c0_14, c1_14 = int(cs14.min()), int(cs14.max()) + 1
    h = r1_14 - r0_14
    w = c1_14 - c0_14

    if h == 3 and w == 4:
        if r0_14 - 1 >= 0 and bool((grid[r0_14 - 1, c0_14:c1_14] == HEAD).all()):
            return (r0_14 - 1, c0_14, 'U')
        if r1_14 < 64 and bool((grid[r1_14, c0_14:c1_14] == HEAD).all()):
            return (r0_14, c0_14, 'D')
    elif h == 4 and w == 3:
        if c0_14 - 1 >= 0 and bool((grid[r0_14:r1_14, c0_14 - 1] == HEAD).all()):
            return (r0_14, c0_14 - 1, 'L')
        if c1_14 < 64 and bool((grid[r0_14:r1_14, c1_14] == HEAD).all()):
            return (r0_14, c0_14, 'R')
    return None


def _stamp_sprite(grid: np.ndarray, r0: int, c0: int, facing: str) -> None:
    block = np.full((SPRITE_SIZE, SPRITE_SIZE), BODY, dtype=np.uint8)
    if facing == 'U':
        block[0, :] = HEAD
    elif facing == 'D':
        block[3, :] = HEAD
    elif facing == 'L':
        block[:, 0] = HEAD
    elif facing == 'R':
        block[:, 3] = HEAD
    grid[r0:r0 + SPRITE_SIZE, c0:c0 + SPRITE_SIZE] = block


def _erase_sprite(grid: np.ndarray, r0: int, c0: int) -> None:
    grid[r0:r0 + SPRITE_SIZE, c0:c0 + SPRITE_SIZE] = BG


def _find_tile_bboxes(grid: np.ndarray) -> List[Tuple[int, int]]:
    """Return [(r0, c0), ...] top-left corners of every 4x4 target tile."""
    mask9 = (grid == TILE_INTERIOR)
    if not mask9.any():
        return []
    out = []
    seen = set()
    rs, cs = np.where(mask9)
    for r, c in zip(rs.tolist(), cs.tolist()):
        # Interior 2x2 means tile top-left in (r-2..r-1, c-2..c-1).
        for r0 in (r - 1, r - 2):
            for c0 in (c - 1, c - 2):
                if r0 < 0 or c0 < 0 or r0 + TILE_SIZE > 64 or c0 + TILE_SIZE > 64:
                    continue
                if (r0, c0) in seen:
                    continue
                win = grid[r0:r0 + TILE_SIZE, c0:c0 + TILE_SIZE]
                if not (win[1:3, 1:3] == TILE_INTERIOR).all():
                    continue
                border = np.concatenate([
                    win[0, :], win[3, :], win[1:3, 0], win[1:3, 3]
                ])
                bset = set(int(v) for v in border)
                if len(bset) != 1:
                    continue
                if next(iter(bset)) not in TILE_COLORS:
                    continue
                out.append((r0, c0))
                seen.add((r0, c0))
    return out


def _bbox_overlap(b1, b2) -> bool:
    return not (b1[1] < b2[0] or b2[1] < b1[0] or b1[3] < b2[2] or b2[3] < b1[2])


def _apply_move(grid: np.ndarray, action_id: int) -> None:
    det = _find_sprite(grid)
    if det is None:
        return
    r0, c0, _facing = det

    if action_id == 1:
        dr, dc, new_face = -SPRITE_SIZE, 0, 'U'
    elif action_id == 2:
        dr, dc, new_face = SPRITE_SIZE, 0, 'D'
    elif action_id == 3:
        dr, dc, new_face = 0, -SPRITE_SIZE, 'L'
    elif action_id == 4:
        dr, dc, new_face = 0, SPRITE_SIZE, 'R'
    else:
        return

    new_r = r0 + dr
    new_c = c0 + dc

    # Grid-boundary block.
    if new_r < 0 or new_r + SPRITE_SIZE > 64 or new_c < 0 or new_c + SPRITE_SIZE > 64:
        _stamp_sprite(grid, r0, c0, new_face)
        return

    # Target-tile block: if proposed footprint overlaps any tile, sprite stays
    # put and only rotates (matches grid-edge fallback semantics).
    sp_proposed = (new_r, new_r + SPRITE_SIZE - 1, new_c, new_c + SPRITE_SIZE - 1)
    for tr, tc in _find_tile_bboxes(grid):
        tbb = (tr, tr + TILE_SIZE - 1, tc, tc + TILE_SIZE - 1)
        if _bbox_overlap(sp_proposed, tbb):
            _stamp_sprite(grid, r0, c0, new_face)
            return

    _erase_sprite(grid, r0, c0)
    _stamp_sprite(grid, new_r, new_c, new_face)


def simulate(state: GridLike, action_id: int, x: int, y: int):
    grid = _to_np(state)

    if action_id in (1, 2, 3, 4):
        _apply_move(grid, action_id)
        reward_class = 1
    elif action_id == 5:
        reward_class = 0
    else:
        reward_class = 1

    done = False
    return grid.tolist(), int(reward_class), bool(done)
