"""Executable world model for ARC-AGI-3 game `cn04` — v2.

v2 over v1 (Rodionov-style, arXiv:2605.05138):
- Adds action-5 rule: 90-degree clockwise rotation of the playable blob
  about its bbox top-left corner. Verified 11/11 on changed-state cases.
- Adds action-6 wall-click rule: when the click cell holds color 14 (the
  static wall), all color-14 cells flip to color 12 (level-clear lock).
  Verified 1/1 on the observed effective wall-click case.

All v1 invariants preserved:
- reward_class = 1 iff state changed (200/200).
- Actions 1..4 translate the largest non-bg/non-wall/non-timer blob by
  3 cells along cardinal directions (46/46 non-trivial shifts).
- NOOP when shifted bbox leaves the grid, or when the blob already
  contains color 12 (lock).
- Timer ticks (row 0) are non-deterministic from a single frame; not
  modelled.
"""
from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np

GridLike = Union[List[List[int]], np.ndarray]

BG = 10
WALL = 14
TIMER = 4
LOCK = 12

ACTION_SHIFTS = {
    1: (-3, 0),
    2: (3, 0),
    3: (0, -3),
    4: (0, 3),
}


def _to_np(state: GridLike) -> np.ndarray:
    if isinstance(state, np.ndarray):
        return state.astype(np.uint8, copy=True)
    return np.asarray(state, dtype=np.uint8)


def _find_large_blob(s: np.ndarray) -> List[Tuple[int, int]]:
    """Largest 4-connected component of non-bg, non-wall, non-timer pixels in rows 1..63."""
    H, W = s.shape
    visited = np.zeros((H, W), dtype=bool)
    visited[0] = True
    visited |= (s == BG) | (s == WALL) | (s == TIMER)

    best: List[Tuple[int, int]] = []
    for i in range(1, H):
        for j in range(W):
            if visited[i, j]:
                continue
            stack = [(i, j)]
            pix: List[Tuple[int, int]] = []
            visited[i, j] = True
            while stack:
                y, x = stack.pop()
                pix.append((y, x))
                if y > 0 and not visited[y - 1, x]:
                    visited[y - 1, x] = True
                    stack.append((y - 1, x))
                if y < H - 1 and not visited[y + 1, x]:
                    visited[y + 1, x] = True
                    stack.append((y + 1, x))
                if x > 0 and not visited[y, x - 1]:
                    visited[y, x - 1] = True
                    stack.append((y, x - 1))
                if x < W - 1 and not visited[y, x + 1]:
                    visited[y, x + 1] = True
                    stack.append((y, x + 1))
            if len(pix) > len(best):
                best = pix
    return best


def _blob_has_lock(grid: np.ndarray, blob: List[Tuple[int, int]]) -> bool:
    for y, x in blob:
        if grid[y, x] == LOCK:
            return True
    return False


def _apply_shift(grid: np.ndarray, blob: List[Tuple[int, int]], dr: int, dc: int) -> bool:
    if not blob:
        return False
    if _blob_has_lock(grid, blob):
        return False

    rmin = min(y for y, _ in blob)
    rmax = max(y for y, _ in blob)
    cmin = min(x for _, x in blob)
    cmax = max(x for _, x in blob)

    if rmin + dr < 1 or rmax + dr > 63 or cmin + dc < 0 or cmax + dc > 63:
        return False

    colours = [grid[y, x] for y, x in blob]
    for y, x in blob:
        grid[y, x] = BG
    for (y, x), c in zip(blob, colours):
        grid[y + dr, x + dc] = c
    return True


def _apply_rotate_cw(grid: np.ndarray, blob: List[Tuple[int, int]]) -> bool:
    """Rotate the blob 90 degrees clockwise about its bbox top-left corner.

    (y, x) -> (rmin + (x - cmin), cmin + (H - 1 - (y - rmin)))
    where H = rmax - rmin + 1. New bbox shape is W x H.

    Returns True if applied, False if NOOP (empty blob, lock present,
    or rotation exits the grid).
    """
    if not blob:
        return False
    if _blob_has_lock(grid, blob):
        return False

    rmin = min(y for y, _ in blob)
    rmax = max(y for y, _ in blob)
    cmin = min(x for _, x in blob)
    cmax = max(x for _, x in blob)
    H = rmax - rmin + 1
    W = cmax - cmin + 1

    # Check rotated bbox fits.
    new_rmax = rmin + (W - 1)
    new_cmax = cmin + (H - 1)
    if rmin < 1 or new_rmax > 63 or cmin < 0 or new_cmax > 63:
        return False

    colours = [grid[y, x] for y, x in blob]
    for y, x in blob:
        grid[y, x] = BG
    for (y, x), c in zip(blob, colours):
        ny = rmin + (x - cmin)
        nx = cmin + (H - 1 - (y - rmin))
        grid[ny, nx] = c
    return True


def _apply_wall_click(grid: np.ndarray, cy: int, cx: int) -> bool:
    """If the click cell currently holds color 14 (wall), flip every
    color-14 cell to color 12 (lock). Returns True if applied.
    """
    if cy < 0 or cy >= 64 or cx < 0 or cx >= 64:
        return False
    if grid[cy, cx] != WALL:
        return False
    mask = grid == WALL
    if not mask.any():
        return False
    grid[mask] = LOCK
    return True


def simulate(state: GridLike, action_id: int, x: int, y: int):
    """Predict (next_state, reward_class, done) for game cn04."""
    grid = _to_np(state)
    original = grid.copy()

    changed = False
    if action_id in ACTION_SHIFTS:
        dr, dc = ACTION_SHIFTS[action_id]
        blob = _find_large_blob(grid)
        changed = _apply_shift(grid, blob, dr, dc)
    elif action_id == 5:
        blob = _find_large_blob(grid)
        changed = _apply_rotate_cw(grid, blob)
    elif action_id == 6:
        # Click semantics: (x, y) from the observation correspond to
        # (col=x, row=y) in grid coordinates. Probe wall-click.
        changed = _apply_wall_click(grid, y, x)

    if not changed:
        if np.array_equal(grid, original):
            reward_class = 0
        else:
            reward_class = 1
    else:
        reward_class = 1

    return grid, reward_class, False
