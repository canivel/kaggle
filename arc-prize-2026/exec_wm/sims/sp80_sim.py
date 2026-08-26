"""Executable world model for ARC-AGI-3 game `sp80` -- v1.

Rodionov-style (arXiv:2605.05138): a hand-derived, self-contained Python
function that maps (state, action_id, x, y) -> (next_state, reward_class, done).

Empirical observations (full diagnosis in sp80_notes_v1.md):
-----------------------------------------------------------
Grid (64x64, values 0..14). Layout:
  row 0       : "fuel/ammo" bar of 64 cells, originally all value 14.
                Each action drains the K rightmost remaining 14s by
                setting them to 0. K is normally 2 but is 3 whenever the
                pre-action count of 14s is in {58, 41, 26, 9} (verified
                on 200/200 tuples).
  rows 1-7    : static colored bands (12 background + decorative 4/6 strip).
  rows 8-63   : playfield -- background 12.
  one solid   : a 4x20 "paddle" of color 9, occupying exactly one
  9-block     : (r0, c0) slot with r0 in {12,16,20,24} and c0 in {0,4,...,36}.

Action effects on the 9-block (verified across 200 tuples):
  action 1 : move paddle UP    by 4 rows  (NOOP if r0 == 12)
  action 2 : move paddle DOWN  by 4 rows  (NOOP if r0 == 24, conservative)
  action 3 : move paddle LEFT  by 4 cols  (NOOP if c0 == 0)
  action 4 : move paddle RIGHT by 4 cols  (NOOP if c0 == 36)
  action 5 : no paddle move (drains fuel only)
  action 6 : no paddle move (drains fuel only, ignores x/y)

Invariants used:
  - Color 9 occurs ONLY inside the paddle (always a solid 4x20 rectangle).
  - Fuel-bar K-rule is fully deterministic from n14_before.
  - reward_class == 1 and done == False on every observed tuple (200/200).
"""
from __future__ import annotations

from typing import List, Union

import numpy as np

GridLike = Union[List[List[int]], np.ndarray]

# --- Constants ----------------------------------------------------------
PADDLE_COLOR = 9
PLAYFIELD_BG = 12
FUEL_FULL = 14
FUEL_EMPTY = 0

PADDLE_H = 4
PADDLE_W = 20

# Movement step is 4 cells.
STEP = 4

# Bounding box limits derived from observations.
PADDLE_R0_MIN = 12   # action 1 NOOPs here (top wall)
PADDLE_R0_MAX = 24   # action 2 NOOPs here (bottom wall, conservative)
PADDLE_C0_MIN = 0    # action 3 NOOPs here (left wall)
PADDLE_C0_MAX = 36   # action 4 NOOPs here (right wall, c1 = 55)

# Fuel-drain K=3 occurs exactly at these pre-action n14 counts.
K3_TRIGGERS = frozenset({58, 41, 26, 9})


def _to_np(state: GridLike) -> np.ndarray:
    if isinstance(state, np.ndarray):
        return state.astype(np.uint8, copy=True)
    return np.asarray(state, dtype=np.uint8)


def _find_paddle(grid: np.ndarray):
    """Return (r0, c0) of the 4x20 paddle, or None if undetectable."""
    where = np.where(grid == PADDLE_COLOR)
    if where[0].size == 0:
        return None
    r0 = int(where[0].min())
    c0 = int(where[1].min())
    r1 = int(where[0].max())
    c1 = int(where[1].max())
    # Validate solid 4x20.
    if (r1 - r0 + 1) != PADDLE_H or (c1 - c0 + 1) != PADDLE_W:
        return None
    patch = grid[r0:r1 + 1, c0:c1 + 1]
    if not (patch == PADDLE_COLOR).all():
        return None
    return r0, c0


def _move_paddle(grid: np.ndarray, r0: int, c0: int, dr: int, dc: int) -> None:
    """Erase paddle at (r0,c0), stamp it at (r0+dr, c0+dc)."""
    grid[r0:r0 + PADDLE_H, c0:c0 + PADDLE_W] = PLAYFIELD_BG
    nr, nc = r0 + dr, c0 + dc
    grid[nr:nr + PADDLE_H, nc:nc + PADDLE_W] = PADDLE_COLOR


def _drain_fuel(grid: np.ndarray) -> None:
    """Drain the K rightmost 14s from row 0, setting them to 0.

    K is 3 if the current count of 14s is in K3_TRIGGERS, else 2. If fewer
    than K 14s remain, drain whatever is available (still rightmost).
    """
    row0 = grid[0]
    fourteen_idx = np.where(row0 == FUEL_FULL)[0]
    n14 = int(fourteen_idx.size)
    if n14 == 0:
        return
    k = 3 if n14 in K3_TRIGGERS else 2
    take = fourteen_idx[-k:] if n14 >= k else fourteen_idx
    grid[0, take] = FUEL_EMPTY


def simulate(state: GridLike, action_id: int, x: int, y: int):
    """Predict next state, reward_class, done for game sp80.

    Parameters
    ----------
    state : 64x64 grid (list[list[int]] or np.ndarray), values 0..14.
    action_id : int in {1, 2, 3, 4, 5, 6}.
    x, y : ints in [0, 64). Unused (action 6 click ignores x/y in observations).
    """
    grid = _to_np(state)

    # Move the paddle first (so the new position is in place before any
    # other rules touch the grid).
    if action_id in (1, 2, 3, 4):
        det = _find_paddle(grid)
        if det is not None:
            r0, c0 = det
            if action_id == 1:
                if r0 > PADDLE_R0_MIN:
                    _move_paddle(grid, r0, c0, -STEP, 0)
            elif action_id == 2:
                if r0 < PADDLE_R0_MAX:
                    _move_paddle(grid, r0, c0, STEP, 0)
            elif action_id == 3:
                if c0 > PADDLE_C0_MIN:
                    _move_paddle(grid, r0, c0, 0, -STEP)
            elif action_id == 4:
                if c0 < PADDLE_C0_MAX:
                    _move_paddle(grid, r0, c0, 0, STEP)
        # If detection fails (shouldn't, per observations), fall through
        # leaving the playfield unchanged -- preserves pixel-match floor.

    # Always drain fuel (every action drains, even NOOP-paddle moves).
    _drain_fuel(grid)

    # Reward and done are constant across all 200 observed tuples.
    reward_class = 1
    done = False
    return grid, reward_class, done
