"""Executable world model for ARC-AGI-3 game `lf52` — v1.

Rodionov-style (arXiv:2605.05138): a hand-derived, self-contained Python
function that maps (state, action_id, x, y) -> (next_state, reward_class, done).

Empirical observations (full diagnosis in lf52_notes_v1.md):
- 200 random-exploration tuples.
- reward_class is 1 for every tuple, done is False, level stays 0.
- For EVERY tuple (regardless of action_id and x/y), exactly ONE cell
  changes. That cell lives on row 0 and follows a single deterministic
  rule:
    let v = min value present on row 0
    let c = leftmost column with state[0, c] == v
    next_state[0, c] = v + 1
- All other rows are untouched on every observed transition.
- Verified on 200/200 tuples — universal across all 6 actions
  (1, 2, 3, 4, 6, 7) and all observed (x, y).

This is a row-0 step counter that increments through the entire row
before "carrying" into the next value tier (0->1 fills row 0 with 1s,
then 1->2 starts overwriting them, etc.).
"""
from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np

GridLike = Union[List[List[int]], np.ndarray]


def _to_np(state: GridLike) -> np.ndarray:
    if isinstance(state, np.ndarray):
        return state.astype(np.uint8, copy=True)
    return np.asarray(state, dtype=np.uint8)


def _advance_row0_counter(grid: np.ndarray) -> bool:
    """Set the leftmost min-valued cell on row 0 to (min+1).

    Returns True iff a change was applied.
    """
    row = grid[0]
    v = int(row.min())
    # leftmost column whose value equals the min
    idx = int(np.argmax(row == v))  # argmax of bool returns first True
    if row[idx] != v:
        return False
    grid[0, idx] = v + 1
    return True


def simulate(state: GridLike, action_id: int, x: int, y: int):
    """Predict next state, reward_class, done for game lf52.

    Parameters
    ----------
    state : 64x64 grid (list[list[int]] or np.ndarray), values 0-15.
    action_id : int in {1, 2, 3, 4, 6, 7}.
    x, y : ints in [0, 64). Ignored — every observed transition is
        independent of action and click coords.

    Returns
    -------
    (next_state : np.ndarray (64,64) uint8, reward_class : int, done : bool)
    """
    grid = _to_np(state)
    changed = _advance_row0_counter(grid)
    # reward_class 1 == "state changed" in this game's labeling.
    # In 200/200 observed tuples reward was always 1.
    reward_class = 1 if changed else 0
    done = False
    return grid, reward_class, done
