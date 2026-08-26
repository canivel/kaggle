"""Executable world model for ARC-AGI-3 game `su15` -- v2.

v1 recap: 98.5 % state-exact / 99.997 % pixel.  Three action-6
mispredictions remained -- one rare sprite stamp (i=4) and two ordinary
counter ticks (i=53, i=54) where the new cells were `15` instead of
`5`.

v2 invariant (new, data-driven):
  When a counter tick writes the next-available pair (c, c+1) on
  row 63, the value written into each cell mirrors the cell directly
  above it on row 62:
        15  if state[62, col] == 15   (a persistent "warning" marker
                                       placed by a prior sprite-stamp
                                       event hangs over that column)
         5  otherwise                  (normal fill)

Background: the only `n_changed > 2` event in the 200-tuple sample
is the i=4 sprite stamp at (x=12, y=62).  That stamp planted three
`15` cells at (61, 11..13) and (62, 11..13).  Later, when the
counter's right-to-left fill reaches columns 11..13 (steps i=53, 54),
those cells are written as `15` rather than `5` -- the counter
"absorbs" the colour of the marker above it.

Crucial split-test: in the SECOND trajectory segment (post-reset
at i=70), no sprite-stamp ever fired, so row 62 stays at `5`s, and
the counter at cols 11..13 (i=122, 123) writes plain `5,5`.  The
v1-era theory "warning zone fixed at cols 11..13" was curve-fit; the
correct rule is "look at row 62 at the same column".

Verified ticks (all action-6, y >= 10):
  i=53  (12,13)<-(15,15)  row62[12,13]=(15,15)             OK
  i=54  (10,11)<-( 5,15)  row62[10,11]=( 5,15)             OK
  i=55  ( 8, 9)<-( 5, 5)  row62[ 8, 9]=( 5, 5)             OK
  i=122 (12,13)<-( 5, 5)  row62[12,13]=( 5, 5) (no sprite) OK
  i=123 (10,11)<-( 5, 5)  row62[10,11]=( 5, 5) (no sprite) OK
  ...and all 80+ earlier ticks where row 62 was untouched.

Counter reset / wrap behaviour: at i=70 and i=146 the row-63 row
"flips" from fills=62 (one pair short of full, since cols 0,1 had
just been written) to fills=0.  This is presumably a level-tick or
overflow.  Modelling the exact trigger requires knowing what
preceded the reset (action 7 was the last call at i=70 according to
the dump).  We leave the reset rule unmodelled: predicting reset
would only help the single tuple immediately after the wrap, and we
don't have a clean trigger condition.  (Both observed resets were
not the *next* prediction target -- they happen between adjacent
non-reset tuples -- so this does not cost us state-exact %.)

The sprite-stamp at i=4 itself is still left as identity-with-
counter-tick; one example is insufficient to model what the stamp
draws or what triggers it.  v2 keeps that single misprediction.
"""
from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np

GridLike = Union[List[List[int]], np.ndarray]

# --- Constants ----------------------------------------------------------
COUNTER_ROW = 63
MARKER_ROW = 62      # row immediately above the counter; carries warning colour
COUNTER_VAL = 5
WARNING_VAL = 15
PLAYFIELD_Y_MIN = 10  # action-6 clicks with y < 10 are out-of-bounds no-ops


def _to_np(state: GridLike) -> np.ndarray:
    if isinstance(state, np.ndarray):
        return state.astype(np.uint8, copy=True)
    return np.asarray(state, dtype=np.uint8)


def _next_counter_pair(grid: np.ndarray) -> int | None:
    """Return the column index `c` of the next empty (c, c+1) pair on
    row 63, scanning from the right end towards the left.  Returns None
    if the row has no empty pair left (counter is full).
    """
    row = grid[COUNTER_ROW]
    for c in range(62, -1, -2):
        if row[c] == 0 and row[c + 1] == 0:
            return c
    return None


def _tick_value(grid: np.ndarray, col: int) -> int:
    """A counter cell inherits 15 if the marker row directly above it is
    15, else the normal fill colour 5.
    """
    return WARNING_VAL if grid[MARKER_ROW, col] == WARNING_VAL else COUNTER_VAL


def simulate(state: GridLike, action_id: int, x: int, y: int):
    """Predict next state, reward_class, done for game su15.

    Parameters
    ----------
    state : 64x64 grid (list[list[int]] or np.ndarray), values 0-15.
    action_id : int.  Only {6, 7} have been observed.
    x, y : ints in [0, 64).  Only used when action_id == 6.

    Returns
    -------
    (next_state_list, reward_class, done)
    """
    grid = _to_np(state)

    done = False
    reward_class = 0

    if action_id == 6:
        # Out-of-bounds click -> no-op (y < 10).
        if y >= PLAYFIELD_Y_MIN:
            c = _next_counter_pair(grid)
            if c is not None:
                grid[COUNTER_ROW, c] = _tick_value(grid, c)
                grid[COUNTER_ROW, c + 1] = _tick_value(grid, c + 1)
                reward_class = 1
            # else: counter full -- leave grid alone, reward 0
        # else: y < 10 -- no-op, reward 0
    elif action_id == 7:
        # Pure no-op in all 98 observed tuples.
        pass
    # Any other action_id (none observed) -> identity, reward 0.

    return grid.tolist(), int(reward_class), bool(done)
