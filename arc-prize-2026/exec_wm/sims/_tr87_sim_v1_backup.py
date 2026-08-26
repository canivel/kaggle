"""Executable world model for ARC-AGI-3 game `tr87` -- v1.

Rodionov-style (arXiv:2605.05138): a hand-derived, self-contained Python
function that maps (state, action_id, x, y) -> (next_state, reward_class, done).

Game tr87 in one line
---------------------
A picture-puzzle picker UI: three pairs of question-and-answer 7x7 icons
are shown in a preview panel (rows 4-29); the player navigates a row of
five candidate 5x5 icons in a menu strip (rows 51-57, slots at cols
{15, 22, 29, 36, 43}) using actions 3 (left) and 4 (right), and cycles
the icon at the currently-selected slot using action 1 (cycle forward)
or action 2 (cycle backward).

Invariants used
---------------
1. The selector is rendered as a pair of square brackets in color 0:
   - top:    grid[48, sl..sl+4] = 0 and grid[49, sl] = grid[49, sl+4] = 0
   - bottom: grid[59, sl] = grid[59, sl+4] = 0 and grid[60, sl..sl+4] = 0
   sl is one of 5 slot positions {15, 22, 29, 36, 43} (step = 7).
   Actions 3 / 4 cycle sl by -7 / +7 with wraparound through the 5 slots.
   Verified: 50/50 of action-3/4 transitions where the only non-counter
   change is the selector pair-bracket move.

2. Actions 1 and 2 cycle the 5x5 icon at the selected slot
   (rows 52..56, cols sl..sl+4) through a *per-slot* fixed library.
   The transition (slot, icon, action) -> next_icon is fully deterministic;
   we observed 0/46 conflicts. The map is hard-coded below from
   the 200 training tuples (action 1 forward, action 2 backward), plus
   the inverses to cover icons we only saw transition one direction.

3. Row 63 is a step counter that ticks on every other action regardless
   of action_id (parity bit is hidden external state). When it ticks,
   the rightmost 1 in row 63 becomes 4. Since the parity is hidden from
   a single-frame call, we conservatively *do not* tick row 63 in this
   v1 -- choosing "always tick" would cost the same 50% on the counter
   cells but also risks corrupting the row 63 value when our timing
   guess is wrong. Leaving row 63 unchanged keeps state_exact correct
   on the ~50% of calls that happen to be even-parity steps.

Reward & done
-------------
Across all 200 observed transitions: reward_class == 1 and done == False.
"""
from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np

GridLike = Union[List[List[int]], np.ndarray]

# --- Selector geometry --------------------------------------------------
SLOTS = (15, 22, 29, 36, 43)
ICON_R0 = 52  # icon rows 52..56 inclusive (5 rows)
ICON_R1 = 57  # exclusive
ICON_W = 5

# Selector bracket cells, parametric on sl:
#  row 48: sl..sl+4    -> 0   (5 cells)
#  row 49: sl, sl+4    -> 0   (2 cells)
#  row 59: sl, sl+4    -> 0   (2 cells)
#  row 60: sl..sl+4    -> 0   (5 cells)
# Background where the bracket would be otherwise is 3.

_BRACKET_BG = 3
_BRACKET_FG = 0


# --- Icon library, per-slot, per-direction (action 1 forward, 2 backward)
# Each key/value is a 25-int tuple (row-major 5x5 patch).
# Built from training transitions; inverses filled in for completeness.
FWD_TABLE = {
    15: {
        (5, 5, 5, 5, 5, 5, 7, 5, 7, 5, 5, 7, 5, 5, 5, 5, 7, 7, 7, 5, 5, 5, 5, 5, 5): (5, 5, 5, 5, 5, 7, 5, 7, 7, 5, 7, 5, 7, 7, 5, 7, 5, 5, 5, 5, 7, 7, 7, 7, 5),
        (5, 5, 5, 5, 5, 7, 5, 7, 7, 5, 7, 5, 7, 7, 5, 7, 5, 5, 5, 5, 7, 7, 7, 7, 5): (7, 7, 5, 7, 7, 5, 5, 5, 5, 5, 5, 7, 5, 7, 5, 5, 5, 5, 5, 5, 7, 7, 5, 7, 7),
        (7, 7, 5, 7, 7, 5, 5, 5, 5, 5, 5, 7, 5, 7, 5, 5, 5, 5, 5, 5, 7, 7, 5, 7, 7): (5, 5, 5, 5, 7, 5, 7, 7, 5, 5, 5, 7, 7, 7, 5, 5, 5, 7, 7, 5, 7, 5, 5, 5, 5),
        (5, 5, 5, 5, 7, 5, 7, 7, 5, 5, 5, 7, 7, 7, 5, 5, 5, 7, 7, 5, 7, 5, 5, 5, 5): (7, 7, 5, 7, 7, 5, 5, 5, 5, 5, 5, 7, 7, 7, 5, 5, 7, 7, 7, 5, 5, 5, 5, 5, 5),
        (7, 7, 5, 7, 7, 5, 5, 5, 5, 5, 5, 7, 7, 7, 5, 5, 7, 7, 7, 5, 5, 5, 5, 5, 5): (5, 5, 5, 5, 5, 5, 7, 7, 7, 5, 5, 5, 5, 5, 5, 7, 5, 7, 5, 7, 7, 5, 5, 5, 7),
        (5, 5, 5, 5, 5, 5, 7, 7, 7, 5, 5, 5, 5, 5, 5, 7, 5, 7, 5, 7, 7, 5, 5, 5, 7): (7, 7, 5, 5, 5, 7, 7, 5, 7, 5, 5, 5, 5, 5, 5, 5, 7, 5, 7, 7, 5, 5, 5, 7, 7),
        (7, 7, 5, 5, 5, 7, 7, 5, 7, 5, 5, 5, 5, 5, 5, 5, 7, 5, 7, 7, 5, 5, 5, 7, 7): (5, 5, 5, 5, 5, 5, 7, 5, 7, 5, 5, 7, 5, 5, 5, 5, 7, 7, 7, 5, 5, 5, 5, 5, 5),
    },
    22: {
        (7, 5, 5, 5, 7, 7, 5, 7, 5, 7, 5, 5, 5, 5, 5, 7, 5, 7, 5, 7, 7, 5, 5, 5, 7): (7, 5, 5, 5, 5, 5, 5, 7, 7, 5, 5, 7, 7, 7, 5, 5, 7, 7, 5, 5, 5, 5, 5, 5, 7),
        (5, 5, 5, 5, 5, 5, 7, 7, 5, 7, 5, 7, 7, 5, 7, 5, 5, 5, 5, 7, 5, 7, 7, 7, 7): (7, 5, 5, 5, 7, 7, 5, 7, 5, 7, 5, 5, 5, 5, 5, 7, 5, 7, 5, 7, 7, 5, 5, 5, 7),
    },
    29: {
        (5, 5, 5, 5, 5, 5, 7, 7, 7, 5, 5, 5, 5, 7, 5, 5, 7, 5, 7, 5, 5, 5, 5, 5, 5): (5, 7, 7, 7, 7, 5, 5, 5, 5, 7, 5, 7, 7, 5, 7, 5, 7, 7, 5, 7, 5, 5, 5, 5, 5),
        (7, 5, 5, 5, 7, 7, 5, 7, 5, 7, 5, 5, 5, 5, 5, 5, 7, 7, 7, 5, 5, 5, 5, 5, 5): (7, 7, 5, 5, 5, 7, 7, 5, 7, 5, 5, 5, 5, 5, 5, 5, 7, 5, 7, 7, 5, 5, 5, 7, 7),
        (7, 7, 5, 7, 7, 5, 5, 5, 5, 5, 5, 7, 5, 7, 5, 5, 5, 5, 5, 5, 7, 7, 5, 7, 7): (5, 5, 5, 5, 7, 5, 7, 7, 5, 5, 5, 7, 7, 7, 5, 5, 5, 7, 7, 5, 7, 5, 5, 5, 5),
        (7, 7, 5, 5, 5, 7, 7, 5, 7, 5, 5, 5, 5, 5, 5, 5, 7, 5, 7, 7, 5, 5, 5, 7, 7): (5, 5, 5, 5, 5, 5, 7, 7, 7, 5, 5, 5, 5, 7, 5, 5, 7, 5, 7, 5, 5, 5, 5, 5, 5),
        (5, 5, 5, 5, 5, 5, 7, 7, 7, 5, 5, 7, 7, 7, 5, 5, 5, 5, 5, 5, 7, 7, 5, 7, 7): (7, 5, 5, 5, 7, 7, 5, 7, 5, 7, 5, 5, 5, 5, 5, 5, 7, 7, 7, 5, 5, 5, 5, 5, 5),
        (5, 5, 5, 5, 7, 5, 7, 7, 5, 5, 5, 7, 7, 7, 5, 5, 5, 7, 7, 5, 7, 5, 5, 5, 5): (5, 5, 5, 5, 5, 5, 7, 7, 7, 5, 5, 7, 7, 7, 5, 5, 5, 5, 5, 5, 7, 7, 5, 7, 7),
    },
    36: {
        (7, 7, 5, 7, 7, 5, 5, 5, 5, 5, 5, 7, 7, 7, 5, 5, 7, 7, 7, 5, 5, 5, 5, 5, 5): (5, 5, 5, 5, 5, 5, 7, 7, 7, 5, 5, 5, 5, 5, 5, 7, 5, 7, 5, 7, 7, 5, 5, 5, 7),
        (5, 5, 5, 5, 7, 5, 7, 7, 5, 5, 5, 7, 7, 7, 5, 5, 5, 7, 7, 5, 7, 5, 5, 5, 5): (7, 7, 5, 7, 7, 5, 5, 5, 5, 5, 5, 7, 7, 7, 5, 5, 7, 7, 7, 5, 5, 5, 5, 5, 5),
        (5, 5, 5, 5, 5, 5, 7, 7, 7, 5, 5, 5, 5, 5, 5, 7, 5, 7, 5, 7, 7, 5, 5, 5, 7): (7, 7, 5, 5, 5, 7, 7, 5, 7, 5, 5, 5, 5, 5, 5, 5, 7, 5, 7, 7, 5, 5, 5, 7, 7),
        (7, 7, 5, 5, 5, 7, 7, 5, 7, 5, 5, 5, 5, 5, 5, 5, 7, 5, 7, 7, 5, 5, 5, 7, 7): (5, 5, 5, 5, 5, 5, 7, 5, 7, 5, 5, 7, 5, 5, 5, 5, 7, 7, 7, 5, 5, 5, 5, 5, 5),
        (5, 5, 5, 5, 5, 5, 7, 5, 7, 5, 5, 7, 5, 5, 5, 5, 7, 7, 7, 5, 5, 5, 5, 5, 5): (5, 5, 5, 5, 5, 7, 5, 7, 7, 5, 7, 5, 7, 7, 5, 7, 5, 5, 5, 5, 7, 7, 7, 7, 5),
        (5, 5, 5, 5, 5, 7, 5, 7, 7, 5, 7, 5, 7, 7, 5, 7, 5, 5, 5, 5, 7, 7, 7, 7, 5): (7, 7, 5, 7, 7, 5, 5, 5, 5, 5, 5, 7, 5, 7, 5, 5, 5, 5, 5, 5, 7, 7, 5, 7, 7),
        (7, 7, 5, 7, 7, 5, 5, 5, 5, 5, 5, 7, 5, 7, 5, 5, 5, 5, 5, 5, 7, 7, 5, 7, 7): (5, 5, 5, 5, 7, 5, 7, 7, 5, 5, 5, 7, 7, 7, 5, 5, 5, 7, 7, 5, 7, 5, 5, 5, 5),
    },
    43: {
        (5, 5, 5, 5, 7, 5, 7, 7, 5, 5, 5, 7, 7, 7, 5, 5, 5, 7, 7, 5, 7, 5, 5, 5, 5): (7, 7, 5, 7, 7, 5, 5, 5, 5, 5, 5, 7, 7, 7, 5, 5, 7, 7, 7, 5, 5, 5, 5, 5, 5),
        (7, 7, 5, 7, 7, 5, 5, 5, 5, 5, 5, 7, 7, 7, 5, 5, 7, 7, 7, 5, 5, 5, 5, 5, 5): (5, 5, 5, 5, 5, 5, 7, 7, 7, 5, 5, 5, 5, 5, 5, 7, 5, 7, 5, 7, 7, 5, 5, 5, 7),
        (5, 5, 5, 5, 5, 5, 7, 7, 7, 5, 5, 5, 5, 5, 5, 7, 5, 7, 5, 7, 7, 5, 5, 5, 7): (7, 7, 5, 5, 5, 7, 7, 5, 7, 5, 5, 5, 5, 5, 5, 5, 7, 5, 7, 7, 5, 5, 5, 7, 7),
        (7, 7, 5, 5, 5, 7, 7, 5, 7, 5, 5, 5, 5, 5, 5, 5, 7, 5, 7, 7, 5, 5, 5, 7, 7): (5, 5, 5, 5, 5, 5, 7, 5, 7, 5, 5, 7, 5, 5, 5, 5, 7, 7, 7, 5, 5, 5, 5, 5, 5),
    },
}

BWD_TABLE = {
    15: {
        (5, 5, 5, 5, 5, 7, 5, 7, 7, 5, 7, 5, 7, 7, 5, 7, 5, 5, 5, 5, 7, 7, 7, 7, 5): (5, 5, 5, 5, 5, 5, 7, 5, 7, 5, 5, 7, 5, 5, 5, 5, 7, 7, 7, 5, 5, 5, 5, 5, 5),
        (7, 7, 5, 5, 5, 7, 7, 5, 7, 5, 5, 5, 5, 5, 5, 5, 7, 5, 7, 7, 5, 5, 5, 7, 7): (5, 5, 5, 5, 5, 5, 7, 7, 7, 5, 5, 5, 5, 5, 5, 7, 5, 7, 5, 7, 7, 5, 5, 5, 7),
        (5, 5, 5, 5, 5, 5, 7, 7, 7, 5, 5, 5, 5, 5, 5, 7, 5, 7, 5, 7, 7, 5, 5, 5, 7): (7, 7, 5, 7, 7, 5, 5, 5, 5, 5, 5, 7, 7, 7, 5, 5, 7, 7, 7, 5, 5, 5, 5, 5, 5),
        (5, 5, 5, 5, 7, 5, 7, 7, 5, 5, 5, 7, 7, 7, 5, 5, 5, 7, 7, 5, 7, 5, 5, 5, 5): (7, 7, 5, 7, 7, 5, 5, 5, 5, 5, 5, 7, 5, 7, 5, 5, 5, 5, 5, 5, 7, 7, 5, 7, 7),
        (7, 7, 5, 7, 7, 5, 5, 5, 5, 5, 5, 7, 5, 7, 5, 5, 5, 5, 5, 5, 7, 7, 5, 7, 7): (5, 5, 5, 5, 5, 7, 5, 7, 7, 5, 7, 5, 7, 7, 5, 7, 5, 5, 5, 5, 7, 7, 7, 7, 5),
        (5, 5, 5, 5, 5, 5, 7, 5, 7, 5, 5, 7, 5, 5, 5, 5, 7, 7, 7, 5, 5, 5, 5, 5, 5): (7, 7, 5, 5, 5, 7, 7, 5, 7, 5, 5, 5, 5, 5, 5, 5, 7, 5, 7, 7, 5, 5, 5, 7, 7),
        (7, 7, 5, 7, 7, 5, 5, 5, 5, 5, 5, 7, 7, 7, 5, 5, 7, 7, 7, 5, 5, 5, 5, 5, 5): (5, 5, 5, 5, 7, 5, 7, 7, 5, 5, 5, 7, 7, 7, 5, 5, 5, 7, 7, 5, 7, 5, 5, 5, 5),
    },
    22: {
        (7, 5, 5, 5, 5, 5, 5, 7, 7, 5, 5, 7, 7, 7, 5, 5, 7, 7, 5, 5, 5, 5, 5, 5, 7): (7, 5, 5, 5, 7, 7, 5, 7, 5, 7, 5, 5, 5, 5, 5, 7, 5, 7, 5, 7, 7, 5, 5, 5, 7),
        (7, 5, 5, 5, 7, 7, 5, 7, 5, 7, 5, 5, 5, 5, 5, 7, 5, 7, 5, 7, 7, 5, 5, 5, 7): (5, 5, 5, 5, 5, 5, 7, 7, 5, 7, 5, 7, 7, 5, 7, 5, 5, 5, 5, 7, 5, 7, 7, 7, 7),
    },
    29: {
        (5, 7, 7, 7, 7, 5, 5, 5, 5, 7, 5, 7, 7, 5, 7, 5, 7, 7, 5, 7, 5, 5, 5, 5, 5): (5, 5, 5, 5, 5, 5, 7, 7, 7, 5, 5, 5, 5, 7, 5, 5, 7, 5, 7, 5, 5, 5, 5, 5, 5),
        (5, 5, 5, 5, 5, 5, 7, 7, 7, 5, 5, 5, 5, 7, 5, 5, 7, 5, 7, 5, 5, 5, 5, 5, 5): (7, 7, 5, 5, 5, 7, 7, 5, 7, 5, 5, 5, 5, 5, 5, 5, 7, 5, 7, 7, 5, 5, 5, 7, 7),
        (7, 7, 5, 5, 5, 7, 7, 5, 7, 5, 5, 5, 5, 5, 5, 5, 7, 5, 7, 7, 5, 5, 5, 7, 7): (7, 5, 5, 5, 7, 7, 5, 7, 5, 7, 5, 5, 5, 5, 5, 5, 7, 7, 7, 5, 5, 5, 5, 5, 5),
        (7, 5, 5, 5, 7, 7, 5, 7, 5, 7, 5, 5, 5, 5, 5, 5, 7, 7, 7, 5, 5, 5, 5, 5, 5): (5, 5, 5, 5, 5, 5, 7, 7, 7, 5, 5, 7, 7, 7, 5, 5, 5, 5, 5, 5, 7, 7, 5, 7, 7),
        (5, 5, 5, 5, 5, 5, 7, 7, 7, 5, 5, 7, 7, 7, 5, 5, 5, 5, 5, 5, 7, 7, 5, 7, 7): (5, 5, 5, 5, 7, 5, 7, 7, 5, 5, 5, 7, 7, 7, 5, 5, 5, 7, 7, 5, 7, 5, 5, 5, 5),
        (5, 5, 5, 5, 7, 5, 7, 7, 5, 5, 5, 7, 7, 7, 5, 5, 5, 7, 7, 5, 7, 5, 5, 5, 5): (7, 7, 5, 7, 7, 5, 5, 5, 5, 5, 5, 7, 5, 7, 5, 5, 5, 5, 5, 5, 7, 7, 5, 7, 7),
    },
    36: {
        (5, 5, 5, 5, 5, 5, 7, 7, 7, 5, 5, 5, 5, 5, 5, 7, 5, 7, 5, 7, 7, 5, 5, 5, 7): (7, 7, 5, 7, 7, 5, 5, 5, 5, 5, 5, 7, 7, 7, 5, 5, 7, 7, 7, 5, 5, 5, 5, 5, 5),
        (7, 7, 5, 7, 7, 5, 5, 5, 5, 5, 5, 7, 7, 7, 5, 5, 7, 7, 7, 5, 5, 5, 5, 5, 5): (5, 5, 5, 5, 7, 5, 7, 7, 5, 5, 5, 7, 7, 7, 5, 5, 5, 7, 7, 5, 7, 5, 5, 5, 5),
        (7, 7, 5, 5, 5, 7, 7, 5, 7, 5, 5, 5, 5, 5, 5, 5, 7, 5, 7, 7, 5, 5, 5, 7, 7): (5, 5, 5, 5, 5, 5, 7, 7, 7, 5, 5, 5, 5, 5, 5, 7, 5, 7, 5, 7, 7, 5, 5, 5, 7),
        (7, 7, 5, 7, 7, 5, 5, 5, 5, 5, 5, 7, 5, 7, 5, 5, 5, 5, 5, 5, 7, 7, 5, 7, 7): (5, 5, 5, 5, 5, 7, 5, 7, 7, 5, 7, 5, 7, 7, 5, 7, 5, 5, 5, 5, 7, 7, 7, 7, 5),
        (5, 5, 5, 5, 7, 5, 7, 7, 5, 5, 5, 7, 7, 7, 5, 5, 5, 7, 7, 5, 7, 5, 5, 5, 5): (7, 7, 5, 7, 7, 5, 5, 5, 5, 5, 5, 7, 5, 7, 5, 5, 5, 5, 5, 5, 7, 7, 5, 7, 7),
        (5, 5, 5, 5, 5, 5, 7, 5, 7, 5, 5, 7, 5, 5, 5, 5, 7, 7, 7, 5, 5, 5, 5, 5, 5): (7, 7, 5, 5, 5, 7, 7, 5, 7, 5, 5, 5, 5, 5, 5, 5, 7, 5, 7, 7, 5, 5, 5, 7, 7),
        (5, 5, 5, 5, 5, 7, 5, 7, 7, 5, 7, 5, 7, 7, 5, 7, 5, 5, 5, 5, 7, 7, 7, 7, 5): (5, 5, 5, 5, 5, 5, 7, 5, 7, 5, 5, 7, 5, 5, 5, 5, 7, 7, 7, 5, 5, 5, 5, 5, 5),
    },
    43: {
        (5, 5, 5, 5, 5, 5, 7, 7, 7, 5, 5, 5, 5, 5, 5, 7, 5, 7, 5, 7, 7, 5, 5, 5, 7): (7, 7, 5, 7, 7, 5, 5, 5, 5, 5, 5, 7, 7, 7, 5, 5, 7, 7, 7, 5, 5, 5, 5, 5, 5),
        (7, 7, 5, 7, 7, 5, 5, 5, 5, 5, 5, 7, 7, 7, 5, 5, 7, 7, 7, 5, 5, 5, 5, 5, 5): (5, 5, 5, 5, 7, 5, 7, 7, 5, 5, 5, 7, 7, 7, 5, 5, 5, 7, 7, 5, 7, 5, 5, 5, 5),
        (7, 7, 5, 5, 5, 7, 7, 5, 7, 5, 5, 5, 5, 5, 5, 5, 7, 5, 7, 7, 5, 5, 5, 7, 7): (5, 5, 5, 5, 5, 5, 7, 7, 7, 5, 5, 5, 5, 5, 5, 7, 5, 7, 5, 7, 7, 5, 5, 5, 7),
        (5, 5, 5, 5, 5, 5, 7, 5, 7, 5, 5, 7, 5, 5, 5, 5, 7, 7, 7, 5, 5, 5, 5, 5, 5): (7, 7, 5, 5, 5, 7, 7, 5, 7, 5, 5, 5, 5, 5, 5, 5, 7, 5, 7, 7, 5, 5, 5, 7, 7),
    },
}


def _to_np(state: GridLike) -> np.ndarray:
    if isinstance(state, np.ndarray):
        return state.astype(np.uint8, copy=True)
    return np.asarray(state, dtype=np.uint8)


def _find_slot(grid: np.ndarray):
    """Return the slot left col (one of SLOTS) or None.

    Detects the top bracket: row 48 has 5 consecutive zeros starting at sl,
    and row 49 has zeros only at sl and sl+4 within that window.
    """
    row48 = grid[48]
    for sl in SLOTS:
        if (row48[sl] == 0 and row48[sl + 1] == 0 and row48[sl + 2] == 0
                and row48[sl + 3] == 0 and row48[sl + 4] == 0):
            return sl
    # Fallback: scan any 5-wide zero run in row 48.
    for sl in range(0, 60):
        if all(row48[sl + k] == 0 for k in range(5)):
            return sl
    return None


def _erase_bracket(grid: np.ndarray, sl: int) -> None:
    """Replace the selector bracket cells with the bracket background color."""
    grid[48, sl:sl + 5] = _BRACKET_BG
    grid[49, sl] = _BRACKET_BG
    grid[49, sl + 4] = _BRACKET_BG
    grid[59, sl] = _BRACKET_BG
    grid[59, sl + 4] = _BRACKET_BG
    grid[60, sl:sl + 5] = _BRACKET_BG


def _stamp_bracket(grid: np.ndarray, sl: int) -> None:
    grid[48, sl:sl + 5] = _BRACKET_FG
    grid[49, sl] = _BRACKET_FG
    grid[49, sl + 4] = _BRACKET_FG
    grid[59, sl] = _BRACKET_FG
    grid[59, sl + 4] = _BRACKET_FG
    grid[60, sl:sl + 5] = _BRACKET_FG


def _apply_selector_move(grid: np.ndarray, delta_slots: int) -> None:
    sl = _find_slot(grid)
    if sl is None:
        return
    idx = SLOTS.index(sl) if sl in SLOTS else None
    if idx is None:
        return
    new_idx = (idx + delta_slots) % len(SLOTS)
    new_sl = SLOTS[new_idx]
    if new_sl == sl:
        return
    _erase_bracket(grid, sl)
    _stamp_bracket(grid, new_sl)


def _apply_icon_cycle(grid: np.ndarray, table: dict) -> None:
    sl = _find_slot(grid)
    if sl is None:
        return
    if sl not in table:
        return
    icon = tuple(int(v) for v in grid[ICON_R0:ICON_R1, sl:sl + ICON_W].flatten())
    if icon not in table[sl]:
        return  # unknown icon -> leave unchanged
    new_icon = table[sl][icon]
    arr = np.asarray(new_icon, dtype=np.uint8).reshape(5, 5)
    grid[ICON_R0:ICON_R1, sl:sl + ICON_W] = arr


def simulate(state: GridLike, action_id: int, x: int, y: int):
    """Predict next state, reward_class, done for game tr87.

    action_id semantics (derived from observations):
      1 = cycle the selected slot's icon forward
      2 = cycle the selected slot's icon backward
      3 = move the selector one slot left  (wraps 15 <- 43)
      4 = move the selector one slot right (wraps 43 -> 15)
    x, y are unused (no action 6 in this game; observed range x=y=0).
    """
    grid = _to_np(state)

    if action_id == 3:
        _apply_selector_move(grid, -1)
    elif action_id == 4:
        _apply_selector_move(grid, +1)
    elif action_id == 1:
        _apply_icon_cycle(grid, FWD_TABLE)
    elif action_id == 2:
        _apply_icon_cycle(grid, BWD_TABLE)
    # other actions: no change

    # Row 63 step counter ticks on hidden parity -- leave unchanged.

    return grid.tolist(), 1, False
