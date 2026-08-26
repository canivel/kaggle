"""Executable world model for ARC-AGI-3 game `ar25` — v2.

Rodionov-style (arXiv:2605.05138). Hand-derived, self-contained Python
function mapping (state, action_id, x, y) -> (next_state, reward_class, done).

GAME (one line): two 9x9 sprites (L=color 5+0, R=color 4) move in mirrored
directions across a split playfield; a "right meter" in column 63 ticks
down by 1 every successful action.

INVARIANTS (verified across 200 sample tuples):

1.  Background layer (static, deterministic):
        - playfield value = 9 by default
        - cols 30,31,32 = 10 (vertical divider)
        - col 63 = 11 (right meter, ticks 11->5 from the top)
        - row 63 = 5 (bottom bar)
        - rows 45-52 cols 51-59: a static L-shaped 11-decoration (right side)

2.  Two sprites: L (9x9 L-shape, colors 5 + 0-holes) and R (9x9 mirrored
    L-shape in color 4). Each placed by top-left (r0, c0).
    Rendering: cell value 5/4 = opaque; 9 = transparent; 0 = renders as 0
    over playfield-9 but as background where bg has a non-9 feature
    (divider). Implementation: keep the input state as-is, erase old sprite
    pixels back to the background layer, then re-stamp at new (r0, c0).

3.  Action effects (each move is 3 cells):
        action 1: BOTH sprites UP 3 rows
        action 2: BOTH sprites DOWN 3 rows
        action 3: L LEFT 3 cols  /  R RIGHT 3 cols (mirrored)
        action 4: L RIGHT 3 cols /  R LEFT 3 cols (mirrored)
        action 5: tick meter only (sprites unchanged)
        action 6: pure NO-OP (reward 0)
        action 7: non-deterministic in a 1-frame window — leave identity

4.  Boundaries (per sprite top-left):
        L: r0 in [0, 27], c0 unconstrained on left edge (allowed off-screen)
        R: r0 in [0, 27], c0 in [33, 51]
    If a sprite's vertical move would go OOB, BOTH sprites NOOP (action 1/2).
    For action 3/4, only the R sprite has a horizontal bound; if its move
    is blocked, R stays put while L still moves (matches observed cases).

5.  Reward / meter:
        reward = 1 iff any sprite moved; 0 otherwise.
        meter: if reward=1, flip the topmost 11 in column 63 (rows 0..62)
        to a 5. If no 11 remains, meter stays empty (refill happens
        out-of-band between episodes — not modelled).
        Action 6 reward=0, no meter tick.

6.  Done = False always (no terminal in 200 samples).
"""
from __future__ import annotations

from typing import List, Tuple, Union, Optional

import numpy as np

GridLike = Union[List[List[int]], np.ndarray]

# --- Sprite templates --------------------------------------------------
L_TEMPLATE = np.array([
    [5, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 0, 5, 5, 0, 5, 5, 0, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 5],
    [9, 9, 9, 9, 9, 9, 5, 5, 5],
    [9, 9, 9, 9, 9, 9, 5, 0, 5],
    [9, 9, 9, 9, 9, 9, 5, 5, 5],
    [9, 9, 9, 9, 9, 9, 5, 5, 5],
    [9, 9, 9, 9, 9, 9, 5, 0, 5],
    [9, 9, 9, 9, 9, 9, 5, 5, 5],
], dtype=np.int64)

R_TEMPLATE = np.array([
    [4, 4, 4, 4, 4, 4, 4, 4, 4],
    [4, 4, 4, 4, 4, 4, 4, 4, 4],
    [4, 4, 4, 4, 4, 4, 4, 4, 4],
    [4, 4, 4, 9, 9, 9, 9, 9, 9],
    [4, 4, 4, 9, 9, 9, 9, 9, 9],
    [4, 4, 4, 9, 9, 9, 9, 9, 9],
    [4, 4, 4, 9, 9, 9, 9, 9, 9],
    [4, 4, 4, 9, 9, 9, 9, 9, 9],
    [4, 4, 4, 9, 9, 9, 9, 9, 9],
], dtype=np.int64)

SPRITE_H = 9
SPRITE_W = 9
STEP = 3

# Sprite top-left bounds
SPRITE_R0_MIN = 0
SPRITE_R0_MAX = 27   # 27+8 = 35 -> still in rows
L_C0_MIN = 3         # observed L c0 in {3,6,...,30}
L_C0_MAX = 30
R_C0_MIN = 33
R_C0_MAX = 51

# Background features
BG_PLAYFIELD = 9
BG_DIVIDER_COLS = (30, 31, 32)
BG_METER_COL = 63
BG_METER_VAL = 11
BG_BOTTOM_ROW = 63
BG_BOTTOM_VAL = 5

# Static 9x9 11-decoration in right zone (rows 45-52 cols 51-59)
STATIC_DECO_R = np.array([
    [11, 11, 11, 11, 11, 11, 11, 11, 11],
    [11, 11, 11, 11, 11, 11, 11, 11, 11],
    [11, 11, 11, 11, 11, 11, 11, 11, 11],
    [11, 11, 11,  9,  9,  9,  9,  9,  9],
    [11, 11, 11,  9,  9,  9,  9,  9,  9],
    [11, 11, 11,  9,  9,  9,  9,  9,  9],
    [11, 11, 11,  9,  9,  9,  9,  9,  9],
    [11, 11, 11,  9,  9,  9,  9,  9,  9],
    [11, 11, 11,  9,  9,  9,  9,  9,  9],
], dtype=np.int64)
STATIC_DECO_R0 = 45
STATIC_DECO_C0 = 51


def _to_np(state: GridLike) -> np.ndarray:
    if isinstance(state, np.ndarray):
        return state.astype(np.int64, copy=True)
    return np.asarray(state, dtype=np.int64)


def _bg_value(r: int, c: int, meter_col: np.ndarray) -> int:
    """Return the background value at (r,c), using the given meter_col
    snapshot for column 63.

    `meter_col` is a length-63 array of the current state's col[:63, 63] —
    we preserve it from the input state so meter ticks are sticky.
    """
    if r == BG_BOTTOM_ROW:
        return BG_BOTTOM_VAL
    if c == BG_METER_COL:
        return int(meter_col[r])
    # Static right-side decoration (rows 45-52 cols 51-59) — but only the
    # non-9 cells of STATIC_DECO_R are part of the bg.
    dr = r - STATIC_DECO_R0
    dc = c - STATIC_DECO_C0
    if 0 <= dr < 9 and 0 <= dc < 9:
        dv = int(STATIC_DECO_R[dr, dc])
        if dv != 9:
            return dv
    if c in BG_DIVIDER_COLS:
        return 10
    return BG_PLAYFIELD


def _find_sprite_topleft(state: np.ndarray, template: np.ndarray,
                         sprite_color: int, col_lo: int, col_hi: int,
                         r0_min: int, r0_max: int,
                         c0_min: int, c0_max: int,
                         ) -> Optional[Tuple[int, int]]:
    """Find (r0,c0) such that the visible cells of `template` placed at
    (r0,c0) best match `state` inside the given column band.

    We do an exhaustive sliding-window search over r0 in [r0_min, r0_max]
    in steps of STEP (sprite always moves on a 3-cell grid) and c0 in
    [c0_min, c0_max] in steps of STEP. For each candidate we score:
       + 2 per opaque template cell that matches state
       + 1 per transparent (9) cell that is plausibly background
       - 5 per opaque template cell that disagrees with state (mismatch)
    The candidate with the highest score wins.
    """
    s = state
    best_score = -10 ** 9
    best = None
    for r0 in range(r0_min, r0_max + 1, STEP):
        for c0 in range(c0_min, c0_max + 1, STEP):
            score = 0
            ok = True
            for tr in range(SPRITE_H):
                for tc in range(SPRITE_W):
                    fr = r0 + tr
                    fc = c0 + tc
                    if not (0 <= fr < 64 and 0 <= fc < 64):
                        continue
                    # Skip the dynamic bars (row 63 / col 63) — they never
                    # carry sprite pixels.
                    if fr == BG_BOTTOM_ROW or fc == BG_METER_COL:
                        continue
                    tv = int(template[tr, tc])
                    sv = int(s[fr, fc])
                    if tv == sprite_color:
                        # opaque sprite-colour cell
                        if sv == sprite_color:
                            score += 2
                        else:
                            score -= 5
                            ok = False
                    elif tv == 0:
                        # 0-hole: shows 0 over playfield 9, divider 10 wins
                        if fc in BG_DIVIDER_COLS:
                            if sv == 10:
                                score += 1
                            else:
                                score -= 2
                        else:
                            if sv == 0:
                                score += 2
                            elif sv == BG_PLAYFIELD:
                                # less likely but possible boundary case
                                score += 0
                            else:
                                score -= 1
                    elif tv == 9:
                        # transparent: anything's ok, mild reward for bg
                        pass
            if score > best_score:
                best_score = score
                best = (r0, c0)
                _ = ok  # ok unused but reserved
    return best


def _erase_sprite(grid: np.ndarray, top_left: Tuple[int, int],
                  template: np.ndarray, meter_col: np.ndarray) -> None:
    """Replace sprite pixels at `top_left` with the underlying background."""
    r0, c0 = top_left
    for tr in range(SPRITE_H):
        for tc in range(SPRITE_W):
            fr = r0 + tr
            fc = c0 + tc
            if not (0 <= fr < 64 and 0 <= fc < 64):
                continue
            grid[fr, fc] = _bg_value(fr, fc, meter_col)


def _stamp_sprite(grid: np.ndarray, top_left: Tuple[int, int],
                  template: np.ndarray, meter_col: np.ndarray) -> None:
    """Stamp `template` at `top_left` over the current grid.

    Rendering rule per cell:
        tv in {4,5}: opaque, frame = tv
        tv == 9: transparent, leave frame as-is
        tv == 0: if bg has a non-9 feature here, leave bg; else frame = 0.
                 We respect what's already in `grid` at this cell — i.e. if
                 a non-playfield value is present (divider / col63 / row63
                 / static deco), keep it; else set to 0.
    """
    r0, c0 = top_left
    for tr in range(SPRITE_H):
        for tc in range(SPRITE_W):
            fr = r0 + tr
            fc = c0 + tc
            if not (0 <= fr < 64 and 0 <= fc < 64):
                continue
            # Never overwrite row 63 / col 63 (special bars)
            if fr == BG_BOTTOM_ROW or fc == BG_METER_COL:
                continue
            tv = int(template[tr, tc])
            if tv in (4, 5):
                grid[fr, fc] = tv
            elif tv == 9:
                # transparent: leave whatever's already there
                pass
            elif tv == 0:
                bv = _bg_value(fr, fc, meter_col)
                if bv == BG_PLAYFIELD:
                    grid[fr, fc] = 0
                else:
                    # let bg feature show
                    grid[fr, fc] = bv


def _tick_meter(grid: np.ndarray) -> bool:
    """Flip the topmost 11 in column 63 (rows 0..62) to 5. Returns True if
    something changed."""
    col = grid[:BG_BOTTOM_ROW, BG_METER_COL]
    idx = np.where(col == 11)[0]
    if idx.size == 0:
        return False
    grid[int(idx[0]), BG_METER_COL] = 5
    return True


def simulate(state: GridLike, action_id: int, x: int, y: int):
    """Predict next state, reward_class, done for game ar25."""
    grid = _to_np(state)
    action_id = int(action_id)

    # Action 6: NOOP, reward 0.
    if action_id == 6:
        return grid.tolist(), 0, False

    # Action 5: tick the meter only.
    if action_id == 5:
        out = grid.copy()
        _tick_meter(out)
        return out.tolist(), 1, False

    # Action 7: non-deterministic from one frame — predict identity. We
    # cannot recover the "previous action" needed to predict the sprite
    # move from a single frame. Reward=1 is the modal value (33/36), so
    # we predict 1. The 3 NOOP cases (reward=0) miss reward but match
    # state.
    if action_id == 7:
        return grid.tolist(), 1, False

    # Movement actions: 1, 2, 3, 4
    # Locate both sprites
    meter_col = grid[:BG_BOTTOM_ROW, BG_METER_COL].copy()
    l_pos = _find_sprite_topleft(
        grid, L_TEMPLATE, 5, 0, BG_METER_COL,
        SPRITE_R0_MIN, SPRITE_R0_MAX, L_C0_MIN, L_C0_MAX,
    )
    r_pos = _find_sprite_topleft(
        grid, R_TEMPLATE, 4, 33, BG_METER_COL,
        SPRITE_R0_MIN, SPRITE_R0_MAX, R_C0_MIN, R_C0_MAX,
    )

    dr_l, dc_l, dr_r, dc_r = 0, 0, 0, 0
    if action_id == 1:
        dr_l = -STEP; dr_r = -STEP
    elif action_id == 2:
        dr_l = +STEP; dr_r = +STEP
    elif action_id == 3:
        dc_l = -STEP; dc_r = +STEP
    elif action_id == 4:
        dc_l = +STEP; dc_r = -STEP
    else:
        # Unknown action — identity
        return grid.tolist(), 1, False

    # Decide new positions with per-sprite bounds.
    new_l = l_pos
    new_r = r_pos
    moved = False

    # For vertical actions (1,2): if EITHER would go OOB, both stay (NOOP).
    if action_id in (1, 2):
        ok = True
        if l_pos is not None:
            nr = l_pos[0] + dr_l
            if not (SPRITE_R0_MIN <= nr <= SPRITE_R0_MAX):
                ok = False
        if r_pos is not None:
            nr = r_pos[0] + dr_r
            if not (SPRITE_R0_MIN <= nr <= SPRITE_R0_MAX):
                ok = False
        if ok:
            if l_pos is not None:
                new_l = (l_pos[0] + dr_l, l_pos[1])
                moved = True
            if r_pos is not None:
                new_r = (r_pos[0] + dr_r, r_pos[1])
                moved = True
    else:
        # Horizontal mirrored actions (3, 4): L always moves; R may be blocked.
        if l_pos is not None:
            nc = l_pos[1] + dc_l
            if L_C0_MIN <= nc <= L_C0_MAX:
                new_l = (l_pos[0], nc)
                moved = True
        if r_pos is not None:
            nc = r_pos[1] + dc_r
            if R_C0_MIN <= nc <= R_C0_MAX:
                new_r = (r_pos[0], nc)
                moved = True
            # else: R stays in place (NOT a global NOOP)

    out = grid.copy()
    if moved:
        # Erase OLD sprites first (both, even unchanged — easier reasoning).
        if l_pos is not None:
            _erase_sprite(out, l_pos, L_TEMPLATE, meter_col)
        if r_pos is not None:
            _erase_sprite(out, r_pos, R_TEMPLATE, meter_col)
        # Stamp NEW positions (R after L so R wins on overlap — matches the
        # canonical rendering observed; in practice they never overlap).
        if new_l is not None:
            _stamp_sprite(out, new_l, L_TEMPLATE, meter_col)
        if new_r is not None:
            _stamp_sprite(out, new_r, R_TEMPLATE, meter_col)
        # Tick the meter
        _tick_meter(out)
        return out.tolist(), 1, False
    else:
        # NOOP (boundary): leave state alone
        return grid.tolist(), 0, False
