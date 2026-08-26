"""Executable world model for ARC-AGI-3 game `ar25` — v2.

Big change from v1: sprites are tracked by their LOGICAL top-left position;
rendering applies clip + z-order from raw templates. Previous-state inference
finds the logical (r0, c0) by template matching, then we translate by the
action delta and re-render. This handles the "cut" sprites correctly.

GAME: two 9x9 sprites (L 5/0, R 4) translate in mirrored directions on the
split playfield. col-63 meter ticks 11→5 with each successful action.

INVARIANTS:
1. Static background: playfield=9, divider cols 30-32 = 10, col 63 = meter
   (sticky 11→5 from top), row 63 = 5, static decoration rows 45-52 cols
   51-59 (the 11-cells of an L-shape).
2. Two sprites L (5/0) and R (4) with fixed 9x9 templates. L is the left
   sprite, R the right. Rendering order: background → R → divider-wins-
   over-R → L → L-over-divider keeps divider through L's 0-holes.
3. Movement actions (3 cells):
   - 1: both UP 3 rows
   - 2: both DOWN 3 rows
   - 3: L LEFT 3, R RIGHT 3 (mirrored)
   - 4: L RIGHT 3, R LEFT 3 (mirrored)
   - 5: tick meter only
   - 6: NOOP, reward 0
   - 7: identity (single-frame ambiguous; modal heuristic = identity)
4. Vertical boundary: if either sprite's new r0 would be < 0 or > 27,
   BOTH sprites NOOP (reward=0).
   Horizontal: per-sprite logical c0 is unconstrained — clipping at render
   handles the "cut" appearance. (Empirically this is what produces the
   sprite-cut behavior.)
5. Reward = 1 if any movement / meter tick succeeded; 0 otherwise.
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

# Sprite vertical bounds (r0 stays in [0, 27])
R0_MIN = 0
R0_MAX = 27

# Background constants
BG_PLAYFIELD = 9
DIVIDER_COLS = (30, 31, 32)
DIVIDER_VAL = 10
METER_COL = 63
METER_VAL = 11
BOTTOM_ROW = 63
BOTTOM_VAL = 5

# Static 9x9 decoration of 11-color at rows 45-52, cols 51-59
DECO_R0 = 45
DECO_C0 = 51
DECO = np.array([
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


def _to_np(state: GridLike) -> np.ndarray:
    if isinstance(state, np.ndarray):
        return state.astype(np.int64, copy=True)
    return np.asarray(state, dtype=np.int64)


def _build_bg(meter_col: np.ndarray, bottom_row: np.ndarray) -> np.ndarray:
    """Reconstruct the background (no sprites) using the current meter and
    bottom-row state, which we carry over from the input state."""
    bg = np.full((64, 64), BG_PLAYFIELD, dtype=np.int64)
    for c in DIVIDER_COLS:
        bg[:, c] = DIVIDER_VAL
    # Static decoration overlay (only its non-9 cells)
    for dr in range(9):
        for dc in range(9):
            v = int(DECO[dr, dc])
            if v != BG_PLAYFIELD:
                bg[DECO_R0 + dr, DECO_C0 + dc] = v
    bg[:BOTTOM_ROW, METER_COL] = meter_col
    bg[BOTTOM_ROW, :] = bottom_row
    return bg


def _render_r_cells(r0: int, c0: int):
    """Yield (fr, fc, 4) cells for R at (r0, c0), excluding divider cols and
    bottom/meter pixels."""
    for tr in range(SPRITE_H):
        for tc in range(SPRITE_W):
            tv = int(R_TEMPLATE[tr, tc])
            if tv != 4:
                continue
            fr = r0 + tr; fc = c0 + tc
            if not (0 <= fr < 64 and 0 <= fc < 64):
                continue
            if fc in DIVIDER_COLS:
                continue  # R can't paint over divider
            if fr == BOTTOM_ROW or fc == METER_COL:
                continue
            yield fr, fc, 4


def _render_l_cells(r0: int, c0: int):
    """Yield (fr, fc, value) cells for L at (r0, c0). Template 5 → 5;
    template 0 → 0 (playfield) or 10 (divider showing through).
    Skip out-of-bounds and bottom/meter pixels."""
    for tr in range(SPRITE_H):
        for tc in range(SPRITE_W):
            tv = int(L_TEMPLATE[tr, tc])
            if tv not in (5, 0):
                continue
            fr = r0 + tr; fc = c0 + tc
            if not (0 <= fr < 64 and 0 <= fc < 64):
                continue
            if fr == BOTTOM_ROW or fc == METER_COL:
                continue
            if tv == 5:
                yield fr, fc, 5
            else:
                # 0-hole: show divider if at divider col, else 0
                if fc in DIVIDER_COLS:
                    yield fr, fc, DIVIDER_VAL
                else:
                    yield fr, fc, 0


def _score_match(state: np.ndarray, cells, missing_penalty: int = 2) -> int:
    """Score: +1 per matching cell; -missing_penalty per non-matching cell.
    Cells that aren't where the template predicts (cut sprite) are partially
    excused — they count as 0 instead of -penalty if the state shows a
    plausible background (9 or 10).
    """
    score = 0
    for fr, fc, v in cells:
        sv = int(state[fr, fc])
        if sv == v:
            score += 1
        elif sv in (BG_PLAYFIELD, DIVIDER_VAL):
            # Possibly cut sprite — no penalty, no bonus
            pass
        else:
            score -= missing_penalty
    return score


def _infer_l_pos(state: np.ndarray) -> Tuple[int, int]:
    """Find L's logical (r0, c0). Search the 3-grid range needed."""
    best = None
    best_score = -10 ** 9
    # L logical c0 can be very negative (sprite mostly clipped off-screen)
    for r0 in range(R0_MIN, R0_MAX + 1, STEP):
        for c0 in range(-21, 34, STEP):
            cells = list(_render_l_cells(r0, c0))
            sc = _score_match(state, cells)
            if sc > best_score:
                best_score = sc
                best = (r0, c0)
    return best


def _infer_r_pos(state: np.ndarray) -> Tuple[int, int]:
    """Find R's logical (r0, c0). R can be anywhere too (template gets clipped
    by divider on render)."""
    best = None
    best_score = -10 ** 9
    for r0 in range(R0_MIN, R0_MAX + 1, STEP):
        for c0 in range(12, 60, STEP):
            cells = list(_render_r_cells(r0, c0))
            sc = _score_match(state, cells)
            if sc > best_score:
                best_score = sc
                best = (r0, c0)
    return best


def _tick_meter(grid: np.ndarray) -> bool:
    col = grid[:BOTTOM_ROW, METER_COL]
    idx = np.where(col == METER_VAL)[0]
    if idx.size == 0:
        return False
    grid[int(idx[0]), METER_COL] = 5
    return True


def simulate(state: GridLike, action_id: int, x: int, y: int):
    """Predict next state, reward_class, done for game ar25."""
    grid = _to_np(state)
    a = int(action_id)

    if a == 6:
        return grid.tolist(), 0, False

    if a == 5:
        out = grid.copy()
        _tick_meter(out)
        return out.tolist(), 1, False

    if a == 7:
        # Modal heuristic: identity (single-frame can't disambiguate prior
        # action). 3 of 36 cases pass via this. Try also "tick meter" as
        # secondary — empirically worse than identity, so stick with id.
        return grid.tolist(), 1, False

    if a not in (1, 2, 3, 4):
        return grid.tolist(), 1, False

    # Movement actions
    meter_col = grid[:BOTTOM_ROW, METER_COL].copy()
    bottom_row = grid[BOTTOM_ROW, :].copy()

    l_pos = _infer_l_pos(grid)
    r_pos = _infer_r_pos(grid)

    dr_l = dc_l = dr_r = dc_r = 0
    if a == 1:
        dr_l = -STEP; dr_r = -STEP
    elif a == 2:
        dr_l = +STEP; dr_r = +STEP
    elif a == 3:
        dc_l = -STEP; dc_r = +STEP
    elif a == 4:
        dc_l = +STEP; dc_r = -STEP

    new_l = (l_pos[0] + dr_l, l_pos[1] + dc_l)
    new_r = (r_pos[0] + dr_r, r_pos[1] + dc_r)

    # Vertical OOB: if either sprite would leave the [0, 27] band, NOOP.
    if a in (1, 2):
        if not (R0_MIN <= new_l[0] <= R0_MAX) or not (R0_MIN <= new_r[0] <= R0_MAX):
            return grid.tolist(), 0, False

    # Render: bg → R cells → L cells (L wins over R).
    out = _build_bg(meter_col, bottom_row)
    for fr, fc, v in _render_r_cells(*new_r):
        out[fr, fc] = v
    for fr, fc, v in _render_l_cells(*new_l):
        out[fr, fc] = v

    _tick_meter(out)
    return out.tolist(), 1, False
