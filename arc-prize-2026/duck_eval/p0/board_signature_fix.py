"""P0.4 — draft fix for the no-op guard's board signature (ARM P0, 2026-08-22).

NOT APPLIED to any shipped bundle. This is the patch text for
``inference/agent/noop_guard.py``: replace its module-level
``board_signature`` with ``interior_signature`` at both call sites in
``tool_agent.py`` (lines ~1737, ~1832, ~1883 in the 08-07/08-15 bundles),
feeding it the exclusions from one per-game ``HudMask``. The interior is
also what ``board_changed`` must be computed on before it reaches
``NoopGuard.observe`` — fixing only the signature leaves the record path
blind (a HUD tick makes every action look like it changed the board, so
no-ops are never recorded in the first place).

Why: the guard keys on ``blake2b(repr(full 64x64 grid))``. 18/25 field
games render a border HUD/timer strip that changes on >=50% of steps
(measured on canivel/arc3-q38-field-eval intermediate_states.pkl), so the
``(level, board_before_sig, action_sig)`` key never recurs and the guard
has fired 0 times in 1639+3616 recorded actions. Re-keying on the board
interior (border-flush strips excluded) yields 22 correct blocks and 0
false blocks on the same real action stream.

The animation exemption in ``NoopGuard.observe`` (``animated=True`` is
never a no-op) is untouched — this module only changes what gets hashed.
"""
from __future__ import annotations

import hashlib
from typing import Any, Iterable, Sequence

# Fallback when no detection has converged yet: exclude nothing. A fixed
# top-strip default was considered and REJECTED on data: in the field
# corpus the ticking strip is bottom row 63 in 11/18 affected games, top
# row 0 in 3, and a side column in 4 — no single fixed strip covers even
# half the cases. Detection is cheap (first ~10 frames) and measured 0
# false blocks; a wrong fixed strip risks blocking on state the game
# actually plays in.
DEFAULT_EXCLUDE_ROWS: tuple[int, ...] = ()
DEFAULT_EXCLUDE_COLS: tuple[int, ...] = ()


def interior_signature(
    grid: Any,
    exclude_rows: Iterable[int] = DEFAULT_EXCLUDE_ROWS,
    exclude_cols: Iterable[int] = DEFAULT_EXCLUDE_COLS,
) -> str:
    """``board_signature`` over the gameplay interior.

    Identical to the shipped ``board_signature`` (blake2b-8 of ``repr`` of a
    tuple-of-tuples of ints) except rows/columns in the exclusion sets are
    dropped before hashing. With empty exclusions it returns byte-identical
    signatures to the shipped function, so a HudMask that detects nothing
    degrades to exactly the current behaviour.
    """
    xr, xc = set(exclude_rows), set(exclude_cols)
    rows = tuple(
        tuple(int(cell) for c, cell in enumerate(row) if c not in xc)
        for r, row in enumerate(grid or ())
        if r not in xr
    )
    digest = hashlib.blake2b(repr(rows).encode("utf-8"), digest_size=8)
    return digest.hexdigest()


def interior_changed(
    grid_before: Any,
    grid_after: Any,
    exclude_rows: Iterable[int],
    exclude_cols: Iterable[int],
) -> bool:
    """The ``board_changed`` the guard must observe: interior-only diff."""
    return interior_signature(grid_before, exclude_rows, exclude_cols) != \
        interior_signature(grid_after, exclude_rows, exclude_cols)


class HudMask:
    """Online detector of border-flush HUD/timer strips (P3 spec).

    Feed every consecutive settled-frame pair via :meth:`observe`. After
    ``min_pairs`` observations, :attr:`exclude_rows` / :attr:`exclude_cols`
    expose the contiguous border strips whose contents changed on
    ``>= threshold`` of observed steps. Before convergence the exclusions
    are empty, so signatures equal the shipped full-grid behaviour —
    strictly no worse than today.

    Exclusions only ever come from re-derivation over the full history, so
    a strip that stops ticking (e.g. a timer that freezes) drops back out.
    Keep one instance per game, reset on level transition if per-level HUDs
    differ (the field data did not require it: the detected strips were
    stable across levels in all 18 affected games).
    """

    def __init__(self, threshold: float = 0.5, min_pairs: int = 10) -> None:
        self.threshold = float(threshold)
        self.min_pairs = max(1, int(min_pairs))
        self._n = 0
        self._row_chg: list[int] = []
        self._col_chg: list[int] = []

    def observe(self, grid_before: Sequence[Sequence[int]],
                grid_after: Sequence[Sequence[int]]) -> None:
        if not grid_before or not grid_after:
            return
        n_rows = min(len(grid_before), len(grid_after))
        n_cols = min(len(grid_before[0]), len(grid_after[0])) if n_rows else 0
        if len(self._row_chg) < n_rows:
            self._row_chg += [0] * (n_rows - len(self._row_chg))
        if len(self._col_chg) < n_cols:
            self._col_chg += [0] * (n_cols - len(self._col_chg))
        for r in range(n_rows):
            if tuple(grid_before[r][:n_cols]) != tuple(grid_after[r][:n_cols]):
                self._row_chg[r] += 1
        for c in range(n_cols):
            if any(grid_before[r][c] != grid_after[r][c] for r in range(n_rows)):
                self._col_chg[c] += 1
        self._n += 1

    def _border_strip(self, counts: list[int]) -> set[int]:
        hits: set[int] = set()
        if self._n < self.min_pairs or not counts:
            return hits
        for i in range(len(counts)):               # flush to low border
            if counts[i] / self._n >= self.threshold:
                hits.add(i)
            else:
                break
        for i in range(len(counts) - 1, -1, -1):    # flush to high border
            if i in hits:
                break
            if counts[i] / self._n >= self.threshold:
                hits.add(i)
            else:
                break
        return hits

    @property
    def exclude_rows(self) -> set[int]:
        return self._border_strip(self._row_chg)

    @property
    def exclude_cols(self) -> set[int]:
        return self._border_strip(self._col_chg)

    def signature(self, grid: Any) -> str:
        """Convenience: interior_signature under the current mask."""
        return interior_signature(grid, self.exclude_rows, self.exclude_cols)
