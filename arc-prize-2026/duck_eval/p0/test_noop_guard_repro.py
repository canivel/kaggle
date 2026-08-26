"""P0.3 reproduction + P0.4 fix tests (ARM P0, 2026-08-22).

Reproduces the §1.7 instrument defect against the REAL shipped NoopGuard
(imported read-only from the vendored 08-15 bundle, byte-identical to the
08-07 anim bundle's copy) and shows the interior signature fix makes it
able to fire. No shipped file is modified.

Run:  py -3.13 -m pytest duck_eval/p0/test_noop_guard_repro.py -q
  or: py -3.13 duck_eval/p0/test_noop_guard_repro.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BUNDLE = REPO / "duck_eval" / "private" / "bundle_20260815" / "src" / "ARC3-Inference"
sys.path.insert(0, str(BUNDLE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inference.agent.noop_guard import NoopGuard, board_signature  # noqa: E402

from board_signature_fix import (  # noqa: E402
    HudMask,
    interior_changed,
    interior_signature,
)

LEVEL = 1
NOOP_ACTION = "ACTION2"


def make_frame(tick: int, player_col: int = 5) -> list[list[int]]:
    """64x64 board with a HUD timer strip in row 63 that changes every step.

    Interior: all zeros except a player pixel at (10, player_col).
    HUD row 63: a shrinking bar, exactly the pattern the system prompt warns
    about ('a long horizontal ... line near an edge ... often shrinks or
    changes each step').
    """
    grid = [[0] * 64 for _ in range(64)]
    grid[10][player_col] = 3
    grid[63] = [7 if c < 64 - tick else 0 for c in range(64)]
    return grid


def drive(guard, sig_fn, changed_fn, n_repeats: int = 3):
    """Replay the same no-op action n times while the HUD ticks.

    Returns the list of is_known_noop() verdicts taken BEFORE each execution
    (what the harness checks at tool_agent.py:1779/1858).
    """
    verdicts = []
    for t in range(n_repeats):
        before, after = make_frame(t), make_frame(t + 1)  # interior identical
        sig = sig_fn(before)
        verdicts.append(guard.is_known_noop(LEVEL, sig, NOOP_ACTION))
        guard.observe(level=LEVEL, board_before_sig=sig, action_sig=NOOP_ACTION,
                      board_changed=changed_fn(before, after), animated=False)
    return verdicts


# ---------------------------------------------------------------------------
# P0.3a — the defect, reproduced: full-grid signature + full-grid
# board_changed (solver.py:818 semantics). The ticking HUD defeats the guard
# through BOTH paths: board_changed is always True so nothing is recorded,
# and the pre-state hash never recurs so nothing could ever match.
# ---------------------------------------------------------------------------

def test_guard_cannot_fire_with_ticking_hud():
    guard = NoopGuard()
    full_changed = lambda b, a: b != a  # noqa: E731
    verdicts = drive(guard, board_signature, full_changed, n_repeats=4)
    assert verdicts == [False, False, False, False], (
        "expected the shipped guard to NEVER recognise the repeated no-op "
        f"under a ticking HUD, got {verdicts}"
    )


def test_guard_cannot_fire_even_if_recording_were_fixed():
    """Fixing only board_changed (not the signature) still never fires:
    the (level, board_before_sig, action) key changes with every HUD tick."""
    guard = NoopGuard()
    mask_rows = {63}
    int_changed = lambda b, a: interior_changed(b, a, mask_rows, ())  # noqa: E731
    verdicts = drive(guard, board_signature, int_changed, n_repeats=4)
    assert verdicts == [False, False, False, False]


# ---------------------------------------------------------------------------
# P0.3b — the fix: signature over the interior (HUD band excluded) and
# board_changed computed on the interior. The guard fires on the repeat.
# ---------------------------------------------------------------------------

def test_guard_fires_once_hud_band_excluded():
    guard = NoopGuard()
    mask_rows = {63}
    sig = lambda g: interior_signature(g, mask_rows, ())  # noqa: E731
    int_changed = lambda b, a: interior_changed(b, a, mask_rows, ())  # noqa: E731
    verdicts = drive(guard, sig, int_changed, n_repeats=3)
    assert verdicts[0] is False          # first attempt: unknown, executes
    assert verdicts[1] is True, "guard must refuse the first repeat"
    assert verdicts[2] is True


def test_real_action_still_executes_under_interior_signature():
    """An action that moves the player is never blocked, HUD or no HUD."""
    guard = NoopGuard()
    mask_rows = {63}
    for t in range(3):
        before = make_frame(t, player_col=5 + t)
        after = make_frame(t + 1, player_col=5 + t + 1)
        sig = interior_signature(before, mask_rows, ())
        assert guard.is_known_noop(LEVEL, sig, "ACTION1") is False
        guard.observe(level=LEVEL, board_before_sig=sig, action_sig="ACTION1",
                      board_changed=interior_changed(before, after, mask_rows, ()),
                      animated=False)


def test_animation_exemption_preserved():
    """Animated actions are never recorded as no-ops (the docstring's ft09/
    sb26 regression). The fix must not change this: identical interior +
    animated=True must stay unblocked."""
    guard = NoopGuard()
    mask_rows = {63}
    for t in range(3):
        before = make_frame(t)
        sig = interior_signature(before, mask_rows, ())
        assert guard.is_known_noop(LEVEL, sig, NOOP_ACTION) is False, (
            "animated action wrongly memorised as no-op"
        )
        guard.observe(level=LEVEL, board_before_sig=sig, action_sig=NOOP_ACTION,
                      board_changed=False, animated=True)


# ---------------------------------------------------------------------------
# HudMask detection
# ---------------------------------------------------------------------------

def test_hudmask_detects_ticking_bottom_row():
    mask = HudMask(threshold=0.5, min_pairs=10)
    for t in range(12):
        mask.observe(make_frame(t), make_frame(t + 1))
    assert mask.exclude_rows == {63}
    assert mask.exclude_cols == set()


def test_hudmask_empty_before_convergence_matches_shipped_signature():
    mask = HudMask(min_pairs=10)
    mask.observe(make_frame(0), make_frame(1))  # 1 < min_pairs
    g = make_frame(0)
    assert mask.exclude_rows == set() and mask.exclude_cols == set()
    assert mask.signature(g) == board_signature(g), (
        "with no exclusions the fixed signature must be byte-identical to "
        "the shipped board_signature"
    )


def test_hudmask_ignores_non_border_activity():
    """A busy row in the middle of the board is gameplay, not HUD."""
    mask = HudMask(threshold=0.5, min_pairs=10)
    for t in range(12):
        b, a = make_frame(t), make_frame(t + 1)
        b[30][t % 64] = 9
        a[30][(t + 1) % 64] = 9
        mask.observe(b, a)
    assert 30 not in mask.exclude_rows


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if fails else 0)
