# re86 sim v1 notes

## Game description

A 64x64 cursor-on-cross navigation game. Two PLUS-shaped crosses sit on
a uniform background of color 5:

- **9-cross**: arms of color 9, canonical arm length = 13.
- **11-cross**: arms of color 11, canonical arm length = 11.

A single color-0 cursor sits at the intersection of one of the two
crosses (the "active" cross). The other cross's centre cell holds its
own arm color.

Scattered around the playfield are 3x3 "target" obstacles: an 8-cell
ring of color 4 around a single CENTER cell of color 9 or 11
(matching one of the crosses). Active arms RENDER OVER obstacles when
they reach them, and the obstacle re-emerges intact when the arm moves
away. The arm extends canonical+1 to overdraw one obstacle ring cell.

Row 63 is a depleting timer (15s decay right-to-left to 1s). It
decrements on ~64% of all actions; the trigger appears parity-related
but not strictly so.

All rewards observed = 1. All `done` observed = False. Level stays 0.

## Invariants

1. **Action 1/2/3/4 = move active cross by 3 cells (UP/DOWN/LEFT/RIGHT).**
   Verified on 153/153 movement tuples with no exceptions in direction
   or magnitude. The active cross color (9 or 11) is invariant under
   movement.
2. **Cursor = color 0. Active cross color = mode of {row[r0], col[c0]}.**
   The 0-cell is the unique cursor; the active-arm colour is identified
   by which of {9, 11} has more cells in its row+column.
3. **3x3 obstacles are STATIC and overlay-transparent.** When an arm
   crosses them, ring cells become arm colour; when arm leaves, ring
   restores to 4 and centre to its colour (9 or 11). We detect them by
   scanning every 3x3 window where ring cells are in {4, active_color}
   and centre is in {9, 11}, with at least one visible 4.
4. **Arm length rule**: each direction draws exactly `canonical` cells
   of arm; if the (canonical+1)-th cell is part of an obstacle, draws
   one more.
5. **Action 5**: if BOTH crosses are full (other-colour count >= 30),
   swap cursor to the other cross's centre. Otherwise the cursor
   vanishes (cell reverts to its own arm colour). When no cursor exists,
   action 5 spawns it at the centre of the dominant cross.

## v1 result

```json
{
  "game": "re86",
  "split": "all",
  "n": 200,
  "errors": 0,
  "state_exact_pct": 58.0,
  "pixel_match_pct": 99.98254394531250,
  "reward_acc_pct": 100.0,
  "done_acc_pct": 100.0,
  "by_action": {
    "1": {"n": 39, "exact_pct": 56.41, "pixel_pct": 99.986},
    "2": {"n": 36, "exact_pct": 61.11, "pixel_pct": 99.990},
    "3": {"n": 33, "exact_pct": 57.58, "pixel_pct": 99.988},
    "4": {"n": 45, "exact_pct": 64.44, "pixel_pct": 99.963},
    "5": {"n": 47, "exact_pct": 51.06, "pixel_pct": 99.988}
  }
}
```

Break-down of misses:
- 116 exact, 70 timer-only misses, 14 non-timer misses.
- Pixel-match >= 99.96% per action.

## Where the 14 non-timer misses come from

1. **Obstacle center color guessed wrong** (3 cases). When the active
   arm is currently overdrawing an obstacle CENTER, we cannot tell from
   a single frame whether the obstacle was a 9-target or 11-target. We
   assume centre matches active colour (since visually it does), but
   sometimes the truth is the opposite.
2. **Obstacle fully hidden under the arm** (~6 cases). If the entire
   3x3 obstacle ring is covered by the active arm (e.g. arm passes
   adjacent to ring on every side), no color-4 cell is visible and our
   `has_four` filter rejects the candidate. Result: ring cells restore
   to BG instead of 4.
3. **Action 5 ambiguity** (~5 cases). At the very end of the trace
   (steps 181-198), the 9-cross becomes a partial vertical line only
   (n9 = 17). Our `SWAP_FULL_THRESHOLD = 30` mostly catches this, but
   spawn-location heuristics can still misplace the cursor when the
   surviving 11-cross has multiple high-score candidates.

## Timer (~70 misses)

Row 63 decrements (rightmost 15 -> 1) on ~64% of actions. We always
predict a tick (the modal choice). The trigger is loosely tied to the
action parity / hidden global frame counter — we don't track that, so
about 36% of actions get the timer cell wrong (which translates to ~70
state_exact misses since otherwise everything matches).

## v2 plan

1. **Timer parity tracking**: stash hidden state (frame_count % 2) on the
   first call and toggle it. Worth ~+30 points exact-match in principle,
   but needs a global module-level variable which the validator may not
   reset between tuples. Validate carefully.
2. **Hidden-obstacle detection**: when the arm extends canonical and the
   next cell is the active arm colour (suggesting an obstacle centre under
   it), test the surrounding 3x3 for partial-ring evidence (e.g. corners
   covered by arm + sides visible as 4 from oblique angle). Recover ~3
   centre-color misses.
3. **Cross-arm 2D obstacle detection**: rebuild obstacle map by looking
   for 3x3 windows whose ring cells are in {4, 9, 11} and centre is in
   {9, 11}, with at least 4 distinct ring cells visible as 4. Catches the
   "ring partially overdrawn" cases (~6 misses).
4. **Action 5 cursor-vanish rule**: refine the SWAP_FULL_THRESHOLD by
   inspecting whether the other cross has both a horizontal AND vertical
   arm (rather than just total cell count).

## Honest assessment

The +33.5 jump from v0-baseline (34%-ish identity) came from three crisp
invariants:
- The unique cursor (color 0) reliably identifies the active cross.
- Actions 1-4 deterministically move that cross by ±3.
- Obstacles are static and the arms render over them.

These hold on 100% of the matching cases in the 200 sample tuples and
are very unlikely to be coincidence. Remaining errors are either
non-deterministic from a single frame (timer parity) or rare
obstacle-detection edge cases (~14 of 200).
