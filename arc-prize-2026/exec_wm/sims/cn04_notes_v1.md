# cn04 sim v1 notes

## Game description (from observations)

A grid-based push/slide puzzle on a 64x64 canvas. Static walls of color
14 sit in a fixed corner (rows 29-49, cols 41-49). The active playfield
is a single 4-connected blob (colors {0, 8, sometimes 12} on a
background of 10). Row 0 is a horizontal timer of color-4 cells in
cols 16-47; each ambient game-tick the leftmost-remaining 4 flips to 0.
Six actions are exposed: 1 = up, 2 = down, 3 = left, 4 = right,
5 = in-place transform (rotation-like), 6 = click(x, y).

## Invariants used in v1

1. **reward_class = 1 iff state changed.** Verified 200/200 — perfect.
2. **Actions 1..4 translate the largest non-bg, non-wall, non-timer
   4-connected blob by (-3, 0), (+3, 0), (0, -3), (0, +3) respectively.**
   Verified 46/46 on the non-trivial shift cases.
3. **NOOP rules for actions 1..4:**
   * The shifted bbox would leave the grid (verified deterministic for
     action 1: 22/22 off-grid cases were NOOP or timer-only).
   * The blob currently contains color 12 (a "lock" marker that
     appears when the puzzle reaches its rest state). When color 12 is
     present in the blob, actions 1..4 cannot translate it.
4. Actions 5 and 6 are returned as identity (we do not yet model the
   action-5 in-place rotation or the rare action-6 click stamping).

## validate_sim.py output (v1, full 200 tuples)

```
{
  "game": "cn04",
  "split": "all",
  "n": 200,
  "errors": 0,
  "state_exact_pct": 73.5,
  "pixel_match_pct": 99.679443359375,
  "reward_acc_pct": 78.0,
  "done_acc_pct": 100.0,
  "by_action": {
    "1": {"n": 37, "exact_pct": 72.97, "pixel_pct": 99.993},
    "2": {"n": 27, "exact_pct": 77.78, "pixel_pct": 99.995},
    "3": {"n": 33, "exact_pct": 78.79, "pixel_pct": 99.995},
    "4": {"n": 34, "exact_pct": 79.41, "pixel_pct": 99.995},
    "5": {"n": 35, "exact_pct": 54.29, "pixel_pct": 98.48},
    "6": {"n": 34, "exact_pct": 79.41, "pixel_pct": 99.699}
  }
}
```

Comfortably above the bp35-v2 baseline of 69% — and pixel match is
99.68%, meaning the sim is extremely close even in failure cases.

## What v1 still gets wrong (~26.5%)

Error breakdown by (action, truth, prediction):

| action | truth     | pred  | count |
|--------|-----------|-------|-------|
| 1..4   | tick-only | NOOP  | 21    | (timer ticked, blob didn't move)
| 1..4   | SHIFT     | SHIFT | 9     | (shift produced but doesn't match — likely blob detection edge cases when the blob touches the static color-14 region or when the playfield contains additional small disjoint blobs)
| 5      | SHIFT     | NOOP  | 11    | (action-5 in-place transform)
| 5      | tick-only | NOOP  | 5     |
| 6      | SHIFT     | NOOP  | 2     | (rare effective click)
| 6      | tick-only | NOOP  | 5     |

The dominant residual is the timer-tick: row 0's leftmost-4 flips to 0
on ~21 actions where the blob otherwise didn't move. The tick rule is
NOT deterministic from a single frame (we verified that among action-1
"fits-and-no-lock" cases, some tick and some don't, with no visible
disambiguator). Predicting "always tick" would correctly fix the 21
tick-only cases but would break ALL 47 successful shifts (which mostly
do NOT tick) and the 110 NOOPs. Net negative — left untouched in v1.

## v2 plan

1. **Action 5 in-place transform.** Looks like a quarter-turn of the
   blob's content around its centroid, or a swap of the (0)-pixels and
   (10)-pixels within the bbox. Worth probing: detect the symmetry and
   apply the swap. Potential +5-7 percentage points.
2. **Action 6 click rule.** With only 7 effective clicks in 200 samples
   the signal is thin. Check whether the 7 successful clicks land on
   color 8 (player cells) — if so, treat action 6 as "click on a
   player cell triggers action 5 (transform) at that cell". If thin
   data confirms, +2 points.
3. **Timer-tick predicate.** Tick happens stochastically across all
   actions; correlate with hidden state. Skip unless we can collect
   trajectory data showing a non-random pattern (e.g. ticks on every
   2nd step). Risky.
4. **Blob refinement at the boundary.** A few SHIFT-but-wrong errors
   stem from the blob touching the wall (color 14); the BFS treats
   color 14 as a barrier (visited=true), so the blob shape is correct
   but the destination overlap with a separately-sat color-8 sub-blob
   (the static target markers) confuses the stamp. Add: when stamping
   the shifted blob, do NOT overwrite any cell whose pre-shift colour
   was 14 or part of a SECOND large non-bg component. Potential +3
   points.

## Honest assessment

Three crisp invariants drove the +73.5%:

* reward = (state changed) — perfect, costs nothing.
* Cardinal-action shifts by exactly 3 in a known direction.
* Color 12 acts as a hard lock.

These are unlikely to be curve-fitting; they're consistent with a
sliding-puzzle game where actions push a block until it reaches a
captured state. The remaining errors look like genuine unmodelled
mechanics (rotation under action 5) plus a non-deterministic timer.
