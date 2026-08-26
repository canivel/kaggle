# wa30 sim v2 notes

## Decision: ship v2, flagged as marginal

v2 adds **target-tile detection** and treats tiles as obstacles for sprite
movement (same fallback semantics as the grid-edge boundary check in v1).
It does NOT predict the row-63 step counter and does NOT attempt to toggle
tile colors (an aggressive toggle heuristic regressed actions 2 and 4 by
6-7 abs pts in pilot run, so we held back).

## Score comparison (validate_sim.py --game wa30, all 200 tuples)

| metric            | v1        | v2        | delta    |
|-------------------|-----------|-----------|----------|
| state_exact_pct   | 65.0      | 65.0      | 0.0      |
| pixel_match_pct   | 99.9722   | 99.9785   | +0.0063  |
| reward_acc_pct    | 89.0      | 89.0      | 0.0      |
| done_acc_pct      | 100.0     | 100.0     | 0.0      |
| errors            | 0         | 0         | 0        |

### By-action exact %

| action | n  | v1 exact | v2 exact | delta |
|--------|----|----------|----------|-------|
| 1      | 39 | 66.67    | 66.67    | 0     |
| 2      | 36 | 69.44    | 69.44    | 0     |
| 3      | 33 | 57.58    | 57.58    | 0     |
| 4      | 45 | 73.33    | 73.33    | 0     |
| 5      | 47 | 57.45    | 57.45    | 0     |

state_exact_pct is unchanged because the 2 tile-collision tuples produced
"wrong sprite" predictions in BOTH v1 and v2 (different wrongs), but
pixel_match improves on those 2 tuples (v2 stops overwriting the tile
border with sprite pixels), bringing per-cell accuracy up ~0.006 abs pts.

## Active sim

`f:/kaggle/arc-prize-2026/exec_wm/sims/wa30_sim.py` (= v2).
Copy of source kept at `wa30_sim_v2.py`.

## What v2 changes structurally

1. `_find_tile_bboxes(grid)` — scans for 4x4 tiles defined by:
   - 2x2 interior of color 9
   - solid border of color in {3, 4}
2. `_apply_move` — after the grid-boundary check, additionally checks if the
   proposed 4x4 footprint overlaps any tile bbox. If so, treat tile as a wall
   and only rotate the sprite (matches the i=163 D->L re-orient case).

## What v2 deliberately does NOT do (and why)

### Row-63 step counter ticks (-64 cases unfixable)

The row-63 counter ticks "rightmost 7 -> 4" roughly every 3 actions but
the tick is paced by a hidden clock not encoded anywhere in the current
frame:

```
k=0 (c_t%3==0): 112/200 correct
k=1 (c_t%3==1): 111/200
k=2 (c_t%3==2): 113/200
```

All near 56% i.e. random for a 1/3 prior. Predicting tick always also
fails (only 32% of tuples tick). The 64 tick-cases are essentially
uncoverable statelessly; they cost 1 pixel each (0.024% per case) so the
99.97% pixel ceiling is real.

### Tile color toggle on adjacency

Empirically when the sprite enters/leaves edge-adjacency with a tile, the
tile flips 3<->4 (i=165,166,167,170,171,188,189). But the trigger also
depends on facing/no-toggle-when-blocked-twice (i=163 toggles, i=164
doesn't, same position and action). An aggressive "toggle on any
adjacency change" rule regressed:

| action | v1   | aggressive v2 |
|--------|------|---------------|
| 2      | 69.44| 58.33 (-11.1) |
| 4      | 73.33| 68.89 (-4.4)  |

so we shipped the conservative variant only.

### Sprite-cascade after tile interactions (i=171, 188-189)

Cases where the sprite is "carried" past the tile (i=171: sprite at
(28..31,44..47) moves through the tile column to (28..31,40..43) AND the
tile toggles). The sprite still lands in a position consistent with a
plain shift, so v1 logic happens to score these correctly on pixel match
when the toggle bleeds only 12 pixels.

## Honest signal vs curve-fitting

- **Genuine signal**: tile detection (4x4 + interior-9 + solid border 3/4)
  is unambiguous; 3 unique tiles, 0 false positives across 200 frames.
  Tile-as-obstacle physics matches at i=163 deterministically. This rule
  WILL hold on fresh observations.
- **Risk surface**: the test bench has only 8 tile-toggle events; no tile
  is ever destroyed or moved. We may underfit any "tile destroyed on
  3rd consecutive collision" rule. Acceptable risk given small event count.
- **Marginal status**: net state-exact gain is 0.0 pts, below the 2-pt bar.
  Pixel match improves +0.006 abs but pixel is already saturated. We ship
  because v2 is at worst identical to v1 on every metric AND introduces
  the correct tile-physics scaffold that v3+ can extend.

## Kaggle-inference usefulness

MODERATE.

- The sim is mostly a translation predictor on a 64x64 grid — useful as a
  forward model for plan-and-act over 4 cardinal directions. Pixel match
  99.98% means ~1 cell off per step, low compounding error over 10-20
  step rollouts.
- Tile-as-obstacle now propagates correctly, so MCTS/BFS over the action
  tree won't waste branches assuming the sprite warps through tiles.
- Row-63 ticks and tile-toggle physics remain unmodelled. For planning
  this matters only if the reward is tied to tile toggles (we have no
  evidence yet — `reward_class=1` consistently for all 1..4 actions).
  If the game's win-condition turns out to be "toggle all tiles to color
  3" or similar, v3 needs the toggle rule (which requires per-tile state
  beyond a single frame).

## v3 plan (if observed)

1. **Per-call toggle state machine**: with sequential calls in the
   harness, a module-level "last seen tile colors" cache can resolve the
   facing-change vs already-bumping ambiguity. Trade-off: state in the
   sim breaks the stateless contract, but boosts state_exact by ~4 cases.
2. **Sprite-through-tile redirection** (i=171): when sprite would
   collide with a tile but is moving along the tile's perpendicular axis,
   shift by 3 cells instead of 4 (squeeze past). Currently 1 case in 200.
3. **Reward=2 / done event**: not observed yet. Re-trigger v3 the moment
   any tuple has `reward_class != 1` for actions 1..4 or `done=True`.
