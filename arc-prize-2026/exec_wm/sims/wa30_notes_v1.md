# wa30 sim v1 notes

## Game in one line

A 4x4 directional sprite (3-row body of color 14 + 1-row "head" of color 0)
moves N/S/E/W in 4-cell steps on a 64x64 background of color 1. Action 5 is
mostly a NOOP / step-counter tick. Reward stays 1 for moves, 0 for noop. No
level-up in 200 sample tuples.

## Invariants used

1. **Unique sprite anchor**: color 14 appears ONLY inside the body of the
   single 4x4 sprite. Its bbox is exactly 3x4 (vertical) or 4x3 (horizontal),
   and the adjacent row/col of 0s on one side unambiguously identifies
   facing direction.
2. **Pure cardinal translation**: actions 1/2/3/4 always shift the full 4x4
   bbox by exactly (-4,0)/(+4,0)/(0,-4)/(0,+4) and force facing to U/D/L/R
   respectively, regardless of prior facing.
3. **Boundary fallback**: when the new 4x4 footprint would leave the
   64x64 grid, the sprite stays in place and only its facing rotates.
   Modal prediction: n_changed in {0,1,6} — verified on 5/5 edge cases of
   action 4 against the right edge.

## v1 result (validate_sim.py --game wa30, full 200 tuples)

```json
{
  "game": "wa30",
  "split": "all",
  "n": 200,
  "errors": 0,
  "state_exact_pct": 65.0,
  "pixel_match_pct": 99.97216796875,
  "reward_acc_pct": 89.0,
  "done_acc_pct": 100.0,
  "by_action": {
    "1": {"n": 39, "exact_pct": 66.67, "pixel_pct": 99.977, "errors": 0},
    "2": {"n": 36, "exact_pct": 69.44, "pixel_pct": 99.974, "errors": 0},
    "3": {"n": 33, "exact_pct": 57.58, "pixel_pct": 99.927, "errors": 0},
    "4": {"n": 45, "exact_pct": 73.33, "pixel_pct": 99.981, "errors": 0},
    "5": {"n": 47, "exact_pct": 57.45, "pixel_pct": 99.990, "errors": 0}
  }
}
```

## What v1 still gets wrong (~35% of cases)

- **Row-63 step counter ticks** (~32% of move actions, n=33 instead of 32).
  The tick is rightmost-7 -> 4 and fires every ~3 actions but the trigger
  depends on a hidden internal clock not recoverable from the current
  frame. Modelling without certainty would hurt pixel_match more than help.
- **Action 5 reward + tick** (~20/47 cases). Currently we predict noop +
  reward=0; the modal action 5 is exactly noop+reward=0 (27/47).
- **Target-tile chain reactions** (~5 steps in 200): when the sprite walks
  into a 4x4 ring of color 3 / 4, an additional 16-44 cells recolor. Needs
  more samples to characterise the trigger condition reliably.

## v2 plan

1. **Counter parity tracker**: add an internal step counter inferred from
   the count of 4-valued cells in row 63 of `state_t`. If `count(4) %3 == 2`
   (after 200 tuples this matches the tick cadence), predict a tick on
   non-action-5 steps. Risk: wrong on edge of the 3-cycle, ~-1 pixel each.
2. **Action 5 toggle**: if `count(4_in_row_63) >= some_threshold` predict
   no tick (the counter saturated towards the end of the sample window).
3. **Target-ring detector**: scan for any 4x4 frame of solid color 3 in
   the playfield. When the sprite enters its bbox, apply the
   recolor-and-shift transform observed at steps 165/167/170.
4. Action 3 underperforms (57.6%) — recheck whether the bbox-detection
   for L-facing fails near the left edge (cols 0..3). Likely an off-by-one
   in `_find_sprite` when c0_14 == 1.

## Honest assessment

The +65 points exact-match came from two crisp invariants:
- 14-body uniqueness (lossless sprite detection).
- Deterministic cardinal translation with new-facing forcing.

Both should hold beyond the 200 tuples. The remaining 35% is dominated by
the hidden step-counter — same architectural ceiling we hit on bp35. v2
should target that ceiling with a parity-based counter heuristic before
investing in target-tile chain logic.
