# g50t sim v1 notes

## Game in one line

64x64 maze grid. A 5x5 sprite of color 9 (with a single bg-5 cell at the
centre) navigates a 6x6 cell grid anchored at (row 8, col 14). Actions
1/2/3/4 = up/down/left/right by 6 cells; action 5 swaps colour-indicator
panels and toggles a secondary playfield region (complex hidden state).

Available actions: {1, 2, 3, 4, 5}. Done is always False; level stays 0.

## Observations

### Player sprite
- 5x5 block of 9s with a single bg-5 cell at offset (2,2). Detected in
  200/200 states by counting 24 nines in a 5x5 window + center-hole check.
- Initial position (8, 14).  Row slots used in 200 tuples: {8, 14, 20, 26}.
  Col slots: {14, 20, 26, 32, 38}. Step = 6 in both axes.

### Movement invariants (verified)
- Whenever the sprite moved between state_t and state_t1 in a non-action-5
  tuple, the displacement was exactly (dr, dc) for the action, **80/80**.
- Action 1 = (-6, 0), 2 = (+6, 0), 3 = (0, -6), 4 = (0, +6).

### Wall/path encoding
- Path cell value = **5**. Wall/door = **8**. Outside (no playfield) = **0**.
  Trail colour cells = **2**. Indicator panel uses **1, 2, 9**.
- Movement NOOP iff the 5x5 target slot is **entirely zero**.  Slots
  containing 5/8/2 are all PASSABLE.
- 78 free moves + 73 NOOPs (all-zero target) + 2 corner cases (target had
  8 or 2, sprite still moved) are consistent with this rule.

### Reward
- reward_class == 1 iff state_t != state_t1, perfectly. (no level / done
  toggles in 200 tuples.)
- Best stateless predictor: `1 if not blocked-NOOP else 0` -> 79% on 200.

### Counter row 63 (NOT modelled in v1)
- A countdown band: rightmost-9 column gets overwritten with 1 on roughly
  every second action.  Cadence is non-deterministic from a single frame
  (~half of moves tick, ~half of NOOPs tick).
- v1 leaves row 63 untouched, eating ~50% of state_exact on tick steps.

### Action 5 (NOT modelled in v1)
- 47 cases, n_changed in {0, 1, 23, 24, 48, 71, 72, 95, 96, 153}.
- Cycles two indicator panels at rows 1-3 (cols 1-3 and cols 5-7) AND swaps
  the on-grid player sprite with a secondary 8-coloured "ghost" sprite that
  lives elsewhere on the playfield.  Trail painting also stamps the
  "previously active" colour into a 5x5 slot we cannot identify from a
  single frame.
- v1 leaves the grid unchanged (identity).  This correctly predicts the 8
  n_changed=0 cases out of 47.

## v1 result

```
{
  "game": "g50t",
  "split": "all",
  "n": 200,
  "errors": 0,
  "state_exact_pct": 39.5,
  "pixel_match_pct": 99.612060546875,
  "reward_acc_pct": 79.0,
  "done_acc_pct": 100.0,
  "by_action": {
    "2": {"n": 36, "exact_pct": 36.11, "pixel_pct": 99.87, "errors": 0},
    "5": {"n": 47, "exact_pct": 17.02, "pixel_pct": 98.67, "errors": 0},
    "1": {"n": 39, "exact_pct": 56.41, "pixel_pct": 99.97, "errors": 0},
    "3": {"n": 33, "exact_pct": 54.55, "pixel_pct": 99.94, "errors": 0},
    "4": {"n": 45, "exact_pct": 40.00, "pixel_pct": 99.84, "errors": 0}
  }
}
```

Pixel-match >99.6% confirms the sprite-move model is correct in shape;
most exact-mismatch cases differ by exactly the 1 counter-tick cell on
row 63.

## v2 plan

1. **Counter-tick parity**.  Track the *parity* of moves since the last
   tick by scanning row 63 itself: the rightmost-9 column tells us the
   tick budget but not the parity.  A purely stateless predictor cannot
   solve this, but if we permit one extra heuristic — "tick iff the move
   succeeds AND the player ends on an odd row-slot" or similar — we can
   probe for a deterministic rule.  Worth +25 to +35 points exact-match.

2. **Trail painting (n=72/73 cases)**.  When a "trail mode" is active
   (active colour != 9), the previous slot (or the slot two-back) gets
   stamped with the alternate colour.  Need to identify which slot from
   the indicator-panel state.  Worth +10 points.

3. **Action 5 swap**.  Implement the panel-cycle (9 -> 2 -> 1 -> 9 or
   similar) and the on-grid colour swap.  Difficult — likely needs
   stateful tracking.  +15 points on the action-5 bucket.

4. **Walls inside playfield (8s) acting as one-way doors**.  Two of the
   "blocked" cases that moved had `8` content; we already allow this.
   Confirm by simulating doors as path on traversal.
