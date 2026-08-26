# m0r0 sim v1 notes

## Game description

A symmetric two-cursor navigation game on a 64x64 grid. The static
background uses {5, 11, 12}: cols 0-31 are 11 in non-playfield rows, cols
32-63 are 12, and the central playfield channel (irregular shape, mirror-
symmetric around col 31) is colour 5. A cursor consists of TWO 5x5 blocks
of colour 10 that slide on a 5-step lattice inside the playfield channel.

## Invariants used

1. **Wall colours are static**: cells with value 11 or 12 are constant
   across all 200 observation tuples (verified). They define the
   non-traversable region.
2. **Cursor = two 5x5 blocks of colour 10**: every observed state has
   exactly two 5x5 solid blocks of 10 (verified 200/200). I call them
   `tl` (smaller col) and `br` (larger col).
3. **Per-block 5-step motion**:
   - A1 (UP):    each block tries (-5, 0)
   - A2 (DOWN):  each block tries (+5, 0)
   - A3 (EXPAND-X): tl tries (0, -5), br tries (0, +5)
   - A4 (CONTRACT-X): tl tries (0, +5), br tries (0, -5)
   A block moves only if its destination 5x5 region contains no wall
   cells (11 or 12) and is in bounds. Verified **131/131 (100%)** on
   all movement actions.
4. **Reward = changed-flag**: reward_class == 1 iff next_state !=
   state (verified 200/200). done always False, level always 0.

## What v1 cannot model

The grid carries a step counter on row 0 (filling right-to-left with 0s)
and row 63 (filling left-to-right, mirror-symmetric). The counter ticks
on roughly half of all actions but the trigger is non-deterministic from
a single frame (hidden state). I leave the counter rows unchanged.

A5 / A6 are pure counter ticks with no spatial effect. Predicting "no
change" is correct for the ~54% of A5/A6 cases where the counter happened
not to tick.

## validate_sim.py output

```
{
  "game": "m0r0",
  "split": "all",
  "n": 200,
  "errors": 0,
  "state_exact_pct": 57.5,
  "pixel_match_pct": 99.979,
  "reward_acc_pct": 78.0,
  "done_acc_pct": 100.0,
  "by_action": {
    "1": {"n": 37, "exact_pct": 64.86, "pixel_pct": 99.983},
    "2": {"n": 27, "exact_pct": 59.26, "pixel_pct": 99.980},
    "3": {"n": 33, "exact_pct": 66.67, "pixel_pct": 99.984},
    "4": {"n": 34, "exact_pct": 47.06, "pixel_pct": 99.974},
    "5": {"n": 35, "exact_pct": 60.00, "pixel_pct": 99.980},
    "6": {"n": 34, "exact_pct": 47.06, "pixel_pct": 99.974}
  }
}
```

## v2 plan

1. **Reward correlate**: reward_class is determined by whether the GROUND
   TRUTH changed, not whether OUR prediction changed. We predict
   reward_class via our own change-flag, which is 78% right. To improve
   this we would have to predict counter ticks too — but those are
   hidden-state driven. Probably skip.
2. **Counter modelling attempt**: try learning a Bernoulli rate per
   action (~45% tick). Naive Bernoulli prediction can't improve
   state_exact, only worsens it on the no-tick majority. Not worth it.
3. **Cleaner: investigate if there is a 2nd-order pattern** (e.g., did
   the cursor move? did the previous action tick?) that conditions the
   counter. From the 200 tuples there is no obvious signal, but with
   stateful play we could keep an internal counter — out of scope for
   this stateless `simulate` API.
4. The +35-point jump on bp35 came from a hidden symmetry (colour 11
   anchor). Here, the analogue would be a way to predict the counter
   tick. Without seeing two consecutive frames of context, this seems
   genuinely hidden.

## Honest assessment

The movement model is **fully correct** on every observed action 1-4
transition. The remaining 42.5% loss is entirely the counter-tick
non-determinism, which is irreducible from a single state frame. The
pixel match (99.98%) confirms the model only ever misses the two
counter cells.
