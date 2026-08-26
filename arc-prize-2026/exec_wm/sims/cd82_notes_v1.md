# cd82 sim v1 notes

## Game in one line

A 64x64 board with a "2"-bordered rectangle that rotates between an
axis-aligned and a diagonal pose; actions 1-4 trigger ~200-cell rotations,
action 5 paints 0-cells to 15s, action 6 is a click. Row 63 is a 64-cell
counter that fills 4 -> 5 from RIGHT to LEFT each engine tick. Hidden state
controls whether any given action ticks (60% do).

## Observed action / reward distribution (200 tuples)

| action | n  | NOOPs (n_changed=0) | tick-only (n=1) | big rotation (n=200/201) | medium paint |
|--------|----|---------------------|-----------------|--------------------------|--------------|
|   1    | 37 |   7                 | 16              | 2 + 12 = 14              |    -         |
|   2    | 27 |   4                 |  5              | 7 + 11 = 18              |    -         |
|   3    | 33 |   5                 |  7              | 8 + 13 = 21              |    -         |
|   4    | 34 |   8                 |  6              | 7 + 13 = 20              |    -         |
|   5    | 35 |  10                 | 18              |    -                     | 7 (11..55)   |
|   6    | 34 |  12                 | 22              |    -                     |    -         |

- `reward_class == 1`  iff  `n_changed > 0`. Verified on all 200.
- `done == False` everywhere; `level == 0` everywhere.

## Invariants used in v1

1. **Counter row (row 63)**: cells start as `4`, each engine tick flips
   the rightmost remaining `4` to `5`. Verified on all 74 `n=1` tuples
   (zero exceptions) AND on the 49 `n=201` tuples (whose extra change
   is always one cell on row 63).
2. **Reward**: predict `1` unconditionally. Matches 154/200 = 77 %.
3. **Done**: predict `False`. Matches 200/200 = 100 %.

## What v1 deliberately does NOT model

- **Big rotations (n=200/201)**: actions 1-4 cycle a 2-bordered rectangle
  between axis-aligned (`rows 24-32, cols 25-39`-style) and diagonal
  poses. Direction of rotation per action and which pose is "next" both
  depend on hidden orientation state. A wrong rotation would corrupt 200
  cells and crater pixel-match, so v1 leaves the playfield untouched.
- **Action 5 paint**: paints a sub-region of `0`-cells to `15`. The
  target sub-region appears to depend on hidden cursor/level state
  (different runs paint different rectangular bands in rows 34-63).
- **Action 6 click**: 22/34 ticked the counter, 12/34 NOOPed; click
  coordinates do not correlate with `state[y,x]` in any obvious way.
- **Per-action tick determinism**: the tick rate is ~60% across every
  action, with no observable trigger from a single frame.

## v1 result on full 200 tuples

```json
{
  "state_exact_pct": 37.0,
  "pixel_match_pct": 98.1845703125,
  "reward_acc_pct": 77.0,
  "done_acc_pct": 100.0,
  "by_action": {
    "1": {"n": 37, "exact_pct": 43.24, "pixel_pct": 98.15},
    "2": {"n": 27, "exact_pct": 18.52, "pixel_pct": 96.73},
    "3": {"n": 33, "exact_pct": 21.21, "pixel_pct": 96.88},
    "4": {"n": 34, "exact_pct": 17.65, "pixel_pct": 97.12},
    "5": {"n": 35, "exact_pct": 51.43, "pixel_pct": 99.85},
    "6": {"n": 34, "exact_pct": 64.71, "pixel_pct": 99.99}
  }
}
```

Exact-match comes from the 74 `n_changed=1` cases (counter-tick-only)
that v1 nails. Pixel match is ~98 % because the 200-cell rotations only
touch ~5 % of the grid.

## v2 plan

1. **Encode the rotation deterministically.** Diagonal pose seems to be
   a fixed shape sweeping a fixed offset; the axis-aligned pose lives at
   a fixed `(rows 24-32, cols 25-39)`-style slot. Build templates for
   both and toggle between them when the playfield clearly contains
   either. This should solve 200-cell cases when current pose is
   detectable.
2. **Predict tick deterministically?** Look for a hidden "tick clock"
   signal -- maybe the 5/15 internal pattern position. If we cannot find
   one, leave tick as unconditional (current best EV).
3. **Action 5 paint target.** Inspect more carefully whether the
   sub-region is anchored by an object in the playfield -- the "5/15"
   ornament at row 8 may indicate the target paint band.
4. **NOOP detector.** If we can detect a deterministic NOOP signature
   (e.g. row 63 is fully ticked already), suppress the tick to recover
   reward and counter exactness on those cases.

## Honest assessment

v1 is the conservative "tick + identity" baseline. The 37 % state-exact
floor comes entirely from row-63 counter logic, which is rock-solid. The
+~30 % we'd need to match bp35's 69 % is locked behind the rotation
encoding, which I deliberately left out because a wrong rotation costs
200 cells * 73 cases = ~5 % pixel match. We should attempt rotation in
v2 only after observing several consecutive object poses to derive the
deterministic toggle rule.
