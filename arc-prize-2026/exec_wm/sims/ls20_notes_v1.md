# ls20 sim v1 notes

## Game description

ls20 is a top-down maze/road game with a 5x5 sprite (top 2 rows color 12,
bottom 3 rows color 9) that moves in 5-cell steps inside a vertical
corridor (cols 34..38) and adjacent lanes. 4 actions = cardinal moves.
A step counter at rows 61-62 decrements one column-pair (11,11 -> 3,3)
per action; when fully depleted the counter refills and one of three
"score" 8-pairs in the right margin (cols 56-57, 59-60, 62-63) is
consumed (3 refill events in 200 tuples; not modelled in v1).

## Invariants (verified)

1. **Color 12 is a unique sprite anchor.** Across all 200 observed
   states, color 12 appears in exactly 10 cells forming a 2x5 block at
   the top of the sprite. `(min_row, min_col)` gives the sprite
   top-left.

2. **Movement is 5-cell cardinal, NOOP iff destination hits wall (4).**
   Verified 197/200: actions 1/2/3/4 translate the sprite by
   (-5,0)/(+5,0)/(0,-5)/(0,+5). The move is rejected iff the destination
   5x5 patch contains any color-4 cell. Other colors in the destination
   (e.g. 5 = lane line) are overwritten by the sprite.

3. **Counter tick: leftmost (61,c)=(62,c)=11 -> (3,3).** 196/196 cases
   where the counter was non-empty.

4. **Lane-line restoration: when sprite vacates a row, restore the
   per-row background color by sampling the cell immediately to the
   left or right of the sprite (i.e. col `sc-1` or `sc+5`) on the same
   row.** Rows 10 and 15 have static color-5 lane lines that show
   through the cleared sprite area. This per-row restoration converted
   3 mismatches into matches vs. the naive "erase to 3" rule.

## Validation result

```json
{
  "game": "ls20",
  "n": 200,
  "errors": 0,
  "state_exact_pct": 98.0,
  "pixel_match_pct": 99.948974609375,
  "reward_acc_pct": 100.0,
  "done_acc_pct": 100.0,
  "by_action": {
    "1": {"n": 50, "exact_pct": 94.0,  "pixel_pct": 99.86},
    "2": {"n": 49, "exact_pct": 100.0, "pixel_pct": 100.0},
    "3": {"n": 45, "exact_pct": 100.0, "pixel_pct": 100.0},
    "4": {"n": 56, "exact_pct": 98.21, "pixel_pct": 99.94}
  }
}
```

## What v1 misses (4/200 = 2%)

All 4 misses are the special "counter empty" events:
- step 42 (action 1, n_changed=138): counter refills, score-pair at
  cols 62-63 consumed.
- step 85 (action 4, n_changed=138): same shape, score pair consumed.
- step 171 (action 1, n_changed=138): same.
- step 128 (action 1, n_changed=4): score pair at cols 56-57 consumed,
  but counter does NOT refill (only one pair remained).

## v2 plan

1. **Model the score-decrement / counter-refill rule.**
   - When counter is fully depleted, look at row 61 cols 55..63 for the
     rightmost remaining 8-pair. Flip it to 3.
   - If at least one 8-pair remains after consumption, also refill the
     counter (cols 13..54 of rows 61-62 -> 11).
   - This single rule should pick up all 4 remaining cases (+2% to
     state-exact).
2. **Optional: pre-compute a static lane mask** from the initial state
   (rows where col-36 is 5 outside sprite). Cheaper than the
   sample-neighbour heuristic and slightly more robust if a future
   level introduces lane lines on rows we haven't seen yet.
3. **Reward / done are constant.** No further work needed there.

## Honest assessment

98% state-exact in one iteration on a game with 12-color palette and a
clear sprite-color uniqueness anchor. The remaining 2% are the
counter-empty endgame events; modelling them is low-risk (the trigger
condition is observable from the input state alone -- "counter all 3s").
