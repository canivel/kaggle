# sk48 sim v1 notes

## Game description (from observations)

64x64 grid. Background is 5 above row 53 and 4 below row 53. A static
"cursor" frame of color 6 sits at rows 36-41, cols 11-16, containing a
checkered 2x6 "piece" pattern (values 1, 2). The playfield (rows 12-41,
cols 17-46) is filled with background 4 plus an assortment of 2x6
checkered pieces in colors {1, 2, 8, 9, 14}. Row 53 is a long progress
bar of color 2 cells that flip to 3 one-at-a-time as the game advances.
Rows 56-62 contain a small "score box" (a 6x6 frame with color 6/0
inside). Actions {1, 2, 3, 4, 7} can stamp / erase / move 2x6 pieces in
the playfield, and tick the progress bar. Action 6 (click) is always a
no-op. Done is False and level is 0 in all 200 tuples.

## Distributions

- `available_actions`: {1, 2, 3, 4, 6, 7}.
- `reward_class`: 0 iff `n_changed == 0` (54 cases); 1 otherwise (146).
- Action 6: 41/41 cases have `n_changed == 0` (always identity).
- Action 7: 34/34 cases have `n_changed > 0` (always changes).
- Action 1, 2, 3, 4 mostly change state but sometimes don't.
- `done` and `level` are constant.

## Per-action `n_changed` buckets

| action | 0 | small (<=5) | mid (12-13, 36-37) | big (>=52) |
|---|---|---|---|---|
| 1 | 4 | 4 | 0 | 26 (72..241) |
| 2 | 2 | 0 | 0 | 24 (72..209) |
| 3 | 3 | 4 | 16 (12-13: 9, 36-37: 12) | 3 (52..53) |
| 4 | 4 | 3 | 24 (12-13: 13, 36-37: 11) | 3 (52) |
| 6 | 41 | 0 | 0 | 0 |
| 7 | 0 | 1 | 16 (12: 5, 36: 10, 52: 1) | 17 (72..208) |

Smallest interesting bucket is `n_changed == 12`: a 2x6 piece is either
erased (action 3, e.g. step 127: (14,23-28) values 2,1,1,2,1,1 / 1,1,2,1,1,2 -> 4)
or stamped (action 4, e.g. step 3: (32,23-28) 4 -> 2,1,1,2,1,1 / 1,1,2,1,1,2).
`n_changed == 13` is the same erase/stamp PLUS one row-53 tick.

## Invariants used in v1 (3)

1. `done` is always False; `level` is always 0.
2. Action 6 is always a no-op (41/41 across all observed states).
3. `reward_class = 1` iff state changes; `reward_class = 0` iff it doesn't.

## Strategy

- Action 6 -> identity, reward 0.
- Actions 1, 2, 3, 4, 7 -> identity, reward 1.
  Why identity: the piece stamp/erase column and the row-53 tick column
  both depend on hidden state (cursor position, step counter) we cannot
  recover from a single frame.  Guessing wrong would damage pixel-match
  more than identity does.

## v1 result (200 tuples, full set)

```json
{
  "game": "sk48",
  "split": "all",
  "n": 200,
  "errors": 0,
  "state_exact_pct": 27.0,
  "pixel_match_pct": 98.61962890625,
  "reward_acc_pct": 93.5,
  "done_acc_pct": 100.0,
  "by_action": {
    "1": {"n": 34, "exact_pct": 11.76, "pixel_pct": 97.46},
    "2": {"n": 26, "exact_pct":  7.69, "pixel_pct": 96.91},
    "3": {"n": 31, "exact_pct":  9.68, "pixel_pct": 99.43},
    "4": {"n": 34, "exact_pct": 11.76, "pixel_pct": 99.48},
    "6": {"n": 41, "exact_pct": 100.0, "pixel_pct": 100.0},
    "7": {"n": 34, "exact_pct":  0.0,  "pixel_pct": 97.83}
  }
}
```

state_exact = 54 zero-change cases (all but action 7 has some) + 0 from
the 146 changing cases. reward_acc: action 6 perfect (41), actions 1-4,7
get 28+24+28+30+34 = 144 correct (lost 13 due to identity guess on
zero-change non-6 tuples) so 144 + 41 = 185...  Actually printed 93.5%
= 187, so the 4 + 2 + 3 + 4 = 13 zero-change tuples in actions 1/2/3/4
each cost 1, and the no-zero-change action 7 costs 0.  Expected = 187,
matches.

## What v2 should try

The big lever is the n_changed=12 / 13 bucket (24 cases total for actions
3 and 4 -- and the 12/36 cases mid bucket for action 7 too -- about 40
total tuples).  For these:

1. **Erase 2x6 piece (action 3 / action 7)**: find a 2x6 piece in the
   playfield whose interior is the canonical checker pattern
   `[2,1,1,2,1,1; 1,1,2,1,1,2]` (or with any of the "piece-colour" sets
   like {1,2}, {8,9}, {14,4}).  Erasing means setting it to background
   4.  We need to pick WHICH piece to erase -- it might be the one the
   cursor (rows 36-41, cols 11-16) "points to" or the closest piece.
2. **Stamp 2x6 piece (action 4)**: write the canonical checker pattern
   at the cursor-targeted 2x6 slot.  Slot grid appears to be on rows
   {14, 20, 26, 32, 38} x cols {17, 23, 29, 35, 41} approximately.
3. **Row-53 progress tick**: when we predict any change, also flip the
   leftmost remaining 2-cell on row 53 to 3 (analogous to bp35's
   row-63 counter).  But the column in observed diffs doesn't follow
   leftmost-first -- it's scattered (cols 24, 27, 33, 36, 38, 40, 41,
   42, 43, 44, 55, 56, 58, 59, 60, 63...).  So it likely tracks a
   counter we can't recover from one frame.  Skip for v2.
4. Decode the cursor: there's a 6-framed 5x6 box at (36-41, 11-16)
   whose **interior** changes between (37,12..15)=0 and (38,12..15)
   showing the current piece-pattern.  This might be the "selected
   piece" that action 4 stamps and action 3 picks up.  Worth a longer
   look at the raw observations.

Honest assessment: v1's 27% is the floor (zero-change matches).  v2
would need to model the cursor + piece slot grid to push state_exact
into the 40-60% range.  The 6 large-n_changed buckets (>=52 -- 70+
tuples across actions 1, 2, 7) are board-restructuring events that
likely require multi-frame state to reverse-engineer.
