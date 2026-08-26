# s5i5 sim v1 notes

## Game description

A click-only game (single action_id=6). 200 random-exploration tuples,
all with `reward_class=1`, `done=False`, `level=0`. Grid values in 2-14
with 5 as the dominant background. The visible state contains a small
object near the top (rows 9-11, cols 30-35), another near rows 30-35,
cols 9-11, and a row-63 step counter that ticks on every click.

## Observations

### Action / reward / level distribution
- actions: {6: 200} (clicks only)
- rewards: {1: 200}
- dones: {False: 200}
- levels: {0: 200}

### n_changed buckets
- 1 cell: 138
- 2 cells: 54
- 10 cells: 6 (9-cell block + 1 row-63 tick)
- 11 cells: 2 (9-cell block + 2 row-63 ticks)

### Row 63 = step counter (the dominant invariant)

Row 63 starts as all 3s. Every click flips the **rightmost** remaining
3-cell to 4. Some columns come in "digit pairs" -- when the rightmost
3-cell lands on a pair column, the buddy cell also flips to 4 in the
same step, producing n_changed = 2.

Fixed pair columns (verified across 200/200 tuples):
```
(1,2) (6,7) (10,11) (15,16) (20,21) (24,25) (29,30)
(33,34) (38,39) (42,43) (47,48) (52,53) (56,57) (61,62)
```

Rule: `cols_to_flip = [rightmost_3] + ([buddy] if rightmost_3 in PAIRS and buddy still 3 else [])`
Verified on **200/200** transitions.

### 3x3 block toggles (8/200, unmodelled)
A small number of clicks toggle a 3x3 block between value 5 and
{11 or 14}. Locations seen: (rows 30-35, cols 9-11) and (rows 9-11,
cols 30-35). Click coords do not align with either block; the trigger
appears to depend on hidden state (perhaps an internal "armed" flag).
Cost of leaving unmodelled: at most 9 pixels per case = 0.22 %.

## Invariants used
1. Row 63 rightmost-3-flips-to-4 rule with fixed buddy table.
2. reward_class = 1 always.
3. done = False always.

## v1 result on full 200-tuple set

```json
{
  "game": "s5i5",
  "split": "all",
  "n": 200,
  "errors": 0,
  "state_exact_pct": 96.0,
  "pixel_match_pct": 99.9912109375,
  "reward_acc_pct": 100.0,
  "done_acc_pct": 100.0,
  "by_action": {
    "6": {
      "n": 200,
      "exact_pct": 96.0,
      "pixel_pct": 99.9912109375,
      "errors": 0
    }
  }
}
```

The 8 misses are exactly the n_changed in {10, 11} cases (3x3 block
toggles). All n_changed in {1, 2} transitions are predicted exactly.

## v2 plan

1. **3x3 block toggles**: collect more samples (current 8/200 is sparse)
   and look for a trigger. Hypotheses to test:
   - Each click at any (x,y) "arms" a counter; the toggle fires on the
     N-th click after row-63 hits a specific position.
   - Click (x,y) modulo some periodic mask matches the block region.
   - The toggle alternates per fixed counter value (e.g., every 32 ticks).
2. **Confirm counter wrap behaviour**: 200 tuples never exhaust row 63
   (which has 64 cells). Get the wrap rule before believing it generalises.
3. **Other rows**: a second counter-like region may exist outside row 63.
   Sample rows 60-62 across all 200 tuples to confirm they really are
   static.

## Honest assessment

The +96 % comes from one crisp lossless invariant (row-63 rightmost-3
+ fixed pair table). Very low risk; the rule was verified on every
single observed transition. The remaining 4 % requires either more
observations or a hidden-state model -- not more pattern mining of
these 200 tuples.
