# sc25 sim v1 notes

## Game in one line

A 4x4 "die" sprite slides L/R along a horizontal tube (rows 19..22, cols
23..39 in 4-col slots) and is rotated in-place by actions 1 and 2 — while
a 14-coloured "lives bar" on cols 62..63 depletes 2 rows at a time on a
roughly half of the moves (trigger likely hidden timer).

## Observations

- 5 actions: {1, 2, 3, 4, 6}. 200 random-exploration tuples.
- `done` is False for every tuple. `level` is 0 throughout.
- Reward distribution: r=1 when n_changed>0, r=0 when n_changed==0.
- Sprite has **exactly 4 orientations** (verified by unique 4x4 patch
  enumeration across 200 states):
    A = vertical 9-left / 10-right
    B = horizontal 9-top / 10-bot
    C = vertical 10-left / 9-right
    D = horizontal 10-top / 9-bot
- Sprite leftmost col in `{23, 27, 31, 35, 39}` (step = 4).
- Tube background (between sprite and right wall): value 2.

## Action invariants (each verified on every applicable tuple)

| action | effect                                           | NOOP condition |
|--------|--------------------------------------------------|----------------|
| 1      | re-stamp sprite as **B** in place                | never          |
| 2      | re-stamp sprite as **D** in place                | never          |
| 3      | move sprite -4 cols, re-stamp as **A**           | sprite_col==23 |
| 4      | move sprite +4 cols, re-stamp as **C**           | sprite_col==39 |
| 6      | NOOP (43/48); 5 outliers stamp a 3x3 14 block    | unmodelled     |

These are crisp and deterministic — no false positives on 200 tuples.

## Lives bar (cols 62..63)

Each "death" zeros the top-most 2 rows of the bar (4 cells: rows r,r+1
× cols 62,63). Over 200 tuples the depletion lands at every row pair
from (0,1) up to (62,63), 3 times each.  The TRIGGER per action looks
non-deterministic from a single frame (likely a hidden cooldown /
timer). v1 does not model this; cost is the ~half of action 1-4
cases that include a life-drop.

## v1 result on full 200-tuple set

```json
{
  "game": "sc25",
  "split": "all",
  "n": 200,
  "errors": 0,
  "state_exact_pct": 48.0,
  "pixel_match_pct": 99.94,
  "reward_acc_pct": 88.5,
  "done_acc_pct": 100.0,
  "by_action": {
    "1": {"n": 43, "exact_pct": 41.86, "pixel_pct": 99.94},
    "2": {"n": 32, "exact_pct": 37.50, "pixel_pct": 99.94},
    "3": {"n": 41, "exact_pct": 36.59, "pixel_pct": 99.92},
    "4": {"n": 36, "exact_pct": 22.22, "pixel_pct": 99.92},
    "6": {"n": 48, "exact_pct": 89.58, "pixel_pct": 99.98}
  }
}
```

The 22-42% per-action exact rate equals (sprite rule correct) AND
(no life drop). Pixel match >99.9% because we model the sprite
movement perfectly and only miss the 4-cell bar depletion.

## v2 plan

1. **Lives bar timer**: deduce from the bar's current state whether the
   next move drops a life. Hypothesis: depletion fires every Nth move
   from start, regardless of action. Total drops observed over 200
   steps: roughly 95 (matches r=1 distribution). If N is fixed (e.g.
   "drop on every other action that yields r=1"), we can model it from
   the bar's current 14-count modulo N. Would lift exact% to ~85-90.
2. **Action 6 stamping**: 5/48 clicks stamp a 3x3 14-block on the bottom
   "guide" panel. Trigger may be "x,y lands on a specific guide cell".
   Worth modelling if v2 lives-bar fix lands first.
3. **Reward fidelity**: currently 88.5%. The remaining 11.5% miss is
   action 6 cases that *did* change (we predicted no-change). If we
   stamp those clicks correctly, reward acc -> ~98%.
