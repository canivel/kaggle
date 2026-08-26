# bp35 sim v2 notes

## v2 result vs v1

| metric | v1 | v2 | delta |
|---|---|---|---|
| state_exact_pct | 34.0 | **69.0** | +35.0 |
| pixel_match_pct | 97.82 | **98.31** | +0.49 |
| reward_acc_pct | 100.0 | 100.0 | 0 |
| done_acc_pct | 100.0 | 100.0 | 0 |

Per-action exact %:

| action | v1 | v2 | delta |
|---|---|---|---|
| 3 (left) | 6.25 | **75.0** | +68.75 |
| 4 (right) | 1.96 | **74.51** | +72.55 |
| 6 (click) | 94.12 | 94.12 | 0 |
| 7 (toggle) | 32.0 | 32.0 | 0 |

## New observations

### Sprite anchor: color 11

Color 11 occurs ONLY inside the 5x5 sprite at rows 38-39, in EVERY observed
state (200/200 tuples). This makes sprite detection trivial:

- 2 cells of color 11 always live at (38, c) and (39, c) for some unique c.
- If c is at sprite_left + 1, the sprite is L-facing.
- If c is at sprite_left + 3, the sprite is R-facing.

### Sprite slots

The sprite occupies one of 7 fixed columns:
`sprite_left in {13, 19, 25, 31, 37, 43, 49}` — step = 6.

### Direction-forcing facing rule

- **Action 3** (move -6) always flips the sprite to L-facing.
- **Action 4** (move +6) always flips the sprite to R-facing.

Verified on 33/33 action-3 simple swaps and 37/37 action-4 simple swaps.

### NOOP rule for 3 and 4

- Action 3 NOOP iff `sprite_left == 13` (cannot go further left).
- Action 4 NOOP iff `sprite_left == 49` (cannot go further right).

### Sprite templates

```
R-facing (5x5):              L-facing (5x5):
 5  5  9  5  5                5  5  9  5  5
 5  9  9 11  5                5 11  9  9  5
 5  9  9 11  5                5 11  9  9  5
 5  5  9  5  5                5  5  9  5  5
10  5  5  5 10               10  5  5  5 10
```
L is R reflected horizontally.

## What v2 still gets wrong (~31% of cases)

### Action 7 simple swap (~26 cases, all unsolved)
Action 7 swap direction depends on hidden state. The empirical pattern:
- L-facing sprite + action 7 -> moves +6 R-facing (14/26)
- R-facing sprite + action 7 -> moves -6 (sometimes flips L, sometimes
  stays R) (11/26)
- L-facing + action 7 -> moves +6 stays L (1/26)

So "move opposite of current facing, flip facing" is ~70% correct on
action-7 swaps but the facing flip is unreliable. We left action 7 as
counter-only (32% exact = the 16 NOOPs).

### Score display refresh at rows 57-62 (~10-15% of cases)
Rows 57-62 contain a graphical score display that occasionally redraws
when the counter ticks. The refresh trigger is non-deterministic from a
single frame (likely depends on the *full* counter value including a
hidden "carry" register since the counter wraps at 64). Modelling this
would require tracking absolute step count.

### Large-n_changed clicks (3 cases)
Three action-6 clicks at specific (x,y) coords stamp a 5x5 sprite at the
click location (in playfield rows 1-5 and 30-36). The trigger condition
is unclear from 200 samples. Currently action 6 leaves the playfield
untouched.

## v3 plan (if pursued)

1. **Score display modeling**: extract the "digit panel" template at
   rows 57-62 and figure out the counter -> panel mapping. Likely worth
   +10 points exact-match.
2. **Action 7 with facing flip**: predict `move opposite of facing, flip
   facing` for action 7 simple cases. Risky — might worsen pixel-match
   on the noop cases since we'd start moving the sprite incorrectly.
3. **Click stamping**: ignore unless we collect more click data.

## Honest assessment

The +35 points exact-match came from two concrete invariants:
- Color 11 is a unique sprite anchor (lossless detection).
- Actions 3/4 deterministically force facing direction.

Both are crisp, low-risk rules that almost certainly hold beyond the 200
sample tuples. This is real signal, not curve-fitting.

The remaining 31% includes ~20% that probably requires tracking hidden
state (action 7, score panel) and ~10% complex playfield-state cases
(stamped clicks, multi-sprite layouts). These are diminishing returns —
v3 would need more observation data, not more analysis of these 200
tuples.
