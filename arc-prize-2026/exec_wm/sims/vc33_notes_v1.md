# vc33 sim v1 notes

## Game description (from observations)

Single-action game (action 6 / click). Each click advances a left-to-right
"fill" timer on row 0: the right portion of the row is color 7 (unfilled),
the left portion is color 4 (filled). Every click flips the rightmost 7
(or two of them on certain positions) to a 4. After 50 clicks the entire
row is 4 and the env auto-resets row 0 to all-7s (visible only at the
*next* state_t, not in our state_t1). The playfield below row 0 is
invariant under action 6 in 199/200 tuples. Reward is always 1, done is
always False, level stays 0. x,y of the click do not affect the outcome.

## Distribution

- 200 random-exploration tuples, all action_id == 6, all reward_class == 1.
- `n_changed` distribution: {1: 143, 2: 56, 265: 1}.
- The lone n_changed=265 outlier is step 146 — a one-off playfield
  redraw not modeled here.

## Invariants used (v1)

1. **Row-0 timer rule.** Only row 0 is mutated. The rightmost 7-cell
   (R7) always flips to 4. A second cell at R7-1 also flips iff
   R7 in DOUBLE_FLIP_R7 = {2,7,11,16,21,25,30,34,39,43,48,53,57,62}.
   This was verified deterministic across 199/199 simple tuples (each
   R7 value seen multiple times gives the same delta every time).
2. **Identity below row 0.** Rows 1-63 are untouched by action 6.
3. **Constant reward/done.** reward_class = 1, done = False, always.

## v1 result on full 200-tuple set

```json
{
  "game": "vc33",
  "split": "all",
  "n": 200,
  "errors": 0,
  "state_exact_pct": 99.5,
  "pixel_match_pct": 99.9677734375,
  "reward_acc_pct": 100.0,
  "done_acc_pct": 100.0,
  "by_action": {
    "6": {
      "n": 200,
      "exact_pct": 99.5,
      "pixel_pct": 99.9677734375,
      "errors": 0
    }
  }
}
```

## What v1 gets wrong (1 case, ~0.5%)

Step 146 only: a single click triggered 265 cell changes — row 0 advanced
normally (1 cell) but additionally 11 cells changed in the lower
playfield (rows 6,12,18,24 toggled 3->0; rows 34,40,45,47,51,57,63 had
the column-55 "ladder/score" cells flip between 0/3/11). Looks like a
periodic in-game animation/score-display refresh, not click-driven.
With only a single sample it's not safe to model.

## v2 plan (if pursued)

1. **Animation/score refresh.** Step 146 hints at a periodic redraw of
   the lower playfield that fires every ~150 clicks. With 200 samples
   we have only one example; collecting another full cycle (steps
   146..296) would let us check if the refresh recurs at step ~296
   and what the trigger is. Probably not worth pursuing — diminishing
   returns from 99.5 -> ~99.7.
2. **DOUBLE_FLIP_R7 closed form.** The set {2,7,11,16,21,25,30,34,...}
   has differences 5,4,5,5,4,5,4,5,4,5,5,4,5 — not a clean modular
   pattern. It looks like a 14/50 Bresenham-style rasterization of
   a constant 64/50 rate, but the explicit set already covers all
   observed R7 values, so no win.

## Honest assessment

The +99 points came from a single crisp invariant (row-0 timer with a
hardcoded double-flip set). The DOUBLE_FLIP_R7 set is derived from
observation — every member appeared 4 times in the 200-tuple sweep,
each consistent, so I trust it on unseen rollouts.
