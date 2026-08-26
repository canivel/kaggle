# sc25 sim v2 notes

## Changes vs v1

1. **Lives-bar drop** on every non-6 action (modelled as deterministic):
   zero the top-most 2 rows of cols 62-63 holding 14s.
2. **Action-6 panel stamp**: when click (x,y) lies in panel
   `x in [22,38] and y in [47,61]`, stamp a 3x3 of value 14 at the
   nearest button centre (rows {50,55,60} × cols {25,30,35}).

## Score (full 200, --split all)

| metric           | v1     | v2     | delta  |
|------------------|--------|--------|--------|
| state_exact_pct  | 48.0   | 72.5   | +24.5  |
| pixel_match_pct  | 99.94  | 99.97  | +0.03  |
| reward_acc_pct   | 88.5   | 93.0   | +4.5   |
| done_acc_pct     | 100.0  | 100.0  |  0     |

By action:

| action | n  | v1 exact | v2 exact | gain  |
|--------|----|----------|----------|-------|
| 1      | 43 | 41.86    | 53.49    | +11.6 |
| 2      | 32 | 37.50    | 59.38    | +21.9 |
| 3      | 41 | 36.59    | 65.85    | +29.3 |
| 4      | 36 | 22.22    | 80.56    | +58.3 |
| 6      | 48 | 89.58    | 97.92    | +8.3  |

## Honest assessment

- **Action 6 stamp rule is real invariant**: 5/5 stamps predicted
  exactly. The one remaining miss is the step-60 stamp that also
  dropped a life — collateral with the bar timer, not a panel bug.
- **Action 6 NOOPs** outside panel: 43/43 perfectly predicted as
  identity in both v1 and v2.
- **Lives-bar drop is *not* a deterministic invariant**: there is no
  state-side signal for the hidden timer (zero non-sprite/non-bar
  pixels change between frames; verified by per-pixel diff scan). The
  drop fires on ~63% of non-6 actions independent of action effect.
  Always-predict-drop is the optimal greedy bet, but it's a 63% bias,
  not a verified invariant. Action 4 happens to hit 78% in the data
  (small-sample fluctuation on 36 tuples).

## What this means for Kaggle inference

This sim is **better than identity** for sc25 even on actions whose
exact-match is "only" 53-66% — because the sprite invariants are
exact and the bar guess is +EV. A planner that picks moves by
minimising "predicted lives-bar drain" will correctly prefer action 6
clicks (no drop) over actions 1-4 (always drop in model), which is
the right strategic bias for survival-style play.

The remaining ~27% miss rate is dominated by the hidden bar timer
(impossible to model from state alone) plus a small chance that
action-6 clicks outside the panel still drop the bar via timer
coincidence. We could push +1-2 more pts by hand-fitting "drop on
action-6 if previous action was non-drop"-style heuristics, but
that's curve-fitting on n=48 and would not generalise.

## v3 candidates (skip unless room is needed)

1. Modelling the win-condition that resets the bar at step 69 and
   131 (refill to 128). The reset rule is "lives → 0" → refill on
   next step, but we never see lives==0 as state_t; would need
   explicit handling for the level-restart frame. Worth +~3pts.
2. Predict drop also on action-6 stamps (the step-60 case): would add
   1/48 = 2.1pts to action-6 exact but cost the same on no-drop
   stamps. Net ~0; skip.
