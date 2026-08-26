# s5i5 sim v2 notes

## What changed vs v1

v1 captured the row-63 step counter perfectly (200/200 transitions) and
left the 8 cases with n_changed in {10, 11} unmodeled, scoring
**96.0 % state_exact / 99.991 % pixel_match**.

v2 adds a controller-panel model that handles 7 of those 8 cases.

## Mechanic discovered

The board has two clickable controller panels and two output regions:

| Panel             | Bbox (r0,r1,c0,c1) | Controls output region                        |
| ----------------- | ------------------ | --------------------------------------------- |
| Bottom-left (BL)  | (35, 47, 21, 27)   | block-11 at rows 28-35, cols 9-11 (lit val 11)|
| Top-right (TR)    | (18, 23, 36, 48)   | block-14 at rows 9-11, cols 28-35 (lit val 14)|

### Trigger rule (verified 200/200)

A click toggles a 3x3 output sub-block iff **(a)** the click `(y, x)` is
inside one of the two panel bboxes **and** **(b)** the grid value at the
click cell is not the background `5`.

- v=4 clicks inside a panel: 6/6 toggle.
- v=2 click inside a panel: 1/1 toggle (idx=76).
- v=11 click inside a panel: 1/1 toggle (idx=13).
- v=11 click *outside* panels (inside output block 11): 1/1 no toggle (idx=78).
- v=14 click *outside* panels (inside output block 14): 1/1 no toggle (idx=99).
- v=5 clicks anywhere: 0/188 toggles.

### Output sub-block selection

- **TR panel** (4/4 verified): `x < 43` -> toggle (9-11, 30-32); `x >= 43` ->
  toggle (9-11, 33-35).
- **BL panel** (4/5 verified): toggle (33-35, 9-11) -- this is the modal
  target across the 5 observed BL clicks.  One edge case
  (`click(x=26, y=36)` -> (30-32, 9-11)) remains mispredicted.  Rather
  than build a 5-sample sub-rule, we predict the mode.

### Toggle direction

Each sub-block flips between background `5` and its lit value (`11` for
the BL output, `14` for the TR output).  Direction is decided by
sampling whether any cell currently equals the lit value.

## v2 result (active sim)

```json
{
  "game": "s5i5",
  "split": "all",
  "n": 200,
  "errors": 0,
  "state_exact_pct": 99.5,
  "pixel_match_pct": 99.997802734375,
  "reward_acc_pct": 100.0,
  "done_acc_pct": 100.0,
  "by_action": {
    "6": {
      "n": 200,
      "exact_pct": 99.5,
      "pixel_pct": 99.997802734375,
      "errors": 0
    }
  }
}
```

Only remaining mispredict: idx=157 (the BL edge case noted above).

## v1 vs v2

| Metric            | v1     | v2     | Delta  |
| ----------------- | ------ | ------ | ------ |
| state_exact_pct   | 96.0   | 99.5   | +3.5   |
| pixel_match_pct   | 99.991 | 99.998 | +0.007 |
| reward_acc_pct    | 100.0  | 100.0  | 0      |
| done_acc_pct      | 100.0  | 100.0  | 0      |

v2 improves exact match by 7 cases out of 200 (the BL edge case stays).

## Honest signal vs curve-fit

- Trigger rule: **strong signal** -- clean separation across 200 tuples
  with three independent indicators (cell value, panel bbox, output
  block).  Low overfit risk.
- TR sub-block selection by `x < 43`: derived from 4 samples but the
  split corresponds exactly to the visible panel structure (two
  side-by-side 3x3 templates with a value-3 divider at col 43).
  Likely correct.
- BL sub-block (always (33-35, 9-11)): the modal pick.  The single
  observed deviation (idx=157, click(x=26, y=36)) maps to a different
  sub-block; we don't have enough samples to tell whether it's a
  column-based, row-based, or controller-template-based rule.  Honest
  call: leave as modal.

## Could this sim help on Kaggle inference?

Yes -- s5i5 is now a usable executable world model for click planning:

1. Row 63 counter is fully deterministic, so any planner can use it to
   project N clicks ahead without exhausting the action budget.
2. Panel toggles are correctly attributed to (click, cell-value, panel)
   triples, so an MCTS/BFS using this sim can search for click
   sequences that turn the output regions on/off in a desired pattern.
3. Two known panels and two known output regions form a small,
   tractable game graph; the remaining 0.5 % miss (idx=157) is a single
   sub-block disambiguation that does not affect the macro game state.

This is not an identity-mostly sim -- it predicts non-trivial
transitions (panel-driven block toggles) correctly.

## v3 hooks (if more observations arrive)

1. Collect more BL-panel clicks at the (row 36, col 26) cell and
   neighbors to pin the sub-block rule.  Hypothesis: the 7-cell-wide
   panel has top-half (rows 36-40) and bottom-half (rows 42-46) sub-areas,
   each mapping to one of (28-30), (30-32), (33-35) of block-11.
2. Counter wrap behaviour: 200 tuples never exhaust row 63, so the
   wrap rule is still unobserved.  v3 should sample longer rollouts.
3. Confirm the panel border (value 2) clicks behave the same as
   interior clicks (idx=76 v=2 case is the only data point).
