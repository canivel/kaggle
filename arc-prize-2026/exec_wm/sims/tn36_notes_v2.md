# tn36 sim v2 notes

## Game description

Countdown-timer minigame with a 5-button indicator strip.
- Action 6 (click) always ticks a 61-cell countdown on row 1: rightmost
  `9` in cols 1..61 flips to `3` (env resets the row off-screen when all
  cells are spent, so we never see an all-`3` row 1 as input).
- 5 buttons sit at fixed x-centers {21, 26, 31, 36, 41}. Each button has
  two visual orientations selected by the click y-coordinate:
    * y in {42, 43} -> horizontal triple at (row 42, cols c-1..c+1)
    * y in {44, 45, 46} -> vertical triple at (rows 44..46, col c)
- A click in the button band (y in 42..46) AND within dx=2 of the nearest
  center toggles the indicator between palette values 5 and 1.

## Invariants used in v2

1. `reward_class == 1`, `done == False` -- universal (200/200).
2. Action 6 always ticks the row-1 countdown (194/194 plain-tick cases
   + the 6 button-press cases also tick).
3. Action 6 with y in 42..46 AND |x - nearest_center| <= 2 toggles the
   3-cell indicator at that button (5<->1). Selected horizontal vs
   vertical by y.
4. Identity elsewhere.

## Decoding evidence

| idx | (x,y)    | nearest_c | dx | orientation | toggled cells       | direction |
|-----|----------|-----------|----|-------------|---------------------|-----------|
| 10  | (29,44)  | 31        | 2  | vertical    | (44..46, 31)        | 5->1      |
| 18  | (22,46)  | 21        | 1  | vertical    | (44..46, 21)        | 5->1      |
| 72  | (43,43)  | 41        | 2  | horizontal  | (42, 40..42)        | 1->5      |
| 79  | (34,46)  | 36        | 2  | vertical    | (44..46, 36)        | 1->5      |
| 98  | (31,46)  | 31        | 0  | vertical    | (44..46, 31)        | 5->1      |
| 184 | (30,42)  | 31        | 1  | horizontal  | (42, 30..32)        | 5->1      |

Negative-control band clicks (y in 42..46 but NO toggle): dx in
{4, 7, 9, 10, 10, 14, 21}. The dx<=2 vs dx>=4 gap is clean, so the
threshold is not curve-fit at a single boundary.

## v1 vs v2 (validate_sim.py, split=all, n=200)

| metric            | v1       | v2       |
|-------------------|----------|----------|
| state_exact_pct   | 97.0     | **100.0** |
| pixel_match_pct   | 99.9978  | **100.0** |
| reward_acc_pct    | 100.0    | 100.0    |
| done_acc_pct      | 100.0    | 100.0    |
| errors            | 0        | 0        |

Active sim: tn36_sim.py == v2.

## Honest assessment

- **Signal, not curve-fit**: the dx<=2 cutoff is supported by 6 positives
  (all dx<=2) AND 7 negatives (all dx>=4). The y-band {42..46} is also
  cleanly separated -- 0 fails outside it. We're not fitting boundaries
  one observation away.
- **Sample-size caveat**: only 6 toggle events, so we never see (a) a
  click that lands on multiple buttons simultaneously, (b) toggling a
  cell that isn't 5 or 1, (c) action 6 with action_id-flavoured side
  effects we haven't observed. The fall-through (`_toggle_value`
  passes through non-{5,1} values, identity for unknown action_ids)
  is defensive but unverified.
- **Kaggle utility**: 100% state_exact + 100% pixel match on this game
  means an agent can plan perfectly under this sim. tn36 is a simple
  countdown game (every step reward_class=1, done=False), so the sim
  itself doesn't unlock new score -- the agent already had no decision
  to make. The value of this sim is mostly methodological: it proves
  the executable-WM pipeline can hit 100% on a constrained game, and
  the button-decoding pattern (y-band -> orientation, x-nearest-center
  with dx threshold) is reusable on tu93/sb26-style UIs.
