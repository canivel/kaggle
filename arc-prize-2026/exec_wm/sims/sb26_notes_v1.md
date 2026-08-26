# sb26 sim v1 notes

## Game description

A clean, minimal counter-style game. Row 53 starts as all-2s and acts as a
right-to-left fill bar: each successful "tick" turns the rightmost 2 into a 3.
Actions 6 and 7 are inert (cosmetic / no-op) in random play; they neither
change the grid nor produce reward. Action 5 always advances the counter
by one cell and yields reward_class=1. No level transitions observed.

## Top-level observations (200 random tuples)

- Available actions: {5, 6, 7}. Distribution: 5=70, 6=72, 7=58.
- reward_class: 1 iff action_id == 5 (70/70); else 0 (130/130).
- done: False always (200/200). Level stays at 0.
- n_changed per action:
  - action 5: ALWAYS 1 cell changes.
  - action 6: ALWAYS 0 cells change.
  - action 7: ALWAYS 0 cells change.

## The action-5 rule (verified 70/70)

For every action-5 tuple, the exact diff is a single pixel:

- row index = 53
- column index = rightmost column on row 53 that currently holds value 2
- value transition: 2 -> 3

(x, y) is always (0, 0) for action 5; the player coords are unused.

Row 53 begins all-2s at step 0. By the last tuple in the trace, columns
58..63 are 3s, the rest 2s — i.e. the fill grows right-to-left in lockstep
with action-5 invocations.

## Invariants used

1. `reward_class = 1` iff `action_id == 5`, else 0.
2. `done = False` always.
3. Action 5 mutates exactly one pixel: `(row=53, col=rightmost-2)` flips
   2 -> 3. If row 53 has no 2s left (counter full), we leave the grid
   unchanged — we have no observation of wrap behaviour, so identity is
   the safe default.
4. Actions 6 and 7 are identity on the grid.

## v1 result on full 200-tuple set

```json
{
  "game": "sb26",
  "split": "all",
  "n": 200,
  "errors": 0,
  "state_exact_pct": 100.0,
  "pixel_match_pct": 100.0,
  "reward_acc_pct": 100.0,
  "done_acc_pct": 100.0,
  "by_action": {
    "5": {"n": 70, "exact_pct": 100.0, "pixel_pct": 100.0, "errors": 0},
    "7": {"n": 58, "exact_pct": 100.0, "pixel_pct": 100.0, "errors": 0},
    "6": {"n": 72, "exact_pct": 100.0, "pixel_pct": 100.0, "errors": 0}
  }
}
```

## v2 plan

There is no headroom on this 200-tuple split — all four metrics are 100%.
Next-iteration work should focus on **regimes the sample didn't cover**,
not on the present trace:

1. **Counter overflow / wrap**: row 53 still had ~60% 2s at the end of the
   trace. We never observed the moment when the last 2 becomes a 3. We
   left that case as identity; if the real game wraps row 53 back to 2s
   (or advances a level), v1 will mispredict. Worth collecting a longer
   exploration log.
2. **Level-up trigger**: `done` and `level` never moved. The fill bar is
   strongly suggestive of a "fill to win" mechanic; the level-up event is
   probably triggered by completing row 53. We should pre-emptively model
   reward_class=2 + done=True when the rightmost-2 was at column 0 prior
   to the action-5 tick.
3. **Action 6 click locations**: 72 clicks at varied (x, y) never affected
   the grid. Possible the game only listens for clicks in a specific
   region (e.g. a "go" button) that random play never hit. Not worth
   modeling without targeted observations.
4. **Action 7**: 58/58 no-ops. Likely a NOOP/cancel action; if the engine
   ever spends a real frame on action 7 (e.g. animated cursor), we'd want
   to detect that. Currently identity is correct.
