# sp80 sim v1 notes

## One-line description
A paddle-mover: a 4x20 block of color 9 slides on a 4-cell grid; every
action also drains a fuel bar on row 0.

## Observations

### Top-level
- 6 available actions: {1, 2, 3, 4, 5, 6}. No (x, y)-sensitive behaviour
  observed for action 6 (clicks don't affect the playfield).
- 200 random-exploration tuples. `reward_class == 1` and `done == False`
  on every single tuple.

### The fuel bar (row 0)
- Initially all 64 cells are value 14. Each action turns the K rightmost
  remaining 14s into 0.
- K is normally 2, but exactly 3 when the pre-action count of 14s is in
  {58, 41, 26, 9}. Verified on 200/200 tuples (zero mismatches).
- When the bar is empty/near-empty, the *next* observation's state_t shows
  the bar re-initialised to 64 cells of 14. This reset happens *between*
  episodes/level-restarts, so the sim does not have to predict it from a
  single transition.

### The 9-paddle (4x20 block of color 9)
- Color 9 appears ONLY inside this solid 4x20 rectangle in every observed
  state (200/200). Detection by `np.where(grid == 9)` is lossless.
- The paddle lives on a 4-cell movement grid:
  - r0 ∈ {12, 16, 20, 24} (step = 4)
  - c0 ∈ {0, 4, 8, ..., 36} (step = 4)
- Action effects on the paddle:

  | action | effect                | NOOP boundary observed |
  |--------|-----------------------|------------------------|
  | 1      | r0 -= 4 (move up)     | r0 == 12 (top wall)    |
  | 2      | r0 += 4 (move down)   | (never hit; r0_max=24 assumed) |
  | 3      | c0 -= 4 (move left)   | c0 == 0  (left wall)   |
  | 4      | c0 += 4 (move right)  | (never hit; c0_max=36 assumed) |
  | 5      | no paddle move        | --                     |
  | 6      | no paddle move        | --                     |

  Verified: action 1 NOOPs in 15/15 r0=12 cases and moves in 22/22 other
  cases; action 3 NOOPs in 2/2 c0=0 cases and moves in 31/31 others.

### Constant outputs
- `reward_class = 1` on every tuple (200/200).
- `done = False` on every tuple (200/200).

## v1 result on full 200-tuple set

```json
{
  "game": "sp80",
  "split": "all",
  "n": 200,
  "errors": 0,
  "state_exact_pct": 100.0,
  "pixel_match_pct": 100.0,
  "reward_acc_pct": 100.0,
  "done_acc_pct": 100.0,
  "by_action": {
    "1": {"n": 37, "exact_pct": 100.0, "pixel_pct": 100.0, "errors": 0},
    "2": {"n": 27, "exact_pct": 100.0, "pixel_pct": 100.0, "errors": 0},
    "3": {"n": 33, "exact_pct": 100.0, "pixel_pct": 100.0, "errors": 0},
    "4": {"n": 34, "exact_pct": 100.0, "pixel_pct": 100.0, "errors": 0},
    "5": {"n": 35, "exact_pct": 100.0, "pixel_pct": 100.0, "errors": 0},
    "6": {"n": 34, "exact_pct": 100.0, "pixel_pct": 100.0, "errors": 0}
  }
}
```

## Why we got 100% on the first iteration
The game has unusually crisp structure for ARC-AGI-3:
1. Color 9 is a unique paddle anchor (zero detection ambiguity).
2. The K=3 vs K=2 fuel rule is purely a function of n14_before -- no hidden
   counter required.
3. Movement is rigid 4-cell stepping on a small slot grid.

Compared to bp35 (which left 31% unsolved due to hidden facing state and a
score-display refresh), sp80 has no observable "hidden state" leaking into
the visible grid.

## v2 plan (if pursued)
The model already scores 100/100/100/100 on the entire 200-tuple set, so v2
mostly means hardening against unseen states:

1. **Wall guesses for r0_max=24 and c0_max=36 are unverified**. If the
   training-time agent ever drives the paddle past these (e.g. by chaining
   action 2 from r0=24 in a state we haven't seen), the current code will
   still NOOP -- which is conservative but might be wrong. If we see a
   counter-example, relax to r0_max=28 / c0_max=40 (next 4-cell slot).
2. **Reward/done are constant in observations**; if the held-out set
   includes level-ups, we'd want to detect "fuel exhausted" -> `reward = 2`
   or trigger `done = True`. Add a hook that checks for the bar-reset
   pattern and bumps `reward_class` accordingly.
3. **Action 6 ignores (x, y) in 34/34 cases**. If a click-on-paddle case
   ever surfaces, we'd add a click-conditional move similar to bp35.

## Honest assessment
This was a textbook Rodionov target -- one anchor color, one rigid movement
rule, one deterministic counter rule, constant reward/done. The +100 came
from three crisp invariants verified on 100% of the data; no curve-fitting,
no hidden-state guessing.
