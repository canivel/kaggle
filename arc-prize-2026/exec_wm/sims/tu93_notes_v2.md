# tu93 sim v2 notes -- NO-OP iteration (v1 retained)

## Decision: v1 retained, v2 not written

## v1 vs v2 score comparison

| metric            | v1     | v2 (not written) |
|-------------------|--------|------------------|
| state_exact_pct   | 100.0  | -- (no headroom) |
| pixel_match_pct   | 100.0  | -- (no headroom) |
| reward_acc_pct    | 100.0  | -- (no headroom) |
| done_acc_pct      | 100.0  | -- (no headroom) |
| errors            | 0/200  | -- (no headroom) |

Active sim file: `tu93_sim.py` (v1 unchanged).

## Why no v2

Re-ran `validate_sim.py --game tu93 --split all` -- v1 is already at the
ceiling on every metric across all 200 tuples and all 4 actions
(1:50, 2:49, 3:45, 4:56 samples). There is no failing bucket to target.

The dataset itself is also already saturated for the dynamics v1 models:
- `reward_class` = {1} on all 200 tuples (no reward diversity)
- `done` = {False} on all 200 tuples (no terminal transition)
- `level` = {0} on all 200 tuples (no level-up event)
- `action_id` in {1, 2, 3, 4} (no click action 5/6/7)
- sprite never adjacent to the color-14 goal at (46-48, 45-47)

The v1 notes already flagged three plausible v2 directions (goal-reach
`done=True`, post-cycle timer reset, click actions) -- ALL THREE require
new tuples to derive invariants from. Writing them blind would be
curve-fitting on imagined dynamics, which the stop criteria explicitly
forbids ("if v2 invariants verify on <90% of their target bucket: revert").

## Honest signal vs curve-fitting

**Signal**: 100% across 4 independent invariants (sprite anchor, connector
binary, action->facing, timer cycle table) over 200 tuples = 4 full
50-action cycles. Each timer pre-count value seen exactly 4 times. The
invariants are crisp, mechanical, and mutually independent.

**Curve-fitting risk**: low for the modelled mechanics; HIGH if we
speculate about `done` / level-up / clicks without data. The right next
step is data collection (targeted policy that walks toward the goal),
not code.

## Kaggle-inference value

Mixed. v1 IS a true executable WM for the maze-walk subset:
- Planner rollouts of <=50 steps deep are exact.
- Useful as a one-step lookahead for any BFS/MCTS that needs to verify a
  proposed move is not a NOOP (connector-color check is the same fast
  invariant the sim uses internally).
- NOT identity-mostly: the action delivers a real 6-cell jump of the
  3x3 sprite and a 1-or-2-cell timer tick -- a non-trivial state change
  on every step.

What it can NOT do on Kaggle inference (until more data):
- Predict `done=True` when the sprite reaches the goal.
- Predict `level+=1` or the next-level layout.
- Handle planner rollouts that cross a >50-step timer reset (best-effort
  reset already in `_advance_timer`, but unverified).
