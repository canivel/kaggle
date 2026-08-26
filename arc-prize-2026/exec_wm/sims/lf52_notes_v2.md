# lf52 sim v2 notes

## Decision: no v2 sim written; v1 remains active.

## Score comparison

| metric              | v1     | v2 |
|---------------------|--------|----|
| state_exact_pct     | 100.0  | -  |
| pixel_match_pct     | 100.0  | -  |
| reward_acc_pct      | 100.0  | -  |
| done_acc_pct        | 100.0  | -  |
| errors / 200        | 0      | -  |
| worst by_action pct | 100.0  | -  |

Re-ran `validate_sim.py --game lf52` on 2026-06-26: still 0 errors,
100% across the board, 100% in every per-action bucket
(1/2/3/4/6/7). No regressions, no headroom.

## Why no v2

The brief asks "aim higher state_exact_pct without regressing
pixel_match_pct". Both are pinned at 100.0 on every available tuple
across every action bucket. There is no failing tuple to inspect:
`errors == 0`. Any change to the sim falls into one of two categories:

1. **Cosmetic / equivalent code path** — same outputs, no metric
   movement. Wasted churn.
2. **Speculative extension** (wrap at 15, playfield activation,
   level-up, reward_class != 1, action-specific branches) — these
   address transitions we have never observed. With zero failing
   tuples to anchor on, any such branch is curve-fitting against
   imagined physics. The STOP CRITERIA explicitly call this out:
   "If v2 invariants verify on <90% of their target bucket: it's
   curve-fitting, revert." We would be at 0/0.

## What would unlock a real v2

A longer rollout (>=600 steps) that surfaces at least one of:
- row 0 saturated at value 15 (test wrap),
- `reward_class != 1` (separate the "no-op" branch),
- `level > 0` (multi-level state machine),
- `n_changed > 1` (playfield wake-up).

Until one of those is in the observation file, v1 is provably optimal
on the data and v2 is undefined.

## Kaggle-inference utility

Honest read: this sim is a counter, not a planner-relevant model. It
predicts state perfectly but every action gives the same reward (1),
so it offers **no action discrimination signal** to a search policy.
On Kaggle inference it would:
- score 100% on a WM-fidelity probe of lf52 (good for an ensemble of
  per-game sims),
- contribute nothing to action selection inside lf52,
- generalize to no other game (single-game memorization of a trivial
  rule).

Net: ship v1 as the lf52 entry of a per-game sim bank; do not expect
it to lift the agent's lf52 score. The real lift needs richer
observations first, then a sim that branches on action.
