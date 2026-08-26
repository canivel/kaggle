# lp85 sim v2 notes

## Decision: no v2 sim written, v1 remains active

### Score comparison

| version | state_exact_pct | pixel_match_pct | reward_acc_pct | done_acc_pct | errors |
|---------|-----------------|-----------------|----------------|--------------|--------|
| v1      | 100.0           | 100.0           | 100.0          | 100.0        | 0      |
| v2      | (not written)   |                 |                |              |        |

v1 already saturates the metric on all 200 observed tuples. No failure
bucket exists to attack:

- `by_action[6]`: 200/200 exact, 0 errors.
- reward_class distribution {0:197, 1:3} — all 3 successful clicks
  perfectly reconstructed.
- done is False on every tuple — no done-prediction signal to learn.

### Why no v2 file

Writing speculative changes against zero failing tuples is curve-fitting
by construction. The candidate v2 ideas from the v1 notes
(lives-exhaustion, level-up target permutation, tighter/wider button
bboxes, actions 1..5/7) all need at least one observed counter-example
before they are anything more than guesses; otherwise we risk REGRESSING
the 100% v1 score on a future re-validation.

### Active sim

`f:/kaggle/arc-prize-2026/exec_wm/sims/lp85_sim.py` (v1, unchanged).

`validate_sim.py --game lp85` rerun confirms 100.0 / 100.0 / 100.0 / 100.0,
errors=0.

### Honest signal vs curve-fitting assessment

The 3 reward=1 tuples are crisp positive signal (CW and CCW rotations
both verified, lives consumption verified). The 197 reward=0 tuples are
weak signal individually (the identity rule is trivial) but collectively
confirm both button bboxes are tight enough that no non-button click was
mis-classified as a rotation. Combined: real signal, not overfit.

### Kaggle-inference usefulness

LIMITED. The sim is 197/200 identity, so as a forward model for planning
it mostly tells the agent "this click does nothing." That is still
useful for two narrow things:

1. **Pruning**: any click outside the two 8x6 bboxes is provably NOOP, so
   the agent can skip 99% of the click grid and focus on rows 29..36 at
   x in [2,7] U [56,61].
2. **Goal search**: with only 20 possible conveyor permutations and an
   exact CW/CCW model, once we know the target permutation we can plan
   in <=10 clicks (shortest path on a 20-cycle). We don't yet know what
   the target is, but the sim is ready to score candidates the moment a
   level-up event is observed.

If the Kaggle agent is allowed to learn the target online (one
exploration episode to find reward_class=2 or done), the sim becomes
high-value for planning. Without that, it is a NOOP-pruner, which is
modest but real.

## Re-trigger for v3

Build v3 only when one of these is observed in fresh data:
- a successful click that v1 predicts as identity (bbox too tight)
- an identity click that v1 predicts as a rotation (bbox too wide)
- a `done=True` or `reward_class=2` event (target permutation found)
- a `level` change (multi-level game)
- a click after lives are exhausted (rejection vs reset behaviour)
- any action_id in {1,2,3,4,5,7} with non-identity effect
