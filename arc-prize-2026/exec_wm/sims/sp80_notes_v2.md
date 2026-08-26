# sp80 sim v2 notes -- NO-OP iteration (v1 retained)

## Decision
v1 already scores **100/100/100/100** on the full 200-tuple set, with 100% on
every individual action bucket (1..6). Per the stop criteria:
- Improvement headroom is 0 absolute pts. The threshold "<2 pts -> marginal"
  collapses to "any change is risk-only".
- Any new wall-relaxation, click-handler, or fuel-exhausted hook would be
  fitted on **zero** in-distribution failures -- pure speculation about
  held-out behaviour. That is exactly the "curve-fitting" failure mode the
  workflow warns against.

So v2 = v1. No `sp80_sim_v2.py` was written; the v1 file remains active.

## v1 vs v2 comparison

| metric            | v1     | v2 (would-be) | active |
|-------------------|--------|---------------|--------|
| state_exact_pct   | 100.0  | (n/a)         | v1     |
| pixel_match_pct   | 100.0  | (n/a)         | v1     |
| reward_acc_pct    | 100.0  | (n/a)         | v1     |
| done_acc_pct      | 100.0  | (n/a)         | v1     |
| worst action bkt  | 100.0  | (n/a)         | v1     |

Re-ran `validate_sim.py --game sp80` 2026-06-26: still 200/200 errors=0,
identical by-action breakdown.

## Honest signal-vs-curve-fitting assessment
v1's three invariants (paddle = unique color-9 4x20 rect; movement = rigid
4-cell step; fuel-drain K = f(n14_before)) are *structural*, not coincidental
-- each verified on 200/200 with zero exceptions. This is real signal.

The only unverified pieces are:
- `r0_max=24` / `c0_max=36` walls (never hit in train -- NOOP is the
  conservative guess; truth could be "wraps" or "moves further").
- `reward_class=1`, `done=False` constants (could flip at level-up).

These are documented as v2-future hooks in v1 notes; touching them now would
fit noise.

## Kaggle-inference relevance
This sim is **not** identity-mostly. It actively predicts:
- 6 distinct fuel-drain patterns per step (K=2 or K=3 from a state-dependent rule).
- 4 paddle-translation outcomes vs 2 stationary actions, with wall-NOOP logic.

A planner that calls `simulate` to score candidate action sequences gets a
useful, lossless forward model on this game. It plausibly helps Kaggle
inference if the held-out level shares the same paddle mechanic (likely --
sp80 is a "drive the paddle to a target" puzzle).

The one risk to inference value is if the held-out level *adds* a new
mechanic (e.g. a moving enemy on rows 8-63) that this sim ignores. v1 leaves
rows 1-63 unchanged outside the paddle stamp; any new dynamic object there
would silently desync. We accept this risk -- adding speculative dynamics
without evidence is the worse failure mode.
