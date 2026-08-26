# su15 sim v2 notes

## What changed from v1

v1's three remaining mispredictions were i=4 (sprite stamp), i=53, i=54
(counter ticks where the new cells were `15` instead of `5`).

Drilling into i=53/54 I first hypothesised a fixed "warning zone" at
cols 11..13 -- but a draft that hard-coded those columns immediately
broke at i=122/123 (cols 12,13 written as `5,5` in the second
trajectory segment).  That was a clear curve-fitting signal.

The actual rule:

> A counter cell at (63, c) inherits its colour from the cell
> directly above it on row 62.  If `state[62, c] == 15`, write `15`;
> else write the normal counter colour `5`.

Background: the lone sprite-stamp event at i=4 planted `15` cells
into rows 61-62 at cols 11-13.  Later, when the counter's right-to-
left fill reached those columns (i=53, 54), the counter "absorbed"
that colour.  After the i=70 reset there was no sprite event, so
row 62 stayed at `5`s and the counter at cols 11..13 wrote plain
`5,5` (i=122, 123) -- which the v2 rule predicts correctly.

## v1 vs v2 (full 200-tuple split)

| metric            | v1     | v2     | delta  |
|-------------------|--------|--------|--------|
| state_exact_pct   | 98.50  | 99.50  | +1.00  |
| pixel_match_pct   | 99.9972| 99.9976| +0.0004|
| reward_acc_pct    | 100.0  | 100.0  | 0      |
| done_acc_pct      | 100.0  | 100.0  | 0      |
| action 6 exact %  | 97.06  | 99.02  | +1.96  |
| action 7 exact %  | 100.0  | 100.0  | 0      |

Only one tuple now fails: i=4, the n_changed=22 sprite stamp.  One
example is still not enough to derive a rule for what gets drawn or
what triggers it; I refuse to over-fit a 5x5 stamp from a single
sample.

## Honest assessment

The +1.0 pt gain falls just below the "ship but flag" 2-pt heuristic
threshold.  However the new invariant is principled (not curve-fit),
verified on 5/5 near-warning ticks AND on the 50+ untouched-row-62
ticks in the second segment, AND on the killer split-test that broke
the wrong-but-tempting "fixed cols 11..13" hypothesis.  Net: signal,
not noise.

## What is NOT yet modelled

1. **Sprite-stamp trigger (i=4)**.  Single example; cannot generalise.
2. **Counter reset / level wrap** at i=70 and i=146 (fills jumps
   62 -> 0).  These resets fall between adjacent tuples, so they
   don't cost us a state-exact tuple in the current 200-sample.
   But on a longer rollout the sim will get stuck at "counter full"
   and start mispredicting.  Cheapest fix: if the counter is full
   AND action 7, treat as reset.  Not done in v2 because there is no
   action-7 tuple at a full counter to verify on.
3. **Click on row 63 within the playfield**: untested.

## Kaggle inference value

This sim is now ~99.5 % state-exact and 100 % reward-accurate.  It
is "identity-mostly" (115/200 tuples are no-ops), but the active-tick
rule (counter increments + warning inheritance) is enough to let a
search-based agent simulate forward `K` action-6 calls and choose
when to stop ticking.  Since the only reward signal is action-6 on
the playfield, the inference-time policy is essentially "spam action
6 with any y >= 10" -- the sim does not change that.  Marginal
inference value unless the level-transition (counter wrap) becomes
the gating mechanism on private LB.
