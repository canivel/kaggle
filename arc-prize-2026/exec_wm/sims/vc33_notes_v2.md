# vc33 sim v2 notes

## Decision: NO v2 shipped — v1 retained as active.

## What I investigated

v1 misses exactly 1/200 tuples (step 146, R7=4, n_changed=265). All
other 199 tuples are pixel-perfect under v1's row-0 timer rule.

Re-examining step 146 in detail (v1 notes were imprecise — the actual
diff is much larger than "11 cells"):

- **Row 0**: predicted correctly (R7=4, single flip — matches v1).
- **Col 4, row 0**: part of the row-0 flip (correct).
- **Cols 28-31, rows 1-27**: 4x27 = 108 cells flip 3 -> 0 (a vertical
  "score/progress bar" top segment).
- **Cols 46-55, rows 32-63**: a ~10x32 rectangular region of mixed
  3/0/4/11 values toggles. Transitions observed:
  `(3,0): 108, (0,3): 104, (4,3): 16, (0,4): 12, (0,11): 12, (11,3): 8,
   (11,4): 4, (7,4): 1`

This is plainly an in-game UI/score redraw, not a click-driven mechanic.

## Why I didn't ship a v2

Per stop criteria:

> "If v2 invariants verify on <90% of their target bucket: it's
> curve-fitting, revert."

With only **one** example of the redraw event (step 146 out of 200), any
rule I write — be it "fire at step % N == 0", "fire when R7==4",
"redraw when score crosses threshold" — would be supported by exactly 1
sample. There is no second instance to confirm periodicity, no third to
rule out alternatives. Hard-coding the step-146 delta as a lookup would
literally memorize the training set: 100% curve-fit, zero generalization.

The expected v2 gain is at most +0.5 absolute pts on state_exact
(99.5 -> 100.0), well below the +2pt "ship-but-flag" threshold. And
that's only on the *training* observations — on held-out rollouts the
hard-coded rule would almost certainly fire at the wrong step, *lowering*
exact match.

## What would actually unlock v2 here

Collect another ~200 clicks (steps 200..400). If the redraw recurs at
step ~292 (period 146), ~292..293 (period ~146), or any consistent
trigger, we'd have a falsifiable rule. Until then: identity-on-redraw
is the lowest-regret prediction.

## v1 vs v2 score table

| version | state_exact_pct | pixel_match_pct | reward | done | notes              |
| ------- | --------------: | --------------: | -----: | ---: | ------------------ |
| v1      |            99.5 |          99.968 |    100 |  100 | active             |
| v2      |             N/A |             N/A |    N/A |  N/A | not written (curve-fit risk) |

## Honest assessment

v1's row-0 timer rule is a real, generalizable invariant: 199/199
non-redraw tuples are perfectly predicted, and the DOUBLE_FLIP_R7 set
was independently observed at every member ≥4 times.

The remaining 0.5% gap is a UI redraw whose trigger cannot be identified
from a single sample. Shipping a memorized fix would mean accepting
99.5 -> 100.0 on the training set in exchange for likely regression on
held-out data. Not worth it.

**For Kaggle inference**: vc33 is action-6-only with deterministic
row-0 dynamics and a constant reward of 1. A sim that perfectly predicts
row-0 progression is useful for planning the *number* of clicks needed
to reach the next reset (every 50 clicks), but since reward is always 1
regardless of state, planning gains nothing here. The sim is correct
but the *game* doesn't reward sim-based planning. Low-value for LB.
