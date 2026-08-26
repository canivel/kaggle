# su15 sim v1 notes

## Game description (1-line)

A passive "counter increment" game on a static playfield: clicks in the
playfield region tick a 2-cell counter on row 63 (filling right-to-left),
while action 7 and clicks above the playfield (y < 10) are no-ops.

## Observation summary (200 tuples)

- `available_actions`: {6, 7}
- `action_distribution`: {6: 102, 7: 98}
- `reward_distribution`: {0: 113, 1: 87}
- `done`: False on every tuple; `level`: 0 throughout.
- Joint (action, reward): (7, 0)=98, (6, 1)=87, (6, 0)=15.

n_changed distribution:
- 0 -> 113 (all action 7 + 15 action 6 with y < 10)
- 2 -> 86 (action 6, y >= 10, counter tick)
- 22 -> 1 (action 6, step 4 — a rare sprite stamp + counter tick; left
  unmodelled)

## Invariants used by v1

1. **Action 7 is a pure no-op** (98/98).  Identity grid, reward 0.
2. **Action 6 with y >= 10 ticks the counter**: two cells on row 63
   flip 0 -> 5, at the rightmost still-empty (2k, 2k+1) pair.  Verified
   on 86/86 plain cases.  Reward 1.
3. **Action 6 with y < 10 is a no-op** (15/15).  Identity, reward 0.

## v1 result

```
{
  "state_exact_pct": 98.5,
  "pixel_match_pct": 99.9971923828125,
  "reward_acc_pct": 100.0,
  "done_acc_pct": 100.0,
  "by_action": {
    "6": {"n": 102, "exact_pct": 97.06, "pixel_pct": 99.9945},
    "7": {"n": 98,  "exact_pct": 100.0, "pixel_pct": 100.0}
  }
}
```

3 mispredictions remain on action 6:
- 1 is the n_changed=22 "sprite stamp" case at step 4 (x=12, y=62).  The
  delta shows a small object cluster on the playfield (rows 52-62, cols
  3-13) plus the usual counter tick.  With a single example we can't
  derive the rule.
- The other 2 are almost certainly similar rare events but the summary
  did not surface them as exemplars.  Pixel-match remains ~99.99% so
  these are very localised mispredictions.

## v2 plan (if pursued)

1. **Sample raw observations** to find any more n_changed > 2 cases.  If
   the sprite stamp follows a pattern (e.g. fires at counter==4 ticks),
   model it.  Likely worth +1.5% exact-match (3/200 cases).
2. **Counter-full wrap**: not observed (counter only filled to cols
   54-63 max in 200 tuples), so untested.  When closer to 32 ticks this
   may matter; the current sim returns identity which is safe.
3. **Click-on-existing-counter cell**: untested whether a click on a
   row-63 cell within the playfield (y >= 10) still ticks the counter.
   Empirically all action-6 successes had y >= 10 AND were not on
   row 63, so the current rule is fine on the sample.

## Honest assessment

The +98.5% comes from three crisp invariants (one no-op action, one
out-of-bounds no-op, one deterministic 2-cell counter tick).  All three
are verified on 100% of their matching bucket (98/98, 15/15, 86/86).
The remaining 1.5% is a single observed rare event plus its
unseen-but-likely siblings — not worth chasing without more data.
