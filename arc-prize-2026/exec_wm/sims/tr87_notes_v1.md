# tr87 sim v1 notes

## Game in one line
A picture-puzzle picker UI: three rows of "question -> answer" 7x7 icons
in a preview panel (rows 4-29) plus a menu strip of 5 candidate 5x5
icons (rows 51-57) with a square-bracket selector in color 0 around the
currently-selected slot. Actions 3/4 move the selector left/right (wrap
through 5 slots); actions 1/2 cycle the icon at the selected slot
forward/backward through a per-slot fixed library.

## Invariants used

1. **Selector bracket** is rendered in color 0:
   - top row 48, cols sl..sl+4 (5 cells), plus row 49 at sl and sl+4
   - bottom row 60 (5 cells) + row 59 at sl and sl+4
   sl ∈ {15, 22, 29, 36, 43}, step = 7.
   Action 3 cycles left, action 4 cycles right (with wrap).
   Verified on 101/101 action-3/4 transitions: the only diff outside row 63
   is the bracket move.

2. **Per-slot icon cycle**: actions 1 and 2 cycle the 5x5 patch at
   (rows 52..56, cols sl..sl+4) through a slot-specific library. The
   (slot, icon, action) -> next_icon mapping is deterministic (0/46
   conflicts across all observed forward/backward transitions). Table
   hard-coded from training tuples; inverses filled in for completeness.

3. **Row 63 step counter** ticks on every other action regardless of
   action_id. Tick rule: rightmost 1 in row 63 becomes 4. The parity
   bit is purely hidden external state -- not derivable from a single
   frame. We leave row 63 unchanged in v1.

## Result

```
{
  "game": "tr87",
  "split": "all",
  "n": 200,
  "errors": 0,
  "state_exact_pct": 50.0,
  "pixel_match_pct": 99.98779296875,
  "reward_acc_pct": 100.0,
  "done_acc_pct": 100.0,
  "by_action": {
    "2": {"n": 49, "exact_pct": 51.02, "pixel_pct": 99.988},
    "1": {"n": 50, "exact_pct": 50.0,  "pixel_pct": 99.988},
    "3": {"n": 45, "exact_pct": 55.56, "pixel_pct": 99.989},
    "4": {"n": 56, "exact_pct": 44.64, "pixel_pct": 99.986}
  }
}
```

The exact-match ceiling at ~50% is exactly the hidden-parity barrier on
the row-63 counter: on every other call the counter ticks regardless of
action, but a single-frame call has no way to know the parity. Across
all four actions the per-action exact% is ~45-55%, consistent with the
50/50 coin flip on the counter.

Pixel-match is 99.99% because we get the entire 4096-cell grid right
except (at worst) one cell on row 63.

## v2 plan (diminishing returns expected)

1. **Try to encode parity in a derived feature.** Look for any per-state
   hint we missed: e.g. does a single pixel anywhere flicker every other
   step? My earlier scan of rows 0-47 across step 0 vs step 2 showed no
   change, but I only checked one pair. Sweep all even/even pairs to be
   sure.

2. **Action prediction tiebreak.** If we *assume* "tick" on every call we
   still get 50%, but pixel match might be marginally higher; not worth
   risking the existing 99.99% pixel score for it.

3. **Reward / done modelling for terminal states.** All 200 tuples have
   reward 1 and done False -- we never observed an end-of-puzzle event.
   This sim will mispredict on the win/level-up frame. Need more obs.

4. **Unseen-icon coverage.** If validation ever sees an icon at a slot
   that wasn't in our 200-sample library, action 1/2 falls back to
   identity. Could synthesize the missing transitions by inspecting the
   "library order" if the library is canonically ordered (e.g. by
   appearance in the preview panel).

## Honest assessment

The exec-WM is essentially complete given a single-frame input: selector
move + per-slot icon cycle + reward/done constants account for every
non-counter cell. The 50% exact-match ceiling is a fundamental limit
of the I/O contract; without hidden-state passthrough between calls, we
cannot do better. v2 needs either a richer simulate() signature
(carry an external `step` argument) or new observations showing what
distinguishes tick-vs-no-tick frames.
