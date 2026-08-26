# cn04 sim v2 notes

## What changed from v1

1. **Action 5 = 90 degree clockwise rotation of the playable blob about
   its bbox top-left corner.** Mapping: `(y, x) -> (rmin + (x - cmin),
   cmin + (H - 1 - (y - rmin)))`. New bbox shape is W x H (dimensions
   swap). NOOP if the blob contains color 12 (lock) or if the rotated
   bbox would exit the grid. Verified 11/11 on the changed-state cases
   and 24/24 on the locked NOOP cases (the existing lock predicate
   already covers them).

2. **Action 6 wall-click rule:** when the click cell holds color 14,
   every color-14 cell flips to color 12 (level-clear lock). Verified
   1/1 on the only effective wall-click observed in the 200 tuples.
   Non-wall clicks remain identity (matches 32/34 observed clicks).

## v1 vs v2 scores

| metric             | v1     | v2     | delta  |
|--------------------|--------|--------|--------|
| state_exact_pct    | 73.50  | 77.50  | +4.00  |
| pixel_match_pct    | 99.68  | 99.94  | +0.26  |
| reward_acc_pct     | 78.00  | 84.00  | +6.00  |
| done_acc_pct       | 100.00 | 100.00 |  0.00  |
| a=1 exact_pct      | 72.97  | 72.97  |  0.00  |
| a=2 exact_pct      | 77.78  | 77.78  |  0.00  |
| a=3 exact_pct      | 78.79  | 78.79  |  0.00  |
| a=4 exact_pct      | 79.41  | 79.41  |  0.00  |
| a=5 exact_pct      | 54.29  | 77.14  | +22.85 |
| a=6 exact_pct      | 79.41  | 79.41  |  0.00  |

Action 5 jumped 22.85 points — the rotation rule was the dominant
unmodeled mechanic. Pixel match improved across the board (rotation now
matches the full ~150-pixel payload instead of stamping it as identity).

## Residual errors (45 / 200)

Same structural cause as v1: row-0 timer ticks that are
non-deterministic from a single frame.

- Actions 1..4: ~21 "tick-only" cases (NOOP truth but timer ticked)
  + 9 "shift+tick" cases (correct shift, missed tick) = 30 errors.
- Action 5: 8 residual errors are also timer-tick only.
- Action 6: 7 errors are timer-tick on NOOP clicks.

Predicting "always tick" breaks ~157 NOOPs/correct-shifts; predicting
"never tick" loses ~43 (current behaviour). Net: -114. Skip.

## Honest assessment

- **Signal, not curve-fitting.** The action-5 rotation rule was derived
  from a structural property (blob mass preserved, bbox transposed,
  top-left pixel pivot) and verified 11/11 on the changed-state cases
  plus 24/24 on the lock-NOOP cases. The wall-click rule is thin
  (n=1) but mechanistically sensible (clicking the static wall caches
  it as a lock barrier).

- **Plausible Kaggle impact.** cn04 looks like a sliding-rotation
  puzzle where the agent must (a) translate the blob into alignment
  with the wall, then (b) rotate it, then (c) click the wall to lock
  the position. With v2 we can now plan rotation actions, not just
  translations. That is the difference between identity-mostly (which
  cannot help inference) and an actual planner; v1 already crossed
  that threshold and v2 widens the gap on the only non-NOOP rule we
  hadn't modelled.

- **Stop criterion check.** v2 improved exact-match by 4.0 absolute
  points (above the 2-pt marginal threshold) and pixel-match by 0.26
  points (no regression). All new invariants verified on >=100% of
  their target buckets. Ship v2 as the active sim.

## Active sim

`cn04_sim.py` now contains the v2 implementation. `cn04_sim_v2.py`
preserved as the labeled source. v1 retained as the git-tracked
history; no separate backup file is kept.
