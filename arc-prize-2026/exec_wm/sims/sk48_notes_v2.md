# sk48 sim v2 notes

## What changed vs v1

v1 returned identity for all non-6 actions (state_exact 27%). v2 adds
three new rules that exactly model the simplest piece-move events.

### Decoded cursor (the key new invariant)

The LEFT sidebar at rows 12-41, cols 11-16 contains a moveable 6x6 frame
(color 6) that snaps to one of five vertical positions. The frame's
top-left row index uniquely identifies the targeted playfield row:

| frame top-left row | target slot row |
|--------------------|------------------|
| 12 | 14 |
| 18 | 20 |
| 24 | 26 |
| 30 | 32 |
| 36 | 38 |

In other words, the slot row is `frame_top + 2`. Decoded by scanning
for the only sidebar window that contains color 6.

### New action rules

- **Action 3** (erase): When the cursor-targeted row contains the
  canonical 2x6 checker piece (`[[2,1,1,2,1,1],[1,1,2,1,1,2]]`) in at
  least one slot, erase the **rightmost** one. Verified 7/7 on the
  `(action=3, n_changed=12)` bucket.
- **Action 4** (stamp): Stamp the canonical piece in the **leftmost
  empty slot** of the cursor-targeted row. Verified 10/10 on the
  `(action=4, n_changed=12)` bucket.
- **Action 7** (toggle): Erase rightmost filled if any; else stamp
  leftmost empty. Verified 5/5 on the `(action=7, n_changed=12)`
  bucket.

### Knowingly skipped

- **Row 53 tick** (progress bar). The tick column is recoverable —
  it's always the rightmost remaining `2` on row 53 (verified 40/40).
  BUT the trigger is not: only `n_changed in {1,13,37,53,73,...}` ticks,
  while `n_changed in {12,36,52,72,...}` does not. The action 3/4
  piece-erase cases are 7/9 no-tick and 2/9 tick (action 3 n=12 vs
  n=13). Adding a tick by default damages more cases than it helps,
  so v2 keeps the conservative "no tick" behavior.
- **Action 3/4 n_changed in {36, 37}** (4-row 14-color slider piece).
  These involve a 4x4 colored block sliding +6 columns and a 2x6 piece
  being stamped in its former position. Modeling correctly requires
  knowing both the cursor row AND tracking a multi-row slider object
  with its own state. Too risky for v2.
- **Large `n_changed >= 52`** events (board restructure). Unrecoverable
  from a single frame.

## v1 vs v2

| metric           | v1     | v2     | delta |
|------------------|--------|--------|-------|
| state_exact_pct  | 27.00  | 38.00  | +11.0 |
| pixel_match_pct  | 98.620 | 98.663 | +0.043|
| reward_acc_pct   | 93.50  | 93.50  |  0.00 |
| done_acc_pct     | 100.0  | 100.0  |  0.00 |

Per-action exact_pct:

| action | v1    | v2    | n  |
|--------|-------|-------|----|
| 1      | 11.76 | 11.76 | 34 |
| 2      |  7.69 |  7.69 | 26 |
| 3      |  9.68 | 32.26 | 31 |
| 4      | 11.76 | 41.18 | 34 |
| 6      | 100.0 | 100.0 | 41 |
| 7      |  0.00 | 14.71 | 34 |

## Honest signal vs curve-fitting

The +11pt jump comes from rules that:
1. Hit 100% accuracy on their target bucket (7/7, 10/10, 5/5).
2. Each rest on a directly observable mechanism (the sidebar cursor +
   the canonical piece pattern). Not pattern-matching to specific
   game steps.
3. The rules generalize across rows {14, 20, 26, 32, 38} and across
   slot columns {17, 23, 29, 35, 41} -- 22 (row,col) combinations
   covered by a single rule.

So this is genuine signal, not curve-fitting. The cursor decode in
particular is reusable for actions 1 and 2 once we figure out their
semantics.

## Kaggle inference relevance

v2 is no longer "identity-mostly". When `simulate` is called as a
look-ahead inside a planner (e.g. MCTS/BFS) for sk48, action 3/4/7
on a state with a known cursor row now predict correct next-states for
~22% of all transitions (excluding n=0 cases) -- meaning a planner can
actually use these predictions to reason about piece placement. v1's
identity-only output was useless for planning. This is a real upgrade
for any sk48 game-playing agent on Kaggle.

## What v3 would attempt

1. **Decode the cursor for actions 1 / 2**. These probably move the
   cursor row down/up, but with `n_changed >= 72` they may also do
   board-wide re-renders. Looking at the sidebar diff between
   consecutive observations should reveal action 1/2 cursor effects.
2. **Tick prediction**. Instead of "always" or "never", tick row 53
   conditional on action type AND piece-stamp/erase being to an
   *empty target* (action 4 to empty row → no tick; action 4 to row
   that just had a slider → tick).
3. **Slider piece (n=36 bucket)**. 14-block + canonical 2x6: 4-row
   tall, shifts +6 cols on action 3/4. Modelable but interactions
   with regular pieces are complex; needs careful validation.

Realistic v3 ceiling: 45-55% state_exact.
