# lp85 sim v1 notes

## Game summary (from observations)

- 200 random-exploration tuples. ONLY `action_id == 6` (click) available.
- Reward distribution: 0 -> 197 (NOOP), 1 -> 3 (successful clicks).
- `done` always False; `level` always 0.
- Grid 64x64; background colour 4. Visible palette: {1,2,3,4,5,8,9,10,11,14,15}.

## Structure

The playfield is a **20-tile clockwise conveyor belt** of 4x4 monochrome tiles
laid out as a rectangle:

- Top row: 7 tiles at rows 19-22, column-anchors {12,18,24,30,36,42,48}.
- Right column: 3 tiles at cols 48-51, row-anchors {25,31,37}.
- Bottom row: 7 tiles at rows 43-46, column-anchors {12,18,24,30,36,42,48}.
- Left column: 3 tiles at cols 12-15, row-anchors {25,31,37}.

Cycle order (clockwise from top-left):
top L->R, right_col T->B, bottom R->L, left_col B->T.

Two click "buttons":
- LEFT button: rows 29-36, cols 2-7  (rendered with colour 8).
- RIGHT button: rows 29-36, cols 56-61 (rendered with colour 14).

Vertical "lives" counter at col 0: rows of 14 = remaining, rows of 5 = used.

## Invariants used (3 crisp rules, all verified 100%)

1. **NOOP rule** (197/197 reward=0 cases): any click outside the two button
   bboxes -> identity transition, reward=0.
2. **Right-button rule** (2/2 events): click in (y in [29,36], x in [56,61])
   rotates the 20-tile cycle CLOCKWISE by 1 step
   (`new[i] = old[(i-1) mod 20]`).
3. **Left-button rule** (1/1 event): click in (y in [29,36], x in [2,7])
   rotates the 20-tile cycle COUNTER-CLOCKWISE by 1 step (the exact inverse).
4. Every successful click also consumes 5 lives (top 5 remaining-14 cells in
   col 0 flip to 5).

## v1 result on full 200-tuple set

```
state_exact_pct: 100.0   pixel_match_pct: 100.0
reward_acc_pct: 100.0    done_acc_pct: 100.0
by_action: 6 -> 100.0 % exact
errors: 0
```

## v2 plan

There is no obvious v2 — every observed tuple is already exact. Things to
watch for if/when more observations arrive:

1. **Lives exhaustion**: with only 3 successful clicks we never saw the
   lives counter run out. If a 13th successful click happens we expect
   either: the counter resets, the click is rejected, or `done=True`.
   Currently `_consume_lives` silently no-ops if no 14-cells remain, which
   is the safest fallback.
2. **Goal state / level-up**: we never observed `level` change. If a future
   conveyor rotation matches some hidden target permutation, we may see
   `reward_class=2` and/or `done=True`. The cycle has 20 states (one per
   rotation) so brute force is cheap once we know what to match.
3. **Click bbox tightness**: the 3 observed successful clicks were at
   (x,y) = (58,34), (6,31), (56,30). Our 8x6 bbox is conservative; if a
   click *between* the buttons or at the edge ever produces a reward, we
   may need to tighten or widen.
4. **Other actions**: `available_actions == [6]`, but if action ids 1..5 or
   7 ever appear in deeper play, our `action_id != 6 -> identity` default
   is the correct floor until we observe their effects.
