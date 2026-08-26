# ar25 sim v1 notes

## Game in one line

Two 9x9 sprites (L = colors 5/0, R = color 4) sit in left/right playfields
split by a 3-col vertical divider; D-pad-style actions move BOTH sprites
simultaneously in mirrored horizontal directions and same vertical direction.
A "right meter" in col 63 ticks 11 -> 5 from the top with every successful
action.

## Invariants used

1. **Static background**: playfield=9; divider cols 30..32 = 10; col 63 = 11
   (the meter); row 63 = 5; rows 45-52 cols 51-59 = a static L-shaped
   11-decoration. Verified unchanging across 200 tuples.

2. **Two sprites with fixed 9x9 templates**:
   - L template: 3-row top bar (color 5 with 0-holes at cols 1,4,7) plus
     a 6-row right column (cols 6..8 with 0-holes at rows 4,7).
   - R template: L's mirror — 3-row top bar (all color 4) plus 6-row
     left column (cols 0..2).
   Sprite positions are on a 3-cell grid (top-left r0,c0 multiples of 3).
   Detection done via opaque-cell template matching with a strict score.

3. **Action transitions** (verified bucket-wise on 200 tuples):
   - action 1: both sprites UP 3 rows  (28/31 succeed; rest = top boundary)
   - action 2: both sprites DOWN 3 rows (18/19)
   - action 3: L LEFT 3 cols, R RIGHT 3 cols (mirrored)
   - action 4: L RIGHT 3 cols, R LEFT 3 cols (mirrored)
   - action 5: tick meter (top 11 -> 5), sprites unchanged
   - action 6: pure NOOP, reward 0
   - action 7: non-deterministic from one frame — leave identity

4. **Boundaries**: r0 ∈ [0, 27], L c0 ∈ [3, 30], R c0 ∈ [33, 51]. If a
   vertical action would push a sprite OOB, BOTH sprites NOOP. Horizontal
   actions independently clamp each sprite.

5. **Rendering rule**: opaque sprite color (5 for L, 4 for R) over divider
   is treated as L > divider > R (L wins, divider wins over R). Sprite-0
   cells render as 0 over playfield-9 but as the underlying bg feature
   (e.g. divider 10) otherwise.

## v1 result on full 200-tuple set

```
state_exact_pct: 67.5   pixel_match_pct: 99.44
reward_acc_pct: 98.0    done_acc_pct:   100.0
by_action:
  1 (up)     n=31  exact=80.6 %  pixel=99.87 %
  2 (down)   n=19  exact=89.5 %  pixel=99.88 %
  3 (left)   n=27  exact=66.7 %  pixel=99.56 %
  4 (right)  n=29  exact=48.3 %  pixel=99.48 %
  5 (tick)   n=27  exact=100  %  pixel=100  %
  6 (noop)   n=31  exact=100  %  pixel=100  %
  7 (mirror) n=36  exact=8.3  %  pixel=97.80 %
```

## What v1 still gets wrong

### Action 7 (~33/36 unsolved)
Action 7 mirrors the LAST action (which we cannot recover from a single
state). Empirically:
- L-facing + action 7 -> often moves down
- R-facing + action 7 -> often moves up or left
Without a history register we cannot decide. v1 returns identity, scoring
on the 3 NOOP cases only.

### Sprite-cut on boundary push (action 3 / action 4)
When R sprite is pushed left against c0=33 (or L against c0=3 / c0=30),
the sprite does NOT just "stay" — it gets PUZZLE-CUT, losing the right 3
cols of the top bar AND its bottom-left 6×3 column (45 cells -> 18 cells).
Another blocked push reduces 18 -> 0 (sprite vanishes). v1 leaves the
sprite intact, mispredicting 7/27 action 3 cases (R can't move right past
51) and ~15/29 action 4 cases (R blocked at c0=33; sprite shrinks).

### Sprite occlusion in overlap zones
When the L sprite top bar crosses the divider, it overwrites the divider
10 (L wins). But when R sprite top bar crosses the divider, the divider
WINS over R (R 4 -> 10 displayed). Modeled correctly. However when L and
R sprites overlap in the playfield (rare; ~5 cases), the truth shows R's
top bar partially erased (cols overlapping with L disappear). v1 stamps
L on top of R; this matches in most overlaps but not all.

## v2 plan (if pursued)

1. **Implement piece-cutting**: when a horizontal action would push a
   sprite past its c0 limit, do not stay — instead remove the protruding
   columns and rows of that sprite. Likely +25 % on action 4 and +15 %
   on action 3.
2. **Action 7 hidden-state guess**: track the previous-action symbol by
   diffing two consecutive states (not possible with single-frame API).
   Without that, model action 7 as "most likely up" (modal in data).
3. **Score / level events**: monitor row-63 / col-63 for done-trigger
   conditions; none observed in 200 tuples so probably nothing to model.
