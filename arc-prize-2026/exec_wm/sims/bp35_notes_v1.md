# bp35 sim v1 notes

## Observations

### Top-level

- 4 available actions: {3, 4, 6, 7}
- 200 random-exploration tuples. Distribution: 4=51, 3=48, 6=51, 7=50.
- `reward_class` is **1** for every single tuple. No `done` true. Level
  stays at 0 throughout. So a constant `(reward=1, done=False)` is
  optimal for these two outputs (and indeed scores 100 % on both).

### Row 63 is a monotonic step counter

Every `n_changed == 1` transition (the simple case) was exactly one
cell on row 63 flipping from 0 -> 15. The column that flipped equals the
leftmost remaining 0-cell on row 63. In the 6 rendered exemplars I saw
the counter at:

- step 0: row63 = all 0 -> step 1 has (63,0)=15
- step 137: cols 0..38 are 15, rest 0
- step 149: cols 0..50 are 15, rest 0

Increments are not always `step%64`; the counter only advances on
"interesting" events but in the 200 tuples it advanced once per action.
So: `next_state[63, first_zero] = 15`.

### Sprite (frog) toggle in actions 3 / 4 / 7

A 5x5 sprite made of colors {3, 5, 9, 11, 14} sits inside the playfield
on a uniform background of 10. Actions 3 and 4 swap the sprite between
two horizontally adjacent 5x5 blocks (rows 37..41, cols 19..23 <->
cols 25..29 in the exemplars). Action 7 sometimes mirrors the previous
sprite move, sometimes does nothing (n_changed=1).

Larger `n_changed` values (300+, 1400+) occur when the sprite is farther
from its initial spot and the swap blocks overlap less, OR when multiple
sprites/objects exist. The summary alone is insufficient to derive the
exact movement rule for general state.

### Action 6 (click)

48/51 clicks only ticked the step counter. The other 3 clicks (steps
116, 147, 165) drew a non-trivial 26..36-cell pattern at the click
location - looks like the game "stamps" the sprite onto the click site,
but only sometimes. Modelling this from the summary is risky.

## Simulator v1 behaviour

1. Always advance row-63 step counter.
2. Optional sprite-translate hook (currently inert because bbox
   heuristic fails on dense states - this is intentional; including a
   bad sprite predictor regresses pixel-match).
3. `reward_class = 1`, `done = False` always.

## v1 result on full 200-tuple set

```
state_exact_pct: 34.0   pixel_match_pct: 97.82
reward_acc_pct: 100.0   done_acc_pct: 100.0
by_action: 3 -> 6.25 % exact, 4 -> 1.96 %, 6 -> 94.12 %, 7 -> 32.0 %
```

Exact-match comes entirely from the cases where the only change was
the counter tick. Pixel-match is ~98 % because the sprite swap touches
only ~47/4096 = 1.1 % of the grid.

## v2 plan

1. **Detect sprite by template**, not bbox. Cache the 5x5 sprite found
   in the initial state. For actions 3 and 4, swap the sprite cells
   with the equally-sized block 6 columns to the right (action 4) or
   left (action 3), preserving non-sprite cells in both blocks.
2. **Action 7 = repeat last action**: probably toggles the same swap
   that action 3 or 4 just did. Add a tiny stateful trick or look at
   the current sprite position and snap it back.
3. **Action 6**: if (x,y) lands on the sprite, behave like action 4;
   otherwise tick counter only. (Hypothesis to test.)
4. Bring an analytic disambiguator: compute the modular distance between
   sprite columns and frame and pick the direction that conserves the
   number of `5`/`9`/`11` pixels.
5. If sprite detection still misclassifies, fall back to "no playfield
   change" - never inject a wrong translation, because pixel-match
   penalizes false positives more than misses.
