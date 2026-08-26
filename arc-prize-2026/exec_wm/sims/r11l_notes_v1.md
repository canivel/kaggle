# r11l sim v1 notes

## Game description
Single-action click game. Only action 6, reward_class=1, done=False, level=0
on every observed tuple. Grid 64x64, values 0..15. Column 0 is a vertical
step-counter (cells flip 0->5 from row 0 downward). The rest of the grid
is a playfield with background color 2 and interior color 5 containing
sprite-like objects (colors 1, 3, 6, 15).

## Action distribution
- 6 (click): 200/200

## n_changed distribution (modal buckets)
- n=1: 43 tuples (pure counter tick)
- n=2: 4 tuples (counter ticks twice)
- n in [25, 153]: 153 tuples (counter tick + playfield change because the
  click landed on an active interior object)

## Invariants used in v1
- INV-1 "counter tick": topmost 0-valued cell in column 0 becomes 5 every
  transition. Holds on 43/43 n=1 cases and 151/153 large-n cases.
- INV-2 "constant reward/done": reward_class=1, done=False, every tuple.

## v1 strategy
- Always advance col-0 counter.
- Leave playfield untouched (we cannot recover the per-frame piece dynamics
  from a single observation -- the click->effect mapping depends on which
  interior sprite is currently "live").
- This gives 100% on the n=1 bucket and high pixel match (~99%) on the
  large-n bucket because the playfield change typically touches <150 of
  4096 cells.

## v1 result
```
{
  "game": "r11l",
  "split": "all",
  "n": 200,
  "errors": 0,
  "state_exact_pct": 21.5,
  "pixel_match_pct": 98.13720703125,
  "reward_acc_pct": 100.0,
  "done_acc_pct": 100.0,
  "by_action": {
    "6": {
      "n": 200,
      "exact_pct": 21.5,
      "pixel_pct": 98.13720703125,
      "errors": 0
    }
  }
}
```

## What was lost vs. ceiling
- 43 n=1 cases captured (100% exact for that bucket).
- 4 n=2 cases missed (need an extra counter tick rule -- likely "click in
  border-color-2 region near top-left edge" but step 172 has y=2 with click
  on color 5, so the rule is not just border-based).
- 153 large-n cases missed exact -- need piece dynamics. We could try
  detecting the active piece by anchor color (15 is rare and clustered,
  could be the "lead" sprite) and predicting a simple translation when the
  click lands on or near it.

## v2 plan
1. **n_changed=2 rule**: clicks at (x, y) with x in {1,3,37,62}, y in {0,1,2}.
   Three of four have y<=2 -- candidate rule "if y < 3, tick twice". Force-test
   on every tuple before shipping (only 4 positive cases makes this fragile).
2. **Active-piece detection**: scan the playfield for clusters of color 15.
   In exemplar 0 we see two 15-shapes (a chevron near rows 18-24, a diamond
   near rows 45-49). If a click lands inside a bbox containing 15s, predict
   the bbox shifts by (dx, dy) toward the click. Test before shipping.
3. **Trail rule**: many large-n diffs are "5->0" cells along a path -- the
   active piece leaves a trail of erased cells. Worth measuring whether a
   straight-line model from previous-position to click-target reproduces
   the trail.
4. Wrap behaviour: col 0 fill==64 -> next col 0 all zero. v1 leaves the
   full state intact (no observed transition starts from a full-col-0
   state, so no points lost in this sample).

## Honest assessment
v1 (~22% exact, ~98% pixel) is the safe pixel-match floor for a game where
the click-to-piece-effect mapping depends on hidden state. To break past
50% exact we need to identify the active piece sprite and predict its
motion -- that requires more raw observations or actually solving the game
mechanic. The counter tick alone is rock-solid signal.
