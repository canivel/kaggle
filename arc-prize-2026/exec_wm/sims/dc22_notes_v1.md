# dc22 sim v1 notes

## Game model (from 200 observations)

A 2x2 block of colour 14 (the "player") slides on a small lower playfield
(rows 38-43, cols 8-13) of colour 2 (path) embedded in a 64x64 grid of
colour 4 (wall). Above row 38 lies an L-shaped narrow corridor of 9-walls
that reconfigures whenever the player traverses it. A 4x4 region of colour
13 sits north of the playfield (static). A 2x2 region of colour 11 sits
further north (static). Row 63 is a left-to-right counter that flips a 0
to a 3 on roughly every other action.

## Invariants used

1. **Player anchor**: `np.where(state == 14)` returns exactly 4 cells in
   every observed state (200/200) - the 2x2 player block. Top-left =
   `(ys.min(), xs.min())`.
2. **Action -> delta** in the lower playfield (verified deterministic):
   - Action 1 (UP) : (r-2, c)  - deterministic from rows 40, 42
   - Action 2 (DOWN) : (r+2, c) - deterministic
   - Action 3 (LEFT) : (r, c-2) - deterministic
   - Action 4 (RIGHT) : (r, c+2) - deterministic
3. **Bounds (lower playfield)**: col in {8, 10, 12}, row in {38, 40, 42}.
   LEFT NOOP at col 8 (15/15); RIGHT NOOP at col 12 (11/11);
   DOWN NOOP at row 42 (1/1 simple). UP from row 38 is mixed
   (4/13 succeed) - we conservatively NOOP from row 38 (gain 9/13).
4. **Reward** = 1 iff state changed, else 0 (155+45 = 200/200 matched).
5. **Done = False**, level = 0 (200/200).

## Counter (row 63) - the unmodeled half

Each tick: leftmost non-3 cell becomes 3. Empirically ticks on every
other action regardless of action_id, but the parity flips after a
special action-6 click event at step ~56 (n_changed=129, full UI repaint).
The best parity heuristic available from state alone is
`tick iff count_of_3s_in_row_63 is even and < 64`, which gives 51%
counter accuracy - all that's recoverable without external hidden state.

## v1 result on full 200-tuple set

```json
{
  "game": "dc22",
  "split": "all",
  "n": 200,
  "errors": 0,
  "state_exact_pct": 48.5,
  "pixel_match_pct": 99.955810546875,
  "reward_acc_pct": 80.5,
  "done_acc_pct": 100.0,
  "by_action": {
    "1": {"n": 43, "exact_pct": 37.21, "pixel_pct": 99.97},
    "2": {"n": 32, "exact_pct": 43.75, "pixel_pct": 99.98},
    "3": {"n": 41, "exact_pct": 43.90, "pixel_pct": 99.99},
    "4": {"n": 36, "exact_pct": 61.11, "pixel_pct": 99.99},
    "6": {"n": 48, "exact_pct": 56.25, "pixel_pct": 99.88}
  }
}
```

## Error attribution (LEFT and UP)

- LEFT errors: 23/23 are counter-tick mispredictions only. Block move
  is 100% correct.
- UP errors: 23 counter only, 3 block only (from row 38 cases that
  successfully moved up to row 36), 1 mixed.

So almost all per-action error is the counter coin-flip, which is the
fundamental ceiling without hidden state.

## What we didnt model

1. **Counter ticking parity is hidden state.** Searched all non-row-63
   non-player cells for cells that distinguish pre-tick from pre-no-tick
   states with the same row-63 count - found none. So 51% counter
   prediction is the natural cap from the state alone.
2. **UP from row 38 -> row 36 / 34**: succeeds in 4/13 cases. Above
   row 38 the 9-walls reconfigure (verified - initial walls are a
   checkerboard, after entry they collapse into solid bars). We do not
   model this morphing - leaving NOOP is the higher-EV bet (69%).
3. **Action 6 click outcome**: the rare (1/48) n_changed=129 click is
   probably a UI repaint triggered by clicking a specific button (the
   13 or 11 sprite up north). We do not model it.

## v2 plan

1. **Counter as Bernoulli(0.5) ALWAYS-TICK variant**: if I always tick,
   I get 49% counter accuracy (basically same as 51% but inverted on
   which cases). Result roughly identical. Not worth it.
2. **UP from row 38 deterministic when possible**: track the 9-wall
   pattern in rows 34-37 and only emit UP if the cell at (r-2, c) and
   (r-2, c+1) are PATH-coloured (2 or 9 cleared). The 9-wall morphing
   on entry is the part we'd need to model - costs +~6% (4/200 -> 12/200
   or so if we add row-38 deterministic UP plus a guess at the morphing).
3. **Action 6 click on 13 or 11 sprite**: if click (x,y) lies inside the
   4x4 block of 13s at (30,18)-(33,21), it might trigger the n_changed=129
   event. Test on the single example.

Current 48.5% state_exact / 99.96% pixel is dominated by the counter
ceiling. v2 should focus on UP-from-row-38 first (3% upside) and the
counter ceiling is unlikely to be breakable without temporal info.
