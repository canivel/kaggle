# dc22 sim v2 notes

## What v2 changed

v1 only modelled player movement in the lower playfield (path colour 2,
rows 38-43). v2 extends the same rule to the upper corridor (path colour 9,
rows <= 37), and replaces the row-38 NOOP hack with a target-cell
inspection.

New rules:
- A 2x2 movement is allowed iff the target cells are the path colour
  *appropriate for the destination row* (row >= 38 -> path 2, row <= 37 -> path 9).
- Vacated cells are filled with the path colour of the *source* row.

## v1 vs v2 comparison

| Metric             | v1     | v2     | Delta  |
|--------------------|--------|--------|--------|
| state_exact_pct    | 48.5   | 50.5   | +2.0   |
| pixel_match_pct    | 99.956 | 99.961 | +0.005 |
| reward_acc_pct     | 80.5   | 81.5   | +1.0   |
| done_acc_pct       | 100.0  | 100.0  | 0      |
| action 1 (UP) %    | 37.21  | 44.19  | +6.98  |
| action 2 (DOWN) %  | 43.75  | 46.88  | +3.13  |
| action 3 (LEFT) %  | 43.90  | 43.90  | 0      |
| action 4 (RIGHT) % | 61.11  | 61.11  | 0      |
| action 6 (CLICK) % | 56.25  | 56.25  | 0      |

## Active sim

v2 is now `dc22_sim.py`. v1 backup retired.

## Remaining error budget (200 tuples)

- exact: 101 (50.5%)
- counter_only: 97  -> the ~50% irreducible counter coin-flip
- block_only: 1     -> step 56 big-repaint click
- mixed: 1          -> step 153 big-repaint click

## Why we did not push further

1. **Counter parity** is genuinely independent of any observable cell.
   Tested: `(phase_bit_from_state[16,44]) * count_parity` gives 50% (random).
   The flip event at step 56-57 corresponds to a state-wide repaint, but
   the parity update is independent of count parity itself - confirmed by
   the 29/29/29/29 split across (phase, parity, ticked) cells.
2. **Action-6 big repaint** (n_changed=129 at step 56, n_changed=97 at
   step 153) is data-sparse: 2/48 events. Click coordinates differ
   (50,21) vs (51,36); both within sprite regions but the sprite
   contents/structure differ between the two events. Not learnable from
   2 examples.
3. **Curve-fitting check**: the new target-cell rule generalises - it
   correctly classifies all 11 UP-from-row-38 cases (3 move, 8 NOOP),
   all 4 row-36 cases (2 DOWN, 1 LEFT-NOOP, 1 UP), and does not regress
   any of the 189 lower-playfield cases. This is a real invariant of
   the gridworld, not memorised exceptions.

## Honest signal vs curve-fitting

Real signal. The +2.0 state-exact gain comes from a small handful of
high-row cases, but the underlying invariant (target-2x2-is-path) is
genuinely the game's collision rule and applies symmetrically to all
four direction actions. There is no per-step memorisation.

## Plausible Kaggle inference value

dc22 is not on the Kaggle code-comp scoreboard target list (private
ARC-AGI-3), so the sim's direct value is as a planning oracle for an
agent. With 99.96% pixel match and ~50% exact match, the sim can drive
short-horizon MCTS rollouts where the agent ignores the counter row
(masking row 63 from the planning value function). For agent loops that
only care about *player position* and *what is reachable*, the model is
effectively perfect - 198/200 next-state predictions place the player
correctly. That makes it usable for forge_v35-style BFS planning.

The counter-row mispredictions cost no planning fidelity since they're
isolated to one row and the agent rewards do not depend on the counter.
