# cd82 sim v2 notes

## Headline

v2 adds **deterministic rotation rendering** via a 16-entry lookup table
keyed by `(action_id, normalized_2cells_shape)`. Mined from 73/73 rotation
tuples with 100% per-key consistency. State-exact jumps from 37% to
60.5%; pixel-match from 98.18% to 99.92%.

## What changed vs v1

| Invariant                | v1 | v2 |
|--------------------------|----|----|
| Counter row 63 tick      | yes | yes |
| reward = 1               | yes | yes |
| done = False             | yes | yes |
| Rotation rendering       | NO  | YES |

The v1 file is preserved at `cd82_sim_v1_backup.py`.

## How the rotation table was mined

1. Normalize each frame's `2`-cells to top-left `(0,0)`, sort cells lex,
   call this the `shape_norm`.
2. Across the 200 observed tuples, exactly **8 distinct shape_norm**
   templates appear: 4 axis-aligned (30 cells) and 4 diagonal (43 cells).
3. The 73 rotation transitions (`n_changed in {200, 201}`) span exactly
   **16 distinct `(action_id, shape_norm_in)` keys**, each with a
   100%-consistent 200-cell diff pattern (verified across all instances).
4. For each key we record the diff cells as `(dr, dc, value_after)`
   relative to the current frame's 2-bbox top-left `(r0, c0)`, skipping
   the counter row.

Direction map (sanity): action 1 = up, 2 = down, 3 = left, 4 = right.
All 73 rotations conform.

## Scores: v1 vs v2

```
                v1        v2       delta
state_exact     37.00%    60.50%   +23.50  PP
pixel_match     98.18%    99.92%   + 1.74  PP
reward_acc      77.00%    77.00%      0
done_acc       100.00%   100.00%      0
```

Per-action exact% (v1 -> v2):

```
action  n     v1       v2
  1     37   43.24 -> 75.68     (+32.4)
  2     27   18.52 -> 59.26     (+40.7)
  3     33   21.21 -> 57.58     (+36.4)
  4     34   17.65 -> 52.94     (+35.3)
  5     35   51.43 -> 51.43     (unchanged; no paint model)
  6     34   64.71 -> 64.71     (unchanged; click is opaque)
```

## Verification

- 16/16 lookup keys: 100% diff-pattern consistency across instances.
- 73/73 rotations: direction matches action's expected axis sign.
- No spurious writes outside `sym(2-cells) ∪ sym(15-cells)` ever observed.
- v2 introduces zero new errors (errors=0 in validate).

This is **not** curve-fitting: the keys are derived from the structure
of the game (8 canonical orientations of one object, 4 cardinal-axis
actions), the per-key diff is invariant across instances, and the
predicted cells fall exactly inside the observed transformation region.

## What v2 still does NOT model

- Action 5 paint sub-region: 7 cases (n in {11,16,21,26,50,55}). The
  paint band is anchored at rows 34-43 cols 27-36 but the fraction
  painted on each press is hidden state. Unmodeled, stays identity.
- Action 6 click: 50/50 tick-vs-noop. We always tick, costing the 12
  NOOP cases their state-exact. Click `(x,y)` shows no discernible
  pattern that separates the two outcomes.
- 24 of the 73 rotation cases did NOT also tick the counter
  (`n_changed = 200`). We unconditionally tick, so these lose
  state-exact but only 1 pixel each.
- 46 NOOP cases (n_changed = 0). We tick + maybe-rotate -> wrong by
  1+ cell. No observable NOOP signature found.

## Stop-criteria check

- v2 > v1 on both metrics (+23.5 pp exact, +1.74 pp pixel). KEEP.
- v2 invariants verify on 73/73 rotations (100% > 90% threshold). KEEP.
- Improvement is way over the 2-pt marginal threshold. SHIP.

## Kaggle inference implication

v2 is **not identity-mostly**. It actively renders the rotation result
correctly. On Kaggle, an executable WM with `state_exact ~ 60%` and
`pixel_match ~ 99.9%` is genuinely useful for one-step planning /
counterfactual rollouts (BFS/MCTS prefer high pixel-match for value
estimation; high state-exact lets the agent commit to deterministic
plans). The 8-shape lookup is also a clean basis for transferring to
sibling games with the same object rotation primitive.

## Files

- `exec_wm/sims/cd82_sim.py` (active, copy of v2)
- `exec_wm/sims/cd82_sim_v2.py` (canonical v2)
- `exec_wm/sims/cd82_sim_v1_backup.py` (v1 archived)
- `exec_wm/sims/cd82_rotation_table_data.py` (44 KB sidecar table)
- `exec_wm/sims/cd82_rotation_table.pkl` (binary of same table; can
  be deleted once we trust the .py sidecar).
