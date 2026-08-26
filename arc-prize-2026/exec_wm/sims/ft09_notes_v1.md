# ft09 sim v1 notes

## Observations

### Top-level
- Only **action 6 (click)** is available. action_distribution = {6: 200}.
- 200 random-exploration tuples.
- reward_distribution = {0: 185, 1: 15}. done is False everywhere. Level stays at 0.
- n_changed distribution is bimodal and pristine: 185 NOOPs with n_changed=0,
  and 15 successful clicks with n_changed=38 (36 tile cells + 2 counter cells).

### The 3x3 tile grid (the playfield)
A 3x3 grid of 6x6 tiles centered at:
- rows in {(36,41), (44,49), (52,57)}
- cols in {(36,41), (44,49), (52,57)}

Eight outer tiles are uniformly colour 8 or 9. The centre tile (1,1) has a
static "X" cross pattern of values {0, 2, 8} and is inert.

15/15 successful clicks landed inside an outer tile. The whole 6x6 tile
flipped uniformly between 8 and 9 — direction determined by the value at
the click cell (8 → 9 or 9 → 8). 2/2 inside-NOOPs landed in the centre
tile (1,1). 183/200 NOOPs landed outside the 3x3 tile region. **No outer
tile click ever produced a NOOP.**

### Row 63 counter
Row 63 is a horizontal countdown counter, initialised to 64 cells of
value 12. Every successful flip turns the two RIGHTMOST remaining 12-cells
into 11. Counter only ever depletes (no 11 → 12 transitions). At step 200,
26 of 64 cells have been used (≈15 flips × 2).

## Invariants used
1. **Fixed tile grid**: clicks map deterministically to a (tile_row, tile_col)
   via fixed bounds. Outside → identity. Centre tile → identity.
2. **Uniform tile flip**: the value at (y, x) tells us the colour of the
   tile, swap to the other (8 ↔ 9). Stamp uniformly across the 6x6 patch.
3. **Counter consumption rule**: consume exactly two rightmost 12s per flip.

## v1 result (validate_sim.py output, verbatim)

```json
{
  "game": "ft09",
  "split": "all",
  "n": 200,
  "errors": 0,
  "state_exact_pct": 100.0,
  "pixel_match_pct": 100.0,
  "reward_acc_pct": 100.0,
  "done_acc_pct": 100.0,
  "by_action": {
    "6": {
      "n": 200,
      "exact_pct": 100.0,
      "pixel_pct": 100.0,
      "errors": 0
    }
  }
}
```

## v2 plan

The simulator already perfectly predicts the 200 training tuples. v2 would
only be needed if held-out (test split) tuples reveal hidden behaviours we
have not observed. Candidates worth checking with more data:

1. **What happens when the counter is exhausted?** No tuple here reaches
   the empty-counter state. The game presumably advances to a new level or
   transitions `done=True`. v2 would need a counter-empty test case.
2. **Centre tile activation**: tile (1,1) might become active once the
   outer tiles reach a specific colour pattern (e.g. all 8s). v2 could
   look for that puzzle goal once more data is available.
3. **Level-up / done semantics**: this game presumably has a win/lose
   trigger we have not seen. Run a longer rollout (300+ steps) to look
   for level transitions.

The two clean invariants (fixed tile grid + uniform 8↔9 flip + counter
deplete-2) are crisp and almost certainly generalise. This is not
curve-fitting on 200 tuples — the playfield is genuinely this simple.
