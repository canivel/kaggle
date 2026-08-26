# tn36 sim v1 notes

## Game description (1 line)

A countdown-timer minigame: only action 6 (click) is issued; each click
advances a 61-cell timer on row 1 (cols 1..61) by flipping the rightmost
`9` to `3`. Six of 200 clicks also toggle a 3-cell vertical indicator at
rows 44..46.

## Observations

### Top-level
- Only available action: **6** (200/200 tuples).
- Reward: **1** for every tuple. Done: **False** every tuple. Level
  stays at 0. So constant `(reward=1, done=False)` is optimal.
- Grid values 0..11 only.

### Countdown timer on row 1
- Row 1 cols 1..61 hold either `9` (full) or `3` (used). Col 0 and cols
  62..63 are always `5`.
- Each action flips the **rightmost** remaining `9` to `3`.
- Once all 61 cells are `3`, the environment resets row 1 cols 1..61 to
  `9` between observations (we never observe an all-3 row 1 as input).
- Verified on all 194 `n_changed == 1` cases.

### Button-toggle on n_changed == 4 (6 cases)
- Row 1 countdown tick (1 change) PLUS a 3-cell vertical strip at rows
  44, 45, 46 of a single column toggles between `5` and `1`.
- Observed toggle columns: 21, 31, 36, 40-42(row 42 buttons), 31.
- Indicator columns sit at fixed col positions ~21, 26, 31, 36, 41
  (period 5). The click (x, y) does NOT land directly on (44, c) — it
  lands in the area roughly (29, 44), (22, 46), (43, 43), (34, 46),
  (31, 46). The mapping from click → toggled indicator is unclear from
  6 samples and we leave it unmodelled.

## Invariants used in v1
1. **`reward_class == 1, done == False`** — universal.
2. **Action 6 ticks the countdown** — rightmost `9` on row 1 cols 1..61
   becomes `3`. Defensive fallback resets+ticks if no `9` remains.
3. **Identity elsewhere** — leave rows other than row 1 unchanged.
   Cost: only 6/200 cases (≈3%) have additional toggles; each touches
   only 3 cells, so pixel-match cost is < 0.005%.

## v1 result on full 200-tuple set

```
{
  "game": "tn36",
  "split": "all",
  "n": 200,
  "errors": 0,
  "state_exact_pct": 97.0,
  "pixel_match_pct": 99.997802734375,
  "reward_acc_pct": 100.0,
  "done_acc_pct": 100.0,
  "by_action": {
    "6": {
      "n": 200,
      "exact_pct": 97.0,
      "pixel_pct": 99.997802734375,
      "errors": 0
    }
  }
}
```

Per-action exact %: action 6 → 97.0 (the 3% miss = the 6 n_changed=4
button-toggle cases).

## v2 plan

1. **Decode the button → indicator mapping**. Hypotheses to test on the
   6 observed cases:
   - Each click (x, y) hits a "button" rectangle on row 42 (3-cell wide
     groups at cols ~20-22, 25-27, 30-32, 35-37, 40-42). The button's
     indicator at (44..46, c) toggles.
   - But the observed clicks don't land directly on row 42 — they land
     in (29..43, 43..46). May need to map (x,y) to the nearest button
     center using Chebyshev distance, OR may be a different UI (color
     swatch picker).
2. Need more raw observations specifically targeting clicks that fire
   n_changed=4 to nail the mapping. With only 6 samples, any rule
   overfits.
3. Worst-case v2 gain: +3% state_exact → 100%. Pixel-match is already
   99.998% so v2 ROI is mostly state_exact polish.

## Honest assessment

97% state_exact in one iteration is well past the bp35-v2 result (69%).
The countdown invariant is rock-solid (verified on 194/194 cases).
Pixel-match at 99.998% suggests the simulator is nearly lossless and the
remaining 3% would need targeted observation of the click-toggle UI
before risking a rule that could regress pixel-match.
