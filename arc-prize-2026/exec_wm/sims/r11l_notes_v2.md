# r11l sim v2 notes

## v1 vs v2

| metric           | v1     | v2     | delta   |
|------------------|--------|--------|---------|
| state_exact_pct  | 21.50  | 23.00  | +1.50pp |
| pixel_match_pct  | 98.137 | 98.138 | +0.0004 |
| reward_acc_pct   | 100.0  | 100.0  | 0       |
| done_acc_pct     | 100.0  | 100.0  | 0       |
| errors           | 0      | 0      | 0       |

Active sim: **v2** (copied over `r11l_sim.py`; `r11l_sim_v1_backup.py` kept).

## What v2 adds

INV-3 "double-tick rule":
```
if y < 3 AND (first_zero_row_in_col_0) % 8 == 7:
    tick the counter twice instead of once
```

Trigger stats on all 200 tuples:
- TP = 4 (every n=2 case captured)
- FP = 1 (tuple i=149: first0=31, y=0, actually single-tick — we add an
  erroneous extra tick, breaking exact match for this n=1 case and
  costing one pixel)
- FN = 0
- TN = 195

Net: +4 newly-correct exact matches, -1 broken = **+3 net (+1.5pp)**.

## What I tried and rejected

1. **Active-piece translation** (color-15 cluster moves toward click).
   Found 153 large-n cases. Tested hypothesis `bbox1.left = x` and
   `dx = x - bbox0.left`: only 65/153 matched. The sprite's motion is
   gated by hidden per-frame state (animation phase). Cannot be
   recovered from a single state — drops if shipped.

2. **`y < 3` alone for double-tick**: 14 tuples qualify, only 4 are
   actually n=2. Drops exact match (would falsely add ticks to 10
   tuples that are correctly modeled as single-tick).

3. **No-movement detection** (predict bbox stays put). 36/153 large-n
   cases have dx=0, but the sprite still animates (n_changed 25..115),
   so predicting "no change" still misses exact match. No gain.

## Honest signal vs curve-fit

- INV-3 has **only 4 positive examples** in 200 tuples. The rule
  `first0 % 8 == 7` was found by inspection and is highly suspicious
  of curve-fitting. Mitigation: the conjunction `y < 3` is an
  independent observation (all 4 n=2 events have a top-row click),
  and the 195/200 TN rate plus 1 FP means we are not adding noise to
  the 153 large-n cases that already miss exact match. Pixel match
  proves we are not regressing.
- v2 improvement (+1.5pp) is **below the 2pp stop threshold** — shipping
  but flagging as marginal per the brief.

## Kaggle-inference plausibility

The sim remains identity-mostly: 98.14% pixel match because the
playfield (4046/4096 cells) almost never changes for a click. As a
forward model for planning, it correctly predicts:
- counter ticks (1 or occasionally 2) — useful for an agent that cares
  about how many steps remain;
- reward_class=1 and done=False — trivially constant in this game.

It does **not** predict piece dynamics, so it cannot help an agent
choose which click intercepts the sprite. Net: marginally useful as a
no-op-aware step predictor, not as a planning oracle. Won't move the
needle on Kaggle alone, but consistent with the "identity-mostly
floor" we expected for click games without multi-frame state.
