# ka59 sim v1 notes

## Game description (from observations)

A 2D puzzle game with a 3x3 player sprite (color 14, hole 0 at centre)
that translates by 3 cells per action through a playfield bounded by
walls of color 15 and outer background color 2. There is a 3x3 goal
sprite (color 14 with centre = 5) that is static. Row 63 is a depleting
step counter that starts as all 4s and ticks one cell from the right to 0
on each "active" frame. Five actions are available: 1 (UP), 2 (DOWN),
3 (LEFT), 4 (RIGHT), 6 (click). Movement is a clean 3x3 swap with the
adjacent block in the move direction. NOOP iff target 3x3 is uniformly
outer-bg (2) or uniformly wall (15). reward_class = 0 iff state unchanged.

## Invariants used (v1)

1. **Sprite anchor**: in 97% of observed states the unique 0-cell in rows
   0..62 IS the player sprite's centre (verified 111/114 18-change
   transitions). When 0 or >1 such cells exist, we fall back to identity.
2. **Move = 3x3 swap**: actions 1/2/3/4 swap the sprite 3x3 with the 3x3
   block at (drow, dcol) in {(-3,0),(+3,0),(0,-3),(0,+3)}. Verified on
   all 114 observed 18-change transitions.
3. **NOOP rule**: target 3x3 is uniform color 2 (outer bg) or 15 (wall).
   Verified on all 13 observed action-1..4 NOOP cases.
4. **Counter tick**: depleting from the right (rightmost 4 -> 0). Tick on
   any rewarded action (always tick for 1..4-moves and for action 6).

## v1 result on full 200-tuple set

```json
{
  "game": "ka59",
  "split": "all",
  "n": 200,
  "errors": 0,
  "state_exact_pct": 54.0,
  "pixel_match_pct": 99.977,
  "reward_acc_pct": 76.0,
  "done_acc_pct": 100.0,
  "by_action": {
    "1": {"n": 43, "exact_pct": 53.49, "pixel_pct": 99.986},
    "2": {"n": 32, "exact_pct": 50.00, "pixel_pct": 99.984},
    "3": {"n": 41, "exact_pct": 58.54, "pixel_pct": 99.957},
    "4": {"n": 36, "exact_pct": 47.22, "pixel_pct": 99.966},
    "6": {"n": 48, "exact_pct": 58.33, "pixel_pct": 99.990}
  }
}
```

## Where the 46% state-exact miss comes from

| failure                                             | count |
|-----------------------------------------------------|-------|
| Counter mispredicted (1-cell diff on row 63)        | 75    |
| Sprite "absorbed" by goal (4-zero edge state)       | ~5    |
| Action-6 surprise stamping (3..22 cells changed)    | ~6    |
| Other rare animation frames                         | ~6    |

The dominant miss is the **counter tick**: even after a successful move
the counter ticks only ~60% of the time. Whether it ticks depends on
hidden state we cannot recover from a single frame (the gap pattern is
2,1,2,2,1,2,1,2,1,2 -- no parity / sprite-position / counter-state
indicator fits cleanly).

## v2 plan

1. **Action-6 splitter**: investigate whether the click (x,y) lands on a
   specific colour or coordinates correlated with the 20 NOOP / 28 tick
   outcomes. If a deterministic predicate emerges (e.g. y < some
   threshold), branch action 6 between identity and counter-tick.
2. **Adjacent-goal transient**: the 4-zero "sprite next to goal" state
   (~3 distinct configurations) -- collect those and either model the
   pre-merge animation or special-case identity.
3. **Counter tick**: requires hidden-state tracking. Without prev-action
   info passed to `simulate`, this caps state_exact around 60% even with
   perfect playfield prediction. We could detect "the counter just
   ticked" from a *pair* of frames but the harness signature is single-
   frame so the gain is irreducible here.
4. **Large-diff action-6 events**: a handful of click events change
   3..22 cells (likely stamping a sprite shape at the click site).
   Trigger condition unclear from 48 samples.

## Honest assessment

Three crisp invariants captured the playfield dynamics cleanly:
- Unique-0 anchor for sprite centre
- 3x3 cardinal swap rule
- All-2/all-15 NOOP detection

Pixel-match at 99.98% confirms the playfield model is essentially
correct. The 46% exact-match gap is mostly the irreducible counter
noise plus a few rare animation frames. v2 might recover +5..+10
points via the click splitter and 4-zero special case.
