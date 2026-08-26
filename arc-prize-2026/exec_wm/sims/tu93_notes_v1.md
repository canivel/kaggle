# tu93 sim v1 notes

## Game in one sentence

A 6x6 grid maze with a 3x3 sprite (color 9 body + color 4 arrow) navigating
between cells; actions 1/2/3/4 = UP/DOWN/LEFT/RIGHT; row 63 is a 64-cell
countdown timer; goal is a 3x3 color-14 block at the bottom-right but is
never reached in the 200 random-exploration tuples (level stays 0).

## Invariants used

1. **Unique sprite anchor (color 4)**: every state has exactly one cell of
   color 4 -- the "arrow" inside the 3x3 sprite block. The 3x3 window
   around it contains exactly 1 four and 8 nines, giving lossless sprite
   detection.
2. **Connector colour determines move success**: the 3x3 strip between
   adjacent cells is either color 2 (corridor -> move succeeds) or color 5
   (outside the maze -> NOOP). Verified on 95/95 moves vs 105/105 NOOPs.
3. **Action -> facing mapping**: 1=UP arrow at (0,1), 2=DOWN arrow at
   (2,1), 3=LEFT arrow at (1,0), 4=RIGHT arrow at (1,2). On NOOP the
   facing does NOT change. Verified on all 200 tuples.
4. **Deterministic timer pattern**: row 63 has a 50-action cycle. The
   set of pre-action 6-counts triggering a "double-tick" decrement is
   `{3, 8, 12, 17, 22, 26, 31, 35, 40, 44, 49, 54, 58, 63}`. Every other
   pre-count tics by 1. Each value seen exactly 4 times in the 200
   samples (= 4 full cycles), so the table is complete.

## v1 result on full 200-tuple set

```json
{
  "game": "tu93", "split": "all", "n": 200, "errors": 0,
  "state_exact_pct": 100.0,
  "pixel_match_pct": 100.0,
  "reward_acc_pct": 100.0,
  "done_acc_pct": 100.0,
  "by_action": {
    "2": {"n": 49, "exact_pct": 100.0, "pixel_pct": 100.0, "errors": 0},
    "1": {"n": 50, "exact_pct": 100.0, "pixel_pct": 100.0, "errors": 0},
    "3": {"n": 45, "exact_pct": 100.0, "pixel_pct": 100.0, "errors": 0},
    "4": {"n": 56, "exact_pct": 100.0, "pixel_pct": 100.0, "errors": 0}
  }
}
```

## v2 plan (if anything were to fail in extended testing)

1. **Goal reaching**: random play never landed at (46-48, 45-47). If the
   sprite ever sits adjacent and moves into the 14-block we may need to
   emit `done=True` and/or `level+=1`. Probe this with a targeted policy.
2. **Timer reset on n6==0**: the in-game observation resets row 63 to
   all-6s between tuples whose `state_t1` reaches all-zeros and the next
   tuple's `state_t`. Currently simulate(state_t with n6=0) is never
   called by validate_sim, but a planner using simulate() chained more
   than 50 deep will need an extra refresh tick after n6 hits 0.
3. **(x, y) clicks**: not used in tu93 (always 0,0). If a later level
   exposes click actions 5/6/7, capture fresh tuples first.

## Honest assessment

100% is real because the four invariants above are crisp and independent:
sprite detection is lossless, connector detection is binary, action-to-
facing is mechanical, and the timer table was complete after 200 samples
(4 cycles x 50 actions). Both action 6/7 and goal mechanics are absent
from the data, so any extension targeting level-up will require new
observations rather than analysis of these 200 tuples.
