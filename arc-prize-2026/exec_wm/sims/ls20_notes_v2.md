# ls20 sim v2 notes

## Delta vs v1

v1 left 4/200 mismatches: the "counter empty" events at steps 42, 85, 128,
171. v2 models them with one additional rule that triggers iff the counter
band (rows 61-62, cols 13..54) is entirely color 3.

## New invariants (verified 4/4)

5. **Score-margin layout.** Rows 61-62 cols 56-57, 59-60, 62-63 each hold
   an (8,8)/(8,8) score pair, separated by color-5 dividers at cols 55,
   58, 61.

6. **Counter-empty trigger.** Iff `state[61, 13:55].all() == 3` and same
   for row 62, the action consumes the RIGHTMOST remaining 8-pair
   (right-to-left order: 62-63, 59-60, 56-57) flipping it to (3,3)/(3,3).

7. **Refill + respawn (n>=2 score pairs remain at start of step).**
   - Counter band rows 61-62 cols 13..54 -> 11.
   - Sprite teleports to (45, 34); previous sprite is erased with the
     usual per-row lane-line restore.
   - The action's normal move is SKIPPED; the v1 tick is also skipped
     (counter just refilled).

8. **Final pair (n==1 score pair at start of step).** Pair is consumed,
   no refill, no respawn. The normal sprite move is then attempted (in
   step 128 the destination was all walls so it NOOPed). v1 tick is
   skipped (counter is already all-3).

## Validation result

```json
{
  "game": "ls20", "n": 200, "errors": 0,
  "state_exact_pct": 100.0,
  "pixel_match_pct": 100.0,
  "reward_acc_pct": 100.0,
  "done_acc_pct": 100.0,
  "by_action": {
    "1": {"n": 50, "exact_pct": 100.0, "pixel_pct": 100.0},
    "2": {"n": 49, "exact_pct": 100.0, "pixel_pct": 100.0},
    "3": {"n": 45, "exact_pct": 100.0, "pixel_pct": 100.0},
    "4": {"n": 56, "exact_pct": 100.0, "pixel_pct": 100.0}
  }
}
```

## Signal vs curve-fitting

The new rule fires on 4/200 = 2% of tuples and converts all 4 to exact
match. Trigger condition (counter all-3) is observable from the input
state and has obvious in-game semantics (round timer expired -> respawn).
Refill-vs-final-pair branch is decided by counting remaining 8-pairs,
also fully observable. Respawn point (45,34) matches the initial sprite
spawn (step 0 `state_t`), so it is consistent with a documented game
mechanic rather than a memorised constant.

Risk: only 3 of the 4 events test the refill+respawn branch (step 128
tests the final-pair branch). If a future level changes the respawn
position the rule will mispredict, but the counter+score logic itself
is on solid ground.

## Active sim

`ls20_sim.py` now mirrors `ls20_sim_v2.py`. v1 is preserved at
`ls20_notes_v1.md`; v2 source lives at `ls20_sim_v2.py`.

## Inference utility

ls20 is NOT identity-mostly: every action ticks the counter, sprite
moves on ~all non-trigger steps, and the trigger event materially
changes 138 cells. A planner using this sim can credit-assign across
respawns and avoid wasting actions when the round is about to roll. The
sim is a useful forward model.
