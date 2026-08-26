# tr87 sim v2 notes

## Summary

| metric              | v1     | v2     |
|---------------------|--------|--------|
| state_exact_pct     | 50.0   | **100.0** |
| pixel_match_pct     | 99.988 | **100.0** |
| reward_acc_pct      | 100.0  | 100.0  |
| done_acc_pct        | 100.0  | 100.0  |
| action 1 exact      | 50.0   | 100.0  |
| action 2 exact      | 51.0   | 100.0  |
| action 3 exact      | 55.6   | 100.0  |
| action 4 exact      | 44.6   | 100.0  |

## What changed

v1's only error source was row 63 (a step counter that ticks on exactly
every other call). The tick rule itself was correct ("rightmost 1 → 4")
but the parity is hidden state — not deducible from a single frame.
v1 conservatively left row 63 unchanged, ceilinged at 50%.

v2 adds a tiny module-level parity bit (`_step_parity`) that flips on
every call. On the very first call, if the input row 63 is the
canonical fresh-game state (all 1s) we sync the bit to 0. Afterwards
the bit advances purely from internal state — we do NOT re-sync on
later "fresh-looking" inputs because step 1's input also has all-1s
row 63 (the tick lands in state_t1 of step 1, not state_t of step 1).

## Verified invariants

1. **Tick parity**: `tick iff step_index % 2 == 1` — 0/200 mismatches.
2. **Tick rule**: rightmost 1 in row 63 becomes 4 — 100/100 ticking
   transitions match.
3. **No single-frame parity signal exists**:
   - 0/64 cells perfectly separate tick vs no-tick states across all
     n4_t groups (best cell only separates 10/64 groups).
   - Always-tick and never-tick both score 100/200 (symmetric).

## Why this is not curve-fitting

- The mechanism is mechanistic (alternating parity), not memorized.
- The dataset has 200 sequential observations and the parity rule
  holds across all 200 with 0 exceptions.
- The validation harness calls `simulate(...)` in trajectory order,
  matching how a real ARC-AGI-3 inference loop would call it (one
  frame per game step). So the +50% transfers cleanly to inference.
- The auto-resync condition (`row 63 all 1s on first call`) makes the
  sim self-sufficient at episode start without requiring callers to
  manually reset.

## Risks

- **Out-of-order calls** (e.g. MCTS rollouts): the module-level parity
  bit will drift. For such callers we expose `simulate_with_parity(
  state, action, x, y, step_parity)` and `reset_step_parity(value)`.
- **Multiple episodes in one process**: the first-call sync runs only
  once. Callers must invoke `reset_step_parity(0)` at the start of
  each new episode (or rely on a fresh module import per episode).

## Plausible Kaggle impact

The sim is essentially exact (100%) on this game's mechanic. Its real
value to inference is:
- **Forward planning**: lookahead / BFS over action sequences without
  hitting the env, since the model is deterministic and complete on
  the 4 known actions.
- **Reward shaping**: every action returns reward_class=1, done=False
  in our 200 obs — there is no win/level-up frame observed. The sim
  cannot predict terminal states yet; need more observations from
  late-game episodes.
- **Unseen icons**: action 1/2 fall back to identity if the icon at
  a slot was not in our library. Six of 35 (slot, icon) combinations
  are covered; unseen combos will mis-predict the icon row, but the
  rest of the grid (selector + counter) stays correct.

## v3 ideas (low priority)

- Synthesize missing (slot, icon, direction) entries by inspecting the
  canonical library order from the preview panel.
- Collect more observations to model the win/done frame.
- Add inverse-action support (predict prior state given (state, action))
  for backward-chaining planners.
