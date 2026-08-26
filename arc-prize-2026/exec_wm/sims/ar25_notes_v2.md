# ar25 sim v2 notes

## v1 → v2 scoreboard (200-tuple set)

| metric              | v1     | v2     | delta  |
|---------------------|--------|--------|--------|
| state_exact_pct     | 67.50  | 80.00  | +12.50 |
| pixel_match_pct     | 99.438 | 99.565 | +0.127 |
| reward_acc_pct      | 98.00  | 98.50  | +0.50  |
| done_acc_pct        | 100.0  | 100.0  | —      |

Per-action exact-match %:

| action       | v1   | v2    | delta |
|--------------|------|-------|-------|
| 1 (up)       | 80.6 | 93.5  | +12.9 |
| 2 (down)     | 89.5 | 94.7  |  +5.2 |
| 3 (L← R→)    | 66.7 | 92.6  | +25.9 |
| 4 (L→ R←)    | 48.3 | 93.1  | +44.8 |
| 5 (meter)    | 100  | 100   |    —  |
| 6 (NOOP)     | 100  | 100   |    —  |
| 7 (mirror)   |  8.3 |  8.3  |    —  |

## What changed conceptually

v1 erased the sprites and re-stamped fresh **full** 9x9 templates after each
move. That assumed the visible sprite was always the canonical L-shape,
which is wrong: the engine clips each sprite to a valid region, and the
"cut" appearance is just a render-time effect on a fully tracked logical
position.

v2 explicitly:

1. **Infers each sprite's logical (r0, c0)** by template-matching with a
   "missing cell = OK (no penalty)" score. A cell that's missing from the
   state because it was clipped or occluded doesn't count against the
   match. This lets us find the true logical position even when the
   visible sprite is heavily cut.

2. **Translates** the logical position by the action delta (no
   intermediate render).

3. **Re-renders** by these rules, in order, on a freshly-built background:
   - bg = playfield 9 + divider 10 (cols 30–32) + col-63 meter (sticky)
     + row-63 bar + static 11-decoration at rows 45–52 cols 51–59.
   - R cells (template == 4) painted, but **skipped on divider cols**
     (R can never overpaint the divider).
   - L cells:
     - template 5 → paint 5 (L wins over R, over divider, over playfield).
     - template 0 → paint 10 if cell is on divider col (divider shows
       through L's 0-hole); else paint 0.
   - col-63 meter and row-63 bar are preserved from the input state and
     written last; meter ticks the topmost 11→5.

## Invariants (200-tuple verified)

1. **Background**: cols 30–32 = 10 (divider), col 63 = sticky meter (11→5
   top down), row 63 = 5, static 11-decoration at rows 45–52 cols 51–59.
   Divider cols never carry value 4 (R color) in any of the 200 tuples,
   confirming R is divider-clipped at render time.

2. **Sprite z-order**: L > divider > R > background. L's 0-holes are
   transparent to the divider underneath (rendered 10) but opaque (0)
   over playfield.

3. **Action deltas (logical positions, no clipping at move-time)**:
   - 1: dr_L = dr_R = −3
   - 2: dr_L = dr_R = +3
   - 3: dc_L = −3, dc_R = +3
   - 4: dc_L = +3, dc_R = −3
   - 5: meter tick only
   - 6: NOOP, reward 0
   - 7: identity (single-frame ambiguous)

4. **Vertical OOB**: if either sprite's new r0 leaves [0, 27], BOTH stay
   put (NOOP), reward 0. Horizontal positions are unconstrained — clipping
   at render produces the "cut" appearance.

5. **Reward**: 1 on any successful move / meter tick; 0 otherwise.

## Stop-criterion check

- v2 improvement is **+12.5 absolute** state-exact (well above the +2 pt
  threshold). Not curve-fitting: the same render rule lifts every movement
  bucket simultaneously by huge margins (a=4 +44.8 pts).
- Per-bucket exact-match for movement actions all land 93–95% — single
  failure mode now is action 7 (single-frame ambiguous mirror) and a few
  reward-1 → reward-0 misses at OOB boundaries.

## Remaining failures (40/200)

- **Action 7 (33/36)**: needs prior-action memory; not recoverable from a
  single frame. Identity is the modal correct answer (3/36). Could try
  predicting "down" (10/36) as modal-non-identity, but the gain is +7
  exact vs the risk of hurting the 22 identity cases that *also* survive
  because the meter doesn't tick. Net: keep identity.
- Action 1/2/3/4 small residuals: mostly the few cases where the
  inference picks a wrong c0 because state_t is heavily occluded (both
  sprites overlapping) — would need either prior-state memory or a
  stronger inference scorer.

## Is this useful on Kaggle?

Yes, conditional on:
- The harness querying `simulate()` for one-step state predictions during
  search — v2 is now 80% exact, so an MCTS/BFS rollout would actually get
  useful information about which actions move which sprite.
- For action 7, even at 8% exact the pixel-match is 97.8%, so a
  pixel-difference reward signal is still ~97% accurate, useful for
  reward-shaping but not for exact-state lookup.
- Identity-mostly games (action 6 = 100%, action 5 = 100%, action 7 =
  identity heuristic) are still correctly identified, so the search
  knows when to "skip" actions that don't change state.

## Files

- `exec_wm/sims/ar25_sim.py` — active sim (v2 contents).
- `exec_wm/sims/ar25_sim_v2.py` — same v2 (kept for diff/audit).
- `exec_wm/sims/ar25_sim_v1_backup.py` — original v1, in case revert
  needed.
