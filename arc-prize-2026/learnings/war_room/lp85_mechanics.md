# lp85 (looping_chains) — Level 2 mechanics

Source: `kaggle-data/environment_files/lp85/305b61c3/lp85.py` (levels line 1069+, engine line 21317+).
Click-only game (ACTION6). 8 levels; baseline_actions `[33,22,31,23,33,34,73,173]`.

## Engine (deobfuscated)

- `izutyjcpih` = per-level chain maps: grids of slot numbers 1..N tracing a loop; sprite pos = map cell x3 (`crxpafuiwp=3`).
- Buttons carry tag `button_<chain>_<R|L>`. Clicking one moves **whatever sprite sits exactly at each slot position** to the next slot (`R`: k -> k+1, wrap N -> 1; `L`: reverse). Tiles are anonymous 2x2 colored squares; **goal tiles are color 11** (`odkpvwbihk`, tag `goal`).
- Win (`khartslnwa`): every 4x4 corner-bracket marker (tag `bghvgbtwcb`, corner color 11, static) must have a `goal` sprite at `(x+1, y+1)` — i.e. a color-11 tile inside every color-11 bracket. (`fdgmtkfrxl`/`goal-o` variant unused in L2.)
- StepCounter (L2: **60**) decrements only on clicks that hit a button; 0 -> lose. Off-button clicks are free no-ops. Top-row pixel bar (colors 5/14) shows budget; short bars at y=1 show level progress.

## Level 2 (`cecdsipmha`, 41x41 grid)

- Chain **A**: 26-slot rectangular ring (map cols x=4,8, rows y=2..11). Chain **B**: 10-slot horizontal row at map y=5. Chain **C**: 10-slot row at y=8.
- **B and C each cross ring A twice**: A8=B7 (24,15), A24=B3 (12,15), A11=C7 (24,24), A21=C3 (12,24). A sprite parked on a crossing belongs to *both* chains — this is the transfer mechanism.
- Brackets sit exactly on crossings A8 (23,14) and A11 (23,23). Goals start on ring A at slots 4 and 20.
- **Pure-A rotation is unsolvable**: goals are 16 apart on the ring; targets are 3 apart. One goal must leave the ring via B or C.
- **Trap**: 4x A_R puts goal1 on bracket1 (A8=B7) but simultaneously puts goal2 on B3 — any B click then drags *both* goals onto row B (we hit this; cost 19 wasted clicks in first attempt).

## Verified scripted policy

`scratchpad/lp85_policy.py` drives the real `arcengine` (wheel 0.9.3) via `perform_action(ACTION6)` with display coords brute-forced through `camera.display_to_grid`:

- L1 (`kdrsqrvpwb`, 20-slot single ring, 13 steps): goal slot 6 -> target slot 1 = 5x A_L. Passed.
- L2: **1x A_R** (goal2 -> C3 crossing, goal1 -> A5), **1x C_R** (park goal2 off-ring at C4), **3x A_R** (goal1 -> A8 = bracket1), **3x C_R** (goal2 -> C7 = bracket2). **8 effective clicks, 52/60 steps left, `levels_completed=2`.** Deterministic, no RNG.

## LLM failure hypothesis

1. **Parity/impossibility blindness**: the obvious policy (spin ring A watching color-11 tiles approach brackets) can never satisfy both brackets; an LLM grinds A clicks until the 60-step budget kills it (the step bar is a subtle 1px row it likely never reads).
2. **Crossings are invisible**: rows B/C look like decorative distractor stripes of shuffling colored tiles; nothing marks (24,15)/(12,15)/(24,24)/(12,24) as shared slots. Discovering transfer requires clicking a B/C button *while a goal sits on a crossing* and attributing the diff correctly amid ~10 tiles moving per click.
3. **Diff noise**: every effective click moves 10-26 tiles; tracking the two color-11 tiles among six shuffling colors across frames exceeds typical visual-diff summarization.

## Minimal unlock cue

Tell the agent: "Tiles sit on numbered loop tracks; loops **share cells where they cross** — rotate a loop to place the color-11 tile on a crossing, then rotate the *other* loop to carry it off/on. Get one color-11 tile inside each color-11 corner bracket simultaneously. Only clicks on arrow buttons consume the (limited) step bar." The single load-bearing fact is **crossings transfer tiles between chains**; everything else is greedy per-chain rotation.
