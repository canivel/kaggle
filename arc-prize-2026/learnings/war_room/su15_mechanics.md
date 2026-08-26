# su15 ("Suika Vacuum") — Level 2 Mechanics

Source: `kaggle-data/environment_files/su15/4c352900/su15.py` (9 levels, actions ACTION6=click, ACTION7=undo). All claims below verified by scripted policy on the local arcengine (0.9.3), scripts in scratchpad (`drive_su15.py`, `fail_su15.py`, `trap_su15.py`).

## Core mechanic: click = vacuum
- ACTION6 at (x,y): every fruit/enemy whose bbox is within **radius 8** of the click is pulled **to the click point** over 4 frames (speed 4/frame; fruits arrive exactly). Clicks with y<=9 or y>=63 are no-ops. Valid-action lattice: x∈{0,4,…,60}, y∈{10,14,…,62}.
- After the pull, any **same-level fruits that overlap merge into ONE fruit of level+1** at their centroid (n-way: 3 fruits still yield ONE fruit — mass is lost).
- **Mixed-level overlap = fault**: fruits flash, step bar loses 2+2·penalty (escalates: −2, −4, −6…), and the board **auto-rolls-back** to the pre-click snapshot. No merge.
- Each normal click costs 1 step from the level's budget. Win check runs after every vacuum completes.

## Level 2 (index 1) exact layout
- Eight level-0 fruits (1px, color 10) in four natural pairs: (41,37)/(37,40), (18,37)/(16,41), (14,55)/(16,57), (49,54)/(47,56).
- Goal ring (9×9, color 9) at (29,23) — win region for fruit **center**: x∈[29,38), y∈[23,32). Key swatch at top (color 11 = level-3 fruit) shows the target.
- `goal=[3,1]`: **exactly one level-3 fruit whose center is inside the ring**. Steps: 32. No enemies.
- Feasibility invariant: Σ 2^level over fruits must stay ≥ 8 (=2³). Any 3-way merge breaks it → goal/key sprites turn **gray (color 2)** = unwinnable; ACTION7 (undo) restores positions/levels (free, does NOT restore spent steps) and un-grays.

## Verified winning policy (18 clicks, budget 32)
1. Four pair-merge clicks at pair midpoints: (39,38),(17,39),(15,56),(48,55) → four L1.
2. Drag one L1 toward its partner in ≤7px hops (each hop-click must catch only the dragged fruit), then a midpoint click catching both → L2. Twice → two L2.
3. Same to merge L2+L2 → one L3.
4. Drag L3 toward (33,27); win fired as soon as its center entered the ring.
Also verified: Level 1 = drag single L2 fruit into ring at (48,15), 8 clicks. Traps reproduced: 3-way click at (16,49) → single L1, `grayed=True`, undo recovered; mixed-level catch → −2 steps + rollback.

## Why the LLM fails Level 2
1. **Invisible mechanic**: no avatar; click semantics (radius-8 vacuum that drops things AT the click point) must be induced from 1-px fruit motion — nearly invisible at 64×64.
2. **Silent mass loss**: catching 3+ zeros in one click looks like progress (a bigger fruit appears) but makes the level unwinnable; the only cue is the goal/key turning gray, which the LLM doesn't decode, and it never uses ACTION7.
3. **Escalating flash penalty**: careless clicks near mixed levels burn 2/4/6 steps and roll back state, so the LLM sees "nothing happened" while the budget collapses.
4. **Exact-count goal**: it must finish with the L3 fruit's center inside the ring — random final merges land outside.

## Minimal unlock cue (inject into prompt)
"Each click is a vacuum: everything within 8 pixels is dragged to the click point. Two — and only two — same-color-tier fruits that meet merge into the next tier (3+ merge into one and waste fruit; if the goal turns gray it's unwinnable — press ACTION7 to undo). Never let different tiers touch (costs extra turns). Build 8×tiny → 4 → 2 → 1 big fruit by clicking between pairs, moving fruits in short hops, and make the final fruit's center land inside the goal ring."
