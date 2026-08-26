# sb26 mechanics — level 2 solved analytically (2026-07-13)

Source read: `kaggle-data/environment_files/sb26/7fbdac44/sb26.py` (obfuscated names deobfuscated).
Verified: scripted no-LLM policy on the real local engine (arc_agi OFFLINE + taaf GameAPI)
clears L1 then L2 first try — **24 actions total** (L1 = 8 clicks + RUN, L2 = 14 clicks + RUN).
Script: scratchpad `sb26_policy.py` (pattern from `duck_eval/scheduler/smoke_test.py` S4).

## Core mechanic: "program the color sequence" (Human-Resource-Machine-lite)
- **Top row** (`quhhhthrri` cells, sorted by y,x) = target color sequence.
- **Frames** (tag `pkpgflvjel`; last char of name = slot count, e.g. `qdmvvkvhaz4` = 4 slots
  at `(frame.x+2+i*6, frame.y+2)`) hold items.
- **Bottom tray** (y=56) holds colored 6x6 items (`lngftsryyw`).
- **ACTION6** click: first click selects an item, second click on an empty slot spot
  (`susublrply`) places it (−1 energy); clicking another placed item swaps (−1 energy).
  Valid click coords = `sprite.x+1, sprite.y+1`.
- **ACTION5 = RUN** (−1 energy): interpreter walks the MAIN frame (topmost) left→right,
  comparing each item's color to the next target cell. Match fills the cell; mismatch or
  empty slot = fail flash, cells reset (placements survive). All cells filled → `next_level()`.
- **ACTION7 = undo** placement history. Energy = 64/level; 0 → `lose()`.

## Level 2: the CALL / subroutine mechanic
Two 4-slot frames. Main frame (border 8, y=20) slot 2 is pre-occupied by an **immovable**
`vgszefyyyp` "call" item (box-with-hole, interior color 14). RUN, on hitting it, jumps into
the frame whose border color equals its interior color (frame 14, y=34), executes its 4
slots inline, then resumes the main frame. Flattened output:
`main0, main1, f14[0..3], main3` — 7 outputs for 7 targets `12,15,8,9,14,11,6`.

**Winning assignment**: main = `12, 15, CALL, 6`; frame14 = `8, 9, 14, 11`.
Winning action pattern (7 select+place pairs, then RUN):
```
12@(29,56)->(20,20)  15@(15,56)->(26,20)  6@(36,56)->(38,20)
 8@(8,56)->(20,34)    9@(43,56)->(26,34)  14@(22,56)->(32,34)  11@(50,56)->(38,34)
then ACTION5
```

## Why an LLM agent plausibly misses it
1. **Perception**: the call item renders as just another small colored square; nothing marks
   it as a jump. The link is color-identity (item interior == frame border) plus a subtle
   2px color-14 connector (`pqezjimbse`) between frames.
2. **Wrong prior from L1**: L1 is a trivial 1:1 copy of the top row into one frame. The
   natural generalization — "fill slots left-to-right to match targets reading order" —
   places frame-14's colors as targets 4..7 directly, which fails because execution order
   is main-with-inline-expansion, not visual reading order (target index 2..5 map to
   frame 14, and main slot 3 maps to target 6).
3. **Trial-and-error is punished**: each move and each RUN costs energy (64); the fixed
   call item can't be clicked, which confuses agents that try to move it; repeated failed
   RUNs + reshuffles burn out the bar → `lose()`.
4. Multi-step memory: attributing which of 7 placements caused a fail (fail flash shows
   position, but agent must track pointer-into-two-frames state).

## Minimal cue that unlocks it
One prompt line: *"A box-with-a-hole item is a CALL: when RUN reaches it, execution jumps
into the frame whose border color matches the hole color, runs its slots, then returns.
The flattened sequence must equal the top row. Call items are fixed; every move costs
energy."* Plus: pass `_get_valid_actions` coords (item/spot centers) so clicks land.
