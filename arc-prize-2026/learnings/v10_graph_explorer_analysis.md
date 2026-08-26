# v10 Decision: Graph-Explorer Architecture Analysis

**Date:** 2026-04-17
**Context:** v9 = 0.26 (-0.01 from v7 0.27, noise range). Need bigger change.

## Source code analysis: arXiv 2512.24156

Repo: `github.com/dolphin-in-a-coma/arc-agi-3-just-explore`. 3rd place on **PRIVATE** LB.

### Architecture (different from ours)
- **No game source simulation** — interacts only via real agent API
- **Frame-by-frame exploration** — one action per turn, no lookahead
- **Graph of masked-frame-hashes** — each node is a distinct game state
- **Priority tiers (5 groups)** — segment-based heuristics
- **Error threshold** — close nodes after repeated failures

### FrameProcessor (100 lines)
```python
segment_frame(frame):
    # 4-connected components by color → list of {bbox, color, area, is_rectangle, twins}
    
identify_status_bars_with_rule(segments):
    # Rule-based: on edge + (long ratio OR ≥3 twins)
    # Returns mask of status-bar pixels

frame_segments_to_action_groups(segments, n=5):
    # G0: salient_color (6-15) + medium_width (2-32px) → most likely interactive
    # G1: medium_width but not salient → maybe interactive
    # G2: salient but weird size → UI candidates
    # G3: everything else
    # G4: status bar (lowest)
```

### GraphExplorer (300 lines)
- `NodeInfo`: per-state edge tracking (tested/untested, priority group, distance-to-frontier)
- `choose_edge`: untested edges in current group → else navigate to frontier
- `record_test`: update graph, rebuild distances on new finds
- `_maybe_advance_group`: escalate tier when current tier is exhausted

## Why it beats us on PRIVATE LB (hypothesis)

Our BFS:
- Simulates game class via `copy.deepcopy` — super fast exploration (17k states in 90s)
- Finds OPTIMAL paths when it can
- But: limited by branching factor × depth for hard games
- Plus: lots of budget spent on BFS/MCTS for games it can't solve at all

Graph-explorer:
- Real env interaction only — much slower per state
- Exploits priority tiers → avoids wasting clicks on obvious UI
- Keeps going until budget runs out — no "give up" from timeout
- Solves MORE levels but with MORE actions → lower per-level RHAE but more total levels

**On PRIVATE LB (55 games)**, BFS might fail entirely on some games (deep planning required), while graph-explorer at least solves L0-L2 of most games. Total RHAE comes from breadth × depth, and graph-explorer wins on breadth.

## 3 Paths for v10

### Path A: Quick v10 (1 day) — "v9 cleanup"
- Rule-based mask (replace data-driven)  
- Tune back-label reward to [0.0, 0.5] (match existing scale)
- Keep everything else from v7

**Risk:** low. **Expected:** 0.26 → 0.27 at best.

### Path B: Hybrid v10 (3-4 days) — "forge_v35 + graph-explorer fallback"
- **Keep** BFS+MCTS+CNN hybrid exactly as v7
- **Add** GraphExplorer strategy invoked ONLY when BFS fails AND RepeatAction fails AND MCTS can't find path
- Budget: ~300s per stuck level for graph exploration

**Risk:** medium. **Expected:** 0.27+ on games we already solve, plus new level completions on BFS-unsolvable games. Est +0.05-0.10.

### Path C: Full rewrite v10 (1-2 weeks) — "pure graph-explorer"
- Replace entire agent with port of HeuristicAgent
- No BFS, no MCTS, no CNN
- Just segmentation + priority tiers + graph exploration

**Risk:** HIGH. Could drop back to 0.10. **Upside:** matches what won 3rd on PRIVATE.

## Recommendation: **Path B**

- Preserves v7's 0.27 ceiling (falls back to exactly v7 behavior on solvable games)
- Adds the private-LB-proven technique where it matters (unsolvable-by-BFS games)
- Manageable effort (3-4 days)
- Clear local validation path (test on games we can't solve now)

## v10 Path B: concrete implementation plan

### Step 1 (Day 1): Port FrameProcessor
- Copy `segment_frame`, `identify_status_bars_with_rule`, `frame_segments_to_action_groups`
- Replace current data-driven `_compute_counter_mask` with rule-based
- Local test: verify status bar detection on VC33/LP85 matches expected patterns

### Step 2 (Day 1-2): Port GraphExplorer (simpler standalone module)
- Copy NodeInfo and GraphExplorer classes from graph_explorer.py
- Expose as `graph_explorer.py` module alongside forge_v35_tips.py
- Unit test with synthetic transitions

### Step 3 (Day 2-3): Wire fallback into MyAgent
- After BFS timeout, RepeatAction stalled, MCTS couldn't find — enter graph-explorer mode
- Graph-explorer runs until budget exhausted or level completes
- When level completes, exit to v7's normal flow (MCTS shortening, back-label)

### Step 4 (Day 3): Local validation
- Run on all 25 games at 1140s each
- Compare v10 vs v7 per-game scores
- Goal: no regression on solvable games, new completions on hard games

### Step 5 (Day 4): Push + submit

## Decision needed from user
Proceed with Path B? Or prefer Path A (quick cleanup) first, Path B after?
