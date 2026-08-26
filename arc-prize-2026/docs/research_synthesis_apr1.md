# Research Synthesis - April 1, 2026

## Leaderboard
- Top: 0.50 (Sergei Fironov)
- Us: 0.10 (v18), targeting 0.30+ with v24

## Three Tiers of Approaches (from notebook analysis)

### Tier 1: Game-specific hand-coded solvers (CHRONOS, 0.39)
- 85 "cognitive rungs" in modular pipeline
- OBSERVE->CLASSIFY->EXTRACT_GOAL->MAP_EFFECTS->PLAN->EXECUTE->VERIFY
- Causal click mapping: learn what each click does
- Analytical cycle solving: no search for tile puzzles
- Only 3 game IDs hardcoded - mostly generalizable

### Tier 2: BFS via importlib game loading (FORGE, 0.30-0.46)
- Load game Python source, instantiate directly
- BFS with hidden field probing (game.__dict__)
- Win condition extraction from source code
- Counter A* when counter field detected
- CNN fallback when BFS fails

### Tier 3: CNN online learning (StochasticGoose, 0.10-0.17)
- Frame-change prediction via BCE
- Graph-based state tracking
- ResNet value models (graph+value)
- Our evolved greedy_novelty approach

## Our v24: FORGE + 13 Improvements
1. Full MD5 hash (no truncation)
2. Multi-level solution transfer
3. Fixed multiplier bug
4. Added 4x multiplier
5. More BFS time (35%/15%)
6. Object centroid scanning (scipy ndimage)
7. ACMD threshold 500
8. CNN persists across levels
9. LR decay on level transitions
10. Higher CNN exploration (0.25)
11. Slower epsilon decay
12. Higher epsilon floor
13. Causal click mapping

## What We Still Don't Have (vs leaders)
- Game-specific cognitive rungs (would need per-game analysis)
- Analytical cycle solving (planned for v25+)
- Fuel-aware BFS for resource games
- Anti-oscillation mechanisms
- Confidence-weighted rung selection

## Evolution Status
- 50+ generations, best=12 levels (greedy_novelty)
- Plateau'd for heuristic approaches
- BFS approach is the right path forward
