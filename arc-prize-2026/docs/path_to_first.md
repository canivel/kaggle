# Path to #1 (0.50+)

## Current: 0.10 (v18 CNN-only) -> v29 PENDING (BFS + CNN)

## The Math
- Each perfectly solved game = +0.04 RHAE (1.0 / 25 games)
- Need ~12 perfect games for 0.50
- Local BFS solves 3/25 perfectly (FT09, VC33, LP85)
- Full FORGE on Kaggle should solve more (hidden fields, CNN fallback)

## #1 Strategy (Sergei Fironov, 0.50)
- BFS via importlib for deterministic games (~50-60% of games)
- DQN value model for BFS fallback (when state space too large)
- Beam search for constrained exploration
- Background: Kaggle Master, Yandex, RL expert (CayleyPy RL paper)

## Our Plan
1. v29: FORGE BFS + 23 improvements (PENDING, should be 0.30+)
2. v30: Add DQN/value-guided BFS for large state spaces
3. v31: Game-type specific heuristics (CHRONOS approach)
4. v32+: Iterate based on scores

## Key Techniques Still Needed
- DQN value model (Sergei's edge)
- Beam search when BFS times out
- Better hidden state detection
- Per-game analytical solvers where possible
- Adaptive time budgeting based on game complexity
