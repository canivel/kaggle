# ARC Prize 2026 - Competition Plan

## Current Position (March 29, 2026)
- Competition launched 4 days ago - we're EARLY
- Exp 0001: Random baseline -> 1 level
- Exp 0002: CNN-PPO (RL) -> 0 levels (wrong approach)
- Exp 0003: Winner reproduction (BCE) -> 10 levels (3min/game, GPU)
- Kaggle submission v6: scored 0.00 (bugs: GPU OOM + missing FrameData fields)
- Kaggle submission v7: ready with fixes + 3 improvements (submit tomorrow)

## Milestone Targets
| Milestone | Date | Target | Prize |
|-----------|------|--------|-------|
| M1 | June 30 | Top 3 score | $25K/$10K/$2.5K |
| M2 | Sept 30 | Top 3 score | $25K/$10K/$2.5K |
| Final | Nov 2 | Top 5 | $40K/$15K/$10K/$5K/$5K |
| Grand | Nov 2 | 100% RHAE | $700K |

## Research Synthesis

### What Works (Evidence-Based)
1. **Supervised BCE frame-change prediction** >> RL/PPO (12.58% vs <1%)
2. **CNN with one-hot encoding** (16ch) + dual head (action + coordinate)
3. **Hash-based dedup** maximizes experience diversity
4. **Click games easiest** (67% L1 solve rate vs 0% keyboard-only)
5. **Graph-based exploration** can solve 30/52 levels training-free (3rd place)

### What Doesn't Work
1. LLM-based agents (<1% even with frontier models)
2. RL/PPO (too slow to learn, wrong reward signal)
3. Custom harnesses don't generalize (97% known -> 0% unknown)
4. Pure random (1/182 levels)

### Key Bottlenecks (Ordered by Impact)
1. **Can't get past L1** - model resets lose all knowledge per level
2. **Time allocation** - 110 games in 6hrs = 3min each (not enough)
3. **No available_actions on Kaggle** - defaulting to all 6 actions wastes exploration
4. **GPU OOM on Kaggle** - 110 parallel threads, forced CPU (~20x slower)

## Improvement Roadmap

### Phase 1: Quick Wins (This Week) -> v7-v8
- [x] Fix Kaggle bugs (CPU, FrameData compat, try/except)
- [x] Don't reset model between levels
- [x] Frame segmentation for click guidance
- [x] Action sequence replay on GAME_OVER
- [ ] Verify v7 scores > 0 on Kaggle

### Phase 2: Graph + Planning (April) -> v9-v12
- [ ] State hash table (track visited frames, avoid revisiting)
- [ ] Directed state graph (nodes=frames, edges=actions)
- [ ] Pruning: skip actions that create loops or don't change state
- [ ] Back-labeling: when level completes, mark path with distances
- [ ] Value model (ResNet9/18) trained on distance-to-goal labels
- [ ] Adaptive time allocation (triage games in 30s, deep dive promising ones)

### Phase 3: World Model (May-June) -> v13+
- [ ] Delta-IRIS: encode frame deltas as discrete tokens
- [ ] Transformer dynamics model predicts next frame
- [ ] Model-based planning (imagine future states before acting)
- [ ] Combine with CBET (frame-change exploration) for explore/exploit balance

### Phase 4: Meta-Learning (July-Sept) -> v20+
- [ ] Reptile/PEARL for fast adaptation to new environments
- [ ] AMAGO-2 architecture (Transformer meta-RL)
- [ ] Cross-environment pretraining on public games
- [ ] Transfer learning curriculum

## Technical Architecture (Target)

```
Frame (64x64, 16 colors)
    |
    v
[One-Hot Encoding] -> [CNN/ResNet Backbone] -> [Feature Vector]
    |                                               |
    |                                    +----------+----------+
    |                                    |          |          |
    |                               [Action    [Coord     [Value
    |                                Head]      Head]      Head]
    |                                  |          |          |
    v                                  v          v          v
[State Graph] ---- [Transition Table] ---- [Planning Module]
    |                                               |
    v                                               v
[Experience Buffer (hash-dedup)] --------> [BCE Training]
    |
    v
[Action Replay Store] -> replay on GAME_OVER
```

## Compute Strategy
- **Local (RTX 3080)**: Development + testing, 42 act/s, 10min/game experiments
- **RunPod A40**: Extended training, 1-2hr/game, pre-training shared models
- **Kaggle P100**: Submission, must use CPU for 110 parallel games
- **Budget**: Milestone 1 is 3 months away, plenty of iteration time

## Key Metrics to Track
- Levels completed (total across all games)
- L2+ completions (our weakness)
- RHAE score on public games
- Actions/sec on different hardware
- Buffer utilization ratio (signal for stuck vs exploring)
- Time to L1 per game type (click vs keyboard vs hybrid)
