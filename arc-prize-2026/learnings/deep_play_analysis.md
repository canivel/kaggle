# Deep Play Analysis - What It Takes to Reach 0.1 RHAE

## Current Best Results (10min/game, GPU, CNN+Graph)
| Game | Levels | L1 acts | Human L1 | Efficiency | RHAE |
|------|--------|---------|----------|------------|------|
| R11L | 1/6 | 9 | 7 | 1.3x | 0.029 |
| LP85 | 2/8 | 282 | 33 | 8.5x | 0.000 |
| VC33 | 1/7 | 246 | 6 | 41x | 0.000 |
| TN36 | 1/7 | 2668 | 23 | 116x | 0.000 |
| AR25 | 1/8 | 14577 | 17 | 857x | 0.000 |

## The Core Problem
We're 10-800x less efficient than humans. RHAE squares the ratio:
- 2x human: RHAE = 0.25 (decent)
- 3x human: RHAE = 0.11 (ok)
- 10x human: RHAE = 0.01 (terrible)
- 100x human: RHAE = 0.0001 (zero)

## What Humans Do Differently
1. **Understand the game rules** from 1-2 actions (we take thousands)
2. **Plan ahead** (we react to single frames)
3. **Generalize** level patterns (we reset and re-explore each level)
4. **Goal-directed** (we explore randomly, hoping to stumble on solutions)

## Path to 0.1 RHAE
Need ~3x human efficiency across multiple levels in several games.
This requires UNDERSTANDING, not just exploration.

### Approach 1: State graph + shortest path replay
If we find the winning path, replay it optimally.
- R11L: Found L1 path in 9 actions (human 7). If we could reliably find
  such short paths, RHAE would be high.
- Problem: finding the path still takes thousands of exploration actions.
  But we only need to report the REPLAY count for scoring.

### Wait... does RHAE count exploration actions or just the winning run?
KEY QUESTION: In competition mode, does the agent get ONE shot per game,
or can it explore + then replay?

If ONE shot: current approach is doomed (exploration = wasted actions)
If replay allowed: explore first, then replay winning path = high RHAE

### Approach 2: Better world model for faster rule learning
- DreamerV3/Delta-IRIS: predict dynamics, plan in latent space
- Would reduce exploration from 10000 to maybe 100 actions per level
- Implementation: 4-6 weeks

### Approach 3: Transfer learning across games
- Pre-train on public games, fast-adapt to private games
- Doesn't help with efficiency within a single game
