# ARC-AGI-3 Improvement Priorities

## Ranked by Impact

| # | Improvement | Impact | Effort | Notes |
|---|-----------|--------|--------|-------|
| 1 | Don't reset model between levels | HIGH | LOW | Winner's own TODO. Later levels weighted more in RHAE. Just clear buffer, keep weights. |
| 2 | Action sequence replay | HIGH | MED | Store L1 solution, replay on GAME_OVER instead of re-exploring. |
| 3 | Frame segmentation for click masking | HIGH | MED | scipy ndimage connected components -> click on objects not background. 4096->50 targets. |
| 4 | GPU semaphore for parallel games | MED | LOW | `threading.Semaphore(2)` around GPU ops. Prevents OOM with 110 threads. |
| 5 | State graph (2nd place approach) | HIGH | HIGH | Track visited states, avoid revisiting. Enables planning. |
| 6 | ResNet18 backbone | MED | MED | Better features than 4-layer CNN. Proven for grid reasoning. |
| 7 | Smart time budgeting | HIGH | MED | Triage games in 30s, allocate time to promising ones. |
| 8 | Undo action (ACTION7) | MED | MED | Undo bad actions instead of recovering. |

## Key Insight: RHAE Optimization
- Level 6 worth 6x level 1 in scoring
- Squaring: 2x human = 0.25 score, 1x human = 1.0 score
- Not resetting model is the #1 RHAE win (transfers knowledge to harder, higher-weighted levels)

## Blind Squirrel (2nd Place) Approach
- Directed state graph (nodes=frames, edges=actions)
- ResNet18 value model (how close to winning?)
- Back-labels states with distance-to-goal when level completes
- Much smarter than pure frame-change prediction

## For iter3: Implement #1, #3, #7
- Don't reset model (3 lines change)
- Add frame segmentation for click games
- Add adaptive time allocation in the Kaggle notebook
