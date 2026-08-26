# ARC-AGI-3 Data Analysis - Research Findings

## Game Solvability by Control Type
| Type | Games | L1 Solved | Rate |
|------|-------|-----------|------|
| click | 6 (+FT09) | 4-5 | 67-71% |
| keyboard_click | 12 | 6 | 50% |
| keyboard | 4 | 0 | 0% |

## Fastest L1 Completions (3 min budget)
| Game | Type | Actions to L1 | Time | Human Baseline |
|------|------|--------------|------|----------------|
| R11L | click | 15 | 1s | 167 |
| SP80 | kb_click | 314 | 7s | 472 |
| LP85 | click | 542 | 12s | 422 |
| M0R0 | kb_click | 567 | 13s | 970 |
| FT09 | click | 710 | 17s | 163 |

## Key Metrics
- GPU throughput: ~42 actions/sec (RTX 3080)
- CPU throughput: ~2 actions/sec (20x slower)
- 3 min/game -> ~7,300 actions
- Train steps: ~8/sec (1 train step per 5 actions)

## Time Budget Extrapolation
| Budget | Actions | Est. L1 Solved | Est. Total Levels |
|--------|---------|----------------|-------------------|
| 3 min | 7,300 | 10/25 (40%) | 10 |
| 30 min | 73,000 | 15-20 | 20-30 |
| 6 hrs | 864,000 | 20-23 | 40-60 |

## Critical Finding: More time alone won't crack L2+
R11L stuck at L1 with 25K actions (10 min). The model resets between levels, so L2
requires re-learning from scratch with harder patterns. Need smarter strategies.

## Optimal Kaggle Time Allocation (6hrs, 110 games)
1. 60s initial scan per game (110 min)
2. 5 min for ~44 games showing progress (220 min)
3. Remaining 30 min: top 5 easiest for deeper levels
Expected: ~44 L1 + 5-10 L2 = ~50-55 levels

## Buffer Utilization as Signal
- Low unique/total ratio = agent stuck in loops (SB26: 5.7%)
- High ratio = diverse exploration but game is hard (S5I5: 97%)
- Use this to triage games during time allocation
