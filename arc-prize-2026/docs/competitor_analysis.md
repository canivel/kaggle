# Competitor Analysis - Top Kaggle Notebooks

## Leaderboard (April 1, 2026)
| Rank | Team | Score |
|------|------|-------|
| 1 | Sergei Fironov | 0.50 |
| 2 | NiyatiSingla | 0.48 |
| 3 | Pg0106 | 0.46 |
| Us | canivel | 0.10 |

## Key Notebook: Graph Exploration w/ Value Learning (parthenos)
**Architecture**: Combines 3rd place graph + 2nd place value learning

### What they have that we don't:
1. **ResNet18 ValueNet** - trained online to predict win-direction
2. **Back-labeling** - when level won, BFS backward labels distance-to-win on all states
3. **UCB exploration** (upper confidence bound) - balances explore/exploit
4. **5-tier segment priority** - small objects first (more likely interactive)
5. **Status bar masking** - detects and masks status bar rows
6. **Frame processor** - connected component analysis per frame
7. **Experience buffer** with value training every 4 actions

### What might break on Kaggle:
- Uses GPU (ResNet18 per game thread)
- 110 parallel = 110 ResNets = GPU OOM
- But they scored - so either they handle it or Kaggle gives more resources

### Key Takeaway:
The VALUE LEARNING is what gets 0.25+. Pure exploration plateaus around 0.10.
We need to add value prediction to break through.
