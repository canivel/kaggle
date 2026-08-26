# ARC-AGI-3 Techniques Catalog
## Compiled 2026-04-01

---

## 1. LEADERBOARD CONTEXT (ARC-AGI-3 Preview)

| Rank | Agent | Score | Approach | Actions Used |
|------|-------|-------|----------|-------------|
| 1st | StochasticGoose (Tufa Labs) | 12.58% | CNN RL action prediction | 255,964 |
| HM | Fluxonian | 8.04% | DSL + LLM hybrid | - |
| 2nd | Blind Squirrel | 6.71% | State graph + ResNet18 value model | 109,108 |
| HM | Play Zero Agent | 4.37% | Random + LLM video analysis | - |
| 3rd | Just Explore (Rudakov) | 3.64% | Graph-based frame exploration | - |
| * | Symbolica Arcgentica | 36.08% (unverified, public set) | Orchestrator-subagent LLM | $1,005 |

Frontier LLMs: Gemini Pro 0.37%, GPT-5.4 0.26%, Opus 4.6 0.25%, Grok-4.20 0%
Humans: 100%

---

## 2. NOTEBOOK ANALYSIS

### 2A. StochasticGoose Pruned (brenorb/arc3-stochasticgoose-pruned-frontier)

**Source:** DriesSmit/ARC3-solution (Tufa Labs, 1st place preview)

**Architecture:**
- ActionModel CNN: 4-layer backbone (32->64->128->256 channels)
- Dual-head: Action head (MaxPool -> FC -> 5 logits) + Coordinate head (Conv 128->64->32->1, preserves 2D spatial bias)
- Input: 16-channel one-hot encoding of 64x64 grid
- ~200K params

**Key Techniques:**
1. **Frame-change prediction**: Binary classification - predict which (state, action) pairs cause frame changes
2. **Hash-based experience dedup**: MD5 of (frame_bytes + action_idx), maxlen=200K buffer
3. **Coordinate pruning via ranked candidates** (MAX_COORD_CANDIDATES=96):
   - Object centroids, bounding box corners/edges scored 1.9-2.6
   - Changed-region pixels scored 3.2 (highest)
   - Successful prior clicks scored 2.8
   - Coarse lattice (16-step, 8-step) scored 0.35-0.6
   - Grid corners/center/edge-centers scored 0.8
4. **Sigmoid probability sampling**: Actions and coordinates sampled proportionally
5. **Level reset**: Clear buffer + reinit CNN on score change
6. **Entropy regularization**: Small bonus for exploration in loss
7. **Sliding window frames**: Keep only last 10 frames (memory efficiency)

**Novel insight:** Coordinate sampling done purely via convolution preserves 2D grid bias - better than flattening or predicting row/column separately.

---

### 2B. Graph Pruned Frontier (brenorb/arc3-graph-pruned-frontier)

**Architecture:**
- VisualSaliencyNet: Lighter CNN (32->64->128 + AdaptiveAvgPool2d(8))
- Separate action head (FC -> 5) and coord head (FC -> 4096)
- StateGraph: directed graph of frame transitions

**Key Techniques:**
1. **Explicit state graph**: Nodes = MD5(frame_bytes), edges = (state, action) -> next_state
2. **BFS to frontier**: Find shortest path through known graph to reach state with untried actions
   - Max BFS depth = 20 steps
   - Falls back to UCB + saliency when BFS finds nothing
3. **UCB exploration scores**: c*sqrt(ln(n_state)/n_state_action), c=2.0
4. **Saliency-weighted action selection**: UCB + 0.3 * normalized_saliency
5. **Coordinate masking**:
   - Non-zero color pixels + 1-cell neighborhood expansion
   - Changed cells since previous frame
   - Coarse 8-step lattice + corners/center
   - Cap at 768 candidates; downsample busy boards via stride-2
6. **Epsilon-greedy**: 5% random actions for exploration
7. **Available action masking**: Properly handles gateway's raw int action IDs

**Novel insight:** BFS through known state graph to systematically reach frontier states is much more efficient than random exploration.

---

### 2C. Baseline Starter (sigmaborov/arc-agi-3-baseline-starter)

**Architecture:** Offline BFS solver using direct game class instantiation via importlib

**Key Techniques:**
1. **Offline BFS via deepcopy**: Load game .py source, instantiate game class, use copy.deepcopy() for state branching
2. **Action scanning**: Probe each action to find ones that actually change the frame
3. **Click coordinate pruning**: Stride-2 scan on non-background pixels only, with effect dedup
4. **Counter A* search**: When a win-condition counter field is detected in game source:
   - Parse game source code to find `self.next_level()` condition
   - Detect counter direction from `>=` vs `<=` comparisons
   - Priority queue ordered by counter progress toward win condition
5. **Hidden state probing (ACMD)**: Detect scalar fields that change without pixel changes:
   - Compare game.__dict__ before/after actions
   - Filter clock fields (change on every action) vs true state fields
   - Retry BFS with trigger-field-aware hashing
6. **Solution transfer between levels**:
   - Direct replay of previous solution on next level
   - Object matching across levels (color + size similarity)
   - Coordinate offset calculation (mean displacement of matched objects)
   - Solution expansion (2x, 3x, 1.5x repetitions with offset)
7. **Warm-up unlock**: Try each ACTION1-4 as prefix to discover locked initial states
8. **IDDFS fallback**: Iterative deepening DFS for small action spaces (<=6)

**Novel insights:** 
- Source code analysis to detect win conditions is extremely powerful
- Hidden state fields invisible in pixels can completely break frame-only hashing
- Solution patterns often transfer across levels with coordinate offsets

---

### 2D. BitterTruth-AI (thezetaproject/bittertruth-ai-arc-agi-3-competition-submission)

**Architecture:** 7-tier algorithmic solver cascade

**Tiers (in priority order):**
1. **Deepcopy BFS**: Standard BFS with composite state hashing (frame + sprite positions + game state)
2. **Direct-load BFS**: importlib-based game class loading + BFS (independent of SDK wrapper)
3. **Env IDDFS**: Iterative deepening via SDK env.reset()/env.step()
4. **Coordinate BFS**: Analytical solver for click-based games
5. **MCTS**: Monte Carlo Tree Search with UCB1 selection (c=1.414)
6. **Greedy**: One-step lookahead, pick best score delta
7. **Random**: Uniform random with grid-scan coordinates

**Key Techniques:**
1. **Multi-level state hashing** (4 different hash functions!):
   - `_hash_frame`: Raw pixel hash
   - `_game_state_hash`: Introspect all game attributes (positions, lists, dicts)
   - `_sprite_pixel_hash`: Hash sprite pixel data directly
   - `_composite_state_hash`: Frame + sprite positions + level state combined
2. **Game classification**: Detect game type from probe results:
   - Click-only -> analytical, MCTS
   - Move-only -> BFS variants, MCTS
   - Score-improving actions -> analytical first
   - Low frame-changing actions -> analytical, MCTS
3. **Coordinate dedup**: Cluster click coordinates by their effect hash
4. **Multi-level BFS**: Solve levels sequentially with time budget allocation
5. **MCTS rollout**: 25-step random rollouts from leaf nodes, 2000 simulations
6. **BFS parameter tuning from probes**: Adjust max_depth/max_nodes based on action effects
7. **Solution transfer**: Direct replay + coordinate offset matching between levels

**Novel insights:**
- Multiple hash functions for different game types is crucial
- Game classification via action probing drives solver selection
- MCTS as fallback when BFS is intractable
- Sprite-level introspection catches hidden state that frame hashing misses

---

### 2E. FORGE v18 (projectforty2/forge-arc-agi-3-agent) -- 83 votes, Kaggle score 0.39

**Architecture:** Offline BFS + CNN hybrid with attention mechanisms

**Key Techniques:**
1. **CBAM (Convolutional Block Attention Module)**: Channel attention + spatial attention on CNN features
2. **ActionEffectAttention**: Cross-attention between current CNN features and memory of past action effects
   - Diff encoder: Conv(1,8,8,stride=8) -> Conv(8,16,4,stride=4) -> FC(64,32)
   - Query projection from CNN global features
   - Values = concatenated [diff_embedding, reward, action_one_hot]
3. **ForgeNet**: 26-input-channel CNN (16 one-hot + 10 derived features):
   - Background mask, rarity map, edge map, row/col position
4. **CLTI (Cross-Level Training Injection)**: Inject BFS L0 solution demos into CNN replay buffer for L1
   - Replay the BFS solution from previous level
   - Record (frame, action, reward=2.0) tuples as expert demonstrations
   - Pre-train CNN with these demos before online exploration begins
5. **Counter A***: Priority BFS guided by game counter detection
6. **Warm-up unlock**: For locked initial states (sc25-type games)
7. **ACMD trigger finder**: Hidden-state retry mechanism
8. **Object tracking reward**: Detect objects that moved 2-20 pixels, bonus +0.3*min(moved,3)
9. **Template detection**: Detect divider lines (columns/rows with <=2 non-bg pixels) to mask left/top half
10. **Heuristic fallback**: Try directional actions first (steps 0-3), then click object medians

**Novel insights:**
- CLTI (demo injection from solved levels) is a form of curriculum learning
- ActionEffectAttention lets the CNN remember what past actions did
- Rich input features (26 channels) significantly improve CNN performance
- Template detection for split-screen game layouts

---

### 2F. Residual CNN + State Graph (parthenos, score 0.17)

**Architecture:** Hybrid residual CNN with cross-attention + state graph

**Key Techniques:**
1. **ResidualBlock**: Pre-activation residual blocks (BN -> ReLU -> Conv -> BN -> ReLU -> Conv + skip)
2. **ActionModelV2**: ~900K params
   - Stem -> 3 stages with residual blocks (64->128->256)
   - **Cross-attention action head**: 5 learnable action queries attend to 16x16 spatial tokens via MultiheadAttention(4 heads)
   - **Decoder coordinate head**: ConvTranspose2d upsampling 16->32->64 with residual blocks
3. **State graph** (from Rudakov et al.): Track (state, action) -> next_state transitions
   - `novelty_bonus = 1/sqrt(n+1)` for count-based intrinsic reward
   - `untried_action_mask` for systematic exploration
4. **Prioritized Experience Replay**:
   - Priority = (reward + novelty_bonus + epsilon)^alpha
   - Alpha = 0.6 for moderate prioritization
   - Hash-based dedup

**Novel insights:**
- Cross-attention is better than pooling for action prediction (global context)
- Residual blocks + BatchNorm stabilize online training
- Decoder-style coordinate head preserves 2D structure better than FC
- Prioritized replay focuses learning on informative transitions

---

### 2G. QOR Agent (ravikaash/qor-arc-agi-3-agent) -- 22 votes

**Architecture:** Pre-compiled binary agent with .qor DNA files

**What it is:** A compiled binary (`arc3`) that reads `.qor` rule files and `game_rules.json`. This is NOT a Python-based agent -- it's a pre-compiled solver with game-specific rule files (DNA).

**Key observation:** This represents a fundamentally different approach -- hand-crafted game-specific solvers compiled to native code. The DNA files likely contain game-specific strategies/rules discovered through analysis.

---

## 3. GITHUB REPOSITORIES

### DriesSmit/ARC3-solution (1st place)
- StochasticGoose CNN RL agent
- custom_agents/action.py: Core action learning
- Key: off-policy training from all stored transitions

### dolphin-in-a-coma/arc-agi-3-just-explore (3rd place)
- Graph-based exploration, training-free
- Frame segmentation into single-color connected components
- Priority tiering: 5 tiers based on button likelihood (size, color salience)
- Status bar detection and masking
- BFS with priority-based action selection (exhaust high-priority before low)
- Post-eval bugfix improved from 12 to 17 median levels

### symbolica-ai/arcgentica (36% on public set, LLM-based)
- Orchestrator-subagent architecture for ARC-AGI-2 (not ARC-AGI-3 directly)
- Persistent Python REPL agents
- Recursive delegation: call_agent() in scope enables spawning subagents
- Max 10 subagents per task, 2 attempts
- 350 lines of Python for full ARC-AGI agent
- Opus 4.6 at 120k context -> 85.28% on ARC-AGI-2
- Key: state distribution across agents prevents context rot

### arcprize/arc-agi-3-benchmarking
- Official developer harness for ARC-AGI-3
- Benchmarking tools and local testing

### arcprize/ARC-AGI (Toolkit)
- Official Python API for ARC-AGI-3 interactive environments

---

## 4. KEY TECHNIQUES NOT YET IN OUR AGENT

### HIGH PRIORITY (proven effective):

1. **Offline BFS via importlib** (FORGE, BitterTruth, Baseline Starter):
   - Load game .py source directly, instantiate game class
   - deepcopy() for state branching, solve entire levels offline
   - This is the single most impactful technique for deterministic games

2. **Counter A* / Hidden State Detection** (FORGE, Baseline Starter):
   - Parse game source code to find win conditions (self.next_level())
   - Detect counter fields and their direction
   - Priority BFS guided by counter progress

3. **CLTI - Cross-Level Training Injection** (FORGE v18):
   - Replay BFS solutions from solved levels as expert demonstrations
   - Pre-train CNN before online exploration on new level

4. **Solution Transfer with Coordinate Offsets** (FORGE, Baseline Starter, BitterTruth):
   - Match objects across levels by color and size
   - Calculate mean displacement
   - Adjust click coordinates by offset

5. **Multi-tier Solver Cascade** (BitterTruth):
   - BFS -> IDDFS -> Coordinate BFS -> MCTS -> Greedy -> Random
   - Game classification drives solver selection

### MEDIUM PRIORITY (architectural improvements):

6. **Residual CNN + Cross-Attention** (Residual CNN notebook):
   - ResidualBlock with pre-activation
   - 5 learnable action queries with MultiheadAttention for global context
   - Decoder-style coordinate head with upsampling

7. **ActionEffectAttention / CBAM** (FORGE v18):
   - Cross-attention between current state and memory of past action effects
   - CBAM channel+spatial attention on backbone features

8. **Prioritized Experience Replay** (Residual CNN notebook):
   - Priority = (reward + novelty_bonus)^0.6
   - Focuses training on informative transitions

9. **26-channel Input Features** (FORGE):
   - 16 one-hot + background mask + rarity map + edge map + position encoding

10. **State Graph with BFS to Frontier** (Graph Pruned notebook):
    - Explicit directed graph of known transitions
    - BFS to find shortest path to states with untried actions
    - UCB + saliency for action selection when BFS exhausted

### LOWER PRIORITY (specialized):

11. **Template/Divider Detection** (FORGE):
    - Detect split-screen layouts, mask non-interactive half

12. **Sprite-level State Hashing** (BitterTruth):
    - Multiple hash functions for different game types
    - Introspect game object attributes, sprite positions

13. **Warm-up Unlock** (FORGE, Baseline Starter):
    - Try each action as prefix to discover locked initial states

14. **Vision resolution optimization** (HN discussion):
    - Optimal resolution window: 512-1024px for Claude vision
    - 128-256px too small, 2048px too large

---

## 5. COMPETITION INSIGHTS

- **No internet during eval**: Must be fully offline (no LLM API calls)
- **Scoring = action efficiency**: Compared to human baseline per level, normalized per game
- **Preview games were too friendly to random search**: Future games will be harder
- **8-hour time limit** per game
- **Quadratic scoring curve**: Small inefficiencies compound fast
- **Milestone #1**: June 30, 2026 ($25K 1st place) - must open source
- **Games have 8-10 levels each**, progressive difficulty
- **Exploration budget matters**: ~200K actions is the practical limit
