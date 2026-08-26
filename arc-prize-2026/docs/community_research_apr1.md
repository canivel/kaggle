# ARC-AGI-3 Community Research - Exhaustive Discussion Analysis
**Date: April 1, 2026**

## 1. Competition Overview & Current Status

### Launch Metrics (Day 1)
- 675K social media views, 15K website visits
- 700 Kaggle entrants, 70 submissions, 30 teams on leaderboard
- Top Kaggle score on Day 1: **18%**
- Trended #1 on Twitter/X and became #1 Kaggle competition

### Timeline
- **Launch:** March 25, 2026
- **Milestone 1:** June 30, 2026 ($25K / $10K / $2.5K) - must open-source by deadline
- **Milestone 2:** September 30, 2026 ($25K / $10K / $2.5K)
- **Submissions close:** November 2, 2026
- **Results announced:** December 4, 2026

### Prize Structure ($850K Total for ARC-AGI-3 Track)
- Grand Prize: $700K for 100% RHAE (carries forward annually if unclaimed)
- Top Score: $40K / $15K / $10K / $5K / $5K

---

## 2. Leaderboard & Scores

### Preview Competition Results (30-Day Developer Preview)
| Rank | Team | RHAE Score | Levels Completed | Approach |
|------|------|-----------|-----------------|----------|
| 1st | StochasticGoose (Tufa Labs) | 12.58% | 18 levels | CNN + RL frame-change prediction |
| 2nd | Blind Squirrel | 6.71% | 13 levels | State-graph + action pruning + ResNet18 value model |
| 3rd | Explore It Till You Solve It | 3.64% | - | Training-free graph exploration |

### Frontier Model Scores (Semi-Private Set, March 2026)
| Provider | Model | Score |
|----------|-------|-------|
| Google | Gemini 3.1 Pro Preview | 0.37% |
| OpenAI | GPT 5.4 (High) | 0.26% |
| Anthropic | Opus 4.6 (Max) | 0.25% |
| xAI | Grok-4.20 Beta | 0.00% |

**Human baseline: 100%** (all 135 environments fully solvable by at least 2 independent first-time players)

### Key Score Context
- CNN-based approach beat ALL frontier LLMs by 12+ percentage points
- Non-LLM methods dominate the leaderboard
- "Systematic state tracking matters more than raw model size"

---

## 3. Winning Approaches - Detailed Analysis

### A. StochasticGoose (1st Place, 12.58%) - Dries Smit, Tufa Labs
**Source:** https://github.com/DriesSmit/ARC3-solution

**Architecture:**
- 4-layer CNN: 32->64->128->256 channels
- Input: 16-channel one-hot encoded 64x64 frames
- Dual output heads: action classification (ACTION1-5) + spatial coordinate prediction (ACTION6)
- Coordinate head uses convolutional layers (not flattened) to preserve 2D spatial inductive bias

**Training:**
- Supervised learning: (state, action) -> frame_changed binary labels
- Binary cross-entropy loss with light entropy regularization
- Experience buffer: 200K unique state-action pairs, hash-based deduplication
- Dynamic resets: model and buffer clear on new levels to prevent catastrophic forgetting

**Exploration Strategy:**
- Stochastic probability-guided: actions predicted to cause frame changes get higher selection probability
- Hierarchical: first sample action type via sigmoid probabilities, then coordinates if ACTION6
- Explicitly avoids LLMs due to token explosion

**Key Insight:** Predicting which actions cause frame changes enables dramatically more efficient exploration than random selection.

### B. Blind Squirrel (2nd Place, 6.71%) - Will Dick

**Architecture:**
- Directed state graph builder
- ResNet18-based value model for ranking (state, action) pairs

**Strategy:**
- Builds state graph from observed frames
- Prunes actions that create loops or don't change state
- When score improves, back-labels level with distances
- Retrains value model to rank toward next milestone
- Repeats until win or cap

**Performance:** Used only 109,108 total actions vs StochasticGoose's 255,964 - more action-efficient but completed fewer levels.

### C. Graph-Based Exploration (3rd Place approach, arxiv 2512.24156)

**Algorithm - Hierarchical Action Selection:**
1. Test untested actions in current state at current priority level
2. If exhausted, navigate to nearest reachable state with untested high-priority actions
3. If none exist, increment priority threshold and recurse

**State Graph:**
- Nodes = unique game states (perceptual hashing)
- Edges = state transitions from actions
- Tracks: action space, priority levels, exploration status, successor states, shortest distances

**Action Prioritization (Frame Processor):**
- Image segmentation into single-color connected components
- Visual stratification into 5 priority tiers (segment size, morphology, color salience)
- Status bar detection and masking to eliminate UI elements
- Reduces 4,096 potential click actions to salient interactive elements

**Results:** Median 30/52 levels (8-hour runtime). No learning or world model needed - pure exploration suffices.

### D. Arcgentica / Symbolica AI (Not Competition-Eligible)

**Architecture:** Orchestrator-subagent
- Top-level orchestrator does NOT interact with environment directly
- Delegates to specialized subagents returning compressed textual summaries
- Constrains context growth, maintains higher-level planning
- Uses Claude Opus 4.6 as backbone

**Performance:** 36% (unverified), 113/182 levels, 7/25 complete games
**Cost:** $1,005 per evaluation ($8,900 naive)
**Limitation:** Requires API calls - banned in Kaggle submission (no internet)

### E. Duke University Harness

**Innovation:** Allows LLM to execute arbitrary Python code to selectively retrieve and transform information from action history
- Solves context management for large reasoning models
- Achieved 97.1% on known environment TR87
- BUT 0.0% on unfamiliar environment BP35
- **Critical finding:** Performance on seen environments does NOT transfer to unseen

---

## 4. Scoring Deep Dive (RHAE)

### Formula
```
Per-Level: S = min(1.0, human_actions / ai_actions)^2
Per-Game:  E = sum(level_weight * S) / sum(weights)  [level 1 = weight 1, level 2 = weight 2, etc.]
Total:     T = average across all environments
```

### Key Thresholds
| AI Actions vs Human | Score |
|---------------------|-------|
| 1x (matching human) | 100% |
| 1.5x | 44% |
| 2x | 25% |
| 3x | 11% |
| 5x (hard cutoff) | 4% |
| 10x | 1% |

### Critical Scoring Rules
1. **5x action cap**: Hard cutoff at 5x human actions per level - agent is stopped
2. **Score capped at 1.0**: Even if AI outperforms human, no bonus
3. **Later levels weighted more**: Level 1 = 1x weight, Level N = Nx weight - tutorial levels barely matter
4. **Unfinished levels = zero**: Completion before efficiency
5. **Second-best human baseline**: Not the best, not median - 2nd best from 10 testers
6. **Only environment-altering actions count**: Internal reasoning/tool calls don't count

### Strategic Implications
- **Focus on completing levels first**, then optimize efficiency
- **Later levels dominate the score** - level 5 is worth 5x level 1
- **Small inefficiencies compound quadratically** - every extra action hurts badly
- **Matching human speed is the ceiling** - no reward for being faster

---

## 5. Environment & Technical Specifications

### Game Format
- **Grid:** 64x64, 16 colors
- **Actions:** ACTION1-ACTION6 (5 directional keys + click coordinate) + ACTION7 (added v0.9.2) + RESET
- **Levels:** Minimum 6 per game, tutorial level 1 is intentionally easy
- **No instructions, no rules, no stated goals**
- **Core knowledge only:** Objectness, geometry, physics, agentness - NO numbers, letters, clip-art, or cultural conventions

### Dataset Split (Anti-Overfitting Design)
| Category | Count | Purpose |
|----------|-------|---------|
| Public Demo | 25 | Format demonstration, intentionally easier |
| Semi-Private | 55 | External API testing, moderate leakage risk |
| Fully Private | 55 | Official competition evaluation, maximum security |

**Critical:** Public set deliberately avoids mechanics overlap with private sets. Previous ARC versions suffered from overfitting between similar public/private sets. ARC-AGI-3 inverts the ratio (25 public vs 110 private).

### Submission Requirements
- **Platform:** Kaggle notebook only
- **Runtime:** Must complete in <12 hours
- **Internet:** NO access during evaluation (no API calls to external models)
- **Compute cost cap:** $10,000 USD maximum per evaluation run
- **Open source:** All code must be MIT or CC0 licensed to qualify for prizes
- **External compute:** Third-party providers (Modal, Lambda, RunPod) are allowed
- **Must be one-click runnable** with automated setup

---

## 6. Community Discussions & Perspectives

### HackerNews (news.ycombinator.com)

**Thread: "ARC-AGI-3" (item 47521150)**
- Active discussion about the benchmark's significance
- Debate about whether interactive RL is truly measuring intelligence vs game-playing ability
- Some argue AGI benchmarks are inherently unfalsifiable if the goalpost moves each iteration

**Thread: "Day 1 of ARC-AGI-3" (item 47538078)**
- Discussion of harness development approaches
- Mention of Agentica SDK as "a meta-harness that makes things easy"
- Debate about whether the benchmark selects for search algorithms rather than genuine learning

### Common HackerNews Criticisms
1. "The Arc prize/benchmark is a terrible judge of whether we got to AGI"
2. Moving goalpost criticism: if the bar moves every time it's cleared, it's not measuring AGI
3. Concern that it selects for search algorithms, not adaptive learning
4. Some view the human baseline framing as misleading

### ARC Prize Discord
- **Server:** discord.gg/9b77dPAmcA
- Active community for collaboration and discussion
- Researchers sharing approaches and debugging help

### Blog/Media Analysis

**Adam Holter's Critique ("The Human Baseline Is Rigged")**
- Second-best human run, not median, used as 100% baseline
- Even 2nd best human would score ~50% by their own metric
- Quadratic penalty makes <1% scores partly a scoring artifact, not purely a capability gap
- "A benchmark slightly adversarial to everyone is more honest than one calibrated to make AI look good"

**DEV Community Article**
- Highlights that interactive format requires sustained sequential reasoning, state tracking across hundreds of steps
- LLMs face practical token limits: hundreds of interaction steps = millions of tokens
- Concluded: "Systematic state tracking matters more than raw model size"

---

## 7. Strategy Recommendations from Community

### What Works (Proven)
1. **CNN frame-change prediction** (StochasticGoose approach) - best known method
2. **State graph construction** with action pruning (Blind Squirrel)
3. **Hierarchical action prioritization** based on visual salience
4. **Hash-based state deduplication** to maximize information per action
5. **Level-aware model resets** to prevent catastrophic forgetting
6. **Perceptual hashing** for state identification

### What Doesn't Work
1. **Frontier LLMs** - all score <1%, too expensive, no internet allowed
2. **Random exploration** - preview games were "too friendly to random search" but private games won't be
3. **Memorizing public games** - private set is deliberately out-of-distribution
4. **Curiosity-driven RL** - no guarantee of correlation with task progress in novel environments
5. **Pure world models (Dreamer family)** - limited by sparse reward signal (only level completion provides feedback)

### Promising But Unproven Approaches
1. **Hybrid architecture** - RL exploration core + reasoning layer + coordination protocol
2. **Meta-learning** for rapid adaptation across games
3. **Orchestrator-subagent** pattern (proven at 36% but requires API calls)
4. **Code execution for context compression** (Duke harness approach)
5. **Chollet's vision:** "programmer-like meta-learner" merging neural nets with discrete program search

### Tactical Tips
1. **Optimize for action efficiency from day one** - quadratic penalty means small inefficiencies compound fast
2. **Think like a human player**: test one thing, notice what changes, form a theory, test that
3. **Focus on later levels** - they dominate scoring due to linear weighting
4. **Completion > efficiency**: zero score for unfinished levels regardless of efficiency
5. **Use external compute** (RunPod, Modal) for serious training - Kaggle P100 is limited
6. **Action parsing defense**: use regex to extract valid action tokens, fallback to random when parsing fails
7. **Remove render_mode for speed** - 2K+ FPS without terminal rendering
8. **Cross-level transfer**: carry learned productive actions forward between levels

---

## 8. What Organizers Expect

From Chollet and the ARC Prize team:
1. **Genuine exploration and adaptation**, not memorization or brute-force
2. **Hybrid approaches** combining pattern recognition with rule-governed reasoning
3. **Test-time adaptation (TTA)** - agents that modify their own state on the fly
4. **Open-source everything** - required for prizes, philosophy of advancing the field
5. **Novel systems** preferred over wrappers around existing frontier models
6. **Efficiency over brute force** - the RHAE metric explicitly penalizes wasteful exploration

The competition organizers are watching for:
- Harness innovation (introduced community leaderboard specifically for this)
- Novel architectures that don't rely on existing LLMs
- Transfer learning between environments/levels
- Approaches that generalize to the private set, not just memorize public games

---

## 9. Known Issues & Gotchas

### Technical Issues
- **Kaggle JS-rendered pages**: Discussion forums hard to scrape - use browser or Kaggle MCP
- **FrameData field names changed in v0.9.2**: `score` -> `levels_completed`, `win_score` -> `win_levels`
- **ACTION7 added in v0.9.2**: Check SDK version
- **arcengine imports**: Use `from arcengine import GameAction`, not `from agents.structs`
- **Loop bugs in graph exploration**: Reset-inducing actions must be marked tested to prevent infinite resets

### Scoring Gotchas
- The quadratic penalty makes the <1% frontier scores seem worse than they are architecturally
- But the 5x action cap is a HARD stop - budget your actions carefully
- Tutorial levels (Level 1) are worth very little - don't over-optimize them
- Human baseline is 2nd-best from 10 testers, which is a high bar

### Submission Gotchas
- NO internet during Kaggle evaluation - no API calls to any external model
- Must be one-click runnable - all setup automated in notebook
- External compute providers ARE allowed (Modal, Lambda, RunPod)
- $10K max runtime cost cap
- 12-hour notebook runtime limit
- Must open-source by milestone deadline to be eligible for milestone prizes

---

## 10. Resources & Links

### Official
- Competition: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3
- Documentation: https://docs.arcprize.org/
- Scoring: https://docs.arcprize.org/methodology
- LLM Agents Guide: https://docs.arcprize.org/llm_agents
- ARC-AGI-3 Info: https://arcprize.org/arc-agi/3
- Play Games: https://three.arcprize.org/
- Discord: discord.gg/9b77dPAmcA

### Code & Repos
- ARC-AGI Toolkit: https://github.com/arcprize/arc-agi
- Agent Templates: https://github.com/arcprize/ARC-AGI-3-Agents
- StochasticGoose Solution: https://github.com/DriesSmit/ARC3-solution
- Arcgentica (Symbolica): https://github.com/symbolica-ai/ARC-AGI-3-Agents

### Papers
- Technical Report: https://arxiv.org/abs/2603.24621
- Graph-Based Exploration: https://arxiv.org/abs/2512.24156

### Analysis & Discussion
- 30-Day Learnings Blog: https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings
- Launch Blog: https://arcprize.org/blog/arc-agi-3-launch
- Day 1 Update: https://arcprize.org/blog/day-1-update
- Verified Testing Policy: https://arcprize.org/policy
- HN Thread 1: https://news.ycombinator.com/item?id=47521150
- HN Thread 2: https://news.ycombinator.com/item?id=47538078
- Scoring Critique: https://adam.holter.com/arc-agi-3-launch-sota-models-score-under-1-and-the-human-baseline-is-rigged/
- DEV Community: https://dev.to/codepawl/gpt-5-claude-gemini-all-score-below-1-arc-agi-3-just-broke-every-frontier-model-5dbj
