# AIMO3 Research: ippeiogawa (46/50) and Top Approaches

**Research Date**: 2026-04-01
**Researcher**: Claude Sonnet 4.6
**Goal**: Find anything useful about how ippeiogawa scored 46/50 and how to close the gap

---

## Part 1: What We Found About ippeiogawa

### Identity

- **Kaggle username**: ippeiogawa
- **Team ID**: 14801984
- **AIMO3 best score**: 46/50 (submitted 2026-03-31 17:34:48)
- **AIMO2 score**: 27/50 (rank ~17th)
- **AIMO2 notebook**: "generate code from deepseekR1" (last modified 2025-03-15)
  - Used `deepseek-r1-distill-qwen-14b-awq-casperhansen` in AIMO2
- **ARC Prize 2025**: Ranked 8th (LB 10.00/100) using test-time training (TTT)
  - Modified the ARChitects 2024 solution
  - Key ARC technique: 24 epoch TTT, DFS with top_k=4, 2 subprocess parallelism

### What We Know About Their AIMO3 Approach

**Almost nothing.** ippeiogawa has NO public AIMO3 notebooks. Their LB entry is entirely private.

**Background clues from ARC Prize work**:
- They are technically sophisticated (TTT, DFS, subprocess parallelism)
- They iterate heavily on existing top solutions (took ARChitects' code and tweaked it)
- They focus on hyperparameter optimization over architectural novelty
- They are willing to use aggressive inference tricks (parallelism, compute efficiency)

**Speculation based on jump from AIMO2 (27) to AIMO3 (46)**:
ippeiogawa's AIMO2 27/50 is mediocre - they were 17th. Their AIMO3 46/50 is #1. This is an extraordinary improvement. Possible explanations:
1. They adopted the full huikang pipeline with the fine-tuned model AND added something else
2. They implemented a superior voting/selection mechanism (GenSelect-style?)
3. They ran many more attempts by optimizing the inference loop
4. They used a different model configuration that nobody else found

### Submitted 2026-03-31 - Final Week

Their best score was submitted March 31, just before the end of competition. This suggests they may have been iterating throughout and found something late.

---

## Part 2: Full Leaderboard (as of 2026-04-01)

From Kaggle CLI:

| Rank | Team | Score | Last Submitted |
|------|------|-------|----------------|
| 1 | ippeiogawa | 46 | 2026-03-31 17:34 |
| 2 | just public 44, all is luck | 45 | 2026-03-01 09:06 |
| 3 | Batman's Butler | 45 | 2026-04-01 13:48 |
| 4 | Riku Suzuki | 45 | 2026-04-01 08:08 |
| 5 | Seungjun Lee | 45 | 2026-04-01 01:32 |
| 6 | i.won.a.maths.debate | 44 | 2026-03-31 23:31 |
| 7-20 | Various | 44 | 2026-03-31 |

**Key observation**: "just public 44, all is luck" reached 45 on March 1 - over a month before deadline. Their team name suggests they believe variance (luck) is a major factor. This is consistent with the arxiv paper finding σ=1.7 points across runs.

---

## Part 3: Key Scientific Paper (arxiv 2603.27844)

"Model Capability Dominates: Inference-Time Optimization Lessons from AIMO 3"
by Natapong Nitarach

This paper is essential reading. It ran 23+ experiments on AIMO3 and documents every failure.

### Three Models Tested

| Model | Parameters | Active Params | LB Score |
|-------|-----------|---------------|----------|
| gpt-oss-120b | 116.8B | 5.1B (MoE) | 39.7 mean, peak 44 |
| Qwen3.5-35B-A3B | 35B | 3B (MoE) | 23 |
| Nemotron-Super-120B | 120B | 12B | 23 |

**Conclusion**: Only gpt-oss-120b can reach 40+. The capability gap is 17 points (69% vs 46% per-attempt accuracy).

### Temperature Ablation (definitive)

| Temperature | Score | Delta vs Baseline |
|------------|-------|-------------------|
| T=0.5 | 38 | -1.7 |
| T=0.8 | 40 | +0.3 (BEST) |
| T=1.0 | 39.7 | baseline (13-run mean) |
| T=1.2 + min_p=0.03 | 37 | -2.7 |

### What Actually Works

The paper found one genuine improvement: T=0.8 gives +0.3 vs T=1.0 baseline.

Everything else they tried failed:
- Diverse prompt strategies: every single one reduced score
- Equal prompt mixing (2+2+2+2): 36 (-3.7)
- Small-cases-first: 37 (-2.7)
- Code-first variant: avg 37.7 (-2.0)
- N=16 attempts: no improvement over N=8
- T=0.5: 38 (-1.7)

### Sample Count Analysis

- N=8, T=1.0: error correlation ρ̂ = -0.113 to -0.258 (NEGATIVE)
- This means attempts are already anti-correlated
- Effective N at ρ=-0.258 is already near the ceiling
- Adding more attempts provides diminishing returns

**Baseline configuration (paper's finding)**:
```
N=8, T=1.0, min_p=0.02, 16 parallel workers, Jupyter sandbox
Code: tool_call tags → Python execution
Voting: majority vote
Time budget: 342 seconds per problem (~5.7 min)
```

---

## Part 4: Pawan Mali's 50+ Experiments (42/50 best, 41/50 current best)

Source: notebook `pawanmali/chasing-47-50-aimo3-journey-of-50-experiments` (pulled 2026-04-01)

**Best score**: V40 = 42/50 (Feb 6, 2026), current best V125 = 41/50

### What Worked

1. **Temperature 1.0**: V41 with T=0.5 scored 34 (-8 regression). High temperature is essential.
2. **5-component weighted entropy** (+5-6 points vs simple mean):
   ```python
   # 40% weight on position-weighted (exponential decay, recent tokens count more)
   # 30% weight on base mean entropy
   # 20% weight on variance penalty
   # 30%*3.0 high-entropy-stretch penalty
   # -10% low-entropy-streak bonus
   ```
3. **Simple prompts** (+3-5 points vs complex): "You are a world-class IMO competitor. Final answer must be 0-99999 in \\boxed{}."
4. **context_tokens = 65536** NOT 131K (131K cuts concurrency from 7.11x to 3.35x, scored -6)
5. **Inverse entropy weighted voting**: weight = 1.0 / max(entropy, 1e-9)
6. **8 attempts, early_stop=4** (stop when 4 of 8 agree)

### What Failed

| Experiment | Score Change |
|------------|-------------|
| top_p=0.9 | -8 (catastrophic) |
| 131K context | -6 |
| Simple mean entropy (vs 5-component) | -6 |
| huikang fine-tuned model alone | -6 |
| 6 techniques combined | -6 |
| Self-refinement loop | -4 |
| Answer verification (CORRECT/WRONG) | 0 (no improvement) |
| Very verbose prompts | -9 |

### Key Insights from 50+ Experiments

1. **huikang fine-tuned model alone = -6**: "Fine-tuning requires matching the entire inference pipeline, not just swapping models." The fine-tuned model was trained with huikang's 131K context, specific KV cache, and specific prompts. Drop-in replacement fails.

2. **Answer verification adds 0 improvement**: "The model is equally likely to verify wrong answers as correct."

3. **Self-refinement hurts (-4)**: "The model 'corrects' correct answers to wrong ones."

4. **Temperature 1.0 + weighted entropy = best combination for this author's pipeline.**

---

## Part 5: The Gap Analysis — 42 to 47

### Current Reality

| Score | Who | How |
|-------|-----|-----|
| 46 | ippeiogawa | Private, unknown |
| 45 | 4 teams | Unknown approaches |
| 44 | 14 teams | Mix of approaches |
| 42 | Pawan Mali's best | T=1.0 + weighted entropy + simple prompts |
| 41 | nihilisticneuralnet (public 44/50 is latest, earlier shows 41-42) | T=0.5, simple entropy |
| 39.7 | arxiv paper baseline | N=8, T=1.0, majority vote |

### What 44/50 vs 46/50 Requires

From per-attempt accuracy p̂ = 0.69 (gpt-oss-120b baseline):
- Expected majority vote score = 39.7 (actual data)
- 44/50 requires p̂ to be ~0.78 at the problem level (better per-attempt accuracy)
- 46/50 requires p̂ ~0.83+ or a fundamentally better selection mechanism

**The 2-point gap between 44 and 46 is likely due to one of**:
1. A fine-tuned model with higher per-attempt accuracy on the specific 6 problems missed
2. A much better answer selection/verification that recovers 2 near-misses
3. Longer timeouts allowing more TIR tool calls on the hardest problems
4. Lucky run (σ=1.7 means 46 is ~3.7σ above mean — possible but rare)

---

## Part 6: Approaches Nobody Has Tried Publicly

### 6.1 EAGLE-3 Speculative Decoding for GPT-OSS-120B

**GitHub**: `juemifuji/eagle3-aimo3`

This is a fork of SpecForge that:
- Trains a draft model to predict gpt-oss-120b tokens
- Achieves 36-42% speedup on H100 (776 → 1059 tokens/sec at 8 concurrent requests)
- Enables context lengths beyond 40K on 8xH800

**AIMO3 impact if this works**: With 40% speedup, you can run N=11-12 attempts instead of N=8 in the same 5-hour window. Or equivalently, each problem gets ~40% more token budget.

**Why it might not matter**: The arxiv paper showed N=16 doesn't improve over N=8 (negative correlation already saturates). The speedup would need to be directed at longer per-problem timeouts, not more attempts.

**Actionable**: Deploy EAGLE-3 to get longer per-problem timeouts (e.g., 380s vs 270s) on the hardest problems. This may unlock 1-2 extra solutions on problems that need very long TIR chains.

### 6.2 GenSelect (AIMO2 Winner's Technique)

From the AIMO2 winning paper (arxiv 2504.16891):
- After generating N candidates, a *learned* selector model picks the best answer
- The selector was trained on (problem, candidate_solutions) → correctness labels
- Significantly outperformed majority voting baseline

**For AIMO3**: A selector fine-tuned on AIMO3-style problems could replace entropy-weighted voting. With 8 candidate solutions per problem, the selector sees all 8 and picks the most likely correct one.

**Why this could beat 44**: Entropy-weighted voting is a heuristic. A trained verifier/selector may catch cases where the high-entropy solution is actually correct (when the model is exploring a difficult but correct approach).

**Implementation complexity**: Requires a separate smaller model (1-7B) trained on (problem, solution) → {correct, wrong}. The training data can be generated from AIME/AMC problems with known answers.

### 6.3 Two-Model Pipeline: GPT-OSS-120B + Specialized Math Model

**Current best (ZaynYu, ~40/50)**: GPT-OSS-120B + Qwen3-30B-A3B in parallel

**The idea**: Use GPT-OSS-120B for most problems, but route specific problem types to a specialized model.

For geometry problems specifically, DeepSeek-Math-V2 (which scored gold-level on IMO 2025) may have been specifically fine-tuned on geometric reasoning in a way that GPT-OSS-120B lacks.

**Practical constraint**: Running two 120B models won't fit. But running GPT-OSS-120B + a smaller specialized model (7-14B) is feasible if the smaller model only handles a few problems.

**Problem classification**: Detect geometry/combinatorics problems in the prompt, route to specialized model, use GPT-OSS-120B for algebra/number theory.

### 6.4 Reasoning Token Analysis for Voting

**The concept**: GPT-OSS-120B (like o3) produces reasoning tokens before the final answer. The reasoning quality itself is a signal for answer correctness.

**Untested idea**: Instead of using token entropy to weight votes, use the *coherence* of the reasoning chain as a weight:
- Parse the reasoning tokens and count "contradiction" phrases ("wait, this is wrong")
- Count successful tool call outputs (code ran without error)
- Use these as weights in addition to final entropy

**Why this might work**: Entropy measures uncertainty at each token, but a logically coherent reasoning chain with confident final tokens is a better signal than entropy alone.

---

## Part 7: What Probably Explains ippeiogawa's 46/50

### Most Likely Hypothesis

Based on all evidence:

1. **They use huikang's FULL pipeline** (not just the model) — the 131K context, the streaming inference, the VOI stopping — because the fine-tuned model was designed for that specific pipeline. Pawan Mali's V128 failure was using the fine-tuned model with the *wrong* pipeline.

2. **Plus the 5-component weighted entropy** or a Bayesian posterior variant (Jonathan Chan style), adding 1-2 points on top of the already-strong huikang pipeline.

3. **Or** they found a way to run more effective attempts — perhaps using EAGLE-3 or a custom serving setup that gives them 40% more compute per problem.

### Why the ARC Prize Background Matters

ippeiogawa's ARC work (8th place, 10% on ARC Prize 2025) shows they:
- Build on the best existing public solution and tweak hyperparameters
- Implement test-time adaptation (TTT in ARC context)
- Use aggressive parallelism (2 subprocesses)

Applied to AIMO3: they likely took the best public AIMO3 pipeline (huikang or nihilisticneuralnet), studied it carefully, and added a specific improvement that others missed.

### The "just public 44, all is luck" Clue

This team's name is significant: they are saying their public score (44) is pure luck and their private score might differ significantly. They submitted on March 1 and haven't improved since. This team name is a signal that the variance in scores is real and high.

---

## Part 8: Actionable Recommendations

### Priority 1: Fix the huikang Model Integration

Pawan Mali's V128 failed (-6) because he used huikang's model with the wrong pipeline. The correct approach is to use huikang's **full pipeline** (131K context, streaming, his voting mechanism).

- Use `huikang/gpt-oss-120b-aimo3/Transformers/160a/20`
- Use `--max-model-len 131072` (not 65536)
- Use huikang's VOI stopping criterion
- Use huikang's 3-factor Bayesian weighting

**Expected impact**: +1 to +3 points (restores the fine-tuning benefit that gets lost without the matching pipeline).

### Priority 2: 5-Component Weighted Entropy

Pawan Mali's V136 showed simple entropy → -6 points. The 5-component version adds +5-6 points. Our current notebooks may be using simple entropy or a suboptimal weighting.

```python
def compute_5_component_entropy(logprobs_sequence):
    entropies = [-sum(p * log(p) for p in dist) for dist in logprobs_sequence]
    n = len(entropies)
    
    # 1. Base mean (30%)
    mean_ent = sum(entropies) / n
    
    # 2. Position-weighted (40%) - exponential decay, recent tokens weighted more
    weights = [0.995 ** (n - 1 - i) for i in range(n)]
    pos_weighted = sum(e * w for e, w in zip(entropies, weights)) / sum(weights)
    
    # 3. Variance penalty (20%)
    variance = sum((e - mean_ent)**2 for e in entropies) / n
    std_dev = variance ** 0.5
    
    # 4. High entropy penalty (30% * 3.0)
    high_ent_ratio = sum(1 for e in entropies if e > 2.0) / n
    
    # 5. Low entropy streak bonus (-10%)
    max_streak = 0
    cur_streak = 0
    for e in entropies:
        if e < 0.5:
            cur_streak += 1
            max_streak = max(max_streak, cur_streak)
        else:
            cur_streak = 0
    streak_bonus = -0.1 * (max_streak / n)
    
    return (0.3 * mean_ent +
            0.4 * pos_weighted +
            0.2 * std_dev +
            0.3 * high_ent_ratio * 3.0 +
            streak_bonus)
```

**Expected impact**: +5-6 points if currently using simple mean entropy. Already implemented in some of our notebooks but verify it's the full 5-component version.

### Priority 3: Keep Simple System Prompts

Pawan Mali's V127 with verbose prompts scored -9. Our prompts should be 3 lines max:

```python
system_prompt = (
    'You are a world-class IMO competitor. '
    'The final answer must be a non-negative integer 0-99999. '
    'Place your final answer inside \\boxed{}.'
)
```

Remove all multi-step instructions, category-specific guidance, and "think carefully" filler.

### Priority 4: Temperature 1.0 (Not 0.5 or 0.8)

Despite the arxiv paper showing T=0.8 as best for *mean score*, Pawan Mali's V41 with T=0.5 scored -8. Competition scores are peaks, not means. T=1.0 with good weighting seems to be what works in practice.

**Exception**: If using top_p, use either top_p=0.8 or no top_p. top_p=0.9 scored -8.

### Priority 5: Do NOT Use These Techniques

Based on evidence from 50+ experiments and the arxiv paper:

| Technique | Evidence for Avoiding |
|-----------|----------------------|
| Answer verification (CORRECT/WRONG) | 0 improvement in V133, reduces time |
| Self-refinement | -4 in V134 |
| 131K context without matching pipeline | -6 in V135 |
| Diverse prompts across attempts | Every experiment in arxiv paper failed |
| More than 8 attempts | Diminishing returns, time cost |
| top_p=0.9 | -8 in V126 |
| Verbose/structured prompts | -9 in V127 |
| Simple mean entropy | -6 in V136 |

---

## Part 9: The Ceiling Question — Can We Reach 46?

### Mathematical Analysis

From arxiv paper (p̂ = 0.69 per attempt for gpt-oss-120b):
- P(score ≥ 46) at mean 39.7, σ=1.7 ≈ 0.01% (1 in 10,000)
- This makes 46/50 essentially impossible through luck alone

**ippeiogawa's 46 implies a higher p̂ value.** If we estimate their baseline p̂:
- 46/50 with N=8 and majority vote requires p̂ ≈ 0.82-0.85 per attempt
- This is 13-16 percentage points above the base model's 0.69

This means ippeiogawa is almost certainly using either:
(a) A fine-tuned model with significantly higher per-attempt accuracy, OR
(b) A selection mechanism that is much better than majority voting

### What 45 Requires

45/50 with T=1.0 and N=8 requires mean p̂ ≈ 0.80. Given our current approach may be achieving 0.69-0.73, we need a 7-11 percentage point improvement in per-attempt accuracy.

Sources of this improvement:
- Fine-tuned model matching full pipeline: +3-6 points on per-attempt accuracy
- Better geometry/combinatorics prompting: +1-2 points per-attempt on those categories
- Longer problem timeouts via EAGLE-3: +1-2 points on hardest problems

Combined, these might push us to 44-45.

### Time Budget (14 days remaining as of 2026-04-01)

The competition deadline is April 15, 2026. We have 14 days and ~5 more submissions per day = ~70 total submissions.

Priority actions:
1. Wait for v25 score to determine if huikang model + VOI stopping works
2. If v25 fails, implement full huikang pipeline with 131K and 5-component entropy
3. Try EAGLE-3 for per-problem timeout extension
4. Minimize approach combinations — test one change at a time

---

## Sources

1. Kaggle CLI leaderboard: `kaggle competitions leaderboard ai-mathematical-olympiad-progress-prize-3 --show`
2. Arxiv paper 2603.27844 (HTML version): Full ablation table, temperature data, correlation analysis
3. Pawan Mali notebook (pulled via `kaggle kernels pull`): 50+ experiment full results
4. ippeiogawa notebook `lb10-00-with-tweak-to-2024-1st-solution`: ARC Prize 2025 approach
5. AIMOPRIZE.COM gap analysis: o3-preview commercial performance (46-50/50)
6. Existing research: `huikang_model_research.md`, `deep_research_42_to_46.md`
7. EAGLE-3 GitHub (`juemifuji/eagle3-aimo3`): 40% speedup on gpt-oss-120b
8. Kaggle search: `kaggle kernels list -s "ippeiogawa"` — confirmed no public AIMO3 notebooks
9. tonghuikang GitHub: Template code using vLLM + Modal deployment
10. AIMO2 leaderboard: ippeiogawa was 17th/27 in AIMO2
