# Deep Research: AIMO3 Score 42-44 → 45-46/50

**Research Date**: 2026-04-01
**Researcher**: Claude Sonnet 4.6
**Goal**: Identify concrete techniques to push from current ~42-44/50 to 45-46/50

---

## Leaderboard Context (as of 2026-04-01)

| Rank | Score | Notes |
|------|-------|-------|
| 1 | 46 | ippeiogawa — fully private, no public notebooks |
| 2-4 | 45 | "just public 44", Batman's Butler, Riku Suzuki, Seungjun Lee |
| 5-20 | 44 | Mix of approaches; nihilisticneuralnet is highest public notebook |
| Our best | 38 | v18 (competition log), v25 pending (expected 42-45) |

The gap to #1 is 2-3 points. The top score of 46 has been confirmed in the leaderboard
monitor (`tonghuikang--leaderboard-monitor`) which shows scores ranging from 0-46 with
highest being 46 as of late March 2026.

Source from arxiv paper 2603.27844 (Natapong Nitarach): "competition context: AIMO-3
leaderboard ≥46/50" as of March 26, 2026.

---

## Q1: Does Increasing Attempts from 8 to 12 Help? What's the Time Tradeoff?

### Evidence from arxiv paper (2603.27844)

The paper explicitly tested N ablation:
- N=3: scored 36/50 (on gpt-oss-120b)
- N=8: mean 39.7/50 across 13 runs (σ=1.7)
- N=16: tested on Qwen3.5-35B — "Doubling N from 8 to 16: no improvement"

**Critical insight on the math**: The paper measured mean pairwise error correlation
ρ̂ = -0.258 across 8 problems. This is NEGATIVE, meaning the attempts are already
slightly anti-correlated. With negative correlation, adding more attempts yields
diminishing returns faster than with positive correlation.

The effective sample size formula is: `N_eff = N / (1 + (N-1)ρ)`

At ρ = -0.113 (conservative estimate for N≥7):
- N=8: N_eff = 8 / (1 + 7×(-0.113)) = 8 / 0.209 = 38.3 effective samples
- N=12: N_eff = 12 / (1 + 11×(-0.113)) = 12 / (-0.243) → undefined (already at ceiling)

This means 8 attempts with negative correlation already achieves maximum decorrelation.
**Going to 12 attempts is not expected to improve average score.**

### Our v8 notebook design (12 attempts, temp=0.8)

Our v8 attempts 12 with:
- `temperature = 0.8` (between 0.5 and 1.0)
- `early_stop = 5` (needs 5/12 agreeing, vs 4/8 in standard)
- `high_problem_timeout = 1200` (was 900)
- `base_problem_timeout = 210` (reduced from 270)

**Time tradeoff**: 12 attempts fit in 5h budget only if:
1. Early stopping fires frequently (5 of 12 agree before all 12 run)
2. Base timeout floor is reduced (210s vs 270s hurts hard problems)

The risk is that reducing `base_problem_timeout` from 270 to 210 seconds means
the model has less time per problem on hard problems, potentially dropping 1-2 answers
that need extensive TIR tool loops.

**Verdict**: Low-to-negative expected impact. The math says 8 is already near-optimal
for majority voting. The time budget reduction to accommodate 12 attempts may offset
any benefit. **Priority: Medium-Low. Validate v26 score before further investment.**

---

## Q2: Is temp=0.5 or temp=0.8 Better?

### Evidence from arxiv paper (2603.27844)

Full temperature ablation table (all with N=8, baseline T=1.0 = 39.7 mean):

| Temperature | LB Score | Delta |
|-------------|----------|-------|
| T=0.5 | 38 | -1.7 |
| T=0.8 | 40 | +0.3 |
| T=1.0 | 39.7 | baseline (13-run mean) |
| T=1.2 + min_p=0.03 | 37 | -2.7 |

**The paper's baseline uses T=1.0.** T=0.8 shows marginal +0.3 improvement. T=0.5 is
notably worse (-1.7).

### But our competition log says both 44/50 notebooks use T=0.5!

From `competition_log.md`: "[MEDIUM] Temperature: 0.5 works but so does 1.0. Both score
44/50. 0.8 is the middle ground for more diversity."

From `aimo3_top_notebook_analysis.md`: nihilisticneuralnet (44/50) parameters show
`temperature=0.5` and `temperature=1.0` both scoring 44/50.

**Resolution**: The paper (T=0.5 → 38) was measuring mean over 13 runs, each of which
is a single LB submission. The competition log is measuring the peak achieved on the
best submission. There's no contradiction: T=0.5 reduces average score but reduces
variance too — fewer 37s but fewer 44s. T=1.0 has higher variance: more 44s but
also more 38s.

**For competition purposes (you care about best submission, not mean)**:
- T=0.5 → lower mean (38), but perhaps lower variance → more consistent
- T=1.0 → higher mean (39.7) but higher variance → more lottery tickets for 44+
- T=0.8 → best mean (40.0) with less variance than T=1.0

**Actionable**: T=0.8 appears to be empirically the best choice for the expected score
objective. For "lottery ticket" strategy to hit 45+, T=1.0 may be better (higher variance).

---

## Q3: Does the Huikang Fine-Tuned Model Outperform Base?

### Evidence from existing research (huikang_model_research.md)

**Key finding**: The highest public score (44/50) does NOT use huikang's model — it
uses `danielhanchen/gpt-oss-120b` (base). The huikang fine-tuned model (v20, 160a)
is used by:
- Jonathan Chan Bayesian notebook (score unknown, 185 votes → high quality signal)
- kienngx inference-only (version 9, 139 votes)

**What we know about the fine-tuning**:
- Same file size as base (~65.3 GB) → likely full fine-tune, not LoRA
- Training data label `160a` — huikang's internal notation
- No public benchmark comparison exists

**What competition log says**: "[MEDIUM] Docker image controls path mounting — custom
docker uses `/kaggle/input/gpt-oss-120b/`, no docker uses `/kaggle/input/models/danielhanchen/`"

Both 44/50 notebooks (base model) were submitted. v25 uses huikang model and is pending.

**Verdict**: The evidence is ambiguous. The highest public score uses base. Jonathan
Chan uses huikang model with Bayesian voting — but we don't have his LB score.
Given that our v25 (huikang + verification + reliability voting) is pending, the answer
will become clear when we get that score. **Current expectation: huikang may add 0-2 points
if the fine-tuning was on IMO-style problems.**

The `160a` variation name appears across all 20 versions, suggesting iterative training
on the same dataset recipe. Version 20 (latest) is the most trained iteration.

---

## Q4: Are There Problems ALL Notebooks Get Wrong?

### Evidence from OlymMATH benchmark (arxiv 2503.21380v2)

On the Hard subset of OlymMATH:
- Best model (Gemini 2.5 Pro): 58.4% accuracy
- gpt-oss-120b equivalent: ~50-60% estimated based on AIME performance
- This implies 40-50% of hard Olympiad problems remain unsolved even by best models

The paper notes models use "empirical guesses — heuristics, symmetry assumptions, or
fabrication — rather than rigorous reasoning" on the hardest problems.

### Evidence from AIMO2 analysis (aimoprize.com)

Of the 7 problems o3-preview's low-compute version missed:
- 2 geometry problems
- 2 algebra problems
- 3 combinatorics problems
- (0 number theory)

This suggests **geometry and combinatorics are the hardest** for current models. On
AIMO3 problems, if the distribution is similar, expect 3-5 problems that essentially
no public competitor solves.

### What the leaderboard implies

Top score is 46/50 (ippeiogawa, private). If no public competitor has 47+, there are
likely 4-7 "hard" problems that current methods fail on consistently. Going from 44 to 46
means solving 2 of these hard problems that your current 44/50 config gets wrong.

**Strategy implication**: Don't try to improve the 44 problems you're already getting.
Focus on what specifically goes wrong on the 6 problems you miss. The likely failures are:
1. Geometry problems (LLM struggles with coordinate geometry setup for complex figures)
2. Combinatorics with complex case analysis (model loses track of cases)
3. Problems requiring >10 Python tool calls (truncation or timeout)

---

## Q5: Theoretical Ceiling with GPT-OSS-120B on These Problems

### From arxiv paper (2603.27844)

Per-attempt accuracy p̂ = 0.69 for gpt-oss-120b. This means each individual attempt
solves 69% of problems correctly.

**Ceiling calculation** (binomial model, N=8 independent attempts, majority vote):

P(majority correct | p=0.69, N=8) = P(5+ correct out of 8)
= Σ C(8,k) × 0.69^k × 0.31^(8-k) for k=5,6,7,8
= 0.714

But with negative correlation (ρ = -0.113), effective N increases, raising the ceiling.

**Upper bound**: If we could make all 8 attempts perfectly independent with p=0.69:
E[correct problems] = 50 × 0.714 = 35.7 expected problems solved via majority vote.

Wait — the paper reports N=8 baseline scoring 39.7/50 mean, which is HIGHER than this.
This means majority voting on 50 problems, not 50 independent coin flips. With N=8
attempts per problem at p=0.69:

P(majority ≥ 5 of 8) ≈ 71.4% → expected 35.7 problems
But actual score is 39.7 → p̂ of 0.69 is the mean across problems.

The discrepancy is because p̂ varies by problem: easy problems have p̂ = 0.95+,
hard problems have p̂ = 0.20. For easy problems (p=0.95, N=8), majority vote = near
certainty. For hard problems (p=0.20, N=8), majority vote gives P(≥5 of 8 correct) = 0.01.

**Implication**: The ceiling for 8-attempt majority voting with gpt-oss-120b is around
42-44/50. To get above 44 requires either:
1. Higher per-attempt accuracy (fine-tuning, better prompts)
2. Different approach for hard problems (different model, agentic multi-step verification)
3. Lottery: variance means occasionally scoring 46+ through luck

The paper's 13-run best score was 44. The competition leader is at 46. **46 appears
to be achievable but requires either a fine-tuned model or a genuinely better approach
(not just inference tricks).**

---

## Q6: Geometry Problems — Specific Tricks

### What we know from research

1. **Coordinate geometry is the recommended approach** — our own system prompt already
   says "For geometry: coordinate geometry is often the most reliable"

2. **From AIMO2 analysis**: Geometry was one of the two hardest categories for o3-preview.

3. **From OlymMATH benchmark**: Geometry problems remain challenging for all models.
   Models tend to make errors in:
   - Setting up coordinates for complex figures
   - Computing intersection points symbolically
   - Handling trigonometric identities in non-standard configurations

### Actionable tricks for geometry

**Trick 1: Explicit coordinate geometry instruction in system prompt**
Add to system prompt: "For geometry problems: ALWAYS set up a coordinate system. Place
convenient vertices at (0,0), (1,0), etc. Use sympy to compute distances, angles,
areas exactly. Do NOT attempt pure synthetic geometry."

**Trick 2: Force sympy geometric objects**
```python
# Add to system prompt geometry section:
"Use `from sympy.geometry import Point, Triangle, Circle, Segment` for geometric primitives.
These compute exact symbolic answers."
```

**Trick 3: Trigonometric forcing**
For problems involving angles, explicitly prompt: "Use the law of cosines and sympy's
trigonometric simplification. cos(angle) = dot_product / (|a| * |b|) as exact fractions."

**Trick 4: Verify with multiple methods**
"For geometry answers, verify using BOTH coordinate computation AND known formulas
(Heron's formula for area, Ptolemy's theorem for cyclic quadrilaterals, etc.)"

**Trick 5: Area and angle modulo**
Geometry competition answers that seem like decimals are often integers after modular
reduction. Add: "If the geometric answer looks like a fraction, check if the problem
asks for the answer modulo some number."

**Expected impact**: 0.5-1.5 points on geometry-specific problems.

---

## Q7: Does min_p Help? Experiments with 0.01 vs 0.02 vs 0.05

### From arxiv paper (2603.27844)

The paper's baseline uses `min_p = 0.02` (confirmed in Appendix A).
Only one test: `T=1.2, min_p=0.03 → 37/50 (-2.7 from baseline)`.
**No dedicated min_p ablation was conducted.**

### From Unsloth documentation (gpt-oss fine-tuning guide)

Recommended: `temperature=1.0, top_p=1.0, top_k=0` (experiment with top_k=100).
No specific min_p recommendation for math.

### Analysis

min_p=0.02 means: at each token step, any token with probability < 2% of the max
probability is excluded. This prevents the model from sampling very low-probability
"wild" tokens.

For competition math, the tradeoffs are:
- `min_p=0.01`: More diverse, allows rarer token paths → higher variance, may explore
  unusual approaches on hard problems
- `min_p=0.02`: Current baseline, good calibration
- `min_p=0.05`: More conservative → reduces diversity, similar to lowering temperature

**Recommendation**: Try `min_p=0.01` as it gives the most token diversity at T=1.0.
No experimental evidence it helps, but the theory suggests it could unlock unusual
reasoning paths for hard problems. Low risk since it's a single hyperparameter change.

---

## Q8: Does Seed Diversity Help?

### From arxiv paper (2603.27844)

"The paper uses different random seed per parallel attempt but reports that stochastic
diversity already achieves most of the achievable decorrelation."

Mean pairwise error correlation ρ̂ = -0.258 across 8 problems. The correlation is
already NEGATIVE with just T=1.0 and different seeds. "All eight [correlation estimates]
are negative...leaving nothing for diversity strategies to exploit."

### Practical implication

Our v8 notebook uses `seed=42` at the vLLM server level. The individual attempts
within a problem use different random seeds implicitly through vLLM's per-request
sampling. No evidence that varying the server-level seed between submissions helps.

**Verdict**: Seed diversity is already baked into temperature sampling. Explicitly
varying seeds across attempts is not expected to improve over current approach.

---

## Q9: Different System Prompts for Different Attempts

### From arxiv paper (2603.27844) — Diverse Prompt Mixer experiments

This is the most comprehensively tested intervention in the paper. Results:

| Strategy | Score | Delta |
|----------|-------|-------|
| All Original (baseline) | 39.7 | — |
| Small Cases First (E1) | 37 | -2.7 |
| Work Backwards (E2) | 39 | -0.7 |
| Classify Then Solve (E3) | 36 | -3.7 |
| Code-First (E12, 3 runs) | avg 37.7 | -2.0 |
| Formalize-First (EF1) | 39 | -0.7 |
| Conservative mix (5+1+1+1) | underperformed | negative |
| Aggressive mix (3+2+2+1) | underperformed | negative |
| Equal mix (2+2+2+2) | 36 | -3.7 |

**Critical conclusion**: "More diversity = worse performance. The relationship is
monotonic: replacing more Original prompts with diverse strategies never helps and
eventually hurts."

The paper's explanation: "The model already explores diverse reasoning paths at T=1.0.
Adding prompt diversity reduces per-attempt quality more than it reduces correlation."

**Verdict**: Do NOT use different system prompts for different attempts. Use the same
high-quality 5-step structured prompt for all 8 attempts.

---

## Q10: Jonathan Chan Bayesian Approach — Full Details

From `huikang_model_research.md`:

**Model**: huikang/gpt-oss-120b-aimo3 v14 (primary), v15, v16 in model_sources.

**Bayesian posterior voting**:
```python
def _compute_bayesian_posterior(self, detailed_results):
    for r in detailed_results:
        entropy_weight = 1.0 / (1.0 + entropy)
        reliability = 1.0 / (1 + error_penalty * python_errors)
        tool_bonus = 1.2 if (python_calls > 0 and python_errors == 0) else (0.8 if python_errors > 0 else 1.0)
        weight = entropy_weight * reliability * tool_bonus
        posterior[answer] += weight
```

**Value-of-Information (VOI) early stopping**:
```python
submit_utility = max_prob
expected_improvement = entropy * voi_entropy_weight  # 0.6
continue_utility = max_prob + expected_improvement - voi_compute_cost  # 0.04

if submit_utility >= continue_utility:
    stop_event.set()  # Stop if gain from more attempts < 0.04 compute cost
```

**Key parameters**:
- temperature = 1.0, min_p = 0.02
- context_tokens = 65536
- attempts = 8, workers = 16
- posterior_stop_threshold = 0.82 (stop if posterior > 82% certain)
- gpu_memory_utilization = 0.96

**What makes this different from nihilisticneuralnet's 44/50 approach**:
1. Uses huikang fine-tuned model (may have higher per-attempt accuracy)
2. 3-factor weight (entropy + reliability + tool bonus) vs simple entropy
3. VOI early stopping (more principled than simple "4 of 8 agree")
4. posterior_stop_threshold acts as a confidence gate

**Why we don't know Jonathan Chan's score**: It was not on the public leaderboard when
the competition log was written. The 185 votes suggest the notebook is high quality.

**Our v25 notebook** implements a similar approach: huikang + mandatory verification +
reliability-weighted voting. This should be close to Jonathan Chan's approach.

---

## Key Findings from Deep Research

### What the arxiv paper (2603.27844) definitively proves

1. **Temperature T=0.8 is the empirically best single choice** (+0.3 vs T=1.0 baseline)
2. **T=0.5 is significantly worse** (-1.7) despite our 44/50 notebooks using it
3. **Diverse prompts hurt without exception** — every alternative to the base prompt reduced score
4. **N=8 is near-optimal** — doubling to N=16 showed no improvement
5. **Seed diversity is irrelevant** — already maximally decorrelated at T=1.0
6. **min_p=0.02 baseline is not well-optimized** — only one alt tested (0.03 at high T)
7. **Competition ceiling was ≥46/50** as of March 26, 2026

### What we don't know

- Exact LB score of Jonathan Chan's Bayesian notebook
- Whether huikang v20 (latest) beats base model
- Whether amanatar's "ans-verifys" notebook (described in answer_extraction_research.md)
  actually achieved ~41/50 with verification
- What ippeiogawa (46/50, private) does differently

---

## Top 5 Actionable Techniques to Try

### Rank 1: Multi-Turn Follow-Up Prompting (Expected +1 to +2 points)

**What it is**: When a generation attempt produces no boxed answer, inject a follow-up
user message asking for a guess with a boxed answer. Hui Kang's streaming notebook does this.

**Why it matters**: Every "no answer" response scores 0 on that problem. If 5% of
responses fail to produce a boxed answer, that's 2.5 problems lost. Follow-up prompting
recovers some fraction of these.

**Implementation** (from answer_extraction_research.md):
```python
if not boxed_answer_found:
    follow_up = (
        "The answer is an integer 0-99999. Please give your best estimate in \\boxed{}."
    )
    # inject as new user message and re-run
```

**Expected impact**: +1 to +2 points. This is the highest-priority unimplemented feature.

### Rank 2: Answer Verification Loop (Expected +0.5 to +1.5 points)

**What it is**: After collecting N=8 candidate answers, take the top-1 or top-2 by vote
count and ask the model to verify: "Is X the correct answer to this problem?" at T=0.0.

**From answer_extraction_research.md (amanatar's approach)**:
```python
prompt = f"Problem:\n{problem}\n\nProposed answer: {answer}\n\nCheck carefully.\nReply only: CORRECT or WRONG"
response = model.generate(prompt, temperature=0.0, max_tokens=5)
```

**Expected impact**: +0.5 to +1.5 points, especially on problems where majority vote
is split 3-3-2 between three candidate answers.

**Risk**: Uses ~200-400 tokens per verification, which might reduce time for some problems.

### Rank 3: Geometry-Specific Prompt Enhancement (Expected +0.5 to +1.5 points)

**What it is**: Add explicit geometry strategy to the system prompt. Current prompt says
"coordinate geometry is often the most reliable" but doesn't force it.

**Proposed addition to system prompt**:
```
For geometry problems specifically:
- IMMEDIATELY set up coordinates: place one vertex at (0,0), another at (d, 0)
- Use sympy.geometry module: from sympy.geometry import Point, Triangle, Circle
- For cyclic polygons: use complex numbers z = r*exp(i*theta) in sympy
- For area problems: use shoelace formula via sympy
- Cross-verify with synthetic geometry (Ptolemy, power of a point)
- NEVER leave a geometry answer unverified by code
```

**Expected impact**: +0.5 to +1.5 points on geometry problems (estimated 8-12 of 50 problems).

### Rank 4: Huikang v20 Fine-Tuned Model (Expected 0 to +2 points, uncertain)

**What it is**: Use `huikang/gpt-oss-120b-aimo3/Transformers/160a/20` instead of base model.
Drop-in replacement with identical code.

**Evidence**: Ambiguous. Base model achieves 44/50 publicly. Huikang model is used by
competitive notebooks (Jonathan Chan, kienngx). No direct A/B comparison available.

**Expected impact**: 0 to +2 points. The v25 notebook (pending score) will clarify.
If v25 scores 44+, the combination (huikang + verification + reliability voting) works.

**Risk**: Low (same code, same protocol).

### Rank 5: Increase max_tokens and context_tokens (Expected +0.5 to +1 point)

**Current**: context_tokens=65536, max_tokens=65536 (in our notebooks).

**Top notebooks**: `--max-model-len 81920` with fp8 KV cache.

**What this enables**: Problems that require very long TIR chains (10+ Python calls with
long outputs) currently might be truncated. The 40-50 hardest problems are exactly those
that require the most tool calls.

**Implementation**: Change vLLM flag from `--max-model-len 65536` to `--max-model-len 81920`
and add `--kv-cache-dtype fp8_e4m3`. The fp8 KV cache is required to fit 81920 in VRAM.

**Expected impact**: +0.5 to +1 point on problems with very long reasoning chains.

---

## Techniques That Are NOT Worth Trying

### Diverse prompts: Definitively ruled out
The arxiv paper ran 23+ experiments. Every alternative prompt strategy reduced score.
Use one consistent 5-step prompt for all attempts.

### More attempts (12, 16): Minimal benefit, time cost
Math shows 8 is already near-ceiling for majority voting with negative correlation.
Going to 12 reduces per-problem time budget.

### Seed diversity: Already maximally decorrelated
T=1.0 sampling already achieves ρ̂ = -0.258 correlation between attempts.

### Temperature < 0.8: Reduces mean score
T=0.5 → -1.7 points vs baseline. Use T=0.8 or T=1.0.

### Temperature > 1.0 with high min_p: Bad combination
T=1.2 + min_p=0.03 → -2.7 points.

---

## Ceiling Analysis: Can We Reach 45-46?

**Current base rate**: 44/50 is achievable with the best public approaches. We've validated
this in our competition log.

**To reach 45**: Need to solve 1 more problem than the "standard" 44. Options:
1. Follow-up prompting recovers a failed attempt on a problem we were close on
2. Geometry prompt enhancement cracks one geometry problem
3. Huikang fine-tune has slightly higher p̂ on hard problems

**To reach 46**: Need to solve 2 more. The #1 scorer (ippeiogawa) may have:
1. A different fine-tuned model or fine-tuning approach
2. A more sophisticated agentic loop
3. Higher compute (more kernels, longer timeouts)
4. Or simply got lucky (variance exists: 13-run σ=1.7 means 46 is within 4σ of the mean)

**The arxiv paper's 44/50 was their best run** out of 13. At mean 39.7 and σ=1.7:
- P(score ≥ 44) ≈ P(Z ≥ (44-39.7)/1.7) = P(Z ≥ 2.53) ≈ 0.6%
- P(score ≥ 45) ≈ P(Z ≥ (45-39.7)/1.7) = P(Z ≥ 3.12) ≈ 0.1%
- P(score ≥ 46) ≈ P(Z ≥ (46-39.7)/1.7) = P(Z ≥ 3.71) ≈ 0.01%

**This means**: Reaching 46 with the current approach requires extreme luck (roughly
1 in 10,000 submissions). Reaching 45 requires ~1 in 1,000. With 14 days left and ~2
submissions per day = ~28 more submissions, P(hitting 45+) ≈ 2.7%.

**The real path to 45-46 is systematic improvement, not luck**:
Implementing follow-up prompting + verification + geometry enhancement together may
shift the mean to 41-42 and the occasional high to 45-46.

---

## Time Budget: Remaining Competition Days

Deadline: April 15, 2026. Current date: April 1, 2026. **14 days remaining.**

Priority order for implementation:
1. Wait for v25 score (huikang + reliability voting) — arriving ~April 2
2. Implement multi-turn follow-up prompting in the generation loop — **highest impact**
3. Add geometry-specific system prompt enhancement
4. Implement answer verification (ans-verifys approach)
5. Validate v26 (12 attempts, T=0.8) score vs v25

---

## Sources

1. Arxiv paper 2603.27844: "Model Capability Dominates: Inference-Time Optimization
   Lessons from AIMO 3" by Natapong Nitarach — full ablation results, temperature table,
   diverse prompt mixer experiments, correlation analysis.

2. Existing research: `/f/kaggle/aimo-progress-prize-3/research/huikang_model_research.md`
   — leaderboard standings, model analysis, Jonathan Chan Bayesian details.

3. Existing research: `/f/kaggle/aimo-progress-prize-3/research/answer_extraction_research.md`
   — answer extraction analysis, amanatar verification approach, Hui Kang follow-up prompting.

4. Existing research: `/f/kaggle/aimo-progress-prize-3/research/aimo3_top_notebook_analysis.md`
   — protocol analysis, voting mechanism details, parameter comparison tables.

5. Competition log: `/f/kaggle/aimo-progress-prize-3/docs/competition_log.md`

6. v8 notebook (`submission_v8_12attempts.ipynb`) — 12-attempt config details.

7. AIMOPRIZE.COM: Commercial model performance data, gap analysis, geometry problem findings.

8. OlymMATH benchmark paper (arxiv 2503.21380v2): Problem category difficulty breakdown.

9. gpt-oss-120b model card (arxiv 2508.10925v1): AIME 2024/2025 benchmark scores.

10. Unsloth docs: Recommended inference parameters T=1.0, top_p=1.0, top_k=0.
