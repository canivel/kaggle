# AIMO3 Score Differentiators: 43 vs 44 vs 45/46

**Research date:** 2026-04-01  
**Source:** Direct notebook pulls via `kaggle kernels pull`, leaderboard via `kaggle competitions leaderboard`

---

## Leaderboard (Final, 2026-04-01)

| Rank | Team | Score | Public Submission Date |
|------|------|-------|------------------------|
| 1 | ippeiogawa | **46** | 2026-03-31 |
| 2 | just public 44, all is luck | **45** | 2026-03-01 |
| 3 | Batman's Butler | **45** | 2026-03-31 |
| 4 | Riku Suzuki | **45** | 2026-04-01 |
| 5 | Seungjun Lee | **45** | 2026-04-01 |
| 6-20 | Various | **44** | Various |
| (public ref) | nihilisticneuralnet (parthenos) | **44** | 2026-01-15 |
| (public ref) | nihilisticneuralnet | **43** | 2026-02-05 |
| (public ref) | datasciencegrad | **42** | 2026-02-16 |

**Critical note:** The 46/50 scorer (ippeiogawa) has NO public notebook. The 45/50 scorers also have no public notebooks. The highest publicly available notebook is **44/50** (nihilisticneuralnet "LET ME (over)COOK!!!", 2630 votes).

---

## Notebooks Analyzed

| Notebook | LB Score | Author | Votes |
|----------|----------|--------|-------|
| `nihilisticneuralnet/44-50-let-me-over-cook` | 44 | parthenos | 2630 |
| `nihilisticneuralnet/43-50-aimo-3-gpt-oss-120b-weighted-entropy` | 43 | parthenos | 490 |
| `datasciencegrad/aimo-3-42-50-stable-lb-possible-43-luck` | 42 | Sagar Nagpure | 209 |
| `bhargavaabhi/aimo-3-winner` | ~41 | BhargavaAbhi | 176 |
| `jonathanchan/aimo3-gpt-oss-120b-with-bayesian` | ~40 | Jonathan Chan | 185 |
| `amanatar/ans-verifys` | ~41 | Aman Atar | 542 |
| `shelterw/15-15-aime-2026-i-120b-in-20mins` | 15/15 AIME | ShelterW | 340 |
| `kaanyorgun/44-aimo3` | claimed 44+ | Kaan Yorgun | 8 |

---

## Master Comparison Table

| Parameter | andreasbis (base) | datasciencegrad (42) | nihilist (43, weighted-entropy) | nihilist (44, overcook) | kaanyorgun (44) | amanatar (42-ish) | Jonathan Chan (Bayesian) | ShelterW (AIME) |
|---|---|---|---|---|---|---|---|---|
| **LB Score** | ~40 | 42 | 43 | 44 | 44 | ~41-42 | ~40 | 15/15 AIME |
| **temperature** | 1.0 | 1.0 | 1.0 | **0.5** | **0.5** | **0.8** | 1.0 | 1.0 |
| **min_p** | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |
| **top_p** | - | - | - | - | - | - | 0.95 | - |
| **attempts** | 8 | 8 | 8 | 8 | 8 | **12** | 8 | 8 |
| **early_stop** | 4 | 4 | 4 | 4 | 4 | 4 | N/A (Bayesian) | 4 |
| **workers** | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| **batch_size** | 256 | 256 | 256 | 256 | 256 | **128** | 256 | 256 |
| **context_tokens** | 65536 | 65536 | 65536 | 65536 | 65536 | 65536 | 65536 | 65536 |
| **high_problem_timeout** | 900 | 900 | 900 | 900 | 900 | 900 | 900 | 900 |
| **base_problem_timeout** | 300 | **300** | **270** | 300 | 300 | 300 | 300 | 300 |
| **voting** | plain entropy | plain entropy | **weighted entropy** | plain entropy | plain entropy | plain entropy | **Bayesian posterior** | plain entropy |
| **entropy formula** | simple mean | simple mean | **position-weighted + variance + streak** | simple mean | simple mean | simple mean | entropy_weight * reliability * tool_bonus | simple mean |
| **system prompt** | minimal (~2 lines) | minimal | minimal | **structured 5-step IMO** | **structured 5-step IMO** | **structured INTERNAL PROTOCOL** | force Python mandatory | minimal |
| **preference_prompt** | minimal | minimal | minimal | **detailed sympy/numpy guide** | **detailed sympy/numpy guide** | **detailed with arrows** | minimal | minimal |
| **tool_prompt** | minimal | minimal | minimal | **detailed 5-use-cases** | **detailed 5-use-cases** | **selective use** | minimal | minimal |
| **model** | gpt-oss-120b (default) | gpt-oss-120b (default) | gpt-oss-120b (default) | **danielhanchen variant** | gpt-oss-120b (default) | gpt-oss-120b (default) | **huikang fine-tuned 160a** | **danielhanchen variant** |
| **seed** | 42 | 42 | 42 | **blank/random** | 42 | 42 | 42 | **blank/random** |

---

## Critical Finding 1: Temperature 1.0 vs 0.5

The most clearly documented jump from 43 to 44 is **lowering temperature from 1.0 to 0.5**.

- All 43/50 and lower notebooks: `temperature = 1.0`
- Both 44/50 public notebooks (nihilist 44, kaanyorgun 44): `temperature = 0.5`
- The 42/50 (datasciencegrad) and 43/50 (nihilist): `temperature = 1.0`

This is the single most consistent technical difference between scoring tiers.

**Hypothesis:** At temperature=1.0, the model explores more but makes more arithmetic errors. At 0.5, the model is more deterministic and consistent, especially for the final calculation steps.

**Counter-note:** amanatar uses 0.8 with 12 attempts, which is a different trade-off (more attempts + medium temperature).

---

## Critical Finding 2: System Prompt Engineering

**Baseline (andreasbis, datasciencegrad, nihilist-43):**
```
'You are a world-class International Mathematical Olympiad (IMO) competitor. '
'The final answer must be a non-negative integer between 0 and 99999. '
'You must place the final integer answer inside \\boxed{}.'
```

**44/50 notebooks (nihilist-44, kaanyorgun-44, ShelterW):**
```
'You are an elite mathematical problem solver with expertise at the International 
Mathematical Olympiad (IMO) level. Your goal is to find the correct answer through 
rigorous mathematical reasoning.\n\n'

'# Problem-Solving Approach:\n'
'1. UNDERSTAND: Carefully read and rephrase the problem...\n'
'2. EXPLORE: Consider multiple solution strategies...\n'
'3. PLAN: Select the most promising approach...\n'
'4. EXECUTE: Work through your solution methodically...\n'
'5. VERIFY: Check your answer by substituting back...\n\n'

'# Mathematical Reasoning Principles:\n'
'- Break complex problems into smaller, manageable sub-problems\n'
'- Look for patterns, symmetries, and special cases...\n'
...
'# Verification Requirements:\n'
...
'# Output Format:\n'
'The final answer must be a non-negative integer between 0 and 99999.\n'
'Place your final numerical answer inside \\boxed{}, e.g., \\boxed{42}\n\n'

'Think step-by-step and show your complete reasoning process...'
```

The 44/50 prompt is approximately **10x longer** than the 43/50 prompt and adds:
- Explicit 5-step problem-solving methodology (UNDERSTAND/EXPLORE/PLAN/EXECUTE/VERIFY)
- Mathematical reasoning principles section
- Verification requirements section
- Concrete output format example (`\\boxed{42}`)

**amanatar's variant** adds an "INTERNAL SOLVING PROTOCOL (DO NOT REVEAL)" framing plus verification constraints ("Accept an answer ONLY if...").

**Jonathan Chan (Bayesian)** instead forces Python use: "You MUST use the python tool for calculations, algebra, or enumeration. Before giving the final answer: 1. Write Python code... 2. Execute the code... 3. Only then produce the boxed answer."

---

## Critical Finding 3: Weighted Entropy (nihilist-43 innovation)

The nihilist-43 notebook introduced a multi-component entropy metric (which is its main claim to fame), but it actually **did NOT improve over the plain entropy** in nihilist-44. The 43->44 jump came from the system prompt + temperature, not the complex entropy formula.

**nihilist-43 weighted entropy formula:**
```python
final_entropy = (
    0.3 * mean_ent +                    # Base uncertainty level
    0.4 * position_weighted_ent +       # Exponential decay toward end (MOST IMPORTANT)
    0.2 * std_dev +                     # Consistency penalty
    0.3 * high_ent_ratio * 3.0 +        # Sustained high-entropy penalty
    streak_bonus                         # Bonus for low-entropy streaks
)
```

**nihilist-44, datasciencegrad-42, kaanyorgun-44, amanatar, ShelterW - plain entropy:**
```python
return total_entropy / token_count  # simple mean of per-token Shannon entropy
```

The nihilist-44 notebook **reverted to plain entropy** and achieved a higher score (44 vs 43). This suggests the complex entropy formula was not helpful and may have hurt.

---

## Critical Finding 4: Bayesian Stopping (Jonathan Chan)

The Bayesian notebook replaces the simple "4-vote early stop" with a Value-of-Information decision:

```python
# After each completed attempt:
posterior = self._compute_bayesian_posterior(detailed_results)
# posterior[answer] = entropy_weight * reliability * tool_bonus (normalized)

if len(detailed_results) >= min_attempts_before_stop:  # 3
    max_prob = max(posterior.values())
    entropy = self._posterior_entropy(posterior)
    
    submit_utility = max_prob
    expected_improvement = entropy * voi_entropy_weight  # 0.6
    continue_utility = max_prob + expected_improvement - voi_compute_cost  # 0.04
    
    if submit_utility >= continue_utility:  # stop when confident enough
        stop_event.set()
```

The reliability weight includes: `reliability = 1/(1 + error_penalty * python_errors)` and `tool_bonus = 1.2` if Python used with no errors, `0.8` if errors.

**Note:** This uses `model_path = '/kaggle/input/models/huikang/gpt-oss-120b-aimo3/transformers/160a/14'` - a fine-tuned variant from Tong Hui Kang (a competition organizer-adjacent figure). This is a different model entirely.

---

## Critical Finding 5: Model Variants

Three different model paths appear across notebooks:

1. **Default** (`/kaggle/input/gpt-oss-120b/transformers/default/1`): Used by most notebooks including all scoring tiers up to 43.
2. **danielhanchen variant** (`/kaggle/input/models/danielhanchen/gpt-oss-120b/transformers/default/1`): Used by nihilist-44 and ShelterW. This is Daniel Han's (Unsloth) optimized variant.
3. **huikang fine-tuned** (`/kaggle/input/models/huikang/gpt-oss-120b-aimo3/transformers/160a/14`): Used by Jonathan Chan Bayesian notebook. This is a fine-tuned AIMO3-specific checkpoint.

The danielhanchen path is used by the top public 44/50 notebook, suggesting the Unsloth variant may provide marginal improvements.

---

## Critical Finding 6: Attempts Count

- All notebooks: 8 attempts except `amanatar`: **12 attempts** with `batch_size=128`

Reducing batch_size from 256 to 128 and increasing attempts from 8 to 12 is a different resource trade-off: fewer parallel sequences but more total attempts. The entropy voting over 12 attempts vs 8 should improve answer selection.

---

## What the 45/46 Scorers Likely Did

Since the 45/46 scorers (ippeiogawa, Batman's Butler, Riku Suzuki, Seungjun Lee, "just public 44") have no public notebooks, we can infer from the progression:

1. **43->44 clearly requires:** Better system prompt (5-step structure) + temperature=0.5
2. **44->45 likely requires:** Some combination of:
   - More attempts (12+) with appropriate batch_size reduction
   - Further prompt engineering (verification forcing, code-mandatory)
   - Fine-tuned model (huikang 160a or similar)
   - Better seed diversity (the ShelterW notebook uses `seed = ` blank/None, implying random seeds per run)
   - Possibly larger context window (the `danielhanchen` variant may support this)

3. **45->46 (ippeiogawa):** This is the most elusive. Ippeiogawa's profile shows prior work on AIMO1/AIMO2 with tweaked solutions. The 46 was submitted 2026-03-31 (last day). Likely involves:
   - Luck/variance (50 problems, at 46/50 only 4 wrong - could be a good sample)
   - Or a genuinely different technique (different model fine-tuning, different stopping, longer thinking)

---

## Code Diff: 43 vs 44 (The Key Changes)

### Change 1: Temperature
```python
# 43/50 (and baseline):
temperature = 1.0

# 44/50:
temperature = 0.5
```

### Change 2: System Prompt (full 5-step vs 2-line)
The system prompt grew from 3 lines to ~30 lines with structured reasoning protocol.

### Change 3: Preference Prompt
```python
# 43/50 baseline:
preference_prompt = 'You have access to `math`, `numpy` and `sympy` to solve the problem.'

# 44/50:
preference_prompt = (
    'You have access to `math`, `numpy`, and `sympy` for:\n\n'
    '# Symbolic Computation (sympy):\n'
    '- Algebraic manipulation and simplification\n'
    '- Solving equations and systems of equations\n'
    '- Symbolic differentiation and integration\n'
    '- Number theory functions (primes, divisors, modular arithmetic)\n'
    '- Polynomial operations and factorization\n'
    '...(full guide for numpy, math, and best practices)...'
)
```

### Change 4: Tool Prompt
```python
# 43/50 baseline:
tool_prompt = ('Use this tool to execute Python code. The environment is a stateful Jupyter notebook. '
              'You must use print() to output results.')

# 44/50:
tool_prompt = (
    'Use this tool to execute Python code for:\n'
    '- Complex calculations that would be error-prone by hand\n'
    '- Numerical verification of analytical results\n'
    '- Generating examples or testing conjectures\n'
    '- Visualizing problem structure when helpful\n'
    '- Brute-force verification for small cases\n\n'
    'The environment is a stateful Jupyter notebook. Code persists between executions.\n'
    'Always use print() to display results. Write clear, well-commented code.\n\n'
    'Remember: Code should support your mathematical reasoning, not replace it. '
    'Explain what you\'re computing and why before running code.'
)
```

### Change 5: Model Path (in 44/50 notebooks)
```python
# 43/50: 
model_path = '/kaggle/input/gpt-oss-120b/transformers/default/1'

# 44/50 (nihilist, ShelterW):
model_path = '/kaggle/input/models/danielhanchen/gpt-oss-120b/transformers/default/1'
```

---

## Ranked List of Techniques to Try (by Expected Impact)

### 1. Temperature = 0.5 [HIGHEST IMPACT, proven 43->44]
**Evidence:** Both 44/50 public notebooks use temperature=0.5 vs temperature=1.0 in all 43/50 and below notebooks.
**Action:** Change `temperature = 1.0` to `temperature = 0.5` in CFG.

### 2. Structured 5-step System Prompt [HIGH IMPACT, proven 43->44]
**Evidence:** All 44/50 notebooks use the 5-step UNDERSTAND/EXPLORE/PLAN/EXECUTE/VERIFY prompt.
**Action:** Replace minimal system prompt with the full structured prompt (see Code Diff Change 2).

### 3. Enhanced preference_prompt and tool_prompt [MODERATE IMPACT]
**Evidence:** Both 44/50 notebooks have 10x longer tool and preference prompts.
**Action:** Replace minimal prompts with structured guides (see Code Diff Changes 3 & 4).

### 4. More attempts (12) with lower batch_size (128) [MODERATE IMPACT]
**Evidence:** amanatar uses 12 attempts + batch_size=128. More attempts = better voting.
**Action:** Change `attempts=8, batch_size=256` to `attempts=12, batch_size=128`.
**Risk:** Needs testing - time budget may not allow 12 full attempts on hard problems.

### 5. danielhanchen model variant [UNKNOWN IMPACT]
**Evidence:** Top 44/50 public notebooks use `/kaggle/input/models/danielhanchen/gpt-oss-120b/transformers/default/1`.
**Action:** Switch model_path to danielhanchen's variant if available on your Kaggle instance.

### 6. Mandatory Python verification before final answer [MODERATE, unproven]
**Evidence:** Jonathan Chan's system prompt forces: "You MUST use the python tool for calculations... Before giving the final answer: 1. Write Python code... 2. Execute the code... 3. Only then produce the boxed answer."
**Action:** Try adding mandatory code verification step to system prompt.

### 7. Bayesian VoI stopping (replace early_stop=4) [UNCERTAIN]
**Evidence:** Bayesian approach in Jonathan Chan notebook, but also uses different model (huikang fine-tuned).
**Action:** Worth testing the VoI stopping logic independently.

### 8. Random seeds instead of fixed seed=42 [LOW, unproven]
**Evidence:** ShelterW notebook shows `seed = ` (blank - Python None), implying random seeds each run.
**Action:** Try using `random.randint(0, 99999)` as seed for non-deterministic diversity.

---

## Things That Did NOT Help (Based on Evidence)

- **Complex weighted entropy** (nihilist-43): The position-weighted + variance + streak formula scored LOWER (43) than simple mean entropy (44). Revert to simple mean.
- **Bayesian posterior voting** (Jonathan Chan): Adds complexity without proven LB improvement over the public 44 notebooks. Also uses different model, making it hard to isolate.

---

## The 46 Gap: What's Unknown

The gap from 44 to 46 (2 additional correct answers) remains unaccounted for by any public notebook. Possible explanations:
1. **Fine-tuned model**: huikang's 160a checkpoint or a private fine-tune specifically trained for AIMO3 problem style
2. **Higher attempt count**: e.g., 16-20 attempts per problem with correspondingly smaller batch_size
3. **Problem-type routing**: Different prompts for geometry vs. algebra vs. combinatorics
4. **Luck**: At 50 problems, getting 2 more correct can be partly stochastic (boundary problems where model is 40-60% accurate)
5. **Context length**: Some problems may require longer thinking chains (increasing context_tokens beyond 65536 would require GPU headroom)

---

## Recommended Next Experiments (Ordered)

1. `temperature=0.5` + structured 5-step prompt: Best ROI, proven to push 43->44
2. `temperature=0.5` + structured prompt + `attempts=10, batch_size=160`: Push toward 45
3. Mandatory Python verification in system prompt: May push boundary problems to correct
4. Try huikang fine-tuned model if accessible: Could be the hidden factor for 45->46
5. Problem-type detection: Detect geometry (use different coordinate tools), number theory (modular arithmetic hints), combinatorics (enumeration hints)

---

*Sources: Direct notebook analysis via kaggle kernels pull. Leaderboard via kaggle competitions leaderboard CLI.*
