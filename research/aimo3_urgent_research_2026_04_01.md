# AIMO3 Urgent Research: How to Jump from 38 to 45+
## Date: 2026-04-01 | Deadline: 2026-04-15 | 14 days, ~13 submissions remaining

---

## Situation

- Our best LB: 38/50 (v18). We have v25 pending (expected 42-45).
- Top LB: 46/50 (ippeiogawa, fully private).
- Best public notebook: 44/50 (nihilisticneuralnet/44-50-let-me-over-cook).
- We need p≈0.85 per-attempt to consistently hit 46. Current base model gives p≈0.69.

---

## MOST IMPORTANT NEW FINDINGS (discovered today)

### 1. CRITICAL: Pawan Mali's 50-Experiment Log Shows What DOES and DOES NOT Work

Source: `pawanmali/chasing-47-50-aimo3-journey-of-50-experiments` (30 votes)
Best score achieved by this team: 42/50.

**What works (confirmed with multiple experiments):**

| Technique | Score Change | Notes |
|-----------|-------------|-------|
| Temperature = 1.0 | +2-3 pts | V40 (42/50): T=1.0. V41 (34/50): T=0.5 |
| 5-component weighted entropy | +5-6 pts | vs simple mean. V136 simple mean → 35/50 |
| Base problem timeout = 270s | Stable | NOT 300. 270 is more reliable |
| 65536 context (NOT 131K) | +6 pts | 131K halves concurrency (7x → 3.35x) |
| Simple prompts (3 lines) | +3-5 pts | vs complex structured prompts |

**What does NOT work:**

| Technique | Score Change | Notes |
|-----------|-------------|-------|
| huikang fine-tuned model (alone) | -6 pts | V128: 35/50 vs V125: 41/50 |
| Answer verification (yes/no) | 0 pts | V133: same as V132 |
| Self-refinement | -4 pts | V134: model "corrects" correct answers |
| Combined techniques (6 changes) | -6 pts | V131 |
| top_p=0.9 | -8 pts | V126: 33/50 |
| 131K context | -6 pts | V135 |
| Simple mean entropy | -6 pts | V136 |

**Gap analysis (their estimate, March 30 2026):**
- Rank 1: ippeiogawa 46/50 (+4 above their 42/50)
- Rank 2: nihilisticneuralnet 44/50 (+2 above them)
- The 44/50 solution uses temperature=0.5 + SIMPLE entropy (opposite from what works for them)
- **Key insight: "nihilisticneuralnet uses temp=0.5 + simple entropy, we use temp=1.0 + weighted entropy. These are OPPOSITE approaches but both work."**

**Untested by them (highest potential):**

| Approach | Risk | Potential |
|----------|------|-----------|
| MCTS (tree search over solution paths) | High | +2-4 pts |
| Problem classification + type-specific strategies | Medium | +1-2 pts |
| Full huikang pipeline (not just model swap) | Medium | +1-2 pts |
| temp=0.5 + weighted entropy hybrid | Low | +1-2 pts |

---

### 2. CRITICAL: Actual Code from 44/50 Notebooks - Exact Differences

I pulled and analyzed the two 44/50 notebooks by nihilisticneuralnet.

**44/50 "overcook" (latest, Jan 15 2026):**
```python
class CFG:
    temperature = 0.5        # KEY: 0.5, not 1.0
    min_p = 0.02
    # NO top_p
    context_tokens = 65536
    attempts = 8
    early_stop = 4
    workers = 16
    batch_size = 256
    base_problem_timeout = 300   # 300s
    high_problem_timeout = 900
    model_path = '/kaggle/input/gpt-oss-120b/transformers/default/1'  # BASE model
    kv_cache_dtype = 'fp8_e4m3'
    stream_interval = 200
    top_logprobs = 5
    gpu_memory_utilization = 0.96
    seed = 42
    turns = 128
    # entropy = SIMPLE MEAN (not weighted)
```

System prompt: **detailed 5-step structured prompt** (UNDERSTAND→EXPLORE→PLAN→EXECUTE→VERIFY) with detailed tool guidance, preference prompt with specific library instructions (sympy, numpy, math).

**43/50 weighted entropy notebook (Feb 5 2026):**
- Almost identical CFG as above, except temperature=1.0
- Uses the **5-component weighted entropy** formula (with position weighting, variance penalty, high-entropy ratio, streak bonus)
- This scored 43, NOT 44 — so complex entropy with T=1.0 → 43, simple entropy with T=0.5 → 44

**Critical observation**: The 44/50 notebook uses temperature=0.5 with SIMPLE mean entropy. The 43/50 uses temperature=1.0 with COMPLEX entropy. Our research log note "T=0.5 → 38 per arxiv paper" likely reflects the arxiv paper's configuration (which used simple mean entropy at T=0.5). With the detailed structured prompt and different seed setup, T=0.5 can reach 44.

---

### 3. IMPORTANT: Nemotron-3 Super 120B (MoE) as Alternative Architecture

Source: `khoinguyennguyen/nvidia-nemotron-3-super-120b-accelerated-tir` (4 votes)

This notebook uses **NVIDIA Nemotron-3 Super 120B (MoE)** — NOT GPT-OSS-120B. Key facts:
- Model: `zaynyu/nvidia-nemotron-3-super-120b-a12b-nvfp4` (80.4 GB with NVFP4 quantization)
- 12B active parameters per token (MoE) → much faster inference than 120B dense
- Fits on single H100 80GB with NVFP4
- Score: not reported by this author (utility notebook), but forked from a 9/10 AIME notebook
- Uses standard HuggingFace chat template (not Harmony protocol) → different code path
- Selective thinking: `enable_thinking = (turn_id % OVERTHINKING_PERIOD) == OVERTHINKING_MODULO`
- Temperature 0.6, attempts=6, context=3.5×65536=229376 tokens

**Verdict**: Different architecture, potentially better throughput. NOT tested for AIMO3 scoring. GPT-OSS-120B with Harmony protocol still dominates public leaderboard.

---

### 4. IMPORTANT: TurboQuant (2-bit KV cache) Testing

Source: `yeoyunsianggeremie/aimo3-testing-turboquant` (30 votes) — testing only notebook

TurboQuant is a new vLLM KV cache quantization:
```python
kv_cache_dtype="turboquant"
os.environ['TQ_BITS'] = '2'  # 2-bit KV cache
```

With Qwen3.5-35B-A3B model. Geremie Yeo was testing if TurboQuant (2-bit) allows longer context or more concurrent sequences on H100. **This is a throughput/memory experiment, not a scored submission.** It may allow more parallel attempts. Not yet proven to improve accuracy.

---

### 5. CONFIRMED: Bayesian Notebook Uses huikang v14, Not Base Model

The Jonathan Chan Bayesian notebook uses:
```python
model_path = '/kaggle/input/models/huikang/gpt-oss-120b-aimo3/transformers/160a/14'
```

This is **version 14**, not the latest version 20. The notebook also adds:
- `ANSWER_ONLY_PROMPT`: a second prompt for a "silent reasoning" verification pass
- `scipy`, `gmpy2`, `z3`, `fractions` added to sandbox (beyond just math/numpy/sympy/mpmath)
- `posterior_stop_threshold = 0.82`: stop if posterior probability exceeds 82%
- `voi_entropy_weight = 0.6, voi_compute_cost = 0.04` for VOI early stopping

The Bayesian solve method is NOT shown in the notebook cells I retrieved (the key `_compute_bayesian_posterior` method is truncated). But from prior research in `huikang_model_research.md`, we have the full implementation.

---

### 6. CONFIRMED: Pawan Mali Data on huikang Fine-Tune

Pawan Mali directly tested huikang fine-tuned model (V128):
- V125 (base model, T=1.0): **41/50**
- V128 (huikang fine-tuned, same everything else): **35/50** → -6 points

His explanation: "The fine-tuned model was trained with huikang's specific pipeline (131K context, KV cache management, specific prompts). Using it with our pipeline breaks the optimization."

This confirms our prior research: **swapping only the model_path without matching the full training pipeline causes regression**.

However: Jonathan Chan uses huikang model with the Harmony protocol — and our competition log says his notebook has 185 votes (high quality). His setup is:
- huikang model + Harmony + T=1.0 + Bayesian voting

So huikang model CAN work with Harmony if used correctly. The issue is likely that Pawan Mali's pipeline wasn't configured for the fine-tuned model's expected prompt format.

---

## Actionable Strategies for Next 14 Days

### Priority 1: GET THE TEMPERATURE RIGHT (Immediate, 0 risk)

Our v25 uses T=1.0. The 44/50 notebook uses T=0.5 with simple entropy.

The correct approach based on ALL evidence:
- **T=0.5 + simple mean entropy + detailed structured prompt = 44/50** (confirmed public)
- **T=1.0 + 5-component weighted entropy = 43/50** (confirmed public)
- **T=1.0 + simple mean entropy = 35/50** (Pawan Mali V136)
- **T=0.5 + 5-component weighted entropy = UNTESTED** (Pawan Mali's suggestion)

**Action**: Our v23 (base model, exact 44/50 config) should score around 42-44. If v25 (our huikang + reliability voting) scores < 42, switch immediately to the exact 44/50 config. If v25 scores 42+, keep the direction.

### Priority 2: HUIKANG MODEL REQUIRES FULL PIPELINE MATCH

If we use huikang model (v20), we must:
1. Use the correct model_path: `/kaggle/input/models/huikang/gpt-oss-120b-aimo3/transformers/160a/20`
2. NOT just swap model_path alone — the training was done with specific prompts
3. The Harmony protocol appears to be compatible (Jonathan Chan uses it successfully)
4. Consider adding extra sandbox packages: `scipy`, `gmpy2`, `z3`
5. Use VOI early stopping or the 3-factor weight (entropy × reliability × tool bonus)

**For now**: The safest submission is the base model with exact 44/50 parameters. The huikang model experiment should be secondary.

### Priority 3: SIMPLE ENTROPY IS BETTER THAN COMPLEX (Counter-intuitive)

Based on confirmed data:
- Simple mean entropy (44/50 overcook): **better**
- 5-component weighted entropy (43/50): **worse by 1 point**

Simple entropy implementation:
```python
def _compute_mean_entropy(self, logprobs_buffer):
    if not logprobs_buffer:
        return float('inf')
    total_entropy = 0.0
    token_count = 0
    for top_logprobs_dict in logprobs_buffer:
        if not isinstance(top_logprobs_dict, dict) or not top_logprobs_dict:
            continue
        token_entropy = sum(-math.exp(lp) * math.log2(math.exp(lp)) 
                           for lp in top_logprobs_dict.values() 
                           if math.exp(lp) > 0)
        total_entropy += token_entropy
        token_count += 1
    return total_entropy / token_count if token_count > 0 else float('inf')
```

### Priority 4: MCTS FOR HARD PROBLEMS (High risk, high reward)

Pawan Mali identifies MCTS as the highest-potential untested technique (+2-4 pts).

Theory: Instead of 8 independent parallel attempts, use a tree search where:
- Early promising paths get more compute allocated
- Failed paths (Python errors, contradictions) are pruned
- Best partial solution is tracked and extended

This requires significant engineering. NOT recommended with 14 days left unless we have capacity. The time budget per problem is the main constraint.

### Priority 5: PROBLEM DIFFICULTY ROUTING (Medium risk)

Allocate time dynamically based on estimated problem difficulty:
- Easy problems (identifiable keywords: "find the value", "compute", definite numeric bounds) → 270s
- Hard problems (combinatorics, geometry, "how many integers") → 900s
- Budget rebalancing: save time on easy to spend on hard

The current implementation already does partial version of this with `base_problem_timeout` and `high_problem_timeout`. The question is whether we can identify "hard" problems from text alone.

Simple heuristic: problems mentioning "geometry", "circle", "triangle", "polygon" → hard (900s). Problems with simple arithmetic setup → easy (270s).

### Priority 6: EXTRA SANDBOX PACKAGES

The Bayesian notebook adds to the Jupyter sandbox:
```python
import scipy, gmpy2, z3, fractions, functools
```

`z3` is a constraint solver that can solve combinatorial problems. `gmpy2` is faster arbitrary-precision arithmetic. These could help on specific problem types.

Implementation: add to sandbox init:
```python
self.execute('import math, numpy, sympy, mpmath, itertools, collections, scipy, fractions\n'
             'try: import gmpy2\nexcept: pass\n'
             'try: import z3\nexcept: pass\n'
             'mpmath.mp.dps = 64\n')
```

---

## The Two Approaches in Competition (Both Scored 44/50)

| Aspect | nihilisticneuralnet 44/50 | nihilisticneuralnet 43/50 |
|--------|--------------------------|--------------------------|
| Temperature | 0.5 | 1.0 |
| Entropy | Simple mean | 5-component weighted |
| System prompt | 5-step structured | 5-step structured |
| Tool prompt | Detailed with examples | Brief |
| Preference prompt | Detailed (sympy/numpy/math guide) | Brief |
| Model | Base GPT-OSS-120B | Base GPT-OSS-120B |
| context_tokens | 65536 | 65536 |
| attempts | 8 | 8 |
| early_stop | 4 | 4 |
| base_problem_timeout | 300 | 270 |

**Key difference**: T=0.5 with simple entropy beats T=1.0 with complex entropy by 1 point.

---

## What ippeiogawa (46/50) Likely Does

Based on all available evidence, the #1 scorer likely:
1. Uses a proprietary fine-tuned model (not the public huikang or base models)
2. OR uses a significantly different architecture (Nemotron MoE, different inference setup)
3. OR has solved 2 specific "hard" problems that the standard approach fails on, using:
   - A different system prompt specifically crafted for those problem types
   - MCTS or search-based approach
   - Model with higher per-attempt accuracy (p≈0.80+)
4. OR got lucky (p=46/50 is within 4σ of mean for current methods — extremely rare but possible)

The most likely answer: proprietary fine-tuned model + either better voting or more attempts.

---

## Model Comparison Summary

| Model | Size | Available | Known LB | Notes |
|-------|------|-----------|----------|-------|
| danielhanchen/gpt-oss-120b | 120B | Public (Kaggle) | 44/50 | Best confirmed public score |
| huikang/gpt-oss-120b-aimo3 (v20) | 120B | Public (Kaggle) | Unknown (regression when used without matching pipeline) | Use with care |
| huikang/huikang-use-only (v13) | 120B | Public but not tested | Unknown | Separate experiment model |
| zaynyu/nvidia-nemotron-3-super-120b-a12b-nvfp4 | 120B MoE (12B active) | Public | Unknown for AIMO3 | Different protocol needed |
| Qwen3-30B-A3B | 30B MoE (3B active) | Public | 40/50 (with GPT-OSS) | Used as secondary by ZaynYu |
| reyvan14/gpt-oss-sft-aimo3 | 120B | Public (v1 only) | Unknown | Jan 2026, may be outdated |

---

## Recommended Submission Order (Next 13 Days)

**Priority ranking based on all research:**

| Day | Action | Expected Score | Confidence |
|-----|--------|---------------|------------|
| Apr 2 | v25 result arrives (huikang + reliability voting) | Expected 40-44 | Medium |
| Apr 3 | Submit: T=0.8, 12 attempts, simple entropy + _verify_answer cascade (ans_verifys config) | 42-45 | High |
| Apr 4 | v26 result (12 attempts, T=0.8) if submitted | 40-44 | Medium |
| Apr 5 | Submit: exact 44/50 config (T=0.5, 8 attempts, simple entropy, detailed structured prompt) | 42-44 | High |
| Apr 6 | Submit: T=0.8, 12 attempts, _verify_answer + z3/gmpy2/scipy sandbox | 43-45 | Medium |
| Apr 7 | Submit: T=0.5, 12 attempts, simple entropy + _verify_answer (hybrid best-of-both) | 43-45 | Medium |
| Apr 8 | Submit: huikang v20 + T=0.8 + Bayesian voting (Jonathan Chan config) | 42-45 | Low |
| Apr 9-10 | Submit: geometry/combinatorics difficulty routing + per-type prompts | 42-44 | Low |
| Apr 11-12 | Submit best current config (vary seed for diversity) | 38-46 | Stochastic |
| Apr 13-14 | Repeat best config (luck factor at top end) | 38-46 | Stochastic |
| Apr 15 | Final submission before deadline | Best achieved | - |

**High confidence actions** (proven by public notebooks):
1. T=0.8 + 12 attempts + _verify_answer cascade (ans_verifys, 545 votes)
2. T=0.5 + simple entropy + detailed structured prompt (44/50 overcook, 2635 votes)

**Medium confidence** (combination of proven elements):
3. T=0.8 + 12 attempts + z3/gmpy2 sandbox (novel combination)
4. T=0.5 + 12 attempts + _verify_answer (reduces wrong answers without hurting right ones)

---

## NEW FINDING: ans_verifys Notebook (Aman Atar, 545 votes — 2nd most popular)

Source: `amanatat/ans-verifys` — pulled and analyzed 2026-04-01.
**545 votes** makes this the #2 most popular AIMO3 notebook.

### Key novelty: `_verify_answer` method

This notebook implements a **dedicated model-based answer verification step** at temperature=0.0:

```python
def _verify_answer(self, problem: str, answer: int) -> bool:
    prompt = f"Problem:\n{problem}\n\nProposed answer: {answer}\n\nCheck the answer carefully.\nReply with only ONE word:\nCORRECT or WRONG"
    prompt_ids = self.encoding.encode(prompt)
    resp = self.client.completions.create(
        model=self.cfg.served_model_name,
        prompt=prompt_ids,
        temperature=0.0,    # greedy decoding for deterministic judgment
        max_tokens=5
    )
    text = resp.choices[0].text.strip().upper()
    return "CORRECT" in text and "WRONG" not in text
```

### Full solve flow (unique 4-step cascade):

1. **HARD ACCEPT**: If >=4 of 12 attempts agree (unanimous by early_stop=4) → return immediately
2. **CANDIDATE FILTER**: Keep only answers with >=2 votes
3. **ENTROPY SORT**: Among candidates, sort by average entropy (ascending = most confident first)
4. **VERIFY**: Call `_verify_answer` for each candidate in order → return first CORRECT
5. **FALLBACK**: If no CORRECT, use entropy-weighted voting (`_select_answer`)

This is the **only notebook** implementing explicit answer verification using the model itself.

### CFG differences vs 44/50 notebook:

| Parameter | ans_verifys (545 votes) | 44/50 overcook (2635 votes) |
|-----------|------------------------|----------------------------|
| temperature | **0.8** | 0.5 |
| attempts | **12** | 8 |
| batch_size | 128 | 256 |
| early_stop | 4 | 4 |
| workers | 16 | 16 |
| context_tokens | 65536 | 65536 |
| entropy | simple mean | simple mean |
| verify_answer | YES (novel) | no |
| sandbox_timeout | 3 | 3 |
| notebook_limit | 17400 | not set |

### Why temperature=0.8 matters:

The arxiv paper on AIMO3 (2603.27844) found T=0.8 is optimal for this model. Aman Atar appears to have found the same empirically. The `_verify_answer` call uses T=0.0 (greedy) which is safe for binary CORRECT/WRONG judgment.

### Critical note on verify_answer effectiveness:

Pawan Mali (V133) tested answer verification and got 0 improvement. However, his implementation may have been different. Aman Atar's approach:
1. Only verifies candidates with >=2 votes (not all answers)
2. Sorts by entropy first (most confident answer verified first)
3. Uses pure greedy T=0.0 for the verification judge
4. Verification is a lightweight 5-token completions call (very fast)

With 545 votes, the community clearly found this approach valuable. The null result in Pawan Mali's V133 may be because he used a different verification prompt or called it differently.

### CONFIRMED: ans_verifys scores 44/50

Cross-referenced leaderboard: "Aman" team (teamId 14953109) has score 44/50, last submission 2026-03-31.
This is Aman Atar (`amanatar`), author of `ans_verifys`. So the approach works in practice.

Note on `encoding.encode()` in `_verify_answer`: This passes raw tokenized text to the `completions` endpoint (not `chat/completions`). Since GPT-OSS-120B is a completion model, this produces a valid completion from raw text. The model responds correctly because the verification prompt is readable plain text and the model is capable of raw text completion. This is a different code path than the main Harmony-framed conversation but still valid.

### Action item:

Consider combining `_verify_answer` with the 44/50 config:
- Use T=0.8 (or T=0.5) with 8-12 attempts
- After unanimous/voting, verify top candidates with T=0.0 judge
- Only takes a few tokens per call (max_tokens=5)
- Proven to work: achieves 44/50 from a different starting config than nihilisticneuralnet

---

## What NOT to Do

1. Do NOT use 131K context — halves concurrency
2. Do NOT combine multiple experimental changes at once
3. Do NOT use answer verification loops (0 improvement per Pawan Mali, but see ans_verifys above)
4. Do NOT use self-refinement (hurts, model overcorrects)
5. Do NOT use top_p > 0.85 (top_p=0.9 → -8 points)
6. Do NOT use temperature < 0.5 (too conservative)
7. Do NOT use different system prompts per attempt
8. Do NOT increase to 12 attempts blindly — only if combined with _verify_answer style cascade

---

## ZaynYu Nemotron Local Testing Notebook (9/10 AIME)

Source: `zaynyu/9-10-nvidia-nemotron3-super-120b-optimal-tir` (March 17, 2026, 31 votes)

This is ZaynYu's **local AIME testing notebook** — NOT an AIMO3 competition submission.
- `attempts=1, workers=2, early_stop=1` — single-attempt local mode
- **Saves all token IDs and decoded text** to `solutions/{question_id}/` for offline analysis
- Has `</think>` detection (Nemotron's thinking tags)
- Uses HuggingFace tokenizer + chat template (NOT Harmony protocol)
- Model: `zaynyu/nvidia-nemotron-3-super-120b-a12b-nvfp4/transformers/v1/1`
- "9/10" refers to AIME accuracy in local testing, NOT AIMO3

**Key technique**: Saves complete conversation traces (token IDs + decoded text) to disk. This allows:
1. Post-hoc analysis of which problems fail and why
2. Studying the model's reasoning chains
3. Fine-tuning data collection

**Verdict**: This is ZaynYu's offline analysis infrastructure. Not directly usable as a competition submission. Their AIMO3 best public score is 40/50 from January 2026.

---

## Sources

1. `nihilisticneuralnet/44-50-let-me-over-cook` — pulled and analyzed (44/50 public, 2635 votes)
2. `nihilisticneuralnet/43-50-aimo-3-gpt-oss-120b-weighted-entropy` — pulled and analyzed (43/50, 491 votes)
3. `jonathanchan/aimo3-gpt-oss-120b-with-bayesian` — pulled and analyzed (185 votes, huikang model)
4. `pawanmali/chasing-47-50-aimo3-journey-of-50-experiments` — pulled and analyzed (30 votes)
5. `khoinguyennguyen/nvidia-nemotron-3-super-120b-accelerated-tir` — pulled and analyzed (4 votes)
6. `yeoyunsianggeremie/aimo3-testing-turboquant` — pulled and analyzed (30 votes)
7. `amanatat/ans-verifys` — pulled and analyzed (545 votes, _verify_answer method, T=0.8, 12 attempts, CONFIRMED 44/50)
8. `zaynyu/9-10-nvidia-nemotron3-super-120b-optimal-tir` — pulled and analyzed (local testing rig, 9/10 AIME)
9. `kaggle models list -s "gpt-oss"` — full model registry scan
10. `kaggle models instances versions list huikang/gpt-oss-120b-aimo3/transformers/160a` — v1-v20
11. Leaderboard cross-reference: "Aman" = Aman Atar = 44/50, confirms ans_verifys works
12. Prior research: `/f/kaggle/aimo-progress-prize-3/research/deep_research_42_to_46.md`
13. Prior research: `/f/kaggle/aimo-progress-prize-3/research/huikang_model_research.md`
