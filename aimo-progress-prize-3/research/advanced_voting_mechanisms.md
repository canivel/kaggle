# Advanced Voting Mechanisms for AIMO3: Research Report

**Research date**: 2026-04-01
**Goal**: Find voting methods better than plain 1/entropy for AIMO3 answer selection.
**Current state**: `voting.py` uses majority vote + quality weights (code executed, boxed).
**Key question**: Is it worth changing from 1/entropy weighted voting?

---

## Executive Summary

**Current state (v14 notebook):** 1/entropy weighted voting is already implemented in
`_select_answer` in `submission_v14_verify.ipynb`. The method:
- Computes per-token entropy from top-k vLLM logprobs (already requesting `logprobs=top_logprobs`)
- Weights each attempt by `1 / max(entropy, 1e-9)`
- Selects the answer with highest total weight

The standalone `voting.py` in `src/aimo3/` is a simpler module NOT used by the submission.

**Verdict: 1/entropy is already running. The question is whether anything BETTER exists.**

The most important finding: a peer-reviewed AIMO3-specific paper (arxiv 2603.27844) directly
tested entropy-weighted voting (`w = 1 + 1/(entropy + 0.1)`) on the actual 50 AIMO3 problems
with GPT-OSS-120B and found it produced "marginal gains at best" -- within the natural variance
of ±1.7 problems. Model capability dominated everything by an order of magnitude.
No inference-time voting trick reliably moved the needle by more than ±2 problems.

That said, there are three mechanisms with evidence of improvement that have NOT been tried.
They are listed by expected impact below.

---

## 1. The Papers and What They Actually Say

### 1.1 CISC: Confidence Improves Self-Consistency (arxiv 2502.06233, ACL 2025)

**Method:** Weighted majority vote with softmax-normalized confidence scores.

**Formula:**
```
a* = argmax_a  sum_i  1[answer_i == a] * c_tilde_i

c_tilde_i = exp(c_i / T) / sum_j exp(c_j / T)
```

Where `c_i` is the raw confidence for response `i` and `T` is a temperature hyperparameter.

**Three confidence extraction options tested:**
1. **Response Probability** -- length-normalized logprobs of the ENTIRE reasoning path
2. **Verbal Confidence** -- model rates itself 0-100; score = that number
3. **P(True)** -- append a verification prompt, extract P(token="1")

**How P(True) works:**
- After generating the response, append: "Is this answer correct? Reply 1 for yes, 0 for no."
- Use vLLM to generate ONE token and read its logprob for token "1"
- This is the confidence score for that response
- Requires a second LLM call per sample (but only 1 output token = cheap with prefix caching)

**Results:**
- Reduces required samples by 40% on average vs plain self-consistency
- P(True) achieved 34% cost reduction on MATH dataset specifically
- Works across 9 models and 4 datasets
- Math datasets tested: GSM8K (grade-school), MATH (competition difficulty up to AMC/AIME)
- NOT tested directly on AIMO3-level IMO problems

**vLLM compatibility:** Yes. P(True) requires `logprobs=1` on the verification call to read
the probability of the "1" token.

**Verdict:** P(True) is implementable but adds latency (one extra vLLM call per sample).
For AIMO3's 30-minute budget, this is a real cost. The 40% sample reduction benefit is
irrelevant if we're already computing N=8 samples regardless.

---

### 1.2 Inverse-Entropy Voting (arxiv 2511.02309, "The Sequential Edge")

**Method:** Weight each response by 1/entropy of its token distribution.

**Exact formula:**
```python
# Per-token entropy averaged across the chain
H_i = -(1 / len(tokens_i)) * sum(
    sum(p_t_j * log2(p_t_j) for j in vocab)
    for t in range(len(tokens_i))
)

# Weight
w_i = 1 / max(H_i, 1e-10)
```

**Then apply weighted vote:**
```python
# For each candidate answer a:
score[a] = sum(w_i for i if answer_i == a)
# Select: argmax(score)
```

**Results on parallel sampling (our use case):**
- Outperformed majority voting across all tested models
- Improvements were **modest: +0.5 to +3.4 percentage points** on parallel samples
- Larger gains (+46.7pp) were only for SEQUENTIAL chains building on each other
- Sequential means each chain reads all previous chains -- we cannot do this in parallel

**vLLM compatibility:** Yes. Requires `logprobs=N` where N is large enough to compute entropy.
Problem: vLLM only returns top-k logprobs, not the full distribution. True entropy requires the
full vocab distribution. Workaround: use only the top-k distribution to estimate entropy
(this is the approach DeepConf and most papers actually use in practice).

**Current state in voting.py (standalone module):** NOT implemented. But the submission
notebook (`submission_v14_verify.ipynb`) DOES implement this in `_select_answer`.

**Critical caveat from AIMO3 paper (2603.27844):**
The AIMO3-specific test used formula `w = 1 + 1/(entropy + 0.1)` and found it "showed promise
but lacked significant improvement over basic majority voting." The gains were within noise
(±2 problems on 50-problem set, baseline stdev = 1.7).

**Verdict:** Implement as the PRIMARY voting method but don't expect miracles. It's the
right theoretical move, but AIMO3 evidence says impact is small.

---

### 1.3 CER: Confidence Enhanced Reasoning (arxiv 2502.14634, ACL 2025)

**Method:** Weight paths by the joint probability of INTERMEDIATE numerical answers, not the
full response logprob.

**Formulas:**
```
# Step-wise confidence: multiply token probs for the numerical answer at step j
c_aj = product(p(token_k) for k in answer_tokens_at_step_j)

# Path-wise confidence: linearly weight later steps more
C_path = sum(j * c_aj for j=1..n) / sum(j for j=1..n)

# Final vote: sum path confidences for each final answer
score[a] = sum(C_path(y) for y in paths if y.final_answer == a)
```

**Key insight:** CER focuses on numbers/entities WITHIN the reasoning trace, not the
full token sequence. For math, "numerical tokens" are the intermediate computed values.

**Results:**
- Up to +7.4% improvement over plain self-consistency on MATH dataset (OLMo-2-7B)
- Stronger gains for weaker models; smaller for strong models
- Uses K=10 paths (same as our N=8)
- Training-free, inference-time only

**vLLM compatibility:** Yes. Requires `logprobs=1` to get probability of each generated token.
Implementation requires parsing the reasoning trace to find intermediate numerical answers, then
multiplying their token probabilities.

**Verdict:** This is the most NOVEL approach not yet tried. The key advantage is that it
focuses on numerical intermediate answers -- directly relevant to math competition reasoning.
The implementation complexity is moderate: need to identify "critical tokens" (numbers) in
each reasoning trace.

---

### 1.4 DeepConf: Deep Think with Confidence (arxiv 2508.15260, Meta AI)

**Method:** Filter low-confidence reasoning traces before voting, using group-level confidence.

**Formulas:**
```python
# Token confidence (negative avg logprob of top-k tokens)
conf_t = -mean(logprob of top-1 token at position t)

# Group confidence: sliding window average
group_conf_i = mean(conf_t for t in window_i)  # window = 2048 tokens

# Lowest group confidence (weakest link)
lowest_gc_i = min(group_conf_window for all windows in chain i)

# Threshold: keep top-eta% most confident chains
s = percentile(100 - eta, [conf_0, conf_1, ..., conf_{N-1}])

# After filtering, apply plain majority vote on survivors
final_answer = majority_vote(answers for chains with conf >= s)
```

**Results:**
- 99.9% on AIME 2025 (vs 97.0% for plain majority vote) using GPT-OSS-120B at N=64
- 43-79% token reduction by stopping low-confidence chains early
- The "lowest group confidence" metric is the best predictor of chain quality

**vLLM compatibility:** Yes, but complex. The full DeepConf requires modifying the inference
loop to track group confidence DURING generation and stop early. For post-hoc filtering, just
use the mean token confidence of completed outputs.

**Verdict:** The filtering approach (discard bottom X% by confidence before voting) is
straightforward to implement post-hoc on vLLM outputs. This is the second highest priority.

---

### 1.5 Self-Certainty with Borda Voting (arxiv 2502.18581)

**Method:** KL divergence from uniform distribution per token, then Borda rank voting.

**Formula:**
```python
# Self-certainty: how far is distribution from uniform?
SC_i = -1/(n*V) * sum(sum(log(V * p(j|context)) for j in vocab) for token in chain)

# Borda voting: rank by SC, assign weight = (N - rank + 1)^p, p=1.2
borda_weight_i = (N - rank_i + 1) ** 1.2
```

**Results (Llama-3.1-8B, N=64):**
| Method | LiveBench-Math | GSM8K | MATH | Avg |
|--------|---|---|---|---|
| Majority vote | 26.25% | 90.99% | 63.40% | 56.15% |
| Borda (p=1.2) | 26.69% | 90.95% | 64.10% | 56.51% |

**Verdict:** Very small improvement (+0.36% average). Borda is more complex than inverse-entropy
and the gain is smaller. Skip this.

---

### 1.6 GenSelect (AIMO2 Winner, arxiv 2504.16891 + 2507.17797)

**Method:** Use a reasoning LLM to SELECT the best solution from N candidates.

**Procedure:**
- Generate 64 candidate solutions
- Group into subsets of 16
- Feed each subset to a selection model: "Analyze these solutions, identify the best approach,
  output the index of the best solution."
- Run majority vote over the selection model's chosen indices
- Then majority@8 over final selected answers

**Results:**
| Benchmark | Majority@8 | GenSelect RL 1.7B |
|---|---|---|
| AIME 2024 | 63.33% | 65.00% |
| AIME 2025 | 52.56% | 54.17% |
| HMMT 2025 | 25.00% | 36.25% |

**Key catch:** GenSelect requires EITHER a separately trained selection model (RL fine-tuned)
OR using the same model with a new inference pass per subset. At AIMO3's N=8 samples and
30-minute budget, this adds ~2-4x more compute.

**Off-the-shelf alternative:** Use the reasoning model itself with a selection prompt
(Qwen3-1.7B prompting without RL showed weaker results than trained models). The AIMO2 winners
used a specifically RL-trained 1.7B selector.

**Verdict:** Not practical with current budget. Good for future if compute increases. The
trained selector approach is the right direction but requires additional training.

---

## 2. The AIMO3-Specific Ground Truth (arxiv 2603.27844)

This paper (March 2026, Natapong Nitarach) is the most important document because it directly
tested on the 50 AIMO3 problems with GPT-OSS-120B:

**Tested approaches and outcomes:**
| Approach | Mean Score (out of 50) | Delta vs Baseline |
|---|---|---|
| Baseline (N=8, T=1.0, original prompt) | 39.7 (σ=1.7) | 0 |
| Entropy-weighted voting w=1+1/(H+0.1) | ~39-40 | ~+0.3, within noise |
| Conservative diversity mix | ~40.0 | +0.3 |
| Aggressive diversity mix | ~40.0 | +0.3 |
| Equal prompt mix | ~38.0 | -1.7 |
| Code-first prompt | ~37-38 (noisy) | -1.7 to +1.3 |

**The brutal truth:** No voting trick reliably moved the score by more than 1 problem.
The baseline stdev was 1.7 problems, so ±2 is within noise. Even the "best" alternatives
achieved +0.3 at most, which is less than 1 standard deviation.

**Why entropy weighting showed marginal gains:**
"When models make mistakes, they tend toward diverse incorrect answers rather than correlated
ones." This means erroneous solutions already spread across many wrong answers, so the
entropy signal doesn't cleanly separate wrong from right.

---

## 3. The Key Untested Ideas

### 3.1 Boxed-answer-specific logprobs (NOVEL, not tested anywhere)

**Idea:** Instead of weighting by full-response entropy, weight only by the logprobs of the
tokens INSIDE `\boxed{...}`.

**Rationale:** The model may have low confidence throughout a long reasoning chain but high
confidence in the final boxed answer (or vice versa). The boxed answer tokens are the critical
signal. A model that writes `\boxed{42}` with high token probability is more reliable than
one that writes it with low probability.

**Implementation approach:**

The idea is to find the FINAL `\boxed{...}` span in the decoded response text, then identify
the corresponding token positions in the vLLM logprobs output, and take the mean logprob over
those positions only.

**Critical implementation note:** Do NOT pattern-match on decoded token strings directly.
The GPT-OSS-120B tokenizer (Llama/Qwen BPE) may split `\boxed{42}` into tokens like
`\`, `box`, `ed`, `{`, `42`, `}` or `▁\boxed`, `{`, `42`, `}` or other variants depending
on context. Matching on `"boxed" in decoded_token` will break silently on some tokenizations.

Correct approach:
1. Fully decode the response text
2. Find the character span of the LAST `\boxed{...}` using regex on the full text
3. Reconstruct token spans by accumulating decoded tokens and tracking character offsets
4. Average logprobs over token indices that fall inside the boxed span

```python
import re

def boxed_answer_logprob_weight(tokens: list[str], logprobs: list[float]) -> float:
    """Weight = mean logprob of the final \\boxed{...} tokens.
    Falls back to mean of full response if no boxed found."""
    full_text = ''.join(tokens)
    # Find the LAST \\boxed{...} in the text
    matches = list(re.finditer(r'\\boxed\{([^}]*)\}', full_text))
    if not matches:
        return sum(logprobs) / max(len(logprobs), 1)  # fallback
    span_start, span_end = matches[-1].start(), matches[-1].end()
    # Map character offsets back to token indices
    char_pos = 0
    box_logprobs = []
    for tok, lp in zip(tokens, logprobs):
        tok_start = char_pos
        tok_end = char_pos + len(tok)
        if tok_start >= span_start and tok_end <= span_end:
            box_logprobs.append(lp)
        char_pos = tok_end
    return sum(box_logprobs) / max(len(box_logprobs), 1)
```

**Note:** vLLM returns decoded tokens via `logprob.decoded_token` per position. Accumulate
these to reconstruct text with character offsets, then identify the boxed span.

**Status:** No paper has tested this. The hypothesis is that boxed-answer token confidence
is a cleaner signal than full-chain entropy. It's novel and worth testing.

**Expected impact:** Unknown. Could be +0 to +2 problems. Risk is low (can fall back to
plain majority vote if it underperforms).

---

### 3.2 Code-execution confidence interaction with entropy (NOVEL)

**Proposed formula:**
```
w_i = (1/entropy_i) * code_multiplier_i
where code_multiplier_i = 1.0  (no code executed OR no code success)
                        = 1.2  (code executed successfully)
                        = 0.8  (code executed but failed)
```

**The question from the task:** "Could we weight by (1/entropy) * (python_calls > 0 ? 1.2 : 0.8)?"

**Research says:** The AIMO3 paper tested code-first strategies and found them unreliable.
Plain code-first execution gave mixed results. The current `voting.py` already gives +2.0 weight
to code-succeeded responses. Adding entropy multiplicatively is reasonable but unproven.

**Verdict:** The multiplicative code_multiplier on top of entropy is not supported by evidence.
The +2.0 additive bonus for code success is already in place. Keep them separate rather than
multiplicative -- additive is more interpretable and less risk of amplifying wrong signals.

---

### 3.3 CER on math: number-token confidence (IMPLEMENTABLE, untested for AIMO3)

**Proposed implementation for AIMO3:**
```python
def cer_weight(response_text: str, logprobs: list[float], tokens: list[str]) -> float:
    """CER-style confidence: product of probs of number tokens in reasoning."""
    import re
    number_pattern = re.compile(r'^\d+$')  # pure numeric tokens

    step_confidences = []
    current_step_logprobs = []

    for tok, lp in zip(tokens, logprobs):
        if number_pattern.match(tok.strip()):
            current_step_logprobs.append(lp)
        if tok in ['\n', '.', ')', ']']:  # end of step
            if current_step_logprobs:
                # Joint prob of number tokens in this step
                joint_lp = sum(current_step_logprobs)  # sum of logprobs = log of product
                step_confidences.append(joint_lp)
                current_step_logprobs = []

    if not step_confidences:
        return 1.0  # No intermediate numbers found

    # Later steps weighted more (j * weight)
    n = len(step_confidences)
    weighted_sum = sum((j+1) * c for j, c in enumerate(step_confidences))
    weight_total = sum(range(1, n+1))
    return weighted_sum / weight_total  # Higher (less negative) = more confident
```

**This is implementable TODAY with vLLM's `logprobs=1` parameter.**

---

## 4. What vLLM Can Provide

vLLM's SamplingParams:
```python
SamplingParams(
    logprobs=20,         # Returns top-20 token logprobs per position
    prompt_logprobs=1,   # Can get logprobs for specific prompt tokens
)
```

From `output.outputs[0].logprobs`:
- List of dicts, one per generated token
- Each dict: `{token_id: Logprob(logprob=float, rank=int, decoded_token=str)}`
- The CHOSEN token is always included, plus up to `logprobs` alternatives

**Computing entropy from top-k logprobs (approximation):**
```python
import numpy as np

def approx_entropy_from_topk(logprobs_dict: dict) -> float:
    """Approximate entropy from top-k logprobs. True entropy underestimated
    because remaining probability mass is distributed across vocab-k tokens."""
    lps = [v.logprob for v in logprobs_dict.values()]
    probs = np.exp(lps)
    probs = probs / probs.sum()  # renormalize top-k to sum to 1 (approximation)
    return -np.sum(probs * np.log2(probs + 1e-10))
```

This approximation is what DeepConf, the Sequential Edge paper, and Self-Certainty all use
in practice (none of them have access to the full vocab distribution either).

---

## 5. Jonathan Chan / Bayesian Posterior Voting

No public notebook by Jonathan Chan on Bayesian posterior voting for AIMO3 was found.
The kaggle.com/jonathanchan profile exists but no AIMO3-specific public notebooks appeared
in search results. The term "VOI (Value of Information) early stopping" did not appear in any
AIMO3 research context found.

This was likely a rumor or a private/unpublished approach.

---

## 6. "Proof or Bluff" (arxiv 2503.21934) -- What It Actually Suggests

The paper evaluates reasoning models on USAMO 2025 (proof-required problems, not integer-answer).
Key finding: Gemini-2.5-Pro scored 25%, all others under 5%.

**Voting relevance:** None directly. The paper uses human expert annotation, not automated
voting. The insight for AIMO3 is that "pattern recognition shortcuts" produce correct INTEGER
answers even when the reasoning is wrong. This means entropy of the REASONING may not correlate
with correctness -- a confidently wrong chain may have low entropy too.

**Implication for voting:** If the model can reach the right answer via heuristic shortcuts,
high-entropy reasoning is not necessarily "wrong reasoning." This weakens the theoretical case
for entropy weighting (but doesn't eliminate it empirically).

---

## 7. Recommended Implementation Priority

### Priority 1: Already done -- 1/entropy is live in v14

The submission notebook already runs `w = 1 / max(entropy, 1e-9)` in `_select_answer`.
No change needed here. See `submission_v14_verify.ipynb` cell 13, `_select_answer` method.

### Priority 2: Boxed-Answer Logprob Weight (NOVEL, UNKNOWN RISK)

Weight by the geometric mean logprob of tokens inside `\boxed{...}`.
Untested anywhere -- could be +0 to +2 problems if the hypothesis is right.

### Priority 3: CER-Style Intermediate Number Confidence (MODERATE COMPLEXITY)

Weight by confidence of intermediate numerical computations within each chain.
ACL 2025 paper shows +7.4% on MATH, but that's a weak model (OLMo-2-7B).
GPT-OSS-120B may show smaller gains. Complexity: needs token-level parsing.

### Priority 4: DeepConf Filtering (NOT RECOMMENDED at N=8)

DeepConf filters the bottom X% of chains by confidence before voting. At N=64 this is effective
(discard 16, keep 48 strong chains). At N=8 (current AIMO3 budget), discarding 2 chains means
voting on only 6 responses. Given the baseline stdev of 1.7 problems on 50, losing 2 responses
is more likely to HURT than help -- you reduce your effective sample by 25% for at most
marginal confidence-signal benefit.

**Only consider this if N is increased to >= 16.** With N=16, discarding the bottom 2-4
chains has evidence of benefit (DeepConf showed 99.9% vs 97% at N=64 on AIME 2025).

If implemented with a fallback:
```python
def filter_and_vote(answers, logprobs_per_response, keep_fraction=0.75, min_keep=6):
    if len(answers) < min_keep / keep_fraction:
        return majority_vote(answers)  # too few -- skip filtering
    chain_confs = [mean_logprob(lps) for lps in logprobs_per_response]
    threshold = np.percentile(chain_confs, (1 - keep_fraction) * 100)
    kept = [a for a, c in zip(answers, chain_confs) if c >= threshold]
    if len(kept) < min_keep:
        return majority_vote(answers)  # fallback if filter too aggressive
    return majority_vote(kept)
```

### Already Live (no action needed):
- 1/entropy weighted voting -- in v14 `_select_answer`

### Do NOT Implement:

- Code-execution multiplier on top of entropy (additive bonus already exists, multiplicative
  combination unproven and risks amplifying noise)
- Diverse prompt mixing (AIMO3 paper showed this HURTS by -1.7 problems)
- GenSelect (requires separate trained model, too expensive for current budget)
- CISC P(True) verification (adds latency with marginal absolute gain on hard problems)
- Borda count with self-certainty (minimal improvement, high complexity)

---

## 8. The Honest Bottom Line

The research consensus (especially the AIMO3-specific paper) is clear:

1. **Entropy weighting is already the right approach** -- it's what "1/entropy" does. The
   current voting.py does NOT implement this. Adding it is the right move.

2. **The gain is likely small** -- ±1 problem on 50. The AIMO3 paper found +0.3 problems
   (within noise). Don't expect a breakthrough.

3. **No voting method beats model capability** -- a better base model beats any voting trick
   by 10-17 problems. If GPT-OSS-120B is the model, voting optimization has a ceiling.

4. **The one genuinely novel idea** is boxed-answer-specific logprobs. No paper has tested
   this. It's the most interesting experiment because it directly weights the CRITICAL TOKENS
   (the answer digits) rather than the noisy reasoning chain.

5. **CER is the most principled untested method** for math -- weighting by intermediate
   numerical token confidence is theoretically sound and has paper evidence on similar tasks.

---

## References

- CISC (Confidence Improves Self-Consistency): arxiv 2502.06233, ACL 2025 Findings
  https://arxiv.org/abs/2502.06233

- The Sequential Edge (Inverse-Entropy Voting): arxiv 2511.02309
  https://arxiv.org/abs/2511.02309

- AIMO3 Inference-Time Optimization Lessons: arxiv 2603.27844
  https://arxiv.org/abs/2603.27844

- CER (Confidence Enhanced Reasoning): arxiv 2502.14634, ACL 2025
  https://arxiv.org/abs/2502.14634

- DeepConf (Deep Think with Confidence, Meta AI): arxiv 2508.15260
  https://arxiv.org/abs/2508.15260

- Self-Certainty Best-of-N: arxiv 2502.18581
  https://arxiv.org/abs/2502.18581

- AIMO2 Winning Solution + GenSelect: arxiv 2504.16891
  https://arxiv.org/abs/2504.16891

- GenSelect A Generative Approach to Best-of-N: arxiv 2507.17797
  https://arxiv.org/abs/2507.17797

- Proof or Bluff (USAMO 2025 evaluation): arxiv 2503.21934
  https://arxiv.org/abs/2503.21934

- Ranked Voting based Self-Consistency: ACL 2025 Findings
  https://aclanthology.org/2025.findings-acl.744

- CISC GitHub implementation: https://github.com/taubenfeld/CISC
- CER GitHub implementation: https://github.com/sharif-ml-lab/CER
