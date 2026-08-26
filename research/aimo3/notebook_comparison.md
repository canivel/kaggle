# AIMO3 Public Notebook Technical Comparison
**Research date:** 2026-04-01
**Method:** Direct `kaggle kernels pull` on all 40+ scoring notebooks

---

## Leaderboard Context

| Rank | Team | Score | Public? |
|------|------|-------|---------|
| 1 | ippeiogawa | **46** | No |
| 2-5 | Various | **45** | No |
| 6-20 | Various | **44** | Some |
| (best public) | nihilisticneuralnet / kaanyorgun | **44** | Yes |
| (reference) | datasciencegrad | **42** | Yes |
| (reference) | amanatar | ~41 | Yes |

---

## Master Comparison Table

| Parameter | nihilist-44 (overcook) | nihilist-43 (weighted-entropy) | nihilist-41 (confidence) | kaanyorgun-44 | amanatar (~41) | jonathanchan (~40, Bayesian) | bhargavaabhi (~41, winner) | datasciencegrad-42 | andreasbis-with-tools (~40) | gelart (multitemp) | shelterw-AIME | **OUR v10 (current)** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **LB Score** | **44** | 43 | 41 | **44** | ~41 | ~40 | ~41 | 42 | ~40 | unknown | 15/15 AIME | unknown |
| **temperature** | **0.5** | 1.0 | 1.0 | 1.0 | 0.8 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 + top_p=0.95 | 1.0 | **0.8** |
| **min_p** | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |
| **top_p** | - | - | - | - | - | **0.95** | - | - | - | 0.95 | - | - |
| **attempts** | 8 | 8 | 8 | 8 | **12** | 8 | 8 | 8 | 8 | 8 | 8 | 8 |
| **early_stop** | 4 | 4 | 4 | 4 | 4 | VoI-based | 4 | 4 | 4 | >4 | 4 | 4 |
| **workers** | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| **batch_size** | 256 | 256 | 256 | 256 | **128** | 256 | 256 | 256 | **64** | 256 | 256 | 256 |
| **context_tokens** | 65536 | 65536 | 65536 | 65536 | 65536 | 65536 | 65536 | 65536 | **81920** | 65536 | 65536 | 65536 |
| **max-num-batched-tokens** | - | - | - | - | - | - | - | - | **2048** | - | - | - |
| **max-cudagraph-capture-size** | - | - | - | - | - | - | - | - | **64** | - | - | - |
| **high_problem_timeout** | 900 | 900 | 900 | 900 | 900 | 900 | 900 | 900 | 900 | 900 | 900 | 900 |
| **base_problem_timeout** | 300 | **270** | **270** | 300 | 300 | 300 | 300 | **270** | **270** | 300 | 300 | 300 |
| **model** | **default** | default | default | default | default | **huikang 160a/14** | default | default | default | default | **danielhanchen** | **huikang 160a/20** |
| **voting** | plain entropy-weighted | position-weighted entropy + variance + streak | AdvancedVoting (exp conf) | plain entropy-weighted | plain entropy-weighted | **Bayesian posterior (entropy * reliability * tool_bonus)** | plain entropy-weighted | **AdvancedVoting (exp conf)** | plain entropy-weighted | plain entropy-weighted | plain entropy-weighted | entropy * reliability * code_bonus |
| **verify step** | no | no | no | no | **YES: _verify_answer greedy call** | no | no | no | no | no | no | **YES: follow-up greedy call** |
| **system prompt** | 5-step UNDERSTAND/EXPLORE/PLAN/EXECUTE/VERIFY | minimal | 5-step | 5-step | 5-step + INTERNAL PROTOCOL + verification standard | minimal (force Python mandatory) | minimal | 5-step | minimal | **6-step (+ REFLECT)** + SELF-CHECK | 5-step | 5-step + domain-specific + verification |
| **seed** | 42 | 42 | 42 | 42 | 42 | 42 | 42 | 42 | 42 | 42 | 42 | 42 |
| **_scan_for_answer extras** | + "final answer is" | + "final answer is" | + "final answer is" | + "final answer is" | + "final answer is" | + "final answer is" | + "final answer is" | + "final answer is" | + "final answer is" | + "answer: X" + "final answer is" | + "final answer is" | **+ negatives mod 100000 + "answer is" + "final answer is"** |

---

## Key Technical Findings

### Finding 1: The 43 -> 44 Jump (ANALYSIS - Multiple Confounded Variables)
The transition from 43 to 44 involves multiple changes. No single factor is proven in isolation.

Nihilist's progression:
- nihilist-41 (41/50): `5-step UNDERSTAND/EXPLORE/PLAN/EXECUTE/VERIFY prompt` + T=1.0 + AdvancedVoting (complex confidence-based)
- nihilist-43 (43/50): `SHORT 2-line prompt` ("You are a world-class IMO competitor...") + T=1.0 + complex position-weighted entropy voting
- nihilist-44 (44/50): `5-step prompt` (reverted) + T=0.5 + **simple mean entropy** (reverted from complex)

**IMPORTANT**: nihilist-41 also used the 5-step prompt and scored LOWER (41) than nihilist-43's short prompt (43). The prompt alone cannot explain the difference. The voting mechanism changed too:
- nihilist-41: AdvancedVoting (complex) -> 41
- nihilist-43: complex position-weighted entropy -> 43
- nihilist-44: **simple mean entropy** -> 44

The evidence suggests the combination of (5-step prompt + simple mean entropy voting) at T=0.5 drives 44. You cannot isolate which change matters most.

**kaanyorgun-44 (44/50)**: Uses the 5-step prompt with "Mathematical Reasoning Principles" + T=1.0 + plain entropy-weighted voting = 44.
This CONFIRMS: T=0.5 is not required. The 5-step prompt + plain entropy voting is sufficient for 44.

**Key pattern**: Both 44-scoring notebooks use **plain `1/entropy` weighted voting** (not the complex formulas in nihilist-41 or nihilist-43). This is the strongest consistent factor.

**OUR v10**: Uses `entropy * reliability * code_bonus` weighted voting. This is MORE complex than plain `1/entropy`. Based on the evidence, our weighted voting might be HURTING us vs plain entropy. This is a cheap test to run.

**kaanyorgun vs our v10**: Both use 5-step prompt + similar structure. Key difference: (1) they use default model vs our huikang 160a/20, (2) they use plain entropy vs our reliability/code_bonus weighting.

### Finding 2: System Prompt Length & Structure
The 5-step UNDERSTAND/EXPLORE/PLAN/EXECUTE/VERIFY prompt appears in both 44/50 notebooks and in nihilist-41 (41/50). The short 2-line prompt appears in nihilist-43 (43/50). This means neither prompt form has a clean correlation with score on its own.

**CORRECTED CLAIM**: The short prompt is NOT associated with low scores; nihilist-43 scored 43 with it (best performance at that time among nihilist's notebooks). The prompt appears to be a secondary factor, not the primary driver. What changed between 43 and 44 was both the prompt AND the voting mechanism (back to plain entropy).

Short 2-line prompts for bhargavaabhi, jonathanchan, andreasbis (all ~40): These notebooks also lack the 5-step prompt, but also use different models or different configs. Cannot isolate prompt effect.

**Notable variant (gelart, scoring unknown):** Adds a 6th step "REFLECT: Review your entire solution. Question each step." and adds "Critical Requirements" block with SELF-CHECK instructions. This is the only notebook with explicit self-checking prompting.

### Finding 3: Answer Verification (amanatar vs our follow-up)
**amanatar `_verify_answer`:**
```python
def _verify_answer(self, problem, answer):
    prompt = f"Problem:\n{problem}\n\nProposed answer: {answer}\n\nCheck the answer carefully.\nReply with only ONE word:\nCORRECT or WRONG"
    resp = self.client.completions.create(
        model=..., prompt=prompt_ids, temperature=0.0, max_tokens=5
    )
    return "CORRECT" in text and "WRONG" not in text
```
Called only on candidates with >=2 votes, sorted by average entropy. Fallback to entropy-weighted voting if no candidate passes.

**Our follow-up (v10):** If no answer extracted, injects "Based on your analysis so far, what is the final integer answer?" at temperature=0.0.

Key difference: amanatar verifies AFTER the attempt completes, using a separate single-token call to check a proposed answer. Our approach tries to extract an answer from an existing conversation. They solve different problems.

### Finding 4: Jonathan Chan's Bayesian VoI Stopping
Replaces `early_stop = 4` with adaptive stopping:
```python
posterior[answer] = entropy_weight * reliability * tool_bonus  # normalized
# Stop when submit_utility >= continue_utility
# submit_utility = max(posterior.values())
# continue_utility = max_prob + entropy * voi_entropy_weight(0.6) - voi_compute_cost(0.04)
```
Stops earlier when one answer is highly confident, later when posterior is still uncertain.
- `reliability = 1/(1 + error_penalty * python_errors)` penalizes error-prone attempts
- `tool_bonus = 1.2` if Python used with no errors, `0.8` if errors, `1.0` otherwise

### Finding 5: Model Variants
Three model paths (CONFIRMED from actual notebook code):
1. **Default** `/kaggle/input/gpt-oss-120b/transformers/default/1`: Used by nihilist-44, nihilist-43, kaanyorgun-44, amanatar, datasciencegrad, bhargavaabhi, andreasbis, gelart, nurikw3
2. **danielhanchen** `/kaggle/input/models/danielhanchen/gpt-oss-120b/transformers/default/1`: Used by shelterw AIME 15/15 ONLY
3. **huikang 160a/14** `/kaggle/input/models/huikang/gpt-oss-120b-aimo3/transformers/160a/14`: Used by Jonathan Chan only. Version 20 (`/14` -> `/20`) is what our v10 uses.

**CRITICAL CORRECTION**: nihilist-44 uses the DEFAULT model, NOT danielhanchen. Both public 44/50 notebooks (nihilist-44 and kaanyorgun-44) use the default model. The confirmed differentiators between 43 and 44 are (1) the 5-step system prompt structure and (2) temperature=0.5 in nihilist-44's case. Model variants are NOT responsible for the 44/50 score.

### Finding 6: andreasbis vLLM Flags (unique)
Only notebook with non-default vLLM flags:
- `context_tokens = 81920` (vs 65536 standard) - 25% longer context window
- `batch_size = 64` (vs 256 standard) - 4x smaller batch to support longer context
- `--max-num-batched-tokens 2048`
- `--max-cudagraph-capture-size 64`

This trades throughput for context length. No LB score known.

### Finding 7: amanatar's 12 Attempts
`attempts=12, batch_size=128` vs standard `attempts=8, batch_size=256`. Amanatar also has an `ANSWER_ONLY_PROMPT` defined but there's no evidence it's used in the main inference loop. The temperature is 0.8 (same as our current).

### Finding 8: Gelart's Multi-Temperature Framework (commented out but revealing)
The gelart notebook has commented-out multi-temperature sampling groups:
```python
# sampling_groups = [
#     (1.1, 0.01),  
#     (1.0, 0.02),  # standard
#     (0.9, 0.03),   
#     (0.8, 0.04),
#     (0.7, 0.05),
#     (0.6, 0.06),
#     (0.5, 0.07),
#     (0.4, 0.08),
#     (0.3, 0.09),
# ]
```
But the active config is `sampling_groups = [(1.0, 0.95)]` with top_p=0.95. The commented structure suggests the author explored per-attempt diversity via different temperatures. This is currently not used.

### Finding 9: Gelart's Enhanced _scan_for_answer
The ResultExtractor adds a third pattern:
```python
r'answer\s*[:=]\s*([0-9,]+)'
```
This catches "answer: 42" and "answer = 42" formats that plain `\boxed{}` and "final answer is" miss.

---

## What Our v10 Does vs What 44+ Does

| Feature | Our v10 | nihilist-44 | kaanyorgun-44 | Gap? |
|---------|---------|-------------|---------------|------|
| temperature | 0.8 | **0.5** | 1.0 | Try 0.5 (may help) |
| model | huikang 160a/20 | **default** | **default** | YES - both use default; we use fine-tuned |
| system prompt | 5-step + domain | 5-step | 5-step + principles | SIMILAR - no gap here |
| follow-up for no-answer | greedy turn injection | no | no | We add extra coverage |
| answer verification | no | no | no | amanatar has this |
| _scan extras | negatives mod 100k, "answer is" | "final answer is" | "final answer is" | We're better |
| attempts | 8 | 8 | 8 | Same |
| voting | entropy * reliability * code_bonus | plain entropy-weighted | plain entropy-weighted | We add reliability/bonus |

---

## Top Techniques We're Missing

### Priority 1: Plain Entropy Voting vs Our Weighted Formula (CHEAPEST TEST)
Both 44/50 notebooks use plain `weight = 1.0 / entropy` voting (no reliability or code_bonus multipliers).
Our v10 uses `entropy * reliability * code_bonus` which is MORE complex than what scores 44.
The evidence from nihilist's progression shows that complex voting formulas HURT performance:
- nihilist-41: AdvancedVoting (complex) -> 41
- nihilist-43: complex position-weighted entropy -> 43
- nihilist-44: simple 1/entropy -> 44
Test: Change `_select_answer` to use plain `weight = 1.0 / max(entropy, 1e-9)`. No infrastructure change needed.

### Priority 2: Temperature = 0.5 (nihilist-44 path; worth testing)
nihilist-44 scored 44/50 with temperature=0.5. kaanyorgun-44 scores 44/50 with T=1.0.
Temperature is NOT required for 44, but may provide an additional boost.
Our current 0.8 is between. Experiment: try 0.5 on our infrastructure.

### Priority 3: Default Model vs huikang 160a/20 (circumstantial but worth testing)
Both 44/50 public notebooks use the DEFAULT model. We use huikang 160a/20.
This is correlation, not causation (most teams use default as the path of least resistance).
The huikang model is fine-tuned specifically for AIMO3 which could be better or worse.
Only Jonathan Chan uses huikang (~40) but he has many other differences; can't isolate model effect.
Test: Submit with default model + our v10 config.

### Priority 4: amanatar's Answer Verification Pass
After collecting all attempt results, run a separate greedy verification call:
"Problem: X. Proposed answer: Y. Check carefully. Reply CORRECT or WRONG."
Only costs time on 2-vote candidates (usually 1-2 calls per problem).

### Priority 5: 6-step REFLECT prompt (gelart)
Adding a REFLECT step and explicit SELF-CHECK requirement may improve reasoning quality.
Gelart's score is unknown but the approach is theoretically sound.

### Priority 6: 12 Attempts with batch_size=128 (amanatar)
More total attempts at the cost of smaller concurrent batch. May improve
answer quality through better voting with more diverse samples.

### Priority 7: Multi-temperature Diversity (gelart framework, commented out)
Instead of all 8 attempts at the same temperature, use different temperatures:
e.g., 4 attempts at 0.5, 2 at 0.8, 2 at 1.0. This would give the model both
the precision of T=0.5 and the diversity of T=1.0 in a single run.

### Priority 8: Bayesian VoI Stopping (Jonathan Chan)
More principled early stopping that considers answer confidence vs cost of more attempts.
Saves time when already confident, spends more time when uncertain.

---

## What Did NOT Help (Evidence-Based)

1. **nihilist-43's weighted entropy formula** (position-weighted + variance + streak): Scored 43 vs nihilist-44's simple mean at 44. The complexity hurt.
2. **AdvancedVotingMechanism** (datasciencegrad-42, nihilist-41): Both use it, scoring 42 and 41 respectively. Lower than nihilist-44 with plain entropy. May be neutral or slightly harmful.
3. **Bayesian posterior voting** (Jonathan Chan): Unknown LB score, but uses a different model (huikang 160a/14), making it impossible to isolate the voting effect.
4. **danielhanchen model variant**: Only used by shelterw (AIME 15/15 which is a different task). NOT used by nihilist-44 or kaanyorgun-44. The earlier claim that nihilist-44 used danielhanchen was WRONG.
5. **SHORT 2-line system prompt as the bottleneck**: nihilist-43 scored 43 WITH the short prompt; the prompt alone didn't cap performance at 41. The short prompt is likely suboptimal but is not clearly the main driver of the 43->44 gap.
6. **Complex voting formulas** (position-weighted entropy, streak bonuses, AdvancedVoting, reliability * code_bonus): All evidence points to plain `1/entropy` weighting outperforming complex formulas. Three different complex voting implementations (nihilist-41, nihilist-43, our v10) all score below or equal to plain entropy users.
7. **huikang fine-tuned model as a proven improvement**: Jonathan Chan uses huikang 160a/14 and scores ~40; all 41-44 public notebooks use the default model. However, model effect is confounded with other config differences - this is not proven negative, just unproven positive.

---

## The 45-46 Gap (Speculation)

No public 45/46 notebooks exist. Based on the progression:
- 43->44: Two paths confirmed: (a) nihilist: change only T from 1.0 to 0.5; (b) kaanyorgun: T=1.0, different system prompt/logic
- 44->45: Likely involves more attempts (10-12) or better model or ensemble approach
- 45->46: ippeiogawa's submission on the last day suggests either:
  - Optimal hyperparameter combination discovered
  - Fine-tuned model not accessible publicly
  - Lucky problem sample (at 46/50 = 92%, only 4 wrong)

---

## Recommended Experiment Order

1. **Plain entropy voting** - change `_select_answer` to `weight = 1.0 / max(entropy, 1e-9)` only; no other changes. Cheapest test, strongest empirical support.
2. **temp=0.5** on current v10 infrastructure (nihilist-44 path)
3. **plain entropy + temp=0.5** combined (exact nihilist-44 voting + temperature)
4. **default model** (switch from huikang 160a/20 to `/kaggle/input/gpt-oss-120b/transformers/default/1`)
5. **amanatar verification pass** (add _verify_answer greedy check on >=2-vote candidates)
6. **gelart 6-step REFLECT prompt** (add REFLECT + SELF-CHECK to our 5-step)
7. **12 attempts + batch_size=128** (amanatar path)
8. **Multi-temperature: 4x0.5 + 2x0.8 + 2x1.0**

---

*Sources: Direct notebook pulls via `kaggle kernels pull` on 2026-04-01.*
*Notebooks pulled to: F:/kaggle/research/aimo3/notebooks/*
