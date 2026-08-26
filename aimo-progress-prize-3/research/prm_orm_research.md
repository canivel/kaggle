# Process Reward Models (PRM) & Outcome Reward Models (ORM) for AIMO3
## Research Date: 2026-04-01

---

## Executive Summary

Using a PRM to rerank 8 candidate solutions instead of majority voting is a proven
technique that consistently adds 1-14+ accuracy points on math benchmarks. The
concrete AIMO3 path: load Qwen2.5-Math-PRM-7B (~14GB fp16, ~7GB 4-bit) as a
second process alongside the solver, score each solution trace step-by-step, pick
the highest-scoring answer. This "best-of-N with PRM" approach is well-established
and feasible within the H100 80GB VRAM budget.

---

## 1. PRM vs ORM: What's the Difference?

| | ORM (Outcome Reward Model) | PRM (Process Reward Model) |
|---|---|---|
| Scores | Final answer only | Each reasoning step |
| Signal density | Sparse (one score per solution) | Dense (per-step scores) |
| Best-of-N quality | Good | Better on hard problems |
| Reward hacking risk | Low | Higher (can repeat correct steps) |
| Training data cost | Cheap | Expensive (step annotations needed) |

For competition math (hard, multi-step, rare correct answers), **PRMs consistently
outperform ORMs in best-of-N selection**. Research shows 10%+ advantage on hard
benchmarks (MATH-500, OlympiadBench) where correct solutions are rare and need to
be distinguished from plausibly-wrong ones.

---

## 2. Key Models Identified

### 2a. Qwen2.5-Math-PRM-7B [RECOMMENDED PRIMARY]

- **HuggingFace**: `Qwen/Qwen2.5-Math-PRM-7B`
- **Paper**: "The Lessons of Developing Process Reward Models in Mathematical
  Reasoning" (arXiv:2501.07301)
- **Base**: Fine-tuned from Qwen2.5-Math-7B-Instruct
- **Architecture**: 7B parameters, BF16, Qwen2.5 architecture
- **VRAM**: ~14GB fp16, ~7GB in 4-bit (no official quant released by Qwen,
  but bitsandbytes/AWQ should work since base arch is standard Qwen2.5)
- **Performance**: Outperforms maj@8 across all 7 benchmarks by avg 1.4%.
  On AIME2024 with a suitable policy model, RM helped reach 21/30 problems.
  Superior to all open-source PRMs at equivalent scale, outperforms GPT-4o-0806
  as judge.
- **Scoring method**: Insert `<extra_0>` after each step. Extract probability
  that token is classified "positive". Scores are in [0,1] per step.
- **Best aggregation**: Product of step scores (or min) — not average.

### 2b. Skywork-o1-Open-PRM-Qwen-2.5-7B [ALTERNATIVE]

- **HuggingFace**: `Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B`
- **Base**: Qwen2.5-Math-7B-Instruct (same as Qwen PRM)
- **VRAM**: ~14GB fp16
- **Performance** (Best-of-64 with Skywork-o1-Open-8B policy):
  - GSM8K: 96.7%, MATH: 87.0%, AIME-24: 23.3%
- **Note**: Better suited for code tasks as well (multi-task). Uses step_token="\n"
  as separator (simpler than Qwen's `<extra_0>` approach).
- **Requires**: Custom `model_utils` from
  https://github.com/SkyworkAI/skywork-o1-prm-inference

### 2c. Qwen2.5-Math-PRM-72B [TOO LARGE]

- **HuggingFace**: `Qwen/Qwen2.5-Math-PRM-72B`
- **VRAM**: ~144GB fp16, ~36GB 4-bit — does NOT fit in 14GB headroom
- **Performance**: Slightly better than PRM-7B, but not dramatically so
- **Verdict**: Unusable alongside GPT-OSS-120B on single H100

### 2d. ThinkPRM [EMERGING, COMPLEX]

- **Paper**: "Process Reward Models That Think" (arXiv:2504.16828)
- **Mechanism**: Generates a verification CoT before scoring each step
  (long chain-of-thought verifier)
- **Performance**: Outperforms discriminative PRMs by 7-8% on MATH-500 and
  AIME-24 using only 1% of PRM800K labels. Beats RLHFFlow-DeepSeek-PRM by
  7%+ across all beam sizes.
- **VRAM**: Unknown model size (fine-tuned from a reasoning model); likely 7B+
- **Code**: https://github.com/mukhal/ThinkPRM
- **Verdict**: Stronger quality but slower (generates verification CoT per step).
  Could be used offline for N=8 reranking if latency budget allows.

### 2e. Math-Shepherd [Historical Baseline]

- **Paper**: arXiv:2312.08935, ACL 2024
- **Mechanism**: Step-level reward via automated MC estimation (no human labels)
- **Performance**: Verification improved Mistral-7B from 77.9% to 89.1% on GSM8K
- **Status**: Superseded by Qwen/Skywork PRMs on hard benchmarks. Good conceptual
  baseline.

---

## 3. VRAM Feasibility Analysis for H100 80GB

### Current allocation:
- GPT-OSS-120B (solver): ~66GB fp16 (or quantized variant)
- H100 total: 80GB
- Headroom: **~14GB**

### Options that fit:

| Model | fp16 VRAM | 4-bit VRAM | Fits in 14GB? |
|---|---|---|---|
| Qwen2.5-Math-PRM-7B | ~14GB | ~7GB | Yes (tight fp16, comfortable 4-bit) |
| Skywork-o1-PRM-7B | ~14GB | ~7GB | Yes (same) |
| ThinkPRM-7B (estimate) | ~14GB | ~7GB | Yes (estimate) |
| Any 72B PRM | ~144GB | ~36GB | No |

### VRAM calculation:
- 7B params x 2 bytes (bf16) = 14GB for weights
- Plus KV cache and activations for scoring ~8 solutions
- 4-bit via bitsandbytes: 7B x 0.5 bytes = ~3.5GB + overhead ~7GB total
- **Recommendation**: Load PRM in 4-bit (bitsandbytes NF4) to leave headroom for
  solver's KV cache spill.

### Critical VRAM caveat: vLLM pre-allocates KV cache

vLLM fills all remaining GPU memory with KV cache blocks at startup. During active
solver inference, the ~14GB headroom is **not freely available** — it is consumed
by KV cache. The actual VRAM budget for a second model must be carved out explicitly:

- Set solver vLLM `gpu_memory_utilization=0.80` (leaves ~16GB free instead of 14GB)
- Or use vLLM sleep mode (https://blog.vllm.ai/2025/10/26/sleep-mode.html) which
  releases KV cache when the model is not actively generating
- With sleep mode: solver generates all 8 solutions, enters sleep, PRM scores them,
  solver wakes back up for next problem

### Deployment architecture options:

**Option A: vLLM sleep mode (recommended)**
1. Solver starts with `gpu_memory_utilization=0.82` (reserving ~14GB)
2. PRM loaded separately in 4-bit (~7GB) into reserved space
3. Solver generates 8 solutions, enters vLLM sleep (frees KV cache)
4. PRM scores all 8 solutions
5. Solver wakes, continues
6. Total peak: ~66GB (solver) + ~7GB (PRM) = ~73GB

**Option B: Reduced gpu_memory_utilization + always-on PRM**
1. Solver: `gpu_memory_utilization=0.82` — smaller KV cache, slightly slower generation
2. PRM: 4-bit Qwen-PRM-7B as persistent separate process
3. Both coexist at ~73GB total
4. Risk: reduced solver batch throughput due to smaller KV cache

**Option C: Same model for generation and verification (no extra VRAM)**
- GenSelect: Use the solver itself to read all N candidates and select the best
- See Section 6 — but note this requires a model trained on GenSelect, not zero-shot
- No VRAM change; accuracy gain depends heavily on whether solver has GenSelect training

---

## 4. Best-of-N with PRM: Expected Impact

### Empirical results from literature:

| Benchmark | Majority Vote | PRM Best-of-8 | Gain |
|---|---|---|---|
| MATH-500 | ~80% | ~82-84% | +2-4% |
| OlympiadBench | ~45% | ~50%+ | +5%+ |
| GSM8K | ~95% | ~96-97% | +1-2% |
| AIME 2024 (BoN) | depends on N | higher | significant |

- Average +1.4% gain across 7 tasks (Qwen PRM paper, N=8)
- Up to +6.7% on MATH-500 (Pairwise RM paper)
- +8.5% across 6 datasets (R-PRM paper, N=varies)
- The gain increases with problem difficulty — best effect on olympiad-level problems

### Critical caveat (arXiv:2502.00271):
"Verifier-guided search exhibits diminishing advantages as sample size increases and
eventually underperforms repeated sampling." The failure mode: at large N (N=64+),
the PRM may erroneously prune all valid paths if many wrong solutions exploit
reward hacking. At N=8 (our budget), this is NOT a problem — the PRM excels here.

---

## 5. How to Score Solutions with Qwen2.5-Math-PRM-7B

**Critical architecture note**: This model uses `modeling_qwen2_rm.py` (a custom
reward head with `nn.Linear(hidden_size, 2)` outputting 2 logits per token, NOT
vocabulary logits). The step scores are extracted at `<extra_0>` token positions.
The positive class is index `[:, 1]`. The code below is verified against the
official README.

```python
import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F


def make_step_rewards(logits, token_masks):
    """
    Official scoring function from Qwen2.5-Math-PRM-7B README.
    logits: (batch, seq_len, 2)  — reward head outputs, NOT vocab logits
    token_masks: (batch, seq_len) bool — True at <extra_0> positions
    Returns: list of lists, one per sample, each list = per-step scores in [0,1]
    """
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1)  # zero out non-step positions
    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i]  # (seq_len, 2)
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1]  # (n_steps,) — positive class
        all_scores_res.append(positive_probs.cpu().tolist())
    return all_scores_res


# Load model — trust_remote_code required (uses custom modeling_qwen2_rm.py)
model_name = "Qwen/Qwen2.5-Math-PRM-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    # Add for 4-bit to fit in ~7GB (saves VRAM on H100):
    # load_in_4bit=True,
).eval()

step_sep_id = tokenizer.encode("<extra_0>", add_special_tokens=False)[0]


def score_solution(problem: str, solution_steps: list[str]) -> float:
    """
    Score a solution given as a list of step strings.
    Returns the minimum step score (most conservative aggregation).
    Higher score = more likely correct.
    """
    # Interleave steps with <extra_0> separators
    response = "<extra_0>".join(step.strip() for step in solution_steps) + "<extra_0>"
    messages = [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": response},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
    token_masks = (input_ids == step_sep_id)  # shape: (1, seq_len)

    with torch.no_grad():
        output = model(input_ids=input_ids)
        step_rewards = make_step_rewards(output[0], token_masks)

    scores = step_rewards[0] if step_rewards else [0.5]
    return min(scores)  # use min: weakest step determines correctness


def split_into_steps(solution: str) -> list[str]:
    """
    Split a CoT solution into steps.
    Use double newlines as step boundaries. Adjust for your solver's format.
    For TIR solutions with code blocks, consider stripping code and keeping
    only the reasoning text before scoring.
    """
    steps = [s.strip() for s in solution.split("\n\n") if s.strip()]
    return steps if steps else [solution]


def pick_best_solution(problem: str, solutions: list[str]) -> str:
    """Given N candidate solutions, return the one with highest PRM score."""
    scores = []
    for sol in solutions:
        steps = split_into_steps(sol)
        scores.append(score_solution(problem, steps))
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return solutions[best_idx]
```

### TIR solution compatibility warning
The PRM was trained on pure CoT math solutions. TIR solutions interleave Python
code blocks (`<code>...</code>`) with reasoning text. Before scoring a TIR
solution, strip code blocks and keep only the natural language reasoning:

```python
import re

def strip_code_blocks(solution: str) -> str:
    """Remove code execution blocks from TIR solution before PRM scoring."""
    # Remove <code>...</code> or ```python ... ``` blocks
    solution = re.sub(r"<code>.*?</code>", "[code executed]", solution, flags=re.DOTALL)
    solution = re.sub(r"```python.*?```", "[code executed]", solution, flags=re.DOTALL)
    solution = re.sub(r"<output>.*?</output>", "", solution, flags=re.DOTALL)
    return solution
```

Use PRM scoring primarily on CoT-mode attempts; for TIR-only problems, fall back
to majority voting.

---

## 6. GenSelect: The Same-Model Selection Alternative

GenSelect (NVIDIA, arXiv:2602.02143) uses the solver itself to select among N candidates
by posing selection as a reasoning problem:

```
Prompt: "Here are 8 candidate solutions to [PROBLEM]. Systematically analyze 
each solution and identify the best approach. Output only the index (1-8)."

[Solution 1]: ...
[Solution 2]: ...
...
[Solution 8]: ...
```

### IMPORTANT: GenSelect requires a trained selector, not zero-shot prompting

NVIDIA's GenSelect numbers come from models fine-tuned on selection tasks using
DeepSeek R1 reasoning traces. Zero-shot prompting of an untrained model to "pick
the best of 8" is equivalent to LLM-as-a-judge, which the Qwen PRM paper found
performs WORSE than their PRM on hard problems. Do not assume the +14% numbers
apply to our solver without verifying it has GenSelect-style training.

**Check**: Does our solver (GPT-OSS-120B or OpenMath-Nemotron-14B) have GenSelect
training? For OpenMath-Nemotron-14B-Kaggle specifically, check whether the Kaggle
variant includes GenSelect fine-tuning.

### Performance (NVIDIA OpenMath-Nemotron-14B, models trained on GenSelect):
- maj@64 baseline: AIME24 86.7%, AIME25 73.3%, HMMT 64.8%, HLE-Math 6.5%
- + Self GenSelect (fine-tuned): AIME24 86.7%, AIME25 76.7%, HMMT 72.4%, HLE-Math 14.1%
- + 32B GenSelect (fine-tuned): AIME24 90.0%, AIME25 76.7%, HMMT 71.9%, HLE-Math 13.7%

Key result on hard problems: **HLE-Math 6.5% → 14.1%** with fine-tuned selector.
This is more relevant to AIMO3's IMO-level problems than AIME benchmarks.

### Timing consideration:
GenSelect reads all N solutions in a single forward pass, but the context is long
(8 full solutions concatenated = potentially 16K+ tokens). On H100, this takes
~5-15 seconds per problem. Budget this into the 30-min-per-problem limit.

### How to implement:
1. Generate 8 solutions as usual
2. Feed all 8 to solver (or a dedicated small GenSelect model) with selection prompt
3. Pick the indexed solution
4. No additional VRAM beyond reducing solver KV cache slightly for the long context

---

## 7. Recommended AIMO3 Pipeline

### Current pipeline:
```
Problem -> Solver (8 attempts) -> Majority vote -> Answer
```

### Upgraded pipeline (Option A: PRM reranking):
```
Problem -> Solver (8 attempts) -> PRM scores each solution -> Best score wins -> Answer
```

### Upgraded pipeline (Option B: GenSelect, no extra VRAM):
```
Problem -> Solver (8 attempts) -> Solver reads all 8, picks best -> Answer
```

### Upgraded pipeline (Option C: Combined):
```
Problem -> Solver (8 attempts) -> PRM scores -> Top 3 candidates -> GenSelect picks final -> Answer
```

### Rough time budget per problem (30 min limit):
- 8 CoT/TIR attempts: ~20-25 min (most of budget)
- PRM scoring 8 solutions (7B, 4-bit, ~1-3K tokens each): ~5-15 sec total (negligible)
- GenSelect (all 8 concatenated, ~16K tokens): ~15-30 sec (acceptable)
- ThinkPRM (generates verification CoT per step): ~1-3 min total (significant)

PRM reranking adds minimal latency. GenSelect adds ~30 sec. ThinkPRM may conflict
with the time budget on hard problems that already take full 30 min.

### Implementation priority:
1. **Qwen-PRM-7B in 4-bit** (proven +1.4-8.5% on hard math, reliable signal, fast scoring)
2. **GenSelect** ONLY if solver has GenSelect training; otherwise weak signal
3. **ThinkPRM** only if problems have spare time budget (easier problems)

---

## 8. Risks and Pitfalls

1. **PRM reward hacking at large N**: Don't go to N=64+ with PRM alone. At N=8
   it's well-behaved.
2. **Step format mismatch**: PRMs are sensitive to how steps are delimited.
   The solver must produce clearly delimited steps (double newlines or "\n\nStep X:").
   TIR solutions mix code+text which may confuse step scoring.
3. **OOM risk**: Loading PRM in fp16 alongside solver may cause OOM. Always use
   4-bit quantization for the PRM.
4. **Scoring TIR solutions** [POTENTIAL BLOCKER]: PRM trained on pure CoT math will
   likely score TIR solutions (which interleave Python code blocks with reasoning)
   poorly or incorrectly. If most of your 8 attempts are TIR, PRM reranking may be
   ineffective. Mitigation: strip code blocks before scoring (see Section 5 code),
   or run a mix of CoT and TIR attempts and only apply PRM to CoT ones.
5. **Wrong PRM aggregation**: Average of step scores is worse than min or product.
   Use product or min.
6. **vLLM sleep mode**: If using vLLM for solver, use sleep mode to temporarily
   free KV cache before loading PRM. (See: https://blog.vllm.ai/2025/10/26/sleep-mode.html)

---

## 9. Key Papers

| Paper | arXiv | Key contribution |
|---|---|---|
| "Let's Verify Step by Step" (OpenAI) | 2305.20050 | Original PRM paper |
| Math-Shepherd | 2312.08935 | Automated step labels via MC |
| "The Lessons of Developing PRMs" (Qwen) | 2501.07301 | Best practices + Qwen-PRM-7B |
| "Scaling Flaws of Verifier-Guided Search" | 2502.00271 | PRM fails at large N |
| "Process Reward Models That Think" (ThinkPRM) | 2504.16828 | Long-CoT verifier |
| "R-PRM: Reasoning-Driven PRM" | 2503.21295 | +8.5pts across 6 datasets |
| "Scaling LLM Test-Time Compute Optimally" | 2408.03314 | Compute-optimal BoN |
| "Learning Generative Selection for BoN" (GenSelect) | 2602.02143 | NVIDIA GenSelect |

---

## 10. Quick Decision Guide

**Step 1: Check solver for GenSelect training**
- If OpenMath-Nemotron-14B-Kaggle has GenSelect fine-tuning: implement GenSelect first,
  it's free (no extra VRAM) and proven +14% on hard problems with that model.
- If solver has no GenSelect training: skip GenSelect, LLM-as-judge is unreliable.

**Step 2: Load Qwen2.5-Math-PRM-7B in 4-bit**
- Set solver `gpu_memory_utilization=0.82` to carve out ~14GB
- PRM at 4-bit takes ~7GB, leaving 7GB margin
- Use vLLM sleep mode so KV cache is released during PRM scoring
- Expected gain: **+1.4% to +8.5%** on hard olympiad problems

**Step 3: TIR compatibility test**
- Run PRM on 10 problems with known answers using both CoT and TIR attempts
- Verify that PRM correctly prefers the right solution
- If PRM consistently scores TIR solutions badly: apply code-stripping before scoring
  or restrict PRM to CoT-mode attempts only

**Expected overall gain from PRM reranking**: +1.4% to +8.5% depending on problem
difficulty. For AIMO3 (IMO-level), gains should be at the higher end since PRM
advantage grows with problem difficulty.

---

*Sources: arXiv papers listed above, HuggingFace model cards for Qwen2.5-Math-PRM-7B
and Skywork-o1-PRM, NVIDIA OpenMath-Nemotron technical report (arXiv:2504.16891),
GenSelect paper (arXiv:2602.02143)*
