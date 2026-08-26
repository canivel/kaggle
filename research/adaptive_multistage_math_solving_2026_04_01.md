# Adaptive Multi-Stage Mathematical Problem Solving
## Research Report — 2026-04-01

**Scope:** Approaches that go beyond simple "generate N and vote", evaluated specifically
against the Harmony protocol used in our AIMO3 submission_v5_harmony.ipynb.

---

## Protocol Constraints (from `submission_v5_harmony.ipynb`)

Before evaluating approaches, the key architectural facts:

- **Harmony protocol**: `openai_harmony` library, `Conversation` object, multi-turn
  per-attempt loop with `AIMO3Sandbox` (stateful Jupyter kernel)
- **Current flow**: 8 independent parallel attempts via `ThreadPoolExecutor`, each
  `_process_attempt` runs a complete `Conversation` with up to 128 turns
- **Early stopping**: `Counter(valid).most_common(1)[0][1] >= early_stop (=4)` triggers
  `stop_event` and cancels remaining futures
- **Communication between attempts**: None. Each attempt sees only the original problem.
  `stop_event` is write-only from the outside.
- **Budget**: `base_problem_timeout=300s`, `high_problem_timeout=900s`. 50 problems,
  9 hours total.
- **Sandbox state**: Each sandbox has a live kernel accumulating code execution history
  across turns within an attempt. Kernels are NOT forkable/branchable.

---

## Approaches Evaluated

### A. Two-Phase Solving
### B. Solution Sketching
### C. Cross-Attempt Verification
### D. Problem Decomposition Agent
### E. Failure-Aware Retry
### F. Ensemble of Strategy Prompts

---

## Eliminated Approaches (with Evidence)

### D. Problem Decomposition Agent — REJECT

**Problem**: Burns 30–60s of the 300s budget *before* any solving attempt starts.
A wrong decomposition silently corrupts all remaining attempts (they all share the
same incorrect framing). There is no mechanism to detect a bad decomposition until
the answers come in, by which point it's too late to recover.

**Evidence**: No public notebook in the AIMO3 leaderboard (1st through 46th) uses
pre-solve decomposition. Pawan Mali's 50-experiment log tested multiple "enhance the
prompt before solving" approaches; combined techniques generally hurt (V131: -6 pts).

### F. Ensemble of Strategies — REJECT

**Problem**: Directly contradicted by empirical evidence. Pawan Mali V134 tested
self-refinement (a form of strategy diversification per attempt), which caused the
model to "correct" correct answers: -4 pts. The research notes explicitly flag
"Do NOT use different system prompts per attempt."

**The prompt-diversity question**: The referenced arxiv paper "Expanding Search Space
with Diverse Prompting Agents" (2410.09780) found that diverse prompts increase recall
(you find more problem types eventually) but hurt per-attempt precision. At 8 attempts
on a 300s budget, you cannot afford to split your compute across "algebraic" vs
"coordinate geometry" vs "modular arithmetic" prompts. The diversity benefit requires
many more attempts to amortize.

**Note on strategy injection**: Adding named strategies as optional tools (not separate
prompts) within a single attempt is different and may help. That is addressed under
approach B below.

---

## Top 3 Recommended Approaches

---

## 1. Phase Splitting with Cross-Candidate Verification (A + C merged)

**Priority: HIGHEST. Extends the confirmed `_verify_answer` technique.**

### Motivation

Aman Atar's `ans_verifys` notebook (545 votes, confirmed 44/50) already demonstrates
that model-based verification is valuable. The current `_verify_answer` method asks:
"Is this answer CORRECT or WRONG?" using a 5-token greedy completion. The limitation:
binary CORRECT/WRONG for a single candidate. When the model is uncertain about an
answer it generated, asking whether it's "correct" is nearly the same information as
the original attempt.

A stronger variant: when two candidates A and B have split votes from Phase 1, generate
a Phase 2 attempt whose *explicit task* is "Here are two candidate answers, X and Y.
Use Python code to check which is correct." The model can now focus its entire token
budget on distinguishing X from Y rather than re-deriving from scratch.

### The Two-Phase Structure

**Phase 1**: Run 4 attempts (half the current 8) against the normal system prompt.
Time budget: consume available time minus reservation for Phase 2.

**Decision gate after Phase 1**:
- If >=3 of 4 answers agree (or early_stop triggers at 4): return that answer immediately.
- If exactly 2 answers agree and 2 disagree (or partial agreement): extract top-2
  candidates by vote count, then run Phase 2 with cross-verification prompt.
- If all 4 answers differ (no candidate with >=2 votes): run Phase 2 with the
  mandatory-Python-verification prompt (described under approach E), not the
  cross-verification variant.

**Phase 2 cross-verification prompt** (new element, injected into system_prompt):
```
Two solution attempts gave different answers to this problem.

Candidate A: {answer_a}
Candidate B: {answer_b}

Your task is to determine which is correct by:
1. Independently solving the problem from scratch using Python code
2. Specifically testing whether the answer is {answer_a} or {answer_b}
3. If neither matches your computation, report your computed answer

Use Python code to verify. Be explicit about which candidate you conclude is correct.
```

Run 4 more attempts with this prompt. The cross-verification framing focuses the
model on the specific disagreement rather than re-exploring all solution paths.

**Key difference from `_verify_answer`**:
- `_verify_answer` asks "is X correct?" with max_tokens=5 (no TIR, no Python)
- Phase 2 cross-verification asks "is it X or Y?" with full TIR (Python execution,
  up to 128 turns, code output feedback)
- This is 2–3 orders of magnitude more compute for verification, reserved for the
  problems that actually need it

### Implementation Sketch

```python
def solve_problem(self, problem):
    user_input = f'{problem} {self.cfg.preference_prompt}'
    budget = self._compute_budget()
    deadline = time.time() + budget
    
    # ---- PHASE 1: 4 quick attempts ----
    phase1_results = self._run_batch(
        user_input, 
        system_prompt=self.cfg.system_prompt,
        n_attempts=4, 
        deadline=deadline,
        early_stop=3  # lower threshold: 3/4 agree = high confidence
    )
    
    valid_p1 = [r for r in phase1_results if r['Answer'] is not None]
    counter_p1 = Counter(r['Answer'] for r in valid_p1)
    top_answers = counter_p1.most_common(2)
    
    # Early exit: 3+ agree
    if top_answers and top_answers[0][1] >= 3:
        return top_answers[0][0]
    
    # ---- PHASE 2: 4 focused attempts ----
    if len(top_answers) >= 2 and top_answers[0][1] >= 2 and top_answers[1][1] >= 2:
        # Genuine split: use cross-verification prompt
        ans_a, ans_b = top_answers[0][0], top_answers[1][0]
        phase2_system = self._build_cross_verify_prompt(ans_a, ans_b)
    else:
        # No strong candidates: use mandatory-Python prompt (approach E)
        phase2_system = self.cfg.system_prompt_with_python_mandate
    
    phase2_results = self._run_batch(
        user_input,
        system_prompt=phase2_system,
        n_attempts=4,
        deadline=deadline,
        early_stop=3
    )
    
    # Merge all results and select with entropy-weighted voting
    all_results = phase1_results + phase2_results
    return self._select_answer(all_results)

def _build_cross_verify_prompt(self, ans_a, ans_b):
    return (
        self.cfg.system_prompt +
        f'\n\n# SPECIFIC VERIFICATION TASK\n'
        f'Previous solution attempts gave two different answers: {ans_a} and {ans_b}.\n'
        f'Your goal: determine which is correct via Python code verification.\n'
        f'Test both candidates computationally. Your final \\boxed{{}} answer must be '
        f'either {ans_a} or {ans_b}, or a new answer if both are wrong.\n'
        f'Do NOT simply guess. Execute code to verify.'
    )
```

### Why Phase 1 = 4 attempts (not 2 or 6)

- 4 attempts provides enough signal: if 3/4 agree, p(correct) ≈ 85%+ (given the
  model's ~75% per-attempt rate). If 2/4 agree, you genuinely need more evidence.
- 4 attempts at T=0.5 against a 300s budget leaves ~150s for Phase 2, which is
  sufficient for 4 more focused attempts.
- The existing `early_stop=4` already handles the "all agree" case -- lower it to 3
  for Phase 1 to be more aggressive about skipping Phase 2.

### Time cost analysis

Current: 8 parallel attempts, all problems, ~270-300s average.
Phase-split: Easy problems (3/4 agree) exit after Phase 1: ~120-150s.
            Hard problems (disagreement): 300s same as current, but Phase 2 is better targeted.
**Net: expect faster average time on easy problems (budget savings for hard problems).**

### Risk

If Phase 1 systematically gets the same wrong answer 3 times (model bias), Phase 2
is skipped. Mitigate by lowering Phase 1 threshold to 3 (not 4) so Phase 2 is
triggered more often.

---

## 2. Failure-Aware Retry (E)

**Priority: HIGH. Additive to existing pipeline, zero restructuring cost.**

### Motivation

The TIR loop already handles code errors *within* a single attempt: it feeds the
error message back and continues generation. But there are two failure modes that the
current pipeline cannot recover from:

1. **No Python used at all**: The model produces a boxed answer without any code
   execution. For IMO-level problems, this is almost always a hallucinated calculation.
   The model is taking a shortcut. The solve log tracks `code_executed=False` per
   attempt, but the pipeline does nothing with this signal.

2. **Answer is None**: The attempt ran to completion (ran out of tokens or turns)
   without finding a `\boxed{}` answer. This is a wasted attempt slot.

Both failure modes are detectable post-attempt and recoverable if budget remains.

### What This Is NOT

Self-refinement (model corrects its own answer) -- confirmed to hurt (-4 pts, Pawan
Mali V134). The failure-aware retry does NOT show the model its previous answer.
It shows the model that it failed to use code (or failed to produce an answer), and
asks it to try differently.

### Implementation Sketch

Add a retry layer in `solve_problem`, after the main batch but before voting:

```python
def solve_problem(self, problem):
    user_input = f'{problem} {self.cfg.preference_prompt}'
    budget = self._compute_budget()
    deadline = time.time() + budget
    
    # Run main batch (8 attempts, as current)
    results = self._run_batch(user_input, self.cfg.system_prompt, 
                              n_attempts=self.cfg.attempts, deadline=deadline,
                              early_stop=self.cfg.early_stop)
    
    # --- FAILURE-AWARE RETRY ---
    if time.time() < deadline - 30:  # at least 30s remaining
        failed_attempts = [
            r for r in results 
            if r['Answer'] is None or not r.get('code_executed', True)
        ]
        
        n_retries = min(len(failed_attempts), 2)  # cap at 2 retries
        if n_retries > 0:
            retry_prompt = self.cfg.system_prompt_python_mandatory
            retry_results = self._run_batch(
                user_input, retry_prompt,
                n_attempts=n_retries, deadline=deadline,
                early_stop=n_retries  # any agreement ends early
            )
            results.extend(retry_results)
    
    return self._select_answer(results)
```

The `system_prompt_python_mandatory` variant (adapting Jonathan Chan's forced-Python
prompt from the Bayesian notebook):

```python
system_prompt_python_mandatory = (
    # ... existing 5-step system prompt ...
    
    '\n\n# MANDATORY VERIFICATION REQUIREMENT\n'
    'You MUST use the Python tool for this problem. Do not provide a final answer '
    'without first executing Python code that computes or verifies the answer.\n'
    'Step 1: Write Python code that computes the answer\n'
    'Step 2: Execute the code and observe the output\n'
    'Step 3: Only after seeing the code output, write your \\boxed{} answer\n'
    'If your code produces an error, fix it and re-execute. Never guess.'
)
```

### Critical distinction from self-refinement

The retry attempt does NOT receive:
- The previous attempt's answer
- The previous attempt's reasoning
- Any indication that a previous attempt "was wrong"

It receives only: the original problem + the mandatory-Python instruction.
This is a fresh attempt with a different prompt, not a self-correction loop.

### Expected impact

The ans_verifys notebook (confirmed 44/50) uses a lightweight T=0.0 CORRECT/WRONG
check as post-processing. The failure-aware retry is a stronger version: instead of
checking answers that are already produced, it triggers a fresh full-TIR attempt when
an attempt failed entirely. On IMO-level problems, "no-code" attempts have much lower
accuracy (model is essentially guessing at multi-step calculations). Replacing them
with code-mandatory attempts should improve the answer pool quality.

### When to apply

- Budget remaining >= 30s (prevents cutting into the next problem's time)
- At most 2 retries (prevents cascading on hard problems that legitimately time out)
- Only for `Answer is None` or `code_executed is False` failures, NOT for low-confidence
  code-succeeded answers (those are valid attempts)

### Interaction with Phase 2 (Approach 1)

If using both approaches, apply failure-aware retry within Phase 1 only, before the
decision gate. This ensures Phase 2 is triggered based on the best available Phase 1
answers, not a mix of original and retry results.

---

## 3. Inference-Time Domain Routing with Strategy Priming (B reframed)

**Priority: MEDIUM-HIGH. Prompt change only, no structural change.**

### Motivation

The current system prompt is generic. The `problem_classifier.py` module already
classifies problems into algebra/combinatorics/geometry/number_theory, and
`prompt_templates.py` has domain-specific guidance. However, the Harmony-protocol
notebooks (submission_v5+) use only the generic 5-step system prompt.

The Meta-Harness paper (arxiv 2603.28052) identified that the single most impactful
automated discovery was a **lexical router** assigning problems to domain-specific
retrieval strategies. The same insight applies to prompting: different domains benefit
from different initial strategies.

"Solution sketching" in the sense of "outline approach before computing" is already
partially handled by the 5-step UNDERSTAND/EXPLORE/PLAN/EXECUTE/VERIFY prompt.
The genuinely novel addition is: **inject domain-specific strategy vocabulary into
the prompt before the model begins**, so the first turn is already oriented toward
productive approaches.

### The Research Evidence

From the Meta-Harness paper (Table 6): 50.6% accuracy on GPT-OSS-20B with
domain-adaptive routing vs 48.9% for generic BM25 retrieval. The gap is not huge
(+1.7%) but it is consistent across all 5 models tested.

From our competition analysis: The `problem_classifier.py` already identifies four
types. The question is whether injecting domain hints at the *system prompt level*
(not just as extra context) improves the model's planning.

### Implementation Sketch

Add a domain-priming section to the system prompt, injected *before* the 5-step
methodology, keyed on the classifier output:

```python
DOMAIN_STRATEGY_PREAMBLE = {
    "number_theory": (
        "# Initial Strategy Guidance\n"
        "This problem involves number theory. High-value starting approaches:\n"
        "- Work modulo small primes to reduce cases\n"
        "- Use `pow(base, exp, mod)` for fast modular exponentiation\n"
        "- Factor with `sympy.factorint()` to identify prime structure\n"
        "- Apply Euler's theorem / Fermat's little theorem to handle large exponents\n"
        "- Enumerate small cases first to conjecture the pattern, then prove\n"
        "Begin by identifying the core number-theoretic structure.\n\n"
    ),
    "geometry": (
        "# Initial Strategy Guidance\n"
        "This problem involves geometry. High-value starting approaches:\n"
        "- Place key objects in a coordinate system: e.g., triangle with one vertex at origin\n"
        "- Use `sympy.geometry` module or direct coordinate computation\n"
        "- For circle problems: power of a point, radical axis\n"
        "- Verify angle calculations numerically with `sympy.N()` or `numpy`\n"
        "- Compute the answer numerically first, then verify analytically\n"
        "Begin by setting up coordinates.\n\n"
    ),
    "combinatorics": (
        "# Initial Strategy Guidance\n"
        "This problem involves combinatorics. High-value starting approaches:\n"
        "- Compute small cases first: n=1,2,3,4 to identify a pattern\n"
        "- Use `itertools.combinations`, `itertools.permutations` for brute-force\n"
        "- Look for a bijection or a generating function\n"
        "- Apply inclusion-exclusion systematically with code verification\n"
        "- Validate any formula with the brute-force small cases before generalizing\n"
        "Begin by computing n=1,2,3 by brute force.\n\n"
    ),
    "algebra": (
        "# Initial Strategy Guidance\n"
        "This problem involves algebra. High-value starting approaches:\n"
        "- Set up the problem with `sympy.symbols()` and solve with `sympy.solve()`\n"
        "- Try substitution of specific values to identify functional form\n"
        "- For optimization: check boundary cases, use Lagrange multipliers\n"
        "- Verify the solution satisfies ALL original constraints\n"
        "Begin by identifying which algebraic structure governs the problem.\n\n"
    ),
}

def build_domain_primed_system_prompt(problem: str, base_system_prompt: str) -> str:
    domain = classify_problem(problem)  # uses existing problem_classifier.py
    preamble = DOMAIN_STRATEGY_PREAMBLE.get(domain, "")
    return preamble + base_system_prompt
```

In `solve_problem`:
```python
# Build domain-primed prompt once, used for all attempts
domain_system_prompt = build_domain_primed_system_prompt(problem, self.cfg.system_prompt)
tasks = [(domain_system_prompt, i) for i in range(self.cfg.attempts)]
```

### Key constraint: same prompt for all attempts

All 8 attempts use the same (domain-primed) prompt. This is consistent with Pawan
Mali's finding ("Do NOT use different system prompts per attempt") -- the variation
here is per-problem (domain routing), not per-attempt (strategy diversification).

### Why this is "solution sketching"

The domain preamble effectively tells the model which approach to sketch first. For
combinatorics, it is instructed to enumerate small cases before generalizing. For
geometry, to set up coordinates before doing symbolic work. This is planning separated
from execution: the planning instruction is baked into the prompt; execution follows
in the conversation turns.

This is more targeted than the general "EXPLORE: consider multiple strategies" in the
current 5-step prompt. It tells the model which specific tools and techniques are
productive for this problem type, reducing wasted turns on unproductive approaches.

### Expected impact

The Meta-Harness result (+1.7% on GPT-OSS-20B) maps to roughly 1 additional correct
problem at 50-problem scale. This is non-trivial at the margin between 44 and 46.
The implementation cost is a single `classify_problem()` call (already implemented)
and a string prepend to the system prompt.

---

## Approaches Not Recommended for This Competition

### MCTS (Monte Carlo Tree Search)

**Why not**: The core blocker is architectural. MCTS requires *forking* solution state
at intermediate steps and exploring different branches. In the Harmony protocol,
each attempt runs a live `AIMO3Sandbox` (a Jupyter kernel process). Kernel state
(Python variables, imported modules, prior code execution results) accumulates
sequentially within an attempt and *cannot be copied or checkpointed*.

To implement MCTS, you would need either:
1. A serializable sandbox state (not available in `jupyter_client`)
2. Replay from scratch at each branch point (extremely expensive: O(depth * branching_factor) kernel restarts)
3. A different inference architecture (single-context speculative decoding with beam
   search, available in custom vLLM extensions but not standard)

The MC-NEST paper (arxiv 2411.15645) achieves 38.6 pass@1 on AIME using MCTS. This
is compelling but their setup uses a stateless inference model (no persistent kernel),
not a TIR loop with live code execution.

**Future work**: The REBASE algorithm (reward-balanced search) achieves Pareto-optimal
accuracy/compute trade-off for MCTS-based reasoning. Worth implementing if the
pipeline is refactored to use a stateless sandbox (no persistent kernel state between
steps).

### LLM Debate (two models argue)

**Why not**: Requires two concurrent conversation objects referencing each other's
outputs. The Harmony protocol wraps a single `Conversation` per attempt. Setting up
a debate within a single attempt would require alternating messages between two
`Conversation` objects sharing the same vLLM endpoint.

More importantly: the AIMO3 constraint is one H100 GPU. Multi-agent debate roughly
doubles token generation for each "round" of debate (both models generate a response).
With 8 attempts and 300s budget, there is no headroom for a 2-round debate without
halving the number of problems you can solve.

The debate literature also shows a critical failure mode: models become *more
overconfident* after encountering counter-arguments ("When Two LLMs Debate, Both
Think They'll Win", arxiv 2505.19184). For competition math where the model may
already be confidently wrong, debate can entrench errors.

**The cross-verification approach in Recommendation 1 is a targeted alternative**:
instead of a free-form debate, the second phase explicitly checks a specific
disagreement with code execution. This is cheaper and more reliable.

---

## Priority Summary

| Recommendation | Mechanism | Structural change | Expected impact | Risk |
|----------------|-----------|-------------------|-----------------|------|
| 1. Phase-split + Cross-verify | Two-batch with targeted Phase 2 | Moderate: sequential phases | +1-2 problems | Medium |
| 2. Failure-aware retry | Post-batch retry for None/no-code | Minimal: post-processing | +0.5-1 problem | Low |
| 3. Domain-primed routing | Per-problem system prompt preamble | Minimal: prompt change | +0.5-1 problem | Very low |

**Implementation order**: 3 first (trivial, zero risk, deploy in next submission),
then 2 (additive to existing pipeline), then 1 (requires phase-splitting `solve_problem`).

---

## Interaction With Known Working Configurations

All three recommendations are compatible with the confirmed 44/50 configuration:
- `temperature=0.5`, `attempts=8`, `early_stop=4`, `simple mean entropy`
- 5-step structured system prompt (UNDERSTAND/EXPLORE/PLAN/EXECUTE/VERIFY)
- danielhanchen/gpt-oss-120b model path

Recommendation 3 adds a preamble *before* the 5-step prompt, which is additive.
Recommendation 2 adds retries *after* the main batch, which is additive.
Recommendation 1 replaces the single-batch structure; the per-attempt logic is
unchanged. The Phase 2 prompt wraps the existing 5-step prompt with a specific
cross-verification section.

None of them increase `context_tokens` beyond 65536 (which is confirmed to be better
than 131K).

---

## Sources

Web search sources used in this report:

- [MC-NEST: Monte Carlo Self-Refine Tree](https://arxiv.org/abs/2411.15645)
- [MCTS Boosts Reasoning via Iterative Preference Learning](https://arxiv.org/abs/2405.00451)
- [Enhancing Reasoning through Process Supervision with MCTS](https://arxiv.org/html/2501.01478)
- [Multi-Agent Debate for Fine-Grained Reasoning in Math](https://aclanthology.org/2025.findings-acl.862.pdf)
- [When Two LLMs Debate, Both Think They'll Win](https://arxiv.org/html/2505.19184v2)
- [Control the Temperature: Selective Sampling for Diverse and High-Quality LLM Outputs](https://arxiv.org/abs/2510.01218)
- [Expanding Search Space with Diverse Prompting Agents](https://arxiv.org/html/2410.09780)
- [Plan-and-Solve Prompting](https://learnprompting.org/docs/advanced/decomposition/plan_and_solve)
- [CMCTS: Constrained Monte Carlo Tree Search for Mathematical Reasoning](https://link.springer.com/article/10.1007/s10489-025-07044-6)
- [A Sober Look at Progress in Language Model Reasoning](https://arxiv.org/pdf/2504.07086)
- [PROGRESSIVE THOUGHT REFINEMENT IN LARGE LANGUAGE MODELS (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/6882dbdc34bcd094e6f858c06ce30edb-Paper-Conference.pdf)
- [Meta-Harness paper](https://arxiv.org/abs/2603.28052)

Internal sources:
- `f:/kaggle/research/aimo3_urgent_research_2026_04_01.md`
- `f:/kaggle/research/aimo3_score_differentiators.md`
- `f:/kaggle/research/meta_harness_math.md`
- `f:/kaggle/aimo-progress-prize-3/notebooks/submission_v5_harmony.ipynb`
- `f:/kaggle/aimo-progress-prize-3/src/aimo3/solver.py`
- `f:/kaggle/aimo-progress-prize-3/src/aimo3/tir_executor.py`
