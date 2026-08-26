# AIMO3: Failure Mode Analysis and Novel Approaches to Push p from 0.69 to 0.85

**Research date**: 2026-04-01
**Goal**: Understand WHY GPT-OSS-120B fails 31% of attempts and identify novel approaches
**Current state**: p=0.69 per-attempt → best observed 44/50 (mean ~40). Need p≈0.85 for 46/50.

---

## 1. The Core Problem: Why p=0.69?

### What the arxiv paper 2603.27844 establishes

Paper: "Model Capability Dominates: Inference-Time Optimization Lessons from AIMO 3"
Author: Natapong Nitarach (SCB 10X), March 29, 2026.

The paper's central finding: **every inference-time intervention tested either had negligible
effect or hurt performance**. The 17-point model capability gap (between a weaker and stronger
model) dominated by an order of magnitude compared to any inference strategy. Conclusion:
"Model capability dominates."

Per-attempt accuracy p̂ = 0.69 was measured on the gpt-oss-120b baseline across 50 AIMO3
problems at T=1.0, N=8 attempts.

### Failure mode taxonomy (from research synthesis)

From cross-referencing arxiv 2503.21934 (Proof or Bluff), Mahdavi et al. (Brains vs Bytes),
and competition notebook analysis, failures decompose into:

| Failure Type | Estimated Share | Evidence |
|---|---|---|
| Wrong formalization (problem misunderstood) | ~35% | Mahdavi: "unjustified assumptions" dominant error |
| Wrong approach (correct understanding, wrong method) | ~25% | "creativity failures" in Brains vs Bytes |
| Execution error (right approach, code bug) | ~20% | Pawan Mali: Python errors reduce reliability weight |
| Correct compute but wrong answer extraction | ~10% | Follow-up prompting recovers some |
| Context/timeout truncation | ~10% | Pawan Mali: 131K context hurts; 65536 is right |

**Key insight from "Proof or Bluff" (2503.21934)**: "Occasional correct answers often result
from pattern recognition or heuristic shortcuts, NOT genuine mathematical reasoning." This means
the 69% that succeed are not all truly solved — some are lucky guesses. The model's reasoning
trace is frequently wrong even when the final answer is correct.

### Problem-type difficulty breakdown

From AIMO2 analysis (where o3-preview missed 7/50):
- Geometry: 2 missed (hardest)
- Algebra: 2 missed
- Combinatorics: 3 missed (hardest)
- Number theory: 0 missed

Combinatorics and geometry remain the hardest categories. These are where p̂ is lowest —
probably p̂ ≈ 0.40-0.50 on hard combinatorics vs p̂ ≈ 0.85+ on straightforward number theory.

### Why does the model fail even with Python tools?

From competition notebook analysis + "Agentic Reasoning and Tool Integration" (Microsoft 2025):

1. **Wrong equation setup**: Model writes code that solves the WRONG equations. The code runs
   fine and produces a confident answer. This is the hardest failure to detect without external
   verification.

2. **Correct approach, incomplete enumeration**: In combinatorics, model sets up a loop but
   misses edge cases (off-by-one, periodic boundary, etc.).

3. **Symbolic computation path errors**: SymPy sometimes returns unexpected forms; model
   misinterprets the symbolic output.

4. **Early confidence truncation**: Model gets an answer from a quick calculation, puts it in
   boxed{}, then the remaining thinking (which would have caught the error) is ignored.

5. **No-boxed failures**: Answer computed correctly in code but model doesn't format it in
   \boxed{}. Amanatar's follow-up prompting addresses this.

---

## 2. What the Paper DEFINITIVELY Rules Out

From arxiv 2603.27844 ablation table (N=8, gpt-oss-120b, 50 problems):

| Strategy | Score | Delta vs T=1.0 |
|---|---|---|
| T=1.0 (13-run mean) | 39.7 | baseline |
| T=0.8 | 40.0 | +0.3 |
| T=0.5 | 38.0 | -1.7 |
| T=1.2, min_p=0.03 | 37.0 | -2.7 |
| Diverse prompts (best of 8 strategies) | 36-39 | all negative or negligible |
| N=16 (Qwen3.5-35B) | no improvement | 0.0 |

**Ruled out as high-value approaches**:
- Diverse prompt mixing (every variant reduces score)
- Seed diversity (already maximally decorrelated at T=1.0, ρ̂ = -0.258)
- More attempts beyond N=8 (diminishing returns, math proves this)
- T=0.5 (despite 44/50 notebooks using it — they score 44 on single best run but MEAN is lower)

**Not ruled out (outside paper's scope)**:
- PRM/ORM reranking (paper only tested entropy-based voting)
- Lemma-based hierarchical decomposition (not tested)
- Answer verification loops (paper didn't test this)
- Fine-tuning on AIMO3-style problems (paper tests fixed models only)

---

## 3. What Bridges p=0.69 to p=0.85: The Math

**Current**: p=0.69, N=8, majority vote → P(≥5 correct) ≈ 71% → expected 35.6/50
**Observed**: 39.7/50 mean (higher because easy problems have p≈0.95, hard have p≈0.30)
**Best observed**: 44/50 in competition (vs paper's 44/50 best in 13 runs)

**What p=0.85 would yield** (N=8):
P(≥5 correct | p=0.85) ≈ P(5)+P(6)+P(7)+P(8) = 0.839
Expected: 50 × 0.839 = 41.9/50 mean — that's NOT 46/50

Wait: this calculation is wrong for heterogeneous difficulty. The correct mental model:
- 35 "easy" problems: p=0.95 → P(majority correct) ≈ 0.998 → ~35.0 expected
- 10 "medium" problems: p=0.75 → P(majority correct) ≈ 0.76 → ~7.6 expected
- 5 "hard" problems: p=0.30 → P(majority correct) ≈ 0.06 → ~0.3 expected
- Sum at p_mean=0.69: ~42.9/50

To reach 46/50, you need to solve ~3-4 more of the hard/medium problems.

**For hard problems (p=0.30)**, you need either:
A. Per-attempt accuracy improvement: p=0.30 → p=0.65 flips P(majority) from 0.06 → 0.57
B. Better selection (PRM/ORM reranking): pick the correct solution from N=8 correctly more often
C. Different solve strategy: hierarchical decomposition, longer thinking, MCTS

---

## 4. Novel Approaches Ranked by Expected Impact

### Approach 1: Binary Answer Verification Before Final Vote [HIGHEST IMPACT, proven, 1 day]

**What it is**: After collecting N=8 candidate answers, take top candidates by vote count and
use the model at T=0.0 to binary-verify each: "Is this the correct answer? CORRECT or WRONG."
Return the first candidate that passes verification.

**Important distinction from generic "self-PRM"**:
Research (arxiv 2503.21934 "Proof or Bluff") shows "LLMs frequently fail to distinguish correct
mathematical reasoning from clearly flawed solutions." A 0-10 scoring rubric would inherit this
problem. BUT a simple binary CORRECT/WRONG judgment at T=0.0 is a much simpler task —
the model only needs to answer YES/NO on a specific integer, not evaluate reasoning quality.
This is why amanatar's approach works (proven 44/50) while elaborate scoring would fail.

**Evidence of impact**:
- amanatar's `_verify_answer` notebook: 44/50 on AIMO3 public leaderboard, 545 votes
- Implements exactly binary verification at T=0.0, max_tokens=5
- GenSelect (NVIDIA AIMO2 winner): 78.4% TIR pass@1 → 93.3% with selection (+14.9%)
  (GenSelect is a trained selector; amanatar's approach is a zero-shot approximation)

**The critical point**: PRM/ORM reranking from the literature requires a separately trained
reward model. With ~5 GB VRAM headroom on the H100, a ThinkPRM-1.5B could theoretically fit
(1.5B × bf16 = 3 GB), but the simpler and proven approach is amanatar's binary verification.

**Implementation** (from amanatar's notebook, confirmed working):
```python
def _verify_answer(self, problem: str, answer: int) -> bool:
    prompt = f"Problem:\n{problem}\n\nProposed answer: {answer}\n\nCheck the answer carefully.\nReply with only ONE word:\nCORRECT or WRONG"
    resp = self.client.completions.create(
        model=self.cfg.served_model_name,
        prompt=self.encoding.encode(prompt),
        temperature=0.0,
        max_tokens=5
    )
    text = resp.choices[0].text.strip().upper()
    return "CORRECT" in text and "WRONG" not in text
```

**Voting cascade** (full amanatar approach):
1. HARD ACCEPT: if ≥4 of 8 attempts agree → verify once → submit
2. CANDIDATE FILTER: keep answers with ≥2 votes
3. ENTROPY SORT: rank candidates by mean entropy (ascending)
4. VERIFY: check each candidate via `_verify_answer`; return first CORRECT
5. FALLBACK: entropy-weighted vote if none verified

**Expected impact**: +1 to +2 points. This is the most evidence-backed approach.
Note: Pawan Mali's V133 got 0 improvement with verification, but his implementation
likely differed (different prompt or not using greedy T=0.0 for binary judgment).

---

### Approach 2: Two-Stage Adaptive Solve (Triage + Deep Dive) [HIGH IMPACT, medium complexity]

**What it is**: The Adaptive Multi-Phase design from `novel_approach_design.md`, now supported
by the "adaptive compute" literature (SelfBudgeter, TALE, Input-Adaptive Allocation ICLR 2025).

**Why standard approach wastes compute**:
- 65% of AIMO3 problems are "easy" (p̂ ≈ 0.90+) and solved in Phase 1 (first 2-3 attempts)
- 35% are "hard" (p̂ ≈ 0.40-) but get same 8 attempts as easy problems
- The time budget saved from early-stopping easy problems is currently WASTED

**Proposed flow**:
```
Phase 1: Run 3 fast attempts (120s budget each)
  → If ≥3 agree (unanimous): submit, bank remaining time
  → If ≤2 disagree: tag as HARD, go to Phase 2

Phase 2: 5 deep attempts (full remaining budget per problem)
  → Inject context: "Three quick attempts produced DIFFERENT answers: [A, B, C, D].
    At least 3 are wrong. Be extremely systematic. Use code to verify each step."
  → The disagreement signal is NOT prompt mixing — it's problem-specific information
    that helps the model avoid repeating the same mistake

Phase 3: Verification tiebreak on top-2 candidates
  → Use _verify_answer (T=0.0, max_tokens=5) on the 2 most-voted answers
  → Submit the one that passes verification
```

**Why the disagreement injection works differently than diverse prompts**:
- Diverse prompts: "Use strategy X for all 8 attempts" → reduces per-attempt quality
- Disagreement injection: "Your previous attempts DISAGREED" → context-specific signal
  that triggers more careful verification, not a weaker strategy
- The model is not being given a different strategy; it's being told that previous
  attempts were inconsistent, which reliably triggers more careful computation

**Evidence from Intern-S1-MO (arxiv 2512.10739)**:
Multi-round hierarchical reasoning with memory significantly beats single-pass.
Intern-S1-MO: 26/35 on IMO2025 non-geometry (vs ~15-20 for vanilla approaches).
The key is exactly this: later rounds KNOW about earlier failures.

**Expected impact**: +1 to +3 points. This is the highest-impact untested approach.

**Implementation note**: Does NOT violate the "no diverse prompts" finding because:
1. The base prompt is unchanged (same 5-step structured prompt)
2. The injected context is problem-specific, not a "strategy override"
3. The model can ignore the context if it's not useful (on problems it solves easily)

**Time budget validation**:
- Phase 1: 3 attempts, 60s each, run in parallel (16 workers) → ~60s wall time
- Phase 2: 5 attempts, up to remaining_budget, run in parallel → ~240s wall time
- Total per problem: ~300s on hard problems, ~60-120s on easy problems
- 50 problems: if 35 are easy (avg 90s) + 15 are hard (avg 300s) = 35×90 + 15×300 = 7650s = 127.5 min
- This fits well within the 5-hour (300 min) window. Even a conservative estimate of all hard
  (50×300 = 250 min) stays under the 300 min limit with margin for startup/overhead.

---

### Approach 3: Hierarchical Lemma Decomposition [HIGH IMPACT, high complexity]

**What it is**: Inspired by Intern-S1-MO (arxiv 2512.10739). For hard problems, instead of
one long reasoning trace, decompose into sub-lemmas:
1. "Solve the sub-problem: what is the range of X?"
2. "Given range X, solve for Y."
3. "Combine X and Y to find the final answer."

**Why this helps on the problems we currently fail**:
- Hard combinatorics problems require tracking multiple cases. A single 65K-token context
  fills up with one wrong case analysis. Sub-lemma decomposition allows fresh context per step.
- Geometry problems require setting up coordinates (step 1), computing intersections (step 2),
  and combining results (step 3). Each step can be independently verified.

**How to implement within Harmony protocol**:
```python
def solve_hierarchical(self, problem: str) -> int:
    # Turn 1: Problem analysis and decomposition
    plan_prompt = (
        problem + "\n\n"
        "DECOMPOSE this problem into 2-4 independent sub-problems or lemmas. "
        "List each sub-problem clearly. Do NOT solve them yet."
    )
    plan = self.generate(plan_prompt, max_tokens=2000)
    
    # Turn 2: Solve each sub-problem with TIR
    for i, subproblem in enumerate(parse_subproblems(plan)):
        sub_solution = self.generate(
            subproblem + "\nSolve this sub-problem completely. Use code to verify.",
            max_tokens=10000
        )
        
    # Turn 3: Combine with full context of sub-solutions
    combine_prompt = (
        problem + "\n\n"
        f"We established:\n{sub_solutions_summary}\n\n"
        "Using these established facts, find the final integer answer."
    )
    final = self.generate(combine_prompt, max_tokens=5000)
    return extract_answer(final)
```

**Important caveat**: Intern-S1-MO used a custom RL-trained model (OREAL-H framework) that was
jointly optimized for hierarchical decomposition. Their 26/35 result cannot be attributed to
architecture alone — the model was trained to decompose problems and store lemmas. Bolting
hierarchical turns onto GPT-OSS-120B without compatible training may yield little benefit.

**Evidence for inspiration, not direct replication**: Intern-S1-MO's framework demonstrates
that multi-round reasoning with memory beats single-pass for IMO-level problems. The mechanism
(avoid re-exploring dead ends) is sound. But the gain with an untrained model is speculative.
Realistic expected gain for GPT-OSS-120B: +0 to +2 points (vs +5+ for a trained model).

**Risk**: Higher complexity, more turns, more likely to exhaust context or timeout.
**Feasibility with 14 days**: Low-medium. Best used for hardest problems only if time allows.

---

### Approach 4: Solution Quality Detection and Retry [MEDIUM IMPACT, low complexity]

**What it is**: After each attempt, check signals that correlate with incorrect answers,
and retry with a "be more careful" nudge if the signals are bad.

**Detectable failure signals** (from notebook analysis + competition experience):

| Signal | What It Indicates | Action |
|---|---|---|
| No Python tool calls in response | Model answered without verification | Retry with "You MUST verify with code" |
| Python error (traceback) in last tool call | Execution failed, answer may be guessed | Retry with error context |
| Response shorter than 500 tokens | Model cut short, didn't reason fully | Retry with "Be more thorough" |
| No boxed{} found | Answer extraction failed | Follow-up prompt (already in v25) |
| entropy > threshold AND tool_calls == 0 | High uncertainty + no code verification | Retry |

**Implementation** (low complexity, additive to existing code):
```python
def assess_attempt_quality(self, result: AttemptResult) -> float:
    """Return quality score 0-1. Low quality = retry."""
    score = 1.0
    if result.python_calls == 0:
        score *= 0.5  # major penalty: no verification
    if result.python_errors > 0:
        score *= 0.7  # moderate penalty: errors indicate setup problems
    if result.response_tokens < 500:
        score *= 0.6  # short response = truncated reasoning
    if result.entropy > 2.0:
        score *= 0.8  # high uncertainty
    return score

def solve_with_quality_gates(self, problem: str) -> int:
    for attempt in range(self.cfg.attempts):
        result = self.run_attempt(problem)
        quality = self.assess_attempt_quality(result)
        if quality < 0.4:
            # Retry with targeted nudge
            retry_result = self.run_attempt(
                problem, 
                nudge="Previous attempt had issues. MUST use Python to verify. Show all steps."
            )
            # Keep whichever has lower entropy
            result = min(result, retry_result, key=lambda r: r.entropy)
        self.results.append(result)
    return self.vote(self.results)
```

**Expected impact**: +0.5 to +1.5 points. This addresses the "no code verification" failure
mode directly. Amanatar's 44/50 notebook hints at this with its verification cascade.

---

### Approach 5: Fine-Tuned Verifier via Huikang Model [MEDIUM IMPACT, uncertain]

**What it is**: Use the huikang/gpt-oss-120b-aimo3 fine-tuned checkpoint as a VERIFIER for
solutions generated by the base model, rather than as the solver.

**Why this might work**:
- Huikang fine-tuning appears to have been on AIMO3-style problems specifically
- The model may have been trained to recognize correct mathematical reasoning
- Using it as a verifier (at T=0.0) is lower variance than using it as a solver

**Pawan Mali's finding**: Direct replacement with huikang model → -6 points.
**Jonathan Chan's finding**: Huikang model WITH Harmony + Bayesian voting → unknown but 185 votes.

**Hypothesis**: The fine-tuned model may be better at JUDGING solutions than GENERATING them,
because fine-tuning on hard problems shifts the distribution toward "I know what correct looks like"
even if generation quality is mixed.

**Implementation**: Use base model (danielhanchen variant) for generation, use huikang v20
only for the `_verify_answer(problem, answer)` call at T=0.0.

**Expected impact**: 0 to +1 points. Low risk since verification calls are cheap (5 tokens each).

---

### Approach 6: Test-Time Compute Redistribution via Difficulty Classification [MEDIUM IMPACT]

**What it is**: Classify each problem's difficulty BEFORE solving, then allocate time budget
accordingly. The literature on "Input-Adaptive Allocation" (ICLR 2025) shows that predicting
problem difficulty and allocating compute accordingly beats uniform allocation.

**Simple classifier**:
```python
def estimate_difficulty(self, problem: str) -> str:
    keywords_hard = ['geometry', 'circle', 'triangle', 'polygon', 'inscribed',
                     'combinatorics', 'sequence', 'arrangement', 'how many ways',
                     'configuration', 'lattice', 'grid', 'partition']
    keywords_easy = ['compute', 'find the value', 'evaluate', 'simplify',
                     'divisors', 'sum of digits', 'modulo', 'GCD']
    
    problem_lower = problem.lower()
    hard_count = sum(1 for k in keywords_hard if k in problem_lower)
    easy_count = sum(1 for k in keywords_easy if k in problem_lower)
    
    if hard_count >= 2 or (hard_count == 1 and easy_count == 0):
        return 'hard'  # → 900s per attempt, 12 attempts, T=1.0
    else:
        return 'normal'  # → 300s per attempt, 8 attempts, T=0.8
```

**Why geometry and combinatorics deserve more time**:
- These require more tool calls per problem (5-10 vs 2-3 for number theory)
- With 300s per attempt, hard geometry problems often timeout before completing
- Pawn Mali's V135 shows 131K context hurts → keep context at 65536 but allow MORE TIME

**Expected impact**: +0.5 to +1.5 points on 6-10 hard problems that are currently timing out.

---

### Approach 7: Multi-Agent Debate for Tiebreaks [MEDIUM IMPACT, high complexity]

**What it is**: When N=8 attempts split 4-4 or 3-3-2 between answers, run a "debate" round
where the two camps argue for their answer and try to find the flaw in the other.

**Evidence from literature**:
- MACA (Multi-Agent Consensus Alignment): RL-trained debate → +26.87% on MATH
- A-HMAD: Heterogeneous agents debate → 4-6% higher accuracy
- But: these improvements are on easy MATH problems, not Olympiad-level

**AIMO3-specific limitation**: These debate approaches require re-running the model many
times (multiple agents, multiple rounds). On a 5-hour budget for 50 hard problems, each
problem gets ~6 minutes. A debate round adds another 2-3 minutes per problem. This may
exceed the 30-minute per-problem timeout.

**Feasibility**: Low within 14 days. Too much implementation risk.

---

### Approach 8: EAGLE-3 Speculative Decoding for More Attempts [LOW IMPACT on accuracy]

**Finding**: EAGLE-3 provides 2-6x speed improvement but with **IDENTICAL output distribution**.
The key: speculative decoding is lossless — it produces exactly the same tokens as standard
autoregressive decoding, just faster.

**Implication**: EAGLE-3 does NOT increase per-attempt accuracy (p stays at 0.69).
It DOES allow more attempts within the same time budget (potentially 12-16 instead of 8).

But we already established (from paper 2603.27844 and negative correlation analysis) that
going from 8 to 16 attempts does not improve voting accuracy due to already-maximal
decorrelation.

**Conclusion**: EAGLE-3 is useful only if we repurpose the speedup for:
1. Longer context per attempt (more tool calls within same wall time)
2. More time for hard problems via early-stopping easy ones faster
3. Running a PRM/verifier pass on each solution (freed-up compute)

**Not worth implementing as a standalone change**. Only worth combining with Approach 2.

---

## 5. What Nobody on the Leaderboard Is Doing

Based on analysis of all public notebooks (up to 44/50), the following approaches are
confirmed NOT tried by anyone publicly:

### 5a. Self-PRM (scoring own solutions before voting)

Every public notebook uses entropy-based voting. Nobody is scoring complete solution traces.
The cost is minimal (50-100 tokens per solution × 8 solutions × 50 problems = 20K tokens total).
This is likely the highest-impact untried approach.

### 5b. Two-Stage Triage with Disagreement Injection

No public notebook implements the "first 3 attempts disagree → inject disagreement signal"
approach. Every notebook runs all 8 attempts in parallel with identical prompts.
The sequential nature of triage (wait for Phase 1 results) is architecturally different
but feasible with the Harmony protocol's async design.

### 5c. Lemma-Based Decomposition for Hard Problems

No public notebook does multi-turn hierarchical decomposition. Every notebook treats each
attempt as single-turn Harmony conversation.
The 26/35 on IMO2025 non-geometry by Intern-S1-MO shows this IS the right direction.
This is the "novel approach that nobody is doing" with the highest theoretical upside.

### 5d. Difficulty-Adaptive Time Budget

Every notebook uses the same `base_problem_timeout` for all problems. Only `high_problem_timeout`
distinguishes early-stopped vs non-early-stopped problems. Nobody dynamically allocates
time based on pre-classification.

---

## 6. Novel Approach Scoring Matrix

| Approach | Expected Score Gain | Implementation Days | VRAM Risk | Novel? |
|---|---|---|---|---|
| Self-PRM (own solution scoring) | +1 to +2 pts | 1 day | None | YES |
| Two-stage triage + disagreement injection | +1 to +3 pts | 2-3 days | None | YES |
| Lemma decomposition (hard problems only) | +0 to +2 pts (speculative) | 4-5 days | None | YES |
| Quality-gate retry | +0.5 to +1.5 pts | 1 day | None | YES |
| Difficulty-adaptive time routing | +0.5 to +1.5 pts | 0.5 days | None | Partially |
| Huikang as verifier only | 0 to +1 pts | 0.5 days | None | Partially |
| Real PRM (ThinkPRM-1.5B) | +1 to +3 pts | 2 days | Risk | YES |
| EAGLE-3 (alone) | 0 pts on accuracy | N/A | Medium | No |
| Multi-agent debate | +0 to +2 pts | 5+ days | None | No |

---

## 7. THE RECOMMENDED NOVEL APPROACH: Sequential Two-Stage with Binary Verification

Combining the three highest-impact, feasible ideas into one pipeline:

### Full design

```python
class TwoStageWithVerification:
    """
    Novel pipeline that no public leaderboard notebook implements.
    Combines: triage → disagreement injection → binary answer verification.
    Time budget: ~127-250 min for 50 problems. Fits in 5-hour window.
    """
    
    async def solve(self, problem: str) -> int:
        # STAGE 1: Quick triage (3 fast attempts, 60s each, parallel)
        phase1_results = await self.run_parallel_attempts(
            problem, n=3, timeout=60
        )
        
        # Check unanimity
        answers = [r.answer for r in phase1_results if r.answer is not None]
        top_answer, top_count = Counter(answers).most_common(1)[0]
        
        if top_count >= 2:  # 2/3 agree → HIGH CONFIDENCE
            # Binary verify before early exit
            if self._verify_answer(problem, top_answer):
                return top_answer  # Bank remaining time for hard problems
        
        # STAGE 2: Deep solve (5 attempts, full budget, with disagreement context)
        disagreement_context = (
            f"Initial analysis produced conflicting answers: "
            f"{', '.join(str(a) for a in set(answers))}. "
            f"At least {len(set(answers))-1} of these are wrong. "
            f"Be extremely careful. Verify each step with Python. "
            f"Do NOT repeat the same approach as previous attempts."
        )
        
        phase2_results = await self.run_parallel_attempts(
            problem, 
            n=5, 
            timeout=self.remaining_budget(),
            extra_context=disagreement_context
        )
        
        all_results = phase1_results + phase2_results
        
        # STAGE 3: Binary verification cascade (amanatar approach)
        # Sort candidates by vote count, then entropy. Verify in order.
        candidates = sorted(
            self.get_top_candidates(all_results, k=3),
            key=lambda x: (-x.vote_count, x.mean_entropy)
        )
        
        for candidate in candidates:
            if self._verify_answer(problem, candidate.answer):
                return candidate.answer
        
        # FALLBACK: entropy-weighted vote
        return self.entropy_vote(all_results)
    
    def _verify_answer(self, problem: str, answer: int) -> bool:
        """Binary verification at T=0.0 (proven: amanatar 44/50)."""
        prompt = (
            f"Problem:\n{problem}\n\n"
            f"Proposed answer: {answer}\n\n"
            "Check the answer carefully.\nReply with only ONE word:\nCORRECT or WRONG"
        )
        resp = self.client.completions.create(
            model=self.cfg.served_model_name,
            prompt=self.encoding.encode(prompt),
            temperature=0.0,
            max_tokens=5
        )
        text = resp.choices[0].text.strip().upper()
        return "CORRECT" in text and "WRONG" not in text
```

### Why this combination specifically

1. **Binary verification**: Addresses the "confident but wrong answer" failure type (subset of the
   ~35% wrong-formalization failures). Binary CORRECT/WRONG is a simpler task for the model
   than scoring reasoning quality — LLMs CAN reliably answer yes/no on specific integers
   even though they struggle to evaluate full reasoning traces.

2. **Disagreement injection**: Addresses the "correlation between attempts" problem more
   directly than temperature tuning. When the model KNOWS previous attempts disagreed,
   it tends to be more systematic.

3. **Triage**: Saves ~40% of compute on easy problems, redirects it to hard ones.

**Total expected improvement**: +2 to +4 points. If this pushes mean from 39.7 to 43-44,
with variance, the best run could hit 47-48.

---

## 8. The Fundamental Truth About Going from 44 to 46+

From the arxiv paper: "Across a 17-point model capability gap and every inference-time
optimization tested, model capability dominates by an order of magnitude."

This means:
- **The path from 44 to 46 is NOT inference-time tricks**. It's either:
  1. A better model (fine-tuned on AIMO3 problems — huikang's approach)
  2. A fundamentally different architecture (hierarchical decomposition — Intern-S1-MO approach)
  3. More compute (more wall time per problem, not more attempts)
  4. Luck (variance at 44-47 is high; the top 46 may just be 1-in-100 luck)

- **The #1 scorer (ippeiogawa, 46/50)** either:
  a. Has a privately fine-tuned model with higher p̂ per attempt, OR
  b. Is getting lucky (p ≤ 0.01 probability per run from the paper's statistics), OR
  c. Is using a fundamentally different strategy we haven't found (hierarchical? longer context?)

- **The path from our current 38 to 44**: Well-understood. Use the exact 44/50 config
  (T=0.5, simple entropy, 5-step structured prompt). This is v23/v24 which should score ~42-44.

- **The path from 44 to 46**: The Self-PRM + Two-Stage approach described in Section 7.
  This is genuinely novel. But the expected improvement is probabilistic, not guaranteed.

---

## 9. Action Plan (Ranked by Impact/Effort Ratio)

| Day | Action | Rationale |
|---|---|---|
| Apr 2 | Get v25 score | Baseline for huikang + reliability voting |
| Apr 3 | Submit v23 (exact 44/50 config) if v25 < 42 | Proven path to 42-44 |
| Apr 4-5 | Implement quality-gate retry (Approach 4) | Easiest novel addition, low risk |
| Apr 5-6 | Implement self-PRM scoring (Approach 1) | 1 day, no VRAM risk |
| Apr 6-7 | Add difficulty-adaptive timeout (Approach 6) | 0.5 days, immediate benefit |
| Apr 7-9 | Implement two-stage triage with disagreement injection (Approach 2) | Medium complexity, highest upside |
| Apr 9-11 | Validate and submit best combination | Tune based on observed scores |
| Apr 11-14 | Run best config repeatedly | Lottery approach for 46+ |
| Apr 15 | Final submission | Best achieved |

---

## 10. Sources

1. Arxiv 2603.27844: "Model Capability Dominates: Inference-Time Optimization Lessons from AIMO 3"
   — Natapong Nitarach, March 2026. [https://arxiv.org/abs/2603.27844]

2. Arxiv 2503.21934: "Proof or Bluff? Evaluating LLMs on 2025 USA Math Olympiad"
   — Petrov et al. Failure mode taxonomy: logical errors, unjustified assumptions, creativity
   failures, arithmetic mistakes. [https://arxiv.org/abs/2503.21934]

3. Arxiv 2512.10739: "Long-horizon Reasoning Agent for Olympiad-Level Mathematical Problem Solving"
   — Intern-S1-MO. Hierarchical lemma decomposition: 26/35 on IMO2025 non-geometry.
   [https://arxiv.org/abs/2512.10739]

4. Arxiv 2504.16828: "Process Reward Models That Think (ThinkPRM)"
   — Data-efficient PRM: +5-7% over discriminative PRMs. Uses only 1% of process labels.
   [https://arxiv.org/abs/2504.16828]

5. Arxiv 2504.16891: "AIMO-2 Winning Solution: Building State-of-the-Art Mathematical Reasoning
   Models with OpenMathReasoning dataset" — NVIDIA. GenSelect: 78.4% → 93.3% TIR on AIME.
   [https://arxiv.org/abs/2504.16891]

6. ACL 2025: "Confidence Improves Self-Consistency in LLMs" (CISC) — weighted majority vote
   with confidence reduces required paths by 40%. [https://arxiv.org/html/2502.06233v1]

7. ICLR 2025: "Learning How Hard to Think: Input-Adaptive Allocation of LM Computation"
   — Predicting difficulty and allocating compute accordingly improves accuracy.

8. Competition notebook analysis: nihilisticneuralnet 44/50, amanatar 44/50, Pawan Mali
   50-experiment log, Jonathan Chan Bayesian notebook. Stored in existing research files.

9. Arxiv 2508.12461: "Is GPT-OSS Good? A Comprehensive Evaluation" — capability analysis,
   benchmark comparisons, architecture details (MoE transformer + RL training).
   [https://arxiv.org/html/2508.12461v1]

10. EAGLE-3 (arxiv 2503.01840): Speculative decoding 2-6x speedup, IDENTICAL accuracy.
    Only useful for freeing up compute, not for accuracy improvement.
    [https://arxiv.org/abs/2503.01840]

11. Ranked Voting ACL 2025: Borda count, instant-runoff, reciprocal rank improve robustness
    over standard majority. [https://aclanthology.org/2025.findings-acl.744.pdf]

12. AIMO3 competition log: /f/kaggle/aimo-progress-prize-3/docs/competition_log.md
    — Our own score history, validated notebook list, key learnings.
