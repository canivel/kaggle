# v18 Failure Distribution Analysis: Which 12 Problems Were Wrong?

**Date:** 2026-04-01
**Score:** 38/50 (12 wrong)
**Question:** What types of problems did v18 most likely fail on, and do our planned innovations target the right failure modes?

---

## Factual Correction: arxiv 2603.27844 Has No Per-Category Data

The paper "Model Capability Dominates: Inference-Time Optimization Lessons from AIMO 3" (Natapong Nitarach, March 2026) was read in full (17 pages). It contains **zero per-category breakdown** (no table or figure showing algebra vs combinatorics vs geometry vs number theory accuracy). It reports only aggregate statistics:

- Baseline: µ = 39.7/50, σ = 1.7 (13-run mean), range 37-44
- Per-attempt accuracy p̂ ≈ 0.69 on GPT-OSS-120B

Any prior claim that this paper states "combinatorics > geometry > algebra > number theory" failure ordering is incorrect. That ordering may be correct, but its source is a different paper (OlymMATH benchmark, arxiv 2503.21380), not 2603.27844.

---

## Reference Problem Categorization (10 Public Problems)

Downloaded from Kaggle competition data (`reference.csv`). Categorized by problem type:

| ID | Category | Description |
|----|----------|-------------|
| 0e644e | Geometry | Triangle ABC, integer sides, circle intersections, minimal perimeter |
| 641659 | Geometry | Triangle + incircle + circumcircle + Fibonacci sequences + cyclic condition |
| 26de63 | Number Theory | Floor-sum function, M = 2·3·5·7·11·13, largest 2^k divisor |
| 86e8e5 | Number Theory | n-Norwegian integers, divisors, limit/approximation as n → large |
| 424e18 | Combinatorics | Tournament with 2^20 runners, orderings of competitors, 10^k divides N |
| a295e9 | Combinatorics | 500×500 grid, rectangles with distinct perimeters, max count |
| 92ba6a | Algebra | Alice/Bob ages puzzle, system of equations |
| 9c1c5f | Algebra | Functional equation f(m)+f(n)=f(m+n+mn), count values of f(2024) |
| 42d360 | NT/Combo (mixed) | Base conversion moves, largest M across all n ≤ 10^(10^5) |
| dd7f5e | Algebra/Combo (mixed) | "Shifty functions" with shift operators, count functions |

Distribution: 2 geometry, 2 number theory, 2 combinatorics, 2 algebra, 2 mixed.
Consistent with roughly equal distribution across the full 50 (≈12-13 per category).

---

## Category Failure Rates from OlymMATH Benchmark (arxiv 2503.21380)

The strongest available per-category data for IMO-level problems. This is OlymMATH-EN HARD subset:

| Category | Gemini 2.5 Pro | o3-mini (high) | DeepSeek-R1 |
|----------|---------------|----------------|-------------|
| Algebra | 71.5% correct | 29.5% correct | 30.0% correct |
| Geometry | 75.5% correct | 29.0% correct | 25.5% correct |
| Number Theory | 59.0% correct | 49.5% correct | 18.5% correct |
| Combinatorics | 27.5% correct | 17.0% correct | 4.0% correct |

Key finding: **Combinatorics is the most challenging domain across all models by a wide margin.** Even the strongest model (Gemini 2.5 Pro) scores only 27.5% on combinatorics vs 71.5% on algebra.

The geometry anomaly: Gemini is very strong on geometry (75.5%) but smaller models are not (DeepSeek-R1: 25.5%). This suggests geometry difficulty is more model-size dependent than combinatorics difficulty.

From CombiBench (arxiv 2505.03171): AlphaProof solved 4 of 6 IMO 2024 problems but the 2 unsolved were combinatorial. Combinatorics consistently resists automation across model types.

---

## Estimated Failure Distribution for v18's 12 Wrong Answers

**GPT-OSS-120B context:** p̂ = 0.69 overall. This is between o3-mini and Gemini 2.5 Pro in capability. We cannot map OlymMATH directly to AIMO3 problems (different problem set, different difficulty calibration), but the relative ordering across categories is consistent across all benchmarks.

Assumptions:
- ~12-13 problems per category in the 50 public problems
- Failure rate scaling: combinatorics worst, then geometry ≈ algebra ≈ number theory (with geometry slightly harder for smaller models)

| Category | Est. problems (~13 each) | Est. v18 failures | Est. failure rate | Reasoning |
|----------|--------------------------|-------------------|-------------------|-----------|
| Combinatorics | ~13 | ~5 | ~38% | Consistently hardest across all benchmarks; multi-level case analysis resists TIR |
| Geometry | ~13 | ~3-4 | ~27% | Synthetic insight problems (like ref 641659) not solvable by coordinate bash; but many geometry problems ARE solvable numerically |
| Algebra | ~13 | ~2 | ~15% | Generally solvable with sympy; lowest failure rate |
| Number Theory | ~13 | ~2 | ~15% | Fast modular arithmetic available; sympy handles most NT tasks |

**Most likely: ~5 combinatorics + ~4 geometry + ~2 algebra + ~1 NT = 12 wrong**

Note: This is an estimate. Without seeing which specific problems v18 answered and their labels, exact categorization is impossible.

---

## What the arxiv Paper (2603.27844) DOES Tell Us About GPT-OSS-120B Failures

Even without per-category data, the paper gives important structural information:

### 1. The Model Is at Its Capability Ceiling

Per-attempt accuracy p̂ = 0.69 means the model gets each problem right ~69% of the time in a single attempt. With 8 attempts:
- Easy problems (p≈1): always correct regardless of prompting
- Hard problems (p≈0): never correct regardless of prompting
- Boundary problems (p≈0.3-0.7): where all the variance lives

The 12 wrong answers are problems where p < 0.5 — where even after 8 attempts + majority vote, the model cannot reach a correct majority. These are problems the model fundamentally cannot solve, not problems it solves inconsistently.

### 2. Inference-Time Tricks Do Not Help on Capability-Limited Problems

Table 4 in the paper: across 23 experiments, NO configuration reliably beats the 39.7/50 baseline. The best single run was 44 but the same config replicated at 38 and 34 (mean 37.7).

This means: the 12 problems v18 got wrong are likely the same 12 problems nihilisticneuralnet's baseline gets wrong. The boundary between 38 and 44 is stochastic variance, not systematic improvement from any particular technique.

### 3. The 44-→46 Gap Requires p̂ Improvement, Not Better Voting

The paper shows that the 17-point capability gap between GPT-OSS-120B (p̂=0.69) and the next tier models dwarfs any inference-time optimization (±2 points). To reliably reach 45-46, you need p̂ ≈ 0.80-0.85, which requires a better base model or fine-tuning, not prompt engineering.

---

## Critical Finding: Domain Routing Has Negative Evidence in This Competition

**This directly contradicts the proposed "domain routing" innovation.**

From Table 4 / Table 1 in arxiv 2603.27844 (same model, same competition):

| Strategy | LB Score | Delta from baseline |
|----------|----------|---------------------|
| Baseline (8x original) | 39.7 (mean) | — |
| 8x Classify Then Solve | **36** | **-3.7** (worst individual strategy) |

The "Classify Then Solve" (E3) prompt was: "First classify: is this number theory, algebra, combinatorics, or geometry? Then recall canonical techniques. Then apply."

This scored 36/50, 3.7 points BELOW baseline. It is the worst-performing single strategy in the entire experiment set.

**Why domain routing hurt:**
1. The model is not "prompt-robust" — training data uses mostly CoT and TIR formats, not classification-then-solve formats
2. The classification step consumes tokens/turns that could be used for solving
3. At T=1.0, the model already explores different approaches stochastically; explicit classification is redundant and shifts the prompt distribution away from training data

**The key reconciliation:** There is a distinction between:
- A separate classification SYSTEM PROMPT that replaces the solving prompt (E3 approach: scores 36) — this HURTS
- A domain-specific PREAMBLE prepended to the existing 5-step prompt that adds "here are useful tools for this problem type" — this is untested but lower risk

However, even the preamble approach should be applied with extreme caution given the clear evidence that any prompt divergence from the training distribution costs points.

---

## Do Our Innovations Target the Right Failure Modes?

### Innovation 1: Domain Routing (geometry routing, combinatorics routing)

**Assessment: WRONG TARGET, with caveats**

The 12 wrong problems are likely concentrated in combinatorics and geometry. Domain routing sounds intuitive. But:

- The direct experimental evidence (E3: -3.7 pts) shows classification-then-solve HURTS on this exact model
- Combinatorics failures are a CAPABILITY gap (multi-level case analysis), not a PROMPTING gap. The model cannot build the insight needed; telling it "use itertools.combinations" does not help
- Geometry failures split between: (a) coordinate-solvable problems (TIR already handles these), and (b) synthetic insight problems (like ref 641659 — no coordinate framing helps)
- A domain preamble prepended to the existing prompt is lower risk than E3 but the evidence for benefit is weak (+1.7% from a different model, Meta-Harness paper)

**Recommendation:** Keep domain routing only as a minimal preamble (2-4 lines of tool hints) on the existing 5-step prompt. Do NOT replace the system prompt with a domain-specific one.

### Innovation 2: Phase Splitting (Phase 1 quick → Phase 2 targeted cross-verification)

**Assessment: CORRECT TARGET**

The 12 wrong answers include:
- ~5 combinatorics where p ≈ 0.2: Phase 2 will also fail (capability gap), no impact
- ~4 geometry: same issue for synthetic problems, but coordinate-solvable ones may benefit
- ~2-3 boundary problems across categories where the model gets it right sometimes but wrong more often

Phase 2 cross-verification specifically helps boundary problems: when Phase 1 produces two competing candidates, Phase 2 with "is it X or Y?" framing focuses compute on the distinction. This is more efficient than 8 independent re-attempts.

The question is: what fraction of the 12 wrong answers are genuine boundary problems (p ≈ 0.3-0.5) vs capability-floor problems (p ≈ 0)?

From the baseline distribution (range 37-44 across 13 runs on same config), the variance of ±1.7 implies roughly 3-4 boundary problems per run. These are the problems where phase splitting can flip outcomes.

**Expected impact: +0.5-1.5 correct answers on the boundary problems, zero impact on capability-floor problems.**

### Innovation 3: Failure-Aware Retry (retry None/no-code attempts)

**Assessment: CORRECT TARGET, well-scoped**

This is additive and low-risk. On problems where the model fails to produce any code or any boxed answer, a fresh attempt with mandatory-Python instruction recovers wasted slots. The key is it does NOT show the model its previous answer (avoids the self-refinement failure mode).

Consistent with Aman Atar's `_verify_answer` (confirmed 44/50): using the model for post-hoc checking is valuable if done correctly.

---

## Summary: Estimated Failure Profile and Innovation Targeting

| Wrong problem type | Estimated count | Our innovation targets it? |
|-------------------|-----------------|---------------------------|
| Combinatorics (capability floor, p<0.3) | ~4 | No — not addressable via prompting or phase splitting |
| Combinatorics (boundary, p≈0.4) | ~1 | Partially — phase splitting may help 1 |
| Geometry (synthetic insight, p<0.3) | ~2-3 | No — requires a better model or formal geometry |
| Geometry (coordinate-solvable, p≈0.4) | ~1-2 | Phase splitting may help 1 |
| Algebra/NT (boundary problems) | ~2 | Phase splitting + domain routing may help 1 |

**Realistic expectation from our innovations:** +1 to +2 additional correct answers (from 38 to 39-40), not the +6 needed to reach 44.

**The 44-line is largely a model capability threshold.** The jump from ~38 (our v18) to ~44 (nihilisticneuralnet) is achieved primarily by:
1. 5-step structured system prompt (proven: +4-6 pts)
2. Temperature tuning (0.5 or 0.8) (proven: ±2 pts)
3. Lucky run variance (σ=1.7, max=44 in 13 runs)

The innovations (phase splitting, domain routing, failure-aware retry) contribute at the margin (+1-2 pts) on top of an already well-configured baseline. They are not the primary path to 44+.

---

## Actionable Prioritization Given This Analysis

1. **First**: Ensure v25 uses the exact proven 44/50 configuration (T=0.5, 5-step prompt, simple entropy). This is worth ~+4-6 points from our baseline, not from our innovations.

2. **Second**: Add failure-aware retry to the existing pipeline (Approach 2 from adaptive multistage research). Low risk, additive, addresses the "no-code/no-answer" failure mode.

3. **Third**: Add a minimal domain preamble (2-4 lines of tool hints) to the 5-step prompt. Keep it SHORT and additive. Do NOT classify-then-solve (E3 approach = -3.7 pts confirmed).

4. **Fourth**: Implement phase splitting for submissions Apr 6+, after confirming the base config works. Only use if time budget allows.

5. **Avoid**: Any approach that replaces the system prompt with a domain-specific one. The E3 evidence is direct and damning.

---

## Sources

- arxiv 2603.27844 (PDF, read in full): Natapong Nitarach, "Model Capability Dominates: Inference-Time Optimization Lessons from AIMO 3", March 2026
- arxiv 2503.21380: OlymMATH benchmark, per-category LLM accuracy data
- arxiv 2505.03171: CombiBench, combinatorics as hardest domain for LLMs
- reference.csv: AIMO3 10 public reference problems (downloaded 2026-04-01)
- Competition log: /f/kaggle/aimo-progress-prize-3/docs/competition_log.md
- Score differentiators: /f/kaggle/research/aimo3_score_differentiators.md
- Urgent research: /f/kaggle/research/aimo3_urgent_research_2026_04_01.md
- Adaptive multistage research: /f/kaggle/research/adaptive_multistage_math_solving_2026_04_01.md
