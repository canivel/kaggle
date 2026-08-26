# Research Task: harness-architect

You are designing the OPTIMAL harness architecture for AIMO3.

CONTEXT:
- Model: GPT-OSS-120B via vLLM on H100 80GB
- Protocol: Harmony (completions API, NOT chat)
- 16 Jupyter kernels for code execution (TIR)
- 9 hour time limit, 50 problems
- Current best: 39/50 (flat 8 attempts, entropy vote)
- Theoretical ceiling at p=0.69: ~44/50 with perfect voting at N=8
- To reach 47+: need p=0.85 per attempt (model ceiling issue)

KEY QUESTION: What is the theoretically optimal harness given fixed model capability?

RESEARCH TASKS:
1. MATH: Compute expected score as function of (p, N, voting_strategy) for p in [0.60, 0.90] and N in [4, 16]
2. MATH: What's the optimal N given time budget? More attempts = better voting but less time per attempt
3. DESIGN: If we could perfectly classify problems into easy (p=0.95) and hard (p=0.40), what's the optimal strategy?
4. DESIGN: What's the information-theoretic limit? Given N attempts with error rate (1-p), what's the best possible accuracy?
5. CODE: Write the optimal harness as a solve_problem() function that:
   - Adapts N based on early consensus
   - Adapts time budget based on difficulty
   - Uses the best voting strategy
   - Handles edge cases (no answer, all disagree, timeout)

Write analysis to /optimal_harness.md and code to /optimal_solver.py
