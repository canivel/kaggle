# Research Task: failure-analyst

You are analyzing WHY GPT-OSS-120B fails on hard IMO-level math problems.

CONTEXT:
- GPT-OSS-120B solves 4/10 on the AIMO3 Reference Bench (only the easy P1-P4)
- It fails on P5-P10 (tournament counting, digit-sum optimization, Fibonacci geometry, etc.)
- On the actual 50-problem Kaggle test, it scores ~39/50 (so most competition problems are easier)
- Per-attempt accuracy is p=0.69 — meaning 31% of attempts get the WRONG answer

KEY QUESTION: What are the failure modes? Categorize into:
1. Wrong mathematical approach (picked wrong theorem/technique)
2. Calculation error (right approach, arithmetic mistake)
3. Code execution error (Python code crashed or gave wrong result)
4. Answer extraction failure (solved correctly but answer not extracted)
5. Timeout (ran out of tokens/time)
6. Misread problem (misunderstood what was asked)

For each failure mode, propose a SPECIFIC code change to solve_problem() that would fix it.
Do NOT propose vague ideas. Write actual Python code snippets.

Write your analysis to /failure_analysis.md and proposed code changes to /proposed_fixes.py
