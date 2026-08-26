# Research Task: time-budget-optimizer

You are optimizing time allocation across 50 problems in a 9-hour Kaggle kernel.

CONTEXT:
- 50 problems, 9 hours total, 16 parallel Jupyter kernels
- Current: flat 300s per problem = 15000s total (4.17 hours) — wastes half the budget
- Easy problems (p=0.95) solve in 1-2 attempts (30-60s each)
- Hard problems (p=0.40) need all attempts + retries (600-900s)
- Early_stop=4: when 4 attempts agree, stop immediately

KEY QUESTION: What's the optimal time allocation strategy?

RESEARCH TASKS:
1. Calculate: if easy problems take 60s and hard ones take 600s, how many of each can we fit in 9 hours?
2. How do we DETECT easy vs hard early? (after 2 attempts, if both agree → easy)
3. What's the expected score improvement from adaptive time vs flat time?
4. Write a concrete adaptive_budget() function that replaces the current flat allocation

Write analysis to /time_optimization.md and code to /adaptive_budget.py
