"""AIMO3 Autonomous Research Lab — KAOS v0.3.0

Multi-agent research loop:
1. Spawn N research agents in parallel, each exploring a different hypothesis
2. Each agent writes a solve_problem() variant to its VFS
3. Evaluate each variant against the AIMO3 Reference Bench (10 problems)
4. Score, compare, checkpoint winners
5. Propose next iteration based on what worked

Research directions:
- failure-analysis: WHY does GPT-OSS-120B fail on P5-P10? What's the failure mode?
- time-allocation: Can we allocate more time to hard problems, less to easy ones?
- answer-extraction: Are we losing points to extraction failures, not reasoning?
- multi-turn: Can a follow-up turn recover wrong first attempts?
- ensemble-diversity: Does seed diversity actually help at N=8-12?

Usage:
    cd f:/kaggle/aimo-progress-prize-3
    f:/kaggle/kaos/.venv/Scripts/python.exe scripts/aimo3_research_lab.py
"""

from __future__ import annotations

import json
import io
import sys
import time
from pathlib import Path

KAOS_DIR = Path(__file__).parent.parent.parent / "kaos"
sys.path.insert(0, str(KAOS_DIR))

from kaos import Kaos

DB_PATH = Path(__file__).parent.parent / "aimo3-learnings.db"
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "experiments"

# The base solve_problem from v16 (our proven 39/50 config)
BASE_SOLVER = '''
def solve_problem(self, problem):
    """Baseline solver: 8 parallel attempts, entropy voting, early_stop=4."""
    user_input = f'{problem} {self.cfg.preference_prompt}'
    elapsed = time.time() - self.notebook_start_time
    left = self.cfg.notebook_limit - elapsed
    reserved = max(0, self.problems_remaining - 1) * self.cfg.base_problem_timeout
    budget = min(max(left - reserved, self.cfg.base_problem_timeout), self.cfg.high_problem_timeout)
    deadline = time.time() + budget

    tasks = [(self.cfg.system_prompt, i) for i in range(self.cfg.attempts)]
    detailed, valid = [], []
    stop = threading.Event()
    ex = ThreadPoolExecutor(max_workers=self.cfg.workers)
    try:
        futs = [ex.submit(self._process_attempt, user_input, sp, ai, stop, deadline)
                for sp, ai in tasks]
        for f in as_completed(futs):
            try:
                r = f.result()
                detailed.append(r)
                if r['Answer'] is not None:
                    valid.append(r['Answer'])
                c = Counter(valid).most_common(1)
                if c and c[0][1] >= self.cfg.early_stop:
                    stop.set()
                    for ff in futs: ff.cancel()
                    break
            except Exception as e:
                print(f'Error: {e}')
    finally:
        stop.set()
        ex.shutdown(wait=True, cancel_futures=True)
        self.problems_remaining = max(0, self.problems_remaining - 1)
    if not valid:
        return 0
    return self._select_answer(detailed)
'''

RESEARCH_DIRECTIONS = [
    {
        "name": "failure-analyst",
        "task": """You are analyzing WHY GPT-OSS-120B fails on hard IMO-level math problems.

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
""",
    },
    {
        "name": "time-budget-optimizer",
        "task": """You are optimizing time allocation across 50 problems in a 9-hour Kaggle kernel.

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
""",
    },
    {
        "name": "voting-innovator",
        "task": """You are researching novel voting/answer-selection strategies for math competitions.

CONTEXT:
- Current: simple majority vote with 1/entropy weighting
- 8 attempts per problem, each produces an integer answer + entropy score
- The CORRECT answer is often in the candidate set but gets outvoted
- Pawan Mali: complex entropy weighting = 0 improvement
- Amanatar: binary verify cascade for top candidates = 44/50

KEY QUESTION: Is there a voting strategy that extracts more signal from 8 attempts?

RESEARCH TASKS:
1. Analyze: if p=0.69 per attempt and N=8, what's P(correct answer appears at least once)? P(majority vote correct)?
2. Research "weighted majority" schemes: logprob-weighted, code-execution-weighted, answer-confidence-weighted
3. Research "best-of-N with verification": instead of voting, verify top-K candidates independently
4. Can we use the MODEL ITSELF to break ties? ("Given answers A=5, B=7, which is more likely correct for this problem?")
5. What about answer clustering by value proximity? (if 5 attempts say 42 and 3 say 43, the 42s are likely right even at lower entropy)

Write analysis to /voting_research.md and code to /novel_voter.py
""",
    },
    {
        "name": "multi-turn-researcher",
        "task": """You are researching multi-turn prompting strategies for math problem solving.

CONTEXT:
- Current: single-turn — give problem, get answer, done
- Follow-up prompt exists for no-boxed cases: ask for boxed answer at T=0.0
- Self-refinement: PROVEN HARMFUL (-4 pts in Pawan Mali experiments)
- But multi-turn with NEW INFORMATION (not self-refinement) might help

KEY QUESTION: Can we design a multi-turn strategy that adds value without self-refinement?

SPECIFIC IDEAS TO RESEARCH:
1. "Verification turn": After attempt produces answer X, ask "Verify that X satisfies [problem constraints]. If not, what's the correct answer?"
   - Different from self-refinement because it gives a SPECIFIC claim to verify
2. "Approach suggestion turn": If first attempt fails (no code, no answer), provide a hint: "Try using modular arithmetic" or "Consider the generating function approach"
3. "Cross-attempt synthesis": After 4 attempts with split votes, show the model the conflicting approaches and ask it to adjudicate
4. "Decomposition turn": For complex problems, first ask "What are the key sub-problems?", then solve each sub-problem separately

CRITICAL CONSTRAINT: any multi-turn strategy must NOT be self-refinement (asking "are you sure?" always hurts).
The key is providing NEW information in the follow-up that the model didn't have before.

Write analysis to /multi_turn_research.md and code to /multi_turn_solver.py
""",
    },
    {
        "name": "harness-architect",
        "task": """You are designing the OPTIMAL harness architecture for AIMO3.

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
""",
    },
]


def main():
    print("=" * 60)
    print("AIMO3 Autonomous Research Lab — KAOS v0.3.0")
    print("=" * 60)

    db = Kaos(str(DB_PATH))
    agents = db.list_agents()
    main_agent = agents[0]["agent_id"] if agents else None

    # Store research plan
    if main_agent:
        db.set_state(main_agent, "research_lab", {
            "status": "launching",
            "directions": [d["name"] for d in RESEARCH_DIRECTIONS],
            "base_solver": "v16 exact 44/50 params",
            "benchmark": "AIMO3 Reference (10 problems)",
            "goal": "Find innovations that improve score beyond 39/50",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        })

    # Output research tasks for Claude Code agents
    for d in RESEARCH_DIRECTIONS:
        print(f"\n{'='*60}")
        print(f"Agent: {d['name']}")
        print(f"{'='*60}")
        print(d["task"][:300] + "...")

    # Write tasks to files for agent consumption
    for d in RESEARCH_DIRECTIONS:
        outfile = RESULTS_DIR / f"research_task_{d['name']}.md"
        with io.open(outfile, "w", encoding="utf-8") as f:
            f.write(f"# Research Task: {d['name']}\n\n")
            f.write(d["task"])
        print(f"  -> {outfile}")

    if main_agent:
        db.checkpoint(main_agent, label="research-lab-launched-2026-04-05")

    db.close()
    print(f"\n{'='*60}")
    print("Research tasks written. Launch via Claude Code Agent tool.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
