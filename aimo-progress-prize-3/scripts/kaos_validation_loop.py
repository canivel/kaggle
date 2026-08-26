"""KAOS v0.3.1 Validation Loop for AIMO3.

Orchestrates build → validate → iterate using KAOS agents and checkpoints.
Each experiment is a KAOS agent with:
  - VFS: stores the notebook variant + config
  - State: scores, per-problem results, comparison with baseline
  - Checkpoints: snapshots at each validation run

Loop:
  1. Register experiment as KAOS agent
  2. Run validation against AIMO3 reference bench (10 problems)
  3. Score and compare with best known result
  4. If improved: checkpoint as new best, build notebook
  5. If regressed: restore previous checkpoint, try different approach

Usage:
    cd f:/kaggle/aimo-progress-prize-3
    f:/kaggle/kaos/.venv/Scripts/python.exe scripts/kaos_validation_loop.py --experiment strategy-diversity
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

DB_PATH = Path(__file__).parent.parent / "aimo3-validation.db"
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "experiments"


# ── Experiment definitions ──────────────────────────────────────────────────

EXPERIMENTS = {
    "baseline": {
        "description": "v16 clean — identical to 44/50 notebooks",
        "preference_strategies": {
            "standard": {
                "count": 8,
                "preference": (
                    "You have access to `math`, `numpy`, and `sympy` for:\n\n"
                    "# Symbolic Computation (sympy):\n"
                    "- Algebraic manipulation and simplification\n"
                    "- Solving equations and systems of equations\n"
                    "- Number theory functions (primes, divisors, modular arithmetic)\n"
                    "- Polynomial operations and factorization\n\n"
                    "# Numerical Computation (numpy):\n"
                    "- Array operations and linear algebra\n"
                    "- Efficient numerical calculations\n\n"
                    "Best Practices:\n"
                    "- Use sympy for exact symbolic answers when possible\n"
                    "- Use numpy for numerical verification\n"
                    "- Combine symbolic and numerical approaches\n"
                    "- Validate results against known cases"
                ),
            },
        },
    },
    "strategy-diversity": {
        "description": "Multi-strategy: 4 standard + 2 code-first + 2 small-cases",
        "preference_strategies": {
            "standard": {
                "count": 4,
                "preference": (
                    "You have access to `math`, `numpy`, and `sympy` for:\n\n"
                    "# Symbolic Computation (sympy):\n"
                    "- Algebraic manipulation and simplification\n"
                    "- Solving equations and systems of equations\n"
                    "- Number theory functions (primes, divisors, modular arithmetic)\n"
                    "- Polynomial operations and factorization\n\n"
                    "# Numerical Computation (numpy):\n"
                    "- Array operations and linear algebra\n"
                    "- Efficient numerical calculations\n\n"
                    "Best Practices:\n"
                    "- Use sympy for exact symbolic answers when possible\n"
                    "- Use numpy for numerical verification\n"
                    "- Combine symbolic and numerical approaches\n"
                    "- Validate results against known cases"
                ),
            },
            "code_first": {
                "count": 2,
                "preference": (
                    "Solve this by writing a complete Python program. Go directly to code.\n"
                    "Available: math, numpy, sympy.\n"
                    "Your program must:\n"
                    "1. Compute the answer step by step\n"
                    "2. Verify it satisfies the problem constraints\n"
                    "3. Print the final answer\n"
                    "Prefer exact computation with sympy over floating point."
                ),
            },
            "small_cases": {
                "count": 2,
                "preference": (
                    "Start by testing small cases to find a pattern.\n"
                    "For example, if the problem involves n, try n=1,2,3,...,10.\n"
                    "Write Python code to:\n"
                    "1. Compute results for small cases\n"
                    "2. Identify the pattern or formula\n"
                    "3. Verify the pattern holds for larger cases\n"
                    "4. Compute the final answer\n"
                    "Available: math, numpy, sympy"
                ),
            },
        },
    },
    "strategy-diversity-v2": {
        "description": "Multi-strategy v2: 3 standard + 2 code-first + 2 small-cases + 1 brute-force",
        "preference_strategies": {
            "standard": {
                "count": 3,
                "preference": (
                    "You have access to `math`, `numpy`, and `sympy`.\n"
                    "Use sympy for exact symbolic answers when possible.\n"
                    "Combine symbolic and numerical approaches.\n"
                    "Validate results against known cases."
                ),
            },
            "code_first": {
                "count": 2,
                "preference": (
                    "Solve this by writing a complete Python program. Go directly to code.\n"
                    "Your program must compute the answer and verify constraints.\n"
                    "Available: math, numpy, sympy."
                ),
            },
            "small_cases": {
                "count": 2,
                "preference": (
                    "Start by testing small cases (n=1,2,3,...) to find a pattern.\n"
                    "Write Python code to verify the pattern, then compute the final answer.\n"
                    "Available: math, numpy, sympy"
                ),
            },
            "brute_force": {
                "count": 1,
                "preference": (
                    "If the search space is manageable, write a Python brute-force program "
                    "that checks all possible values. Otherwise, enumerate systematically.\n"
                    "Available: math, numpy, sympy, itertools"
                ),
            },
        },
    },
}


def register_experiment(db: Kaos, name: str, config: dict) -> str:
    """Register an experiment as a KAOS agent."""
    agents = db.list_agents()
    # Check if already exists
    for a in agents:
        if a["name"] == name:
            print(f"  Agent '{name}' already exists: {a['agent_id'][:16]}...")
            return a["agent_id"]

    a_id = db.spawn(name)
    db.set_state(a_id, "config", config)
    db.set_state(a_id, "status", "registered")
    db.set_state(a_id, "created", time.strftime("%Y-%m-%dT%H:%M:%SZ"))
    db.checkpoint(a_id, label=f"{name}-registered")
    print(f"  Registered agent '{name}': {a_id[:16]}...")
    return a_id


def load_benchmark():
    """Load the AIMO3 reference problems."""
    bench_path = DATA_DIR / "aimo3_reference_bench.json"
    with io.open(bench_path, encoding="utf-8") as f:
        return json.load(f)


def record_result(db: Kaos, agent_id: str, problem_id: str, expected: int,
                  predicted: int, strategy: str, correct: bool):
    """Record a single problem result."""
    results = db.get_state(agent_id, "results") or []
    results.append({
        "problem_id": problem_id,
        "expected": expected,
        "predicted": predicted,
        "strategy": strategy,
        "correct": correct,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    db.set_state(agent_id, "results", results)


def compute_score(db: Kaos, agent_id: str) -> dict:
    """Compute aggregate score from results."""
    results = db.get_state(agent_id, "results") or []
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    by_strategy = {}
    for r in results:
        s = r["strategy"]
        by_strategy.setdefault(s, {"total": 0, "correct": 0})
        by_strategy[s]["total"] += 1
        if r["correct"]:
            by_strategy[s]["correct"] += 1

    score = {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0,
        "by_strategy": by_strategy,
    }
    db.set_state(agent_id, "score", score)
    return score


def main():
    import argparse
    parser = argparse.ArgumentParser(description="KAOS AIMO3 Validation Loop")
    parser.add_argument("--experiment", choices=list(EXPERIMENTS.keys()), required=True)
    parser.add_argument("--list", action="store_true", help="List all experiments")
    args = parser.parse_args()

    print("=" * 60)
    print("KAOS v0.3.1 — AIMO3 Validation Loop")
    print("=" * 60)

    db = Kaos(str(DB_PATH))

    if args.list:
        agents = db.list_agents()
        for a in agents:
            score = db.get_state(a["agent_id"], "score")
            print(f"  {a['name']}: {score}")
        db.close()
        return

    exp_name = args.experiment
    exp_config = EXPERIMENTS[exp_name]
    print(f"\nExperiment: {exp_name}")
    print(f"Description: {exp_config['description']}")

    # Register
    agent_id = register_experiment(db, exp_name, exp_config)

    # Load benchmark
    problems = load_benchmark()
    print(f"Benchmark: {len(problems)} AIMO3 reference problems")

    # Generate the per-problem prompts for each strategy
    strategy_assignments = []
    for strat_name, strat_cfg in exp_config["preference_strategies"].items():
        for _ in range(strat_cfg["count"]):
            strategy_assignments.append((strat_name, strat_cfg["preference"]))

    print(f"Strategy assignments: {[(s[0], i) for i, s in enumerate(strategy_assignments)]}")

    # Write problem prompts for agent evaluation
    prompts_out = []
    for p in problems:
        problem_prompts = []
        for strat_name, pref in strategy_assignments:
            problem_prompts.append({
                "strategy": strat_name,
                "user_prompt": f"{p['problem']}\n\n{pref}",
            })
        prompts_out.append({
            "id": p["id"],
            "expected": p["answer"],
            "attempts": problem_prompts,
        })

    # Save to VFS
    prompts_json = json.dumps(prompts_out, indent=2, ensure_ascii=False)
    db.write(agent_id, "/prompts.json", prompts_json.encode("utf-8"))

    # Also save to experiments dir for agent consumption
    outfile = RESULTS_DIR / f"kaos_prompts_{exp_name}.json"
    with io.open(outfile, "w", encoding="utf-8") as f:
        f.write(prompts_json)
    print(f"Prompts written: {outfile}")

    db.set_state(agent_id, "status", "prompts_ready")
    db.checkpoint(agent_id, label=f"{exp_name}-prompts-ready")

    print(f"\nNext: Run validation agents against these prompts.")
    print(f"When results arrive, call record_result() and compute_score().")
    print(f"Agent ID: {agent_id[:16]}...")

    db.close()


if __name__ == "__main__":
    main()
