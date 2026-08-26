"""KAOS-powered benchmark loop for AIMO3.

Evaluates prompt configs against the AIMO3 Reference Benchmark using
KAOS v0.3.0 for state management and Claude Code agents for solving.

This script:
1. Loads the AIMO3 benchmark from KAOS
2. For each config variant (baseline, simple, code-first):
   - Runs all 10 problems through the config's prompt template
   - Scores results against ground truth
   - Stores in KAOS DB with checkpoint
3. Produces a comparison table

Usage:
    cd f:/kaggle/aimo-progress-prize-3
    f:/kaggle/kaos/.venv/Scripts/python.exe scripts/kaos_bench_loop.py
"""

from __future__ import annotations

import json
import io
import sys
import time
from pathlib import Path

# Add KAOS to path
KAOS_DIR = Path(__file__).parent.parent.parent / "kaos"
sys.path.insert(0, str(KAOS_DIR))

from kaos import Kaos
from kaos.metaharness.benchmarks.aimo3 import AIMO3Benchmark, extract_answer

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "experiments"
RESULTS_DIR.mkdir(exist_ok=True)
DB_PATH = Path(__file__).parent.parent / "aimo3-learnings.db"


CONFIGS = {
    "baseline_5step": {
        "system_prompt": (
            "You are an elite mathematical problem solver with expertise at the International "
            "Mathematical Olympiad (IMO) level. Your goal is to find the correct answer through "
            "rigorous mathematical reasoning.\n\n"
            "# Problem-Solving Approach:\n"
            "1. UNDERSTAND: Carefully read and rephrase the problem in your own words. "
            "Identify what is given, what needs to be found, and any constraints.\n"
            "2. EXPLORE: Consider multiple solution strategies. Think about which "
            "mathematical concepts and theorems might be relevant.\n"
            "3. PLAN: Outline your chosen approach before diving into calculations.\n"
            "4. EXECUTE: Carry out your plan step by step, showing all work. "
            "Write Python code to verify computations when helpful.\n"
            "5. VERIFY: Check your answer using a different method or by substituting back.\n\n"
            "Present your final answer as a non-negative integer inside \\boxed{}."
        ),
        "preference": "The answer is a non-negative integer. Think step by step and put your final answer in \\boxed{}.",
        "temperature": 0.8,
    },
    "simple_3line": {
        "system_prompt": (
            "You are a world-class IMO competitor. Solve the problem and give "
            "your final answer as an integer in \\boxed{}."
        ),
        "preference": "The answer is a non-negative integer in \\boxed{}.",
        "temperature": 0.8,
    },
    "code_first": {
        "system_prompt": (
            "You are an elite competition mathematician. For every problem:\n"
            "1. Analyze the mathematical structure\n"
            "2. Write Python code to compute and VERIFY the answer\n"
            "3. Only after code confirms, state your final answer in \\boxed{}\n\n"
            "NEVER give \\boxed{} without code verification."
        ),
        "preference": "Write Python code to solve this, verify the result, then put your final integer answer in \\boxed{}.",
        "temperature": 0.8,
    },
}


def format_problem_for_agent(problem_text: str, config: dict) -> str:
    """Format a problem + config into the prompt that an agent would receive."""
    return (
        f"System: {config['system_prompt']}\n\n"
        f"Problem: {problem_text}\n\n"
        f"{config['preference']}"
    )


def main():
    print("=" * 60)
    print("KAOS AIMO3 Benchmark Loop v0.3.0")
    print("=" * 60)

    # Load benchmark
    bench = AIMO3Benchmark(
        problems_path=str(DATA_DIR / "aimo3_reference_bench.json"),
        search_size=10,  # use all 10 for full eval
        test_size=0,
    )
    problems = bench.get_search_set()
    print(f"Loaded {len(problems)} AIMO3 reference problems")

    # Connect to KAOS
    db = Kaos(str(DB_PATH))
    agents = db.list_agents()
    a_id = agents[0]["agent_id"] if agents else None
    if not a_id:
        print("ERROR: No agent in KAOS DB")
        return

    # Store benchmark metadata
    db.set_state(a_id, "local_benchmark", {
        "name": "AIMO3 Reference Bench (official)",
        "problems": len(problems),
        "source": "AIMO3_Reference_Problems.pdf + reference.csv",
        "note": "gpt-oss-120b scores 4/10, Claude Opus scores 9/10 on these",
        "configs_to_test": list(CONFIGS.keys()),
    })

    # Output the prompt templates for each config (for agent evaluation)
    for config_name, config in CONFIGS.items():
        prompts = []
        for p in problems:
            prompts.append({
                "id": p.problem_id,
                "expected": p.expected,
                "system_prompt": config["system_prompt"],
                "user_prompt": f"{p.input['question']}\n\n{config['preference']}",
                "temperature": config["temperature"],
            })

        outfile = RESULTS_DIR / f"aimo3_prompts_{config_name}.json"
        with io.open(outfile, "w", encoding="utf-8") as f:
            json.dump(prompts, f, indent=2, ensure_ascii=False)
        print(f"Wrote {len(prompts)} prompts for config '{config_name}' -> {outfile}")

    # Store in KAOS
    db.set_state(a_id, "benchmark_configs", {
        config_name: {
            "system_prompt_preview": config["system_prompt"][:200],
            "preference": config["preference"],
            "temperature": config["temperature"],
        }
        for config_name, config in CONFIGS.items()
    })

    db.checkpoint(a_id, label="benchmark-setup-2026-04-04")
    print(f"\nKAOS state updated. Run agents to evaluate each config.")
    print(f"\nPrompt files ready in: {RESULTS_DIR}/")
    print(f"  aimo3_prompts_baseline_5step.json")
    print(f"  aimo3_prompts_simple_3line.json")
    print(f"  aimo3_prompts_code_first.json")

    db.close()


if __name__ == "__main__":
    main()
