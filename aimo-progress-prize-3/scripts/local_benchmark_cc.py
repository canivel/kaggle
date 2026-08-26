"""Local benchmark for AIMO3 — designed to be called by Claude Code agents.

Each problem is solved by a Claude Code agent subprocess. Results are collected
into experiments/bench_*.json.

Usage (from Claude Code):
  Run this via Agent tool — it reads problems and writes results.
  NOT designed to be run standalone (no API key needed).
"""

from __future__ import annotations
import json
import io
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "experiments"
RESULTS_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT_BASELINE = (
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
)

PREFERENCE = "The answer is a non-negative integer. Think step by step and put your final answer in \\boxed{}."


def load_problems(path: str | None = None, n: int | None = None, offset: int = 0) -> list[dict]:
    p = Path(path) if path else DATA_DIR / "benchmark_v1.json"
    with io.open(p, encoding="utf-8") as f:
        problems = json.load(f)
    problems = problems[offset:]
    if n:
        problems = problems[:n]
    return problems


def save_results(results: list[dict], config_name: str = "baseline") -> Path:
    import time
    ts = time.strftime("%Y%m%d_%H%M%S")
    outfile = RESULTS_DIR / f"bench_{config_name}_{ts}.json"
    with io.open(outfile, "w", encoding="utf-8") as f:
        json.dump({
            "config": config_name,
            "backend": "claude_code_agent",
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    return outfile
