"""Local AIMO3 testing using kaos + Claude CLI.

Adapts the kaos meta_harness_math pattern to test our solver innovations
locally using Claude CLI as a proxy for GPT-OSS-120B.

Tests:
1. Domain routing effectiveness
2. Phase split vs flat solving
3. Verify cascade accuracy
4. Different domain hint formulations

Usage:
    uv run python scripts/local_aimo_test.py
    uv run python scripts/local_aimo_test.py --problems data/math_corpus.jsonl --max 10
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, "f:/kaggle/kaos")
from kaos import Kaos

# ── AIME/AMC problems with known answers ──

TEST_PROBLEMS = [
    {"question": "Find the remainder when 2^100 is divided by 7.", "answer": "2", "domain": "number_theory"},
    {"question": "How many 4-digit positive integers have digits that sum to 10?", "answer": "219", "domain": "combinatorics"},
    {"question": "In triangle ABC, AB=5, BC=12, CA=13. What is the area?", "answer": "30", "domain": "geometry"},
    {"question": "Solve for x: x^2 - 5x + 6 = 0. Give the larger root.", "answer": "3", "domain": "algebra"},
    {"question": "Find the remainder when 17^2024 is divided by 5.", "answer": "1", "domain": "number_theory"},
    {"question": "How many ways can 5 people sit in a row?", "answer": "120", "domain": "combinatorics"},
    {"question": "A circle has radius 7. What is its area, rounded to the nearest integer?", "answer": "154", "domain": "geometry"},
    {"question": "If f(x) = 2x+3, find f(f(2)).", "answer": "17", "domain": "algebra"},
]

DOMAIN_HINTS = {
    "number_theory": "This is a number theory problem. Use pow(base,exp,mod) for modular arithmetic. Check patterns in small cases first.",
    "geometry": "This is a geometry problem. Set up coordinates. Use the Pythagorean theorem and area formulas.",
    "combinatorics": "This is a combinatorics problem. Enumerate small cases. Use factorial and binomial.",
    "algebra": "This is an algebra problem. Use sympy.solve() or factoring. Verify by substitution.",
}


def call_claude(prompt, timeout=60):
    claude_bin = shutil.which("claude") or shutil.which("claude.cmd") or r"C:\Users\dcani\AppData\Roaming\npm\claude.cmd"
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    try:
        result = subprocess.run(
            [claude_bin, "--print", "--output-format", "text"],
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=timeout,
            env=env,
        )
        return result.stdout.decode("utf-8", errors="replace").strip()
    except Exception as e:
        return f"Error: {e}"


def extract_answer(response):
    """Extract integer answer from response."""
    # Try boxed
    ms = re.findall(r'\\boxed\{(\d+)\}', response)
    if ms: return ms[-1]
    # Try "answer is N"
    ms = re.findall(r'answer\s+is\s*[:\s]*(\d+)', response, re.IGNORECASE)
    if ms: return ms[-1]
    # Last number
    ms = re.findall(r'\b(\d+)\b', response)
    if ms: return ms[-1]
    return None


def test_harness(harness_fn, problems, label):
    """Test a harness function on problems."""
    correct = 0
    total = len(problems)
    for prob in problems:
        prompt = harness_fn(prob)
        response = call_claude(prompt, timeout=90)
        predicted = extract_answer(response)
        is_correct = str(predicted) == str(prob["answer"])
        if is_correct:
            correct += 1
        status = "OK" if is_correct else "WRONG"
        print(f"  [{status}] {prob['domain']}: pred={predicted} actual={prob['answer']}")
    accuracy = correct / total if total else 0
    print(f"  {label}: {correct}/{total} = {accuracy:.1%}")
    return {"label": label, "correct": correct, "total": total, "accuracy": accuracy}


def harness_bare(prob):
    """No domain hint."""
    return f"Solve: {prob['question']}\nGive ONLY the integer answer."


def harness_domain(prob):
    """With domain hint."""
    hint = DOMAIN_HINTS.get(prob["domain"], "")
    return f"{hint}\n\nSolve: {prob['question']}\nGive ONLY the integer answer."


def harness_domain_code(prob):
    """Domain hint + code instruction."""
    hint = DOMAIN_HINTS.get(prob["domain"], "")
    return (
        f"{hint}\n\n"
        f"Solve: {prob['question']}\n"
        f"Write Python code to compute the answer. Show the code and its output.\n"
        f"Give ONLY the integer answer at the end."
    )


def harness_verify(prob):
    """Domain hint + code + verify."""
    hint = DOMAIN_HINTS.get(prob["domain"], "")
    return (
        f"{hint}\n\n"
        f"Solve: {prob['question']}\n"
        f"Step 1: Solve with code.\n"
        f"Step 2: Verify with a DIFFERENT method.\n"
        f"Give ONLY the integer answer."
    )


def main(max_problems=None):
    db = Kaos("aimo3-local-test.db")
    agent_id = db.spawn("local-aimo-test")

    problems = TEST_PROBLEMS[:max_problems] if max_problems else TEST_PROBLEMS

    print("=" * 60)
    print(f"LOCAL AIMO3 HARNESS TEST ({len(problems)} problems)")
    print("=" * 60)

    results = []

    print("\n--- Bare (no hints) ---")
    r = test_harness(harness_bare, problems, "bare")
    results.append(r)

    print("\n--- Domain Routing ---")
    r = test_harness(harness_domain, problems, "domain")
    results.append(r)

    print("\n--- Domain + Code ---")
    r = test_harness(harness_domain_code, problems, "domain_code")
    results.append(r)

    print("\n--- Domain + Code + Verify ---")
    r = test_harness(harness_verify, problems, "domain_verify")
    results.append(r)

    print("\n" + "=" * 60)
    print("SUMMARY")
    for r in sorted(results, key=lambda x: x["accuracy"], reverse=True):
        print(f"  {r['label']:20s}: {r['correct']}/{r['total']} = {r['accuracy']:.1%}")

    # Store in kaos
    db.set_state(agent_id, "results", results)
    db.set_state(agent_id, "timestamp", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    db.close()

    # Also store in main kaos DB
    main_db = Kaos("aimo3-learnings.db")
    main_agent = main_db.list_agents()[0]["agent_id"]
    main_db.set_state(main_agent, "local_harness_test", {
        "results": results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "Claude CLI proxy on AIME-level problems",
    })
    main_db.close()

    print(f"\nResults in aimo3-local-test.db")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=None)
    args = parser.parse_args()
    main(args.max)
