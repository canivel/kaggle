"""Local evaluation on AMC/AIME benchmark problems.

Runs the full TIR pipeline on a set of validation problems
and reports accuracy by problem type.

Usage:
    uv run python scripts/run_local_eval.py --model models/numinamath-7b-tir/
    uv run python scripts/run_local_eval.py --model models/openmath-nemotron-14b-kaggle/ --n-samples 16
"""

import argparse
import json
import time
from pathlib import Path

# Validation problems: (problem_text, expected_answer, problem_type)
# These are public AMC/AIME problems for testing the pipeline
VALIDATION_PROBLEMS = [
    # AMC 10A 2024 #25 - Number Theory
    (
        "How many positive integers $n \\leq 600$ can be written in the form "
        "$\\lfloor x \\rfloor + \\lfloor 2x \\rfloor + \\lfloor 3x \\rfloor = n$ "
        "for some real number $x$?",
        "399",
        "number_theory",
    ),
    # AIME I 2024 #1 - Algebra
    (
        "Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop "
        "afterwards. When she walks at a constant speed of $s$ kilometers per hour, the "
        "walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she "
        "walks at $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, "
        "including $t$ minutes spent in the coffee shop. Suppose Aya walks at "
        "$s+\\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes "
        "her, including the $t$ minutes spent in the coffee shop.",
        "204",
        "algebra",
    ),
    # AIME I 2024 #3 - Combinatorics
    (
        "Alice and Bob play the following game. A stack of $n$ tokens lies before them. "
        "The players take turns with Alice going first. On each turn, the player removes "
        "either $1$ token or $4$ tokens from the stack. Whoever removes the last token "
        "wins. Find the number of positive integers $n$ less than or equal to $2024$ for "
        "which there exists a strategy for Bob that guarantees that Bob will win the game "
        "regardless of Alice's strategy.",
        "809",
        "combinatorics",
    ),
    # AIME I 2024 #8 - Number Theory
    (
        "Eight circles of radius $34$ are sequentially tangent, and two of the circles "
        "are tangent to $AB$ and $BC$ of triangle $ABC$, respectively. $2024$ circles of "
        "radius $1$ can be arranged in the same manner. The inradius of triangle $ABC$ "
        "can be written as $\\frac{m}{n}$, where $m$ and $n$ are relatively prime positive "
        "integers. Find $m+n$.",
        "197",
        "geometry",
    ),
    # Simple test: 2 + 2
    (
        "What is $2 + 2$?",
        "4",
        "algebra",
    ),
]


def run_evaluation(model_path: str, n_samples: int = 8, max_problems: int = -1):
    """Run evaluation on validation problems."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from aimo3.solver import AIMOSolver

    solver = AIMOSolver(
        model_path=model_path,
        n_samples=n_samples,
        total_time_limit=7200,  # 2 hour local eval limit
        n_problems=len(VALIDATION_PROBLEMS),
    )
    solver.setup()

    problems = VALIDATION_PROBLEMS
    if max_problems > 0:
        problems = problems[:max_problems]

    correct = 0
    total = len(problems)
    results = []

    for i, (problem, expected, ptype) in enumerate(problems):
        print(f"\n{'='*60}")
        print(f"Problem {i+1}/{total} [{ptype}]:")
        print(f"  {problem[:100]}...")
        print(f"  Expected: {expected}")

        answer = solver.solve(problem)
        expected_int = int(expected) % 100_000
        is_correct = answer == expected_int

        status = "CORRECT" if is_correct else "WRONG"
        print(f"  Got: {answer} [{status}]")

        if is_correct:
            correct += 1

        results.append({
            "problem_idx": i,
            "problem_type": ptype,
            "expected": expected_int,
            "predicted": answer,
            "correct": is_correct,
        })

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS: {correct}/{total} correct ({100*correct/total:.1f}%)")

    # Per-type breakdown
    type_results = {}
    for r in results:
        t = r["problem_type"]
        if t not in type_results:
            type_results[t] = {"correct": 0, "total": 0}
        type_results[t]["total"] += 1
        if r["correct"]:
            type_results[t]["correct"] += 1

    print("\nPer-type:")
    for t, counts in sorted(type_results.items()):
        pct = 100 * counts["correct"] / counts["total"]
        print(f"  {t:20s}: {counts['correct']}/{counts['total']} ({pct:.0f}%)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Local eval for AIMO3")
    parser.add_argument("--model", required=True, help="Path to model weights")
    parser.add_argument("--n-samples", type=int, default=8, help="Samples per problem")
    parser.add_argument("--max-problems", type=int, default=-1, help="Max problems to eval")
    args = parser.parse_args()

    run_evaluation(args.model, args.n_samples, args.max_problems)


if __name__ == "__main__":
    main()
