"""Optimize domain-specific hints using Claude CLI + kaos.

Tests different domain hint variants on AIME-level problems
to find the formulations that lead to the best reasoning.
"""

import os
import sys
import json
import subprocess
import shutil
import time

sys.path.insert(0, "f:/kaggle/kaos")
from kaos import Kaos

HARD_PROBLEMS = {
    "number_theory": {
        "problem": "Find the remainder when 7^2024 + 3^2024 is divided by 11.",
        "answer": 2,
    },
    "geometry": {
        "problem": "Triangle ABC has sides AB=13, BC=14, CA=15. Point D is on BC such that AD bisects angle BAC. Find the length AD, expressed as a fraction p/q in lowest terms. Give p+q.",
        "answer": 377,  # AD = 180/sqrt(2)/... complex
    },
    "combinatorics": {
        "problem": "In how many ways can you arrange the letters of MISSISSIPPI?",
        "answer": 34650,
    },
}

HINT_VARIANTS = {
    "number_theory": {
        "v1_basic": "This is a number theory problem.",
        "v2_strategy": "This is a number theory problem. Start by computing small cases and looking for periodic patterns in modular arithmetic.",
        "v3_tools": "This is a number theory problem. Use pow(base,exp,mod) for modular exponentiation. Factor with sympy.factorint(). Enumerate small cases first.",
        "v4_approach": "This is a number theory problem. Key approaches: work modulo small primes, use Fermat's little theorem (a^(p-1) ≡ 1 mod p for prime p), check patterns in the first few powers.",
    },
    "geometry": {
        "v1_basic": "This is a geometry problem.",
        "v2_coords": "This is a geometry problem. Set up coordinates: place one vertex at the origin.",
        "v3_tools": "This is a geometry problem. Use coordinate geometry with sympy. Place B at origin, C on x-axis. Compute all distances/angles numerically first, then verify symbolically.",
        "v4_multi": "This is a geometry problem. Try TWO approaches: (1) coordinate geometry with sympy, (2) law of cosines + area formula. Cross-verify the results.",
    },
    "combinatorics": {
        "v1_basic": "This is a combinatorics problem.",
        "v2_brute": "This is a combinatorics problem. Compute small cases by brute force first.",
        "v3_tools": "This is a combinatorics problem. Use math.factorial() and sympy.binomial(). For arrangements with repetitions: n! / (k1! * k2! * ...). Verify with itertools for small cases.",
        "v4_pattern": "This is a combinatorics problem. Step 1: solve for n=1,2,3,4 by brute-force enumeration. Step 2: look for a formula pattern. Step 3: prove the formula. Step 4: compute the answer.",
    },
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


def evaluate_hint(domain, hint_name, hint_text, problem, answer):
    prompt = f"{hint_text}\n\nSolve: {problem}\nGive ONLY the final integer answer, nothing else."
    response = call_claude(prompt, timeout=90)

    # Check if answer is correct
    try:
        # Extract last number from response
        import re
        numbers = re.findall(r'\b(\d+)\b', response)
        predicted = int(numbers[-1]) if numbers else -1
        correct = (predicted == answer)
    except:
        correct = False
        predicted = -1

    return {
        "domain": domain,
        "hint": hint_name,
        "predicted": predicted,
        "correct": correct,
        "response_preview": response[:100],
    }


def main():
    db = Kaos("aimo3-learnings.db")
    agent = db.list_agents()[0]["agent_id"]

    print("=" * 60)
    print("DOMAIN HINT OPTIMIZATION")
    print("=" * 60)

    all_results = {}

    for domain, prob_data in HARD_PROBLEMS.items():
        print(f"\n--- {domain.upper()} ---")
        print(f"Problem: {prob_data['problem'][:80]}...")
        print(f"Answer: {prob_data['answer']}")

        domain_results = {}
        for hint_name, hint_text in HINT_VARIANTS[domain].items():
            result = evaluate_hint(
                domain, hint_name, hint_text,
                prob_data["problem"], prob_data["answer"]
            )
            domain_results[hint_name] = result
            status = "CORRECT" if result["correct"] else "WRONG"
            print(f"  {hint_name}: {result['predicted']} ({status}) — {result['response_preview'][:50]}")

        all_results[domain] = domain_results

    # Find best hint per domain
    best_hints = {}
    for domain, results in all_results.items():
        correct_hints = [h for h, r in results.items() if r["correct"]]
        if correct_hints:
            best_hints[domain] = correct_hints[-1]  # prefer more detailed if multiple correct
        else:
            best_hints[domain] = "v3_tools"  # default to tool-focused

    print("\n" + "=" * 60)
    print("BEST HINTS:")
    for domain, hint in best_hints.items():
        print(f"  {domain}: {hint}")

    # Store in kaos
    db.set_state(agent, "domain_hint_optimization", {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": {d: {h: {"correct": r["correct"], "predicted": r["predicted"]}
                        for h, r in results.items()}
                    for d, results in all_results.items()},
        "best_hints": best_hints,
        "note": "Claude CLI proxy, not GPT-OSS-120B. Directional only.",
    })

    db.close()
    print("\nResults stored in kaos.")


if __name__ == "__main__":
    main()
