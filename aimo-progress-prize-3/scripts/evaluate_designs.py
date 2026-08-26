"""Evaluate notebook design choices using Claude CLI as a mathematical reasoning proxy.

Tests specific design decisions:
1. Does domain routing help? (provide strategy hints vs not)
2. Does disagreement context help Phase 2? (informed vs blind)
3. Does Python-mandatory retry help? (nudge vs no nudge)

Uses Claude to simulate the model's reasoning on sample problems.
"""

import os
import sys
import json
import subprocess
import shutil
import time

sys.path.insert(0, "f:/kaggle/kaos")
from kaos import Kaos

SAMPLE_PROBLEMS = [
    {
        "problem": "Find the remainder when 2^2024 is divided by 17.",
        "answer": 1,
        "domain": "number_theory",
    },
    {
        "problem": "How many ways can you place 8 non-attacking rooks on an 8x8 chessboard?",
        "answer": 40320,
        "domain": "combinatorics",
    },
    {
        "problem": "In triangle ABC, AB=13, BC=14, CA=15. Find the area.",
        "answer": 84,
        "domain": "geometry",
    },
]


def call_claude(prompt: str, timeout: int = 90) -> str:
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


def test_domain_routing():
    """Test if domain hints improve reasoning."""
    print("=== Testing Domain Routing ===")

    hints = {
        "number_theory": "Start by working modulo small primes. Use pow(base,exp,mod).",
        "combinatorics": "Compute small cases first. Use brute force to verify.",
        "geometry": "Set up coordinates with one vertex at origin.",
    }

    for prob in SAMPLE_PROBLEMS:
        # Without hint
        prompt_bare = f"Solve: {prob['problem']}. Give ONLY the integer answer."
        resp_bare = call_claude(prompt_bare)

        # With domain hint
        hint = hints.get(prob["domain"], "")
        prompt_hint = f"{hint}\n\nSolve: {prob['problem']}. Give ONLY the integer answer."
        resp_hint = call_claude(prompt_hint)

        print(f"\n  [{prob['domain']}] {prob['problem'][:60]}...")
        print(f"    Without hint: {resp_bare[:80]}")
        print(f"    With hint:    {resp_hint[:80]}")
        print(f"    Correct: {prob['answer']}")


def test_disagreement_context():
    """Test if informing about disagreement improves accuracy."""
    print("\n=== Testing Disagreement Context ===")

    prob = SAMPLE_PROBLEMS[0]  # remainder problem

    # Without context
    prompt_bare = f"Solve: {prob['problem']}. Give ONLY the integer answer."
    resp_bare = call_claude(prompt_bare)

    # With disagreement context
    prompt_ctx = (
        f"NOTE: Previous attempts gave conflicting answers: 1, 4, 16. "
        f"At least some are wrong. Be extra careful and verify with computation.\n\n"
        f"Solve: {prob['problem']}. Give ONLY the integer answer."
    )
    resp_ctx = call_claude(prompt_ctx)

    print(f"  Without context: {resp_bare[:80]}")
    print(f"  With context:    {resp_ctx[:80]}")
    print(f"  Correct: {prob['answer']}")


def main():
    print("AIMO3 Design Evaluation (via Claude CLI)")
    print("=" * 60)

    test_domain_routing()
    test_disagreement_context()

    # Store results in kaos
    db = Kaos("aimo3-learnings.db")
    agent = db.list_agents()[0]["agent_id"]
    db.set_state(agent, "local_design_eval", {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tests_run": ["domain_routing", "disagreement_context"],
        "note": "Claude CLI proxy, not actual GPT-OSS-120B. Directional only.",
    })
    db.close()

    print("\nDone. Results are directional only (Claude != GPT-OSS-120B).")


if __name__ == "__main__":
    main()
