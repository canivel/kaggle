"""Local validation of AIMO3 notebook improvements.

Uses Claude Code CLI to evaluate how well different system prompts
and strategies would work on sample AIME/AMC problems.

This doesn't test the actual model — it tests the SCAFFOLDING:
- Does the prompt encourage code verification?
- Does the follow-up recover answers?
- Does the voting logic pick the right answer?

Usage:
    uv run python scripts/validate_locally.py
"""

import os
import sys
import json
import subprocess
import time
import shutil
from pathlib import Path

sys.path.insert(0, "f:/kaggle/kaos")
from kaos import Kaos

# Sample AIME problems with known answers for validation
VALIDATION_PROBLEMS = [
    {
        "problem": "Every morning Aya goes for a 9-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of s km/h, the walk takes her 4 hours, including t minutes spent in the coffee shop. When she walks at s+2 km/h, the walk takes her 2 hours and 24 minutes, including t minutes in the coffee shop. Find the number of minutes the walk takes her when walking at s+1/2 km/h, including t minutes in the coffee shop.",
        "answer": 204,
        "type": "algebra"
    },
    {
        "problem": "Alice and Bob play the following game. A stack of n tokens lies before them. The players take turns with Alice going first. On each turn, the player removes either 1 token or 4 tokens from the stack. Whoever removes the last token wins. Find the number of positive integers n less than or equal to 2024 for which there exists a strategy for Bob that guarantees that Bob will win.",
        "answer": 809,
        "type": "combinatorics"
    },
    {
        "problem": "Find the number of triples of nonneg integers (a,b,c) satisfying a+b+c=300 and a^2b+a^2c+b^2a+b^2c+c^2a+c^2b=6000000.",
        "answer": 601,
        "type": "algebra"
    },
]


def evaluate_prompt_with_claude(problem: str, system_prompt: str, answer: int) -> dict:
    """Use Claude CLI to evaluate if a system prompt leads to correct reasoning."""
    claude_bin = shutil.which("claude") or shutil.which("claude.cmd") or r"C:\Users\dcani\AppData\Roaming\npm\claude.cmd"

    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    env.pop("CLAUDE_CODE", None)

    prompt = (
        f"You are evaluating a math competition system prompt. "
        f"Given this system prompt:\n\n{system_prompt[:500]}\n\n"
        f"And this problem:\n{problem}\n\n"
        f"The correct answer is {answer}.\n\n"
        f"Rate 1-10: How likely is this system prompt to lead the model to the correct answer? "
        f"Consider: Does it encourage code verification? Domain-specific strategies? "
        f"Respond with ONLY a JSON object: {{\"score\": N, \"reason\": \"...\"}}"
    )

    try:
        result = subprocess.run(
            [claude_bin, "--print", "--output-format", "text"],
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=60,
            env=env,
        )
        response = result.stdout.decode("utf-8", errors="replace").strip()
        # Try to parse JSON
        try:
            return json.loads(response)
        except:
            return {"score": 5, "reason": response[:200]}
    except Exception as e:
        return {"score": 0, "reason": str(e)}


def validate_config(config_name: str, system_prompt: str) -> dict:
    """Validate a config against all validation problems."""
    scores = []
    for prob in VALIDATION_PROBLEMS:
        result = evaluate_prompt_with_claude(
            prob["problem"], system_prompt, prob["answer"]
        )
        scores.append({
            "problem_type": prob["type"],
            "score": result.get("score", 0),
            "reason": result.get("reason", ""),
        })
        print(f"  {prob['type']}: {result.get('score', 0)}/10 - {result.get('reason', '')[:80]}")

    avg = sum(s["score"] for s in scores) / len(scores) if scores else 0
    return {"config": config_name, "avg_score": avg, "details": scores}


def main():
    db = Kaos("aimo3-learnings.db")
    agent = db.list_agents()[0]["agent_id"]

    # Load current configs to compare
    configs = {
        "v18_base": (
            'You are a world-class International Mathematical Olympiad (IMO) competitor. '
            'The final answer must be a non-negative integer between 0 and 99999. '
            'You must place the final integer answer inside \\boxed{}.'
        ),
        "v27_optimal": (
            'You are an elite mathematical problem solver with expertise at the International '
            'Mathematical Olympiad (IMO) level. Your goal is to find the correct answer through '
            'rigorous mathematical reasoning.\n\n'
            '# Problem-Solving Approach:\n'
            '1. UNDERSTAND: Carefully read and rephrase the problem.\n'
            '2. EXPLORE: Consider multiple strategies.\n'
            '3. PLAN: Select the most promising approach.\n'
            '4. EXECUTE: You MUST use the Python tool for ALL computations.\n'
            '5. VERIFY: Check answer with an INDEPENDENT method.\n\n'
            '# CRITICAL Rules:\n'
            '- NEVER do arithmetic in your head — always use Python code\n'
            '- ALWAYS verify your answer with a second computation before \\boxed{}\n'
            '- For geometry: ALWAYS set up coordinates FIRST\n'
            'Place your final answer inside \\boxed{}, e.g., \\boxed{42}'
        ),
    }

    print("=" * 60)
    print("LOCAL PROMPT VALIDATION")
    print("=" * 60)

    results = []
    for name, prompt in configs.items():
        print(f"\n--- {name} ---")
        result = validate_config(name, prompt)
        results.append(result)
        print(f"  Average: {result['avg_score']:.1f}/10")

    # Store in kaos
    db.set_state(agent, "local_validation", {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": results,
    })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    for r in sorted(results, key=lambda x: x["avg_score"], reverse=True):
        print(f"  {r['config']:20s}: {r['avg_score']:.1f}/10")

    db.close()


if __name__ == "__main__":
    main()
