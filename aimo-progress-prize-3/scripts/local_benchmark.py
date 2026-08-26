"""Local benchmark harness for AIMO3.

Evaluates solve configurations against benchmark_v1.json.
Supports multiple backends:
  - local vLLM (DeepSeek-R1-8B via docker-compose)
  - Claude CLI (claude --print)
  - OpenAI-compatible API (any endpoint)

Usage:
  # Quick test (5 problems, Claude CLI):
  python scripts/local_benchmark.py --backend claude --n 5

  # Full benchmark (local vLLM):
  python scripts/local_benchmark.py --backend vllm --endpoint http://localhost:8000/v1

  # Custom endpoint (RunPod, etc.):
  python scripts/local_benchmark.py --backend api --endpoint http://1.2.3.4:8000/v1 --model gpt-oss --n 25

  # Test a specific config variant:
  python scripts/local_benchmark.py --backend claude --config two-phase --n 10
"""

from __future__ import annotations

import argparse
import json
import io
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "experiments"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Answer extraction (same as Kaggle notebook) ──────────────────────────────

def extract_answer(text: str, default: int | None = None) -> int | None:
    """Extract integer answer from model response."""
    # Priority 1: \boxed{N}
    boxed = re.findall(r'\\boxed\{(\d+)\}', text)
    if boxed:
        return int(boxed[-1]) % 100000

    # Priority 2: explicit answer markers
    for pat in [
        r'[Ff]inal [Aa]nswer[:\s]+(\d+)',
        r'[Aa]nswer[:\s]+(\d+)',
        r'[Tt]he answer is[:\s]+(\d+)',
        r'= (\d+)\s*$',
    ]:
        m = re.search(pat, text, re.MULTILINE)
        if m:
            return int(m.group(1)) % 100000

    # Priority 3: last standalone integer
    ints = re.findall(r'\b(\d+)\b', text)
    if ints:
        return int(ints[-1]) % 100000

    return default


# ── Backends ──────────────────────────────────────────────────────────────────

def call_claude(system_prompt: str, user_prompt: str) -> str:
    """Call Anthropic API (Claude) directly."""
    try:
        import anthropic
    except ImportError:
        sys.exit("pip install anthropic")

    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var
    try:
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return msg.content[0].text
    except Exception as e:
        return f"ERROR: {e}"


def call_api(system_prompt: str, user_prompt: str, endpoint: str, model: str,
             temperature: float = 0.8, max_tokens: int = 4096) -> str:
    """Call OpenAI-compatible API (vLLM, RunPod, etc.)."""
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("pip install openai")

    client = OpenAI(base_url=endpoint, api_key="unused")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        return f"ERROR: {e}"


# ── Configs to test ───────────────────────────────────────────────────────────

CONFIGS = {
    "baseline": {
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
        "preference_prompt": (
            "The answer is a non-negative integer. Think step by step and put "
            "your final answer in \\boxed{}."
        ),
        "attempts": 1,
        "temperature": 0.8,
    },
    "simple": {
        "system_prompt": (
            "You are a world-class IMO competitor. Solve the problem and give "
            "your final answer as an integer in \\boxed{}."
        ),
        "preference_prompt": "The answer is a non-negative integer in \\boxed{}.",
        "attempts": 1,
        "temperature": 0.8,
    },
    "two-phase": {
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
        "preference_prompt": (
            "The answer is a non-negative integer. Think step by step and put "
            "your final answer in \\boxed{}."
        ),
        "attempts": 1,
        "temperature": 0.8,
        "note": "Same prompt as baseline — the two-phase logic is in the harness, not the prompt",
    },
}


# ── Main evaluation loop ─────────────────────────────────────────────────────

def evaluate(problems: list[dict], config: dict, backend: str,
             endpoint: str = "", model: str = "", n_attempts: int = 1) -> list[dict]:
    """Run evaluation on problems with given config."""
    results = []
    system_prompt = config["system_prompt"]
    pref = config["preference_prompt"]
    temp = config.get("temperature", 0.8)

    for i, prob in enumerate(problems):
        pid = prob.get("id", f"p{i}")
        user_prompt = f"{prob['problem']} {pref}"
        expected = prob["answer"]

        answers = []
        responses = []
        t0 = time.time()

        for attempt in range(n_attempts):
            if backend == "claude":
                resp = call_claude(system_prompt, user_prompt)
            elif backend in ("vllm", "api"):
                resp = call_api(system_prompt, user_prompt, endpoint, model, temp)
            else:
                resp = f"ERROR: unknown backend {backend}"

            ans = extract_answer(resp)
            answers.append(ans)
            responses.append(resp[:500])  # truncate for storage

        elapsed = time.time() - t0

        # Majority vote
        valid = [a for a in answers if a is not None]
        if valid:
            final = Counter(valid).most_common(1)[0][0]
        else:
            final = 0

        correct = (final == expected)
        result = {
            "id": pid,
            "source": prob.get("source", ""),
            "domain": prob.get("domain", ""),
            "difficulty": prob.get("difficulty", 0),
            "expected": expected,
            "predicted": final,
            "correct": correct,
            "attempts": len(answers),
            "valid_answers": len(valid),
            "all_answers": answers,
            "elapsed_s": round(elapsed, 1),
        }
        results.append(result)

        status = "OK" if correct else "XX"
        print(f"  [{i+1:2d}/{len(problems)}] {status} {pid}: "
              f"pred={final} exp={expected} ({elapsed:.1f}s)")

    return results


def summary(results: list[dict]) -> dict:
    """Compute summary statistics."""
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    by_domain = {}
    by_difficulty = {}
    for r in results:
        d = r["domain"]
        by_domain.setdefault(d, {"total": 0, "correct": 0})
        by_domain[d]["total"] += 1
        if r["correct"]:
            by_domain[d]["correct"] += 1

        diff_bucket = "easy" if r["difficulty"] <= 5 else ("medium" if r["difficulty"] <= 7 else "hard")
        by_difficulty.setdefault(diff_bucket, {"total": 0, "correct": 0})
        by_difficulty[diff_bucket]["total"] += 1
        if r["correct"]:
            by_difficulty[diff_bucket]["correct"] += 1

    return {
        "score": f"{correct}/{total}",
        "accuracy": round(correct / total, 4) if total else 0,
        "by_domain": {k: f"{v['correct']}/{v['total']}" for k, v in by_domain.items()},
        "by_difficulty": {k: f"{v['correct']}/{v['total']}" for k, v in by_difficulty.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="AIMO3 local benchmark")
    parser.add_argument("--backend", choices=["claude", "vllm", "api"], default="claude")
    parser.add_argument("--endpoint", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="gpt-oss")
    parser.add_argument("--config", choices=list(CONFIGS.keys()), default="baseline")
    parser.add_argument("--n", type=int, default=None, help="Number of problems (default: all)")
    parser.add_argument("--attempts", type=int, default=1, help="Attempts per problem")
    parser.add_argument("--benchmark", default=str(DATA_DIR / "benchmark_v1.json"))
    parser.add_argument("--offset", type=int, default=0, help="Skip first N problems")
    args = parser.parse_args()

    # Load problems
    with io.open(args.benchmark, encoding="utf-8") as f:
        problems = json.load(f)

    if args.offset:
        problems = problems[args.offset:]
    if args.n:
        problems = problems[:args.n]

    config = CONFIGS[args.config]
    print(f"Benchmark: {len(problems)} problems")
    print(f"Config: {args.config} | Backend: {args.backend} | Attempts: {args.attempts}")
    print(f"Temp: {config.get('temperature', 0.8)}")
    print("=" * 60)

    results = evaluate(problems, config, args.backend,
                       args.endpoint, args.model, args.attempts)

    s = summary(results)
    print("\n" + "=" * 60)
    print(f"Score: {s['score']} ({s['accuracy']:.1%})")
    print(f"By domain: {s['by_domain']}")
    print(f"By difficulty: {s['by_difficulty']}")

    # Save results
    ts = time.strftime("%Y%m%d_%H%M%S")
    outfile = RESULTS_DIR / f"bench_{args.config}_{args.backend}_{ts}.json"
    with io.open(outfile, "w", encoding="utf-8") as f:
        json.dump({
            "config": args.config,
            "backend": args.backend,
            "model": args.model,
            "attempts": args.attempts,
            "summary": s,
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {outfile}")


if __name__ == "__main__":
    main()
