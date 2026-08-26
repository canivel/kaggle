"""Run Meta-Harness search using Claude Code CLI + local vLLM.

Claude Code (subscription) proposes harness code.
Local vLLM (DeepSeek-R1-Qwen3-8B) evaluates math problems.
kaos orchestrates everything in SQLite.

Usage:
    # Ensure vLLM is running:
    docker compose up -d

    # Run search:
    uv run python scripts/run_meta_harness_claude.py \
        --corpus data/math_corpus.jsonl \
        --iterations 5

    # With custom vLLM endpoint:
    uv run python scripts/run_meta_harness_claude.py \
        --corpus data/math_corpus.jsonl \
        --vllm-endpoint http://localhost:8000/v1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, "f:/kaggle/kaos")

import httpx
from kaos import Kaos


# ── Seed harnesses ──────────────────────────────────────────

SEED_NO_RETRIEVAL = '''
"""No retrieval baseline."""
def run(problem):
    question = problem["question"]
    prompt = f"Solve step by step. Put answer in \\\\boxed{{}}\\n\\nProblem: {question}\\n\\nSolution:"
    return {"prompt": prompt, "context_tokens": len(prompt.split())}
'''

SEED_BM25 = '''
"""BM25 retrieval with word overlap."""
def run(problem):
    question = problem["question"]
    corpus = problem.get("corpus", [])
    query_words = set(question.lower().split())
    scored = []
    for doc in corpus:
        doc_words = set(doc["question"].lower().split())
        overlap = len(query_words & doc_words)
        total = len(query_words | doc_words) or 1
        scored.append((overlap / total, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    top_k = [doc for _, doc in scored[:3]]
    examples = ""
    for ex in top_k:
        examples += f"Example: {ex['question']}\\nSolution: {ex.get('solution', 'N/A')[:400]}\\nAnswer: {ex.get('answer', 'N/A')}\\n\\n"
    prompt = f"Reference examples:\\n{examples}\\nSolve step by step. Put answer in \\\\boxed{{}}.\\n\\nProblem: {question}\\n\\nSolution:"
    return {"prompt": prompt, "context_tokens": len(prompt.split())}
'''

SEED_DOMAIN_ROUTING = '''
"""Domain-aware retrieval (Meta-Harness inspired)."""
import re, math

GEO = ["triangle","circle","angle","perpendicular","inscribed","tangent","polygon"]
NT = ["prime","divisible","modulo","gcd","remainder","congruent","coprime"]
COMBO = ["how many","number of ways","permutation","combinat","probability"]

def classify(q):
    q = q.lower()
    scores = {"geometry": sum(1 for k in GEO if k in q),
              "number_theory": sum(1 for k in NT if k in q),
              "combinatorics": sum(1 for k in COMBO if k in q)}
    for d in ["combinatorics","geometry","number_theory"]:
        if scores[d] >= 2: return d
    best = max(scores, key=scores.get)
    return best if scores[best] >= 1 else "algebra"

SOL_MAX = {"combinatorics":800, "geometry":300, "number_theory":400, "algebra":400}

def run(problem):
    question = problem["question"]
    corpus = problem.get("corpus", [])
    domain = classify(question)
    # Filter by domain
    domain_corpus = [d for d in corpus if classify(d["question"]) == domain]
    if len(domain_corpus) < 3:
        domain_corpus = corpus
    # Score
    query_words = set(question.lower().split())
    scored = []
    for doc in domain_corpus:
        doc_words = set(doc["question"].lower().split())
        overlap = len(query_words & doc_words)
        scored.append((overlap / math.sqrt(max(len(doc_words),1)), doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    top_k = [doc for _, doc in scored[:3]]
    max_c = SOL_MAX.get(domain, 400)
    examples = ""
    for ex in top_k:
        sol = ex.get("solution", "")[:max_c]
        examples += f"[{domain.upper()}] Problem: {ex['question']}\\nSolution: {sol}\\nAnswer: {ex.get('answer', 'N/A')}\\n\\n"
    prompt = f"Domain: {domain}\\nReference:\\n{examples}\\nSolve step by step. Put answer in \\\\boxed{{}}.\\n\\nProblem: {question}\\n\\nSolution:"
    return {"prompt": prompt, "context_tokens": len(prompt.split())}
'''


# ── Claude Code proposer ────────────────────────────────────

def claude_propose(archive_text: str, iteration: int) -> str:
    """Use Claude Code CLI to propose a new harness."""
    system = (
        "You write Python retrieval harnesses for math problems. "
        "The harness must define run(problem) -> {'prompt': str, 'context_tokens': int}. "
        "problem has 'question' (str) and 'corpus' (list of dicts with question/solution/answer). "
        "Use techniques: domain routing, math-aware tokenization, difficulty filtering, "
        "Jaccard diversity, adaptive truncation. Output ONLY Python code."
    )

    prompt = f"Iteration {iteration}. Archive:\n{archive_text}\n\nPropose an improved harness:"

    result = subprocess.run(
        ["claude", "--print", "--system-prompt", system, "--output-format", "text"],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        print(f"  Claude CLI error: {result.stderr[:200]}")
        return ""

    response = result.stdout.strip()
    # Extract Python code
    if "```python" in response:
        code = response.split("```python")[1].split("```")[0]
    elif "```" in response:
        code = response.split("```")[1].split("```")[0]
    else:
        code = response

    return code.strip()


# ── vLLM evaluator ──────────────────────────────────────────

def vllm_solve(prompt: str, endpoint: str, model: str) -> str:
    """Send prompt to vLLM and get response."""
    resp = httpx.post(
        f"{endpoint}/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 4096,
        },
        timeout=120.0,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def extract_boxed(text: str) -> str | None:
    """Extract answer from \\boxed{...}."""
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    bs = text.find("{", idx)
    if bs == -1:
        return None
    depth, end = 0, bs
    for i in range(bs, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    return text[bs + 1 : end].strip()


def score_harness(
    harness_code: str,
    problems: list[dict],
    corpus: list[dict],
    endpoint: str,
    model: str,
) -> dict:
    """Evaluate a harness on a set of problems.

    Returns {"accuracy": float, "context_cost": float, "errors": int}.
    """
    # Compile harness
    namespace = {}
    try:
        exec(harness_code, namespace)
    except Exception as e:
        return {"accuracy": 0.0, "context_cost": 0.0, "errors": len(problems), "error": str(e)}

    run_fn = namespace.get("run")
    if not callable(run_fn):
        return {"accuracy": 0.0, "context_cost": 0.0, "errors": len(problems), "error": "No run() function"}

    correct = 0
    total_tokens = 0
    errors = 0

    for prob in problems:
        try:
            result = run_fn({"question": prob["question"], "corpus": corpus})
            prompt = result["prompt"]
            total_tokens += result.get("context_tokens", 0)

            # Send to vLLM
            response = vllm_solve(prompt, endpoint, model)
            predicted = extract_boxed(response)

            if predicted is not None:
                # Normalize comparison
                try:
                    if float(predicted) == float(prob["answer"]):
                        correct += 1
                except (ValueError, TypeError):
                    if predicted.strip() == str(prob["answer"]).strip():
                        correct += 1
        except Exception as e:
            errors += 1
            print(f"    Error on problem: {e}")

    n = len(problems)
    return {
        "accuracy": correct / n if n > 0 else 0.0,
        "context_cost": total_tokens / n if n > 0 else 0.0,
        "errors": errors,
        "correct": correct,
        "total": n,
    }


# ── Main search loop ────────────────────────────────────────

async def main(
    corpus_path: str | None,
    iterations: int,
    vllm_endpoint: str,
    vllm_model: str,
    search_size: int,
):
    db = Kaos("aimo3-meta-search.db")

    # Load corpus
    corpus = []
    if corpus_path and Path(corpus_path).exists():
        with open(corpus_path) as f:
            for line in f:
                if line.strip():
                    corpus.append(json.loads(line))
        print(f"Loaded corpus: {len(corpus)} problems")
    else:
        print("WARNING: No corpus, using synthetic data")
        corpus = [
            {"question": "What is 6*9?", "answer": "54", "solution": "6*9=54"},
            {"question": "Solve x^2=81", "answer": "9", "solution": "x=sqrt(81)=9"},
        ]

    # Split into search set and eval corpus
    import random
    rng = random.Random(42)
    shuffled = list(corpus)
    rng.shuffle(shuffled)
    search_problems = shuffled[:search_size]
    retrieval_corpus = shuffled[search_size:search_size + 500]

    print(f"Search problems: {len(search_problems)}")
    print(f"Retrieval corpus: {len(retrieval_corpus)}")

    # Check vLLM is up
    print(f"\nChecking vLLM at {vllm_endpoint}...")
    try:
        resp = httpx.get(f"{vllm_endpoint}/models", timeout=10)
        models = resp.json()
        print(f"  vLLM ready: {[m['id'] for m in models['data']]}")
    except Exception as e:
        print(f"  vLLM not ready: {e}")
        print("  Start it with: docker compose up -d")
        db.close()
        return

    # Seeds
    seeds = [
        ("no_retrieval", SEED_NO_RETRIEVAL),
        ("bm25", SEED_BM25),
        ("domain_routing", SEED_DOMAIN_ROUTING),
    ]

    # Initialize search agent
    search_id = db.spawn("aimo3-meta-harness")
    results = []

    print("\n" + "=" * 60)
    print("META-HARNESS SEARCH: Math Retrieval Optimization")
    print("=" * 60)

    # Evaluate seeds
    print("\n--- Evaluating seed harnesses ---")
    for name, code in seeds:
        print(f"\n  Evaluating: {name}")
        scores = score_harness(code, search_problems, retrieval_corpus, vllm_endpoint, vllm_model)
        results.append({"name": name, "code": code, "scores": scores, "iteration": 0})
        print(f"    accuracy={scores['accuracy']:.3f}, cost={scores['context_cost']:.0f}, errors={scores['errors']}")

        # Store in kaos
        db.write(search_id, f"/harnesses/{name}/source.py", code.encode())
        db.write(search_id, f"/harnesses/{name}/scores.json", json.dumps(scores).encode())

    # Main loop: Claude proposes, vLLM evaluates
    for iteration in range(1, iterations + 1):
        print(f"\n--- Iteration {iteration}/{iterations} ---")

        # Build archive summary for Claude
        archive_lines = []
        for r in results:
            s = r["scores"]
            archive_lines.append(
                f"Harness '{r['name']}' (iter {r['iteration']}): "
                f"accuracy={s['accuracy']:.3f}, context_cost={s['context_cost']:.0f}, errors={s['errors']}"
            )
            archive_lines.append(f"  Code preview: {r['code'][:200]}...")
        archive_text = "\n".join(archive_lines)

        # Claude proposes
        print("  Proposing via Claude Code...")
        new_code = claude_propose(archive_text, iteration)
        if not new_code:
            print("  Claude returned empty, skipping")
            continue

        # Validate
        try:
            ns = {}
            exec(new_code, ns)
            if not callable(ns.get("run")):
                print("  No run() function, skipping")
                continue
        except Exception as e:
            print(f"  Invalid code: {e}")
            continue

        # Evaluate
        name = f"claude_iter{iteration}"
        print(f"  Evaluating: {name}")
        scores = score_harness(new_code, search_problems, retrieval_corpus, vllm_endpoint, vllm_model)
        results.append({"name": name, "code": new_code, "scores": scores, "iteration": iteration})
        print(f"    accuracy={scores['accuracy']:.3f}, cost={scores['context_cost']:.0f}, errors={scores['errors']}")

        # Store
        db.write(search_id, f"/harnesses/{name}/source.py", new_code.encode())
        db.write(search_id, f"/harnesses/{name}/scores.json", json.dumps(scores).encode())

        # Checkpoint
        db.checkpoint(search_id, label=f"iter-{iteration}")

    # Final report
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    results.sort(key=lambda r: r["scores"]["accuracy"], reverse=True)
    for r in results:
        s = r["scores"]
        print(f"  {r['name']:25s} acc={s['accuracy']:.3f} cost={s['context_cost']:.0f} err={s['errors']}")

    # Export best
    best = results[0]
    out_path = Path("data/best_harness.py")
    out_path.write_text(best["code"])
    print(f"\nBest harness: {best['name']} (accuracy={best['scores']['accuracy']:.3f})")
    print(f"Exported to: {out_path}")

    db.close()
    print(f"All data in aimo3-meta-search.db")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Meta-Harness search: Claude + vLLM")
    parser.add_argument("--corpus", default="data/math_corpus.jsonl")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--vllm-endpoint", default="http://localhost:8000/v1")
    parser.add_argument("--vllm-model", default="hxac/DeepSeek-R1-0528-Qwen3-8B-AWQ-4bit")
    parser.add_argument("--search-size", type=int, default=20, help="Problems for evaluation")
    args = parser.parse_args()

    asyncio.run(main(args.corpus, args.iterations, args.vllm_endpoint, args.vllm_model, args.search_size))
