"""Claude Code SDK integration for meta-harness proposer.

Uses the `claude` CLI (subscription-based, no API key needed) as the
proposer LLM for generating harness candidates. The local vLLM handles
math problem evaluation only.

Architecture:
    Claude Code (proposer) → generates harness code
    Local vLLM (evaluator) → solves math problems to score harnesses
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path


def claude_generate(
    prompt: str,
    system: str = "",
    max_tokens: int = 16384,
    temperature: float = 0.7,
) -> str:
    """Call Claude Code CLI to generate a response.

    Uses `claude --print` for non-interactive single-turn generation.
    This uses your Claude Code subscription, not an API key.

    Args:
        prompt: The user prompt.
        system: Optional system prompt.
        max_tokens: Max response tokens.
        temperature: Sampling temperature.

    Returns:
        The assistant's response text.
    """
    cmd = [
        "claude",
        "--print",  # Non-interactive, print response and exit
        "--output-format", "text",
    ]

    if system:
        cmd.extend(["--system-prompt", system])

    cmd.extend(["--max-tokens", str(max_tokens)])

    # Pass prompt via stdin
    result = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        timeout=300,  # 5 minute timeout
    )

    if result.returncode != 0:
        raise RuntimeError(f"Claude CLI failed: {result.stderr}")

    return result.stdout.strip()


def propose_harness(
    archive_summary: str,
    frontier_summary: str,
    iteration: int,
    benchmark_name: str = "math_rag",
) -> str:
    """Use Claude Code to propose a new retrieval harness.

    Args:
        archive_summary: Summary of prior harnesses and their scores.
        frontier_summary: Current Pareto frontier.
        iteration: Current search iteration.
        benchmark_name: Name of the benchmark.

    Returns:
        Python source code for the new harness.
    """
    system = """You are an expert at designing retrieval-augmented generation (RAG)
systems for mathematical reasoning. You are participating in a meta-harness search
to find the optimal retrieval strategy for solving IMO-level math problems.

Your task: Given the archive of prior harness candidates and their scores,
propose an improved harness. The harness must define a `run(problem)` function
that takes a dict with 'question' and 'corpus' keys, and returns a dict with
'prompt' (str) and 'context_tokens' (int) keys.

Key techniques that work well (from the Meta-Harness paper):
- Domain-specific routing (geometry, combinatorics, number theory, algebra)
- Math-aware BM25 tokenization (preserve LaTeX tokens like \\frac, \\sum)
- Solution-indexed cross-field retrieval for geometry proofs
- Difficulty-based filtering
- Greedy Jaccard diversity deduplication
- Adaptive solution truncation per domain
- OR-max dual-query BM25 (primary + named math terms)

Output ONLY the Python source code. No markdown, no explanation outside the code."""

    prompt = f"""Meta-Harness Search — Iteration {iteration}

## Current Pareto Frontier
{frontier_summary}

## Archive of Prior Harnesses
{archive_summary}

## Task
Propose a new harness that improves on the current frontier. The harness must
define `run(problem)` that returns {{"prompt": str, "context_tokens": int}}.

The `problem` dict contains:
- "question": the math problem text (may contain LaTeX)
- "corpus": list of dicts with "question", "solution", "answer" keys

Focus on improving accuracy while keeping context_cost reasonable.
Generate the complete Python source code:"""

    response = claude_generate(prompt, system=system)

    # Clean up response - extract just the Python code
    if "```python" in response:
        code = response.split("```python")[1].split("```")[0]
    elif "```" in response:
        code = response.split("```")[1].split("```")[0]
    else:
        code = response

    return code.strip()


def evaluate_with_vllm(
    prompt: str,
    vllm_endpoint: str = "http://localhost:8000/v1",
    model: str = "hxac/DeepSeek-R1-0528-Qwen3-8B-AWQ-4bit",
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> str:
    """Send a math problem prompt to the local vLLM for evaluation.

    Args:
        prompt: The formatted prompt with retrieval context + problem.
        vllm_endpoint: Local vLLM OpenAI-compatible endpoint.
        model: Model name served by vLLM.
        temperature: Sampling temperature.
        max_tokens: Max response tokens.

    Returns:
        The model's response text.
    """
    import httpx

    response = httpx.post(
        f"{vllm_endpoint}/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=120.0,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]
