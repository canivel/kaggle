"""Tool-Integrated Reasoning (TIR) executor.

Implements the core TIR loop:
1. LLM generates reasoning text + Python code block
2. Code is extracted and executed in sandbox
3. Output is fed back to LLM for continuation
4. Repeat until \\boxed{} answer is found or max retries reached

This is the single most impactful technique for math competitions,
roughly doubling performance vs. pure chain-of-thought.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from .sandbox import execute_code, extract_code_blocks, ExecutionResult
from .answer_extraction import extract_answer, extract_boxed
from .prompt_templates import format_tir_continuation


@dataclass
class TIRResult:
    """Result of a single TIR solution attempt."""

    full_text: str  # Complete reasoning trace
    answer: int | None  # Extracted integer answer
    code_executed: bool  # Whether any code was run
    code_succeeded: bool  # Whether code execution succeeded
    has_boxed: bool  # Whether \\boxed{} was found
    n_code_blocks: int  # Number of code blocks executed
    execution_outputs: list[str]  # Outputs from each code execution


def run_tir_single(
    engine,
    prompt: str,
    max_retries: int = 3,
    code_timeout: int = 30,
    max_tokens_per_step: int = 4096,
) -> TIRResult:
    """Run a single TIR solution attempt with code execution loop.

    This generates a single solution path, executing code blocks
    as they're encountered and feeding results back to the model.

    Args:
        engine: InferenceEngine instance.
        prompt: The initial formatted prompt.
        max_retries: Maximum code execution + continuation cycles.
        code_timeout: Timeout per code execution in seconds.
        max_tokens_per_step: Max tokens per generation step.

    Returns:
        TIRResult with the full trace and extracted answer.
    """
    full_text = ""
    code_executed = False
    code_succeeded = False
    n_code_blocks = 0
    execution_outputs = []

    current_prompt = prompt

    for step in range(max_retries + 1):
        # Generate next chunk (stops at ```output or end)
        completions = engine.generate_single(
            current_prompt,
            n_samples=1,
            temperature=0.0,  # greedy for single TIR path
            max_tokens=max_tokens_per_step,
            stop=["```output"],
        )

        if not completions:
            break

        chunk = completions[0]
        full_text += chunk

        # Check if we have a boxed answer already (before code)
        if extract_boxed(full_text) is not None:
            break

        # Extract and execute code blocks from this chunk
        code_blocks = extract_code_blocks(chunk)
        if not code_blocks:
            # No code block and stopped - model is done or stuck
            # Try one more generation without the stop sequence
            if step < max_retries:
                continuation = engine.generate_single(
                    current_prompt + chunk,
                    n_samples=1,
                    temperature=0.0,
                    max_tokens=max_tokens_per_step,
                    stop=None,  # no stop, let it finish
                )
                if continuation:
                    full_text += continuation[0]
            break

        # Execute the last code block (most recent)
        code = code_blocks[-1]
        result = execute_code(code, timeout=code_timeout)
        n_code_blocks += 1
        code_executed = True

        if result.success:
            code_succeeded = True
            output_text = result.stdout.strip() if result.stdout.strip() else "(no output)"
            execution_outputs.append(output_text)
        else:
            output_text = result.stderr.strip()[:500]  # truncate long errors
            execution_outputs.append(f"Error: {output_text}")

        # Build continuation prompt with code output
        output_block = f"\n```output\n{output_text}\n```\n"
        full_text += output_block
        current_prompt = current_prompt + chunk + output_block

    # Extract final answer
    answer = extract_answer(full_text)
    has_boxed = "\\boxed" in full_text

    return TIRResult(
        full_text=full_text,
        answer=answer,
        code_executed=code_executed,
        code_succeeded=code_succeeded,
        has_boxed=has_boxed,
        n_code_blocks=n_code_blocks,
        execution_outputs=execution_outputs,
    )


def run_tir_batch(
    engine,
    prompt: str,
    n_samples: int = 32,
    max_retries: int = 3,
    code_timeout: int = 30,
    temperature: float = 0.7,
    max_tokens: int = 8192,
) -> list[TIRResult]:
    """Run batch TIR: generate N samples, execute code for each.

    This is the main entry point for TIR-based solving. It:
    1. Generates N diverse initial completions
    2. For each completion that has code, executes it
    3. If code output is available, feeds it back for continuation
    4. Extracts answers from all N paths

    Args:
        engine: InferenceEngine instance.
        prompt: The initial formatted prompt.
        n_samples: Number of parallel solution attempts.
        max_retries: Max code execution cycles per solution.
        code_timeout: Timeout per code execution.
        temperature: Sampling temperature for diversity.
        max_tokens: Max tokens per generation.

    Returns:
        List of TIRResult objects, one per sample.
    """
    # Step 1: Generate N initial completions (batched, fast)
    initial_completions = engine.generate_single(
        prompt,
        n_samples=n_samples,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["```output"],
    )

    results = []

    for completion in initial_completions:
        full_text = completion
        code_executed = False
        code_succeeded = False
        n_code_blocks = 0
        execution_outputs = []

        # Check for code blocks
        code_blocks = extract_code_blocks(completion)

        if code_blocks:
            # Execute the code
            code = code_blocks[-1]
            exec_result = execute_code(code, timeout=code_timeout)
            n_code_blocks = 1
            code_executed = True

            if exec_result.success:
                code_succeeded = True
                output_text = exec_result.stdout.strip() if exec_result.stdout.strip() else "(no output)"
            else:
                output_text = exec_result.stderr.strip()[:500]

            execution_outputs.append(output_text)

            # Append output and continue generation
            output_block = f"\n```output\n{output_text}\n```\n"
            full_text += output_block

            # Continue generation after code output (greedy)
            continuation_prompt = prompt + full_text
            for retry in range(max_retries):
                continuation = engine.generate_single(
                    continuation_prompt,
                    n_samples=1,
                    temperature=0.0,
                    max_tokens=max_tokens // 2,
                    stop=["```output"],
                )

                if not continuation or not continuation[0].strip():
                    break

                chunk = continuation[0]
                full_text += chunk

                # Check for another code block
                new_blocks = extract_code_blocks(chunk)
                if new_blocks:
                    code = new_blocks[-1]
                    exec_result = execute_code(code, timeout=code_timeout)
                    n_code_blocks += 1

                    if exec_result.success:
                        code_succeeded = True
                        output_text = exec_result.stdout.strip() if exec_result.stdout.strip() else "(no output)"
                    else:
                        output_text = exec_result.stderr.strip()[:500]

                    execution_outputs.append(output_text)
                    output_block = f"\n```output\n{output_text}\n```\n"
                    full_text += output_block
                    continuation_prompt = prompt + full_text
                else:
                    break

                # If we found an answer, stop
                if extract_boxed(full_text) is not None:
                    break

        answer = extract_answer(full_text)
        has_boxed = "\\boxed" in full_text

        results.append(TIRResult(
            full_text=full_text,
            answer=answer,
            code_executed=code_executed,
            code_succeeded=code_succeeded,
            has_boxed=has_boxed,
            n_code_blocks=n_code_blocks,
            execution_outputs=execution_outputs,
        ))

    return results
