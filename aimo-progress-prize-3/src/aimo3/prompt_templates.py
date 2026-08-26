"""Prompt templates for AIMO3 math problem solving.

Provides system prompts and problem-type-specific instructions for
Tool-Integrated Reasoning (TIR).
"""

SYSTEM_PROMPT = """\
You are a world-class mathematician solving competition-level problems.
You MUST use Python code to verify your reasoning and compute the answer.

Rules:
1. Think step by step about the problem.
2. Write Python code using sympy, numpy, or pure Python to solve it.
3. Put your code in a ```python ... ``` block.
4. After the code executes, you will see the output in a ```output ... ``` block.
5. Based on the output, provide your final answer inside \\boxed{}.
6. The answer must be a non-negative integer.
7. If the problem asks for an answer mod N, compute it.

Always verify your answer with code. Never guess."""


TIR_SYSTEM_PROMPT = """\
You are a world-class mathematician. Solve the given problem step by step.

IMPORTANT INSTRUCTIONS:
- Write Python code to help solve the problem. Use ```python ... ``` blocks.
- After code execution, you will see results in ```output ... ``` blocks.
- Use sympy for symbolic computation, numpy for numerical work.
- You may write multiple code blocks, each building on previous results.
- After reaching the answer, put it inside \\boxed{N} where N is an integer.
- If the problem says "find the remainder when X is divided by Y", compute X % Y.
- Double-check your answer with a verification code block before giving \\boxed{}."""


PROBLEM_TYPE_INSTRUCTIONS = {
    "algebra": """
This is an algebra problem. Useful strategies:
- Set up equations with sympy.symbols() and solve with sympy.solve()
- For polynomial problems, use factor(), expand(), roots()
- For inequalities, consider boundary cases
- For functional equations, substitute specific values to find patterns
- Verify by substituting your answer back into the original equation""",
    "combinatorics": """
This is a combinatorics problem. Useful strategies:
- Start by computing small cases to find a pattern
- Use itertools.combinations, itertools.permutations for enumeration
- Use sympy.binomial() for counting
- For probability, compute numerator and denominator separately
- Verify by brute-force enumeration on small instances""",
    "geometry": """
This is a geometry problem. Useful strategies:
- Set up coordinates: place key points at convenient locations
- Use sympy geometry module or direct coordinate computation
- For triangle problems: law of cosines, law of sines, area formulas
- For circle problems: power of a point, radical axes
- Compute distances, angles, areas numerically and verify with code
- When possible, use trigonometric identities""",
    "number_theory": """
This is a number theory problem. Useful strategies:
- Use sympy.factorint() for prime factorization
- Use sympy.gcd(), sympy.lcm() for GCD/LCM
- Use pow(base, exp, mod) for modular exponentiation
- For "find the remainder" problems, work in modular arithmetic
- Use Chinese Remainder Theorem (sympy.crt) when applicable
- Enumerate small cases first to find patterns
- Use sympy.isprime(), sympy.nextprime(), sympy.primerange()""",
    "default": """
Solve this problem step by step. Use Python code to compute and verify.""",
}


def format_problem_prompt(
    problem: str, problem_type: str = "default", include_system: bool = True
) -> str:
    """Format a math problem into a complete prompt for the LLM.

    Args:
        problem: The LaTeX math problem text.
        problem_type: One of 'algebra', 'combinatorics', 'geometry',
                      'number_theory', or 'default'.
        include_system: Whether to prepend the system prompt.

    Returns:
        Formatted prompt string.
    """
    type_instruction = PROBLEM_TYPE_INSTRUCTIONS.get(
        problem_type, PROBLEM_TYPE_INSTRUCTIONS["default"]
    )

    parts = []
    if include_system:
        parts.append(TIR_SYSTEM_PROMPT)
    parts.append(type_instruction.strip())
    parts.append(f"\nProblem:\n{problem}\n\nSolution:")

    return "\n\n".join(parts)


def format_tir_continuation(
    problem: str,
    previous_reasoning: str,
    code_output: str,
    was_error: bool = False,
) -> str:
    """Format a TIR continuation prompt after code execution.

    Used when the model needs to continue reasoning after seeing
    the output of its code execution.

    Args:
        problem: Original problem text.
        previous_reasoning: Model's reasoning so far.
        code_output: Output from code execution (stdout or error).
        was_error: Whether the code execution resulted in an error.

    Returns:
        Continuation prompt string.
    """
    if was_error:
        output_block = f"```output\nError: {code_output}\n```\n\nThe code had an error. Fix the code and try again."
    else:
        output_block = f"```output\n{code_output}\n```"

    return f"{previous_reasoning}\n{output_block}\n"


def build_chat_messages(
    problem: str,
    problem_type: str = "default",
    retrieved_examples: str = "",
) -> list[dict[str, str]]:
    """Build chat-format messages for models that use chat templates.

    Args:
        problem: The LaTeX math problem text.
        problem_type: Problem category.
        retrieved_examples: Formatted few-shot examples from retrieval.

    Returns:
        List of message dicts with 'role' and 'content'.
    """
    type_instruction = PROBLEM_TYPE_INSTRUCTIONS.get(
        problem_type, PROBLEM_TYPE_INSTRUCTIONS["default"]
    )

    user_parts = [type_instruction.strip()]
    if retrieved_examples:
        user_parts.append(retrieved_examples)
    user_parts.append(f"Problem:\n{problem}")

    return [
        {"role": "system", "content": TIR_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]
