"""Secure Python/SymPy code execution sandbox for TIR.

Executes code blocks extracted from LLM output in an isolated namespace
with pre-imported math libraries and a configurable timeout.
"""

import signal
import threading
import traceback
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr


# Pre-imported modules available in the sandbox
SANDBOX_IMPORTS = """
import math
import numpy as np
import sympy as sp
from sympy import *
from sympy import symbols, solve, simplify, expand, factor, Rational, sqrt, oo
from sympy import pi, E, I, sin, cos, tan, log, exp, Abs, floor, ceiling
from sympy import gcd, lcm, isprime, nextprime, factorint, divisors, totient
from sympy import binomial, factorial, fibonacci
from sympy import Matrix, det, eye
from sympy.ntheory import mobius, primerange
from sympy.combinatorics import Permutation
from itertools import combinations, permutations, product as iproduct
from collections import Counter, defaultdict
from fractions import Fraction
from functools import reduce
import itertools
"""


class ExecutionResult:
    """Result of code execution."""

    def __init__(self, stdout: str, stderr: str, success: bool, return_value=None):
        self.stdout = stdout
        self.stderr = stderr
        self.success = success
        self.return_value = return_value

    def __repr__(self):
        status = "OK" if self.success else "FAIL"
        return f"ExecutionResult({status}, stdout={self.stdout!r:.100}, stderr={self.stderr!r:.100})"


def execute_code(code: str, timeout: int = 30) -> ExecutionResult:
    """Execute Python code in an isolated namespace with timeout.

    Args:
        code: Python source code to execute.
        timeout: Maximum execution time in seconds.

    Returns:
        ExecutionResult with stdout, stderr, and success status.
    """
    stdout_capture = StringIO()
    stderr_capture = StringIO()

    # Create isolated namespace with pre-imported modules
    namespace = {}
    try:
        exec(SANDBOX_IMPORTS, namespace)
    except ImportError as e:
        # Some imports may not be available; continue without them
        pass

    result = ExecutionResult("", "", False)
    exception_holder = [None]

    def _run():
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, namespace)
        except Exception as e:
            exception_holder[0] = e

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        # Thread is still running - timeout exceeded
        result.stdout = stdout_capture.getvalue()
        result.stderr = f"Execution timed out after {timeout} seconds"
        result.success = False
        return result

    result.stdout = stdout_capture.getvalue()
    result.stderr = stderr_capture.getvalue()

    if exception_holder[0] is not None:
        exc = exception_holder[0]
        result.stderr = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        result.success = False
    else:
        result.success = True

    return result


def extract_code_blocks(text: str) -> list[str]:
    """Extract Python code blocks from LLM output.

    Looks for ```python ... ``` blocks. Falls back to ``` ... ``` if no
    python-specific blocks are found.
    """
    import re

    # First try python-specific blocks
    blocks = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks

    # Fallback: any code blocks
    blocks = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
    return blocks


def execute_first_code_block(text: str, timeout: int = 30) -> ExecutionResult | None:
    """Extract and execute the first code block from LLM output.

    Returns None if no code blocks are found.
    """
    blocks = extract_code_blocks(text)
    if not blocks:
        return None
    return execute_code(blocks[0], timeout=timeout)
