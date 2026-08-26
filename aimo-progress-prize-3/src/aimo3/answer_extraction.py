"""Extract integer answers from LLM output text.

Handles multiple output formats:
1. \\boxed{N} LaTeX notation (primary)
2. Code execution output (from TIR)
3. "The answer is N" natural language patterns
4. Last integer in text (fallback)

All answers are normalized to int and taken mod 100000.
"""

import re


ANSWER_MOD = 100_000


def extract_boxed(text: str) -> int | None:
    """Extract answer from \\boxed{...} notation, handling nested braces."""
    # Find the last \\boxed occurrence (most likely to be the final answer)
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None

    # Find the opening brace
    brace_start = text.find("{", idx)
    if brace_start == -1:
        return None

    # Match balanced braces
    depth = 0
    end = brace_start
    for i in range(brace_start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    content = text[brace_start + 1 : end].strip()
    return _parse_number(content)


def extract_from_code_output(text: str) -> int | None:
    """Extract answer from code execution output block.

    Looks for ```output ... ``` blocks and extracts the last number.
    """
    # Match ```output ... ``` blocks
    pattern = r"```output\s*(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return None

    # Take the last output block
    last_output = matches[-1].strip()
    return _parse_number(last_output)


def extract_natural_language(text: str) -> int | None:
    """Extract answer from natural language patterns."""
    patterns = [
        r"(?:the\s+)?(?:final\s+)?answer\s+is\s*[:\s]*(\-?\d+(?:\.\d+)?)",
        r"(?:therefore|thus|hence|so)\s*,?\s*(?:the\s+answer\s+is\s+)?(\-?\d+(?:\.\d+)?)",
        r"answer\s*[=:]\s*(\-?\d+(?:\.\d+)?)",
        r"=\s*\\boxed\{(\-?\d+(?:\.\d+)?)\}",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return _parse_number(matches[-1])
    return None


def extract_last_integer(text: str) -> int | None:
    """Fallback: extract the last integer in the text."""
    # Find all integers (possibly negative)
    matches = re.findall(r"(?<![.\d])(-?\d+)(?![.\d])", text)
    if matches:
        return _parse_number(matches[-1])
    return None


def _parse_fraction(s: str) -> int | None:
    r"""Try to evaluate \frac{N}{M} -> N // M if M divides N exactly."""
    match = re.match(r"\\frac\s*\{([^}]+)\}\s*\{([^}]+)\}", s.strip())
    if not match:
        return None
    try:
        num = int(match.group(1).strip())
        den = int(match.group(2).strip())
        if den != 0 and num % den == 0:
            return num // den
    except ValueError:
        pass
    return None


def _parse_number(s: str) -> int | None:
    """Parse a string to integer, handling various formats."""
    s = s.strip()

    # Try \frac{N}{M} evaluation before stripping
    frac_result = _parse_fraction(s)
    if frac_result is not None:
        return frac_result

    # Remove LaTeX formatting
    s = s.replace("\\,", "").replace("\\;", "").replace("\\!", "")
    s = s.replace(",", "")  # Remove thousands separators
    s = re.sub(r"\\text\{.*?\}", "", s)
    s = re.sub(r"\\mathrm\{.*?\}", "", s)

    # Try direct integer parse
    try:
        return int(s)
    except ValueError:
        pass

    # Try float → int (for "123.0" style)
    try:
        f = float(s)
        if f == int(f) and not (f != f):  # not NaN
            return int(f)
    except (ValueError, OverflowError):
        pass

    # Try extracting first number from string
    match = re.search(r"(-?\d+(?:\.\d+)?)", s)
    if match:
        try:
            f = float(match.group(1))
            if f == int(f):
                return int(f)
        except (ValueError, OverflowError):
            pass

    return None


def extract_answer(text: str, default: int | None = None) -> int | None:
    """Extract integer answer from model output using multiple strategies.

    Tries in order:
    1. \\boxed{} notation
    2. Code output blocks
    3. Natural language patterns
    4. Last integer in text

    Returns the answer mod 100000, or `default` if no answer found.
    Pass default=0 to always return an integer (safe for competition submission).
    """
    for extractor in [
        extract_boxed,
        extract_from_code_output,
        extract_natural_language,
        extract_last_integer,
    ]:
        result = extractor(text)
        if result is not None:
            return result % ANSWER_MOD

    return default
