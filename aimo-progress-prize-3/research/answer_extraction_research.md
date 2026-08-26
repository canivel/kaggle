# AIMO3 Answer Extraction Research

**Research date**: 2026-04-01
**Competition**: AI Mathematical Olympiad - Progress Prize 3

---

## Summary

Our existing `answer_extraction.py` is already technically superior to every top public notebook's extraction logic. The key finding is that all top notebooks (44/50, 43/50, 42/50, 40/50) use the same canonical regex pattern that contains a subtle Python bug, while our code uses a string-based `rfind` approach that is immune to this bug. The highest-impact improvements available are not in extraction itself but in the generation loop: multi-turn follow-up prompting (Hui Kang's 40/50 baseline), verification of top candidates (ans-verifys), and handling of `\frac{N}{M}` in `_parse_number`.

---

## 1. What All Top Notebooks Use (and the Bug They Share)

### The Canonical Pattern

Every top notebook except ZaynYu uses this two-pattern approach:

```python
def _scan_for_answer(self, text: str) -> int | None:
    pattern = r'\\boxed\s*\{\s*([0-9,]+)\s*\}'
    matches = re.findall(pattern, text)
    if matches:
        try:
            clean_value = matches[-1].replace(',', '')
            value = int(clean_value)
            if 0 <= value <= 99999:
                return value
        except ValueError:
            pass
    pattern = r'final\s+answer\s+is\s*([0-9,]+)'
    matches = re.findall(pattern, text, re.IGNORECASE)
    if matches:
        try:
            clean_value = matches[-1].replace(',', '')
            value = int(clean_value)
            if 0 <= value <= 99999:
                return value
        except ValueError:
            pass
    return None
```

This pattern is copy-pasted across notebooks by: nihilisticneuralnet (44/50, 43/50), datasciencegrad (42/50), jonathanchan, shelterw (15/15 AIME), and others.

### The Regex Backslash Bug

In Python, `r'\\boxed'` is a raw string containing two characters: a backslash `\` and the letter `b` followed by `oxed`. In regex, `\b` is a **word boundary** assertion, not a literal backslash. So `r'\\boxed'` compiles to a word-boundary-before-`oxed` pattern, NOT "a literal backslash followed by `boxed`".

This means the canonical pattern:
- Matches `boxed{42}` where `b` follows a word boundary (e.g., in `$\boxed{42}$` or at line start)
- Fails silently for `\boxed{42}` if the character preceding `b` is something that doesn't create a word boundary

In practice, it seems to work because the GPT-OSS-120B model typically writes `\boxed{42}` in a LaTeX math environment (e.g., `$$\boxed{42}$$`) where `\b` word boundary still matches. But it is fragile.

### ZaynYu's Fix (40/50)

ZaynYu (40/50) independently discovered the same issue and fixed it:

```python
pattern = r'oxed\s*\{\s*([0-9,\s]+)\s*\}'
```

By dropping the `\\b` prefix entirely and matching from `oxed`, the pattern works regardless of what precedes `\boxed`. This is the cleanest fix for the regex approach.

### Our Existing Approach (Better Than Both)

Our `extract_boxed()` uses `text.rfind("\\boxed")` — a Python string search, not a regex. This is immune to the backslash issue entirely. It then uses balanced brace counting to find the matching `}`, which handles nested braces that neither the canonical regex nor ZaynYu's pattern can handle (e.g., `\boxed{\frac{a}{b}}`).

---

## 2. Edge Cases Our Code Handles vs Misses

### Handled Correctly (our code does better than top notebooks)

| Edge case | Top notebooks | Our code |
|-----------|---------------|----------|
| `\boxed{42,000}` (comma-separated) | Handles (replace comma) | Handles (remove comma in `_parse_number`) |
| `\boxed{\frac{a+b}{c+d}}` (nested braces) | FAILS (regex) | Handles (balanced brace counting) |
| Multiple `\boxed{}` in text | Takes last match | Takes last occurrence (rfind) |
| `\boxed{-42}` (negative) | FAILS (`[0-9,]+` pattern) | Handles (negative in `extract_last_integer`) |
| `\text{...}` inside boxed | FAILS | Handles (strips via `_parse_number`) |
| `\mathrm{...}` inside boxed | FAILS | Handles (strips via `_parse_number`) |
| LaTeX spacing `\,` `\;` `\!` | FAILS | Handles |

### Gaps in Our Code

| Gap | Impact | Fix |
|-----|--------|-----|
| `\frac{N}{M}` inside boxed — we extract just N | Medium — occurs for geometry/combinatorics answers that happen to be fractions before being reduced to integer | See Section 4 |
| Returns `None` instead of a default integer | Medium — downstream code must handle `None` or problems score 0 | Use a non-zero default (e.g., `8687`) |
| No follow-up prompting on missing answer | High — prevents "no answer" entirely | See Section 3 |
| Numbers > 99999 accepted in natural-language / last-integer extractors | Low | Apply `% ANSWER_MOD` consistently (already done at call site in `extract_answer`) |

---

## 3. Hui Kang's Multi-Turn Follow-Up Prompting (Highest Impact)

Hui Kang's streaming notebook (basis of the competition baseline) includes multi-turn follow-up prompting when the model fails to produce a `\boxed{}` answer. This is the highest-impact improvement because it eliminates most "no answer" cases at the source.

### How It Works

After generating a response that contains no valid boxed answer, inject a follow-up user message:

```python
if not is_valid_answer_string(extract_boxed_text(text_response)):
    # Choose follow-up based on remaining time budget
    if iteration == 0 and time_remaining > 90:
        user_follow_up = (
            "The answer is expected to be an integer between 0 and 99999 inclusive. "
            "Please make an educated guess (e.g. lower bound, upper bound, current best answer, ...) "
            "and put your final answer in \\boxed{}."
        )
    elif iteration == 1 and time_remaining > 30:
        user_follow_up = (
            "The answer is expected to be an integer between 0 and 99999 inclusive. "
            "Please guess a reasonable answer and put in \\boxed{} as soon as possible."
        )
    else:
        user_follow_up = (
            "The answer is expected to be an integer between 0 and 99999 inclusive. "
            "Place your final answer in \\boxed{}. Do not guess the answer."
        )
elif int(boxed_text) <= 10:
    # Sanity check: suspiciously small answer
    user_follow_up = "Are you sure that is the answer? Do not guess the answer."
elif iteration == 0 and token_count < 3200:
    # Short response — prompt for verification
    user_follow_up = "Have you verified your answer?"
```

### Key Observations

1. The follow-up is tiered by remaining time: aggressive guessing allowed only if enough time remains; otherwise demand a boxed answer.
2. A special check for answers <= 10 (suspiciously small for 0-99999 range) triggers a re-verification prompt.
3. Short responses (< 3200 tokens) trigger a verification prompt — short responses correlate with incorrect reasoning.
4. **Fallback default**: Hui Kang returns `12453` when `force_answer=True` and no boxed answer is found after all follow-ups.

### Why This Belongs in the Generation Loop, Not Extraction

This technique cannot be implemented purely in `answer_extraction.py` — it requires access to the model's generation API to inject follow-up messages. It belongs in the inference/generation code that calls the model.

---

## 4. ans-verifys Verification Step (Moderate Impact)

The `ans-verifys` notebook (amanatar, ~41/50) adds a verification step after collecting candidate answers from majority voting:

### Flow

1. Generate N=16+ solutions per problem
2. Collect all extracted answers
3. Find candidates with >= 2 votes
4. Sort candidates by average entropy (low entropy = high confidence)
5. For each candidate (in entropy order), ask the model: "Is this answer correct? Reply CORRECT or WRONG."
6. Return the first verified answer
7. If verification fails for all candidates, fall back to entropy-weighted selection

### Verification Prompt

```python
prompt = (
    f"Problem:\n{problem}\n\n"
    f"Proposed answer: {answer}\n\n"
    "Check the answer carefully.\n"
    "Reply with only ONE word:\n"
    "CORRECT or WRONG"
)
response = model.generate(prompt, temperature=0.0, max_tokens=5)
return "CORRECT" in response and "WRONG" not in response
```

### Trade-offs

- Uses model capacity for verification instead of additional sampling
- Most effective when candidates are close (e.g., 42 vs 43) and majority vote is uncertain
- Cheap: temperature=0.0, max_tokens=5
- Diminishing returns if N is already high (majority vote becomes reliable at N=32+)

---

## 5. No-Answer Defaults Across Top Notebooks

| Notebook | Score | No-answer default | Strategy |
|----------|-------|-------------------|----------|
| nihilisticneuralnet | 44/50 | `0` | Hardcoded |
| nihilisticneuralnet | 43/50 | `0` | Hardcoded |
| datasciencegrad | 42/50 | `0` | Hardcoded |
| ZaynYu | 40/50 | `8687` | Config `default_answer` |
| Hui Kang | 38/50 baseline | `12453` | Force-answer follow-up first, then default |
| ans-verifys | ~41/50 | `0` | Hardcoded |
| **Our code** | - | `None` | Returns None (caller decides) |

### Recommendation

Return `0` as the default (matches the majority approach and is a neutral guess in the 0-99999 range). The non-zero defaults (8687, 12453) are arbitrary choices that happen to be right if the actual answer matches — there is no principled reason to prefer them.

---

## 6. Hui Kang's Log-Weighting in Voting

Hui Kang applies a log-weight modifier when aggregating majority votes:

```python
modified_counter[value] = (
    modified_counter.get(value, 0.0) + math.log(1.25 + abs(value)) * count
)
```

This down-weights small answers (e.g., `0`, `1`, `2`) relative to larger ones. The comment in the code is "smaller answers seems to be wrong" — empirical observation that the model sometimes outputs trivially small answers as a failure mode.

Practical note: the effect is small (log(1.25) ≈ 0.22, log(100001.25) ≈ 11.5), so this matters mainly when two answers tie in raw count and one is much smaller.

---

## 7. Recommended Improvements to `answer_extraction.py`

### Priority 1: `\frac{N}{M}` Evaluation in `_parse_number`

Currently, `_parse_number` strips content from `\text{}` and `\mathrm{}` but does not evaluate `\frac{N}{M}`. If the boxed content is `\frac{42}{1}` or a fraction that reduces to an integer, we extract just `42` (the first number found), which happens to be correct for `\frac{42}{1}` but wrong for `\frac{6}{2}` (should be 3, we extract 6).

Fix:

```python
import re

def _parse_fraction(s: str) -> int | None:
    """Try to evaluate \\frac{N}{M} -> N // M if M divides N exactly."""
    match = re.match(r'\\frac\s*\{([^}]+)\}\s*\{([^}]+)\}', s.strip())
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
```

Add this call before the current integer-parse attempts in `_parse_number`.

### Priority 2: Default Return Value

Change `extract_answer` to return `0` instead of `None` when no answer is found, or document clearly that callers must handle `None`:

```python
def extract_answer(text: str, default: int = 0) -> int:
    """..."""
    for extractor in [...]:
        result = extractor(text)
        if result is not None:
            return result % ANSWER_MOD
    return default
```

### Priority 3: Follow-Up Prompting (Generation Layer)

Implement Hui Kang's multi-turn follow-up in the generation/inference code, not in `answer_extraction.py`. The generator should:
1. After each response, attempt extraction
2. If no boxed answer found, inject a follow-up message prompting for a boxed integer
3. Cap at 2-3 follow-up rounds
4. Only allow "educated guess" wording if time budget permits

---

## 8. Key Takeaways

1. **Our extraction code is already better than the top notebooks** — use `rfind` + balanced braces, strip LaTeX macros.

2. **The biggest gap is in the generation loop**, not extraction. Implementing Hui Kang's follow-up prompting is the highest-impact change.

3. **`\frac{N}{M}` is a real edge case** that could cause missed answers on geometry/combinatorics problems where intermediate results are fractions.

4. **Default answer**: Returning `0` (or any hardcoded integer) is equivalent to a random guess in 1/100000 odds. Don't put effort into choosing the "right" default — focus on reducing no-answer rate via follow-up prompting.

5. **Verification step** (ans-verifys) is worth considering if the generation budget allows, especially for problems where majority vote is split between two close candidates.

6. **Temperature matters more than extraction**: All top notebooks use temperature=0.5 (not 1.0). This reduces output diversity but improves answer consistency and reduces malformed outputs.

---

## Sources

- Notebooks analyzed: `44-50-let-me-over-cook.ipynb`, `43-50-aimo-3-gpt-oss-120b-weighted-entropy.ipynb`, `aimo-3-42-50-stable-lb-possible-43-luck.ipynb`, `ans-verifys.ipynb`, `aimo3-gpt-oss-120b-with-bayesian.ipynb`, `15-15-aime-2026-i-120b-in-20mins.ipynb`, `streaming-inference.ipynb`, `40-50-gpt-oss-120b-tir-dynamictime-kernelpool.ipynb`
- All notebooks pulled via `kaggle kernels pull` on 2026-04-01
- Leaderboard scores verified against public AIMO3 leaderboard
