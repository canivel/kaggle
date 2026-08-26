"""Voting strategies for selecting the best answer from multiple candidates.

Implements:
1. Majority voting (mode of all answers)
2. Weighted voting (weight by code execution success, chain length)
3. Confidence-based selection
"""

from __future__ import annotations

from collections import Counter


def majority_vote(answers: list[int | None], default: int = 0) -> int:
    """Simple majority voting: return the most common answer.

    Args:
        answers: List of extracted integer answers (may contain None).
        default: Fallback answer if no valid answers exist.

    Returns:
        The most frequently occurring answer.
    """
    valid = [a for a in answers if a is not None]
    if not valid:
        return default

    counter = Counter(valid)
    return counter.most_common(1)[0][0]


def weighted_vote(
    answers: list[int | None],
    weights: list[float] | None = None,
    default: int = 0,
) -> int:
    """Weighted majority voting.

    Args:
        answers: List of extracted integer answers.
        weights: Per-answer weights (e.g., from code execution confidence).
                 If None, all valid answers get weight 1.0.
        default: Fallback answer.

    Returns:
        The answer with highest total weight.
    """
    if weights is None:
        return majority_vote(answers, default=default)

    weighted_counts: dict[int, float] = {}
    for ans, w in zip(answers, weights):
        if ans is not None:
            weighted_counts[ans] = weighted_counts.get(ans, 0.0) + w

    if not weighted_counts:
        return default

    return max(weighted_counts, key=weighted_counts.get)


def confidence_score(answers: list[int | None]) -> float:
    """Compute confidence as the fraction of answers that agree with the majority.

    Returns a float in [0, 1]. Higher means more agreement.
    """
    valid = [a for a in answers if a is not None]
    if not valid:
        return 0.0

    counter = Counter(valid)
    most_common_count = counter.most_common(1)[0][1]
    return most_common_count / len(valid)


def compute_weights(
    code_executed: list[bool],
    code_succeeded: list[bool],
    has_boxed: list[bool],
) -> list[float]:
    """Compute per-solution weights based on quality signals.

    Args:
        code_executed: Whether each solution attempted code execution.
        code_succeeded: Whether code execution succeeded.
        has_boxed: Whether the solution has a \\boxed{} answer.

    Returns:
        List of weights (higher = more trustworthy).
    """
    weights = []
    for executed, succeeded, boxed in zip(code_executed, code_succeeded, has_boxed):
        w = 1.0
        if executed and succeeded:
            w += 2.0  # Code ran successfully: high trust
        elif executed and not succeeded:
            w += 0.5  # Code attempted but failed: moderate trust
        if boxed:
            w += 0.5  # Has a clear boxed answer
        weights.append(w)
    return weights


def vote_with_quality(
    answers: list[int | None],
    code_executed: list[bool],
    code_succeeded: list[bool],
    has_boxed: list[bool],
    default: int = 0,
) -> tuple[int, float]:
    """Vote with quality-based weighting.

    Returns:
        Tuple of (answer, confidence).
    """
    weights = compute_weights(code_executed, code_succeeded, has_boxed)
    answer = weighted_vote(answers, weights, default=default)
    conf = confidence_score(answers)
    return answer, conf
