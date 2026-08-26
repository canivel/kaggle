"""Binary Answer Verification Cascade for AIMO3.

Implements the amanatar approach (proven 44/50):
1. Run N attempts → collect answers
2. If strong consensus (≥early_stop agree) → return
3. Filter to candidates with ≥2 votes
4. Sort by average entropy (lowest = most confident)
5. Binary verify each candidate at T=0.0: "CORRECT or WRONG?"
6. Return first verified CORRECT
7. Fallback to entropy-weighted vote

This is injected into solve_problem AFTER the phase splitting.
"""

from collections import Counter, defaultdict


def verify_cascade(
    solver,
    problem: str,
    all_detailed: list[dict],
    all_valid: list[int],
    deadline: float,
) -> int | None:
    """Run verification cascade on collected results.

    Returns the verified answer, or None if cascade doesn't help.
    The caller should fall back to _select_answer if None.
    """
    import time

    if not all_valid or time.time() > deadline - 30:
        return None

    counter = Counter(all_valid)

    # Strong consensus — no verification needed
    top_answer, top_count = counter.most_common(1)[0]
    if top_count >= solver.cfg.early_stop:
        return top_answer

    # Filter to candidates with ≥2 votes
    candidates = [a for a, c in counter.items() if c >= 2]
    if not candidates:
        # If no candidate has 2+ votes, take all unique answers
        candidates = list(counter.keys())

    # Sort by average entropy (most confident first)
    entropy_map = defaultdict(list)
    for r in all_detailed:
        if r['Answer'] is not None and r['Entropy'] is not None:
            entropy_map[r['Answer']].append(r['Entropy'])

    avg_entropy = {a: sum(v) / len(v) for a, v in entropy_map.items()}
    candidates.sort(key=lambda x: avg_entropy.get(x, float('inf')))

    # Binary verify each candidate
    for ans in candidates:
        if time.time() > deadline - 15:
            break
        try:
            if _verify_answer(solver, problem, ans):
                print(f'VERIFIED: {ans}')
                return ans
        except Exception:
            pass

    return None  # cascade didn't find a verified answer


def _verify_answer(solver, problem: str, answer: int) -> bool:
    """Binary verification: ask the model if answer is CORRECT or WRONG."""
    prompt = (
        f"Problem:\n{problem}\n\n"
        f"Proposed answer: {answer}\n\n"
        f"Check the answer carefully.\n"
        f"Reply with only ONE word:\nCORRECT or WRONG"
    )
    try:
        prompt_ids = solver.encoding.encode(prompt)
        resp = solver.client.completions.create(
            model=solver.cfg.served_model_name,
            prompt=prompt_ids,
            temperature=0.0,
            max_tokens=5,
        )
        text = resp.choices[0].text.strip().upper()
        return "CORRECT" in text and "WRONG" not in text
    except Exception as e:
        print(f'[Verify Error] {e}')
        return False
