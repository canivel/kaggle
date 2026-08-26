"""
Multi-Turn Code-Based Verification for AIMO3.

Strategy 1: After majority vote produces answer X, ask the model to write
Python verification code that tests X against problem constraints. Code runs
in an existing Jupyter sandbox. If verification fails, try next candidate.

CRITICAL: This is NOT self-refinement (proven harmful, -4 pts).
This is NOT Pawan Mali's V133 binary CORRECT/WRONG (0 improvement).
This uses TIR (tool-integrated reasoning) — the verification CODE RUNS
and provides objective boolean results independent of model reasoning.

Integration point: after _select_answer() in solve_problem().
"""

import re
import time

# ── Verification Prompt Templates ──────────────────────────

VERIFY_CODE_PROMPT = (
    "A student solved the following problem and claims the answer is {answer}.\n\n"
    "Problem:\n{problem}\n\n"
    "Write Python code that tests whether {answer} satisfies ALL the constraints "
    "stated in the problem. Your code should:\n"
    "1. Encode each constraint from the problem as a Python check\n"
    "2. Print 'VERIFIED' if the answer passes ALL checks\n"
    "3. Print 'FAILED: <reason>' if any check fails\n\n"
    "Use sympy, numpy, or math as needed. Be thorough — test every constraint."
)

SYNTHESIS_PROMPT = (
    "Two solution approaches for the same problem gave different answers.\n\n"
    "Problem:\n{problem}\n\n"
    "Approach A concluded: answer = {answer_a}\n"
    "Key reasoning: {reasoning_a}\n\n"
    "Approach B concluded: answer = {answer_b}\n"
    "Key reasoning: {reasoning_b}\n\n"
    "Write Python code to independently verify BOTH answers against the problem "
    "constraints. Then determine which approach has an error.\n"
    "Print the correct answer inside \\boxed{{}}."
)


def verify_answer_with_code(solver, problem, answer, deadline):
    """Run code-based verification on a candidate answer.

    Unlike V133's bare CORRECT/WRONG judgment, this asks the model to
    WRITE AND EXECUTE Python code that tests constraints objectively.

    Args:
        solver: AIMO3Solver instance (has .client, .encoding, .cfg, .sandbox_pool)
        problem: problem text
        answer: integer answer to verify
        deadline: time deadline

    Returns:
        'VERIFIED', 'FAILED', or 'INCONCLUSIVE'
    """
    if time.time() > deadline - 30:
        return 'INCONCLUSIVE'

    prompt = VERIFY_CODE_PROMPT.format(problem=problem, answer=answer)

    try:
        # Use the Harmony completions API
        prompt_ids = solver.encoding.encode(prompt)
        resp = solver.client.completions.create(
            model=solver.cfg.served_model_name,
            prompt=prompt_ids,
            temperature=0.0,  # Greedy for verification
            max_tokens=2048,  # Need enough tokens for code generation
        )
        text = resp.choices[0].text.strip()

        # Extract code block from response
        code_match = re.search(r'```python\s*(.*?)\s*```', text, re.DOTALL)
        if not code_match:
            # Try without language specifier
            code_match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)

        if not code_match:
            return 'INCONCLUSIVE'

        code = code_match.group(1).strip()

        # Execute the verification code in a sandbox
        sandbox = None
        try:
            sandbox = solver.sandbox_pool.get(timeout=solver.cfg.sandbox_timeout)
            output = sandbox.execute(code, timeout=10)  # Short timeout for verification

            output_upper = output.upper()
            if 'VERIFIED' in output_upper and 'FAILED' not in output_upper:
                return 'VERIFIED'
            elif 'FAILED' in output_upper or 'ERROR' in output_upper:
                return 'FAILED'
            else:
                return 'INCONCLUSIVE'
        finally:
            if sandbox:
                sandbox.reset()
                solver.sandbox_pool.put(sandbox)

    except Exception as e:
        print(f'[Verify Code Error] {e}')
        return 'INCONCLUSIVE'


def verification_cascade_with_code(solver, problem, detailed_results, deadline):
    """Post-vote verification cascade using code execution.

    After standard voting, verifies the top candidate(s) using Python code.
    Falls back to original vote if verification is inconclusive.

    Args:
        solver: AIMO3Solver instance
        problem: problem text (WITHOUT preference prompt)
        detailed_results: list of attempt result dicts
        deadline: time deadline

    Returns:
        int: verified answer, or None to fall back to standard voting
    """
    from collections import Counter, defaultdict
    import math as m

    # Get valid answers
    valid = [r['Answer'] for r in detailed_results if r['Answer'] is not None]
    if not valid:
        return None

    counter = Counter(valid)
    top_answer, top_count = counter.most_common(1)[0]

    # Strong consensus (4+) — skip verification, waste of time
    if top_count >= solver.cfg.early_stop:
        return top_answer

    # Get candidates sorted by 1/entropy score
    entropy_map = defaultdict(list)
    for r in detailed_results:
        if r['Answer'] is not None and r['Entropy'] is not None:
            entropy_map[r['Answer']].append(r['Entropy'])

    candidates = []
    for ans in counter:
        avg_ent = sum(entropy_map.get(ans, [float('inf')])) / max(len(entropy_map.get(ans, [1])), 1)
        score = counter[ans] / max(avg_ent, 1e-9)
        candidates.append((ans, counter[ans], avg_ent, score))

    # Sort by score (votes / entropy) descending
    candidates.sort(key=lambda x: x[3], reverse=True)

    # Verify top 2-3 candidates
    for ans, votes, entropy, score in candidates[:3]:
        if time.time() > deadline - 20:
            break

        result = verify_answer_with_code(solver, problem, ans, deadline)
        print(f'  Code-verify {ans} (votes={votes}, ent={entropy:.2f}): {result}')

        if result == 'VERIFIED':
            return ans
        elif result == 'FAILED':
            continue  # Try next candidate
        # INCONCLUSIVE: keep going

    return None  # Fall back to standard voting


def solve_with_verification(solver, problem):
    """Drop-in replacement for solve_problem() that adds code-based verification.

    This wraps the standard solve_problem flow and adds a post-vote
    verification step using TIR (code execution).
    """
    import threading
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print(f'\nProblem: {problem[:300]}\n')

    user_input = f'{problem} {solver.cfg.preference_prompt}'

    # Time budget (same as standard)
    elapsed = time.time() - solver.notebook_start_time
    left = solver.cfg.notebook_limit - elapsed
    reserved = max(0, solver.problems_remaining - 1) * solver.cfg.base_problem_timeout
    budget = min(max(left - reserved, solver.cfg.base_problem_timeout),
                 solver.cfg.high_problem_timeout)
    deadline = time.time() + budget
    # Reserve 30s for verification at the end
    solve_deadline = deadline - 30
    print(f'Budget: {budget:.0f}s (30s reserved for verification)\n')

    # Standard solve: 12 parallel attempts with early stopping
    tasks = [(solver.cfg.system_prompt, i) for i in range(solver.cfg.attempts)]
    detailed, valid = [], []
    stop = threading.Event()
    ex = ThreadPoolExecutor(max_workers=solver.cfg.workers)

    try:
        futs = [ex.submit(solver._process_attempt, user_input, sp, ai, stop, solve_deadline)
                for sp, ai in tasks]
        for f in as_completed(futs):
            try:
                r = f.result()
                detailed.append(r)
                if r['Answer'] is not None:
                    valid.append(r['Answer'])
                c = Counter(valid).most_common(1)
                if c and c[0][1] >= solver.cfg.early_stop:
                    stop.set()
                    for ff in futs:
                        ff.cancel()
                    break
            except Exception as e:
                print(f'Future failed: {e}')
    finally:
        stop.set()
        ex.shutdown(wait=True, cancel_futures=True)

    solver.problems_remaining = max(0, solver.problems_remaining - 1)

    if not valid:
        print('\nResult: 0\n')
        return 0

    # === NEW: Code-based verification ===
    # Only run if we have time and no strong consensus
    counter = Counter(valid)
    top_answer, top_count = counter.most_common(1)[0]

    if top_count >= solver.cfg.early_stop:
        # Strong consensus — skip verification
        print(f'Strong consensus ({top_count}/{len(valid)} -> {top_answer}), skipping verification')
        return top_answer

    # Run code-based verification on top candidates
    print('Running code-based verification...')
    verified = verification_cascade_with_code(
        solver, problem, detailed, deadline
    )

    if verified is not None:
        print(f'Code-verified answer: {verified}')
        return verified

    # Fallback to standard entropy-weighted voting
    print('Verification inconclusive, using standard voting')
    return solver._select_answer(detailed)


# ── Cross-Attempt Synthesis (Strategy 2) ───────────────────

def extract_reasoning_summary(attempt_result, max_chars=500):
    """Extract a concise reasoning summary from an attempt's trace.

    This is a simplified extraction — in production, would need more
    sophisticated trace parsing.
    """
    # The attempt result dict has 'full_text' or we'd need to store it
    # In the current architecture, the full trace is NOT stored in the
    # result dict (only Answer, Entropy, Python Calls, etc.)
    # This is a limitation that makes Strategy 2 harder to implement.
    return f"Answer: {attempt_result.get('Answer', 'unknown')}"


def cross_attempt_synthesis(solver, problem, detailed_results, deadline):
    """When votes are split, show the model conflicting approaches and ask
    it to adjudicate using code.

    Only fires when no answer has 4+ votes (the split-vote case).

    Returns:
        int or None: synthesized answer, or None to fall back
    """
    from collections import Counter

    valid = [r['Answer'] for r in detailed_results if r['Answer'] is not None]
    if not valid:
        return None

    counter = Counter(valid)
    if not counter:
        return None

    # Only fire if genuinely split (no 4+ consensus)
    top_count = counter.most_common(1)[0][1]
    if top_count >= 4:
        return None  # Not a split

    if time.time() > deadline - 60:
        return None  # Not enough time

    # Get top 2 candidates
    top2 = counter.most_common(2)
    if len(top2) < 2:
        return None

    answer_a, count_a = top2[0]
    answer_b, count_b = top2[1]

    # NOTE: In current architecture, we do NOT have access to solution traces
    # (not stored in result dicts). This means we can only provide the answers,
    # not the reasoning summaries. This significantly reduces the value of
    # cross-attempt synthesis.
    #
    # To fully implement Strategy 2, would need to:
    # 1. Store full conversation traces in _process_attempt results
    # 2. Extract key reasoning steps from each trace
    # 3. Summarize into ~500 chars per approach
    # This is the HIGH IMPLEMENTATION COMPLEXITY noted in the analysis.

    prompt = (
        f"Problem:\n{problem}\n\n"
        f"Two groups of solution attempts gave different answers:\n"
        f"Group A ({count_a} attempts): answer = {answer_a}\n"
        f"Group B ({count_b} attempts): answer = {answer_b}\n\n"
        f"Write Python code to independently solve this problem and determine "
        f"which answer is correct. Test both answers against the problem constraints.\n"
        f"Print the correct answer inside \\boxed{{}}."
    )

    try:
        prompt_ids = solver.encoding.encode(prompt)
        resp = solver.client.completions.create(
            model=solver.cfg.served_model_name,
            prompt=prompt_ids,
            temperature=0.0,
            max_tokens=4096,
        )
        text = resp.choices[0].text.strip()

        # Extract code and run it
        code_match = re.search(r'```python\s*(.*?)\s*```', text, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
            sandbox = None
            try:
                sandbox = solver.sandbox_pool.get(timeout=solver.cfg.sandbox_timeout)
                output = sandbox.execute(code, timeout=15)

                # Look for boxed answer in output
                boxed = re.findall(r'\\boxed\s*\{\s*(\d+)\s*\}', output + text)
                if boxed:
                    synth_answer = int(boxed[-1])
                    if 0 <= synth_answer <= 99999:
                        return synth_answer
            finally:
                if sandbox:
                    sandbox.reset()
                    solver.sandbox_pool.put(sandbox)

        # Try to extract answer from text without code
        boxed = re.findall(r'\\boxed\s*\{\s*(\d+)\s*\}', text)
        if boxed:
            synth_answer = int(boxed[-1])
            if 0 <= synth_answer <= 99999:
                return synth_answer

    except Exception as e:
        print(f'[Synthesis Error] {e}')

    return None
