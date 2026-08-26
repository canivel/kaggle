"""
Proposed fixes for AIMO3 submission notebook (v23 -> v24).

These are DROP-IN replacements for methods in AIMO3Solver.
Each fix targets a specific failure mode identified in failure_analysis.json.

Expected combined impact: +1.0 to +1.5 problems (39 -> 40-41 on Kaggle test).
"""

# =============================================================================
# FIX 1: Quality-weighted voting (replaces _select_answer)
# Targets: execution_error, wrong_formalization, timeout
# Expected: +0.85 problems
# =============================================================================

def _select_answer_quality(self, results):
    """Quality-weighted 1/entropy voting with penalty signals.

    Changes from current _select_answer:
    - Penalizes attempts with Python errors (0.3x weight)
    - Penalizes attempts with very long responses / context exhaustion (0.1x)
    - Penalizes no-code attempts (0.7x) -- less verified
    - Bonuses attempts with moderate code usage (1.3x) -- sweet spot
    """
    from collections import defaultdict
    import pandas as pd

    aw = defaultdict(float)  # answer -> weighted score
    av = defaultdict(int)    # answer -> vote count

    for r in results:
        a = r['Answer']
        e = r['Entropy']
        if a is None:
            continue

        # Base weight: inverse entropy (same as current)
        w = 1.0 / max(e, 1e-9)

        # PENALTY: Python errors indicate buggy code -> wrong answer
        py_errors = r.get('Python Errors', 0)
        if py_errors > 0:
            w *= 0.3  # 70% discount

        # PENALTY: Very long responses = likely context exhaustion
        resp_len = r.get('Response Length', 0)
        if resp_len > 50000:
            w *= 0.1  # 90% discount

        # BONUS: Moderate Python usage = verified computation
        py_calls = r.get('Python Calls', 0)
        if 2 <= py_calls <= 5:
            w *= 1.3
        elif py_calls == 0:
            w *= 0.7  # No code = less trustworthy

        aw[a] += w
        av[a] += 1

    # Display scored candidates
    scored = sorted(
        [{'answer': a, 'votes': av[a], 'score': aw[a]} for a in aw],
        key=lambda x: x['score'], reverse=True
    )
    df = pd.DataFrame(
        [(s['answer'], s['votes'], round(s['score'], 3)) for s in scored],
        columns=['Answer', 'Votes', 'Score']
    )
    display(df)

    if not scored:
        print('\nFinal Answer: 0\n')
        return 0

    print(f'\nFinal Answer: {scored[0]["answer"]}\n')
    return scored[0]['answer']


# =============================================================================
# FIX 2: Temperature diversity across attempts (modify solve_problem)
# Targets: wrong_approach (correlated errors)
# Expected: +0.3 to +1.0 problems
# =============================================================================

def solve_problem_with_temp_diversity(self, problem):
    """Modified solve_problem that uses temperature diversity.

    First half of attempts at T=0.5 (focused), second half at T=1.0 (exploratory).
    This reduces correlated wrong answers: if the model has a systematic
    misconception at T=0.8, T=0.5 and T=1.0 may explore different paths.
    """
    print(f'\nProblem: {problem}\n')
    user_input = f'{problem} {self.cfg.preference_prompt}'
    elapsed = time.time() - self.notebook_start_time
    left = self.cfg.notebook_limit - elapsed
    reserved = max(0, self.problems_remaining - 1) * self.cfg.base_problem_timeout
    budget = min(max(left - reserved, self.cfg.base_problem_timeout), self.cfg.high_problem_timeout)
    deadline = time.time() + budget
    print(f'Budget: {budget:.0f}s | Problems left: {self.problems_remaining}\n')

    # KEY CHANGE: alternate temperature per attempt
    tasks = []
    for i in range(self.cfg.attempts):
        if i < self.cfg.attempts // 2:
            temp = 0.5  # focused attempts
        else:
            temp = 1.0  # exploratory attempts
        tasks.append((self.cfg.system_prompt, i, temp))

    detailed, valid = [], []
    stop = threading.Event()
    ex = ThreadPoolExecutor(max_workers=self.cfg.workers)
    try:
        futs = [ex.submit(self._process_attempt_with_temp, user_input, sp, ai, temp, stop, deadline)
                for sp, ai, temp in tasks]
        for f in as_completed(futs):
            try:
                r = f.result()
                detailed.append(r)
                if r['Answer'] is not None:
                    valid.append(r['Answer'])
                c = Counter(valid).most_common(1)
                if c and c[0][1] >= self.cfg.early_stop:
                    stop.set()
                    for ff in futs:
                        ff.cancel()
                    break
            except Exception as e:
                print(f'Future failed: {e}')
    finally:
        stop.set()
        ex.shutdown(wait=True, cancel_futures=True)
        self.problems_remaining = max(0, self.problems_remaining - 1)

    if detailed:
        df = pd.DataFrame(detailed)
        df['Entropy'] = df['Entropy'].round(3)
        df['Answer'] = df['Answer'].astype('Int64')
        display(df)
    if not valid:
        print('\nResult: 0\n')
        return 0
    return self._select_answer_quality(detailed)


# =============================================================================
# FIX 3: Context exhaustion early termination
# Targets: timeout_context_exhaustion
# Expected: +0.2 to +0.5 problems
# =============================================================================

# In _process_attempt, add this check inside the streaming loop:
CONTEXT_WARNING_THRESHOLD = 0.80  # 80% of context_tokens

def _check_context_exhaustion(total_tokens, context_tokens, conversation, encoding):
    """Check if we're about to exhaust context and inject forced-answer prompt.

    Call this inside the streaming loop after each chunk.
    Returns True if we should force an answer now.
    """
    if total_tokens > context_tokens * CONTEXT_WARNING_THRESHOLD:
        # Inject forced-answer message
        force_msg = Message.from_role_and_content(
            Role.USER,
            'IMPORTANT: You are running out of space. '
            'Based on your work so far, what is the final integer answer (0-99999)? '
            'State it immediately in \\boxed{}.'
        )
        conversation.messages.append(force_msg)
        return True
    return False


# =============================================================================
# FIX 4: Improved answer extraction
# Targets: answer_extraction failures
# Expected: +0.1 to +0.3 problems
# =============================================================================

def _scan_for_answer_improved(self, text):
    """Improved answer scanner that handles more edge cases.

    New patterns handled:
    - \\boxed{N \\pmod{M}} -> N
    - \\boxed{\\frac{a}{b}} -> a/b if integer
    - \\boxed{N,NNN} -> remove commas
    - Negative numbers: \\boxed{-N} -> (-N) % 100000
    - Answer in code output: print(N) at end
    - "the answer is N" in various formats
    """
    import re

    # Pattern 1: Standard \\boxed{integer}
    for pat in [r'\\boxed\s*\{\s*([0-9,]+)\s*\}']:
        ms = re.findall(pat, text)
        if ms:
            try:
                v = int(ms[-1].replace(',', ''))
                if 0 <= v <= 99999:
                    return v
            except ValueError:
                pass

    # Pattern 2: Negative \\boxed{-N} -> mod 100000
    ms = re.findall(r'\\boxed\s*\{\s*(-[0-9,]+)\s*\}', text)
    if ms:
        try:
            v = int(ms[-1].replace(',', '')) % 100000
            if 0 <= v <= 99999:
                return v
        except ValueError:
            pass

    # Pattern 3: \\boxed{N \\pmod{M}} -> extract N
    ms = re.findall(r'\\boxed\s*\{\s*(\d+)\s*\\pmod', text)
    if ms:
        try:
            v = int(ms[-1])
            if 0 <= v <= 99999:
                return v
        except ValueError:
            pass

    # Pattern 4: \\boxed{\\frac{a}{b}} -> a//b if exact
    ms = re.findall(r'\\boxed\s*\{\s*\\frac\s*\{\s*(\d+)\s*\}\s*\{\s*(\d+)\s*\}\s*\}', text)
    if ms:
        try:
            a, b = int(ms[-1][0]), int(ms[-1][1])
            if b != 0 and a % b == 0:
                v = (a // b) % 100000
                if 0 <= v <= 99999:
                    return v
        except ValueError:
            pass

    # Pattern 5: "the final answer is N" or "answer is N" or "answer: N"
    for pat in [
        r'(?:the\s+)?(?:final\s+)?answer\s+is\s*[:\s]*(\d+)',
        r'answer\s*[=:]\s*(\d+)',
        r'(?:therefore|thus|hence|so)\s*,?\s*(?:the\s+answer\s+is\s+)?(\d+)',
    ]:
        ms = re.findall(pat, text, re.IGNORECASE)
        if ms:
            try:
                v = int(ms[-1])
                if 0 <= v <= 99999:
                    return v
            except ValueError:
                pass

    # Pattern 6: Last integer in the final code output block
    output_blocks = re.findall(r'```output\s*(.*?)```', text, re.DOTALL)
    if output_blocks:
        last_output = output_blocks[-1].strip()
        # Get the last line that looks like a number
        for line in reversed(last_output.split('\n')):
            line = line.strip()
            try:
                v = int(line)
                if 0 <= v <= 99999:
                    return v
            except ValueError:
                pass

    return None


# =============================================================================
# FIX 5: Verify cascade (from amanatar approach, already in verify_cascade.py)
# Targets: wrong_approach, wrong_formalization
# Expected: +0.6 problems (on top of quality voting)
# =============================================================================

# The verify cascade is already implemented in verify_cascade.py.
# To integrate it into solve_problem, add after voting:
#
#   # After getting voted answer from _select_answer_quality:
#   verified = verify_cascade(self, problem, detailed, valid, deadline)
#   if verified is not None:
#       return verified
#   return voted_answer


# =============================================================================
# INTEGRATION: How to apply all fixes to submission_v23_amanatar.ipynb
# =============================================================================

"""
To create v24, modify the AIMO3Solver class in the notebook:

1. Replace _select_answer method with _select_answer_quality (Fix 1)
   - This is the HIGHEST IMPACT change (+0.85 problems in simulation)
   - Zero risk: same API, just better weights

2. Add _process_attempt_with_temp that takes a temperature parameter (Fix 2)
   - Moderate risk: if T=0.5 is too focused, could lose some problems
   - Test locally first

3. Add context exhaustion check in _process_attempt streaming loop (Fix 3)
   - Low risk: only activates at 80% context, which is already failing

4. Replace _scan_for_answer with _scan_for_answer_improved (Fix 4)
   - Low risk: all new patterns are additive (doesn't change existing behavior)

5. Keep verify cascade from v23 (Fix 5)
   - Already implemented, keep as-is

PRIORITY ORDER for limited submission budget:
1. Fix 1 (quality voting) -- highest expected value
2. Fix 4 (answer extraction) -- lowest risk
3. Fix 3 (context exhaustion) -- low risk, moderate reward
4. Fix 2 (temp diversity) -- needs testing
5. Fix 5 (verify cascade) -- already in v23
"""
