"""Novel AIMO3 Solver v2 — corrected after failure analysis.

REMOVED: Domain routing (Classify Then Solve = -3.7 pts, worst strategy)
KEPT: Phase splitting, failure retry, follow-up, verify cascade

Base: EXACT 43/50 config (5-step prompt proven by both 44/50 notebooks)
+ EAGLE-3 for speed
+ Phase split for time management
+ Failure retry for no-code attempts
+ Follow-up for no-boxed answers
"""

from collections import Counter, defaultdict
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

FOLLOWUP_PROMPT = (
    'You have been working on this problem. Based on your analysis so far, '
    'what is the final integer answer? The answer must be between 0 and 99999. '
    'Please state your answer inside \\boxed{}.'
)

# NO domain routing — proven harmful (-3.7 pts)
# Context injection for disagreement is safe (problem-specific, not prompt-change)
DISAGREE_CONTEXT = (
    'NOTE: Initial analysis produced conflicting results with answers: {answers}. '
    'At least some are wrong. Be extra thorough — verify every step with Python code.'
)

PYTHON_MANDATORY = (
    'You MUST use the Python tool for this problem. Execute Python code '
    'that computes or verifies the answer before giving \\boxed{}.'
)


def novel_solve_v2(solver, problem):
    """Corrected novel solve: no domain routing, keep phase split + retry."""
    print(f'\\nProblem: {problem[:300]}\\n')

    # Standard user input — NO domain hints
    user_input = f'{problem} {solver.cfg.preference_prompt}'

    # Time budget
    elapsed = time.time() - solver.notebook_start_time
    left = solver.cfg.notebook_limit - elapsed
    reserved = max(0, solver.problems_remaining - 1) * solver.cfg.base_problem_timeout
    total_budget = min(max(left - reserved, solver.cfg.base_problem_timeout),
                       solver.cfg.high_problem_timeout)
    overall_deadline = time.time() + total_budget
    print(f'Budget: {total_budget:.0f}s | Left: {solver.problems_remaining}\\n')

    # Phase 1: Quick triage (4 attempts, 25% budget)
    p1_budget = min(180, total_budget * 0.25)
    p1_deadline = time.time() + p1_budget
    d1, v1 = _run_batch(solver, user_input, 0, 4, p1_deadline, consensus=3)

    if v1:
        c1 = Counter(v1)
        top, count = c1.most_common(1)[0]
        if count >= 3:
            print(f'Phase 1: CONSENSUS ({count}/4 -> {top})')
            solver.problems_remaining = max(0, solver.problems_remaining - 1)
            return top
        print(f'Phase 1: SPLIT — {dict(c1)}')
    else:
        print('Phase 1: No answers')

    # Phase 2: Deep (4 attempts, full budget)
    # Add disagreement context ONLY if Phase 1 had conflicting answers
    if v1 and len(set(v1)) > 1:
        ans_str = ', '.join(str(a) for a in sorted(set(v1)))
        user_p2 = f'{problem} {solver.cfg.preference_prompt} {DISAGREE_CONTEXT.format(answers=ans_str)}'
    else:
        user_p2 = user_input

    d2, v2 = _run_batch(solver, user_p2, 4, 4, overall_deadline, consensus=4)

    all_d = d1 + d2
    all_v = v1 + v2

    # Failure retry: re-run no-code/no-answer attempts with Python-mandatory
    if time.time() < overall_deadline - 60:
        failed = [r for r in all_d if r['Answer'] is None or r['Python Calls'] == 0]
        n_retry = min(len(failed), 2)
        if n_retry > 0:
            print(f'Retrying {n_retry} failed attempts')
            retry_input = f'{problem} {solver.cfg.preference_prompt} {PYTHON_MANDATORY}'
            rd, rv = _run_batch(solver, retry_input, 8, n_retry, overall_deadline, consensus=4)
            all_d.extend(rd)
            all_v.extend(rv)

    solver.problems_remaining = max(0, solver.problems_remaining - 1)

    if all_d:
        import pandas as pd
        df = pd.DataFrame(all_d)
        df['Entropy'] = df['Entropy'].round(3)
        df['Answer'] = df['Answer'].astype('Int64')
        from IPython.display import display
        display(df)

    if not all_v:
        print('\\nResult: 0\\n')
        return 0

    return solver._select_answer(all_d)


def _run_batch(solver, user_input, seed_offset, n, deadline, consensus=3):
    tasks = [(solver.cfg.system_prompt, i + seed_offset) for i in range(n)]
    detailed, valid = [], []
    stop = threading.Event()
    ex = ThreadPoolExecutor(max_workers=solver.cfg.workers)
    try:
        futs = [ex.submit(solver._process_attempt, user_input, sp, ai, stop, deadline)
                for sp, ai in tasks]
        for f in as_completed(futs):
            try:
                r = f.result(); detailed.append(r)
                if r['Answer'] is not None: valid.append(r['Answer'])
                c = Counter(valid).most_common(1)
                if c and c[0][1] >= consensus:
                    stop.set()
                    for ff in futs: ff.cancel()
                    break
            except: pass
    finally:
        stop.set(); ex.shutdown(wait=True, cancel_futures=True)
    return detailed, valid
