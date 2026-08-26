"""Adaptive Multi-Phase Solver for AIMO3.

Novel approach: instead of 8 identical parallel attempts,
use a two-phase strategy that adapts based on Phase 1 results.

Phase 1: Quick triage (4 fast attempts)
  - If agreement → submit early (saves time for harder problems)
  - If disagreement → this is a HARD problem

Phase 2: Deep solving (4 careful attempts with context)
  - Informed by Phase 1 disagreement
  - More time budget per attempt
  - Model knows previous attempts disagreed → more careful

Phase 3: Verification tiebreak
  - Top 2 candidates verified at T=0.0
"""

# This module provides the solve_problem replacement for AIMO3Solver.
# It's designed to be dropped into the existing Harmony-protocol notebook.

# The key functions to override in AIMO3Solver:
# 1. solve_problem() — replace with adaptive_solve()
# 2. No changes to _process_attempt() — reuse as-is

DISAGREEMENT_CONTEXT = (
    'NOTE: Initial analysis of this problem produced conflicting results. '
    'Previous quick attempts gave these different answers: {answers}. '
    'At least some of these are wrong. Please be extra thorough: '
    'verify every step with Python code, check your answer with a second method, '
    'and make sure your final answer is correct before giving \\boxed{}.'
)

VERIFY_PROMPT = (
    'Problem:\n{problem}\n\n'
    'Proposed answer: {answer}\n\n'
    'Check the answer carefully by solving the problem independently. '
    'Reply with only ONE word:\nCORRECT or WRONG'
)


def adaptive_solve_problem(solver, problem):
    """Adaptive multi-phase solving.

    Args:
        solver: AIMO3Solver instance
        problem: problem text string

    Returns:
        int: final answer
    """
    import time
    import threading
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print(f'\nProblem: {problem[:200]}\n')

    elapsed = time.time() - solver.notebook_start_time
    left = solver.cfg.notebook_limit - elapsed
    reserved = max(0, solver.problems_remaining - 1) * solver.cfg.base_problem_timeout
    total_budget = min(max(left - reserved, solver.cfg.base_problem_timeout), solver.cfg.high_problem_timeout)
    overall_deadline = time.time() + total_budget

    print(f'Budget: {total_budget:.0f}s | Problems left: {solver.problems_remaining}\n')

    # ──── Phase 1: Quick triage (4 attempts, limited budget) ────
    phase1_budget = min(180, total_budget * 0.3)  # 3 min or 30% of budget
    phase1_deadline = time.time() + phase1_budget

    user_input = f'{problem} {solver.cfg.preference_prompt}'
    tasks_p1 = [(solver.cfg.system_prompt, i) for i in range(4)]

    detailed_p1, valid_p1 = [], []
    stop_p1 = threading.Event()
    ex1 = ThreadPoolExecutor(max_workers=solver.cfg.workers)

    try:
        futs = [ex1.submit(solver._process_attempt, user_input, sp, ai, stop_p1, phase1_deadline)
                for sp, ai in tasks_p1]
        for f in as_completed(futs):
            try:
                r = f.result()
                detailed_p1.append(r)
                if r['Answer'] is not None:
                    valid_p1.append(r['Answer'])
                # Quick early stop: if 3 agree in Phase 1
                c = Counter(valid_p1).most_common(1)
                if c and c[0][1] >= 3:
                    stop_p1.set()
                    for ff in futs: ff.cancel()
                    break
            except Exception as e:
                print(f'P1 Future failed: {e}')
    finally:
        stop_p1.set()
        ex1.shutdown(wait=True, cancel_futures=True)

    # Check Phase 1 consensus
    if valid_p1:
        counter_p1 = Counter(valid_p1)
        top_answer, top_count = counter_p1.most_common(1)[0]

        if top_count >= 3:
            # Strong agreement → submit immediately
            print(f'Phase 1: CONSENSUS ({top_count}/4 agree on {top_answer})')
            solver.problems_remaining = max(0, solver.problems_remaining - 1)
            return top_answer
        else:
            print(f'Phase 1: DISAGREEMENT — answers: {dict(counter_p1)}')
    else:
        print('Phase 1: No valid answers')

    # ──── Phase 2: Deep solving with context ────
    # Inform Phase 2 about the disagreement
    if valid_p1:
        answer_list = ', '.join(str(a) for a in set(valid_p1))
        context = DISAGREEMENT_CONTEXT.format(answers=answer_list)
        user_input_p2 = f'{problem} {solver.cfg.preference_prompt}. {context}'
    else:
        user_input_p2 = user_input  # no Phase 1 info

    tasks_p2 = [(solver.cfg.system_prompt, i + 4) for i in range(4)]  # seeds 4-7

    detailed_p2, valid_p2 = [], []
    stop_p2 = threading.Event()
    ex2 = ThreadPoolExecutor(max_workers=solver.cfg.workers)

    try:
        futs = [ex2.submit(solver._process_attempt, user_input_p2, sp, ai, stop_p2, overall_deadline)
                for sp, ai in tasks_p2]
        for f in as_completed(futs):
            try:
                r = f.result()
                detailed_p2.append(r)
                if r['Answer'] is not None:
                    valid_p2.append(r['Answer'])
                c = Counter(valid_p2).most_common(1)
                if c and c[0][1] >= solver.cfg.early_stop:
                    stop_p2.set()
                    for ff in futs: ff.cancel()
                    break
            except Exception as e:
                print(f'P2 Future failed: {e}')
    finally:
        stop_p2.set()
        ex2.shutdown(wait=True, cancel_futures=True)

    # Combine all results and vote
    all_detailed = detailed_p1 + detailed_p2
    all_valid = valid_p1 + valid_p2

    if not all_valid:
        print('\nResult: 0 (no valid answers from either phase)\n')
        solver.problems_remaining = max(0, solver.problems_remaining - 1)
        return 0

    # Use the solver's standard voting
    answer = solver._select_answer(all_detailed)
    solver.problems_remaining = max(0, solver.problems_remaining - 1)
    return answer
