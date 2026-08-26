"""Novel AIMO3 Solver — combines all innovations.

This is the complete solve_problem replacement that integrates:
1. Domain-primed routing (per-problem strategy hints)
2. Two-phase adaptive solving (quick triage → deep with context)
3. Failure-aware retry (Python-mandatory for no-code attempts)
4. Multi-turn follow-up (recover boxed answers)
5. Binary verification cascade (amanatar proven 44/50)

All on top of the exact 43/50 base config (simple prompt, T=1.0, 81920 context).

The key principle: make ZERO changes to the base that aren't backed by evidence.
Each addition is isolated — if any one fails, the others still work.
"""

from collections import Counter, defaultdict
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Constants ────────────────────────────────────────────

DOMAIN_PREAMBLE = {
    'number_theory': 'This is a number theory problem. Start by: working modulo small primes, using pow(base,exp,mod), factoring with sympy.factorint(), enumerating small cases.',
    'geometry': 'This is a geometry problem. Start by: setting up coordinates (one vertex at origin), using sympy/numpy for calculations, verifying numerically.',
    'combinatorics': 'This is a combinatorics problem. Start by: computing small cases n=1,2,3,4 by brute force, using itertools, validating formulas against brute-force.',
    'algebra': 'This is an algebra problem. Start by: setting up with sympy.symbols() and sympy.solve(), trying specific value substitutions.',
}

DISAGREE_CONTEXT = (
    'NOTE: Initial analysis produced conflicting results with answers: {answers}. '
    'At least some are wrong. Be extra thorough — verify every step with Python code '
    'and check your answer with a second independent method.'
)

PYTHON_MANDATORY = (
    'You MUST use the Python tool for this problem. Do not provide a final answer '
    'without first executing Python code that computes or verifies the answer.'
)

FOLLOWUP_PROMPT = (
    'You have been working on this problem. Based on your analysis so far, '
    'what is the final integer answer? The answer must be between 0 and 99999. '
    'Please state your answer inside \\boxed{}.'
)

VERIFY_PROMPT = (
    'Problem:\n{problem}\n\n'
    'Proposed answer: {answer}\n\n'
    'Check the answer carefully.\n'
    'Reply with only ONE word:\nCORRECT or WRONG'
)

# ── Domain classifier ────────────────────────────────────

GEO_KW = ['triangle','circle','angle','perpendicular','inscribed','tangent','polygon',
          'circumscri','midpoint','altitude','radius','diameter']
NT_KW = ['prime','divisible','modulo','gcd','remainder','congruent','coprime',
         'fermat','euler','residue','digit']
COMBO_KW = ['how many','number of ways','permutation','combinat','probability',
            'expected value','pigeonhole','coloring','partition']

def classify_domain(problem):
    p = problem.lower()
    scores = {'geometry': sum(1 for k in GEO_KW if k in p),
              'number_theory': sum(1 for k in NT_KW if k in p),
              'combinatorics': sum(1 for k in COMBO_KW if k in p)}
    for d in ['combinatorics','geometry','number_theory']:
        if scores[d] >= 2: return d
    best = max(scores, key=scores.get)
    return best if scores[best] >= 1 else 'algebra'


# ── Verify cascade ───────────────────────────────────────

def verify_answer(solver, problem, answer):
    """Binary verify: CORRECT or WRONG at T=0.0."""
    prompt = VERIFY_PROMPT.format(problem=problem, answer=answer)
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
    except Exception:
        return False


def run_verify_cascade(solver, problem, all_detailed, all_valid, deadline):
    """Verify top candidates. Returns verified answer or None."""
    if not all_valid or time.time() > deadline - 30:
        return None

    counter = Counter(all_valid)
    top_answer, top_count = counter.most_common(1)[0]

    # Strong consensus — skip verification
    if top_count >= solver.cfg.early_stop:
        return top_answer

    # Filter candidates with ≥2 votes, sort by entropy
    candidates = [a for a, c in counter.items() if c >= 2]
    if not candidates:
        candidates = list(counter.keys())[:3]

    entropy_map = defaultdict(list)
    for r in all_detailed:
        if r['Answer'] is not None and r['Entropy'] is not None:
            entropy_map[r['Answer']].append(r['Entropy'])
    avg_entropy = {a: sum(v)/len(v) for a, v in entropy_map.items()}
    candidates.sort(key=lambda x: avg_entropy.get(x, float('inf')))

    for ans in candidates:
        if time.time() > deadline - 15:
            break
        if verify_answer(solver, problem, ans):
            print(f'  VERIFIED: {ans}')
            return ans

    return None


# ── Main solve function ──────────────────────────────────

def novel_solve_problem(solver, problem):
    """The full novel solve pipeline."""
    print(f'\nProblem: {problem[:300]}\n')

    # 1. Domain routing
    domain = classify_domain(problem)
    domain_hint = DOMAIN_PREAMBLE.get(domain, '')
    user_input = f'{problem} {solver.cfg.preference_prompt} {domain_hint}'
    print(f'Domain: {domain}')

    # Time budget
    elapsed = time.time() - solver.notebook_start_time
    left = solver.cfg.notebook_limit - elapsed
    reserved = max(0, solver.problems_remaining - 1) * solver.cfg.base_problem_timeout
    total_budget = min(max(left - reserved, solver.cfg.base_problem_timeout),
                       solver.cfg.high_problem_timeout)
    overall_deadline = time.time() + total_budget
    print(f'Budget: {total_budget:.0f}s | Left: {solver.problems_remaining}\n')

    # 2. Phase 1: Quick triage (4 attempts)
    phase1_budget = min(180, total_budget * 0.25)
    phase1_deadline = time.time() + phase1_budget

    detailed_p1, valid_p1 = _run_batch(solver, user_input, 0, 4, phase1_deadline, consensus=3)

    if valid_p1:
        counter_p1 = Counter(valid_p1)
        top, count = counter_p1.most_common(1)[0]
        if count >= 3:
            print(f'Phase 1: CONSENSUS ({count}/4 → {top})')
            solver.problems_remaining = max(0, solver.problems_remaining - 1)
            return top
        print(f'Phase 1: SPLIT — {dict(counter_p1)}')
    else:
        print('Phase 1: No valid answers')

    # 3. Phase 2: Deep solving with disagreement context
    if valid_p1 and len(set(valid_p1)) > 1:
        ans_list = ', '.join(str(a) for a in sorted(set(valid_p1)))
        context = DISAGREE_CONTEXT.format(answers=ans_list)
        user_p2 = f'{problem} {solver.cfg.preference_prompt} {domain_hint} {context}'
    else:
        user_p2 = user_input

    detailed_p2, valid_p2 = _run_batch(solver, user_p2, 4, 4, overall_deadline, consensus=4)

    all_detailed = detailed_p1 + detailed_p2
    all_valid = valid_p1 + valid_p2

    # 4. Failure-aware retry
    if time.time() < overall_deadline - 60:
        failed = [r for r in all_detailed if r['Answer'] is None or r['Python Calls'] == 0]
        n_retry = min(len(failed), 2)
        if n_retry > 0:
            print(f'Retrying {n_retry} failed attempts')
            retry_input = f'{problem} {solver.cfg.preference_prompt} {domain_hint} {PYTHON_MANDATORY}'
            retry_d, retry_v = _run_batch(solver, retry_input, 8, n_retry, overall_deadline, consensus=4)
            all_detailed.extend(retry_d)
            all_valid.extend(retry_v)

    # 5. Verify cascade
    verified = run_verify_cascade(solver, problem, all_detailed, all_valid, overall_deadline)
    if verified is not None:
        solver.problems_remaining = max(0, solver.problems_remaining - 1)
        return verified

    # 6. Fallback: entropy-weighted vote
    solver.problems_remaining = max(0, solver.problems_remaining - 1)
    if not all_valid:
        print('\nResult: 0\n')
        return 0
    return solver._select_answer(all_detailed)


def _run_batch(solver, user_input, seed_offset, n, deadline, consensus=3):
    """Run a batch of attempts."""
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
            except Exception: pass
    finally:
        stop.set(); ex.shutdown(wait=True, cancel_futures=True)
    return detailed, valid
