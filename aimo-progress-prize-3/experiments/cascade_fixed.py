"""
Why does cascade HURT when combined with EV voting?

Hypothesis: When we add more attempts for split-vote problems, the additional
wrong answers (which cluster!) dilute the correct signal. With 8 attempts,
the correct answer might have 3 code-verified votes vs 2 clustered-wrong votes.
Adding 8 more might give 5 correct vs 4 clustered-wrong, but the clustered-wrong
answers now also include more code-verified ones, reducing the EV advantage.

The fix: The cascade should use ONLY code-verified answers from extra waves.
Non-code-verified extra attempts should be discarded entirely.

Also test: What if instead of cascading (more attempts), we use the compute
budget for TEMPERATURE DIVERSITY? Run 4 attempts at T=0.5, 4 at T=1.0.
The T=1.0 attempts explore different solution paths, reducing correlation.
"""
import numpy as np
from collections import defaultdict, Counter
import time

np.random.seed(42)
CORRECT = 42
N_SIMS = 5000

DIFFICULTIES = [0.85]*35 + [0.40]*5 + [0.20]*5 + [0.05]*3 + [0.01]*2
WCP = 0.70
CORRECT_ENT_MU, CORRECT_ENT_STD = 1.5, 0.3
CONFIDENT_WRONG_ENT_MU = 1.8
WRONG_ENT_MU, WRONG_ENT_STD = 2.5, 0.5
CORRECT_CODE_RATE, WRONG_CODE_RATE = 0.72, 0.60
CORRECT_ERROR_RATE, WRONG_ERROR_RATE = 0.05, 0.25

CODE_BOOST = 10.0
ERROR_PEN = 0.1
NO_CODE_PEN = 0.2


def gen_attempt(p, temp_factor=1.0):
    """Generate one attempt. temp_factor > 1 = higher temperature = less correlation."""
    is_correct = np.random.random() < p
    if is_correct:
        ans = CORRECT
        ent = max(0.1, np.random.normal(CORRECT_ENT_MU * temp_factor, CORRECT_ENT_STD))
        uc = np.random.random() < CORRECT_CODE_RATE
        pe = 1 if np.random.random() < CORRECT_ERROR_RATE else 0
    else:
        if np.random.random() < WCP / temp_factor:  # Higher temp = less clustering
            ans = -1  # attractor placeholder
            ent = max(0.1, np.random.normal(CONFIDENT_WRONG_ENT_MU * temp_factor, 0.3))
        else:
            ans = np.random.randint(0, 100000)
            ent = max(0.1, np.random.normal(WRONG_ENT_MU, WRONG_ENT_STD))
        uc = np.random.random() < WRONG_CODE_RATE
        pe = 1 if np.random.random() < WRONG_ERROR_RATE else 0
    pc = max(0, int(np.random.normal(3.0, 1.5))) if uc else 0
    return ans, ent, uc, pe, pc, is_correct


def ev_vote(answers, entropies, used_codes, py_errors):
    w = defaultdict(float)
    for a, e, uc, pe in zip(answers, entropies, used_codes, py_errors):
        wt = 1.0 / max(e, 1e-9)
        if uc and pe == 0:
            wt *= CODE_BOOST
        elif uc and pe > 0:
            wt *= ERROR_PEN
        elif not uc:
            wt *= NO_CODE_PEN
        w[a] += wt
    return max(w, key=w.get) if w else 0


def sim_problem(p, attractor, strategy):
    answers, entropies, used_codes, py_errors = [], [], [], []

    def add(n, temp_factor=1.0):
        for _ in range(n):
            a, e, uc, pe, pc, ic = gen_attempt(p, temp_factor)
            if a == -1: a = attractor
            answers.append(a)
            entropies.append(e)
            used_codes.append(uc)
            py_errors.append(pe)

    if strategy == 'baseline':
        add(8)
        w = defaultdict(float)
        for a, e in zip(answers, entropies):
            w[a] += 1.0 / max(e, 1e-9)
        return max(w, key=w.get)

    elif strategy == 'ev_8':
        add(8)
        return ev_vote(answers, entropies, used_codes, py_errors)

    elif strategy == 'ev_16':
        # Simply 16 attempts with EV voting
        add(16)
        return ev_vote(answers, entropies, used_codes, py_errors)

    elif strategy == 'ev_32':
        add(32)
        return ev_vote(answers, entropies, used_codes, py_errors)

    elif strategy == 'ev_8_diverse':
        # 4 at T=0.5 (low temp) + 4 at T=1.5 (high temp, diverse)
        add(4, temp_factor=0.8)   # Standard
        add(4, temp_factor=1.5)   # Diverse
        return ev_vote(answers, entropies, used_codes, py_errors)

    elif strategy == 'ev_16_diverse':
        add(8, temp_factor=0.8)
        add(8, temp_factor=1.5)
        return ev_vote(answers, entropies, used_codes, py_errors)

    elif strategy == 'ev_32_diverse':
        add(16, temp_factor=0.8)
        add(16, temp_factor=1.5)
        return ev_vote(answers, entropies, used_codes, py_errors)

    elif strategy == 'ev_cascade_code_only':
        # Base 8. If split, add 8 more BUT only code-verified extra count.
        add(8)

        c = Counter(answers)
        if c.most_common(1)[0][1] < 4:
            # Generate extra, only add code-verified ones
            for _ in range(16):
                a, e, uc, pe, pc, ic = gen_attempt(p)
                if a == -1: a = attractor
                if uc and pe == 0:  # ONLY add code-verified
                    answers.append(a)
                    entropies.append(e)
                    used_codes.append(uc)
                    py_errors.append(pe)

        return ev_vote(answers, entropies, used_codes, py_errors)

    elif strategy == 'ev_cascade_code_only_diverse':
        add(8)

        c = Counter(answers)
        if c.most_common(1)[0][1] < 4:
            for _ in range(16):
                a, e, uc, pe, pc, ic = gen_attempt(p, temp_factor=1.5)
                if a == -1: a = attractor
                if uc and pe == 0:
                    answers.append(a)
                    entropies.append(e)
                    used_codes.append(uc)
                    py_errors.append(pe)

        return ev_vote(answers, entropies, used_codes, py_errors)

    elif strategy == 'ev_adaptive_code_only':
        # BEST IDEA: Adaptive cascade, only counting code-verified extra attempts
        add(8)

        # Check code-verified consensus
        cv = [(a, e) for a, e, uc, pe in zip(answers, entropies, used_codes, py_errors) if uc and pe == 0]
        cv_c = Counter([a for a, e in cv])
        need_more = not (cv_c and cv_c.most_common(1)[0][1] >= 3)

        if need_more:
            # Wave 2: 16 more, only keep code-verified
            for _ in range(16):
                a, e, uc, pe, pc, ic = gen_attempt(p, temp_factor=1.3)
                if a == -1: a = attractor
                if uc and pe == 0:
                    answers.append(a)
                    entropies.append(e)
                    used_codes.append(uc)
                    py_errors.append(pe)

            cv = [(a, e) for a, e, uc, pe in zip(answers, entropies, used_codes, py_errors) if uc and pe == 0]
            cv_c = Counter([a for a, e in cv])
            need_more = not (cv_c and cv_c.most_common(1)[0][1] >= 5)

        if need_more:
            # Wave 3: 24 more, only keep code-verified, even more diverse
            for _ in range(24):
                a, e, uc, pe, pc, ic = gen_attempt(p, temp_factor=1.5)
                if a == -1: a = attractor
                if uc and pe == 0:
                    answers.append(a)
                    entropies.append(e)
                    used_codes.append(uc)
                    py_errors.append(pe)

        return ev_vote(answers, entropies, used_codes, py_errors)

    elif strategy == 'oracle_8':
        add(8)
        return CORRECT if CORRECT in answers else answers[0]

    elif strategy == 'oracle_32':
        add(32)
        return CORRECT if CORRECT in answers else answers[0]

    return 0


def run_all():
    print("=" * 110)
    print("CASCADE DESIGN: Why cascade hurts and how to fix it")
    print("=" * 110)

    strategies = [
        'baseline',
        'ev_8',
        'ev_16',
        'ev_32',
        'ev_8_diverse',
        'ev_16_diverse',
        'ev_32_diverse',
        'ev_cascade_code_only',
        'ev_cascade_code_only_diverse',
        'ev_adaptive_code_only',
        'oracle_8',
        'oracle_32',
    ]

    results = {}
    print(f"\n  {'Strategy':<40} {'Mean':>6} {'Delta':>6} {'Std':>5} {'P(>=39)':>7} {'P(>=41)':>7} {'P(>=44)':>7}")
    print(f"  {'-'*90}")

    baseline_mean = None
    for strat in strategies:
        t0 = time.time()
        scores = np.zeros(N_SIMS, dtype=int)
        for sim in range(N_SIMS):
            for prob in range(50):
                p = DIFFICULTIES[prob]
                att = np.random.randint(0, 100000)
                if att == CORRECT: att = CORRECT + 1
                if sim_problem(p, att, strat) == CORRECT:
                    scores[sim] += 1
        elapsed = time.time() - t0

        if baseline_mean is None:
            baseline_mean = float(scores.mean())

        delta = scores.mean() - baseline_mean
        results[strat] = float(scores.mean())
        print(f"  {strat:<40} {scores.mean():>6.2f} {delta:>+6.2f} {scores.std():>5.2f} "
              f"{np.mean(scores>=39):>7.4f} {np.mean(scores>=41):>7.4f} {np.mean(scores>=44):>7.4f} [{elapsed:.0f}s]")

    # Analysis
    print(f"\n\n{'='*110}")
    print("ANALYSIS: Effect of N and diversity")
    print("="*110)

    # EV at different N
    print("\n  Effect of attempt count (standard temp):")
    for strat in ['ev_8', 'ev_16', 'ev_32']:
        delta = results[strat] - results['ev_8']
        print(f"    {strat}: {results[strat]:.2f}  (vs ev_8: {delta:+.2f})")

    print("\n  Effect of attempt count (diverse temp):")
    for strat in ['ev_8_diverse', 'ev_16_diverse', 'ev_32_diverse']:
        delta = results[strat] - results['ev_8']
        print(f"    {strat}: {results[strat]:.2f}  (vs ev_8: {delta:+.2f})")

    print("\n  Code-only cascade variants:")
    for strat in ['ev_cascade_code_only', 'ev_cascade_code_only_diverse', 'ev_adaptive_code_only']:
        delta = results[strat] - results['ev_8']
        print(f"    {strat}: {results[strat]:.2f}  (vs ev_8: {delta:+.2f})")

    # The best N that doesn't waste too much compute
    print(f"\n  Oracle gap analysis:")
    print(f"    Oracle@8: {results['oracle_8']:.2f}  Oracle@32: {results['oracle_32']:.2f}")
    print(f"    Best achievable@8: {results['ev_8']:.2f}  Gap: {results['oracle_8'] - results['ev_8']:.2f}")


if __name__ == '__main__':
    t0 = time.time()
    run_all()
    print(f"\nTotal: {time.time()-t0:.0f}s")
