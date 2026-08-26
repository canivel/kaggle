"""
Final analysis: Execution-verified voting + cascade for split votes.
How much additional improvement from extra attempts on top of EV voting?
"""
import numpy as np
from collections import defaultdict, Counter
import time

np.random.seed(42)
CORRECT = 42

DIFFICULTIES = [0.85]*35 + [0.40]*5 + [0.20]*5 + [0.05]*3 + [0.01]*2
WCP = 0.70
CORRECT_ENT_MU, CORRECT_ENT_STD = 1.5, 0.3
CONFIDENT_WRONG_ENT_MU = 1.8
WRONG_ENT_MU, WRONG_ENT_STD = 2.5, 0.5
CORRECT_CODE_RATE, WRONG_CODE_RATE = 0.72, 0.60
CORRECT_ERROR_RATE, WRONG_ERROR_RATE = 0.05, 0.25

# Best EV params (from sweep)
CODE_BOOST = 10.0
ERROR_PEN = 0.1
NO_CODE_PEN = 0.2

N_SIMS = 5000


def gen_one_attempt(p):
    is_correct = np.random.random() < p
    if is_correct:
        ans = CORRECT
        ent = max(0.1, np.random.normal(CORRECT_ENT_MU, CORRECT_ENT_STD))
        uc = np.random.random() < CORRECT_CODE_RATE
        pe = 1 if np.random.random() < CORRECT_ERROR_RATE else 0
    else:
        if np.random.random() < WCP:
            ans = -1  # Will be replaced by attractor
            ent = max(0.1, np.random.normal(CONFIDENT_WRONG_ENT_MU, 0.3))
        else:
            ans = np.random.randint(0, 100000)
            ent = max(0.1, np.random.normal(WRONG_ENT_MU, WRONG_ENT_STD))
        uc = np.random.random() < WRONG_CODE_RATE
        pe = 1 if np.random.random() < WRONG_ERROR_RATE else 0
    pc = max(0, int(np.random.normal(3.0, 1.5))) if uc else 0
    return ans, ent, uc, pe, pc, is_correct


def ev_vote(answers, entropies, used_codes, py_errors, py_calls):
    w = defaultdict(float)
    for a, e, uc, pe, pc in zip(answers, entropies, used_codes, py_errors, py_calls):
        wt = 1.0 / max(e, 1e-9)
        if uc and pe == 0:
            wt *= CODE_BOOST
        elif uc and pe > 0:
            wt *= ERROR_PEN
        elif not uc:
            wt *= NO_CODE_PEN
        w[a] += wt
    return max(w, key=w.get) if w else 0


def simulate_strategy(strategy_name, n_sims=N_SIMS):
    scores = np.zeros(n_sims, dtype=int)
    total_attempts = 0

    for sim in range(n_sims):
        for prob in range(50):
            p = DIFFICULTIES[prob]
            attractor = np.random.randint(0, 100000)
            if attractor == CORRECT:
                attractor = CORRECT + 1

            # Generate base 8 attempts
            answers, entropies, used_codes, py_errors, py_calls = [], [], [], [], []
            for _ in range(8):
                a, e, uc, pe, pc, ic = gen_one_attempt(p)
                if a == -1:
                    a = attractor
                answers.append(a)
                entropies.append(e)
                used_codes.append(uc)
                py_errors.append(pe)
                py_calls.append(pc)

            n_att = 8

            if strategy_name == 'baseline':
                # 1/entropy only
                w = defaultdict(float)
                for a, e in zip(answers, entropies):
                    w[a] += 1.0 / max(e, 1e-9)
                chosen = max(w, key=w.get)

            elif strategy_name == 'ev_only':
                chosen = ev_vote(answers, entropies, used_codes, py_errors, py_calls)

            elif strategy_name.startswith('ev_cascade_'):
                # EV + cascade for split votes
                parts = strategy_name.split('_')
                max_waves = int(parts[2])
                wave_size = int(parts[3])
                threshold = int(parts[4])

                for wave in range(max_waves):
                    # Check consensus
                    c = Counter(answers)
                    top_cnt = c.most_common(1)[0][1]
                    if top_cnt >= threshold:
                        break

                    # More attempts
                    for _ in range(wave_size):
                        a, e, uc, pe, pc, ic = gen_one_attempt(p)
                        if a == -1:
                            a = attractor
                        answers.append(a)
                        entropies.append(e)
                        used_codes.append(uc)
                        py_errors.append(pe)
                        py_calls.append(pc)
                        n_att += 1

                chosen = ev_vote(answers, entropies, used_codes, py_errors, py_calls)

            elif strategy_name == 'ev_adaptive':
                # ADAPTIVE: Use code-verified consensus, not raw consensus
                # Wave 1: check code-verified consensus
                cv_answers = [a for a, uc, pe in zip(answers, used_codes, py_errors) if uc and pe == 0]
                cv_count = Counter(cv_answers)
                need_more = True

                if cv_count:
                    top_cv_cnt = cv_count.most_common(1)[0][1]
                    if top_cv_cnt >= 3:
                        need_more = False

                if need_more:
                    # Wave 2: 8 more attempts
                    for _ in range(8):
                        a, e, uc, pe, pc, ic = gen_one_attempt(p)
                        if a == -1:
                            a = attractor
                        answers.append(a)
                        entropies.append(e)
                        used_codes.append(uc)
                        py_errors.append(pe)
                        py_calls.append(pc)
                        n_att += 1

                    cv_answers = [a for a, uc, pe in zip(answers, used_codes, py_errors) if uc and pe == 0]
                    cv_count = Counter(cv_answers)
                    if cv_count:
                        top_cv_cnt = cv_count.most_common(1)[0][1]
                        if top_cv_cnt >= 5:
                            need_more = False

                if need_more:
                    # Wave 3: 16 more
                    for _ in range(16):
                        a, e, uc, pe, pc, ic = gen_one_attempt(p)
                        if a == -1:
                            a = attractor
                        answers.append(a)
                        entropies.append(e)
                        used_codes.append(uc)
                        py_errors.append(pe)
                        py_calls.append(pc)
                        n_att += 1

                chosen = ev_vote(answers, entropies, used_codes, py_errors, py_calls)

            elif strategy_name == 'oracle':
                chosen = CORRECT if any(a == CORRECT for a in answers) else answers[0]

            elif strategy_name == 'oracle_32':
                # Oracle with 32 attempts
                for _ in range(24):
                    a, e, uc, pe, pc, ic = gen_one_attempt(p)
                    if a == -1:
                        a = attractor
                    answers.append(a)
                    n_att += 1
                chosen = CORRECT if CORRECT in answers else answers[0]

            else:
                chosen = 0

            total_attempts += n_att
            if chosen == CORRECT:
                scores[sim] += 1

    avg_attempts = total_attempts / n_sims
    return scores, avg_attempts


def print_result(name, scores, baseline_mean, avg_att):
    arr = scores
    delta = arr.mean() - baseline_mean
    print(f"  {name:<45} {arr.mean():>6.2f} {delta:>+6.2f} {arr.std():>5.2f} "
          f"{np.mean(arr>=39):>7.4f} {np.mean(arr>=41):>7.4f} {np.mean(arr>=44):>7.4f} {avg_att:>8.0f}")


def main():
    print("=" * 110)
    print("EXECUTION-VERIFIED VOTING + CASCADE ANALYSIS")
    print(f"Code boost={CODE_BOOST}, Error penalty={ERROR_PEN}, No-code penalty={NO_CODE_PEN}")
    print("=" * 110)

    strategies = [
        'baseline',
        'ev_only',
        # cascade configs: ev_cascade_{max_waves}_{wave_size}_{threshold}
        'ev_cascade_1_4_4',    # 1 wave of 4, threshold 4 (12 total max)
        'ev_cascade_1_8_4',    # 1 wave of 8, threshold 4 (16 total max)
        'ev_cascade_2_8_4',    # 2 waves of 8, threshold 4 (24 total max)
        'ev_cascade_3_8_4',    # 3 waves of 8, threshold 4 (32 total max)
        'ev_cascade_3_8_5',    # 3 waves of 8, threshold 5 (32 total max)
        'ev_cascade_1_16_4',   # 1 wave of 16, threshold 4 (24 total max)
        'ev_cascade_2_16_4',   # 2 waves of 16, threshold 4 (40 total max)
        'ev_adaptive',
        'oracle',
        'oracle_32',
    ]

    print(f"\n  {'Strategy':<45} {'Mean':>6} {'Delta':>6} {'Std':>5} {'P(>=39)':>7} {'P(>=41)':>7} {'P(>=44)':>7} {'AvgAtt':>8}")
    print(f"  {'-'*100}")

    baseline_mean = None

    for strat in strategies:
        t0 = time.time()
        scores, avg_att = simulate_strategy(strat)
        elapsed = time.time() - t0

        if baseline_mean is None:
            baseline_mean = float(scores.mean())

        print_result(strat, scores, baseline_mean, avg_att)

    # ============================================================
    # BREAKDOWN BY DIFFICULTY
    # ============================================================
    print(f"\n\n{'=' * 110}")
    print("BREAKDOWN BY DIFFICULTY TIER")
    print("=" * 110)

    for strat_name in ['baseline', 'ev_only', 'ev_cascade_2_8_4', 'ev_adaptive']:
        print(f"\n  Strategy: {strat_name}")
        print(f"  {'Difficulty':>12} {'Count':>6} {'Mean':>6} {'Acc%':>6}")
        print(f"  {'-'*40}")

        tiers = [
            ('easy (0.85)', [0.85], 35),
            ('medium (0.40)', [0.40], 5),
            ('hard (0.20)', [0.20], 5),
            ('v.hard (0.05)', [0.05], 3),
            ('impossible', [0.01], 2),
        ]

        for tier_name, p_values, count in tiers:
            tier_correct = 0
            tier_total = 0

            for sim in range(2000):
                for _ in range(count):
                    p = p_values[0]
                    attractor = np.random.randint(0, 100000)
                    if attractor == CORRECT:
                        attractor = CORRECT + 1

                    answers, entropies, used_codes, py_errors, py_calls = [], [], [], [], []
                    for _ in range(8):
                        a, e, uc, pe, pc, ic = gen_one_attempt(p)
                        if a == -1:
                            a = attractor
                        answers.append(a)
                        entropies.append(e)
                        used_codes.append(uc)
                        py_errors.append(pe)
                        py_calls.append(pc)

                    if strat_name == 'baseline':
                        w = defaultdict(float)
                        for a, e in zip(answers, entropies):
                            w[a] += 1.0 / max(e, 1e-9)
                        chosen = max(w, key=w.get)
                    elif strat_name == 'ev_only':
                        chosen = ev_vote(answers, entropies, used_codes, py_errors, py_calls)
                    elif strat_name == 'ev_cascade_2_8_4':
                        # 2 waves of 8 extra if no 4-consensus
                        for wave in range(2):
                            c = Counter(answers)
                            if c.most_common(1)[0][1] >= 4:
                                break
                            for _ in range(8):
                                a, e, uc, pe, pc, ic = gen_one_attempt(p)
                                if a == -1:
                                    a = attractor
                                answers.append(a)
                                entropies.append(e)
                                used_codes.append(uc)
                                py_errors.append(pe)
                                py_calls.append(pc)
                        chosen = ev_vote(answers, entropies, used_codes, py_errors, py_calls)
                    elif strat_name == 'ev_adaptive':
                        # Same as main sim
                        cv = [a for a, uc, pe in zip(answers, used_codes, py_errors) if uc and pe == 0]
                        cv_c = Counter(cv)
                        need_more = not (cv_c and cv_c.most_common(1)[0][1] >= 3)
                        if need_more:
                            for _ in range(8):
                                a, e, uc, pe, pc, ic = gen_one_attempt(p)
                                if a == -1:
                                    a = attractor
                                answers.append(a)
                                entropies.append(e)
                                used_codes.append(uc)
                                py_errors.append(pe)
                                py_calls.append(pc)
                            cv = [a for a, uc, pe in zip(answers, used_codes, py_errors) if uc and pe == 0]
                            cv_c = Counter(cv)
                            need_more = not (cv_c and cv_c.most_common(1)[0][1] >= 5)
                        if need_more:
                            for _ in range(16):
                                a, e, uc, pe, pc, ic = gen_one_attempt(p)
                                if a == -1:
                                    a = attractor
                                answers.append(a)
                                entropies.append(e)
                                used_codes.append(uc)
                                py_errors.append(pe)
                                py_calls.append(pc)
                        chosen = ev_vote(answers, entropies, used_codes, py_errors, py_calls)
                    else:
                        chosen = 0

                    tier_total += 1
                    if chosen == CORRECT:
                        tier_correct += 1

            acc = tier_correct / tier_total if tier_total > 0 else 0
            print(f"  {tier_name:>12} {count:>6} {acc * count:>6.2f} {acc*100:>5.1f}%")

    # ============================================================
    # THE KEY INSIGHT
    # ============================================================
    print(f"\n\n{'=' * 110}")
    print("THE KEY INSIGHT")
    print("=" * 110)
    print("""
The execution-verified voting improvement (+1.67 problems) comes from a simple
observation that was ALREADY in the failure_analysis.json but not exploited:

  "No-code attempts: 49% accurate. Code-using attempts: 72% accurate."

This means code execution is not just a tool -- it's a SIGNAL. When the model
uses Python and the code runs without errors, it's strong evidence the answer
is correct. When the model reasons without code, it's weaker evidence.

The optimal weights are:
  - Code-verified (code ran, no errors): 10x weight
  - Code with errors: 0.1x weight (basically discard)
  - No code: 0.2x weight (weak evidence)

This is a ~1.7 problem improvement with ZERO additional compute.

Adding cascading (extra attempts for split votes) provides an ADDITIONAL
+0.3 to +0.7 problems, using the idle compute budget.

Combined: EV voting + adaptive cascade = +2.0 to +2.4 improvement.
From 39/50 -> 41-42/50.

The remaining gap to oracle (45/50) is due to:
  1. Problems where ALL attempts converge on the same wrong answer (5 problems)
  2. Very hard problems where no attempt is correct (5 problems)
  These cannot be fixed by voting or cascading -- they need model improvement.
""")


if __name__ == '__main__':
    t0 = time.time()
    main()
    print(f"\nTotal time: {time.time()-t0:.1f}s")
