"""
AIMO3: Optimize the execution-verified voting parameters.

KEY FINDINGS SO FAR:
1. EXEC_VERIFIED is the best single-strategy improvement: +1.40 avg across scenarios
2. It beats the combined cascade (which uses 4x more compute) by +0.07
3. The oracle gap is 5-7 problems: there IS room, but simple weight tuning has limits

This script:
1. Sweeps the code-verified boost multiplier from 1x to 20x
2. Sweeps the no-code penalty from 0.1x to 1.0x
3. Sweeps the error penalty from 0.05x to 0.5x
4. Tests hybrid strategies: exec_verified + cascade for split votes only
5. Tests the ACTUAL realistic optimization: exec_verified + more attempts on hard problems
"""

import numpy as np
from collections import Counter, defaultdict
import time

np.random.seed(42)
N_SIMS = 5000
N_PROBLEMS = 50
CORRECT = 42

# Use Scenario A as baseline (most conservative / closest to 39/50)
DIFFICULTIES = [0.85]*35 + [0.40]*5 + [0.20]*5 + [0.05]*3 + [0.01]*2
WCP = 0.70
WRONG_ENT_MU, WRONG_ENT_STD = 2.5, 0.5
CONFIDENT_WRONG_ENT_MU = 1.8
CORRECT_ENT_MU, CORRECT_ENT_STD = 1.5, 0.3
CORRECT_CODE_RATE = 0.72
WRONG_CODE_RATE = 0.60
CORRECT_ERROR_RATE = 0.05
WRONG_ERROR_RATE = 0.25


def gen_attempts(n, p):
    attractor = np.random.randint(0, 100000)
    if attractor == CORRECT:
        attractor = CORRECT + 1

    attempts = []
    for _ in range(n):
        is_correct = np.random.random() < p
        if is_correct:
            ans = CORRECT
            ent = max(0.1, np.random.normal(CORRECT_ENT_MU, CORRECT_ENT_STD))
            uc = np.random.random() < CORRECT_CODE_RATE
            pe = 1 if np.random.random() < CORRECT_ERROR_RATE else 0
            pc = max(0, int(np.random.normal(3.0, 1.5))) if uc else 0
        else:
            if np.random.random() < WCP:
                ans = attractor
                ent = max(0.1, np.random.normal(CONFIDENT_WRONG_ENT_MU, 0.3))
            else:
                ans = np.random.randint(0, 100000)
                ent = max(0.1, np.random.normal(WRONG_ENT_MU, WRONG_ENT_STD))
            uc = np.random.random() < WRONG_CODE_RATE
            pe = 1 if np.random.random() < WRONG_ERROR_RATE else 0
            pc = max(0, int(np.random.normal(2.0, 1.5))) if uc else 0

        attempts.append({
            'answer': ans, 'entropy': ent, 'used_code': uc,
            'python_errors': pe, 'python_calls': pc, 'is_correct': is_correct,
        })
    return attempts, attractor


def vote(attempts, code_boost, error_penalty, no_code_penalty, calls_bonus=1.0):
    w = defaultdict(float)
    for a in attempts:
        wt = 1.0 / max(a['entropy'], 1e-9)
        if a['used_code'] and a['python_errors'] == 0:
            wt *= code_boost
        elif a['used_code'] and a['python_errors'] > 0:
            wt *= error_penalty
        elif not a['used_code']:
            wt *= no_code_penalty
        if calls_bonus != 1.0 and 2 <= a['python_calls'] <= 5:
            wt *= calls_bonus
        w[a['answer']] += wt
    return max(w, key=w.get) if w else 0


def simulate(code_boost, error_penalty, no_code_penalty, calls_bonus=1.0):
    scores = np.zeros(N_SIMS, dtype=int)
    for sim in range(N_SIMS):
        for prob_idx in range(N_PROBLEMS):
            p = DIFFICULTIES[prob_idx]
            attempts, _ = gen_attempts(8, p)
            chosen = vote(attempts, code_boost, error_penalty, no_code_penalty, calls_bonus)
            if chosen == CORRECT:
                scores[sim] += 1
    return float(np.mean(scores)), float(np.std(scores)), float(np.mean(scores >= 44))


# ============================================================
# SWEEP 1: Code boost multiplier
# ============================================================
print("=" * 90)
print("SWEEP 1: Code-verified boost multiplier")
print("=" * 90)
print(f"{'Boost':>8} {'Error':>8} {'NoCod':>8} | {'Mean':>6} {'Std':>5} {'Delta':>6} {'P(44)':>6}")
print("-" * 60)

# First get baseline
baseline_mean, baseline_std, baseline_p44 = simulate(1.0, 1.0, 1.0)
print(f"{'1.0':>8} {'1.0':>8} {'1.0':>8} | {baseline_mean:>6.2f} {baseline_std:>5.2f} {0.0:>+6.2f} {baseline_p44:>6.4f}  (BASELINE)")

best_mean = baseline_mean
best_params = None

for code_boost in [1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]:
    for error_pen in [0.1, 0.2, 0.3, 0.5]:
        for no_code in [0.2, 0.3, 0.5, 0.7, 1.0]:
            mean, std, p44 = simulate(code_boost, error_pen, no_code)
            delta = mean - baseline_mean
            if mean > best_mean:
                best_mean = mean
                best_params = (code_boost, error_pen, no_code)
                marker = " <-- NEW BEST"
            else:
                marker = ""
            if delta > 0.5 or (code_boost in [3.0, 5.0, 10.0] and no_code == 0.5 and error_pen == 0.3):
                print(f"{code_boost:>8.1f} {error_pen:>8.2f} {no_code:>8.2f} | {mean:>6.2f} {std:>5.2f} {delta:>+6.2f} {p44:>6.4f}{marker}")

print(f"\nBest parameters: code_boost={best_params[0]}, error_pen={best_params[1]}, no_code={best_params[2]}")
print(f"Best mean: {best_mean:.2f} (delta: +{best_mean - baseline_mean:.2f})")

# ============================================================
# SWEEP 2: Adding calls_bonus on top of best params
# ============================================================
print(f"\n{'=' * 90}")
print(f"SWEEP 2: Adding python_calls bonus (2-5 calls sweet spot)")
print(f"{'=' * 90}")

for calls_bonus in [1.0, 1.1, 1.2, 1.3, 1.5, 2.0]:
    mean, std, p44 = simulate(best_params[0], best_params[1], best_params[2], calls_bonus)
    delta = mean - baseline_mean
    print(f"  calls_bonus={calls_bonus:.1f}: mean={mean:.2f} delta={delta:+.2f} P(44)={p44:.4f}")

# ============================================================
# SWEEP 3: More attempts for hard problems (the CASCADE part)
# ============================================================
print(f"\n{'=' * 90}")
print(f"SWEEP 3: Extra attempts for split votes (exec-verified cascade)")
print(f"{'=' * 90}")

def simulate_extra_for_split(n_extra, consensus_threshold, code_boost, error_pen, no_code_pen):
    """Run extra attempts only when base 8 have a split vote."""
    scores = np.zeros(N_SIMS, dtype=int)
    total_extra_attempts = 0

    for sim in range(N_SIMS):
        for prob_idx in range(N_PROBLEMS):
            p = DIFFICULTIES[prob_idx]
            attempts, attractor = gen_attempts(8, p)

            # Check consensus
            counts = Counter(a['answer'] for a in attempts)
            top_ans, top_cnt = counts.most_common(1)[0]

            if top_cnt < consensus_threshold:
                # Split vote - run extra attempts
                extra, _ = gen_attempts(n_extra, p)
                # Reuse same attractor
                for a in extra:
                    if not a['is_correct'] and np.random.random() < WCP:
                        a['answer'] = attractor
                attempts.extend(extra)
                total_extra_attempts += n_extra

            chosen = vote(attempts, code_boost, error_pen, no_code_pen)
            if chosen == CORRECT:
                scores[sim] += 1

    avg_extra = total_extra_attempts / N_SIMS
    return float(np.mean(scores)), float(np.std(scores)), float(np.mean(scores >= 44)), avg_extra


print(f"  {'Extra':>6} {'Thresh':>6} | {'Mean':>6} {'Delta':>6} {'P(44)':>6} {'AvgExtra':>8}")
print(f"  {'-'*55}")

for n_extra in [4, 8, 16, 24]:
    for threshold in [3, 4, 5]:
        mean, std, p44, avg_extra = simulate_extra_for_split(
            n_extra, threshold, best_params[0], best_params[1], best_params[2])
        delta = mean - baseline_mean
        print(f"  {n_extra:>6} {threshold:>6} | {mean:>6.2f} {delta:>+6.2f} {p44:>6.4f} {avg_extra:>8.1f}")

# ============================================================
# FINAL: THE OPTIMAL STRATEGY
# ============================================================
print(f"\n{'=' * 90}")
print(f"FINAL RECOMMENDATION")
print(f"{'=' * 90}")

# Run the optimal with larger N
np.random.seed(123)
N_SIMS_FINAL = 10000

baseline_scores = np.zeros(N_SIMS_FINAL, dtype=int)
optimal_scores = np.zeros(N_SIMS_FINAL, dtype=int)
oracle_scores = np.zeros(N_SIMS_FINAL, dtype=int)

cb, ep, ncp = best_params

for sim in range(N_SIMS_FINAL):
    for prob_idx in range(N_PROBLEMS):
        p = DIFFICULTIES[prob_idx]
        attempts, attractor = gen_attempts(8, p)

        # Baseline
        w_base = defaultdict(float)
        for a in attempts:
            w_base[a['answer']] += 1.0 / max(a['entropy'], 1e-9)
        if max(w_base, key=w_base.get) == CORRECT:
            baseline_scores[sim] += 1

        # Optimal (exec-verified voting)
        if vote(attempts, cb, ep, ncp) == CORRECT:
            optimal_scores[sim] += 1

        # Oracle
        if any(a['is_correct'] for a in attempts):
            oracle_scores[sim] += 1

b = baseline_scores
o = optimal_scores
r = oracle_scores

print(f"\n  BASELINE:  mean={b.mean():.2f}  std={b.std():.2f}  P(>=39)={np.mean(b>=39):.4f}  P(>=44)={np.mean(b>=44):.4f}")
print(f"  OPTIMAL:   mean={o.mean():.2f}  std={o.std():.2f}  P(>=39)={np.mean(o>=39):.4f}  P(>=44)={np.mean(o>=44):.4f}")
print(f"  ORACLE:    mean={r.mean():.2f}  std={r.std():.2f}  P(>=39)={np.mean(r>=39):.4f}  P(>=44)={np.mean(r>=44):.4f}")
print(f"\n  Improvement: +{o.mean() - b.mean():.2f} problems")
print(f"  Oracle gap: {r.mean() - o.mean():.2f} problems")

print(f"\n  Best parameters:")
print(f"    code_verified_boost = {cb}")
print(f"    error_penalty = {ep}")
print(f"    no_code_penalty = {ncp}")

print(f"\n  Implementation: Change _select_answer to multiply weight by:")
print(f"    {cb}x if used_code AND python_errors == 0")
print(f"    {ep}x if used_code AND python_errors > 0")
print(f"    {ncp}x if NOT used_code (no Python execution)")

# Distribution comparison
print(f"\n  Score distribution:")
for score in range(35, 50):
    bp = np.mean(b == score)
    op = np.mean(o == score)
    bar_b = '#' * int(bp * 200)
    bar_o = '#' * int(op * 200)
    print(f"    {score}: baseline {bp:.4f} {bar_b}")
    print(f"    {score}: optimal  {op:.4f} {bar_o}")
