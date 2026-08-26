"""
AIMO3 Voting Strategy Analysis - REALISTIC VERSION
Accounts for the fact that wrong answers are NOT purely random:
- Some wrong answers cluster (systematic errors, common mistakes)
- Models can produce the same wrong answer multiple times
"""
import json
import numpy as np
from math import comb
from collections import Counter

np.random.seed(42)

N = 8
p = 0.69
q = 1 - p
NUM_PROBLEMS = 100000
CORRECT_ANSWER = 42

print("="*70)
print("PART 1: THEORETICAL (wrong answers perfectly random)")
print("="*70)

# Exact theoretical calculations
majority_strict = sum(comb(N, k) * p**k * q**(N-k) for k in range(5, N+1))
p_k4 = comb(N, 4) * p**4 * q**4
majority_effective = majority_strict + p_k4
appears_once = 1 - q**N
gap = appears_once - majority_effective

print(f"P(strict majority >=5/8)       = {majority_strict:.6f}")
print(f"P(plurality >=4/8, wrongs scatter) = {majority_effective:.6f}")
print(f"P(correct appears at least once)   = {appears_once:.6f}")
print(f"Gap (improvement ceiling)          = {gap:.6f} ({gap/majority_effective*100:.1f}% relative)")

print(f"\n{'='*70}")
print("PART 2: REALISTIC SIMULATION")
print("Wrong answers have partial clustering (systematic errors)")
print("="*70)

def simulate_realistic(num_problems, n_attempts, p_correct, wrong_cluster_prob=0.3):
    """
    Realistic simulation where wrong answers partially cluster.

    wrong_cluster_prob: probability that a wrong answer copies a "systematic error"
    rather than being purely random. This means multiple wrong attempts can
    converge on the SAME wrong answer.
    """
    is_correct = np.random.random((num_problems, n_attempts)) < p_correct
    answers = np.full((num_problems, n_attempts), -1, dtype=np.int64)
    entropies = np.zeros((num_problems, n_attempts))
    python_errors = np.zeros((num_problems, n_attempts), dtype=int)

    for i in range(num_problems):
        # Each problem has 1-3 "attractor" wrong answers (common mistakes)
        n_attractors = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
        attractors = np.random.randint(1, 100000, size=n_attractors)
        # Make sure attractors != correct answer
        attractors = attractors[attractors != CORRECT_ANSWER]
        if len(attractors) == 0:
            attractors = np.array([CORRECT_ANSWER + 1])

        for j in range(n_attempts):
            if is_correct[i, j]:
                answers[i, j] = CORRECT_ANSWER
                entropies[i, j] = max(0.1, np.random.normal(1.5, 0.3))
                python_errors[i, j] = 1 if np.random.random() < 0.05 else 0
            else:
                if np.random.random() < wrong_cluster_prob:
                    # Pick from attractor wrong answers
                    answers[i, j] = np.random.choice(attractors)
                else:
                    # Random wrong answer
                    answers[i, j] = np.random.randint(1, 100000)
                entropies[i, j] = max(0.1, np.random.normal(2.5, 0.5))
                python_errors[i, j] = 1 if np.random.random() < 0.25 else 0

    return answers, entropies, is_correct, python_errors


# ---- All strategies ----
def simple_majority(answers_row):
    c = Counter(answers_row)
    return c.most_common(1)[0][0]

def weighted_majority(answers_row, entropy_row):
    weighted = {}
    for ans, ent in zip(answers_row, entropy_row):
        weighted[ans] = weighted.get(ans, 0) + 1.0 / max(ent, 0.1)
    return max(weighted, key=weighted.get)

def cluster_aware(answers_row, entropy_row):
    c = Counter(answers_row)
    clustered = [(ans, cnt) for ans, cnt in c.items() if cnt >= 2]
    if len(clustered) == 1:
        return clustered[0][0]
    elif len(clustered) > 1:
        return max(clustered, key=lambda x: x[1])[0]
    else:
        idx = np.argmin(entropy_row)
        return answers_row[idx]

def cluster_entropy_tiebreak(answers_row, entropy_row):
    """Cluster-aware with entropy tiebreak for same-count clusters."""
    c = Counter(answers_row)
    clustered = [(ans, cnt) for ans, cnt in c.items() if cnt >= 2]
    if len(clustered) == 0:
        idx = np.argmin(entropy_row)
        return answers_row[idx]
    elif len(clustered) == 1:
        return clustered[0][0]
    else:
        # Multiple clusters: pick by count, then by avg entropy
        def score(ans, cnt):
            avg_ent = np.mean([e for a, e in zip(answers_row, entropy_row) if a == ans])
            return (-cnt, avg_ent)  # higher count wins, lower entropy breaks ties
        clustered.sort(key=lambda x: score(x[0], x[1]))
        return clustered[0][0]

def cluster_weighted_entropy(answers_row, entropy_row):
    """
    Score each answer as: count * (1/avg_entropy)
    This combines cluster size with confidence.
    """
    c = Counter(answers_row)
    scores = {}
    for ans in c:
        cnt = c[ans]
        avg_ent = np.mean([e for a, e in zip(answers_row, entropy_row) if a == ans])
        scores[ans] = cnt * (1.0 / max(avg_ent, 0.1))
    return max(scores, key=scores.get)

def cluster_count_plus_entropy(answers_row, entropy_row):
    """
    Score = count + alpha * (1/avg_entropy)
    Tuned so that count dominates but entropy breaks close ties.
    """
    c = Counter(answers_row)
    alpha = 0.5
    scores = {}
    for ans in c:
        cnt = c[ans]
        avg_ent = np.mean([e for a, e in zip(answers_row, entropy_row) if a == ans])
        scores[ans] = cnt + alpha * (1.0 / max(avg_ent, 0.1))
    return max(scores, key=scores.get)

def error_filtered_cluster(answers_row, entropy_row, error_row):
    """Filter out error attempts, then cluster-aware."""
    filtered_a = []
    filtered_e = []
    for a, e, err in zip(answers_row, entropy_row, error_row):
        if err == 0:
            filtered_a.append(a)
            filtered_e.append(e)
    if not filtered_a:
        filtered_a = list(answers_row)
        filtered_e = list(entropy_row)
    return cluster_entropy_tiebreak(filtered_a, filtered_e)

def error_penalized_cluster(answers_row, entropy_row, error_row):
    """
    Score each answer: for each attempt, add weight.
    Weight = (1/entropy) * (0.3 if has_error else 1.0)
    Then pick highest total weight, but ONLY if it appears >= 2 times.
    """
    c = Counter(answers_row)
    scores = {}
    for a, e, err in zip(answers_row, entropy_row, error_row):
        penalty = 0.3 if err > 0 else 1.0
        scores[a] = scores.get(a, 0) + penalty / max(e, 0.1)

    # Prefer answers appearing >= 2 times
    clustered = {a: s for a, s in scores.items() if c[a] >= 2}
    if clustered:
        return max(clustered, key=clustered.get)
    return max(scores, key=scores.get)

def oracle(answers_row):
    if CORRECT_ANSWER in answers_row:
        return CORRECT_ANSWER
    return answers_row[0]


# Run for different wrong_cluster_prob values
for wcp in [0.0, 0.15, 0.30, 0.50]:
    print(f"\n{'~'*70}")
    print(f"  Wrong answer clustering probability = {wcp:.0%}")
    print(f"{'~'*70}")

    answers, entropies, is_correct, python_errors = simulate_realistic(
        NUM_PROBLEMS, N, p, wrong_cluster_prob=wcp
    )

    strategies = {
        'A_simple_majority': lambda a, e, err: simple_majority(a),
        'B_weighted_1/entropy': lambda a, e, err: weighted_majority(a, e),
        'C_cluster_aware': lambda a, e, err: cluster_aware(a, e),
        'D_cluster_entropy_tiebreak': lambda a, e, err: cluster_entropy_tiebreak(a, e),
        'E_cluster_weighted_entropy': lambda a, e, err: cluster_weighted_entropy(a, e),
        'F_count+0.5/entropy': lambda a, e, err: cluster_count_plus_entropy(a, e),
        'G_error_filtered_cluster': lambda a, e, err: error_filtered_cluster(a, e, err),
        'H_error_penalized_cluster': lambda a, e, err: error_penalized_cluster(a, e, err),
        'Z_oracle': lambda a, e, err: oracle(a),
    }

    strat_results = {}
    for name, func in strategies.items():
        correct_count = 0
        for i in range(NUM_PROBLEMS):
            pred = func(answers[i], entropies[i], python_errors[i])
            if pred == CORRECT_ANSWER:
                correct_count += 1
        acc = correct_count / NUM_PROBLEMS
        strat_results[name] = acc
        print(f"  {name:40s}: {acc:.5f} ({correct_count}/{NUM_PROBLEMS})")

    # Count how often wrong answers cluster
    wrong_cluster_events = 0
    for i in range(NUM_PROBLEMS):
        wrong_answers = [a for a, c in zip(answers[i], is_correct[i]) if not c]
        if len(wrong_answers) > 1:
            wc = Counter(wrong_answers)
            if wc.most_common(1)[0][1] >= 2:
                wrong_cluster_events += 1

    has_wrong = np.sum(np.sum(~is_correct, axis=1) > 1)
    print(f"\n  Wrong answers cluster (>=2 same wrong): {wrong_cluster_events}/{has_wrong} "
          f"= {wrong_cluster_events/has_wrong:.3f}")


print(f"\n\n{'='*70}")
print("PART 3: IMPACT ON COMPETITION SCORE (50 problems)")
print("="*70)

# With 50 problems, what's the expected score?
n_problems_comp = 50

for wcp_label, wcp in [("ideal (0%)", 0.0), ("moderate (30%)", 0.3), ("heavy (50%)", 0.5)]:
    print(f"\n  Wrong clustering = {wcp_label}:")

    # Monte Carlo: simulate 10000 competitions
    n_sims = 10000
    scores = {name: [] for name in ['majority', 'cluster_entropy', 'error_penalized', 'oracle']}

    for _ in range(n_sims):
        answers, entropies, is_correct, python_errors = simulate_realistic(
            n_problems_comp, N, p, wrong_cluster_prob=wcp
        )

        s_maj = sum(simple_majority(answers[i]) == CORRECT_ANSWER for i in range(n_problems_comp))
        s_ce = sum(cluster_entropy_tiebreak(answers[i], entropies[i]) == CORRECT_ANSWER for i in range(n_problems_comp))
        s_ep = sum(error_penalized_cluster(answers[i], entropies[i], python_errors[i]) == CORRECT_ANSWER for i in range(n_problems_comp))
        s_or = sum(oracle(answers[i]) == CORRECT_ANSWER for i in range(n_problems_comp))

        scores['majority'].append(s_maj)
        scores['cluster_entropy'].append(s_ce)
        scores['error_penalized'].append(s_ep)
        scores['oracle'].append(s_or)

    for name in scores:
        arr = np.array(scores[name])
        print(f"    {name:25s}: mean={arr.mean():.2f}/50  std={arr.std():.2f}  "
              f"P(>=44)={np.mean(arr >= 44):.3f}  P(>=45)={np.mean(arr >= 45):.3f}")


print(f"\n\n{'='*70}")
print("PART 4: OPTIMAL STRATEGY RECOMMENDATION")
print("="*70)

# The key realistic scenario
answers, entropies, is_correct, python_errors = simulate_realistic(
    NUM_PROBLEMS, N, p, wrong_cluster_prob=0.30
)

# Best strategies compared
final_strategies = {
    'Current (weighted 1/ent)': lambda a, e, err: weighted_majority(a, e),
    'Cluster+entropy tiebreak': lambda a, e, err: cluster_entropy_tiebreak(a, e),
    'Count * (1/avg_ent)': lambda a, e, err: cluster_weighted_entropy(a, e),
    'Count + 0.5/avg_ent': lambda a, e, err: cluster_count_plus_entropy(a, e),
    'Error-penalized cluster': lambda a, e, err: error_penalized_cluster(a, e, err),
    'Oracle (upper bound)': lambda a, e, err: oracle(a),
}

print(f"\nWith 30% wrong-answer clustering (realistic estimate):")
final_results = {}
for name, func in final_strategies.items():
    correct_count = sum(
        func(answers[i], entropies[i], python_errors[i]) == CORRECT_ANSWER
        for i in range(NUM_PROBLEMS)
    )
    acc = correct_count / NUM_PROBLEMS
    final_results[name] = acc
    print(f"  {name:35s}: {acc:.5f}")

# ============================================================
# FINAL JSON OUTPUT
# ============================================================
best_name = max(
    {k: v for k, v in final_results.items() if 'Oracle' not in k},
    key=final_results.get
)

output = {
    "majority_vote_accuracy_theoretical": round(majority_effective, 6),
    "majority_vote_accuracy_simulated_realistic": round(final_results.get('Current (weighted 1/ent)', 0), 5),
    "appears_at_least_once": round(appears_once, 6),
    "improvement_ceiling": round(gap, 6),
    "best_strategy_name": best_name,
    "best_strategy_accuracy": round(final_results[best_name], 5),
    "oracle_upper_bound": round(final_results['Oracle (upper bound)'], 5),
    "expected_score_50_problems": {
        "current_weighted_majority": round(final_results.get('Current (weighted 1/ent)', 0) * 50, 1),
        "best_strategy": round(final_results[best_name] * 50, 1),
        "oracle": round(final_results['Oracle (upper bound)'] * 50, 1),
    },
    "key_insight": (
        "The critical variable is HOW MUCH wrong answers cluster. "
        "With random wrongs, all cluster-based strategies are near-perfect (~99.95%). "
        "With 30% wrong clustering, the gap between strategies widens. "
        "The best approach: score = count * (1/avg_entropy), which naturally "
        "handles both clustering detection AND confidence weighting. "
        "Error filtering adds marginal value."
    ),
    "recommendation": (
        "Use score = count * (1/avg_entropy) for each unique answer, pick highest. "
        "This is better than pure majority vote AND better than pure 1/entropy weighting. "
        "It exploits the key asymmetry: correct answers cluster AND have lower entropy, "
        "so they score high on both dimensions simultaneously."
    ),
}

print(f"\n{json.dumps(output, indent=2)}")
