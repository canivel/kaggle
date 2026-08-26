"""
AIMO3 Voting Strategy Analysis
N=8 attempts per problem, p=0.69 per-attempt accuracy
Wrong answers are random integers mod 100000
"""
import json
import numpy as np
from math import comb
from collections import Counter

np.random.seed(42)

# ============================================================
# 1. Exact P(majority vote correct) for N=8, p=0.69
# ============================================================
N = 8
p = 0.69
q = 1 - p

# Strict majority = at least 5 of 8 correct
majority_prob = sum(comb(N, k) * p**k * q**(N-k) for k in range(5, N+1))
print(f"1. P(strict majority vote correct, >=5 of 8) = {majority_prob:.6f}")

# With random wrong answers (mod 100000), ties at k=4 are won by correct answer
# because the 4 wrong answers are almost certainly all different
# P(any two wrong match) ~ C(4,2)/100000 ~ 0.00006 ≈ 0
p_k4 = comb(N, 4) * p**4 * q**4
effective_majority = majority_prob + p_k4
print(f"   P(plurality correct, k>=4 since wrongs scatter) = {effective_majority:.6f}")

# ============================================================
# 2. P(correct appears at least once)
# ============================================================
appears_once = 1 - q**N
print(f"\n2. P(correct appears at least once) = {appears_once:.6f}")

# ============================================================
# 3. Improvement ceiling
# ============================================================
gap = appears_once - effective_majority
print(f"\n3. Improvement ceiling (gap) = {gap:.6f}")
print(f"   This is {gap/effective_majority*100:.1f}% relative improvement possible")

# ============================================================
# 4 & 5 & 6. Full simulation
# ============================================================
NUM_PROBLEMS = 100000
CORRECT_ANSWER = 42

def simulate_attempts(num_problems, n_attempts, p_correct):
    is_correct = np.random.random((num_problems, n_attempts)) < p_correct
    answers = np.where(
        is_correct,
        CORRECT_ANSWER,
        np.random.randint(1, 100000, size=(num_problems, n_attempts))
    )
    # Correct answers: lower entropy; Wrong answers: higher entropy
    entropies = np.where(
        is_correct,
        np.random.normal(1.5, 0.3, size=(num_problems, n_attempts)),
        np.random.normal(2.5, 0.5, size=(num_problems, n_attempts))
    )
    entropies = np.clip(entropies, 0.1, 5.0)
    return answers, entropies, is_correct

answers, entropies, is_correct = simulate_attempts(NUM_PROBLEMS, N, p)

# ---- Strategy A: Simple majority vote ----
def simple_majority(answers_row):
    c = Counter(answers_row)
    winner, count = c.most_common(1)[0]
    return winner

# ---- Strategy B: Weighted majority (weight = 1/entropy) ----
def weighted_majority(answers_row, entropy_row):
    weighted = {}
    for ans, ent in zip(answers_row, entropy_row):
        weighted[ans] = weighted.get(ans, 0) + 1.0 / ent
    return max(weighted, key=weighted.get)

# ---- Strategy C: Plurality with lowest-entropy tiebreak ----
def plurality_lowest_entropy(answers_row, entropy_row):
    c = Counter(answers_row)
    max_count = c.most_common(1)[0][1]
    candidates = [ans for ans, cnt in c.items() if cnt == max_count]
    if len(candidates) == 1:
        return candidates[0]
    best_ans = None
    best_ent = float('inf')
    for cand in candidates:
        avg_ent = np.mean([e for a, e in zip(answers_row, entropy_row) if a == cand])
        if avg_ent < best_ent:
            best_ent = avg_ent
            best_ans = cand
    return best_ans

# ---- Strategy D: Cluster-aware voting ----
# Key insight: wrong answers are RANDOM, correct answers CLUSTER
# Any answer appearing >= 2 times is almost certainly correct
def cluster_aware(answers_row, entropy_row):
    c = Counter(answers_row)
    clustered = [(ans, cnt) for ans, cnt in c.items() if cnt >= 2]
    if len(clustered) == 1:
        return clustered[0][0]
    elif len(clustered) > 1:
        return max(clustered, key=lambda x: x[1])[0]
    else:
        # All answers unique - pick lowest entropy
        idx = np.argmin(entropy_row)
        return answers_row[idx]

# ---- Strategy E: Convergence-aware (later attempts weighted more) ----
def convergence_aware(answers_row, entropy_row):
    n = len(answers_row)
    weighted = {}
    for i, (ans, ent) in enumerate(zip(answers_row, entropy_row)):
        pos_weight = 0.5 + (i / (n - 1)) * 1.0
        entropy_weight = 1.0 / ent
        weighted[ans] = weighted.get(ans, 0) + pos_weight * entropy_weight
    return max(weighted, key=weighted.get)

# ---- Strategy F: Cluster + Entropy hybrid ----
def cluster_entropy_hybrid(answers_row, entropy_row):
    c = Counter(answers_row)
    max_count = c.most_common(1)[0][1]
    if max_count >= 2:
        candidates = [ans for ans, cnt in c.items() if cnt == max_count]
        if len(candidates) == 1:
            return candidates[0]
        best_ans = None
        best_ent = float('inf')
        for cand in candidates:
            avg_ent = np.mean([e for a, e in zip(answers_row, entropy_row) if a == cand])
            if avg_ent < best_ent:
                best_ent = avg_ent
                best_ans = cand
        return best_ans
    else:
        idx = np.argmin(entropy_row)
        return answers_row[idx]

# ---- Strategy G: Cluster with threshold 2, entropy fallback ----
def strict_cluster(answers_row, entropy_row, threshold=2):
    c = Counter(answers_row)
    clustered = [(ans, cnt) for ans, cnt in c.items() if cnt >= threshold]
    if clustered:
        clustered.sort(key=lambda x: (-x[1],
            np.mean([e for a, e in zip(answers_row, entropy_row) if a == x[0]])))
        return clustered[0][0]
    else:
        idx = np.argmin(entropy_row)
        return answers_row[idx]

# ---- Strategy H: Oracle (upper bound) ----
def oracle(answers_row):
    if CORRECT_ANSWER in answers_row:
        return CORRECT_ANSWER
    return answers_row[0]

# ---- Strategy I: Cluster-aware + weighted entropy for all-unique case ----
# For the all-unique case, use (1/entropy)^2 to amplify the entropy signal
def cluster_aggressive_entropy(answers_row, entropy_row):
    c = Counter(answers_row)
    clustered = [(ans, cnt) for ans, cnt in c.items() if cnt >= 2]
    if len(clustered) >= 1:
        # Pick highest count, break ties by lowest entropy
        clustered.sort(key=lambda x: (-x[1],
            np.mean([e for a, e in zip(answers_row, entropy_row) if a == x[0]])))
        return clustered[0][0]
    else:
        # All unique: use squared inverse entropy
        scores = [1.0 / (ent ** 2) for ent in entropy_row]
        idx = np.argmax(scores)
        return answers_row[idx]

# ---- Strategy J: Cluster-aware + python_errors filter ----
# Simulate: wrong answers more likely to have python errors
def cluster_with_error_filter(answers_row, entropy_row, error_row):
    # Filter out attempts with errors
    filtered = [(a, e) for a, e, err in zip(answers_row, entropy_row, error_row) if err == 0]
    if not filtered:
        filtered = list(zip(answers_row, entropy_row))
    f_answers = [x[0] for x in filtered]
    f_entropies = [x[1] for x in filtered]
    return cluster_aware(f_answers, f_entropies)

# Simulate error counts: wrong answers more likely to have errors
python_errors = np.where(
    is_correct,
    (np.random.random((NUM_PROBLEMS, N)) < 0.05).astype(int),  # 5% error rate for correct
    (np.random.random((NUM_PROBLEMS, N)) < 0.25).astype(int),  # 25% error rate for wrong
)

# Run all strategies
print(f"\n{'='*60}")
print(f"SIMULATION RESULTS (N={NUM_PROBLEMS} problems, {N} attempts, p={p})")
print(f"{'='*60}")

results = {}
strategy_funcs = {
    'A_simple_majority': lambda a, e, err: simple_majority(a),
    'B_weighted_majority_1/ent': lambda a, e, err: weighted_majority(a, e),
    'C_plurality_low_entropy': lambda a, e, err: plurality_lowest_entropy(a, e),
    'D_cluster_aware_ent_fallback': lambda a, e, err: cluster_aware(a, e),
    'E_convergence_aware': lambda a, e, err: convergence_aware(a, e),
    'F_cluster_entropy_hybrid': lambda a, e, err: cluster_entropy_hybrid(a, e),
    'G_strict_cluster_2': lambda a, e, err: strict_cluster(a, e, 2),
    'H_oracle_upper_bound': lambda a, e, err: oracle(a),
    'I_cluster_aggressive_ent': lambda a, e, err: cluster_aggressive_entropy(a, e),
    'J_cluster_error_filter': lambda a, e, err: cluster_with_error_filter(a, e, err),
}

for name, func in strategy_funcs.items():
    correct_count = 0
    for i in range(NUM_PROBLEMS):
        pred = func(answers[i], entropies[i], python_errors[i])
        if pred == CORRECT_ANSWER:
            correct_count += 1
    acc = correct_count / NUM_PROBLEMS
    results[name] = acc
    print(f"  {name:42s}: {acc:.5f} ({correct_count}/{NUM_PROBLEMS})")

# ============================================================
# Edge case: all 8 answers unique (k=0 or k=1)
# ============================================================
all_unique_total = 0
all_unique_correct_min_ent = 0
all_unique_problems_with_correct = 0

for i in range(NUM_PROBLEMS):
    c = Counter(answers[i])
    if c.most_common(1)[0][1] == 1:
        all_unique_total += 1
        has_correct = CORRECT_ANSWER in answers[i]
        if has_correct:
            all_unique_problems_with_correct += 1
        idx = np.argmin(entropies[i])
        if answers[i][idx] == CORRECT_ANSWER:
            all_unique_correct_min_ent += 1

print(f"\n{'='*60}")
print(f"EDGE CASE: All 8 answers unique")
print(f"{'='*60}")
print(f"  Frequency: {all_unique_total}/{NUM_PROBLEMS} = {all_unique_total/NUM_PROBLEMS:.5f}")
if all_unique_total > 0:
    print(f"  Of these, correct present: {all_unique_problems_with_correct}/{all_unique_total} = {all_unique_problems_with_correct/all_unique_total:.4f}")
    print(f"  Min-entropy picks correct: {all_unique_correct_min_ent}/{all_unique_total} = {all_unique_correct_min_ent/all_unique_total:.4f}")

# ============================================================
# Breakdown by k
# ============================================================
print(f"\n{'='*60}")
print(f"BREAKDOWN BY k (number of correct attempts out of 8)")
print(f"{'='*60}")
k_counts = np.sum(is_correct, axis=1)
for k in range(N+1):
    mask = k_counts == k
    n_problems = int(np.sum(mask))
    if n_problems == 0:
        continue
    correct_cluster = 0
    correct_majority = 0
    for i in np.where(mask)[0]:
        if cluster_aware(answers[i], entropies[i]) == CORRECT_ANSWER:
            correct_cluster += 1
        if simple_majority(answers[i]) == CORRECT_ANSWER:
            correct_majority += 1
    pct_c = correct_cluster / n_problems * 100
    pct_m = correct_majority / n_problems * 100
    theoretical = comb(N, k) * p**k * q**(N-k) * 100
    print(f"  k={k}: {n_problems:6d} problems ({theoretical:5.2f}% expected)  "
          f"majority={pct_m:6.1f}%  cluster_aware={pct_c:6.1f}%")

# ============================================================
# Additional: What if p varies? Sensitivity analysis
# ============================================================
print(f"\n{'='*60}")
print(f"SENSITIVITY: Cluster-aware accuracy vs p (theoretical)")
print(f"{'='*60}")
for p_test in [0.50, 0.55, 0.60, 0.65, 0.69, 0.70, 0.75, 0.80]:
    q_test = 1 - p_test
    # P(cluster works) = P(k>=2) * ~1.0 + P(k=1) * P(entropy picks right) + P(k=0) * 0
    # P(k>=2) = 1 - P(k=0) - P(k=1)
    p_k0 = q_test**8
    p_k1 = 8 * p_test * q_test**7
    p_k2plus = 1 - p_k0 - p_k1
    # When k=1 and all unique, entropy picks correctly ~75% of time (from sim)
    entropy_accuracy_k1 = 0.75  # approximate from entropy separation
    p_cluster = p_k2plus + p_k1 * entropy_accuracy_k1

    # Compare to plain majority (>= 5 of 8)
    p_maj = sum(comb(8, k) * p_test**k * q_test**(8-k) for k in range(5, 9))
    # Effective majority (>= 4, wrongs scatter)
    p_eff = p_maj + comb(8, 4) * p_test**4 * q_test**4

    print(f"  p={p_test:.2f}: majority={p_eff:.4f}  cluster_aware~={p_cluster:.4f}  "
          f"oracle={1-p_k0:.4f}  cluster_gain={p_cluster-p_eff:+.4f}")

# ============================================================
# FINAL OUTPUT
# ============================================================
non_oracle = {k: v for k, v in results.items() if 'oracle' not in k.lower()}
best_real_name = max(non_oracle, key=non_oracle.get)

print(f"\n{'='*60}")
print("FINAL SUMMARY")
print(f"{'='*60}")

output = {
    "majority_vote_accuracy": round(effective_majority, 6),
    "majority_vote_strict_5plus": round(majority_prob, 6),
    "appears_at_least_once": round(appears_once, 6),
    "improvement_ceiling": round(gap, 6),
    "improvement_ceiling_pct": f"{gap/effective_majority*100:.1f}%",
    "best_strategy_name": best_real_name,
    "best_strategy_accuracy": round(non_oracle[best_real_name], 5),
    "oracle_upper_bound": round(results['H_oracle_upper_bound'], 5),
    "all_strategies": {k: round(v, 5) for k, v in results.items()},
    "key_insight": (
        "With random wrong answers (mod 100000), ANY answer appearing >=2 times "
        "is almost certainly correct (P(two random wrongs matching) = 1/99999). "
        "This makes cluster detection nearly as good as oracle. The only losses: "
        "k=0 (no correct attempt, ~0.07% of problems) and k=1 where all 8 answers "
        "are unique (must rely on entropy, ~75% success). Cluster-aware voting "
        "captures nearly all available accuracy."
    ),
    "recommendation": (
        "Replace weighted majority with cluster-aware voting: "
        "(1) If any answer appears >=2 times, pick highest-count answer "
        "(tiebreak: lowest avg entropy). "
        "(2) If all answers unique, pick lowest-entropy answer. "
        "Also filter out attempts with python errors before voting."
    ),
}

print(json.dumps(output, indent=2))
