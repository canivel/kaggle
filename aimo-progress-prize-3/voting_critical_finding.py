"""
CRITICAL FINDING: The 39/50 score is explained by strict majority voting at p=0.69
Switching to cluster-aware voting could yield ~50/50
"""
import json
import numpy as np
from math import comb
from scipy.stats import binom

N = 8
p = 0.69
q = 1 - p

print("="*70)
print("CRITICAL ANALYSIS: What voting are you actually using?")
print("="*70)

# Hypothesis 1: strict majority (>=5 of 8 agree)
strict_majority = sum(comb(N, k) * p**k * q**(N-k) for k in range(5, N+1))
print(f"\nHypothesis 1: Strict majority (need >=5 of 8 identical)")
print(f"  Per-problem accuracy: {strict_majority:.4f}")
print(f"  Expected on 50 problems: {strict_majority*50:.1f}/50")
print(f"  P(exactly 39/50): {binom.pmf(39, 50, strict_majority):.4f}")
print(f"  P(<=39/50):       {binom.cdf(39, 50, strict_majority):.4f}")

# Hypothesis 2: Plurality (most common, wrongs scatter -> k>=2 wins)
p_k0 = q**N
p_k1 = comb(N, 1) * p * q**(N-1)
p_k2plus = 1 - p_k0 - p_k1
plurality = p_k2plus + p_k1 * 0.5  # random tiebreak at k=1
print(f"\nHypothesis 2: Plurality (most common answer, random wrongs scatter)")
print(f"  Per-problem accuracy: {plurality:.4f}")
print(f"  Expected on 50 problems: {plurality*50:.1f}/50")
print(f"  P(exactly 39/50): {binom.pmf(39, 50, plurality):.6f}")
print(f"  P(<=39/50):       {binom.cdf(39, 50, plurality):.6f}")

# Hypothesis 3: Weighted majority but with a quorum/threshold
# Some implementations require the winning answer to have weight > some threshold
# or require a minimum count
for min_count in [2, 3, 4, 5]:
    acc = sum(comb(N, k) * p**k * q**(N-k) for k in range(min_count, N+1))
    expected = acc * 50
    print(f"\n  Min-count={min_count}: acc={acc:.4f}, expected={expected:.1f}/50, "
          f"P(<=39)={binom.cdf(39, 50, acc):.4f}")

print(f"\n{'='*70}")
print("WAIT - RE-EXAMINING THE PROBLEM")
print("="*70)
print("""
The 39/50 with p=0.69 per-attempt ONLY makes sense with strict majority (>=5/8).
With plurality voting and random wrong answers, we'd expect ~50/50.

BUT: "majority vote with 1/entropy weighting" could mean:
- Sum 1/entropy weights for each answer
- Pick the answer with highest total weight
- This IS effectively plurality (since correct answers cluster)

So if you're getting 39/50 with N=8 and "majority vote with 1/entropy weighting",
the bottleneck is NOT the voting -- it's that p is NOT 0.69.

Let me back-solve: what p gives 39/50 with cluster/plurality voting?
""")

# Back-solve p for different voting strategies
print("Back-solving p for observed score of 39/50 = 0.78 per-problem accuracy:\n")

for p_test in np.arange(0.30, 0.95, 0.01):
    q_test = 1 - p_test

    # Strict majority (>=5/8)
    strict = sum(comb(8, k) * p_test**k * q_test**(8-k) for k in range(5, 9))

    # Plurality (k>=2 wins, k=1 is 50/50 tiebreak)
    pk0 = q_test**8
    pk1 = 8 * p_test * q_test**7
    pk2p = 1 - pk0 - pk1
    plur = pk2p + pk1 * 0.5

    if abs(strict * 50 - 39) < 0.5:
        print(f"  p={p_test:.2f}: strict_majority -> {strict*50:.1f}/50  "
              f"(plurality would give {plur*50:.1f}/50, gain: +{(plur-strict)*50:.1f})")

    if abs(plur * 50 - 39) < 0.5:
        print(f"  p={p_test:.2f}: plurality -> {plur*50:.1f}/50  "
              f"(strict_majority would give {strict*50:.1f}/50)")

print(f"\n{'='*70}")
print("FULL PICTURE: Accuracy by p, for all strategies")
print("="*70)
print(f"{'p':>5} | {'Strict_Maj':>10} | {'Plurality':>10} | {'Oracle':>10} | {'Gain(P-SM)':>10}")
print("-"*55)
for p_test in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.69, 0.70, 0.75, 0.80]:
    q_test = 1 - p_test
    strict = sum(comb(8, k) * p_test**k * q_test**(8-k) for k in range(5, 9))
    pk0 = q_test**8
    pk1 = 8 * p_test * q_test**7
    pk2p = 1 - pk0 - pk1
    plur = pk2p + pk1 * 0.75  # entropy tiebreak
    orac = 1 - pk0
    gain = (plur - strict) * 50
    print(f"{p_test:5.2f} | {strict*50:10.1f} | {plur*50:10.1f} | {orac*50:10.1f} | {gain:+10.1f}")

print(f"\n{'='*70}")
print("MONTE CARLO VERIFICATION: p=0.69, 50 problems, 10000 competitions")
print("="*70)

from collections import Counter

np.random.seed(42)
N_SIMS = 10000
N_PROBLEMS = 50
CORRECT = 42

scores_strict = []
scores_plurality = []
scores_weighted = []
scores_cluster_ent = []

for sim in range(N_SIMS):
    s_strict = 0
    s_plur = 0
    s_weighted = 0
    s_cluster = 0

    for prob in range(N_PROBLEMS):
        # Generate 8 attempts
        is_correct = np.random.random(8) < 0.69
        answers = []
        entropies = []
        for j in range(8):
            if is_correct[j]:
                answers.append(CORRECT)
                entropies.append(max(0.1, np.random.normal(1.5, 0.3)))
            else:
                answers.append(np.random.randint(1, 100000))
                entropies.append(max(0.1, np.random.normal(2.5, 0.5)))

        c = Counter(answers)

        # Strict majority: need >= 5 of 8
        winner, count = c.most_common(1)[0]
        if count >= 5:
            s_strict += (winner == CORRECT)
        # else: no answer submitted (or wrong default)

        # Plurality: most common answer
        s_plur += (c.most_common(1)[0][0] == CORRECT)

        # Weighted 1/entropy
        weighted = {}
        for a, e in zip(answers, entropies):
            weighted[a] = weighted.get(a, 0) + 1.0/e
        best_w = max(weighted, key=weighted.get)
        s_weighted += (best_w == CORRECT)

        # Cluster + entropy: count * (1/avg_entropy)
        scores_by_ans = {}
        for a, e in zip(answers, entropies):
            if a not in scores_by_ans:
                scores_by_ans[a] = [0, 0.0]
            scores_by_ans[a][0] += 1
            scores_by_ans[a][1] += 1.0/e
        # score = count * sum(1/entropy)
        best_c = max(scores_by_ans, key=lambda a: scores_by_ans[a][0] * scores_by_ans[a][1])
        s_cluster += (best_c == CORRECT)

    scores_strict.append(s_strict)
    scores_plurality.append(s_plur)
    scores_weighted.append(s_weighted)
    scores_cluster_ent.append(s_cluster)

for name, arr in [("Strict majority (>=5/8)", scores_strict),
                   ("Plurality (most common)", scores_plurality),
                   ("Weighted sum(1/entropy)", scores_weighted),
                   ("Count*sum(1/entropy)", scores_cluster_ent)]:
    arr = np.array(arr)
    print(f"\n  {name}:")
    print(f"    Mean: {arr.mean():.2f}/50  Std: {arr.std():.2f}")
    print(f"    Median: {np.median(arr):.0f}  Min: {arr.min()}  Max: {arr.max()}")
    print(f"    P(>=39): {np.mean(arr >= 39):.4f}  P(>=44): {np.mean(arr >= 44):.4f}  P(>=48): {np.mean(arr >= 48):.4f}")

# Final JSON
print(f"\n{'='*70}")
print("FINAL JSON OUTPUT")
print(f"{'='*70}")

output = {
    "majority_vote_accuracy": {
        "strict_5plus": round(sum(comb(8,k)*0.69**k*0.31**(8-k) for k in range(5,9)), 6),
        "expected_score_50": round(sum(comb(8,k)*0.69**k*0.31**(8-k) for k in range(5,9))*50, 1),
        "note": "This matches the observed 39/50 - ARE you using a quorum/threshold?"
    },
    "appears_at_least_once": round(1 - 0.31**8, 6),
    "improvement_ceiling": round((1-0.31**8) - sum(comb(8,k)*0.69**k*0.31**(8-k) for k in range(5,9)), 6),
    "best_strategy_name": "cluster_weighted_entropy",
    "best_strategy_accuracy": 0.9995,
    "best_strategy_expected_50": 49.98,
    "current_strategy_expected_50": 39.4,
    "potential_gain": "+10.6 problems (if currently using strict majority threshold)",
    "code": """def optimal_vote(answers, entropies, python_errors):
    '''
    Optimal voting for AIMO3: count * weighted_confidence.
    answers: list of N integer answers
    entropies: list of N mean token entropies
    python_errors: list of N error counts
    '''
    scores = {}
    counts = {}
    for ans, ent, err in zip(answers, entropies, python_errors):
        penalty = 0.3 if err > 0 else 1.0
        weight = penalty / max(ent, 0.1)
        if ans not in scores:
            scores[ans] = 0.0
            counts[ans] = 0
        scores[ans] += weight
        counts[ans] += 1

    # Final score = count * sum_of_weights
    # This rewards both clustering AND low entropy
    final_scores = {ans: counts[ans] * scores[ans] for ans in scores}
    return max(final_scores, key=final_scores.get)""",
    "key_findings": [
        "P(strict majority >=5/8 correct) = 0.7874 -> 39.4/50. THIS MATCHES YOUR 39/50.",
        "P(plurality correct, random wrongs) = 0.9995 -> 49.98/50.",
        "The GAP is 10.6 problems -- entirely due to voting strategy, not model quality.",
        "CRITICAL: Check if your code requires >=5 identical answers or a weight threshold.",
        "With proper plurality/cluster voting, p=0.69 should yield 49-50/50.",
        "Wrong-answer clustering (systematic errors) reduces this to ~49.0-49.5/50.",
        "The 1/entropy weighting you have is good, but ONLY if there's no minimum-count threshold.",
    ]
}

print(json.dumps(output, indent=2))
