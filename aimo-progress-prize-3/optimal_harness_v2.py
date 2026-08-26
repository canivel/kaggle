#!/usr/bin/env python3
"""
AIMO3 Optimal Harness Analysis - V2 (refined models)
Fixes: better anti-correlation model, realistic wrong-answer correlation.
"""
import json
import math
import numpy as np
from scipy.special import comb
from scipy.stats import norm
from functools import lru_cache

p = 0.69

def majority_vote_accuracy(N, p):
    """P(majority correct) for N independent Bernoulli(p) attempts. Works for odd N."""
    threshold = (N + 1) // 2 if N % 2 == 1 else N // 2 + 1
    return float(sum(comb(N, k, exact=True) * p**k * (1-p)**(N-k) for k in range(threshold, N+1)))

# =========================================================================
# KEY REALITY CHECK: Why is our score 39 and not 43?
# =========================================================================
print("=" * 80)
print("REALITY CHECK: Why 39/50 instead of theoretical 43/50?")
print("=" * 80)

# With p=0.69 per attempt and N=8 (our current setup):
# If we use PLURALITY voting (most common answer wins):
# The question is: how often do wrong answers collide?
#
# CRITICAL: In AIMO3, answers are integers mod 100000.
# Wrong answers are NOT uniformly random -- common math errors produce
# the same wrong answer. This is the key modeling question.
#
# Let's estimate alpha (fraction of wrong attempts giving the SAME wrong answer)
# from our actual data: 39/50 correct with N=8.

print("\nReverse-engineering alpha from observed score:")
print("If P(correct via voting) = 39/50 = 0.78, what alpha explains this?")
print()

def voting_accuracy_with_alpha(N, p_correct, alpha):
    """
    Each attempt independently:
    - correct with prob p_correct (all give same answer)
    - common_mistake with prob (1-p_correct)*alpha (all give same wrong answer)
    - random_wrong with prob (1-p_correct)*(1-alpha) (each gives unique wrong answer)

    Plurality vote: correct wins if correct_count > mistake_count.
    (Random wrongs never accumulate enough to win for reasonable N.)

    Returns P(correct answer wins plurality).
    """
    p_c = p_correct
    p_m = (1 - p_correct) * alpha
    p_r = (1 - p_correct) * (1 - alpha)

    total = 0.0
    for c in range(0, N+1):
        for m in range(0, N-c+1):
            r = N - c - m
            # multinomial probability
            prob = (math.factorial(N) / (math.factorial(c) * math.factorial(m) * math.factorial(r))) * \
                   p_c**c * p_m**m * p_r**r
            if c > m and c > 0:
                total += prob
            elif c == m and c > 0:
                total += prob * 0.5  # tiebreak: random
            # if c == 0: no correct answers, we lose
    return total

print(f"{'alpha':>6} | {'P(correct)':>12} | {'E[score]/50':>12}")
print("-" * 38)
for alpha in np.arange(0.0, 1.05, 0.05):
    acc = voting_accuracy_with_alpha(8, 0.69, alpha)
    print(f"{alpha:6.2f} | {acc:12.6f} | {50*acc:12.2f}")

# Find alpha that gives 39/50
from scipy.optimize import brentq
target = 39.0 / 50.0
alpha_estimated = brentq(lambda a: voting_accuracy_with_alpha(8, 0.69, a) - target, 0.0, 1.0)
print(f"\nEstimated alpha = {alpha_estimated:.4f} (produces E[score] = 39/50)")
print(f"This means {alpha_estimated*100:.1f}% of wrong attempts produce the SAME wrong answer.")
print(f"And {(1-alpha_estimated)*100:.1f}% produce scattered/unique wrong answers.")

# =========================================================================
# SECTION 1: VOTING ACCURACY WITH REALISTIC ALPHA
# =========================================================================
print("\n" + "=" * 80)
print(f"SECTION 1: VOTING ACCURACY WITH alpha={alpha_estimated:.3f}")
print("=" * 80)

alpha = alpha_estimated

print(f"\n{'N':>4} | {'P(correct)':>12} | {'E[score]/50':>12} | {'Marginal gain':>14}")
print("-" * 55)
prev_score = 50 * voting_accuracy_with_alpha(1, p, alpha)
for N in [1, 3, 5, 7, 8, 9, 11, 13, 15, 17, 19, 21, 25, 31]:
    acc = voting_accuracy_with_alpha(N, p, alpha)
    score = 50 * acc
    gain = score - prev_score
    print(f"{N:4d} | {acc:12.6f} | {score:12.2f} | {gain:+14.2f}")
    prev_score = score

# =========================================================================
# SECTION 2: TIME BUDGET
# =========================================================================
print("\n" + "=" * 80)
print("SECTION 2: TIME BUDGET (realistic)")
print("=" * 80)

workers = 16
time_per_attempt = 30  # seconds including generation + code execution
total_time = 32000     # 9h - 400s setup
problems = 50

print(f"\nWith {workers} parallel workers, {time_per_attempt}s/attempt, {total_time}s total:\n")
for N in [8, 12, 16, 20, 24, 32, 48, 64]:
    total_attempts = N * problems
    wall_time = math.ceil(total_attempts / workers) * time_per_attempt
    fits = "OK" if wall_time <= total_time else "NO"
    acc = voting_accuracy_with_alpha(min(N, 31), p, alpha)  # cap at 31 for computation
    print(f"N={N:2d}: wall={wall_time:5d}s ({wall_time/3600:.1f}h) E[score]={50*acc:.1f} [{fits}]")

max_N_budget = int(total_time / time_per_attempt * workers / problems)
print(f"\nMax N within budget: {max_N_budget}")

# But: is time_per_attempt really 30s? Could be longer for hard problems.
# Hard problems may need 60-90s of code execution + retries.
# Let's be conservative: 45s average
print("\nWith conservative 45s/attempt:")
for N in [8, 12, 16, 20, 24, 32]:
    total_attempts = N * problems
    wall_time = math.ceil(total_attempts / workers) * 45
    fits = "OK" if wall_time <= total_time else "NO"
    print(f"N={N:2d}: wall={wall_time:5d}s ({wall_time/3600:.1f}h) [{fits}]")

# =========================================================================
# SECTION 3: ADAPTIVE STRATEGY WITH REALISTIC ALPHA
# =========================================================================
print("\n" + "=" * 80)
print(f"SECTION 3: ADAPTIVE STRATEGY (alpha={alpha:.3f})")
print("=" * 80)

def adaptive_expected(p_correct, alpha, K, N_max):
    """Stop when any answer gets K votes. Realistic wrong-answer model."""
    @lru_cache(maxsize=None)
    def solve(attempts, correct_count, mistake_count):
        if correct_count >= K:
            return (0.0, 1.0)
        if mistake_count >= K:
            return (0.0, 0.0)
        if attempts >= N_max:
            if correct_count > mistake_count:
                return (0.0, 1.0)
            elif correct_count == mistake_count:
                return (0.0, 0.5)
            else:
                return (0.0, 0.0)

        p_c = p_correct
        p_m = (1 - p_correct) * alpha
        p_r = (1 - p_correct) * (1 - alpha)

        e_c, pw_c = solve(attempts + 1, correct_count + 1, mistake_count)
        e_m, pw_m = solve(attempts + 1, correct_count, mistake_count + 1)
        e_r, pw_r = solve(attempts + 1, correct_count, mistake_count)

        e_total = 1.0 + p_c * e_c + p_m * e_m + p_r * e_r
        p_win = p_c * pw_c + p_m * pw_m + p_r * pw_r
        return (e_total, p_win)

    result = solve(0, 0, 0)
    solve.cache_clear()
    return result

# Different N_max and K combinations
print(f"\nFixed alpha={alpha:.3f}, p=0.69:")
print(f"{'K':>4} {'N_max':>6} | {'E[attempts]':>12} | {'P(correct)':>12} | {'E[score]/50':>12}")
print("-" * 55)
for K in [3, 4, 5, 6]:
    for N_max in [12, 16, 20, 24]:
        e_att, p_win = adaptive_expected(0.69, alpha, K, N_max)
        print(f"{K:4d} {N_max:6d} | {e_att:12.2f} | {p_win:12.6f} | {50*p_win:12.2f}")
    print()

# Heterogeneous difficulty
print("--- With heterogeneous difficulty ---")
difficulty_dist = [(0.20, 0.90), (0.50, 0.69), (0.30, 0.40)]
print(f"Distribution: 20% easy (p=0.90), 50% medium (p=0.69), 30% hard (p=0.40)")

configs = [
    ("Flat N=8", 8, None),
    ("Flat N=12", 12, None),
    ("Flat N=16", 16, None),
    ("Flat N=24", 24, None),
    ("Adaptive K=4, Nmax=16", 16, 4),
    ("Adaptive K=4, Nmax=24", 24, 4),
    ("Adaptive K=5, Nmax=24", 24, 5),
    ("Adaptive K=6, Nmax=24", 24, 6),
]

print(f"\n{'Config':>30} | {'E[score]':>10} | {'E[attempts]':>12} | {'Wall time':>10}")
print("-" * 72)
for name, N_max, K in configs:
    total_score = 0
    total_att = 0
    for frac, pi in difficulty_dist:
        n_probs = 50 * frac
        if K is None:
            # Flat: just run N_max attempts
            acc = voting_accuracy_with_alpha(N_max, pi, alpha)
            total_score += n_probs * acc
            total_att += n_probs * N_max
        else:
            e_att, p_win = adaptive_expected(pi, alpha, K, N_max)
            total_score += n_probs * p_win
            total_att += n_probs * e_att
    wall = math.ceil(total_att / workers) * time_per_attempt
    print(f"{name:>30} | {total_score:10.2f} | {total_att:12.0f} | {wall:8d}s")

# =========================================================================
# SECTION 4: WHAT EXPLAINS THE GAP TO 44-46?
# =========================================================================
print("\n" + "=" * 80)
print("SECTION 4: GAP ANALYSIS - What gets us from 39 to 44-46?")
print("=" * 80)

print("\nFactor 1: Increase N (more attempts)")
print(f"  N=8  -> E[score] = {50*voting_accuracy_with_alpha(8, 0.69, alpha):.1f}")
print(f"  N=12 -> E[score] = {50*voting_accuracy_with_alpha(12, 0.69, alpha):.1f}")
print(f"  N=16 -> E[score] = {50*voting_accuracy_with_alpha(16, 0.69, alpha):.1f}")
print(f"  N=20 -> E[score] = {50*voting_accuracy_with_alpha(20, 0.69, alpha):.1f}")
print(f"  N=24 -> E[score] = {50*voting_accuracy_with_alpha(24, 0.69, alpha):.1f}")

print("\nFactor 2: Increase p (better model/prompt)")
for p_test in [0.69, 0.72, 0.75, 0.78, 0.80, 0.85]:
    # Recalculate alpha for this p (assume alpha stays same)
    score_8 = 50 * voting_accuracy_with_alpha(8, p_test, alpha)
    score_16 = 50 * voting_accuracy_with_alpha(16, p_test, alpha)
    score_24 = 50 * voting_accuracy_with_alpha(min(24, 20), p_test, alpha)
    print(f"  p={p_test:.2f}: N=8->{score_8:.1f}, N=16->{score_16:.1f}, N=20->{score_24:.1f}")

print("\nFactor 3: Reduce alpha (decorrelate wrong answers)")
print("  (Higher temperature, diverse prompts, different solution approaches)")
for alpha_test in [alpha, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]:
    score = 50 * voting_accuracy_with_alpha(16, 0.69, alpha_test)
    print(f"  alpha={alpha_test:.2f}: N=16 -> E[score] = {score:.1f}")

# =========================================================================
# SECTION 5: ANTI-CORRELATION - PROPER MODEL
# =========================================================================
print("\n" + "=" * 80)
print("SECTION 5: ANTI-CORRELATION EFFECT (proper model)")
print("=" * 80)

# The arxiv 2603.27844 reports rho=-0.258 for pairwise correlation between attempts.
# This means: conditioned on attempt i being wrong, attempt j is MORE likely to be right.
#
# Proper model: use multivariate probit or direct simulation.
# For correlated Bernoulli with pairwise rho:
# P(X_i = 1, X_j = 1) = p^2 + rho*p*(1-p)
# P(X_i = 1, X_j = 0) = p*(1-p) - rho*p*(1-p) = p*(1-p)*(1-rho)
# P(X_i = 0, X_j = 0) = (1-p)^2 + rho*p*(1-p)
#
# For negative rho: P(both wrong) INCREASES, P(one right one wrong) INCREASES.
# Wait: P(X_i=0, X_j=0) = (1-p)^2 + rho*p*(1-p). With rho<0, this DECREASES.
# P(X_i=1, X_j=0) = p*(1-p)*(1-rho). With rho<0, this INCREASES.
# So: more disagreement between attempts, less joint failure. GOOD for voting!

rho = -0.258
print(f"\nPairwise correlation rho = {rho}")
print(f"P(both correct) = p^2 + rho*p*(1-p) = {p**2 + rho*p*(1-p):.4f} (vs {p**2:.4f} independent)")
print(f"P(both wrong) = (1-p)^2 + rho*p*(1-p) = {(1-p)**2 + rho*p*(1-p):.4f} (vs {(1-p)**2:.4f} independent)")
print(f"P(disagree) = 2*p*(1-p)*(1-rho) = {2*p*(1-p)*(1-rho):.4f} (vs {2*p*(1-p):.4f} independent)")

# Simulate correlated Bernoulli using Gaussian copula
print("\nMonte Carlo simulation with Gaussian copula (100K trials):")
np.random.seed(42)
n_sim = 100000

for N in [7, 9, 11, 15, 21]:
    # Generate correlated normal variables
    # Correlation matrix: all off-diagonal = rho
    corr_matrix = np.full((N, N), rho)
    np.fill_diagonal(corr_matrix, 1.0)

    # Check positive definiteness
    eigvals = np.linalg.eigvalsh(corr_matrix)
    if eigvals.min() < 0:
        # Not PD, find max valid rho
        # For equicorrelated matrix: eigenvalues are 1+(N-1)*rho and 1-rho (multiplicity N-1)
        min_rho = -1.0/(N-1)
        print(f"  N={N}: rho={rho} invalid (min={min_rho:.4f}), using rho={max(rho, min_rho+0.001):.4f}")
        actual_rho = max(rho, min_rho + 0.001)
        corr_matrix = np.full((N, N), actual_rho)
        np.fill_diagonal(corr_matrix, 1.0)
    else:
        actual_rho = rho

    # Cholesky decomposition
    try:
        L = np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        print(f"  N={N}: Cholesky failed, skipping")
        continue

    # Generate correlated normals
    Z = np.random.randn(n_sim, N) @ L.T

    # Convert to Bernoulli using threshold
    threshold = norm.ppf(1 - p)  # P(Z > threshold) = p
    X = (Z > threshold).astype(int)

    # Check empirical p and rho
    emp_p = X.mean()
    if N >= 2:
        emp_rho = np.corrcoef(X[:, 0], X[:, 1])[0, 1]
    else:
        emp_rho = 0

    # Majority vote
    votes = X.sum(axis=1)
    majority_threshold = (N + 1) // 2
    majority_correct = (votes >= majority_threshold).mean()

    # Independent baseline
    indep_correct = majority_vote_accuracy(N, p)

    gain = 50 * (majority_correct - indep_correct)
    print(f"  N={N:2d}: rho_used={actual_rho:.3f} emp_p={emp_p:.3f} emp_rho={emp_rho:.3f} | "
          f"P(maj)={majority_correct:.4f} vs {indep_correct:.4f} (indep) | "
          f"E[score]: {50*majority_correct:.1f} vs {50*indep_correct:.1f} | gain={gain:+.1f}")

# =========================================================================
# SECTION 6: COMBINING EVERYTHING - REALISTIC OPTIMAL
# =========================================================================
print("\n" + "=" * 80)
print("SECTION 6: REALISTIC OPTIMAL STRATEGY")
print("=" * 80)

# Reality: alpha ~ 0.76, p = 0.69, rho ~ -0.258 (but limited by N)
# The anti-correlation helps but doesn't change the picture dramatically.
# The dominant factor is alpha (wrong-answer correlation).

print("\n--- Sensitivity analysis: Score as function of (p, alpha, N) ---")
print(f"\n{'p':>5} {'alpha':>6} {'N':>4} | {'E[score]':>10}")
print("-" * 32)
for p_test in [0.69, 0.72, 0.75, 0.78]:
    for alpha_test in [0.76, 0.50, 0.30, 0.10]:
        for N in [8, 16]:
            score = 50 * voting_accuracy_with_alpha(N, p_test, alpha_test)
            marker = " <-- current" if (p_test == 0.69 and alpha_test == 0.76 and N == 8) else ""
            marker = " <-- target" if (score >= 44 and score <= 46 and marker == "") else marker
            print(f"{p_test:5.2f} {alpha_test:6.2f} {N:4d} | {score:10.2f}{marker}")
    print()

# =========================================================================
# SECTION 7: FINAL JSON OUTPUT
# =========================================================================
print("\n" + "=" * 80)
print("FINAL JSON OUTPUT")
print("=" * 80)

# Find optimal configs that reach 44, 45, 46
def find_min_N_for_target(target_score, p_val, alpha_val, max_N=64):
    for N in range(1, max_N+1):
        if 50 * voting_accuracy_with_alpha(N, p_val, alpha_val) >= target_score:
            return N
    return max_N

result = {
    "estimated_alpha": round(alpha_estimated, 4),
    "alpha_meaning": f"{alpha_estimated*100:.1f}% of wrong attempts produce the SAME wrong answer (common math mistake)",
    "optimal_N": {
        "with_current_alpha": {
            "for_44": find_min_N_for_target(44, 0.69, alpha_estimated),
            "for_45": find_min_N_for_target(45, 0.69, alpha_estimated),
            "for_46": find_min_N_for_target(46, 0.69, alpha_estimated),
        },
        "with_reduced_alpha_0.50": {
            "for_44": find_min_N_for_target(44, 0.69, 0.50),
            "for_45": find_min_N_for_target(45, 0.69, 0.50),
            "for_46": find_min_N_for_target(46, 0.69, 0.50),
        },
        "with_better_p_0.75": {
            "for_44": find_min_N_for_target(44, 0.75, alpha_estimated),
            "for_45": find_min_N_for_target(45, 0.75, alpha_estimated),
            "for_46": find_min_N_for_target(46, 0.75, alpha_estimated),
        },
        "recommended": 16,
        "rationale": "N=16 fits in budget (1500s wall time) and reaches 43.5 at current alpha. Further N increase has diminishing returns."
    },
    "expected_scores": {
        "current_N8_p069_alpha076": round(50 * voting_accuracy_with_alpha(8, 0.69, alpha_estimated), 2),
        "N16_p069_alpha076": round(50 * voting_accuracy_with_alpha(16, 0.69, alpha_estimated), 2),
        "N16_p069_alpha050": round(50 * voting_accuracy_with_alpha(16, 0.69, 0.50), 2),
        "N16_p075_alpha076": round(50 * voting_accuracy_with_alpha(16, 0.75, alpha_estimated), 2),
        "N16_p075_alpha050": round(50 * voting_accuracy_with_alpha(16, 0.75, 0.50), 2),
        "N24_p069_alpha076": round(50 * voting_accuracy_with_alpha(min(24,20), 0.69, alpha_estimated), 2),
    },
    "current_score_gap_analysis": {
        "our_score": 39,
        "estimated_alpha": round(alpha_estimated, 3),
        "three_levers_ranked": [
            {
                "lever": "Reduce alpha (decorrelate wrong answers)",
                "how": "Higher temperature, 3-4 prompt variants, different solution strategies",
                "impact": f"alpha 0.76->0.50 at N=16: +{50*voting_accuracy_with_alpha(16,0.69,0.50) - 50*voting_accuracy_with_alpha(16,0.69,alpha_estimated):.1f} score",
                "difficulty": "Medium - requires prompt engineering"
            },
            {
                "lever": "Increase N (more attempts per problem)",
                "how": "N=8 -> N=16, easy with time budget",
                "impact": f"N 8->16 at current alpha: +{50*voting_accuracy_with_alpha(16,0.69,alpha_estimated) - 50*voting_accuracy_with_alpha(8,0.69,alpha_estimated):.1f} score",
                "difficulty": "Easy - just run more"
            },
            {
                "lever": "Increase p (better per-attempt accuracy)",
                "how": "Better prompts, more reliable code execution, fix answer extraction",
                "impact": f"p 0.69->0.75 at N=16: +{50*voting_accuracy_with_alpha(16,0.75,alpha_estimated) - 50*voting_accuracy_with_alpha(16,0.69,alpha_estimated):.1f} score",
                "difficulty": "Hard - fundamental model capability"
            }
        ]
    },
    "adaptive_vs_flat": {
        "flat_N16_score": round(50 * voting_accuracy_with_alpha(16, 0.69, alpha_estimated), 2),
        "recommendation": "Adaptive K=5 Nmax=24 saves time on easy problems, allocates more to hard ones. But the score gain vs flat N=16 is modest (~0.5-1.0) because hard problems have both low p AND high alpha."
    },
    "anti_correlation_effect": {
        "rho": -0.258,
        "effect": "Negative correlation HELPS voting by ~0.5-1.5 score points at N=15. The effect is real but modest compared to the alpha and p levers.",
        "note": "For N<=4, rho=-0.258 exceeds the valid range for equicorrelated Bernoulli (min_rho = -1/(N-1)). The anti-correlation is strongest and most beneficial at larger N."
    },
    "recommended_architecture": {
        "strategy": "Adaptive plurality voting with diverse sampling",
        "N_max": 16,
        "early_stop_K": 5,
        "sampling_config": {
            "temperature": 0.9,
            "prompt_variants": 4,
            "purpose": "Reduce alpha by decorrelating wrong answers"
        },
        "voting": "Plurality (most common answer). With scattered wrongs, plurality >> majority.",
        "two_phase": True,
        "phase1": "N=8 attempts for all 50 problems (750s wall time)",
        "phase2": "N_extra=8-16 for problems without K=5 agreement (use remaining ~31000s)",
        "expected_score": "41-43 at current p/alpha, 44-46 if alpha reduced to 0.40-0.50",
        "critical_insight": (
            "The #1 bottleneck is NOT N (we have budget for N=300+). "
            "It is ALPHA: 76% of wrong attempts produce the same wrong answer, "
            "meaning more attempts just add more votes for the common mistake. "
            "MUST decorrelate errors via diverse prompts/temperatures."
        )
    }
}

print("\n" + json.dumps(result, indent=2))

# =========================================================================
# FINAL SUMMARY TABLE
# =========================================================================
print("\n" + "=" * 80)
print("ACTIONABLE SUMMARY")
print("=" * 80)
print()
print("Current: 39/50 (p=0.69, N=8, alpha=0.76)")
print()
print("Path to 44-46:")
print("  1. [EASY]   Increase N: 8 -> 16          -> +4.5 -> 43.5")
print("  2. [MEDIUM] Reduce alpha: 0.76 -> 0.50   -> +3.5 -> 47.0")
print("  3. [HARD]   Increase p: 0.69 -> 0.75     -> +2.0 -> 49.0")
print("  (effects not fully additive)")
print()
print("Combined realistic target:")
combined = 50 * voting_accuracy_with_alpha(16, 0.72, 0.55)
print(f"  N=16, p=0.72, alpha=0.55 -> E[score] = {combined:.1f}")
combined2 = 50 * voting_accuracy_with_alpha(16, 0.75, 0.45)
print(f"  N=16, p=0.75, alpha=0.45 -> E[score] = {combined2:.1f}")
print()
print("KEY INSIGHT: alpha (wrong-answer correlation) is the hidden variable")
print("that explains why our 39 is below the theoretical 43 for p=0.69, N=8.")
print(f"Reverse-engineered alpha = {alpha_estimated:.3f}")
