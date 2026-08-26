#!/usr/bin/env python3
"""
AIMO3 Optimal Harness Analysis
Compute theoretically optimal parameters for majority/plurality voting.
"""
import json
import math
import numpy as np
from scipy.special import comb
from functools import lru_cache

p = 0.69

def majority_vote_accuracy_odd(N, p):
    threshold = (N + 1) // 2
    return float(sum(comb(N, k, exact=True) * p**k * (1-p)**(N-k) for k in range(threshold, N+1)))

# =========================================================================
# SECTION 1: MAJORITY VOTE ACCURACY
# =========================================================================
print("=" * 80)
print("SECTION 1: MAJORITY VOTE ACCURACY (p=0.69)")
print("=" * 80)

odd_Ns = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 31, 47, 63]
print(f"\n{'N':>4} | {'P(correct)':>12} | {'E[score]/50':>12}")
print("-" * 35)
for N in odd_Ns:
    acc = majority_vote_accuracy_odd(N, p)
    print(f"{N:4d} | {acc:12.6f} | {50*acc:12.2f}")

# =========================================================================
# SECTION 1b: PLURALITY VOTE (scattered wrong answers)
# =========================================================================
print("\n" + "=" * 80)
print("SECTION 1b: PLURALITY VOTE (wrong answers fully scattered)")
print("P(correct) = 1 - (1-p)^N")
print("=" * 80)

Ns = [1, 2, 3, 4, 5, 8, 10, 12, 16, 20, 24, 32]
print(f"\n{'N':>4} | {'P(plurality)':>14} | {'E[plur]/50':>11} | {'P(majority)':>13} | {'E[maj]/50':>10}")
print("-" * 62)
for N in Ns:
    p_plur = 1 - (1-p)**N
    N_eff = N if N % 2 == 1 else N - 1
    p_maj = majority_vote_accuracy_odd(N_eff, p)
    print(f"{N:4d} | {p_plur:14.8f} | {50*p_plur:11.4f} | {p_maj:13.6f} | {50*p_maj:10.2f}")

# =========================================================================
# SECTION 2: TIME TRADEOFF
# =========================================================================
print("\n" + "=" * 80)
print("SECTION 2: TIME TRADEOFF (16 parallel workers, 30s per attempt)")
print("=" * 80)

workers = 16
time_per_attempt = 30
total_time = 32000
problems = 50

print(f"\nAvailable time: {total_time}s, Workers: {workers}, Time/attempt: {time_per_attempt}s\n")

for N in [4, 8, 12, 16, 20, 24, 32, 48, 64]:
    total_attempts = N * problems
    batches = math.ceil(total_attempts / workers)
    wall_time = batches * time_per_attempt
    fits = "YES" if wall_time <= total_time else "NO "
    print(f"N={N:2d}: {total_attempts:5d} attempts, {batches:4d} batches, "
          f"wall_time={wall_time:6d}s ({wall_time/3600:.1f}h) [{fits}]")

max_N = int(total_time / time_per_attempt * workers / problems)
print(f"\nMax N (fully parallel): {max_N}")
print(f"  E[score] majority(N={max_N-1 if max_N%2==0 else max_N}) = {50*majority_vote_accuracy_odd(max_N-1 if max_N%2==0 else max_N, p):.2f}")
print(f"  E[score] plurality(N={max_N}) = {50*(1-(1-p)**max_N):.4f}")

# =========================================================================
# SECTION 3: ADAPTIVE N WITH EARLY STOPPING
# =========================================================================
print("\n" + "=" * 80)
print("SECTION 3: ADAPTIVE N (early_stop=K, N_max=16)")
print("=" * 80)

def expected_attempts_early_stop(p_correct, K, N_max):
    """Scattered wrong answers. Stop when K attempts agree."""
    @lru_cache(maxsize=None)
    def solve(attempts, correct_count):
        if correct_count >= K:
            return (0.0, 1.0)
        if attempts >= N_max:
            return (0.0, 1.0 if correct_count >= 1 else 0.0)
        e_c, pw_c = solve(attempts + 1, correct_count + 1)
        e_w, pw_w = solve(attempts + 1, correct_count)
        e_total = 1.0 + p_correct * e_c + (1 - p_correct) * e_w
        p_win = p_correct * pw_c + (1 - p_correct) * pw_w
        return (e_total, p_win)
    result = solve(0, 0)
    solve.cache_clear()
    return result

print("\nScattered wrong answers (best case), early_stop=4, N_max=16:")
print(f"{'p':>6} | {'E[attempts]':>12} | {'P(correct)':>12} | {'E[score]/50':>12} | {'Time saved':>12}")
print("-" * 65)

for p_val in [0.95, 0.85, 0.75, 0.69, 0.60, 0.50, 0.40, 0.30]:
    e_att, p_win = expected_attempts_early_stop(p_val, 4, 16)
    time_saved_pct = (1 - e_att / 16) * 100
    print(f"{p_val:6.2f} | {e_att:12.2f} | {p_win:12.6f} | {50*p_win:12.2f} | {time_saved_pct:11.1f}%")

# Different early stop thresholds
print("\nComparing early_stop thresholds at p=0.69, N_max=16:")
print(f"{'K':>4} | {'E[attempts]':>12} | {'P(correct)':>12} | {'E[score]/50':>12}")
print("-" * 50)
for K in [2, 3, 4, 5, 6, 8]:
    e_att, p_win = expected_attempts_early_stop(0.69, K, 16)
    print(f"{K:4d} | {e_att:12.2f} | {p_win:12.6f} | {50*p_win:12.2f}")

# =========================================================================
# SECTION 3b: CORRELATED WRONG ANSWERS
# =========================================================================
print("\n" + "=" * 80)
print("SECTION 3b: CORRELATED WRONG ANSWERS (alpha=fraction giving same wrong)")
print("=" * 80)

def expected_attempts_correlated(p_correct, alpha, K, N_max):
    """Three types: correct (p), common_mistake ((1-p)*alpha), random_wrong ((1-p)*(1-alpha))"""
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

print("\nN_max=16, early_stop=4, p=0.69, varying alpha:")
print(f"{'alpha':>6} | {'E[attempts]':>12} | {'P(correct)':>12} | {'E[score]/50':>12}")
print("-" * 50)
for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    e_att, p_win = expected_attempts_correlated(0.69, alpha, 4, 16)
    print(f"{alpha:6.2f} | {e_att:12.2f} | {p_win:12.6f} | {50*p_win:12.2f}")

# =========================================================================
# SECTION 4: SCORE GAP ANALYSIS
# =========================================================================
print("\n" + "=" * 80)
print("SECTION 4: SCORE GAP ANALYSIS")
print("=" * 80)

current_score = 39
current_N = 8

# With p=0.69 and N=8 (majority vote, odd=7):
theoretical_majority_7 = 50 * majority_vote_accuracy_odd(7, p)
theoretical_majority_9 = 50 * majority_vote_accuracy_odd(9, p)
theoretical_plurality_8 = 50 * (1 - (1-p)**8)
print(f"\nOur current score: {current_score}/50 with N={current_N}")
print(f"Theoretical majority vote N=7: {theoretical_majority_7:.2f}/50")
print(f"Theoretical majority vote N=9: {theoretical_majority_9:.2f}/50")
print(f"Theoretical plurality vote N=8 (scattered wrongs): {theoretical_plurality_8:.2f}/50")
print(f"\nOur score ({current_score}) vs majority N=7 ({theoretical_majority_7:.1f}): gap = {current_score - theoretical_majority_7:.2f}")
print(f"Our score ({current_score}) vs plurality N=8 ({theoretical_plurality_8:.1f}): gap = {current_score - theoretical_plurality_8:.2f}")

print("\n--- What p would leaderboard leaders need? ---")
for target in [44, 45, 46]:
    # Find p such that majority_vote(N=8) = target/50
    # Binary search
    lo, hi = 0.5, 0.99
    for _ in range(100):
        mid = (lo + hi) / 2
        score = 50 * majority_vote_accuracy_odd(7, mid)
        if score < target:
            lo = mid
        else:
            hi = mid
    p_needed_maj = (lo + hi) / 2

    lo, hi = 0.5, 0.99
    for _ in range(100):
        mid = (lo + hi) / 2
        score = 50 * majority_vote_accuracy_odd(15, mid)
        if score < target:
            lo = mid
        else:
            hi = mid
    p_needed_maj16 = (lo + hi) / 2

    lo, hi = 0.5, 0.99
    for _ in range(100):
        mid = (lo + hi) / 2
        score = 50 * (1 - (1-mid)**16)
        if score < target:
            lo = mid
        else:
            hi = mid
    p_needed_plur16 = (lo + hi) / 2

    print(f"Score {target}/50: need p={p_needed_maj:.4f} (maj,N=7), "
          f"p={p_needed_maj16:.4f} (maj,N=15), p={p_needed_plur16:.4f} (plur,N=16)")

print("\n--- Alternatively: what N do WE need at p=0.69? ---")
for target in [44, 45, 46]:
    for N in range(1, 200, 2):
        if 50 * majority_vote_accuracy_odd(N, p) >= target:
            print(f"Score {target}/50: need N>={N} (majority vote)")
            break

# =========================================================================
# SECTION 5: ANTI-CORRELATION EFFECT
# =========================================================================
print("\n" + "=" * 80)
print("SECTION 5: ANTI-CORRELATION EFFECT (rho = -0.258)")
print("=" * 80)

# With correlated Bernoulli trials, the effective p for majority vote changes.
# If X_i are correlated Bernoulli(p) with pairwise correlation rho:
# Var(sum) = N*p*(1-p) + N*(N-1)*rho*p*(1-p) = N*p*(1-p)*(1 + (N-1)*rho)
#
# For negative rho: variance DECREASES, sum concentrates around mean.
# This means: P(majority correct) INCREASES with negative correlation!
# The sum is more concentrated around Np, which is above N/2 when p>0.5.

rho = -0.258
print(f"\nCorrelation rho = {rho}")
print(f"Per-attempt accuracy p = {p}")
print(f"\nWith N independent trials: Var(S) = N*p*(1-p) = N*{p*(1-p):.4f}")
print(f"With rho={rho}: Var(S) = N*p*(1-p)*(1 + (N-1)*rho)")
print()

# Use normal approximation for correlated case
def majority_vote_correlated_normal_approx(N, p, rho):
    """Normal approximation to P(S > N/2) where S = sum of correlated Bernoulli."""
    mu = N * p
    # Variance of sum of correlated Bernoulli
    var = N * p * (1-p) * (1 + (N-1) * rho)
    if var <= 0:
        # Perfect concentration at mean
        return 1.0 if mu > N/2 else 0.0
    sigma = math.sqrt(var)
    # P(S > N/2) using normal approx with continuity correction
    from scipy.stats import norm
    z = (N/2 + 0.5 - mu) / sigma  # continuity correction
    return float(1 - norm.cdf(z))

print(f"{'N':>4} | {'P(indep)':>12} | {'P(rho={:.3f})':>14} | {'E[indep]/50':>12} | {'E[corr]/50':>12} | {'Gain':>8}".format(rho))
print("-" * 75)
for N in [3, 5, 7, 9, 11, 13, 15, 17, 21, 25, 31]:
    p_indep = majority_vote_accuracy_odd(N, p)
    p_corr = majority_vote_correlated_normal_approx(N, p, rho)
    gain = 50 * (p_corr - p_indep)
    print(f"{N:4d} | {p_indep:12.6f} | {p_corr:14.6f} | {50*p_indep:12.2f} | {50*p_corr:12.2f} | {gain:+8.2f}")

print("\nMeaning of NEGATIVE correlation (rho = -0.258):")
print("  When one attempt gets it WRONG, other attempts are MORE LIKELY to get it RIGHT.")
print("  This is consistent with: model explores DIVERSE solution paths.")
print("  Temperature/sampling diversity creates anti-correlated errors.")
print("  EFFECT: Majority vote is BETTER than independent-trial prediction!")
print("  The sum of correct votes concentrates tighter around the mean (Np),")
print("  reducing the probability of getting fewer than N/2 correct.")

# Compute: with rho=-0.258, what effective N do we get?
print("\n--- Effective N: independent trials that match correlated performance ---")
for N_actual in [7, 9, 11, 15]:
    p_corr = majority_vote_correlated_normal_approx(N_actual, p, rho)
    # Find N_eff such that majority_vote(N_eff, p) = p_corr
    for N_eff in range(N_actual, 200, 2):
        if majority_vote_accuracy_odd(N_eff, p) >= p_corr:
            print(f"N={N_actual} with rho={rho} ~ N_eff={N_eff} independent trials "
                  f"(P={p_corr:.6f}, E[score]={50*p_corr:.2f})")
            break

# =========================================================================
# SECTION 6: FLAT vs ADAPTIVE COMPARISON
# =========================================================================
print("\n" + "=" * 80)
print("SECTION 6: FLAT vs ADAPTIVE STRATEGY")
print("=" * 80)

# Assume problem difficulty is heterogeneous:
# Easy (20%): p=0.90
# Medium (50%): p=0.69
# Hard (30%): p=0.40

difficulty_dist = [(0.20, 0.90), (0.50, 0.69), (0.30, 0.40)]

print("\nProblem difficulty distribution:")
for frac, pi in difficulty_dist:
    print(f"  {frac*100:.0f}% of problems: p={pi}")

# FLAT: N=16 for all problems
print("\n--- FLAT STRATEGY: N=16 for all 50 problems ---")
flat_total_attempts = 0
flat_score = 0
for frac, pi in difficulty_dist:
    n_problems = 50 * frac
    acc = majority_vote_accuracy_odd(15, pi)  # use 15 (odd)
    flat_score += n_problems * acc
    flat_total_attempts += n_problems * 16
print(f"Total attempts: {flat_total_attempts:.0f}")
print(f"Expected score: {flat_score:.2f}/50")
print(f"Wall time: {math.ceil(flat_total_attempts/16)*30:.0f}s")

# ADAPTIVE: early_stop=4, N_max=24
print("\n--- ADAPTIVE STRATEGY: early_stop=4, N_max=24 ---")
adaptive_total_attempts = 0
adaptive_score = 0
for frac, pi in difficulty_dist:
    n_problems = 50 * frac
    e_att, p_win = expected_attempts_early_stop(pi, 4, 24)
    adaptive_score += n_problems * p_win
    adaptive_total_attempts += n_problems * e_att
print(f"Expected total attempts: {adaptive_total_attempts:.0f}")
print(f"Expected score: {adaptive_score:.2f}/50")
print(f"Expected wall time: {math.ceil(adaptive_total_attempts/16)*30:.0f}s")

# ADAPTIVE with correlated wrongs (alpha=0.3)
print("\n--- ADAPTIVE STRATEGY: early_stop=4, N_max=24, alpha=0.3 ---")
adaptive_corr_attempts = 0
adaptive_corr_score = 0
for frac, pi in difficulty_dist:
    n_problems = 50 * frac
    e_att, p_win = expected_attempts_correlated(pi, 0.3, 4, 24)
    adaptive_corr_score += n_problems * p_win
    adaptive_corr_attempts += n_problems * e_att
print(f"Expected total attempts: {adaptive_corr_attempts:.0f}")
print(f"Expected score: {adaptive_corr_score:.2f}/50")
print(f"Expected wall time: {math.ceil(adaptive_corr_attempts/16)*30:.0f}s")

# OPTIMAL ADAPTIVE: Use saved time from easy problems on hard problems
print("\n--- TWO-PHASE ADAPTIVE ---")
print("Phase 1: Run N=8 for all problems (quick)")
print("Phase 2: Run N_extra for problems without clear winner")
phase1_time = math.ceil(50 * 8 / 16) * 30
remaining_time = total_time - phase1_time
print(f"Phase 1 time: {phase1_time}s")
print(f"Remaining time: {remaining_time}s")
extra_attempts_budget = int(remaining_time / 30 * 16)
print(f"Extra attempts budget: {extra_attempts_budget}")

# After phase 1: which problems have clear winner?
phase2_score = 0
phase2_extra_used = 0
for frac, pi in difficulty_dist:
    n_problems = 50 * frac
    # After 8 attempts: P(>=4 agree)
    p_clear = sum(comb(8, k, exact=True) * pi**k * (1-pi)**(8-k) for k in range(4, 9))
    n_clear = n_problems * p_clear
    n_unclear = n_problems * (1 - p_clear)

    # Clear problems: score = P(4+ correct out of 8)
    p_correct_clear = sum(comb(8, k, exact=True) * pi**k * (1-pi)**(8-k) for k in range(5, 9))
    # This is majority of 8... but with scattered wrongs,
    # "clear winner" means the correct answer has 4+ votes
    # which happens when 4+ attempts are correct
    p_correct_given_clear = sum(comb(8, k, exact=True) * pi**k * (1-pi)**(8-k) for k in range(4, 9))

    # Unclear: need more attempts. Budget: up to 16 more
    extra_per_unclear = min(16, extra_attempts_budget / max(n_unclear, 0.01))

    # Score from clear problems
    phase2_score += n_clear  # assuming clear = correct (scattered wrongs)

    # Score from unclear problems with extra attempts
    if n_unclear > 0:
        total_N = 8 + min(16, int(extra_per_unclear))
        p_unclear_correct = 1 - (1-pi)**total_N  # plurality with scattered
        phase2_score += n_unclear * p_unclear_correct
        phase2_extra_used += n_unclear * min(16, int(extra_per_unclear))

print(f"Two-phase expected score: {phase2_score:.2f}/50")

# =========================================================================
# SECTION 7: FINAL OUTPUT JSON
# =========================================================================
print("\n" + "=" * 80)
print("SECTION 7: FINAL OUTPUT JSON")
print("=" * 80)

# Determine truly optimal N given time budget
# With 16 workers: max total attempts = 32000/30 * 16 = 17066
# Per problem: max 17066/50 = 341 attempts (way more than needed)
# So time is NOT the binding constraint. GPU throughput might be.
max_total = int(total_time / time_per_attempt * workers)
max_per_problem = max_total // problems
print(f"\nMax total attempts: {max_total}")
print(f"Max per problem: {max_per_problem}")

# Find optimal flat N
best_N = 1
best_score = 0
for N in range(1, min(max_per_problem+1, 65), 2):
    acc = majority_vote_accuracy_odd(N, p)
    score = 50 * acc
    if score > best_score:
        best_score = score
        best_N = N

# With anti-correlation
best_N_corr = 1
best_score_corr = 0
for N in range(1, min(max_per_problem+1, 65), 2):
    acc = majority_vote_correlated_normal_approx(N, p, rho)
    score = 50 * acc
    if score > best_score_corr:
        best_score_corr = score
        best_N_corr = N

# Adaptive scores
_, adaptive_flat_score_p69 = expected_attempts_early_stop(0.69, 4, 16)
e_att_adaptive, _ = expected_attempts_early_stop(0.69, 4, 16)

result = {
    "optimal_N": {
        "flat_majority_independent": best_N,
        "flat_majority_with_anticorrelation": best_N_corr,
        "adaptive_N_max": 24,
        "adaptive_early_stop_K": 4,
        "note": f"Time budget allows up to N={max_per_problem} per problem. N={best_N} gives diminishing returns; N=15-25 is the sweet spot."
    },
    "expected_score_at_optimal_N": {
        "flat_N15_majority_independent": round(50 * majority_vote_accuracy_odd(15, p), 2),
        "flat_N15_majority_anticorrelated": round(50 * majority_vote_correlated_normal_approx(15, p, rho), 2),
        "flat_N15_plurality_scattered": round(50 * (1-(1-p)**15), 4),
        "flat_N25_majority_independent": round(50 * majority_vote_accuracy_odd(25, p), 2),
        "adaptive_K4_Nmax24_heterogeneous": round(adaptive_score, 2),
    },
    "current_score_gap_analysis": {
        "our_score": 39,
        "theoretical_majority_N7": round(50 * majority_vote_accuracy_odd(7, p), 2),
        "theoretical_majority_N15": round(50 * majority_vote_accuracy_odd(15, p), 2),
        "theoretical_majority_N15_anticorr": round(50 * majority_vote_correlated_normal_approx(15, p, rho), 2),
        "leaderboard_top": "44-46",
        "explanation": (
            "At p=0.69, N=7 majority vote gives 43.03 expected score. "
            "Our 39 is BELOW this, suggesting either: (1) our effective p < 0.69 on some problems, "
            "(2) correlated wrong answers reduce voting effectiveness, or (3) implementation bugs "
            "lose recoverable answers. To reach 44-46, leaders likely have p~0.75-0.80 OR use "
            "N=15+ with anti-correlated errors. The SINGLE biggest lever is increasing p (better prompts, "
            "code execution reliability, answer extraction)."
        )
    },
    "adaptive_vs_flat": {
        "flat_N16_score": round(50 * majority_vote_accuracy_odd(15, p), 2),
        "adaptive_K4_Nmax24_score_heterogeneous": round(adaptive_score, 2),
        "adaptive_time_savings": "Easy problems (p=0.90) stop in ~4.5 attempts vs 16, freeing budget for hard problems",
        "recommendation": "Adaptive saves ~40% time on easy problems, allowing N_max=24 within same budget"
    },
    "anti_correlation_effect": {
        "rho": rho,
        "meaning": (
            "Negative correlation means: when one attempt fails, others are MORE likely to succeed. "
            "This arises from diverse sampling (high temperature, varied prompts). "
            "Effect: majority vote accuracy INCREASES. At N=15, anti-correlation gives "
            f"~{50*majority_vote_correlated_normal_approx(15, p, rho) - 50*majority_vote_accuracy_odd(15, p):.1f} "
            "extra expected score vs independent trials."
        ),
        "N15_independent": round(50 * majority_vote_accuracy_odd(15, p), 2),
        "N15_anticorrelated": round(50 * majority_vote_correlated_normal_approx(15, p, rho), 2),
        "effective_N_independent_equivalent": "N=15 with rho=-0.258 performs like N~25-31 independent trials"
    },
    "recommended_architecture": {
        "strategy": "Two-phase adaptive with anti-correlated sampling",
        "phase1": {
            "N": 8,
            "purpose": "Quick triage - identify easy vs hard problems",
            "time": f"{math.ceil(50*8/16)*30}s"
        },
        "phase2": {
            "N_extra": "up to 16 more for unclear problems",
            "early_stop": 4,
            "purpose": "Concentrate budget on hard problems"
        },
        "sampling": {
            "temperature": "0.8-1.0 to maximize diversity (anti-correlation)",
            "varied_prompts": "Use 3-4 prompt variants to decorrelate errors",
            "code_execution": "Always use TIR (Tool-Integrated Reasoning)"
        },
        "voting": {
            "method": "Plurality (not strict majority) - scattered wrong answers mean plurality ~ 1-(1-p)^N",
            "tiebreak": "Pick answer with highest confidence score if tie"
        },
        "expected_score": f"{round(50 * majority_vote_correlated_normal_approx(15, p, rho), 1)}/50 (conservative), up to {round(50 * (1-(1-p)**24), 1)}/50 (optimistic plurality)",
        "critical_insight": (
            "The #1 lever is INCREASING p, not increasing N. "
            "Going from p=0.69 to p=0.75 with N=15 gains +2.3 score. "
            "Going from N=15 to N=25 at p=0.69 gains +1.5 score. "
            "Focus: better prompts, reliable code execution, robust answer extraction."
        )
    }
}

print("\n" + json.dumps(result, indent=2))

# Final summary table
print("\n" + "=" * 80)
print("SUMMARY: EXPECTED SCORES BY STRATEGY")
print("=" * 80)
strategies = [
    ("Current (N=8, p=0.69)", 39),
    ("Majority N=7, independent", round(50 * majority_vote_accuracy_odd(7, p), 1)),
    ("Majority N=15, independent", round(50 * majority_vote_accuracy_odd(15, p), 1)),
    ("Majority N=15, rho=-0.258", round(50 * majority_vote_correlated_normal_approx(15, p, rho), 1)),
    ("Majority N=25, independent", round(50 * majority_vote_accuracy_odd(25, p), 1)),
    ("Plurality N=16, scattered", round(50 * (1-(1-p)**16), 1)),
    ("Plurality N=24, scattered", round(50 * (1-(1-p)**24), 1)),
    ("Adaptive K=4, Nmax=24, heterogeneous", round(adaptive_score, 1)),
    ("If p=0.75, N=15, majority", round(50 * majority_vote_accuracy_odd(15, 0.75), 1)),
    ("If p=0.80, N=15, majority", round(50 * majority_vote_accuracy_odd(15, 0.80), 1)),
]
for name, score in strategies:
    bar = "#" * int(score)
    print(f"  {name:45s} | {score:5.1f} | {bar}")
