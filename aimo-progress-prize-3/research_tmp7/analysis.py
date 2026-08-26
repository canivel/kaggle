import math

# === QUANTITATIVE ANALYSIS ===

# Base parameters
p_per_attempt = 0.69  # from arxiv paper
N = 12  # current attempts
n_problems = 50

# Distribution of problems by difficulty
easy_count, easy_p = 35, 0.95
med_count, med_p = 10, 0.60
hard_count, hard_p = 5, 0.25

def p_majority(p, N):
    threshold = (N + 1) // 2
    total = 0
    for k in range(threshold, N+1):
        binom = math.comb(N, k) * p**k * (1-p)**(N-k)
        total += binom
    return total

def expected_score(ep, mp, hp, N):
    return (easy_count * p_majority(ep, N) +
            med_count * p_majority(mp, N) +
            hard_count * p_majority(hp, N))

baseline = expected_score(easy_p, med_p, hard_p, N)
print(f"=== BASELINE ===")
print(f"Expected score with N={N}: {baseline:.2f}/50")
print(f"  Easy ({easy_count}p, p={easy_p}): {easy_count * p_majority(easy_p, N):.2f}")
print(f"  Medium ({med_count}p, p={med_p}): {med_count * p_majority(med_p, N):.2f}")
print(f"  Hard ({hard_count}p, p={hard_p}): {hard_count * p_majority(hard_p, N):.2f}")

print()
print("=== STRATEGY 1: VERIFICATION TURN ===")

p_majority_wrong_easy = 1 - p_majority(easy_p, N)
p_majority_wrong_med = 1 - p_majority(med_p, N)
p_majority_wrong_hard = 1 - p_majority(hard_p, N)

print(f"P(majority wrong) - Easy: {p_majority_wrong_easy:.6f}")
print(f"P(majority wrong) - Medium: {p_majority_wrong_med:.4f}")
print(f"P(majority wrong) - Hard: {p_majority_wrong_hard:.4f}")

# Conservative estimates (accounting for Pawan Mali V133 null result)
p_verify_catch_wrong = 0.55
p_verify_false_reject = 0.15
p_2nd_correct_easy = 0.95
p_2nd_correct_med = 0.45
p_2nd_correct_hard = 0.25

recovered = (easy_count * p_majority_wrong_easy * p_verify_catch_wrong * p_2nd_correct_easy +
             med_count * p_majority_wrong_med * p_verify_catch_wrong * p_2nd_correct_med +
             hard_count * p_majority_wrong_hard * p_verify_catch_wrong * p_2nd_correct_hard)
lost = (easy_count * p_majority(easy_p, N) * p_verify_false_reject * (1 - p_2nd_correct_easy) +
        med_count * p_majority(med_p, N) * p_verify_false_reject * (1 - p_2nd_correct_med) +
        hard_count * p_majority(hard_p, N) * p_verify_false_reject * (1 - p_2nd_correct_hard))
net_gain_verify = recovered - lost
print(f"Recovered: {recovered:.3f}")
print(f"Lost (false rejections): {lost:.3f}")
print(f"NET GAIN (conservative): +{net_gain_verify:.2f} problems")

# Optimistic (code-based verification really is better than CORRECT/WRONG)
p_verify_catch_opt = 0.75
p_verify_false_reject_opt = 0.08
recovered_opt = (easy_count * p_majority_wrong_easy * p_verify_catch_opt * p_2nd_correct_easy +
                 med_count * p_majority_wrong_med * p_verify_catch_opt * p_2nd_correct_med +
                 hard_count * p_majority_wrong_hard * p_verify_catch_opt * p_2nd_correct_hard)
lost_opt = (easy_count * p_majority(easy_p, N) * p_verify_false_reject_opt * (1 - p_2nd_correct_easy) +
            med_count * p_majority(med_p, N) * p_verify_false_reject_opt * (1 - p_2nd_correct_med) +
            hard_count * p_majority(hard_p, N) * p_verify_false_reject_opt * (1 - p_2nd_correct_hard))
net_gain_verify_opt = recovered_opt - lost_opt
print(f"NET GAIN (optimistic): +{net_gain_verify_opt:.2f} problems")

print()
print("=== STRATEGY 2: CROSS-ATTEMPT SYNTHESIS ===")

def p_split(p, N, threshold=4):
    return sum(math.comb(N, k) * p**k * (1-p)**(N-k) for k in range(threshold))

p_split_easy = p_split(easy_p, N)
p_split_med = p_split(med_p, N)
p_split_hard = p_split(hard_p, N)
print(f"P(no 4+ consensus) - Easy: {p_split_easy:.8f}")
print(f"P(no 4+ consensus) - Medium: {p_split_med:.4f}")
print(f"P(no 4+ consensus) - Hard: {p_split_hard:.4f}")

n_split = easy_count * p_split_easy + med_count * p_split_med + hard_count * p_split_hard
print(f"Expected problems with split votes: {n_split:.2f}")

p_entropy_correct_split = 0.50
p_synthesis_correct = 0.65
net_synthesis = n_split * (p_synthesis_correct - p_entropy_correct_split)
print(f"Net gain from synthesis: +{net_synthesis:.2f} problems")

time_cost_per_split = 45
total_time_cost = n_split * time_cost_per_split
print(f"Time cost: {total_time_cost:.0f}s total")

p_synthesis_hurts = n_split * p_entropy_correct_split * (1 - p_synthesis_correct)
print(f"Risk (problems hurt): {p_synthesis_hurts:.2f}")

print()
print("=== STRATEGY 3: DECOMPOSITION ===")

p_decomp_boost_hard = 0.15
p_decomp_boost_med = 0.05

score_decomp_6 = (easy_count * p_majority(easy_p, N) +
                  med_count * p_majority(med_p + p_decomp_boost_med, N) +
                  hard_count * p_majority(hard_p + p_decomp_boost_hard, 6))
net_decomp_6 = score_decomp_6 - baseline

score_decomp_8 = (easy_count * p_majority(easy_p, N) +
                  med_count * p_majority(med_p + p_decomp_boost_med, N) +
                  hard_count * p_majority(hard_p + p_decomp_boost_hard, 8))
net_decomp_8 = score_decomp_8 - baseline

print(f"Decomposition (hard only, N=6): NET +{net_decomp_6:.2f}")
print(f"Decomposition (hard only, N=8): NET +{net_decomp_8:.2f}")

p_bad_decomp = 0.20
expected_hurt = hard_count * p_majority(hard_p, N) * p_bad_decomp
print(f"Risk (bad decomposition hurts): {expected_hurt:.2f} problems")
print(f"  p_majority(0.40, 8) = {p_majority(0.40, 8):.4f}")
print(f"  p_majority(0.25, 12) = {p_majority(0.25, 12):.4f}")

print()
print("=== COMBINED S1+S2 ===")
combined = net_gain_verify + net_synthesis
combined_opt = net_gain_verify_opt + net_synthesis
print(f"Combined S1+S2 (conservative): +{combined:.2f} pts")
print(f"Combined S1+S2 (optimistic):   +{combined_opt:.2f} pts")

print()
print("=== TIME BUDGET ===")
print(f"S1: +20-30s per problem = {25*50}s = {25*50/3600:.1f}h overhead")
print(f"S2: +45s per {n_split:.1f} split problems = {total_time_cost:.0f}s = {total_time_cost/3600:.2f}h overhead")
print(f"S3: doubles time for hard problems = costly")
print(f"Total budget: 5h = 18000s")
print(f"S1+S2 overhead: {25*50 + total_time_cost:.0f}s = {(25*50 + total_time_cost)/18000*100:.1f}% of budget")
