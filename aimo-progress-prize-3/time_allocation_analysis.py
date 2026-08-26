"""
AIMO3 Time Allocation Optimization Analysis
50 problems, 9 hours (32400s), 16 parallel kernels
"""
import json
import math
import random

random.seed(42)

###############################################################################
# CONSTANTS
###############################################################################
TOTAL_TIME = 32400  # 9 hours
SETUP_TIME = 400
AVAILABLE_TIME = TOTAL_TIME - SETUP_TIME  # 32000s
PARALLEL_KERNELS = 16
N_PROBLEMS = 50

EASY = 30    # p=0.90 per attempt
MEDIUM = 15  # p=0.70 per attempt
HARD = 5     # p=0.40 per attempt

# Realistic time per attempt (execution time, not timeout)
EASY_TPA = 20     # easy problems solve quickly
MEDIUM_TPA = 50   # medium take longer
HARD_TPA = 80     # hard problems use more tokens/time

N_SIMS = 200000

###############################################################################
# HELPERS
###############################################################################

def p_majority_correct(p, n_attempts):
    """P(majority of n independent Bernoulli(p) trials are 1)."""
    threshold = (n_attempts // 2) + 1
    prob = 0.0
    for k in range(threshold, n_attempts + 1):
        prob += math.comb(n_attempts, k) * (p ** k) * ((1 - p) ** (n_attempts - k))
    return prob


def simulate_problem(p, n_attempts, early_stop, time_per_attempt, n_sims=N_SIMS):
    """Simulate a problem with majority vote and early stopping."""
    correct_count = 0
    total_time = 0
    total_attempts = 0

    for _ in range(n_sims):
        results = []
        attempts_used = 0

        for i in range(n_attempts):
            correct = random.random() < p
            results.append(1 if correct else 0)
            attempts_used += 1

            # Check early stop: do we have early_stop results that agree?
            c = sum(results)
            w = len(results) - c
            if c >= early_stop or w >= early_stop:
                break

        # Majority vote
        c = sum(results)
        if c > len(results) / 2:
            correct_count += 1

        total_time += attempts_used * time_per_attempt
        total_attempts += attempts_used

    return {
        'accuracy': correct_count / n_sims,
        'avg_time': total_time / n_sims,
        'avg_attempts': total_attempts / n_sims
    }


###############################################################################
# TASK 1: FLAT vs ADAPTIVE Expected Scores
###############################################################################
print("=" * 70)
print("TASK 1: FLAT vs ADAPTIVE Expected Scores")
print("=" * 70)

# -- Analytical (majority vote) --
flat_easy_acc = p_majority_correct(0.90, 8)
flat_medium_acc = p_majority_correct(0.70, 8)
flat_hard_acc = p_majority_correct(0.40, 8)
flat_expected_analytical = EASY * flat_easy_acc + MEDIUM * flat_medium_acc + HARD * flat_hard_acc

print(f"\nFLAT STRATEGY (8 attempts, early_stop=4, 300s timeout):")
print(f"  Easy   (n={EASY}, p=0.90): P(majority correct) = {flat_easy_acc:.6f}")
print(f"  Medium (n={MEDIUM}, p=0.70): P(majority correct) = {flat_medium_acc:.6f}")
print(f"  Hard   (n={HARD}, p=0.40): P(majority correct) = {flat_hard_acc:.6f}")
print(f"  Expected score (analytical): {flat_expected_analytical:.2f}")

# -- Flat simulation --
print("\n  Monte Carlo simulation (flat 8 attempts, early_stop=4):")
flat_sim = {}
for label, p, tpa in [('easy', 0.90, EASY_TPA), ('medium', 0.70, MEDIUM_TPA), ('hard', 0.40, HARD_TPA)]:
    result = simulate_problem(p, 8, 4, tpa)
    flat_sim[label] = result
    print(f"    {label}: acc={result['accuracy']:.4f}, avg_attempts={result['avg_attempts']:.1f}, avg_time={result['avg_time']:.0f}s")

flat_sim_score = 30*flat_sim['easy']['accuracy'] + 15*flat_sim['medium']['accuracy'] + 5*flat_sim['hard']['accuracy']
flat_sim_compute = 30*flat_sim['easy']['avg_time'] + 15*flat_sim['medium']['avg_time'] + 5*flat_sim['hard']['avg_time']
flat_sim_wall = flat_sim_compute / PARALLEL_KERNELS
print(f"  Sim score: {flat_sim_score:.2f}, Compute: {flat_sim_compute:.0f}s, Wall: {flat_sim_wall:.0f}s + {SETUP_TIME}s = {flat_sim_wall+SETUP_TIME:.0f}s")


# -- Adaptive: allocate more to harder problems --
# We search over attempt counts to find the best allocation that fits in budget

print(f"\n{'='*70}")
print("OPTIMAL ALLOCATION SEARCH (brute force over attempt counts)")
print("=" * 70)

best_score = 0
best_config = None

for n_e in range(2, 10, 2):
    for n_m in range(4, 14, 2):
        for n_h in range(6, 22, 2):
            es_e = max(2, (n_e // 2) + 1)
            es_m = max(3, (n_m // 2) + 1)
            es_h = max(4, (n_h // 2) + 1)

            # Quick time estimates (approximate based on early stopping patterns)
            # P(stop early at es) ~ p^es + (1-p)^es for all-agree
            p_early_e = 0.9**es_e + 0.1**es_e
            p_early_m = 0.7**es_m + 0.3**es_m
            p_early_h = 0.4**es_h + 0.6**es_h

            eff_att_e = es_e * p_early_e + n_e * (1 - p_early_e)
            eff_att_m = es_m * p_early_m + n_m * (1 - p_early_m)
            eff_att_h = es_h * p_early_h + n_h * (1 - p_early_h)

            time_e = eff_att_e * EASY_TPA
            time_m = eff_att_m * MEDIUM_TPA
            time_h = eff_att_h * HARD_TPA

            total_compute = EASY * time_e + MEDIUM * time_m + HARD * time_h
            wall_time = total_compute / PARALLEL_KERNELS

            if wall_time + SETUP_TIME > TOTAL_TIME:
                continue

            score = (EASY * p_majority_correct(0.90, n_e) +
                     MEDIUM * p_majority_correct(0.70, n_m) +
                     HARD * p_majority_correct(0.40, n_h))

            if score > best_score:
                best_score = score
                best_config = {
                    'n_easy': n_e, 'n_medium': n_m, 'n_hard': n_h,
                    'es_easy': es_e, 'es_medium': es_m, 'es_hard': es_h,
                    'time_easy': time_e, 'time_medium': time_m, 'time_hard': time_h,
                    'wall_time': wall_time + SETUP_TIME,
                    'total_compute': total_compute,
                    'acc_easy': p_majority_correct(0.90, n_e),
                    'acc_medium': p_majority_correct(0.70, n_m),
                    'acc_hard': p_majority_correct(0.40, n_h),
                }

print(f"\n  OPTIMAL ALLOCATION (analytical):")
print(f"    Easy:   {best_config['n_easy']} attempts (es={best_config['es_easy']}), ~{best_config['time_easy']:.0f}s, acc={best_config['acc_easy']:.4f}")
print(f"    Medium: {best_config['n_medium']} attempts (es={best_config['es_medium']}), ~{best_config['time_medium']:.0f}s, acc={best_config['acc_medium']:.4f}")
print(f"    Hard:   {best_config['n_hard']} attempts (es={best_config['es_hard']}), ~{best_config['time_hard']:.0f}s, acc={best_config['acc_hard']:.4f}")
print(f"    Wall time: {best_config['wall_time']:.0f}s / {TOTAL_TIME}s")
print(f"    Expected score: {best_score:.2f}")

# -- Verify with simulation --
ne = best_config['n_easy']
nm = best_config['n_medium']
nh = best_config['n_hard']
es_e = best_config['es_easy']
es_m = best_config['es_medium']
es_h = best_config['es_hard']

print(f"\n  Verifying with Monte Carlo ({N_SIMS} sims):")
adapt_sim = {}
for label, p, tpa, na, es in [
    ('easy', 0.90, EASY_TPA, ne, es_e),
    ('medium', 0.70, MEDIUM_TPA, nm, es_m),
    ('hard', 0.40, HARD_TPA, nh, es_h)
]:
    result = simulate_problem(p, na, es, tpa)
    adapt_sim[label] = result
    print(f"    {label}: acc={result['accuracy']:.4f}, avg_attempts={result['avg_attempts']:.1f}, avg_time={result['avg_time']:.0f}s")

adapt_sim_score = 30*adapt_sim['easy']['accuracy'] + 15*adapt_sim['medium']['accuracy'] + 5*adapt_sim['hard']['accuracy']
adapt_sim_compute = 30*adapt_sim['easy']['avg_time'] + 15*adapt_sim['medium']['avg_time'] + 5*adapt_sim['hard']['avg_time']
adapt_sim_wall = adapt_sim_compute / PARALLEL_KERNELS
print(f"  Sim score: {adapt_sim_score:.2f}, Compute: {adapt_sim_compute:.0f}s, Wall: {adapt_sim_wall:.0f}s + {SETUP_TIME}s = {adapt_sim_wall+SETUP_TIME:.0f}s")


###############################################################################
# TASK 2: Early Detection Probabilities
###############################################################################
print(f"\n{'='*70}")
print("TASK 2: Early Detection Analysis (Bayesian)")
print("=" * 70)

# P(correct | 2/2 agree) using Bayes
# For math problems, if both attempts give the same numeric answer:
# P(2 agree on correct) = p^2
# P(2 agree on wrong) = (1-p)^2 * P(same wrong answer) ~ very small for math
# Assume P(two wrong answers match) ~ 0.03 (integer answers in a small range)

for p in [0.90, 0.70, 0.40, 0.69]:
    p_wrong_agree = 0.03
    p_2_agree = p**2 + (1-p)**2 * p_wrong_agree
    p_correct_given_agree = p**2 / p_2_agree
    p_2_disagree = 1 - p_2_agree

    # If 2 disagree, P(at least 1 correct in remaining 6)
    p_none_correct_6 = (1 - p) ** 6
    p_any_correct_6 = 1 - p_none_correct_6

    print(f"\n  p={p:.2f}:")
    print(f"    P(2/2 agree)             = {p_2_agree:.4f}")
    print(f"    P(correct | 2/2 agree)   = {p_correct_given_agree:.4f}")
    print(f"    P(2/2 disagree)          = {p_2_disagree:.4f}")
    print(f"    P(>=1 correct in next 6) = {p_any_correct_6:.4f}")

# Specific answers for p=0.69
p = 0.69
p_wrong_agree = 0.03
p_2_agree_69 = p**2 + (1-p)**2 * p_wrong_agree
p_correct_given_agree_69 = p**2 / p_2_agree_69
p_any_correct_6_69 = 1 - (1-p)**6

print(f"\n  KEY ANSWERS (p=0.69):")
print(f"    P(correct | 2/2 agree) = {p_correct_given_agree_69:.4f}")
print(f"    P(any correct in remaining 6 | 2 disagree) = {p_any_correct_6_69:.4f}")


###############################################################################
# TASK 3: Optimal Attempt Counts - Diminishing Returns
###############################################################################
print(f"\n{'='*70}")
print("TASK 3: Diminishing Returns by Difficulty")
print("=" * 70)

for label, p, tpa in [("Easy (p=0.90)", 0.90, 20), ("Medium (p=0.70)", 0.70, 50), ("Hard (p=0.40)", 0.40, 80)]:
    print(f"\n  {label}, {tpa}s per attempt:")
    print(f"  {'n':>4s}  {'accuracy':>10s}  {'time':>6s}  {'eff/1000s':>10s}  {'marginal':>10s}")
    prev_acc = p  # 1 attempt baseline
    for n in [2, 4, 6, 8, 10, 12, 16]:
        acc = p_majority_correct(p, n)
        total_t = n * tpa
        eff = acc / total_t * 1000
        marginal = acc - prev_acc
        print(f"  {n:4d}  {acc:10.6f}  {total_t:6d}  {eff:10.4f}  {marginal:+10.6f}")
        prev_acc = acc

# Recommend: stop where marginal gain < 0.005
print("\n  RECOMMENDATION (stop where marginal < 0.005):")
for label, p in [("Easy p=0.90", 0.90), ("Medium p=0.70", 0.70), ("Hard p=0.40", 0.40)]:
    prev_acc = p
    recommended = 2
    for n in range(2, 20, 2):
        acc = p_majority_correct(p, n)
        if acc - prev_acc < 0.005 and n > 2:
            recommended = n - 2
            break
        prev_acc = acc
        recommended = n
    print(f"    {label}: recommend {recommended} attempts (accuracy={p_majority_correct(p, recommended):.4f})")


###############################################################################
# TASK 4: PHASE-BASED ADAPTIVE with difficulty detection
###############################################################################
print(f"\n{'='*70}")
print("TASK 4: Phase-Based Adaptive Strategy (full simulation)")
print("=" * 70)

def simulate_phase_adaptive(n_sims=N_SIMS):
    """
    Phase 1: Run 2 attempts for all 50 problems.
    Phase 2: Classify by agreement, allocate remaining attempts.

    If 2/2 agree -> "easy path": run 2 more (4 total), early stop OK
    If 2/2 disagree -> "hard path": run 10 more (12 total)
    """
    random.seed(42)

    problems = (
        [(0.90, 'easy')] * 30 +
        [(0.70, 'medium')] * 15 +
        [(0.40, 'hard')] * 5
    )

    total_score = 0
    total_compute = 0
    class_counts = {'agree': 0, 'disagree': 0}
    class_correct = {'agree': 0, 'disagree': 0}

    for _ in range(n_sims):
        score = 0
        compute = 0

        for p, true_type in problems:
            # Phase 1: 2 attempts
            r1 = 1 if random.random() < p else 0
            r2 = 1 if random.random() < p else 0
            compute += 2 * 35  # avg time for initial probing

            if r1 == r2:  # 2/2 agree
                class_counts['agree'] += 1
                # Easy path: 2 more attempts (4 total)
                r3 = 1 if random.random() < p else 0
                r4 = 1 if random.random() < p else 0
                compute += 2 * 35
                results = [r1, r2, r3, r4]

                c = sum(results)
                if c >= 3:
                    score += 1
                    class_correct['agree'] += 1
                elif c <= 1:
                    pass  # majority wrong
                else:
                    # Tie (2-2): escalate, run 4 more
                    for _ in range(4):
                        results.append(1 if random.random() < p else 0)
                    compute += 4 * 45
                    c = sum(results)
                    if c > len(results) / 2:
                        score += 1
                        class_correct['agree'] += 1

            else:  # 2/2 disagree
                class_counts['disagree'] += 1
                results = [r1, r2]
                # Hard path: 10 more attempts (12 total)
                for _ in range(10):
                    results.append(1 if random.random() < p else 0)
                compute += 10 * 50
                c = sum(results)
                if c > len(results) / 2:
                    score += 1
                    class_correct['disagree'] += 1

        total_score += score
        total_compute += compute

    avg_score = total_score / n_sims
    avg_compute = total_compute / n_sims
    avg_agree = class_counts['agree'] / n_sims
    avg_disagree = class_counts['disagree'] / n_sims
    acc_agree = class_correct['agree'] / class_counts['agree'] if class_counts['agree'] > 0 else 0
    acc_disagree = class_correct['disagree'] / class_counts['disagree'] if class_counts['disagree'] > 0 else 0

    print(f"  Classification: {avg_agree:.1f} agree, {avg_disagree:.1f} disagree (per run)")
    print(f"  Accuracy given agree path: {acc_agree:.4f}")
    print(f"  Accuracy given disagree path: {acc_disagree:.4f}")
    print(f"  Avg score: {avg_score:.2f}")
    print(f"  Avg compute: {avg_compute:.0f}s, wall: {avg_compute/PARALLEL_KERNELS:.0f}s")

    return avg_score, avg_compute

phase_score, phase_compute = simulate_phase_adaptive()

print(f"\n{'='*70}")
print("COMPARISON SUMMARY")
print("=" * 70)
print(f"  Flat (8 att, es=4):          {flat_sim_score:.2f} / 50  (wall {flat_sim_wall+SETUP_TIME:.0f}s)")
print(f"  Adaptive (optimal static):   {adapt_sim_score:.2f} / 50  (wall {adapt_sim_wall+SETUP_TIME:.0f}s)")
print(f"  Phase-adaptive (2-phase):    {phase_score:.2f} / 50  (wall {phase_compute/PARALLEL_KERNELS+SETUP_TIME:.0f}s)")
print(f"\n  Adaptive improvement:       +{adapt_sim_score - flat_sim_score:.2f} ({(adapt_sim_score - flat_sim_score)/flat_sim_score*100:.1f}%)")
print(f"  Phase-adaptive improvement: +{phase_score - flat_sim_score:.2f} ({(phase_score - flat_sim_score)/flat_sim_score*100:.1f}%)")


###############################################################################
# FINAL JSON OUTPUT
###############################################################################
print(f"\n{'='*70}")
print("FINAL JSON")
print("=" * 70)

# Build the adaptive_budget function
adaptive_budget_code = '''
def adaptive_budget(self, problem_idx, problems_remaining, elapsed, results_so_far):
    """
    Adaptive time/attempt allocation based on early signal detection.

    Called after each attempt to decide whether to continue or stop.

    Args:
        problem_idx: which problem (0-49)
        problems_remaining: how many problems left to process
        elapsed: total wall time elapsed so far (seconds)
        results_so_far: list of (answer, confidence) from attempts so far

    Returns:
        (n_attempts, timeout_per_attempt, total_budget)
    """
    TIME_LIMIT = 32400  # 9 hours
    SETUP_OVERHEAD = 400
    remaining_wall = TIME_LIMIT - elapsed - SETUP_OVERHEAD

    n_done = len(results_so_far)

    # Extract answers (ignoring confidence for now)
    answers = [r[0] for r in results_so_far if r[0] is not None]

    # Phase 1: Initial probing (0 attempts done yet)
    if n_done == 0:
        # Start with 2 fast probing attempts
        # Reserve time: easy=120s, medium=360s, hard=720s
        # But we don't know difficulty yet, so budget for medium
        base_timeout = min(45, remaining_wall / max(problems_remaining, 1) / 8)
        return (2, base_timeout, 2 * base_timeout)

    # Phase 2: After 2 attempts, classify difficulty
    if n_done >= 2:
        # Count agreement
        from collections import Counter
        answer_counts = Counter(answers)
        most_common_count = answer_counts.most_common(1)[0][1] if answer_counts else 0
        agreement_ratio = most_common_count / len(answers) if answers else 0

        # Safety: ensure we don't exceed remaining time
        time_per_problem_remaining = remaining_wall / max(problems_remaining, 1)

        if n_done == 2 and len(set(answers)) == 1:
            # EASY PATH: 2/2 agree -> likely correct (P~0.99)
            # Run 2 more for confirmation (4 total), quick timeout
            timeout = min(30, time_per_problem_remaining / 4)
            return (4, timeout, 4 * timeout)  # 4 total attempts

        elif n_done == 2 and len(set(answers)) > 1:
            # HARD PATH: 2/2 disagree -> need more attempts
            # Allocate generously: 12 attempts with longer timeout
            timeout = min(60, time_per_problem_remaining / 12)
            return (12, timeout, 12 * timeout)

        elif n_done >= 4 and agreement_ratio >= 0.75:
            # CONFIRMED EASY: strong agreement after 4 -> stop
            return (n_done, 0, 0)  # signal: no more attempts needed

        elif n_done >= 4 and agreement_ratio < 0.6:
            # CONFIRMED HARD: still split after 4+ -> extend to 12
            timeout = min(60, time_per_problem_remaining / (12 - n_done))
            return (12, timeout, (12 - n_done) * timeout)

        elif n_done >= 8:
            # Ran 8 already: check if we should do 4 more
            if agreement_ratio < 0.65:
                timeout = min(60, time_per_problem_remaining / 4)
                return (12, timeout, 4 * timeout)
            else:
                return (n_done, 0, 0)  # enough agreement, stop

        else:
            # MEDIUM PATH: moderate agreement
            timeout = min(45, time_per_problem_remaining / (8 - n_done))
            return (8, timeout, (8 - n_done) * timeout)

    # Fallback
    return (8, 45, 360)
'''

output = {
    "flat_expected_score": round(flat_sim_score, 2),
    "adaptive_expected_score": round(adapt_sim_score, 2),
    "phase_adaptive_score": round(phase_score, 2),
    "improvement_adaptive_vs_flat": round(adapt_sim_score - flat_sim_score, 2),
    "improvement_phase_vs_flat": round(phase_score - flat_sim_score, 2),
    "flat_wall_time_s": round(flat_sim_wall + SETUP_TIME),
    "adaptive_wall_time_s": round(adapt_sim_wall + SETUP_TIME),
    "phase_wall_time_s": round(phase_compute / PARALLEL_KERNELS + SETUP_TIME),
    "optimal_static_config": {
        "easy_attempts": best_config['n_easy'],
        "easy_early_stop": best_config['es_easy'],
        "medium_attempts": best_config['n_medium'],
        "medium_early_stop": best_config['es_medium'],
        "hard_attempts": best_config['n_hard'],
        "hard_early_stop": best_config['es_hard'],
    },
    "early_detection_p069": {
        "P_correct_given_2_agree": round(p_correct_given_agree_69, 4),
        "P_any_correct_remaining_6": round(p_any_correct_6_69, 4),
    },
    "adaptive_budget_code": adaptive_budget_code.strip()
}

print(json.dumps(output, indent=2))
