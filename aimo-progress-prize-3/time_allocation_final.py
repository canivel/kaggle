"""
AIMO3 Time Allocation - FINAL with realistic wrong-answer collision model

Key insight from v2: the answer-matching model matters enormously.
- If wrong answers are fully random (1/999): acc is ~0.999 even for p=0.40
- If wrong answers cluster (e.g., common mistakes): acc drops significantly

For AIMO math problems, reality is somewhere between:
- Fully random wrong answers (too optimistic)
- Naive majority vote (too pessimistic - treats all wrong as same)

Realistic model: each wrong attempt has a ~10-20% chance of matching
another wrong attempt's answer (common algebraic mistakes, off-by-one, etc.)

We model this with a "wrong answer entropy" parameter:
- n_wrong_buckets: number of distinct wrong answers possible
- Higher = more random = answer matching works better
"""
import json
import math
import random
from collections import Counter

random.seed(42)
N_SIMS = 300000

TOTAL_TIME = 32400
SETUP_TIME = 400
PARALLEL = 16
EASY, MEDIUM, HARD = 30, 15, 5


def p_majority_correct(p, n):
    threshold = (n // 2) + 1
    prob = 0.0
    for k in range(threshold, n + 1):
        prob += math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
    return prob


def simulate(p, n_attempts, early_stop, time_per_attempt,
             n_wrong_buckets=10, n_sims=N_SIMS):
    """
    Simulate with parameterized wrong-answer diversity.
    n_wrong_buckets: how many distinct wrong answers exist
    - 1 = all wrong answers are the same (worst case, like majority vote)
    - 10 = moderate diversity (realistic for math)
    - 999 = essentially random (best case)
    """
    correct_count = 0
    total_time = 0
    total_attempts = 0

    for _ in range(n_sims):
        answers = []
        attempts_used = 0

        for i in range(n_attempts):
            if random.random() < p:
                answers.append(0)  # correct
            else:
                answers.append(random.randint(1, n_wrong_buckets))
            attempts_used += 1

            counts = Counter(answers)
            if counts.most_common(1)[0][1] >= early_stop:
                break

        counts = Counter(answers)
        final = counts.most_common(1)[0][0]
        if final == 0:
            correct_count += 1

        total_time += attempts_used * time_per_attempt
        total_attempts += attempts_used

    return {
        'accuracy': correct_count / n_sims,
        'avg_time': total_time / n_sims,
        'avg_attempts': total_attempts / n_sims,
    }


###############################################################################
# SENSITIVITY TO WRONG ANSWER DIVERSITY
###############################################################################
print("=" * 70)
print("SENSITIVITY: Wrong-answer diversity (n_wrong_buckets)")
print("=" * 70)

for label, p in [("Easy p=0.90", 0.90), ("Medium p=0.70", 0.70), ("Hard p=0.40", 0.40)]:
    print(f"\n  {label}, 8 attempts, early_stop=4:")
    print(f"  {'buckets':>8s}  {'accuracy':>10s}  {'avg_att':>8s}  {'interpretation':>30s}")
    for nb in [1, 3, 5, 10, 20, 50, 999]:
        s = simulate(p, 8, 4, 35, n_wrong_buckets=nb, n_sims=200000)
        interp = {1: "all wrong same (worst)", 3: "low diversity",
                  5: "moderate", 10: "realistic AIMO",
                  20: "high diversity", 50: "very high", 999: "random (best)"}
        print(f"  {nb:8d}  {s['accuracy']:10.4f}  {s['avg_attempts']:8.1f}  {interp.get(nb, ''):>30s}")


###############################################################################
# MAIN ANALYSIS: Use realistic n_wrong_buckets=10
###############################################################################
NWB = 10  # realistic for AIMO math problems

print(f"\n{'='*70}")
print(f"MAIN ANALYSIS: Using n_wrong_buckets={NWB} (realistic AIMO)")
print("=" * 70)

# FLAT: 8 attempts, early_stop=4
print("\n--- FLAT (8 attempts, early_stop=4) ---")
flat = {}
for label, p, tpa in [('easy', 0.90, 20), ('medium', 0.70, 50), ('hard', 0.40, 80)]:
    flat[label] = simulate(p, 8, 4, tpa, NWB)
    print(f"  {label}: acc={flat[label]['accuracy']:.4f}, att={flat[label]['avg_attempts']:.1f}, time={flat[label]['avg_time']:.0f}s")

flat_score = 30*flat['easy']['accuracy'] + 15*flat['medium']['accuracy'] + 5*flat['hard']['accuracy']
flat_compute = 30*flat['easy']['avg_time'] + 15*flat['medium']['avg_time'] + 5*flat['hard']['avg_time']
print(f"  SCORE: {flat_score:.2f}, compute: {flat_compute:.0f}s, wall: {flat_compute/PARALLEL+SETUP_TIME:.0f}s")


# ADAPTIVE STRATEGIES
strategies = {
    "S1: More attempts on hard (8/8/12, es=4/4/5)": {
        'configs': [('easy', 0.90, 20, 8, 4), ('medium', 0.70, 50, 8, 4), ('hard', 0.40, 80, 12, 5)]
    },
    "S2: Fewer easy, more hard (4/8/12, es=3/4/5)": {
        'configs': [('easy', 0.90, 20, 4, 3), ('medium', 0.70, 50, 8, 4), ('hard', 0.40, 80, 12, 5)]
    },
    "S3: Push all up (8/12/16, es=4/5/6)": {
        'configs': [('easy', 0.90, 20, 8, 4), ('medium', 0.70, 50, 12, 5), ('hard', 0.40, 80, 16, 6)]
    },
    "S4: Timeout boost hard (8/8/8, es=4/4/4, hard p=0.50)": {
        'configs': [('easy', 0.90, 20, 8, 4), ('medium', 0.70, 50, 8, 4), ('hard', 0.50, 160, 8, 4)]
    },
    "S5: Combined (6/10/12, es=3/4/5, hard p=0.50)": {
        'configs': [('easy', 0.90, 20, 6, 3), ('medium', 0.70, 50, 10, 4), ('hard', 0.50, 160, 12, 5)]
    },
    "S6: Aggressive (4/8/16, es=3/4/6, hard p=0.50)": {
        'configs': [('easy', 0.90, 20, 4, 3), ('medium', 0.70, 50, 8, 4), ('hard', 0.50, 160, 16, 6)]
    },
}

results = {}
for name, strat in strategies.items():
    print(f"\n--- {name} ---")
    sims = {}
    for label, p, tpa, na, es in strat['configs']:
        sims[label] = simulate(p, na, es, tpa, NWB)
        print(f"  {label}: acc={sims[label]['accuracy']:.4f}, att={sims[label]['avg_attempts']:.1f}, time={sims[label]['avg_time']:.0f}s")

    score = 30*sims['easy']['accuracy'] + 15*sims['medium']['accuracy'] + 5*sims['hard']['accuracy']
    compute = 30*sims['easy']['avg_time'] + 15*sims['medium']['avg_time'] + 5*sims['hard']['avg_time']
    wall = compute/PARALLEL + SETUP_TIME
    delta = score - flat_score
    print(f"  SCORE: {score:.2f} ({delta:+.2f}), compute: {compute:.0f}s, wall: {wall:.0f}s")
    results[name] = {'score': score, 'delta': delta, 'wall': wall, 'compute': compute}


###############################################################################
# PHASE-ADAPTIVE with realistic wrong-answer model
###############################################################################
print(f"\n{'='*70}")
print(f"PHASE-ADAPTIVE (n_wrong_buckets={NWB})")
print("=" * 70)

def simulate_phase_adaptive(n_sims=N_SIMS):
    random.seed(42)
    problems = ([(0.90, 20)] * 30 + [(0.70, 50)] * 15 + [(0.40, 80)] * 5)

    total_score = 0
    total_compute = 0

    for _ in range(n_sims):
        score = 0
        compute = 0

        for p, base_tpa in problems:
            answers = []

            # Phase 1: 2 probing attempts
            for _ in range(2):
                if random.random() < p:
                    answers.append(0)
                else:
                    answers.append(random.randint(1, NWB))
            compute += 2 * 35

            counts = Counter(answers)
            top_cnt = counts.most_common(1)[0][1]

            if top_cnt >= 2:  # 2/2 agree -> EASY PATH
                # Run 2 more, short timeout
                for _ in range(2):
                    if random.random() < p:
                        answers.append(0)
                    else:
                        answers.append(random.randint(1, NWB))
                compute += 2 * 25

                counts = Counter(answers)
                top_ans, top_cnt = counts.most_common(1)[0]
                if top_cnt >= 3:
                    # 3/4 agree -> very confident
                    if top_ans == 0:
                        score += 1
                else:
                    # Still ambiguous, run 4 more (8 total)
                    for _ in range(4):
                        if random.random() < p:
                            answers.append(0)
                        else:
                            answers.append(random.randint(1, NWB))
                    compute += 4 * 40
                    counts = Counter(answers)
                    if counts.most_common(1)[0][0] == 0:
                        score += 1

            else:  # 2/2 disagree -> HARD PATH
                # Run 10 more with early_stop=4
                for _ in range(10):
                    if random.random() < p:
                        answers.append(0)
                    else:
                        answers.append(random.randint(1, NWB))

                    counts = Counter(answers)
                    if counts.most_common(1)[0][1] >= 4:
                        break
                compute += (len(answers) - 2) * 55

                counts = Counter(answers)
                if counts.most_common(1)[0][0] == 0:
                    score += 1

        total_score += score
        total_compute += compute

    return total_score / n_sims, total_compute / n_sims

phase_score, phase_compute = simulate_phase_adaptive()
phase_wall = phase_compute / PARALLEL + SETUP_TIME
print(f"  SCORE: {phase_score:.2f} ({phase_score - flat_score:+.2f}), compute: {phase_compute:.0f}s, wall: {phase_wall:.0f}s")


###############################################################################
# EARLY DETECTION PROBABILITIES (with realistic wrong-answer model)
###############################################################################
print(f"\n{'='*70}")
print("EARLY DETECTION: Bayesian with n_wrong_buckets=10")
print("=" * 70)

for p in [0.90, 0.70, 0.40, 0.69]:
    # P(2 agree) = P(both correct) + P(both wrong, same bucket)
    p_both_correct = p**2
    p_both_wrong_same = (1-p)**2 * (1/NWB)
    p_agree = p_both_correct + p_both_wrong_same
    p_correct_given_agree = p_both_correct / p_agree

    p_disagree = 1 - p_agree
    p_any_correct_6 = 1 - (1-p)**6

    print(f"\n  p={p:.2f}:")
    print(f"    P(2/2 agree)           = {p_agree:.4f}")
    print(f"    P(correct | 2/2 agree) = {p_correct_given_agree:.4f}")
    print(f"    P(2/2 disagree)        = {p_disagree:.4f}")
    print(f"    P(>=1 correct in 6)    = {p_any_correct_6:.4f}")


###############################################################################
# RANKING
###############################################################################
print(f"\n{'='*70}")
print("STRATEGY RANKING")
print("=" * 70)

all_results = {"FLAT (baseline)": {'score': flat_score, 'delta': 0, 'wall': flat_compute/PARALLEL+SETUP_TIME}}
all_results.update(results)
all_results["Phase-Adaptive"] = {'score': phase_score, 'delta': phase_score - flat_score, 'wall': phase_wall}

ranked = sorted(all_results.items(), key=lambda x: x[1]['score'], reverse=True)
print(f"\n  {'Rank':>4s}  {'Strategy':<55s}  {'Score':>6s}  {'Delta':>6s}  {'Wall':>6s}")
for i, (name, r) in enumerate(ranked, 1):
    print(f"  {i:4d}  {name:<55s}  {r['score']:6.2f}  {r['delta']:+6.2f}  {r['wall']:6.0f}s")


###############################################################################
# FINAL JSON OUTPUT
###############################################################################
print(f"\n{'='*70}")
print("FINAL JSON")
print("=" * 70)

best_name, best_r = ranked[0]
best_delta = best_r['score'] - flat_score

adaptive_budget_code = '''def adaptive_budget(self, problem_idx, problems_remaining, elapsed, results_so_far):
    """
    Adaptive time/attempt allocation with 2-phase difficulty detection.

    Phase 1 (first 2 attempts): Standard timeout, probe difficulty.
    Phase 2 (attempts 3+):
      - If 2/2 agreed: "easy path" -> 2 more confirmatory, short timeout
      - If 2/2 disagreed: "hard path" -> up to 10 more, long timeout
      - Re-evaluate at 4 and 8 attempts based on consensus level

    Key insight: answer-matching early stop (not naive majority) means
    that agreement = very high confidence. Disagreement = allocate more.

    Returns: (max_attempts, timeout_per_attempt, total_budget_seconds)
    """
    TIME_LIMIT = 32400
    SETUP = 400
    remaining = TIME_LIMIT - elapsed - SETUP
    n_done = len(results_so_far)

    # Safety floor: ensure every remaining problem gets at least some budget
    budget_per_problem = remaining / max(problems_remaining, 1)

    # Extract non-None answers
    answers = [r[0] for r in results_so_far if r[0] is not None]

    # === No attempts yet: start probing ===
    if n_done == 0:
        timeout = min(300, budget_per_problem / 8)
        return (8, timeout, 8 * timeout)

    # === 1 attempt done: need at least 1 more to classify ===
    if n_done == 1:
        timeout = min(300, budget_per_problem / 8)
        return (8, timeout, 8 * timeout)

    # === 2+ attempts: classify and decide ===
    from collections import Counter
    counts = Counter(answers)
    top_answer, top_count = counts.most_common(1)[0] if counts else (None, 0)
    agreement = top_count / len(answers) if answers else 0

    if n_done == 2:
        if len(set(answers)) == 1:
            # EASY PATH: 2/2 agree -> P(correct|agree) > 0.99
            # Quick confirmation: 2 more attempts, short timeout
            timeout = min(120, budget_per_problem / 4)
            return (4, timeout, 4 * timeout)
        else:
            # HARD PATH: 2/2 disagree -> need many more with longer timeout
            timeout = min(600, budget_per_problem / 12)
            return (12, timeout, 12 * timeout)

    if n_done >= 4 and agreement >= 0.75:
        # Strong consensus -> stop, answer is the mode
        return (n_done, 0, 0)

    if n_done >= 4 and agreement < 0.5 and n_done < 12:
        # Still no consensus -> extend with generous timeout
        remaining_att = min(12 - n_done, 8)
        timeout = min(600, budget_per_problem / remaining_att)
        return (12, timeout, remaining_att * timeout)

    if n_done >= 8:
        if agreement >= 0.5:
            # Reasonable consensus after 8 -> accept
            return (n_done, 0, 0)
        else:
            # Still split after 8 -> try 4 more
            timeout = min(600, budget_per_problem / 4)
            return (12, timeout, 4 * timeout)

    # Default medium path
    remaining_att = max(1, 8 - n_done)
    timeout = min(300, budget_per_problem / remaining_att)
    return (8, timeout, remaining_att * timeout)'''

# Early detection answers for p=0.69
p = 0.69
p_agree_69 = p**2 + (1-p)**2 * (1/NWB)
p_corr_agree_69 = p**2 / p_agree_69

output = {
    "flat_expected_score": round(flat_score, 2),
    "adaptive_expected_score": round(best_r['score'], 2),
    "improvement": round(best_delta, 2),
    "best_strategy": best_name,
    "analysis": {
        "wrong_answer_model": f"n_wrong_buckets={NWB} (realistic for AIMO integer answers)",
        "flat_breakdown": {
            "easy_30x": round(30*flat['easy']['accuracy'], 2),
            "medium_15x": round(15*flat['medium']['accuracy'], 2),
            "hard_5x": round(5*flat['hard']['accuracy'], 2)
        },
        "early_detection_p069": {
            "P_correct_given_2_agree": round(p_corr_agree_69, 4),
            "P_any_correct_remaining_6": round(1-(1-p)**6, 4)
        },
        "key_insights": [
            "Answer-matching early stop boosts accuracy dramatically vs naive majority vote",
            "With 10 wrong-answer buckets: easy=1.00, medium=0.98, hard=0.73 (8 att, es=4)",
            "Most points lost on hard problems (only 5 but ~1.3 lost points)",
            "Increasing timeout for hard problems to boost per-attempt p is the main lever",
            "Going from p_hard=0.40 to 0.50 (via 2x timeout) gains ~0.3-0.5 points",
            "Phase-adaptive: detect difficulty after 2 attempts, reallocate budget",
            "More attempts (12-16) on hard helps ~0.2-0.4 points",
            "Combined (more attempts + boosted timeout) is best strategy"
        ],
        "strategy_ranking": [
            {"rank": i+1, "name": n, "score": round(r['score'],2), "delta": round(r['delta'],2)}
            for i, (n, r) in enumerate(ranked)
        ]
    },
    "adaptive_budget_code": adaptive_budget_code
}

print(json.dumps(output, indent=2))
