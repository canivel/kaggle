"""
AIMO3 Breakthrough Strategy Simulation
=======================================
Simulates 6 novel strategies to find the one with the biggest edge over
the baseline 8-attempt 1/entropy voting approach.

Key parameters (from real competition data):
- p_correct = 0.69 per attempt (empirically measured)
- 8 attempts per problem, 50 problems
- Wrong answers cluster ~30% (systematic misconceptions)
- Correct answers: entropy ~ N(1.5, 0.3)
- Wrong answers: entropy ~ N(2.5, 0.5)
- Code-using attempts: 72% accurate; no-code: 49% accurate
- Python errors: 5% of correct attempts, 25% of wrong attempts

THE CORE OPPORTUNITY: We use ~2% of 9-hour compute budget. 8.8 hours idle.
"""

import numpy as np
from collections import Counter, defaultdict
from math import comb
import json
import time

np.random.seed(42)

# ============================================================
# SIMULATION PARAMETERS (calibrated to real data)
# ============================================================
N_SIMS = 5000           # Monte Carlo simulations per strategy
N_PROBLEMS = 50         # Competition has 50 problems
CORRECT = 42            # Arbitrary correct answer label
N_ATTEMPTS_BASE = 8     # Standard attempt count

# Problem difficulty distribution (calibrated to score 39-44/50)
# 39 easy, 5 medium, 3 hard, 2 very-hard, 1 impossible
DIFFICULTY_MIX = {
    'easy':      {'count': 39, 'p': 0.85},
    'medium':    {'count': 5,  'p': 0.45},
    'hard':      {'count': 3,  'p': 0.25},
    'very_hard': {'count': 2,  'p': 0.10},
    'impossible':{'count': 1,  'p': 0.02},
}

# Wrong answer clustering probability
WRONG_CLUSTER_PROB = 0.30

# Entropy distributions
CORRECT_ENTROPY_MU, CORRECT_ENTROPY_STD = 1.5, 0.3
WRONG_ENTROPY_MU, WRONG_ENTROPY_STD = 2.5, 0.5

# Python usage rates
CORRECT_CODE_RATE = 0.72     # P(used code | correct)
WRONG_CODE_RATE = 0.60       # P(used code | wrong)
CORRECT_ERROR_RATE = 0.05    # P(python error | correct)
WRONG_ERROR_RATE = 0.25      # P(python error | wrong)
CORRECT_CALLS_MU = 3.0       # Mean python calls if correct
WRONG_CALLS_MU = 2.0         # Mean python calls if wrong


def get_problem_difficulties():
    """Return per-problem p_correct for 50 problems."""
    difficulties = []
    for diff_name, info in DIFFICULTY_MIX.items():
        for _ in range(info['count']):
            difficulties.append((diff_name, info['p']))
    np.random.shuffle(difficulties)
    return difficulties


def generate_attempts(n_attempts, p_correct, wrong_cluster_prob=WRONG_CLUSTER_PROB):
    """
    Generate n_attempts for a single problem.
    Returns: list of dicts with answer, entropy, python_calls, python_errors, used_code, is_correct
    """
    # Generate attractor wrong answers for this problem
    n_attractors = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
    attractors = []
    while len(attractors) < n_attractors:
        a = np.random.randint(0, 100000)
        if a != CORRECT:
            attractors.append(a)

    attempts = []
    for _ in range(n_attempts):
        is_correct = np.random.random() < p_correct

        if is_correct:
            answer = CORRECT
            entropy = max(0.1, np.random.normal(CORRECT_ENTROPY_MU, CORRECT_ENTROPY_STD))
            used_code = np.random.random() < CORRECT_CODE_RATE
            python_errors = 1 if np.random.random() < CORRECT_ERROR_RATE else 0
            python_calls = max(0, int(np.random.normal(CORRECT_CALLS_MU, 1.5))) if used_code else 0
        else:
            if np.random.random() < wrong_cluster_prob:
                answer = np.random.choice(attractors)
            else:
                answer = np.random.randint(0, 100000)
            entropy = max(0.1, np.random.normal(WRONG_ENTROPY_MU, WRONG_ENTROPY_STD))
            used_code = np.random.random() < WRONG_CODE_RATE
            python_errors = 1 if np.random.random() < WRONG_ERROR_RATE else 0
            python_calls = max(0, int(np.random.normal(WRONG_CALLS_MU, 1.5))) if used_code else 0

        attempts.append({
            'answer': answer,
            'entropy': entropy,
            'python_calls': python_calls,
            'python_errors': python_errors,
            'used_code': used_code,
            'is_correct': is_correct,
        })

    return attempts, attractors


# ============================================================
# VOTING STRATEGIES
# ============================================================

def baseline_vote(attempts):
    """Current strategy: 1/entropy weighted voting."""
    weights = defaultdict(float)
    for a in attempts:
        if a['answer'] is not None:
            weights[a['answer']] += 1.0 / max(a['entropy'], 1e-9)
    if not weights:
        return 0
    return max(weights, key=weights.get)


def quality_weighted_vote(attempts):
    """Enhanced: 1/entropy * quality signals."""
    weights = defaultdict(float)
    for a in attempts:
        if a['answer'] is None:
            continue
        w = 1.0 / max(a['entropy'], 1e-9)
        if a['python_errors'] > 0:
            w *= 0.3
        if 2 <= a['python_calls'] <= 5:
            w *= 1.3
        elif a['python_calls'] == 0:
            w *= 0.7
        weights[a['answer']] += w
    if not weights:
        return 0
    return max(weights, key=weights.get)


def oracle_vote(attempts):
    """Upper bound: always picks correct if any attempt produced it."""
    for a in attempts:
        if a['is_correct']:
            return CORRECT
    return attempts[0]['answer'] if attempts else 0


# ============================================================
# STRATEGY 1: PROGRAM SYNTHESIS PASS
# ============================================================
# After standard solving, for split-vote problems, run extra attempts
# that ONLY count if the model's Python code produces a matching answer.
# Code-verified answers get massive weight boost.

def simulate_program_synthesis(attempts, p_correct, attractors):
    """
    For split-vote problems (no answer has >= 4 votes), run 4 extra
    "program synthesis" attempts. Each has:
    - Same p_correct as base
    - But we ONLY count the answer if the code executed successfully
      (used_code=True AND python_errors=0)
    - Verified answers get 5x weight

    Key insight: code-verified correct answers have much higher reliability.
    Correct + code + no errors: p_accurate ~ 0.95
    Wrong + code + no errors: p_still_wrong ~ 0.75
    The differential is the edge.
    """
    # Check if base attempts have clear consensus
    counts = Counter(a['answer'] for a in attempts)
    top_answer, top_count = counts.most_common(1)[0]
    if top_count >= 4:
        # Clear consensus, just use baseline
        return baseline_vote(attempts)

    # Split vote - run synthesis pass
    # Simulate 4 extra attempts focused on code verification
    n_extra = 4
    extra_attempts = []
    for _ in range(n_extra):
        is_correct = np.random.random() < p_correct
        if is_correct:
            answer = CORRECT
            entropy = max(0.1, np.random.normal(CORRECT_ENTROPY_MU, CORRECT_ENTROPY_STD))
            used_code = True  # Force code usage in synthesis pass
            python_errors = 1 if np.random.random() < 0.03 else 0  # Lower error rate for targeted code
            python_calls = max(1, int(np.random.normal(3.5, 1.0)))
        else:
            if np.random.random() < WRONG_CLUSTER_PROB:
                answer = np.random.choice(attractors)
            else:
                answer = np.random.randint(0, 100000)
            entropy = max(0.1, np.random.normal(WRONG_ENTROPY_MU, WRONG_ENTROPY_STD))
            used_code = True
            python_errors = 1 if np.random.random() < 0.30 else 0  # Higher error for wrong approach
            python_calls = max(1, int(np.random.normal(2.0, 1.0)))

        extra_attempts.append({
            'answer': answer,
            'entropy': entropy,
            'python_calls': python_calls,
            'python_errors': python_errors,
            'used_code': used_code,
            'is_correct': is_correct,
            'is_synthesis': True,
        })

    # Combine all attempts with synthesis bonus
    all_attempts = list(attempts) + extra_attempts
    weights = defaultdict(float)

    for a in all_attempts:
        if a['answer'] is None:
            continue
        w = 1.0 / max(a['entropy'], 1e-9)

        # Synthesis pass with code verified = massive boost
        if a.get('is_synthesis', False):
            if a['used_code'] and a['python_errors'] == 0:
                w *= 5.0  # 5x weight for code-verified synthesis
            else:
                w *= 0.1  # Discard failed synthesis attempts

        weights[a['answer']] += w

    if not weights:
        return 0
    return max(weights, key=weights.get)


# ============================================================
# STRATEGY 2: ADVERSARIAL FILTERING
# ============================================================
# For each problem, model also predicts common wrong answers.
# Down-weight answers matching predicted-wrong.

def simulate_adversarial_filter(attempts, p_correct, attractors):
    """
    After standard attempts, ask model: "What is the most common wrong answer?"

    Modeling: The model can identify common wrong answers with some probability.
    p_detect_wrong = 0.40 (model correctly identifies an attractor)
    p_false_positive = 0.05 (model incorrectly identifies correct answer as wrong)

    Effect: Down-weight answers matching predicted-wrong by 0.2x
    """
    p_detect_wrong = 0.40
    p_false_positive = 0.05

    # Simulate adversarial prediction
    predicted_wrong = set()
    for attractor in attractors:
        if np.random.random() < p_detect_wrong:
            predicted_wrong.add(attractor)

    # False positive: might flag the correct answer (catastrophic!)
    if np.random.random() < p_false_positive:
        predicted_wrong.add(CORRECT)

    weights = defaultdict(float)
    for a in attempts:
        if a['answer'] is None:
            continue
        w = 1.0 / max(a['entropy'], 1e-9)
        if a['answer'] in predicted_wrong:
            w *= 0.2  # Heavy downweight
        weights[a['answer']] += w

    if not weights:
        return 0
    return max(weights, key=weights.get)


# ============================================================
# STRATEGY 3: CONSENSUS CASCADING
# ============================================================
# Run in waves. Early consensus = done early.
# More attempts for hard problems. Uses idle compute.

def simulate_consensus_cascade(p_correct, attractors):
    """
    Wave 1: 4 attempts. If 3+ agree -> done.
    Wave 2: 4 more (8 total). If 5+ agree -> done.
    Wave 3: 8 more (16 total). If 10+ agree -> done.
    Wave 4: 16 more (32 total). Pick best.

    Time budget: assume each wave takes ~2 min.
    Total: 4 waves * ~2 min = ~8 min (well within 9 hr budget).
    """
    all_attempts = []
    waves = [(4, 3), (4, 5), (8, 10), (16, 18)]  # (new_attempts, consensus_threshold)

    for wave_size, threshold in waves:
        new_attempts, _ = zip(*[
            (generate_attempts(1, p_correct, WRONG_CLUSTER_PROB)[0][0], None)
            for _ in range(wave_size)
        ]) if wave_size > 0 else ([], [])

        # Use the same attractors for consistency
        for _ in range(wave_size):
            att = generate_attempts(1, p_correct, WRONG_CLUSTER_PROB)[0][0]
            # Override attractor consistency
            if not att['is_correct'] and np.random.random() < WRONG_CLUSTER_PROB:
                att['answer'] = np.random.choice(attractors)
            all_attempts.append(att)

        # Check consensus
        counts = Counter(a['answer'] for a in all_attempts)
        top_answer, top_count = counts.most_common(1)[0]
        if top_count >= threshold:
            return baseline_vote(all_attempts)

    # Final: use all attempts (up to 32)
    return baseline_vote(all_attempts)


# ============================================================
# STRATEGY 4: CROSS-PROBLEM LEARNING
# ============================================================
# After first 10 problems, analyze patterns and adjust strategy.
# Simulated as a slight p_correct boost for remaining problems.

# This one is complex to simulate per-problem, so we model the effect:
# - First 10 problems: standard p_correct
# - If we detect weakness pattern (e.g., 2+ of first 10 wrong),
#   we add "Python-verify everything" for remaining 40 problems
# - Boost: +0.03 to p_correct for remaining problems
# But also risk: wrong detection = -0.02 penalty

def simulate_cross_problem_score():
    """Simulates full 50-problem competition with cross-problem learning."""
    difficulties = get_problem_difficulties()
    score = 0

    # Phase 1: first 10 problems (standard)
    phase1_correct = 0
    for i in range(10):
        diff_name, p = difficulties[i]
        attempts, attractors = generate_attempts(N_ATTEMPTS_BASE, p)
        chosen = baseline_vote(attempts)
        if chosen == CORRECT:
            phase1_correct += 1
            score += 1

    # Analyze phase 1
    phase1_wrong = 10 - phase1_correct

    # Phase 2: remaining 40 problems with adjusted strategy
    for i in range(10, 50):
        diff_name, p = difficulties[i]

        # Cross-problem adjustment
        if phase1_wrong >= 3:
            # Detected weakness -> force code verification -> small boost
            p_adjusted = min(0.95, p + 0.03)
        elif phase1_wrong <= 1:
            # Overconfidence risk -> slightly more cautious
            p_adjusted = p - 0.01
        else:
            p_adjusted = p + 0.01

        attempts, attractors = generate_attempts(N_ATTEMPTS_BASE, p_adjusted)
        chosen = baseline_vote(attempts)
        if chosen == CORRECT:
            score += 1

    return score


# ============================================================
# STRATEGY 5: ANSWER SPACE ANALYSIS
# ============================================================
# Analyze problem text to determine answer constraints.
# Filter answers outside expected range.

def simulate_answer_space_filter(attempts, p_correct, attractors):
    """
    For ~30% of problems, we can determine answer constraints from the text:
    - "remainder when divided by 10^5" -> answer 0-99999 (not useful, always true)
    - "m+n in lowest terms where m/n" -> answer likely < 1000
    - "number of arrangements" -> likely > 0
    - "find a+b+c" -> likely < 10000

    Model this as:
    - 30% of problems: we know answer < 1000 (and 70% of wrong answers are >= 1000)
    - Filtering removes 70% of wrong answers but 0% of correct answers
    """
    # 30% chance this problem has exploitable constraints
    has_constraints = np.random.random() < 0.30
    constraint_filters_wrong = 0.70  # P(wrong answer filtered | has constraints)
    constraint_filters_correct = 0.02  # P(correct answer incorrectly filtered)

    if not has_constraints:
        return baseline_vote(attempts)

    # Filter attempts
    filtered = []
    for a in attempts:
        if a['is_correct']:
            # Correct answer almost never filtered
            if np.random.random() > constraint_filters_correct:
                filtered.append(a)
        else:
            # Wrong answers often filtered
            if np.random.random() > constraint_filters_wrong:
                filtered.append(a)
            # else: filtered out

    if not filtered:
        return baseline_vote(attempts)  # Fallback

    return baseline_vote(filtered)


# ============================================================
# STRATEGY 6: EXECUTION-VERIFIED VOTING
# ============================================================
# Only count attempts where Python code produced the answer.
# Attempts that reason without code get 0.2x weight.

def simulate_execution_verified(attempts, p_correct, attractors):
    """
    Key data from failure_analysis.json:
    - Correct + code + no errors: 78% accurate
    - Correct + no code: 49% accurate
    - Wrong + code: 40% accurate (but errors are signal!)
    - Wrong + no code: still wrong

    Give 5x weight to code-verified (used_code=True, python_errors=0).
    Give 0.2x to no-code attempts.
    """
    weights = defaultdict(float)
    for a in attempts:
        if a['answer'] is None:
            continue
        w = 1.0 / max(a['entropy'], 1e-9)

        if a['used_code'] and a['python_errors'] == 0:
            w *= 3.0  # Code verified = 3x boost
        elif a['used_code'] and a['python_errors'] > 0:
            w *= 0.3  # Code with errors = penalty
        else:
            w *= 0.5  # No code = weak evidence

        weights[a['answer']] += w

    if not weights:
        return 0
    return max(weights, key=weights.get)


# ============================================================
# COMBINED STRATEGY: CONSENSUS CASCADE + EXECUTION-VERIFIED + SYNTHESIS
# ============================================================

def simulate_combined(p_correct, attractors):
    """
    THE BIG IDEA: Consensus Cascade with Execution-Verified + Program Synthesis

    Wave 1: 8 standard attempts. Apply execution-verified voting.
            If consensus (5+ code-verified agree) -> done.
    Wave 2: 8 more attempts (16 total). Check again.
            If consensus (8+ agree or 5+ code-verified agree) -> done.
    Wave 3: For still-split problems, run 8 "program synthesis" attempts.
            These ONLY count if code executes and produces answer.
            24 total attempts. Use combined scoring.
    Wave 4: If still split, 8 more standard + 8 more synthesis (40 total).
            Final: quality-weighted vote across all attempts.
    """
    all_attempts = []
    synthesis_attempts = []

    # Wave 1: 8 standard
    base, _ = generate_attempts(8, p_correct, WRONG_CLUSTER_PROB)
    for a in base:
        if not a['is_correct'] and np.random.random() < WRONG_CLUSTER_PROB:
            a['answer'] = np.random.choice(attractors)
    all_attempts.extend(base)

    # Check consensus on code-verified
    code_verified = [a for a in all_attempts if a['used_code'] and a['python_errors'] == 0]
    if code_verified:
        cv_counts = Counter(a['answer'] for a in code_verified)
        if cv_counts and cv_counts.most_common(1)[0][1] >= 4:
            return simulate_execution_verified(all_attempts, p_correct, attractors)

    # Wave 2: 8 more standard
    base2, _ = generate_attempts(8, p_correct, WRONG_CLUSTER_PROB)
    for a in base2:
        if not a['is_correct'] and np.random.random() < WRONG_CLUSTER_PROB:
            a['answer'] = np.random.choice(attractors)
    all_attempts.extend(base2)

    # Check consensus
    counts = Counter(a['answer'] for a in all_attempts)
    if counts.most_common(1)[0][1] >= 10:
        return simulate_execution_verified(all_attempts, p_correct, attractors)

    code_verified = [a for a in all_attempts if a['used_code'] and a['python_errors'] == 0]
    if code_verified:
        cv_counts = Counter(a['answer'] for a in code_verified)
        if cv_counts and cv_counts.most_common(1)[0][1] >= 6:
            return simulate_execution_verified(all_attempts, p_correct, attractors)

    # Wave 3: 8 synthesis attempts (code-only)
    for _ in range(8):
        is_correct = np.random.random() < p_correct
        if is_correct:
            answer = CORRECT
            entropy = max(0.1, np.random.normal(CORRECT_ENTROPY_MU, CORRECT_ENTROPY_STD))
            python_errors = 1 if np.random.random() < 0.03 else 0
            python_calls = max(1, int(np.random.normal(3.5, 1.0)))
        else:
            if np.random.random() < WRONG_CLUSTER_PROB:
                answer = np.random.choice(attractors)
            else:
                answer = np.random.randint(0, 100000)
            entropy = max(0.1, np.random.normal(WRONG_ENTROPY_MU, WRONG_ENTROPY_STD))
            python_errors = 1 if np.random.random() < 0.35 else 0
            python_calls = max(1, int(np.random.normal(2.0, 1.0)))

        sa = {
            'answer': answer,
            'entropy': entropy,
            'python_calls': python_calls,
            'python_errors': python_errors,
            'used_code': True,
            'is_correct': is_correct,
            'is_synthesis': True,
        }
        synthesis_attempts.append(sa)

    # Combined scoring
    combined = all_attempts + synthesis_attempts
    weights = defaultdict(float)

    for a in combined:
        if a['answer'] is None:
            continue
        w = 1.0 / max(a['entropy'], 1e-9)

        # Execution verification
        if a['used_code'] and a['python_errors'] == 0:
            w *= 3.0
        elif a['used_code'] and a['python_errors'] > 0:
            w *= 0.3
        else:
            w *= 0.5

        # Synthesis bonus
        if a.get('is_synthesis', False):
            if a['python_errors'] == 0:
                w *= 2.0  # Additional synthesis bonus
            else:
                w *= 0.05  # Nearly discard failed synthesis

        weights[a['answer']] += w

    if not weights:
        return 0
    return max(weights, key=weights.get)


# ============================================================
# MAIN SIMULATION
# ============================================================

def run_full_simulation():
    print("=" * 80)
    print("AIMO3 BREAKTHROUGH STRATEGY SIMULATION")
    print("=" * 80)
    print(f"\nParameters: N_SIMS={N_SIMS}, N_PROBLEMS={N_PROBLEMS}")
    print(f"Per-attempt accuracy: mixed difficulty (39 easy@0.85, 5 medium@0.45,")
    print(f"  3 hard@0.25, 2 very-hard@0.10, 1 impossible@0.02)")
    print(f"Wrong answer clustering: {WRONG_CLUSTER_PROB:.0%}")
    print(f"Baseline: 8 attempts, 1/entropy voting\n")

    strategies = {
        '0_BASELINE (8 att, 1/entropy)': 'baseline',
        '1_QUALITY_WEIGHTED (8 att)': 'quality',
        '2_PROGRAM_SYNTHESIS (8+4 att)': 'synthesis',
        '3_ADVERSARIAL_FILTER (8 att)': 'adversarial',
        '4_CONSENSUS_CASCADE (4-32 att)': 'cascade',
        '5_CROSS_PROBLEM_LEARNING': 'cross_problem',
        '6_ANSWER_SPACE_FILTER (8 att)': 'answer_space',
        '7_EXECUTION_VERIFIED (8 att)': 'exec_verified',
        '8_COMBINED_CASCADE+SYNTH+EXEC (8-24 att)': 'combined',
        '9_ORACLE (upper bound)': 'oracle',
    }

    results = {}

    for strat_name, strat_key in strategies.items():
        start_time = time.time()
        scores = []

        for sim in range(N_SIMS):
            difficulties = get_problem_difficulties()
            sim_score = 0

            if strat_key == 'cross_problem':
                sim_score = simulate_cross_problem_score()
            else:
                for prob_idx in range(N_PROBLEMS):
                    diff_name, p = difficulties[prob_idx]
                    attempts, attractors = generate_attempts(N_ATTEMPTS_BASE, p)

                    if strat_key == 'baseline':
                        chosen = baseline_vote(attempts)
                    elif strat_key == 'quality':
                        chosen = quality_weighted_vote(attempts)
                    elif strat_key == 'synthesis':
                        chosen = simulate_program_synthesis(attempts, p, attractors)
                    elif strat_key == 'adversarial':
                        chosen = simulate_adversarial_filter(attempts, p, attractors)
                    elif strat_key == 'cascade':
                        chosen = simulate_consensus_cascade(p, attractors)
                    elif strat_key == 'answer_space':
                        chosen = simulate_answer_space_filter(attempts, p, attractors)
                    elif strat_key == 'exec_verified':
                        chosen = simulate_execution_verified(attempts, p, attractors)
                    elif strat_key == 'combined':
                        chosen = simulate_combined(p, attractors)
                    elif strat_key == 'oracle':
                        chosen = oracle_vote(attempts)
                    else:
                        chosen = baseline_vote(attempts)

                    if chosen == CORRECT:
                        sim_score += 1

            scores.append(sim_score)

        elapsed = time.time() - start_time
        arr = np.array(scores)
        results[strat_name] = {
            'mean': float(arr.mean()),
            'std': float(arr.std()),
            'median': float(np.median(arr)),
            'min': int(arr.min()),
            'max': int(arr.max()),
            'p_ge_44': float(np.mean(arr >= 44)),
            'p_ge_45': float(np.mean(arr >= 45)),
            'p_ge_46': float(np.mean(arr >= 46)),
            'p_ge_47': float(np.mean(arr >= 47)),
            'time_sec': round(elapsed, 1),
        }

        print(f"\n{strat_name}:")
        print(f"  Mean: {arr.mean():.2f}/50  Std: {arr.std():.2f}  Median: {np.median(arr):.0f}")
        print(f"  Range: [{arr.min()}, {arr.max()}]")
        print(f"  P(>=44): {np.mean(arr >= 44):.4f}  P(>=45): {np.mean(arr >= 45):.4f}  "
              f"P(>=46): {np.mean(arr >= 46):.4f}  P(>=47): {np.mean(arr >= 47):.4f}")
        print(f"  Time: {elapsed:.1f}s")

    # ============================================================
    # COMPARISON TABLE
    # ============================================================
    print("\n\n" + "=" * 100)
    print("COMPARISON TABLE")
    print("=" * 100)

    baseline_mean = results['0_BASELINE (8 att, 1/entropy)']['mean']

    print(f"\n{'Strategy':<45} {'Mean':>7} {'Std':>5} {'Delta':>7} {'P(>=44)':>8} {'P(>=45)':>8} {'P(>=46)':>8}")
    print("-" * 100)
    for name, r in results.items():
        delta = r['mean'] - baseline_mean
        print(f"{name:<45} {r['mean']:>7.2f} {r['std']:>5.2f} {delta:>+7.2f} "
              f"{r['p_ge_44']:>8.4f} {r['p_ge_45']:>8.4f} {r['p_ge_46']:>8.4f}")

    # ============================================================
    # WINNER ANALYSIS
    # ============================================================
    print("\n\n" + "=" * 80)
    print("WINNER ANALYSIS")
    print("=" * 80)

    # Exclude oracle
    non_oracle = {k: v for k, v in results.items() if 'ORACLE' not in k}
    winner = max(non_oracle, key=lambda k: non_oracle[k]['mean'])
    winner_data = non_oracle[winner]

    print(f"\nBest strategy: {winner}")
    print(f"Expected score: {winner_data['mean']:.2f}/50")
    print(f"Improvement over baseline: +{winner_data['mean'] - baseline_mean:.2f} problems")
    print(f"P(>=44): {winner_data['p_ge_44']:.4f}")
    print(f"P(>=45): {winner_data['p_ge_45']:.4f}")
    print(f"P(>=46): {winner_data['p_ge_46']:.4f}")
    print(f"P(>=47): {winner_data['p_ge_47']:.4f}")

    oracle_data = results['9_ORACLE (upper bound)']
    print(f"\nOracle ceiling: {oracle_data['mean']:.2f}/50")
    print(f"Gap to oracle: {oracle_data['mean'] - winner_data['mean']:.2f} problems")

    return results


if __name__ == '__main__':
    results = run_full_simulation()
