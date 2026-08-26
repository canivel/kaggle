"""
AIMO3 Breakthrough Strategy Simulation (FAST vectorized version)
================================================================
Simulates 6+ novel strategies to find the biggest edge.
All strategies are vectorized with NumPy for speed.
"""

import numpy as np
from collections import Counter, defaultdict
import time

np.random.seed(42)

# ============================================================
# PARAMETERS (calibrated from real data)
# ============================================================
N_SIMS = 3000
N_PROBLEMS = 50
CORRECT = 42

# Difficulty distribution
DIFFICULTIES = (
    [0.85] * 39 +   # easy
    [0.45] * 5  +   # medium
    [0.25] * 3  +   # hard
    [0.10] * 2  +   # very hard
    [0.02] * 1      # impossible
)

WRONG_CLUSTER_PROB = 0.30

# Entropy params
CORRECT_ENT_MU, CORRECT_ENT_STD = 1.5, 0.3
WRONG_ENT_MU, WRONG_ENT_STD = 2.5, 0.5

# Code usage
CORRECT_CODE_RATE = 0.72
WRONG_CODE_RATE = 0.60
CORRECT_ERROR_RATE = 0.05
WRONG_ERROR_RATE = 0.25


def generate_problem_batch(n_sims, n_problems, n_attempts, difficulties):
    """
    Generate all attempts for all problems for all simulations at once.
    Returns arrays of shape (n_sims, n_problems, n_attempts).
    """
    total = n_sims * n_problems * n_attempts

    # Per-problem p_correct - tile across sims and attempts
    p_arr = np.array(difficulties)  # (n_problems,)
    p_tiled = np.broadcast_to(p_arr[None, :, None], (n_sims, n_problems, n_attempts))

    # Which attempts are correct?
    is_correct = np.random.random((n_sims, n_problems, n_attempts)) < p_tiled

    # Answers
    answers = np.random.randint(0, 100000, size=(n_sims, n_problems, n_attempts))
    answers[is_correct] = CORRECT

    # Wrong answer clustering: for each problem, pick 1-2 attractor wrong answers
    # ~30% of wrong answers copy the attractor
    attractors = np.random.randint(0, 100000, size=(n_sims, n_problems))
    # Make sure attractor != CORRECT
    attractors[attractors == CORRECT] = CORRECT + 1

    # Apply clustering to wrong answers
    wrong_mask = ~is_correct
    cluster_mask = wrong_mask & (np.random.random((n_sims, n_problems, n_attempts)) < WRONG_CLUSTER_PROB)

    # Replace clustered wrong answers with attractor
    attractor_tiled = np.broadcast_to(attractors[:, :, None], (n_sims, n_problems, n_attempts))
    answers[cluster_mask] = attractor_tiled[cluster_mask]

    # Entropies
    entropies = np.where(
        is_correct,
        np.maximum(0.1, np.random.normal(CORRECT_ENT_MU, CORRECT_ENT_STD, (n_sims, n_problems, n_attempts))),
        np.maximum(0.1, np.random.normal(WRONG_ENT_MU, WRONG_ENT_STD, (n_sims, n_problems, n_attempts)))
    )

    # Code usage
    used_code = np.where(
        is_correct,
        np.random.random((n_sims, n_problems, n_attempts)) < CORRECT_CODE_RATE,
        np.random.random((n_sims, n_problems, n_attempts)) < WRONG_CODE_RATE
    )

    # Python errors
    python_errors = np.where(
        is_correct,
        (np.random.random((n_sims, n_problems, n_attempts)) < CORRECT_ERROR_RATE).astype(int),
        (np.random.random((n_sims, n_problems, n_attempts)) < WRONG_ERROR_RATE).astype(int)
    )

    # Python calls (0 if no code)
    python_calls = np.where(
        used_code,
        np.maximum(0, np.random.normal(3.0, 1.5, (n_sims, n_problems, n_attempts)).astype(int)),
        0
    )

    return {
        'answers': answers,
        'entropies': entropies,
        'is_correct': is_correct,
        'used_code': used_code,
        'python_errors': python_errors,
        'python_calls': python_calls,
        'attractors': attractors,
    }


def vote_per_problem(answers, weights):
    """
    For a single problem (one sim), pick the answer with highest total weight.
    answers: (n_attempts,) int
    weights: (n_attempts,) float
    Returns: chosen answer
    """
    w = defaultdict(float)
    for a, wt in zip(answers, weights):
        w[a] += wt
    return max(w, key=w.get)


def run_strategy_vectorized(data, strategy_fn, n_sims, n_problems, label=""):
    """
    Run a voting strategy across all sims/problems.
    strategy_fn(answers, entropies, used_code, python_errors, python_calls) -> weights
    Returns: scores array of shape (n_sims,)
    """
    answers = data['answers']
    entropies = data['entropies']
    used_code = data['used_code']
    python_errors = data['python_errors']
    python_calls = data['python_calls']

    scores = np.zeros(n_sims, dtype=int)

    for sim in range(n_sims):
        for prob in range(n_problems):
            a = answers[sim, prob]
            e = entropies[sim, prob]
            uc = used_code[sim, prob]
            pe = python_errors[sim, prob]
            pc = python_calls[sim, prob]

            w = strategy_fn(a, e, uc, pe, pc)
            chosen = vote_per_problem(a, w)

            if chosen == CORRECT:
                scores[sim] += 1

    return scores


# ============================================================
# STRATEGY WEIGHT FUNCTIONS
# ============================================================

def weights_baseline(answers, entropies, used_code, python_errors, python_calls):
    """Standard 1/entropy voting."""
    return 1.0 / np.maximum(entropies, 1e-9)


def weights_quality(answers, entropies, used_code, python_errors, python_calls):
    """Quality-weighted: 1/entropy * quality signals."""
    w = 1.0 / np.maximum(entropies, 1e-9)
    # Python error penalty
    w = np.where(python_errors > 0, w * 0.3, w)
    # Code usage bonus
    w = np.where((python_calls >= 2) & (python_calls <= 5), w * 1.3, w)
    w = np.where(python_calls == 0, w * 0.7, w)
    return w


def weights_exec_verified(answers, entropies, used_code, python_errors, python_calls):
    """Execution-verified: massive boost for code-verified answers."""
    w = 1.0 / np.maximum(entropies, 1e-9)
    # Code verified = 3x
    code_verified = used_code & (python_errors == 0)
    code_errored = used_code & (python_errors > 0)
    no_code = ~used_code

    w = np.where(code_verified, w * 3.0, w)
    w = np.where(code_errored, w * 0.3, w)
    w = np.where(no_code, w * 0.5, w)
    return w


def weights_exec_verified_strong(answers, entropies, used_code, python_errors, python_calls):
    """Even stronger execution verification: 10x for code-verified."""
    w = 1.0 / np.maximum(entropies, 1e-9)
    code_verified = used_code & (python_errors == 0)
    code_errored = used_code & (python_errors > 0)
    no_code = ~used_code

    w = np.where(code_verified, w * 10.0, w)
    w = np.where(code_errored, w * 0.1, w)
    w = np.where(no_code, w * 0.2, w)
    return w


def weights_combined_quality_exec(answers, entropies, used_code, python_errors, python_calls):
    """Combined quality + execution verification."""
    w = 1.0 / np.maximum(entropies, 1e-9)

    code_verified = used_code & (python_errors == 0)
    code_errored = used_code & (python_errors > 0)
    no_code = ~used_code

    # Execution verification
    w = np.where(code_verified, w * 5.0, w)
    w = np.where(code_errored, w * 0.2, w)
    w = np.where(no_code, w * 0.4, w)

    # Code usage sweet spot bonus
    w = np.where((python_calls >= 2) & (python_calls <= 5), w * 1.3, w)

    return w


# ============================================================
# STRATEGIES REQUIRING EXTRA ATTEMPTS (cascade, synthesis)
# ============================================================

def simulate_cascade_problem(p_correct, attractor):
    """Simulate consensus cascade for one problem."""
    waves = [
        (4, 3),   # 4 attempts, need 3 for consensus
        (4, 5),   # +4 (8 total), need 5
        (8, 8),   # +8 (16 total), need 8
        (16, 12), # +16 (32 total), need 12
    ]

    all_answers = []
    all_entropies = []
    all_used_code = []
    all_py_errors = []

    for wave_size, threshold in waves:
        for _ in range(wave_size):
            is_correct = np.random.random() < p_correct
            if is_correct:
                ans = CORRECT
                ent = max(0.1, np.random.normal(CORRECT_ENT_MU, CORRECT_ENT_STD))
                uc = np.random.random() < CORRECT_CODE_RATE
                pe = 1 if np.random.random() < CORRECT_ERROR_RATE else 0
            else:
                if np.random.random() < WRONG_CLUSTER_PROB:
                    ans = attractor
                else:
                    ans = np.random.randint(0, 100000)
                ent = max(0.1, np.random.normal(WRONG_ENT_MU, WRONG_ENT_STD))
                uc = np.random.random() < WRONG_CODE_RATE
                pe = 1 if np.random.random() < WRONG_ERROR_RATE else 0

            all_answers.append(ans)
            all_entropies.append(ent)
            all_used_code.append(uc)
            all_py_errors.append(pe)

        # Check consensus
        c = Counter(all_answers)
        top, top_count = c.most_common(1)[0]
        if top_count >= threshold:
            # Use execution-verified voting on all collected
            a = np.array(all_answers)
            e = np.array(all_entropies)
            uc_arr = np.array(all_used_code)
            pe_arr = np.array(all_py_errors)
            pc_arr = np.zeros_like(pe_arr)  # simplified
            w = weights_exec_verified(a, e, uc_arr, pe_arr, pc_arr)
            return vote_per_problem(a, w)

    # No consensus after all waves: use all with exec-verified
    a = np.array(all_answers)
    e = np.array(all_entropies)
    uc_arr = np.array(all_used_code)
    pe_arr = np.array(all_py_errors)
    pc_arr = np.zeros_like(pe_arr)
    w = weights_exec_verified(a, e, uc_arr, pe_arr, pc_arr)
    return vote_per_problem(a, w)


def simulate_synthesis_problem(base_answers, base_entropies, base_used_code,
                                base_py_errors, base_py_calls, p_correct, attractor):
    """
    After standard 8 attempts, for split-vote problems, run 4-8 extra
    "synthesis" attempts that only count if code executes cleanly.
    """
    # Check if base has consensus
    c = Counter(base_answers)
    top, top_count = c.most_common(1)[0]
    if top_count >= 4:
        # Clear consensus, use baseline
        w = weights_baseline(np.array(base_answers), np.array(base_entropies),
                            np.array(base_used_code), np.array(base_py_errors), np.array(base_py_calls))
        return vote_per_problem(np.array(base_answers), w)

    # Split vote -> run synthesis pass
    n_extra = 8  # More extra attempts for split problems

    synth_answers = []
    synth_entropies = []
    synth_weights = []

    for _ in range(n_extra):
        is_correct = np.random.random() < p_correct
        if is_correct:
            ans = CORRECT
            ent = max(0.1, np.random.normal(CORRECT_ENT_MU, CORRECT_ENT_STD))
            pe = 1 if np.random.random() < 0.03 else 0  # Lower error for focused code
        else:
            if np.random.random() < WRONG_CLUSTER_PROB:
                ans = attractor
            else:
                ans = np.random.randint(0, 100000)
            ent = max(0.1, np.random.normal(WRONG_ENT_MU, WRONG_ENT_STD))
            pe = 1 if np.random.random() < 0.35 else 0  # Higher error for wrong code

        if pe == 0:  # Only count if code succeeded
            synth_answers.append(ans)
            synth_entropies.append(ent)
            synth_weights.append(5.0 / max(ent, 1e-9))  # 5x weight

    # Combine base + synthesis
    all_answers = list(base_answers) + synth_answers
    base_w = 1.0 / np.maximum(np.array(base_entropies), 1e-9)
    all_weights = list(base_w) + synth_weights

    return vote_per_problem(np.array(all_answers), np.array(all_weights))


def simulate_cascade_plus_synthesis_problem(p_correct, attractor):
    """
    THE COMBINED STRATEGY:
    Wave 1: 8 standard attempts (exec-verified voting).
            If 5+ code-verified agree -> done.
    Wave 2: 8 more standard (16 total). Check consensus.
    Wave 3: 8 synthesis-only (code must run). Verified answers get 5x weight.
    Wave 4: If still split, 8 more synthesis (32+16=48 total attempts).
    """
    all_ans = []
    all_ent = []
    all_uc = []
    all_pe = []
    synth_indices = set()

    def gen_attempt(is_synth=False):
        is_correct = np.random.random() < p_correct
        if is_correct:
            ans = CORRECT
            ent = max(0.1, np.random.normal(CORRECT_ENT_MU, CORRECT_ENT_STD))
            uc = True if is_synth else (np.random.random() < CORRECT_CODE_RATE)
            pe = 1 if np.random.random() < (0.03 if is_synth else CORRECT_ERROR_RATE) else 0
        else:
            if np.random.random() < WRONG_CLUSTER_PROB:
                ans = attractor
            else:
                ans = np.random.randint(0, 100000)
            ent = max(0.1, np.random.normal(WRONG_ENT_MU, WRONG_ENT_STD))
            uc = True if is_synth else (np.random.random() < WRONG_CODE_RATE)
            pe = 1 if np.random.random() < (0.35 if is_synth else WRONG_ERROR_RATE) else 0
        return ans, ent, uc, pe

    def add_attempts(n, is_synth=False):
        for _ in range(n):
            idx = len(all_ans)
            ans, ent, uc, pe = gen_attempt(is_synth)
            all_ans.append(ans)
            all_ent.append(ent)
            all_uc.append(uc)
            all_pe.append(pe)
            if is_synth:
                synth_indices.add(idx)

    def compute_and_vote():
        w = np.zeros(len(all_ans))
        for i in range(len(all_ans)):
            base_w = 1.0 / max(all_ent[i], 1e-9)
            if all_uc[i] and all_pe[i] == 0:
                base_w *= 5.0 if i in synth_indices else 3.0
            elif all_uc[i] and all_pe[i] > 0:
                base_w *= 0.05 if i in synth_indices else 0.3
            elif not all_uc[i]:
                base_w *= 0.5
            w[i] = base_w
        return vote_per_problem(np.array(all_ans), w)

    def check_code_consensus(threshold):
        cv = [all_ans[i] for i in range(len(all_ans)) if all_uc[i] and all_pe[i] == 0]
        if cv:
            c = Counter(cv)
            if c.most_common(1)[0][1] >= threshold:
                return True
        return False

    # Wave 1: 8 standard
    add_attempts(8, is_synth=False)
    if check_code_consensus(4):
        return compute_and_vote()

    # Wave 2: 8 more standard
    add_attempts(8, is_synth=False)
    c = Counter(all_ans)
    if c.most_common(1)[0][1] >= 10:
        return compute_and_vote()
    if check_code_consensus(6):
        return compute_and_vote()

    # Wave 3: 8 synthesis
    add_attempts(8, is_synth=True)
    if check_code_consensus(8):
        return compute_and_vote()

    # Wave 4: 8 more synthesis
    add_attempts(8, is_synth=True)

    return compute_and_vote()


# ============================================================
# MAIN SIMULATION
# ============================================================

def print_results(name, scores, baseline_mean=None):
    arr = np.array(scores)
    delta = f"+{arr.mean() - baseline_mean:.2f}" if baseline_mean is not None else "---"
    print(f"\n{name}:")
    print(f"  Mean: {arr.mean():.2f}/50  Std: {arr.std():.2f}  Median: {np.median(arr):.0f}")
    print(f"  Delta: {delta}  Range: [{arr.min()}, {arr.max()}]")
    print(f"  P(>=44): {np.mean(arr >= 44):.4f}  P(>=45): {np.mean(arr >= 45):.4f}  "
          f"P(>=46): {np.mean(arr >= 46):.4f}  P(>=47): {np.mean(arr >= 47):.4f}")
    return {
        'mean': float(arr.mean()),
        'std': float(arr.std()),
        'p44': float(np.mean(arr >= 44)),
        'p45': float(np.mean(arr >= 45)),
        'p46': float(np.mean(arr >= 46)),
        'p47': float(np.mean(arr >= 47)),
    }


def main():
    print("=" * 90)
    print("AIMO3 BREAKTHROUGH STRATEGY SIMULATION (FAST)")
    print("=" * 90)
    print(f"N_SIMS={N_SIMS}, N_PROBLEMS={N_PROBLEMS}, WRONG_CLUSTER={WRONG_CLUSTER_PROB:.0%}")
    print(f"Difficulty: 39@0.85, 5@0.45, 3@0.25, 2@0.10, 1@0.02")
    print(f"Mean per-attempt p ~ 0.69")

    difficulties = DIFFICULTIES

    # ============================================================
    # PHASE 1: Weight-only strategies (8 attempts, vary weights)
    # These are fast - vectorizable
    # ============================================================
    print("\n" + "=" * 90)
    print("PHASE 1: WEIGHT-ONLY STRATEGIES (8 attempts)")
    print("=" * 90)

    t0 = time.time()
    data = generate_problem_batch(N_SIMS, N_PROBLEMS, 8, difficulties)
    print(f"Data generated in {time.time()-t0:.1f}s")

    weight_strategies = {
        'BASELINE (1/entropy)': weights_baseline,
        'QUALITY_WEIGHTED': weights_quality,
        'EXEC_VERIFIED (3x code)': weights_exec_verified,
        'EXEC_VERIFIED_STRONG (10x code)': weights_exec_verified_strong,
        'COMBINED_QUALITY_EXEC': weights_combined_quality_exec,
    }

    all_results = {}
    baseline_mean = None

    for name, wfn in weight_strategies.items():
        t0 = time.time()
        scores = run_strategy_vectorized(data, wfn, N_SIMS, N_PROBLEMS, name)
        elapsed = time.time() - t0
        r = print_results(name, scores, baseline_mean)
        r['time'] = elapsed
        all_results[name] = r
        if baseline_mean is None:
            baseline_mean = r['mean']
        print(f"  Time: {elapsed:.1f}s")

    # Oracle
    t0 = time.time()
    oracle_scores = np.zeros(N_SIMS, dtype=int)
    for sim in range(N_SIMS):
        for prob in range(N_PROBLEMS):
            if CORRECT in data['answers'][sim, prob]:
                oracle_scores[sim] += 1
    elapsed = time.time() - t0
    r = print_results('ORACLE (upper bound)', oracle_scores, baseline_mean)
    r['time'] = elapsed
    all_results['ORACLE'] = r
    print(f"  Time: {elapsed:.1f}s")

    # ============================================================
    # PHASE 2: Extra-attempt strategies (need per-problem loops)
    # ============================================================
    print("\n" + "=" * 90)
    print("PHASE 2: EXTRA-ATTEMPT STRATEGIES (variable attempts)")
    print("=" * 90)

    # Strategy: Consensus Cascade (up to 32 attempts)
    t0 = time.time()
    cascade_scores = np.zeros(N_SIMS, dtype=int)
    for sim in range(N_SIMS):
        np.random.shuffle(difficulties)
        for prob_idx in range(N_PROBLEMS):
            p = difficulties[prob_idx]
            attractor = np.random.randint(0, 100000)
            if attractor == CORRECT:
                attractor = CORRECT + 1
            chosen = simulate_cascade_problem(p, attractor)
            if chosen == CORRECT:
                cascade_scores[sim] += 1
    elapsed = time.time() - t0
    r = print_results('CONSENSUS_CASCADE (up to 32 att)', cascade_scores, baseline_mean)
    r['time'] = elapsed
    all_results['CONSENSUS_CASCADE'] = r
    print(f"  Time: {elapsed:.1f}s")

    # Strategy: Program Synthesis (8 base + 8 synthesis for split)
    t0 = time.time()
    synth_scores = np.zeros(N_SIMS, dtype=int)
    for sim in range(N_SIMS):
        np.random.shuffle(difficulties)
        for prob_idx in range(N_PROBLEMS):
            p = difficulties[prob_idx]
            # Generate base attempts
            base_ans = []
            base_ent = []
            base_uc = []
            base_pe = []
            base_pc = []
            attractor = np.random.randint(0, 100000)
            if attractor == CORRECT:
                attractor = CORRECT + 1

            for _ in range(8):
                is_correct = np.random.random() < p
                if is_correct:
                    ans = CORRECT
                    ent = max(0.1, np.random.normal(CORRECT_ENT_MU, CORRECT_ENT_STD))
                    uc = np.random.random() < CORRECT_CODE_RATE
                    pe = 1 if np.random.random() < CORRECT_ERROR_RATE else 0
                    pc = max(0, int(np.random.normal(3.0, 1.5))) if uc else 0
                else:
                    if np.random.random() < WRONG_CLUSTER_PROB:
                        ans = attractor
                    else:
                        ans = np.random.randint(0, 100000)
                    ent = max(0.1, np.random.normal(WRONG_ENT_MU, WRONG_ENT_STD))
                    uc = np.random.random() < WRONG_CODE_RATE
                    pe = 1 if np.random.random() < WRONG_ERROR_RATE else 0
                    pc = max(0, int(np.random.normal(2.0, 1.5))) if uc else 0

                base_ans.append(ans)
                base_ent.append(ent)
                base_uc.append(uc)
                base_pe.append(pe)
                base_pc.append(pc)

            chosen = simulate_synthesis_problem(
                base_ans, base_ent, base_uc, base_pe, base_pc, p, attractor
            )
            if chosen == CORRECT:
                synth_scores[sim] += 1
    elapsed = time.time() - t0
    r = print_results('PROGRAM_SYNTHESIS (8+8 for split)', synth_scores, baseline_mean)
    r['time'] = elapsed
    all_results['PROGRAM_SYNTHESIS'] = r
    print(f"  Time: {elapsed:.1f}s")

    # Strategy: Combined Cascade + Synthesis + Exec Verification
    t0 = time.time()
    combined_scores = np.zeros(N_SIMS, dtype=int)
    for sim in range(N_SIMS):
        np.random.shuffle(difficulties)
        for prob_idx in range(N_PROBLEMS):
            p = difficulties[prob_idx]
            attractor = np.random.randint(0, 100000)
            if attractor == CORRECT:
                attractor = CORRECT + 1
            chosen = simulate_cascade_plus_synthesis_problem(p, attractor)
            if chosen == CORRECT:
                combined_scores[sim] += 1
    elapsed = time.time() - t0
    r = print_results('COMBINED_CASCADE+SYNTH+EXEC (up to 32 att)', combined_scores, baseline_mean)
    r['time'] = elapsed
    all_results['COMBINED_CASCADE_SYNTH_EXEC'] = r
    print(f"  Time: {elapsed:.1f}s")

    # ============================================================
    # PHASE 3: Adversarial Filter simulation
    # ============================================================
    print("\n" + "=" * 90)
    print("PHASE 3: ADVERSARIAL FILTER")
    print("=" * 90)

    t0 = time.time()
    adv_scores = np.zeros(N_SIMS, dtype=int)

    for sim in range(N_SIMS):
        np.random.shuffle(difficulties)
        for prob_idx in range(N_PROBLEMS):
            p = difficulties[prob_idx]
            attractor = np.random.randint(0, 100000)
            if attractor == CORRECT:
                attractor = CORRECT + 1

            # Generate 8 attempts
            answers = []
            entropies = []
            for _ in range(8):
                is_correct = np.random.random() < p
                if is_correct:
                    ans = CORRECT
                    ent = max(0.1, np.random.normal(CORRECT_ENT_MU, CORRECT_ENT_STD))
                else:
                    if np.random.random() < WRONG_CLUSTER_PROB:
                        ans = attractor
                    else:
                        ans = np.random.randint(0, 100000)
                    ent = max(0.1, np.random.normal(WRONG_ENT_MU, WRONG_ENT_STD))
                answers.append(ans)
                entropies.append(ent)

            # Adversarial: predict common wrong answers
            p_detect = 0.40
            p_false_positive = 0.05  # KEY RISK

            predicted_wrong = set()
            if np.random.random() < p_detect:
                predicted_wrong.add(attractor)
            if np.random.random() < p_false_positive:
                predicted_wrong.add(CORRECT)  # CATASTROPHIC

            weights = {}
            for a, e in zip(answers, entropies):
                w = 1.0 / max(e, 1e-9)
                if a in predicted_wrong:
                    w *= 0.2
                weights[a] = weights.get(a, 0) + w

            chosen = max(weights, key=weights.get)
            if chosen == CORRECT:
                adv_scores[sim] += 1

    elapsed = time.time() - t0
    r = print_results('ADVERSARIAL_FILTER', adv_scores, baseline_mean)
    r['time'] = elapsed
    all_results['ADVERSARIAL_FILTER'] = r
    print(f"  Time: {elapsed:.1f}s")

    # ============================================================
    # PHASE 4: Answer Space Analysis
    # ============================================================
    print("\n" + "=" * 90)
    print("PHASE 4: ANSWER SPACE ANALYSIS")
    print("=" * 90)

    t0 = time.time()
    ans_space_scores = np.zeros(N_SIMS, dtype=int)

    for sim in range(N_SIMS):
        np.random.shuffle(difficulties)
        for prob_idx in range(N_PROBLEMS):
            p = difficulties[prob_idx]
            attractor = np.random.randint(0, 100000)
            if attractor == CORRECT:
                attractor = CORRECT + 1

            answers = []
            entropies = []
            for _ in range(8):
                is_correct = np.random.random() < p
                if is_correct:
                    ans = CORRECT
                    ent = max(0.1, np.random.normal(CORRECT_ENT_MU, CORRECT_ENT_STD))
                else:
                    if np.random.random() < WRONG_CLUSTER_PROB:
                        ans = attractor
                    else:
                        ans = np.random.randint(0, 100000)
                    ent = max(0.1, np.random.normal(WRONG_ENT_MU, WRONG_ENT_STD))
                answers.append(ans)
                entropies.append(ent)

            # 30% chance we can determine useful constraint
            has_constraint = np.random.random() < 0.30
            if has_constraint:
                # Filter: remove answers > 1000 (70% of wrong answers are > 1000)
                filtered_a = []
                filtered_e = []
                for a, e in zip(answers, entropies):
                    # Correct answer is always in range
                    if a == CORRECT:
                        if np.random.random() > 0.02:  # 2% false filter
                            filtered_a.append(a)
                            filtered_e.append(e)
                    else:
                        # 70% of wrong answers are out of range
                        if np.random.random() > 0.70:
                            filtered_a.append(a)
                            filtered_e.append(e)

                if not filtered_a:
                    filtered_a = answers
                    filtered_e = entropies

                weights = {}
                for a, e in zip(filtered_a, filtered_e):
                    weights[a] = weights.get(a, 0) + 1.0 / max(e, 1e-9)
                chosen = max(weights, key=weights.get)
            else:
                weights = {}
                for a, e in zip(answers, entropies):
                    weights[a] = weights.get(a, 0) + 1.0 / max(e, 1e-9)
                chosen = max(weights, key=weights.get)

            if chosen == CORRECT:
                ans_space_scores[sim] += 1

    elapsed = time.time() - t0
    r = print_results('ANSWER_SPACE_FILTER', ans_space_scores, baseline_mean)
    r['time'] = elapsed
    all_results['ANSWER_SPACE_FILTER'] = r
    print(f"  Time: {elapsed:.1f}s")

    # ============================================================
    # PHASE 5: Cross-Problem Learning
    # ============================================================
    print("\n" + "=" * 90)
    print("PHASE 5: CROSS-PROBLEM LEARNING")
    print("=" * 90)

    t0 = time.time()
    cross_scores = np.zeros(N_SIMS, dtype=int)

    for sim in range(N_SIMS):
        np.random.shuffle(difficulties)
        score = 0
        phase1_correct = 0

        for prob_idx in range(N_PROBLEMS):
            p = difficulties[prob_idx]

            # After first 10, apply learning
            if prob_idx >= 10:
                phase1_wrong = 10 - phase1_correct
                if phase1_wrong >= 3:
                    p_adj = min(0.95, p + 0.03)
                elif phase1_wrong <= 1:
                    p_adj = max(0.01, p - 0.01)
                else:
                    p_adj = min(0.95, p + 0.01)
            else:
                p_adj = p

            # Generate attempts with adjusted p
            attractor = np.random.randint(0, 100000)
            if attractor == CORRECT:
                attractor = CORRECT + 1

            answers = []
            entropies = []
            for _ in range(8):
                is_correct = np.random.random() < p_adj
                if is_correct:
                    ans = CORRECT
                    ent = max(0.1, np.random.normal(CORRECT_ENT_MU, CORRECT_ENT_STD))
                else:
                    if np.random.random() < WRONG_CLUSTER_PROB:
                        ans = attractor
                    else:
                        ans = np.random.randint(0, 100000)
                    ent = max(0.1, np.random.normal(WRONG_ENT_MU, WRONG_ENT_STD))
                answers.append(ans)
                entropies.append(ent)

            weights = {}
            for a, e in zip(answers, entropies):
                weights[a] = weights.get(a, 0) + 1.0 / max(e, 1e-9)
            chosen = max(weights, key=weights.get)

            if chosen == CORRECT:
                score += 1
                if prob_idx < 10:
                    phase1_correct += 1

        cross_scores[sim] = score

    elapsed = time.time() - t0
    r = print_results('CROSS_PROBLEM_LEARNING', cross_scores, baseline_mean)
    r['time'] = elapsed
    all_results['CROSS_PROBLEM_LEARNING'] = r
    print(f"  Time: {elapsed:.1f}s")

    # ============================================================
    # FINAL COMPARISON
    # ============================================================
    print("\n\n" + "=" * 110)
    print("FINAL COMPARISON TABLE")
    print("=" * 110)

    print(f"\n{'Strategy':<50} {'Mean':>6} {'Delta':>7} {'P(>=44)':>8} {'P(>=45)':>8} {'P(>=46)':>8} {'P(>=47)':>8}")
    print("-" * 110)

    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['mean'], reverse=True)
    for name, r in sorted_results:
        delta = r['mean'] - baseline_mean
        sign = '+' if delta >= 0 else ''
        print(f"{name:<50} {r['mean']:>6.2f} {sign}{delta:>6.2f} "
              f"{r['p44']:>8.4f} {r['p45']:>8.4f} {r['p46']:>8.4f} {r['p47']:>8.4f}")

    # ============================================================
    # WINNER
    # ============================================================
    print("\n" + "=" * 90)
    print("WINNER ANALYSIS")
    print("=" * 90)

    non_oracle = {k: v for k, v in all_results.items() if k != 'ORACLE'}
    winner = max(non_oracle, key=lambda k: non_oracle[k]['mean'])
    w = non_oracle[winner]
    oracle = all_results['ORACLE']

    print(f"\nBest strategy: {winner}")
    print(f"Expected score: {w['mean']:.2f}/50 (baseline: {baseline_mean:.2f})")
    print(f"Improvement: +{w['mean'] - baseline_mean:.2f} problems")
    print(f"P(>=44): {w['p44']:.4f}  (baseline: {all_results['BASELINE (1/entropy)']['p44']:.4f})")
    print(f"P(>=45): {w['p45']:.4f}  (baseline: {all_results['BASELINE (1/entropy)']['p45']:.4f})")
    print(f"P(>=46): {w['p46']:.4f}  (baseline: {all_results['BASELINE (1/entropy)']['p46']:.4f})")
    print(f"Oracle: {oracle['mean']:.2f}/50  Gap: {oracle['mean'] - w['mean']:.2f}")

    # Sensitivity analysis skipped for speed - run with different WRONG_CLUSTER_PROB if needed

    print("\n\nDONE.")


if __name__ == '__main__':
    main()
