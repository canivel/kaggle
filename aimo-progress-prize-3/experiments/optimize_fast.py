"""
Fast parameter optimization for execution-verified voting.
Uses smaller sim count for grid search, then validates best with larger run.
"""
import numpy as np
from collections import defaultdict, Counter
import time

np.random.seed(42)
CORRECT = 42

# Scenario A params (calibrated to ~39/50)
DIFFICULTIES = [0.85]*35 + [0.40]*5 + [0.20]*5 + [0.05]*3 + [0.01]*2
WCP = 0.70
CORRECT_ENT_MU, CORRECT_ENT_STD = 1.5, 0.3
WRONG_ENT_MU, WRONG_ENT_STD = 2.5, 0.5
CONFIDENT_WRONG_ENT_MU = 1.8
CORRECT_CODE_RATE, WRONG_CODE_RATE = 0.72, 0.60
CORRECT_ERROR_RATE, WRONG_ERROR_RATE = 0.05, 0.25


def gen_problem_data(n_sims, n_problems, n_att):
    """Vectorized generation of all problem data."""
    p_arr = np.array(DIFFICULTIES[:n_problems])

    # is_correct: (sims, problems, attempts)
    p_tiled = np.broadcast_to(p_arr[None, :, None], (n_sims, n_problems, n_att))
    is_correct = np.random.random((n_sims, n_problems, n_att)) < p_tiled

    # Answers: random, then set correct ones
    answers = np.random.randint(0, 100000, (n_sims, n_problems, n_att))
    answers[is_correct] = CORRECT

    # Attractors and clustering
    attractors = np.random.randint(0, 100000, (n_sims, n_problems))
    attractors[attractors == CORRECT] = CORRECT + 1
    wrong = ~is_correct
    cluster = wrong & (np.random.random((n_sims, n_problems, n_att)) < WCP)
    att_tiled = np.broadcast_to(attractors[:, :, None], (n_sims, n_problems, n_att))
    answers[cluster] = att_tiled[cluster]

    # Entropies
    ent_correct = np.maximum(0.1, np.random.normal(CORRECT_ENT_MU, CORRECT_ENT_STD, (n_sims, n_problems, n_att)))
    ent_wrong_random = np.maximum(0.1, np.random.normal(WRONG_ENT_MU, WRONG_ENT_STD, (n_sims, n_problems, n_att)))
    ent_wrong_confident = np.maximum(0.1, np.random.normal(CONFIDENT_WRONG_ENT_MU, 0.3, (n_sims, n_problems, n_att)))

    entropies = np.where(is_correct, ent_correct,
                         np.where(cluster, ent_wrong_confident, ent_wrong_random))

    # Code usage
    used_code = np.where(is_correct,
                         np.random.random((n_sims, n_problems, n_att)) < CORRECT_CODE_RATE,
                         np.random.random((n_sims, n_problems, n_att)) < WRONG_CODE_RATE)

    # Errors
    py_errors = np.where(is_correct,
                         (np.random.random((n_sims, n_problems, n_att)) < CORRECT_ERROR_RATE).astype(int),
                         (np.random.random((n_sims, n_problems, n_att)) < WRONG_ERROR_RATE).astype(int))

    # Calls
    py_calls = np.where(used_code,
                        np.maximum(0, np.random.normal(3.0, 1.5, (n_sims, n_problems, n_att)).astype(int)),
                        0)

    return answers, entropies, used_code, py_errors, py_calls, is_correct


def score_strategy(answers, entropies, used_code, py_errors, py_calls,
                   code_boost, error_pen, no_code_pen, calls_bonus=1.0):
    """Score a weighting strategy across all sims/problems."""
    n_sims, n_problems, n_att = answers.shape
    scores = np.zeros(n_sims, dtype=int)

    for sim in range(n_sims):
        for prob in range(n_problems):
            # Compute weights
            w = defaultdict(float)
            for j in range(n_att):
                wt = 1.0 / max(entropies[sim, prob, j], 1e-9)

                if used_code[sim, prob, j] and py_errors[sim, prob, j] == 0:
                    wt *= code_boost
                elif used_code[sim, prob, j] and py_errors[sim, prob, j] > 0:
                    wt *= error_pen
                elif not used_code[sim, prob, j]:
                    wt *= no_code_pen

                if calls_bonus != 1.0 and 2 <= py_calls[sim, prob, j] <= 5:
                    wt *= calls_bonus

                w[answers[sim, prob, j]] += wt

            if max(w, key=w.get) == CORRECT:
                scores[sim] += 1

    return scores


def main():
    print("=" * 90)
    print("EXECUTION-VERIFIED VOTING: PARAMETER OPTIMIZATION")
    print("=" * 90)

    # Generate data once
    n_sims_grid = 2000
    n_sims_validate = 10000
    n_att = 8

    t0 = time.time()
    answers, entropies, used_code, py_errors, py_calls, is_correct = \
        gen_problem_data(n_sims_grid, 50, n_att)
    print(f"Grid search data generated in {time.time()-t0:.1f}s")

    # Baseline
    baseline = score_strategy(answers, entropies, used_code, py_errors, py_calls,
                              1.0, 1.0, 1.0)
    baseline_mean = float(baseline.mean())
    print(f"\nBaseline: {baseline_mean:.2f}/50  std={baseline.std():.2f}")

    # Oracle
    oracle = np.sum(np.any(is_correct, axis=2), axis=1)
    print(f"Oracle:   {oracle.mean():.2f}/50  std={oracle.std():.2f}")

    # Grid search
    print(f"\n{'=' * 90}")
    print(f"GRID SEARCH (n={n_sims_grid})")
    print(f"{'=' * 90}")

    best_mean = baseline_mean
    best_params = (1.0, 1.0, 1.0)
    results = []

    for cb in [1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]:
        for ep in [0.05, 0.1, 0.2, 0.3, 0.5]:
            for ncp in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
                s = score_strategy(answers, entropies, used_code, py_errors, py_calls,
                                   cb, ep, ncp)
                mean = float(s.mean())
                delta = mean - baseline_mean
                results.append((cb, ep, ncp, mean, delta))
                if mean > best_mean:
                    best_mean = mean
                    best_params = (cb, ep, ncp)

    # Sort by mean score
    results.sort(key=lambda x: -x[3])

    print(f"\nTop 20 configurations:")
    print(f"{'Rank':>4} {'CodeBoost':>10} {'ErrPen':>8} {'NoCod':>8} {'Mean':>6} {'Delta':>7}")
    print("-" * 50)
    for i, (cb, ep, ncp, mean, delta) in enumerate(results[:20]):
        marker = " ***" if (cb, ep, ncp) == best_params else ""
        print(f"{i+1:>4} {cb:>10.1f} {ep:>8.2f} {ncp:>8.2f} {mean:>6.2f} {delta:>+7.2f}{marker}")

    print(f"\nBottom 5 (worst configs):")
    for i, (cb, ep, ncp, mean, delta) in enumerate(results[-5:]):
        print(f"     {cb:>10.1f} {ep:>8.2f} {ncp:>8.2f} {mean:>6.2f} {delta:>+7.2f}")

    # ============================================================
    # VALIDATE TOP 5 WITH LARGER SIM
    # ============================================================
    print(f"\n{'=' * 90}")
    print(f"VALIDATION: Top 5 configs (n={n_sims_validate})")
    print(f"{'=' * 90}")

    np.random.seed(999)
    answers_v, entropies_v, used_code_v, py_errors_v, py_calls_v, is_correct_v = \
        gen_problem_data(n_sims_validate, 50, n_att)

    baseline_v = score_strategy(answers_v, entropies_v, used_code_v, py_errors_v, py_calls_v,
                                1.0, 1.0, 1.0)
    baseline_mean_v = float(baseline_v.mean())
    oracle_v = np.sum(np.any(is_correct_v, axis=2), axis=1)

    print(f"\nBaseline: {baseline_mean_v:.2f}/50  std={baseline_v.std():.2f}  P(>=44)={np.mean(baseline_v>=44):.4f}")
    print(f"Oracle:   {oracle_v.mean():.2f}/50")

    print(f"\n{'CodeBoost':>10} {'ErrPen':>8} {'NoCod':>8} | {'Mean':>6} {'Std':>5} {'Delta':>7} {'P(39)':>7} {'P(41)':>7} {'P(44)':>7}")
    print("-" * 80)

    for cb, ep, ncp, _, _ in results[:5]:
        s = score_strategy(answers_v, entropies_v, used_code_v, py_errors_v, py_calls_v,
                           cb, ep, ncp)
        mean = float(s.mean())
        delta = mean - baseline_mean_v
        print(f"{cb:>10.1f} {ep:>8.2f} {ncp:>8.2f} | {mean:>6.2f} {s.std():>5.2f} {delta:>+7.2f} "
              f"{np.mean(s>=39):>7.4f} {np.mean(s>=41):>7.4f} {np.mean(s>=44):>7.4f}")

    # Also validate quality_weighted for comparison
    s_quality = score_strategy(answers_v, entropies_v, used_code_v, py_errors_v, py_calls_v,
                               1.0, 0.3, 0.7, calls_bonus=1.3)
    print(f"\nQuality-weighted (for comparison):")
    print(f"     1.0/ent * (err:0.3, noc:0.7, calls:1.3) | {s_quality.mean():.2f} {s_quality.std():.2f} "
          f"{s_quality.mean()-baseline_mean_v:>+7.2f} {np.mean(s_quality>=39):.4f} {np.mean(s_quality>=41):.4f} {np.mean(s_quality>=44):.4f}")

    # ============================================================
    # CASCADE BENEFIT: Extra attempts for split votes
    # ============================================================
    print(f"\n{'=' * 90}")
    print(f"CASCADE BONUS: Extra attempts when base 8 split (using best weighting)")
    print(f"{'=' * 90}")

    cb, ep, ncp = best_params

    for n_extra in [0, 4, 8, 16, 24]:
        for split_thresh in [3, 4, 5]:
            if n_extra == 0 and split_thresh != 3:
                continue

            scores = np.zeros(n_sims_validate, dtype=int)
            extra_count = 0

            for sim in range(n_sims_validate):
                for prob in range(50):
                    p = DIFFICULTIES[prob]
                    # Use pre-generated base data
                    base_answers = answers_v[sim, prob]
                    base_entropies = entropies_v[sim, prob]
                    base_uc = used_code_v[sim, prob]
                    base_pe = py_errors_v[sim, prob]
                    base_pc = py_calls_v[sim, prob]

                    # Check consensus
                    c = Counter(base_answers)
                    top_ans, top_cnt = c.most_common(1)[0]

                    if n_extra > 0 and top_cnt < split_thresh:
                        # Need extra attempts - generate on the fly
                        extra_attempts, _ = gen_problem_data(1, 1, n_extra)
                        # Compute combined vote
                        w = defaultdict(float)

                        # Base attempts
                        for j in range(8):
                            wt = 1.0 / max(base_entropies[j], 1e-9)
                            if base_uc[j] and base_pe[j] == 0:
                                wt *= cb
                            elif base_uc[j] and base_pe[j] > 0:
                                wt *= ep
                            elif not base_uc[j]:
                                wt *= ncp
                            w[base_answers[j]] += wt

                        # Extra attempts
                        for j in range(n_extra):
                            wt = 1.0 / max(extra_attempts[1][0, 0, j], 1e-9)
                            if extra_attempts[2][0, 0, j] and extra_attempts[3][0, 0, j] == 0:
                                wt *= cb
                            elif extra_attempts[2][0, 0, j] and extra_attempts[3][0, 0, j] > 0:
                                wt *= ep
                            elif not extra_attempts[2][0, 0, j]:
                                wt *= ncp
                            w[extra_attempts[0][0, 0, j]] += wt

                        chosen = max(w, key=w.get) if w else 0
                        extra_count += 1
                    else:
                        # Use base only with optimal weights
                        w = defaultdict(float)
                        for j in range(8):
                            wt = 1.0 / max(base_entropies[j], 1e-9)
                            if base_uc[j] and base_pe[j] == 0:
                                wt *= cb
                            elif base_uc[j] and base_pe[j] > 0:
                                wt *= ep
                            elif not base_uc[j]:
                                wt *= ncp
                            w[base_answers[j]] += wt
                        chosen = max(w, key=w.get) if w else 0

                    if chosen == CORRECT:
                        scores[sim] += 1

            mean = float(scores.mean())
            delta = mean - baseline_mean_v
            avg_extra = extra_count / n_sims_validate
            total_avg_attempts = 8 * 50 + avg_extra * n_extra
            label = f"extra={n_extra:>2}, thresh={split_thresh}"
            if n_extra == 0:
                label = "baseline (no extra)"
            print(f"  {label:>25}: mean={mean:.2f} delta={delta:>+.2f} "
                  f"P(44)={np.mean(scores>=44):.4f} "
                  f"avg_extra_problems={avg_extra:.1f} total_att_avg={total_avg_attempts:.0f}")

    # ============================================================
    # FINAL OPTIMAL STRATEGY
    # ============================================================
    print(f"\n{'=' * 90}")
    print(f"FINAL: THE OPTIMAL IMPLEMENTATION")
    print(f"{'=' * 90}")

    print(f"""
Best weight configuration: code_boost={best_params[0]}, error_pen={best_params[1]}, no_code_pen={best_params[2]}

Drop-in replacement for _select_answer in AIMO3Solver:

def _select_answer(self, detailed_results: list) -> int:
    answer_weights = defaultdict(float)
    answer_votes = defaultdict(int)

    for result in detailed_results:
        answer = result['Answer']
        entropy = result['Entropy']
        python_errors = result.get('Python Errors', 0)
        python_calls = result.get('Python Calls', 0)

        if answer is None:
            continue

        # Base weight: inverse entropy
        weight = 1.0 / max(entropy, 1e-9)

        # EXECUTION VERIFICATION (the key innovation)
        has_code = python_calls > 0
        has_errors = python_errors > 0

        if has_code and not has_errors:
            weight *= {best_params[0]}   # Code-verified: {best_params[0]}x boost
        elif has_code and has_errors:
            weight *= {best_params[1]}   # Code with errors: {best_params[1]}x penalty
        elif not has_code:
            weight *= {best_params[2]}   # No code: {best_params[2]}x penalty

        answer_weights[answer] += weight
        answer_votes[answer] += 1

    scored_answers = []
    for answer, total_weight in answer_weights.items():
        scored_answers.append({{
            'answer': answer,
            'votes': answer_votes[answer],
            'score': total_weight
        }})

    scored_answers.sort(key=lambda x: x['score'], reverse=True)

    vote_data = [(item['answer'], item['votes'], item['score']) for item in scored_answers]
    vote_dataframe = pd.DataFrame(vote_data, columns=['Answer', 'Votes', 'Score'])
    vote_dataframe = vote_dataframe.round({{'Score': 3}})
    display(vote_dataframe)

    if not scored_answers:
        print('\\nFinal Answer: 0\\n')
        return 0

    final_answer = scored_answers[0]['answer']
    print(f'\\nFinal Answer: {{final_answer}}\\n')
    return final_answer
""")


if __name__ == '__main__':
    t0 = time.time()
    main()
    print(f"\nTotal time: {time.time()-t0:.1f}s")
