"""
AIMO3 Breakthrough Simulation - CALIBRATED to match observed 39/50
==================================================================
Uses TWO calibrated scenarios that both produce ~39/50 baseline:

Scenario A: "Hard problems, moderate clustering"
  35@0.85, 5@0.40, 5@0.20, 3@0.05, 2@0.01, wcp=0.70, confident_wrong_ent=1.5

Scenario B: "Medium problems, extreme clustering"
  30@0.85, 10@0.45, 5@0.25, 3@0.10, 2@0.02, wcp=0.90

Both produce baseline ~ 39/50, but the improvement from extra attempts
differs dramatically between them.

THE KEY QUESTION: Which scenario is real? And which strategy helps most?
"""

import numpy as np
from collections import Counter, defaultdict
import time

np.random.seed(42)
N_SIMS = 5000
N_PROBLEMS = 50
CORRECT = 42

# Correct answer entropy
CORRECT_ENT_MU, CORRECT_ENT_STD = 1.5, 0.3

# Code usage rates
CORRECT_CODE_RATE = 0.72
WRONG_CODE_RATE = 0.60
CORRECT_ERROR_RATE = 0.05
WRONG_ERROR_RATE = 0.25


def gen_attempts(n, p_correct, wcp, wrong_ent_mu, wrong_ent_std, attractor,
                 is_synthesis=False, confident_wrong_ent_mu=None):
    """Generate n attempts for one problem."""
    attempts = []
    for _ in range(n):
        is_correct = np.random.random() < p_correct
        if is_correct:
            ans = CORRECT
            ent = max(0.1, np.random.normal(CORRECT_ENT_MU, CORRECT_ENT_STD))
            uc = True if is_synthesis else (np.random.random() < CORRECT_CODE_RATE)
            pe = 1 if np.random.random() < (0.03 if is_synthesis else CORRECT_ERROR_RATE) else 0
            pc = max(0, int(np.random.normal(3.0, 1.5))) if uc else 0
        else:
            if np.random.random() < wcp:
                ans = attractor
                # Confidently wrong - key feature
                cwem = confident_wrong_ent_mu if confident_wrong_ent_mu else wrong_ent_mu
                ent = max(0.1, np.random.normal(cwem, 0.3))
            else:
                ans = np.random.randint(0, 100000)
                ent = max(0.1, np.random.normal(wrong_ent_mu, wrong_ent_std))
            uc = True if is_synthesis else (np.random.random() < WRONG_CODE_RATE)
            pe = 1 if np.random.random() < (0.35 if is_synthesis else WRONG_ERROR_RATE) else 0
            pc = max(0, int(np.random.normal(2.0, 1.5))) if uc else 0

        attempts.append({
            'answer': ans, 'entropy': ent, 'used_code': uc,
            'python_errors': pe, 'python_calls': pc, 'is_correct': is_correct,
            'is_synthesis': is_synthesis,
        })
    return attempts


# ============================================================
# VOTING STRATEGIES
# ============================================================

def vote_baseline(attempts):
    """1/entropy weighted sum."""
    w = defaultdict(float)
    for a in attempts:
        w[a['answer']] += 1.0 / max(a['entropy'], 1e-9)
    return max(w, key=w.get) if w else 0


def vote_quality(attempts):
    """Quality-weighted: 1/entropy * code signals."""
    w = defaultdict(float)
    for a in attempts:
        wt = 1.0 / max(a['entropy'], 1e-9)
        if a['python_errors'] > 0:
            wt *= 0.3
        if 2 <= a['python_calls'] <= 5:
            wt *= 1.3
        elif a['python_calls'] == 0:
            wt *= 0.7
        w[a['answer']] += wt
    return max(w, key=w.get) if w else 0


def vote_exec_verified(attempts):
    """3x weight for code-verified, 0.3x for code-errored, 0.5x for no-code."""
    w = defaultdict(float)
    for a in attempts:
        wt = 1.0 / max(a['entropy'], 1e-9)
        if a['used_code'] and a['python_errors'] == 0:
            wt *= 3.0
        elif a['used_code'] and a['python_errors'] > 0:
            wt *= 0.3
        elif not a['used_code']:
            wt *= 0.5
        w[a['answer']] += wt
    return max(w, key=w.get) if w else 0


def vote_synthesis_combo(base_attempts, p_correct, wcp, wrong_ent_mu, wrong_ent_std,
                          attractor, confident_wrong_ent_mu=None):
    """
    PROGRAM SYNTHESIS: For split-vote problems, run extra code-only attempts.
    Code-verified synthesis answers get 5x weight.
    """
    # Check consensus on base
    counts = Counter(a['answer'] for a in base_attempts)
    top_ans, top_cnt = counts.most_common(1)[0]
    if top_cnt >= 4:
        return vote_baseline(base_attempts)

    # Split vote - run 8 synthesis attempts
    synth = gen_attempts(8, p_correct, wcp, wrong_ent_mu, wrong_ent_std, attractor,
                         is_synthesis=True, confident_wrong_ent_mu=confident_wrong_ent_mu)

    # Combined voting with synthesis bonus
    w = defaultdict(float)
    for a in base_attempts:
        wt = 1.0 / max(a['entropy'], 1e-9)
        w[a['answer']] += wt

    for a in synth:
        wt = 1.0 / max(a['entropy'], 1e-9)
        if a['python_errors'] == 0:
            wt *= 5.0  # Code-verified synthesis = massive boost
        else:
            wt *= 0.1  # Failed synthesis = near-discard
        w[a['answer']] += wt

    return max(w, key=w.get) if w else 0


def cascade_solve(p_correct, wcp, wrong_ent_mu, wrong_ent_std, attractor,
                   confident_wrong_ent_mu=None):
    """
    CONSENSUS CASCADE: Waves of attempts, stop at consensus.
    Wave 1: 4 attempts, need 3 agree
    Wave 2: +4 (8 total), need 5 agree
    Wave 3: +8 (16 total), need 8 agree
    Wave 4: +16 (32 total), use all
    """
    all_attempts = []
    waves = [(4, 3), (4, 5), (8, 8), (16, 12)]

    for wave_size, threshold in waves:
        new = gen_attempts(wave_size, p_correct, wcp, wrong_ent_mu, wrong_ent_std,
                          attractor, confident_wrong_ent_mu=confident_wrong_ent_mu)
        all_attempts.extend(new)

        counts = Counter(a['answer'] for a in all_attempts)
        top_ans, top_cnt = counts.most_common(1)[0]
        if top_cnt >= threshold:
            return vote_exec_verified(all_attempts)

    return vote_exec_verified(all_attempts)


def combined_solve(p_correct, wcp, wrong_ent_mu, wrong_ent_std, attractor,
                    confident_wrong_ent_mu=None):
    """
    COMBINED: Cascade + Synthesis + Execution Verification
    Wave 1: 8 standard. Check code-verified consensus.
    Wave 2: 8 more standard (16 total). Check consensus.
    Wave 3: 8 synthesis (code-only). Check.
    Wave 4: 8 more synthesis (32 total). Final vote.
    """
    all_attempts = []
    synth_attempts = []

    # Wave 1: 8 standard
    w1 = gen_attempts(8, p_correct, wcp, wrong_ent_mu, wrong_ent_std, attractor,
                      confident_wrong_ent_mu=confident_wrong_ent_mu)
    all_attempts.extend(w1)

    cv = [a for a in all_attempts if a['used_code'] and a['python_errors'] == 0]
    if cv:
        c = Counter(a['answer'] for a in cv)
        if c.most_common(1)[0][1] >= 4:
            return vote_exec_verified(all_attempts)

    # Wave 2: 8 more standard
    w2 = gen_attempts(8, p_correct, wcp, wrong_ent_mu, wrong_ent_std, attractor,
                      confident_wrong_ent_mu=confident_wrong_ent_mu)
    all_attempts.extend(w2)

    c = Counter(a['answer'] for a in all_attempts)
    if c.most_common(1)[0][1] >= 10:
        return vote_exec_verified(all_attempts)

    cv = [a for a in all_attempts if a['used_code'] and a['python_errors'] == 0]
    if cv:
        c = Counter(a['answer'] for a in cv)
        if c.most_common(1)[0][1] >= 6:
            return vote_exec_verified(all_attempts)

    # Wave 3: 8 synthesis
    s1 = gen_attempts(8, p_correct, wcp, wrong_ent_mu, wrong_ent_std, attractor,
                      is_synthesis=True, confident_wrong_ent_mu=confident_wrong_ent_mu)
    synth_attempts.extend(s1)

    # Wave 4: 8 more synthesis
    s2 = gen_attempts(8, p_correct, wcp, wrong_ent_mu, wrong_ent_std, attractor,
                      is_synthesis=True, confident_wrong_ent_mu=confident_wrong_ent_mu)
    synth_attempts.extend(s2)

    # Final: combined voting
    w = defaultdict(float)
    for a in all_attempts:
        wt = 1.0 / max(a['entropy'], 1e-9)
        if a['used_code'] and a['python_errors'] == 0:
            wt *= 3.0
        elif a['used_code'] and a['python_errors'] > 0:
            wt *= 0.3
        elif not a['used_code']:
            wt *= 0.5
        w[a['answer']] += wt

    for a in synth_attempts:
        wt = 1.0 / max(a['entropy'], 1e-9)
        if a['python_errors'] == 0:
            wt *= 5.0
        else:
            wt *= 0.05
        w[a['answer']] += wt

    return max(w, key=w.get) if w else 0


def oracle(attempts):
    for a in attempts:
        if a['is_correct']:
            return CORRECT
    return attempts[0]['answer'] if attempts else 0


# ============================================================
# RUN SCENARIOS
# ============================================================

scenarios = {
    'A: Hard problems, 70% clustering, confident wrong': {
        'difficulties': [0.85]*35 + [0.40]*5 + [0.20]*5 + [0.05]*3 + [0.01]*2,
        'wcp': 0.70,
        'wrong_ent_mu': 2.5,
        'wrong_ent_std': 0.5,
        'confident_wrong_ent_mu': 1.8,  # Clustered wrong = confident
    },
    'B: Moderate problems, 90% clustering': {
        'difficulties': [0.85]*30 + [0.45]*10 + [0.25]*5 + [0.10]*3 + [0.02]*2,
        'wcp': 0.90,
        'wrong_ent_mu': 2.5,
        'wrong_ent_std': 0.5,
        'confident_wrong_ent_mu': None,
    },
    'C: Hard problems, 70% clustering, normal entropy': {
        'difficulties': [0.85]*35 + [0.40]*5 + [0.20]*5 + [0.05]*3 + [0.01]*2,
        'wcp': 0.70,
        'wrong_ent_mu': 2.5,
        'wrong_ent_std': 0.5,
        'confident_wrong_ent_mu': None,
    },
}


def run_scenario(scenario_name, params):
    difficulties = params['difficulties']
    wcp = params['wcp']
    wrong_ent_mu = params['wrong_ent_mu']
    wrong_ent_std = params['wrong_ent_std']
    cwem = params.get('confident_wrong_ent_mu')

    print(f"\n{'=' * 90}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'=' * 90}")
    print(f"  WCP={wcp}, wrong_ent_mu={wrong_ent_mu}, confident_wrong_ent={cwem}")
    print(f"  Difficulty: {len([d for d in difficulties if d > 0.7])} easy, "
          f"{len([d for d in difficulties if 0.3 < d <= 0.7])} medium, "
          f"{len([d for d in difficulties if 0.1 < d <= 0.3])} hard, "
          f"{len([d for d in difficulties if d <= 0.1])} very hard")

    strategy_scores = {}
    strategy_names = [
        'BASELINE (8 att, 1/ent)',
        'QUALITY_WEIGHTED (8 att)',
        'EXEC_VERIFIED (8 att)',
        'PROGRAM_SYNTHESIS (8+8 for split)',
        'CONSENSUS_CASCADE (4-32 att)',
        'COMBINED (cascade+synth+exec)',
        'ORACLE (upper bound)',
    ]

    for strat in strategy_names:
        t0 = time.time()
        scores = np.zeros(N_SIMS, dtype=int)

        for sim in range(N_SIMS):
            for prob_idx in range(N_PROBLEMS):
                p = difficulties[prob_idx % len(difficulties)]
                attractor = np.random.randint(0, 100000)
                if attractor == CORRECT:
                    attractor = CORRECT + 1

                if strat == 'BASELINE (8 att, 1/ent)':
                    attempts = gen_attempts(8, p, wcp, wrong_ent_mu, wrong_ent_std, attractor,
                                           confident_wrong_ent_mu=cwem)
                    chosen = vote_baseline(attempts)

                elif strat == 'QUALITY_WEIGHTED (8 att)':
                    attempts = gen_attempts(8, p, wcp, wrong_ent_mu, wrong_ent_std, attractor,
                                           confident_wrong_ent_mu=cwem)
                    chosen = vote_quality(attempts)

                elif strat == 'EXEC_VERIFIED (8 att)':
                    attempts = gen_attempts(8, p, wcp, wrong_ent_mu, wrong_ent_std, attractor,
                                           confident_wrong_ent_mu=cwem)
                    chosen = vote_exec_verified(attempts)

                elif strat == 'PROGRAM_SYNTHESIS (8+8 for split)':
                    attempts = gen_attempts(8, p, wcp, wrong_ent_mu, wrong_ent_std, attractor,
                                           confident_wrong_ent_mu=cwem)
                    chosen = vote_synthesis_combo(attempts, p, wcp, wrong_ent_mu, wrong_ent_std,
                                                  attractor, confident_wrong_ent_mu=cwem)

                elif strat == 'CONSENSUS_CASCADE (4-32 att)':
                    chosen = cascade_solve(p, wcp, wrong_ent_mu, wrong_ent_std, attractor,
                                           confident_wrong_ent_mu=cwem)

                elif strat == 'COMBINED (cascade+synth+exec)':
                    chosen = combined_solve(p, wcp, wrong_ent_mu, wrong_ent_std, attractor,
                                            confident_wrong_ent_mu=cwem)

                elif strat == 'ORACLE (upper bound)':
                    attempts = gen_attempts(8, p, wcp, wrong_ent_mu, wrong_ent_std, attractor,
                                           confident_wrong_ent_mu=cwem)
                    chosen = oracle(attempts)

                else:
                    chosen = 0

                if chosen == CORRECT:
                    scores[sim] += 1

        elapsed = time.time() - t0
        arr = scores
        strategy_scores[strat] = {
            'mean': float(arr.mean()),
            'std': float(arr.std()),
            'p44': float(np.mean(arr >= 44)),
            'p45': float(np.mean(arr >= 45)),
            'p46': float(np.mean(arr >= 46)),
        }
        print(f"\n  {strat}:")
        print(f"    Mean: {arr.mean():.2f}  Std: {arr.std():.2f}  P(>=44): {np.mean(arr >= 44):.4f}  "
              f"P(>=45): {np.mean(arr >= 45):.4f}  P(>=46): {np.mean(arr >= 46):.4f}  [{elapsed:.1f}s]")

    # Summary table
    baseline_mean = strategy_scores['BASELINE (8 att, 1/ent)']['mean']
    print(f"\n  {'Strategy':<45} {'Mean':>6} {'Delta':>7} {'P(>=44)':>8} {'P(>=45)':>8} {'P(>=46)':>8}")
    print(f"  {'-'*90}")
    for name in sorted(strategy_scores, key=lambda k: strategy_scores[k]['mean'], reverse=True):
        r = strategy_scores[name]
        delta = r['mean'] - baseline_mean
        print(f"  {name:<45} {r['mean']:>6.2f} {delta:>+7.2f} {r['p44']:>8.4f} {r['p45']:>8.4f} {r['p46']:>8.4f}")

    return strategy_scores


def main():
    print("=" * 90)
    print("AIMO3 CALIBRATED BREAKTHROUGH SIMULATION")
    print(f"N_SIMS={N_SIMS}, N_PROBLEMS={N_PROBLEMS}")
    print("Target: baseline ~ 39/50 (matching observed competition score)")
    print("=" * 90)

    all_scenarios = {}
    for name, params in scenarios.items():
        all_scenarios[name] = run_scenario(name, params)

    # ============================================================
    # FINAL CROSS-SCENARIO ANALYSIS
    # ============================================================
    print("\n\n" + "=" * 90)
    print("CROSS-SCENARIO COMPARISON: Which strategy is robustly the best?")
    print("=" * 90)

    # For each strategy, show mean improvement across scenarios
    strat_improvements = defaultdict(list)
    for scenario_name, scenario_results in all_scenarios.items():
        baseline = scenario_results['BASELINE (8 att, 1/ent)']['mean']
        for strat, r in scenario_results.items():
            if strat != 'ORACLE (upper bound)':
                strat_improvements[strat].append(r['mean'] - baseline)

    print(f"\n{'Strategy':<45} {'Avg Delta':>10} {'Min Delta':>10} {'Max Delta':>10}")
    print("-" * 80)
    for strat in sorted(strat_improvements, key=lambda k: np.mean(strat_improvements[k]), reverse=True):
        deltas = strat_improvements[strat]
        print(f"{strat:<45} {np.mean(deltas):>+10.2f} {min(deltas):>+10.2f} {max(deltas):>+10.2f}")

    # Show the oracle gap
    print(f"\n{'Strategy':<45} {'Scenario A':>12} {'Scenario B':>12} {'Scenario C':>12}")
    print("-" * 90)
    for strat in ['BASELINE (8 att, 1/ent)', 'QUALITY_WEIGHTED (8 att)', 'EXEC_VERIFIED (8 att)',
                   'PROGRAM_SYNTHESIS (8+8 for split)', 'CONSENSUS_CASCADE (4-32 att)',
                   'COMBINED (cascade+synth+exec)', 'ORACLE (upper bound)']:
        vals = []
        for sname in scenarios:
            vals.append(f"{all_scenarios[sname][strat]['mean']:.2f}")
        print(f"  {strat:<43} {'  '.join(f'{v:>12}' for v in vals)}")


if __name__ == '__main__':
    main()
