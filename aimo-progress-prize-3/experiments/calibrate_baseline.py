"""
Calibrate the simulation model to match observed 39/50 baseline.
We need to find the right combination of:
- Per-attempt accuracy distribution
- Wrong answer clustering rate
That produces baseline ~ 39/50 with 8 attempts + 1/entropy voting.
"""

import numpy as np
from collections import defaultdict

np.random.seed(42)
N_SIMS = 10000
N_PROBLEMS = 50
CORRECT = 42

CORRECT_ENT_MU, CORRECT_ENT_STD = 1.5, 0.3
WRONG_ENT_MU, WRONG_ENT_STD = 2.5, 0.5


def simulate_baseline(difficulties, wrong_cluster_prob, n_sims=N_SIMS):
    """Simulate baseline 1/entropy voting."""
    scores = np.zeros(n_sims, dtype=int)

    for sim in range(n_sims):
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
                    if np.random.random() < wrong_cluster_prob:
                        ans = attractor
                    else:
                        ans = np.random.randint(0, 100000)
                    ent = max(0.1, np.random.normal(WRONG_ENT_MU, WRONG_ENT_STD))
                answers.append(ans)
                entropies.append(ent)

            weights = {}
            for a, e in zip(answers, entropies):
                weights[a] = weights.get(a, 0) + 1.0 / max(e, 1e-9)
            if max(weights, key=weights.get) == CORRECT:
                scores[sim] += 1

    return np.mean(scores), np.std(scores)


# Scan parameter space
print("=" * 90)
print("CALIBRATION: Finding difficulty mix that produces baseline ~ 39/50")
print("=" * 90)

# Test 1: Uniform p with different clustering
print("\nTest 1: Uniform p with different clustering")
print(f"{'p':>6} {'wcp':>6} | {'mean':>6} {'std':>5}")
print("-" * 35)

for p in [0.50, 0.55, 0.60, 0.65, 0.69, 0.75, 0.80]:
    for wcp in [0.30, 0.50, 0.70, 0.90]:
        difficulties = [p] * 50
        mean, std = simulate_baseline(difficulties, wcp, n_sims=2000)
        marker = " <-- TARGET" if abs(mean - 39) < 1.0 else ""
        print(f"{p:>6.2f} {wcp:>6.2f} | {mean:>6.2f} {std:>5.2f}{marker}")

# Test 2: Mixed difficulty with different clustering
print("\nTest 2: Mixed difficulty with different clustering")
print(f"{'mix':>20} {'wcp':>6} | {'mean':>6} {'std':>5}")
print("-" * 50)

mixes = [
    ("39@0.85,5@0.45,3@0.25,2@0.10,1@0.02", [0.85]*39 + [0.45]*5 + [0.25]*3 + [0.10]*2 + [0.02]*1),
    ("30@0.85,10@0.45,5@0.25,3@0.10,2@0.02", [0.85]*30 + [0.45]*10 + [0.25]*5 + [0.10]*3 + [0.02]*2),
    ("25@0.80,10@0.50,8@0.30,4@0.15,3@0.05", [0.80]*25 + [0.50]*10 + [0.30]*8 + [0.15]*4 + [0.05]*3),
    ("20@0.80,15@0.50,8@0.25,4@0.10,3@0.03", [0.80]*20 + [0.50]*15 + [0.25]*8 + [0.10]*4 + [0.03]*3),
    ("35@0.85,5@0.40,5@0.20,3@0.05,2@0.01", [0.85]*35 + [0.40]*5 + [0.20]*5 + [0.05]*3 + [0.01]*2),
]

for name, diffs in mixes:
    for wcp in [0.30, 0.50, 0.70, 0.90]:
        mean, std = simulate_baseline(diffs, wcp, n_sims=2000)
        marker = " <-- TARGET" if abs(mean - 39) < 1.0 else ""
        print(f"{name:>50} {wcp:>6.2f} | {mean:>6.2f} {std:>5.2f}{marker}")

# Test 3: What if confidently-wrong has entropy closer to correct?
print("\n\nTest 3: What if wrong answers have LOWER entropy (confidently wrong)?")
print("This models systematic misconception where the model is SURE of wrong answer")

for wrong_ent_mu in [1.5, 1.8, 2.0, 2.5]:
    for wcp in [0.30, 0.50, 0.70]:
        scores = np.zeros(2000, dtype=int)
        for sim in range(2000):
            difficulties = [0.85]*39 + [0.45]*5 + [0.25]*3 + [0.10]*2 + [0.02]*1
            for prob_idx in range(50):
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
                        if np.random.random() < wcp:
                            ans = attractor
                            # Clustered wrong = confidently wrong!
                            ent = max(0.1, np.random.normal(wrong_ent_mu, 0.3))
                        else:
                            ans = np.random.randint(0, 100000)
                            ent = max(0.1, np.random.normal(2.5, 0.5))
                        entropies.append(ent)
                    if is_correct:
                        entropies.append(ent)
                    answers.append(ans)

                # Fix: entropies list was built wrongly, rebuild
                pass

            scores[sim] = 0  # Will rebuild below

        # Redo properly
        scores = np.zeros(2000, dtype=int)
        for sim in range(2000):
            difficulties = [0.85]*39 + [0.45]*5 + [0.25]*3 + [0.10]*2 + [0.02]*1
            for prob_idx in range(50):
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
                        if np.random.random() < wcp:
                            ans = attractor
                            ent = max(0.1, np.random.normal(wrong_ent_mu, 0.3))
                        else:
                            ans = np.random.randint(0, 100000)
                            ent = max(0.1, np.random.normal(2.5, 0.5))
                    answers.append(ans)
                    entropies.append(ent)

                weights = {}
                for a, e in zip(answers, entropies):
                    weights[a] = weights.get(a, 0) + 1.0 / max(e, 1e-9)
                if max(weights, key=weights.get) == CORRECT:
                    scores[sim] += 1

        mean = np.mean(scores)
        marker = " <-- TARGET" if abs(mean - 39) < 1.0 else ""
        print(f"  wrong_ent_mu={wrong_ent_mu:.1f}, wcp={wcp:.2f} -> mean={mean:.2f}/50{marker}")

print("\n\nKEY INSIGHT: The 39/50 is probably explained by:")
print("  1. Higher wrong-answer clustering (50-70%) AND/OR")
print("  2. Confidently wrong answers (entropy ~ 1.5-1.8 for systematic errors)")
print("  3. A harder problem mix than assumed")
