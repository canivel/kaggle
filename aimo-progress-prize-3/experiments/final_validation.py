"""
Final validation: Run large-N simulation of EV voting vs baseline.
Use the calibrated scenario that matches observed 39/50.
"""
import numpy as np
from collections import defaultdict
import time

np.random.seed(42)
N_SIMS = 20000  # Large N for precise estimates
N_PROBLEMS = 50
CORRECT = 42

DIFFICULTIES = [0.85]*35 + [0.40]*5 + [0.20]*5 + [0.05]*3 + [0.01]*2
WCP = 0.70
CORRECT_ENT_MU, CORRECT_ENT_STD = 1.5, 0.3
CONFIDENT_WRONG_ENT_MU = 1.8
WRONG_ENT_MU, WRONG_ENT_STD = 2.5, 0.5
CORRECT_CODE_RATE, WRONG_CODE_RATE = 0.72, 0.60
CORRECT_ERROR_RATE, WRONG_ERROR_RATE = 0.05, 0.25


def simulate(n_sims, use_ev=False):
    scores = np.zeros(n_sims, dtype=int)

    for sim in range(n_sims):
        for prob in range(N_PROBLEMS):
            p = DIFFICULTIES[prob]
            attractor = np.random.randint(0, 100000)
            if attractor == CORRECT:
                attractor = CORRECT + 1

            answers, entropies, py_calls, py_errors = [], [], [], []

            for _ in range(8):
                is_correct = np.random.random() < p
                if is_correct:
                    ans = CORRECT
                    ent = max(0.1, np.random.normal(CORRECT_ENT_MU, CORRECT_ENT_STD))
                    uc = np.random.random() < CORRECT_CODE_RATE
                    pe = 1 if np.random.random() < CORRECT_ERROR_RATE else 0
                else:
                    if np.random.random() < WCP:
                        ans = attractor
                        ent = max(0.1, np.random.normal(CONFIDENT_WRONG_ENT_MU, 0.3))
                    else:
                        ans = np.random.randint(0, 100000)
                        ent = max(0.1, np.random.normal(WRONG_ENT_MU, WRONG_ENT_STD))
                    uc = np.random.random() < WRONG_CODE_RATE
                    pe = 1 if np.random.random() < WRONG_ERROR_RATE else 0
                pc = max(0, int(np.random.normal(3.0, 1.5))) if uc else 0

                answers.append(ans)
                entropies.append(ent)
                py_calls.append(pc)
                py_errors.append(pe)

            w = defaultdict(float)
            for a, e, pc, pe in zip(answers, entropies, py_calls, py_errors):
                wt = 1.0 / max(e, 1e-9)
                if use_ev:
                    if pc > 0 and pe == 0:
                        wt *= 10.0
                    elif pc > 0 and pe > 0:
                        wt *= 0.1
                    elif pc == 0:
                        wt *= 0.2
                w[a] += wt

            if max(w, key=w.get) == CORRECT:
                scores[sim] += 1

    return scores


t0 = time.time()
print("Running baseline (20K sims)...")
baseline = simulate(N_SIMS, use_ev=False)
print(f"  Done in {time.time()-t0:.0f}s")

t0 = time.time()
print("Running EV voting (20K sims)...")
ev = simulate(N_SIMS, use_ev=True)
print(f"  Done in {time.time()-t0:.0f}s")

print("\n" + "=" * 80)
print("FINAL VALIDATED RESULTS (N=20000)")
print("=" * 80)

for name, arr in [("BASELINE (1/entropy)", baseline), ("EV VOTING (10x/0.1x/0.2x)", ev)]:
    print(f"\n  {name}:")
    print(f"    Mean: {arr.mean():.3f}/50  Std: {arr.std():.3f}")
    print(f"    Median: {np.median(arr):.0f}  Min: {arr.min()}  Max: {arr.max()}")
    print(f"    P(>=38): {np.mean(arr >= 38):.4f}")
    print(f"    P(>=39): {np.mean(arr >= 39):.4f}")
    print(f"    P(>=40): {np.mean(arr >= 40):.4f}")
    print(f"    P(>=41): {np.mean(arr >= 41):.4f}")
    print(f"    P(>=42): {np.mean(arr >= 42):.4f}")
    print(f"    P(>=43): {np.mean(arr >= 43):.4f}")
    print(f"    P(>=44): {np.mean(arr >= 44):.4f}")
    print(f"    P(>=45): {np.mean(arr >= 45):.4f}")

delta = ev.mean() - baseline.mean()
print(f"\n  IMPROVEMENT: +{delta:.3f} problems")
print(f"  95% CI for improvement: [{delta - 1.96*np.sqrt(ev.var()/N_SIMS + baseline.var()/N_SIMS):.3f}, "
      f"{delta + 1.96*np.sqrt(ev.var()/N_SIMS + baseline.var()/N_SIMS):.3f}]")

# Score distribution comparison
print(f"\n  Score distribution:")
print(f"  {'Score':>5} {'Baseline':>10} {'EV':>10} {'Delta':>10}")
for s in range(34, 50):
    bp = np.mean(baseline == s)
    ep = np.mean(ev == s)
    if bp > 0.001 or ep > 0.001:
        print(f"  {s:>5} {bp:>10.4f} {ep:>10.4f} {ep-bp:>+10.4f}")
