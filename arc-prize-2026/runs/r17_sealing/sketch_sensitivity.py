# R17 sealing — computed annexes (checklist items 1, 4, 9)
# Sealed sketch model (fixes the R16 SS5R "binomial sketch" to an exact, reproducible form):
#   - positives P: uniform over the branch's integer Delta-pairs range (SS4R-derived)
#   - spurious pairs: baseline S+ ~ U{0,1,2}, S- ~ U{0,1,2} independent (the SS5R
#     assumption (iii) "0-2 spurious nonzero pairs of either sign", made exact);
#     annex regime: total spurious T in {4, 6}, each pair + with prob 1/2 (methodology R4)
#   - pass iff n = P + S+ + S- (nonzero pairs) has a sign-test critical at
#     alpha = 0.05 one-sided (SS3R) and wins = P + S+ meets it
# All outputs exact enumeration (fractions), no MC.
import json, math, os
from fractions import Fraction
from itertools import product

def crit(n, alpha=0.05):
    # smallest k with P(X >= k | Binom(n, 1/2)) <= alpha, else None
    for k in range(n, -1, -1):
        tail = sum(math.comb(n, j) for j in range(k, n + 1)) / 2**n
        if tail > alpha:
            return k + 1 if k < n else None
    return None

CRIT = {n: crit(n) for n in range(1, 16)}

def p_pass_baseline(pos_range):
    tot = Fraction(0)
    for P, sp, sm in product(pos_range, range(3), range(3)):
        pr = Fraction(1, len(pos_range)) * Fraction(1, 3) * Fraction(1, 3)
        n = P + sp + sm
        c = CRIT.get(n)
        if c is not None and P + sp >= c:
            tot += pr
    return tot

def p_pass_annex(pos_range, T):
    tot = Fraction(0)
    for P in pos_range:
        for sp in range(T + 1):  # sp of the T spurious are positive
            pr = Fraction(1, len(pos_range)) * Fraction(math.comb(T, sp), 2**T)
            n = P + T
            c = CRIT.get(n)
            if c is not None and P + sp >= c:
                tot += pr
    return tot

BRANCHES = {  # branch -> positives range (post-holdout, post-replay-cost interim haircut)
    "B-/EWM-out": range(1, 4),   # 1-3 (unchanged from SS4R)
    "B-/EWM-in": range(1, 5),    # 1-4 (EWM CLEAN-carriers adds at most 1: ls20)
    "B+/EWM-out": range(2, 6),   # 2-5 (sc25 replay-cost haircut removes the top pair)
    "B+/EWM-in": range(2, 7),    # 2-6 (EWM's ls20 channel is the only non-overlapping add)
}

out = {"criticals_alpha_0.05": {n: CRIT[n] for n in range(5, 13)},
       "baseline": {}, "annex_T4": {}, "annex_T6": {}, "null_size": {}}
for b, r in BRANCHES.items():
    out["baseline"][b] = float(p_pass_baseline(r))
    out["annex_T4"][b] = float(p_pass_annex(r, 4))
    out["annex_T6"][b] = float(p_pass_annex(r, 6))
# size under the full null (0 true positives), annex regimes
for T in (4, 6):
    n = T
    c = CRIT.get(n)
    size = sum(math.comb(T, sp) for sp in range((c if c else T + 1), T + 1)) / 2**T if c else 0.0
    out["null_size"][f"T={T}"] = size

# --- df=2 sigma sensitivity band (methodology R3) ---
from statistics import NormalDist
Phi = NormalDist().cdf
sig_hat, df = 0.189, 2
chi2_hi, chi2_lo = 5.99146, 0.102587  # chi2 quantiles at 0.95 / 0.05, df=2
band = {"sigma_hat": sig_hat, "df": df,
        "sigma_90CI": [sig_hat * (df / chi2_hi) ** 0.5, sig_hat * (df / chi2_lo) ** 0.5]}
rows = {}
for name, sig in [("lo", band["sigma_90CI"][0]), ("point", sig_hat), ("hi", band["sigma_90CI"][1])]:
    se = sig * (2 / 3) ** 0.5  # 3 ON vs n=3 control (post-fallback band)
    rows[name] = {
        "sigma": sig, "SE_delta": se,
        "dismantle_P_trip_null": Phi(-0.10 / se),
        "dismantle_power_true_-0.25": Phi(0.15 / se),
        "guard_false_kill_per_window_at_frozen_-0.28": Phi(-0.28 / se),
        "guard_familywise_3win": 1 - (1 - Phi(-0.28 / se)) ** 3,
    }
band["rows"] = rows

# --- FULL-REPLAY-ONLY replay-action cost (checklist item 4), from certified benchmark.json ---
runs_ = ["war_eval_v1", "war_eval_v2", "war_eval_v3", "w0_eval_s1"]
replay = {}
for r in runs_:
    d = json.load(open(f"runs/kernel_pulls/{r}/benchmark.json"))
    for g in d["game_runs"]:
        gid = g["game_id"].split("-")[0]
        if gid in ("ka59", "re86", "sc25", "ft09", "tu93"):
            lc = g["levels_completed"]
            cost = sum(g["actions_per_level"][:lc]) if lc else None
            replay.setdefault(gid, {})[r] = {"lc": lc, "replay_to_frontier": cost,
                                             "total_actions": len(g["history"])}
out["df2_band"] = band
out["replay_cost"] = replay

os.makedirs("runs/r17_sealing", exist_ok=True)
json.dump(out, open("runs/r17_sealing/sketch_sensitivity.json", "w"), indent=1)
print(json.dumps(out, indent=1))
