# R19 hygiene computations — feeds learnings/preregistration_amendment_2026-07-24_DRAFT.md
# Declared tail model (methodology R19 Q2): t-predictive, nu = n-1, scale = s*sqrt(1+1/n).
# All Monte Carlo seeded. Run: uv run --with numpy --with scipy python runs/r19_hygiene/r19_hygiene_stats.py
import json
import numpy as np
from scipy import stats

SEED = 20260724
REPS = 200_000
rng = np.random.default_rng(SEED)

out = {"seed": SEED, "reps": REPS, "date": "2026-07-24",
       "declared_model": "t-predictive: (x - xbar) / (s*sqrt(1+1/n)) ~ t(nu=n-1)"}

# ---------------- Ledger data ----------------
frozen = np.array([0.82, 0.89, 0.93, 1.02, 0.95, 1.33, 0.92, 0.93, 1.14, 0.82])  # n=10, chronological
war = np.array([0.91, 1.08, 0.88, 1.05, 0.76])  # n=5
pooled = np.concatenate([frozen, war])  # n=15 (composition strata; time order handled below)

def summ(x):
    return {"n": len(x), "mean": float(np.mean(x)), "sd": float(np.std(x, ddof=1))}

out["frozen"] = summ(frozen)
out["pooled"] = summ(pooled)

def t_pred_p_upper(x0, data):
    """P(new draw >= x0) under t-predictive."""
    n, m, s = len(data), np.mean(data), np.std(data, ddof=1)
    scale = s * np.sqrt(1 + 1 / n)
    t = (x0 - m) / scale
    return float(stats.t.sf(t, df=n - 1)), float(t)

def t_pred_p_lower(x0, data):
    p, t = t_pred_p_upper(x0, data)
    return 1 - p, t

# ---------------- (a) 0.71 draw p; exceedance table ----------------
p71_f, t71_f = t_pred_p_lower(0.71, frozen)
p71_p, t71_p = t_pred_p_lower(0.71, pooled)
out["draw_0.71"] = {
    "frozen": {"t": t71_f, "one_sided_p": p71_f},
    "pooled": {"t": t71_p, "one_sided_p": p71_p},
    "old_gaussian_p_frozen": float(stats.norm.sf(-(0.71 - frozen.mean()) / frozen.std(ddof=1))),
}

exc = {}
for thr in (1.33, 1.44, 1.47, 1.49):
    pf, tf = t_pred_p_upper(thr, frozen)
    pp, tp = t_pred_p_upper(thr, pooled)
    exc[str(thr)] = {"frozen": {"t": tf, "P_single_draw_ge": pf},
                     "pooled": {"t": tp, "P_single_draw_ge": pp}}
out["exceedance_single_draw"] = exc

# ---------------- (a) E[max] over remaining windows — hierarchical (joint) t-predictive MC ----------
# Proper joint predictive: draw (mu, sigma) from the normal-inverse-chi2 posterior
# (noninformative prior), then K iid draws; this preserves cross-window dependence
# through the shared unknown (mu, sigma). Marginals are exactly the t-predictive.
def emax_mc(data, K, reps, rng):
    n, m, s2 = len(data), np.mean(data), np.var(data, ddof=1)
    nu = n - 1
    sig2 = nu * s2 / rng.chisquare(nu, size=reps)
    mu = m + np.sqrt(sig2 / n) * rng.standard_normal(reps)
    draws = mu[:, None] + np.sqrt(sig2)[:, None] * rng.standard_normal((reps, K))
    mx = draws.max(axis=1)
    return mx

for K in (101,):
    mx = emax_mc(frozen, K, REPS, rng)
    out[f"max_over_{K}_windows_frozen"] = {
        "E_max": float(mx.mean()),
        "median_max": float(np.median(mx)),
        "P_touch_1.33": float((mx >= 1.33).mean()),
        "P_touch_1.44": float((mx >= 1.44).mean()),
        "P_touch_1.47": float((mx >= 1.47).mean()),
        "P_touch_1.49": float((mx >= 1.49).mean()),
        "q10_q90": [float(np.quantile(mx, 0.10)), float(np.quantile(mx, 0.90))],
    }
    mxp = emax_mc(pooled, K, REPS, rng)
    out[f"max_over_{K}_windows_pooled"] = {
        "E_max": float(mxp.mean()),
        "P_touch_1.44": float((mxp >= 1.44).mean()),
        "P_touch_1.49": float((mxp >= 1.49).mean()),
    }

# ---------------- (b) Mann-Kendall + CUSUM on time-ordered draws ----------------
def mann_kendall(x):
    x = np.asarray(x)
    n = len(x)
    S = 0
    for i in range(n - 1):
        S += np.sign(x[i + 1:] - x[i]).sum()
    # tie correction
    vals, counts = np.unique(x, return_counts=True)
    tie_term = sum(t * (t - 1) * (2 * t + 5) for t in counts if t > 1)
    var = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    if S > 0:
        z = (S - 1) / np.sqrt(var)
    elif S < 0:
        z = (S + 1) / np.sqrt(var)
    else:
        z = 0.0
    p = 2 * stats.norm.sf(abs(z))
    return {"S": int(S), "var_S": float(var), "z": float(z), "two_sided_p": float(p)}

def cusum_perm(x, rng, nperm=20_000):
    x = np.asarray(x, dtype=float)
    xc = x - x.mean()
    stat = np.abs(np.cumsum(xc)).max() / (x.std(ddof=1) * np.sqrt(len(x)))
    cnt = 0
    for _ in range(nperm):
        xp = rng.permutation(xc)
        sp = np.abs(np.cumsum(xp)).max() / (x.std(ddof=1) * np.sqrt(len(x)))
        if sp >= stat:
            cnt += 1
    return {"stat": float(stat), "perm_p": float(cnt / nperm), "nperm": nperm}

# Frozen order is certain (chronological as listed; f6=1.33 on 07-18, tail 0.92/0.93/1.14/0.82 = 07-20..23).
out["trend_frozen_n10"] = {"mann_kendall": mann_kendall(frozen), "cusum": cusum_perm(frozen, rng)}
# Pooled order ASSUMED (war interleaved mid-July): f1..f5, w1..w4, f6(1.33), w5, f7..f10
orderA = list(frozen[:5]) + list(war[:4]) + [frozen[5]] + [war[4]] + list(frozen[6:])
# robustness: w5 before f6
orderB = list(frozen[:5]) + list(war) + [frozen[5]] + list(frozen[6:])
out["trend_pooled_n15_orderA"] = {"order": orderA, "mann_kendall": mann_kendall(orderA), "cusum": cusum_perm(np.array(orderA), rng)}
out["trend_pooled_n15_orderB"] = {"mann_kendall": mann_kendall(orderB)}

# ---------------- (d) Promotion gate in exceedance currency ----------------
# Proposed rule (draft): within an arm's first 5 scored windows, PROMOTE iff
#   (P1) >= 2 draws > 1.33  OR  (P2) >= 1 draw >= 1.44.
# Error rates under the declared joint predictive (arm == filler), and power under
# genuinely-better arms (mean shift +delta, same sigma), same hierarchical MC.
def promo_mc(shift, reps, rng, K=5):
    n, m, s2 = len(frozen), np.mean(frozen), np.var(frozen, ddof=1)
    nu = n - 1
    sig2 = nu * s2 / rng.chisquare(nu, size=reps)
    mu = m + shift + np.sqrt(sig2 / n) * rng.standard_normal(reps)
    draws = mu[:, None] + np.sqrt(sig2)[:, None] * rng.standard_normal((reps, K))
    p1 = (draws > 1.33).sum(axis=1) >= 2
    p2 = (draws >= 1.44).sum(axis=1) >= 1
    return {"P_promote": float((p1 | p2).mean()), "P_rule1_only": float(p1.mean()),
            "P_rule2_only": float(p2.mean())}

out["promotion_rule"] = {
    "rule": "within first 5 arm windows: PROMOTE iff (>=2 draws > 1.33) OR (>=1 draw >= 1.44)",
    "false_promotion_arm_eq_filler": promo_mc(0.0, REPS, rng),
    "power_shift_+0.10": promo_mc(0.10, REPS, rng),
    "power_shift_+0.15": promo_mc(0.15, REPS, rng),
    "power_shift_+0.20": promo_mc(0.20, REPS, rng),
}

# ---------------- (e) Harm-pause calibration + resume path ----------------
p_pause_f, t_pf = t_pred_p_lower(0.80, frozen)
p_pause_p, t_pp = t_pred_p_lower(0.80, pooled)
# power vs genuinely harmful arm: true mean 0.85 (same predictive scale)
def shifted_p_lower(x0, data, true_mean):
    n, s = len(data), np.std(data, ddof=1)
    scale = s * np.sqrt(1 + 1 / n)
    return float(stats.t.cdf((x0 - true_mean) / scale, df=n - 1))

out["harm_pause"] = {
    "P_pause_healthy_frozen": p_pause_f, "t_frozen": t_pf,
    "P_pause_healthy_pooled": p_pause_p, "t_pooled": t_pp,
    "P_pause_harmful_mean_0.85_frozen": shifted_p_lower(0.80, frozen, 0.85),
    "P_pause_harmful_mean_0.80_frozen": shifted_p_lower(0.80, frozen, 0.80),
    "P_pause_harmful_mean_0.75_frozen": shifted_p_lower(0.80, frozen, 0.75),
}

# Resume rule (draft): two resume draws; RESUME iff both >= 0.80 AND mean >= 0.90.
def resume_mc(shift, reps, rng):
    n, m, s2 = len(frozen), np.mean(frozen), np.var(frozen, ddof=1)
    nu = n - 1
    sig2 = nu * s2 / rng.chisquare(nu, size=reps)
    mu = m + shift + np.sqrt(sig2 / n) * rng.standard_normal(reps)
    d = mu[:, None] + np.sqrt(sig2)[:, None] * rng.standard_normal((reps, 2))
    ok = (d.min(axis=1) >= 0.80) & (d.mean(axis=1) >= 0.90)
    return float(ok.mean())

out["resume_rule"] = {
    "rule": "two resume draws; RESUME iff both >= 0.80 AND two-draw mean >= 0.90; else SHELVED",
    "P_resume_healthy_arm_eq_filler": resume_mc(0.0, REPS, rng),
    "P_resume_harmful_mean_0.85": resume_mc(0.85 - frozen.mean(), REPS, rng),
    "P_resume_harmful_mean_0.80": resume_mc(0.80 - frozen.mean(), REPS, rng),
}

# ---------------- (h) rule-of-three / exact upper bounds on 29/29, 49/49 ----------------
def zero_fail_upper(n, conf=0.95):
    return {"rule_of_three": 3.0 / n, "exact_clopper_pearson": float(1 - (1 - conf) ** (1 / n))}

out["mechanism_counts"] = {"29_of_29": zero_fail_upper(29), "49_of_49": zero_fail_upper(49)}

# ---------------- (g) Nov-2 gold-cutoff forecast ----------------
# Gold/cutoff series (dates -> value), from daily briefs / discussions sweeps:
#  07-18 wall 1.44 (gold-cutoff proxy: band top forming), 07-22 gold ~1.47 (rank 13),
#  07-23 gold ~1.47, 07-24 gold ~1.49 (top 13). Leader flat 1.86 since 07-14.
t_days = np.array([0, 4, 5, 6], dtype=float)   # days since 07-18
gold = np.array([1.44, 1.47, 1.47, 1.49])
T_NOV2 = 107.0  # 07-18 -> 11-02

# linear
A = np.vstack([t_days, np.ones_like(t_days)]).T
slope, icpt = np.linalg.lstsq(A, gold, rcond=None)[0]
lin_fc = icpt + slope * T_NOV2
# saturating exponential toward ceiling C: g(t) = C - (C - g0) * exp(-k t), g0 = fitted intercept
def sat_fit(C):
    g0 = gold[0]
    ks = np.linspace(1e-4, 0.2, 4000)
    sse = [((C - (C - g0) * np.exp(-k * t_days)) - gold) for k in ks]
    sse = [float((e ** 2).sum()) for e in sse]
    k = float(ks[int(np.argmin(sse))])
    return k, float(C - (C - g0) * np.exp(-k * T_NOV2))

sat = {str(C): dict(zip(["k", "nov2"], sat_fit(C))) for C in (1.61, 1.86, 2.00)}
out["nov2_gold_forecast"] = {
    "series": {"2026-07-18": 1.44, "2026-07-22": 1.47, "2026-07-23": 1.47, "2026-07-24": 1.49},
    "linear": {"slope_per_day": float(slope), "nov2": float(lin_fc),
               "note": "exceeds current leader 1.86 -> treated as implausible upper bound"},
    "saturating_to_ceiling": sat,
}

# rank bleed: (07-20, 40), (07-22, 44), (07-23, 45), (07-24, 49)
rt = np.array([0, 2, 3, 4], dtype=float)
rk = np.array([40, 44, 45, 49], dtype=float)
Ar = np.vstack([rt, np.ones_like(rt)]).T
rs, ri = np.linalg.lstsq(Ar, rk, rcond=None)[0]
pred = ri + rs * rt
ss_res = float(((rk - pred) ** 2).sum()); ss_tot = float(((rk - rk.mean()) ** 2).sum())
out["rank_bleed"] = {"series": {"07-20": 40, "07-22": 44, "07-23": 45, "07-24": 49},
                    "ols_slope_ranks_per_day": float(rs), "r2": 1 - ss_res / ss_tot,
                    "note": "single-overnight 4-rank delta is the max daily step, not the trend"}

path = "runs/r19_hygiene/r19_hygiene_stats.json"
with open(path, "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
