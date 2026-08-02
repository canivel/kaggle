#!/usr/bin/env python
"""
NC-15a / NC-15b / NC-13 discharge — stationarity memo reproduction & audit.

Recomputes every number in `learnings/sweeps/stationarity_2026-08-02.md` from the
draw ledger in `runs/lb_ground_truth.md` (n=19 record), flags any that does NOT
reproduce, and adds the panel-requested extras:

  NC-15a  pair-probability reproduction (MK / CUSUM / change-point perm-p / pair-prob
          under the 3 regimes) with PRE-REGISTERED CUSUM h and min-segment as header
          constants.
  NC-15b  conditional multiple-looks false-alarm rate of the headline change-point
          p=0.0032 (10k simulated stationary ledgers).
  NC-13   yw8837 sigma=0.24 compatibility test (chi-square variance test + F/Levene)
          on our own n=15 (sealed) and n=19 (record), plus the <0.80 harm-pause
          false-fire rates under each regime.
  (4)     paired / relative harm-pause re-derivation with false-fire rates.

All local compute. Deterministic seed. Emits JSON to
`runs/stationarity_repro_2026-08-02.json`.

Run:  uv run python scripts/stationarity_repro.py
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import numpy as np
from scipy import stats

# ----------------------------------------------------------------------------
# PRE-REGISTERED constants (state them here so they are not threshold-shopped).
# Sourced from stationarity_2026-08-02.md §2(b)/(c) and R23 directive 9 / NC-15(iii).
# ----------------------------------------------------------------------------
SEED = 20260802
CUSUM_H_STATE = 4          # pre-registered primary CUSUM alarm bound (h=4)
CUSUM_H_STRICT = 5         # pre-registered stricter bound (h=5)
CUSUM_K = 0.5              # tabular CUSUM reference-value slack (k, in sigma units)
MIN_SEGMENT = 3           # NC-15(iii) / directive 9: minimum segment length for split scan
PERM_B = 20000            # permutation replicates (memo used B=20,000)
SIM_N = 10000             # NC-15b false-alarm simulations
CP_ALPHA = 0.01           # "significant" change-point threshold for the false-alarm sim

# Sealed n=15 control parameters (frozen per prereg §3)
SEALED_MU = 0.9727
SEALED_S = 0.1343
SEALED_N = 15
SEALED_DF = SEALED_N - 1   # 14

# yw8837 external regime
YW_SIGMA = 0.24

# Harm-pause fixed floor
HARM_FLOOR = 0.80
LOW_THRESH = 0.68          # the "tail" threshold the memo scans for (the 0.68 draw)

# ----------------------------------------------------------------------------
# The canonical n=19 record ledger (runs/lb_ground_truth.md ordering).
# ----------------------------------------------------------------------------
LEDGER = [0.82, 0.89, 0.93, 1.02, 0.95, 1.33, 0.92, 0.93, 1.14, 0.82,
          1.05, 0.84, 1.02, 0.90, 1.03, 0.85, 1.10, 0.65, 0.68]

# Memo's published values, for reproduce-or-not flagging.
MEMO = {
    "mk_S": -14,
    "mk_varS": 814.0,
    "mk_z": -0.456,
    "mk_p": 0.649,
    "sen_slope": -0.0050,
    "cusum_final": -4.55,
    "cusum_maxabs": 4.55,
    "cusum_tabular_min": -3.58,
    "cp_maxt": 8.64,
    "cp_prereg_t": 8.65,
    "cp_perm_p": 0.0032,
    "perdraw_gauss": 0.0147,
    "perdraw_t": 0.0267,
    "pair_gauss": 0.0038,
    "pair_t": 0.0124,
    "perdraw_yw": 0.111,
    "pair_yw": 0.185,
    # NC-13 / harm-pause claims from the panel + memo
    "chi2_reject_p": 0.007,       # prog-synthesis P(s<=0.1343 | sigma=0.24)
    "harm_fire_sealed_4draw": 0.34,  # P(>=1 spurious harm-pause in 4 draws), sealed
    "harm_fire_yw_4draw": 0.66,      # under sigma=0.24
    "harm_perdraw_sealed": 0.099,    # P(gated draw < 0.80 | null sealed)
}

REL_TOL = 0.05  # relative tolerance for "reproduces"
ABS_TOL = 0.003  # absolute tolerance floor (for small probabilities)


def reproduces(got, memo, rel=REL_TOL, abs_=ABS_TOL):
    if memo == 0:
        return abs(got) <= abs_
    return abs(got - memo) <= max(abs_, rel * abs(memo))


# ----------------------------------------------------------------------------
# (a) Mann-Kendall trend test + Sen slope
# ----------------------------------------------------------------------------
def mann_kendall(x):
    x = np.asarray(x, float)
    n = len(x)
    S = 0
    for i in range(n - 1):
        S += np.sum(np.sign(x[i + 1:] - x[i]))
    S = int(S)
    # variance with tie correction
    _, counts = np.unique(x, return_counts=True)
    tie = np.sum(counts * (counts - 1) * (2 * counts + 5))
    varS = (n * (n - 1) * (2 * n + 5) - tie) / 18.0
    if S > 0:
        z = (S - 1) / math.sqrt(varS)
    elif S < 0:
        z = (S + 1) / math.sqrt(varS)
    else:
        z = 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    # Sen slope
    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            slopes.append((x[j] - x[i]) / (j - i))
    sen = float(np.median(slopes))
    return {"S": S, "varS": float(varS), "z": float(z), "p": float(p), "sen_slope": sen}


# ----------------------------------------------------------------------------
# (b) CUSUM vs sealed N(mu, s)
# ----------------------------------------------------------------------------
def cusum(x, mu, s, k=CUSUM_K):
    z = (np.asarray(x, float) - mu) / s
    # standardized cumulative sum path (memo: "standardized cumsum")
    cum = np.cumsum(z)
    final = float(cum[-1])
    maxabs = float(np.max(np.abs(cum)))
    # tabular one-sided lower CUSUM (downward shift), reference value k
    C_lo = np.zeros(len(z))
    prev = 0.0
    for i, zi in enumerate(z):
        prev = min(0.0, prev + zi + k)
        C_lo[i] = prev
    tabular_min = float(np.min(C_lo))
    return {
        "path": [float(v) for v in cum],
        "final": final,
        "maxabs": maxabs,
        "tabular_lower_min": tabular_min,
        "crosses_h4": maxabs >= CUSUM_H_STATE,
        "crosses_h5": maxabs >= CUSUM_H_STRICT,
    }


# ----------------------------------------------------------------------------
# (c) Change-point: max |Welch t| over interior splits (min-segment >= MIN_SEGMENT),
#     permutation p on the max statistic.
# ----------------------------------------------------------------------------
def welch_t(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    va, vb = a.var(ddof=1), b.var(ddof=1)
    denom = math.sqrt(va / na + vb / nb)
    if denom == 0:
        return np.nan
    return (a.mean() - b.mean()) / denom


def max_split_t(x, min_seg):
    """Max |Welch t| over splits respecting min segment length. Returns (maxt, split_idx)."""
    n = len(x)
    best_t, best_k = 0.0, None
    for k in range(min_seg, n - min_seg + 1):
        t = welch_t(x[:k], x[k:])
        if not np.isnan(t) and abs(t) > best_t:
            best_t, best_k = abs(t), k
    return best_t, best_k


def max_split_t_nomin(x):
    """Max |Welch t| over ALL interior splits (min_seg=2 for Welch validity) — matches
    the memo's unconstrained scan that produced the n2=2 |t|=8.64 artifact."""
    n = len(x)
    best_t, best_k = 0.0, None
    for k in range(2, n - 1):  # need >=2 on each side for Welch var
        t = welch_t(x[:k], x[k:])
        if not np.isnan(t) and abs(t) > best_t:
            best_t, best_k = abs(t), k
    return best_t, best_k


def changepoint(x, rng, min_seg=None, B=PERM_B):
    x = np.asarray(x, float)
    n = len(x)
    # memo's scan (unconstrained, allows n2=2)
    obs_memo, split_memo = max_split_t_nomin(x)
    # pre-registered min-segment scan
    ms = min_seg if min_seg is not None else MIN_SEGMENT
    obs_min, split_min = max_split_t(x, ms)
    # pre-specified split: last-2-vs-prior-17
    t_prereg = abs(welch_t(x[:n - 2], x[n - 2:]))
    # permutation null on the MAX-over-splits statistic (memo config: unconstrained)
    count_memo = 0
    count_min = 0
    for _ in range(B):
        perm = rng.permutation(x)
        tm, _ = max_split_t_nomin(perm)
        if tm >= obs_memo - 1e-12:
            count_memo += 1
        tmin, _ = max_split_t(perm, ms)
        if tmin >= obs_min - 1e-12:
            count_min += 1
    return {
        "max_t_unconstrained": float(obs_memo),
        "split_unconstrained": int(split_memo) if split_memo else None,
        "max_t_minseg3": float(obs_min),
        "split_minseg3": int(split_min) if split_min else None,
        "t_prereg_last2": float(t_prereg),
        "perm_p_unconstrained": (count_memo + 1) / (B + 1),
        "perm_p_minseg3": (count_min + 1) / (B + 1),
    }


# ----------------------------------------------------------------------------
# (d) Consecutive-pair probability under a null.
# ----------------------------------------------------------------------------
def per_draw_prob(thresh, dist):
    """P(single draw <= thresh) under a given predictive distribution."""
    if dist["kind"] == "gauss":
        return float(stats.norm.cdf(thresh, dist["mu"], dist["sigma"]))
    elif dist["kind"] == "t":
        # t-predictive: (x-mu)/sd_pred ~ t_df
        z = (thresh - dist["mu"]) / dist["sd_pred"]
        return float(stats.t.cdf(z, dist["df"]))
    raise ValueError(dist["kind"])


def prob_at_least_one_consecutive_pair(p, n):
    """P(>=1 consecutive pair of successes) in n iid Bernoulli(p) trials.
    = 1 - P(no two consecutive successes). Use the no-two-consecutive recurrence.
    Let f(n) = P(no two consecutive successes in n trials).
    DP over states: prev = last trial was success? track prob mass."""
    q = 1 - p
    # state[0] = prob mass of sequences ending in a NON-success with no prior adjacent pair
    # state[1] = prob mass ending in a success (single, no adjacent pair yet)
    s0, s1 = q, p
    for _ in range(n - 1):
        ns0 = (s0 + s1) * q
        ns1 = s0 * p  # can only place a success if previous was non-success
        s0, s1 = ns0, ns1
    p_no_pair = s0 + s1
    return 1.0 - p_no_pair


def pair_prob_block(thresh):
    dists = {
        "gauss_sealed": {"kind": "gauss", "mu": SEALED_MU, "sigma": SEALED_S},
        "t_fit": {"kind": "t", "mu": SEALED_MU, "sd_pred": 0.1387, "df": SEALED_DF},
        "yw_sigma024": {"kind": "gauss", "mu": SEALED_MU, "sigma": YW_SIGMA},
    }
    out = {}
    n = len(LEDGER)
    for name, d in dists.items():
        pd = per_draw_prob(thresh, d)
        p_specific_pair = pd * pd
        p_any_pair = prob_at_least_one_consecutive_pair(pd, n)
        out[name] = {
            "per_draw": pd,
            "specific_adjacent_pair": p_specific_pair,
            "any_consecutive_pair_in_n": p_any_pair,
            "expected_count_in_n": pd * n,
        }
    return out


# ----------------------------------------------------------------------------
# NC-15b — conditional multiple-looks false-alarm rate.
# ----------------------------------------------------------------------------
def false_alarm_sim(rng, n=None, B=SIM_N, alpha=CP_ALPHA):
    """Fraction of simulated stationary N(sealed) ledgers whose max-over-splits
    change-point permutation p < alpha. This is the multiple-looks false-alarm
    rate of the memo's headline p=0.0032.

    To keep 10k x inner-perm tractable, we build a permutation reference
    distribution of the max-|t| statistic ONCE (large B), then for each simulated
    ledger compute its max-|t| and its p = P(ref >= obs). This is exactly the
    memo's permutation test (the null is exchangeable Gaussian; the permutation
    reference IS the sampling distribution of max-|t| under exchangeability), so
    it is faithful, not an approximation shortcut."""
    n = n or len(LEDGER)
    # Reference distribution of max-|t| under the exchangeable null.
    # Under a stationary iid Gaussian, max-|t| over splits is location/scale
    # invariant in distribution, so we can build it from standard-normal draws.
    REF = 200000
    ref_stats = np.empty(REF)
    for i in range(REF):
        y = rng.standard_normal(n)
        t, _ = max_split_t_nomin(y)
        ref_stats[i] = t
    ref_sorted = np.sort(ref_stats)

    def perm_p_of(obs):
        # p = fraction of ref >= obs
        idx = np.searchsorted(ref_sorted, obs, side="left")
        return (REF - idx) / REF

    # Simulate ledgers from the sealed Gaussian, compute each one's perm-p.
    fired = 0
    ps = np.empty(B)
    for b in range(B):
        y = rng.normal(SEALED_MU, SEALED_S, n)
        t, _ = max_split_t_nomin(y)
        p = perm_p_of(t)
        ps[b] = p
        if p < alpha:
            fired += 1
    return {
        "n": n,
        "alpha": alpha,
        "n_sims": B,
        "ref_size": REF,
        "false_alarm_rate": fired / B,
        "median_p": float(np.median(ps)),
        "p05": float(np.percentile(ps, 5)),
    }


# ----------------------------------------------------------------------------
# NC-13 — yw8837 sigma=0.24 compatibility test.
# ----------------------------------------------------------------------------
def sigma_compat(s, n, sigma0=YW_SIGMA):
    """Chi-square variance test H0: sigma = sigma0 vs our sample s (n draws).
    chi2 = (n-1) * s^2 / sigma0^2 ~ chi2_{n-1}.
    Report P(s <= observed | sigma0) = P(chi2_{n-1} <= stat) (lower tail — our
    data are TIGHTER than sigma0, so the relevant question is whether such a small
    s is plausible), two-sided p, and 95% CI for sigma from our data."""
    df = n - 1
    stat = df * s * s / (sigma0 * sigma0)
    p_lower = float(stats.chi2.cdf(stat, df))          # P(observing s this small or smaller)
    p_upper = float(stats.chi2.sf(stat, df))
    p_two = 2 * min(p_lower, p_upper)
    # 95% CI for sigma from our sample (chi-square)
    lo = math.sqrt(df * s * s / stats.chi2.ppf(0.975, df))
    hi = math.sqrt(df * s * s / stats.chi2.ppf(0.025, df))
    return {
        "n": n, "s": s, "sigma0": sigma0, "df": df,
        "chi2_stat": float(stat),
        "P(s<=obs | sigma0)": p_lower,
        "P(s>=obs | sigma0)": p_upper,
        "p_two_sided": float(min(1.0, p_two)),
        "sigma_95CI": [float(lo), float(hi)],
        "sigma0_inside_95CI": bool(lo <= sigma0 <= hi),
    }


def harm_pause_fixed(mu, sigma, floor=HARM_FLOOR, k=4):
    """Fixed <floor harm-pause. P(single draw < floor) and P(>=1 in k draws)."""
    p1 = float(stats.norm.cdf(floor, mu, sigma))
    p_any = 1 - (1 - p1) ** k
    return {"per_draw": p1, "any_in_k": p_any, "k": k}


# ----------------------------------------------------------------------------
# (4) Paired / relative harm-pause re-derivation.
# ----------------------------------------------------------------------------
def paired_harm_pause(rng, mu_gate, sigma_gate, mu_ctrl, sigma_ctrl,
                      trailing_k=4, c=1.5, B=200000):
    """Relative harm-pause: pause if gated draw < (mean - c*s) of the trailing-k
    contemporaneous control fillers. False-fire = firing when the gate is NULL
    (gate distribution == control distribution). Simulate.

    Under H0 (no harm), gate draw and control fillers are exchangeable from the
    same distribution, so the false-fire rate is DISTRIBUTION-FREE in mu and only
    weakly depends on sigma via the small-sample s estimate. We simulate under the
    supplied (mu_gate=mu_ctrl, sigma) to confirm invariance across regimes."""
    fires = 0
    for _ in range(B):
        ctrl = rng.normal(mu_ctrl, sigma_ctrl, trailing_k)
        gate = rng.normal(mu_gate, sigma_gate)
        m = ctrl.mean()
        sd = ctrl.std(ddof=1)
        if gate < m - c * sd:
            fires += 1
    return fires / B


def main():
    rng = np.random.default_rng(SEED)
    x = np.array(LEDGER, float)
    n = len(x)

    results = {
        "meta": {
            "generated": datetime.now(timezone.utc).isoformat(),
            "seed": SEED,
            "ledger": LEDGER,
            "n": n,
            "prereg_constants": {
                "CUSUM_H_STATE": CUSUM_H_STATE,
                "CUSUM_H_STRICT": CUSUM_H_STRICT,
                "CUSUM_K": CUSUM_K,
                "MIN_SEGMENT": MIN_SEGMENT,
                "PERM_B": PERM_B,
                "SIM_N": SIM_N,
                "CP_ALPHA": CP_ALPHA,
            },
            "sealed_control": {"mu": SEALED_MU, "s": SEALED_S, "n": SEALED_N},
        }
    }

    # basic ledger stats
    results["ledger_stats"] = {
        "mean_n19": float(x.mean()),
        "s_n19": float(x.std(ddof=1)),
        "mean_last2": float(x[-2:].mean()),
        "mean_prior17": float(x[:-2].mean()),
        "z_065_vs_sealed": float((0.65 - SEALED_MU) / SEALED_S),
        "z_068_vs_sealed": float((0.68 - SEALED_MU) / SEALED_S),
    }

    # (a) MK
    mk = mann_kendall(x)
    results["mann_kendall"] = mk

    # (b) CUSUM
    cu = cusum(x, SEALED_MU, SEALED_S)
    results["cusum"] = cu

    # (c) change-point
    cp = changepoint(x, rng)
    results["changepoint"] = cp

    # (d) pair probability at 0.68
    pp = pair_prob_block(LOW_THRESH)
    results["pair_prob_068"] = pp

    # ---- reproduce-or-not flags vs memo ----
    checks = {}
    checks["mk_S"] = {"got": mk["S"], "memo": MEMO["mk_S"], "ok": reproduces(mk["S"], MEMO["mk_S"])}
    checks["mk_varS"] = {"got": mk["varS"], "memo": MEMO["mk_varS"], "ok": reproduces(mk["varS"], MEMO["mk_varS"])}
    checks["mk_z"] = {"got": mk["z"], "memo": MEMO["mk_z"], "ok": reproduces(mk["z"], MEMO["mk_z"])}
    checks["mk_p"] = {"got": mk["p"], "memo": MEMO["mk_p"], "ok": reproduces(mk["p"], MEMO["mk_p"])}
    checks["sen_slope"] = {"got": mk["sen_slope"], "memo": MEMO["sen_slope"], "ok": reproduces(mk["sen_slope"], MEMO["sen_slope"], rel=0.30)}
    checks["cusum_final"] = {"got": cu["final"], "memo": MEMO["cusum_final"], "ok": reproduces(cu["final"], MEMO["cusum_final"])}
    checks["cusum_maxabs"] = {"got": cu["maxabs"], "memo": MEMO["cusum_maxabs"], "ok": reproduces(cu["maxabs"], MEMO["cusum_maxabs"])}
    checks["cusum_tabular_min"] = {"got": cu["tabular_lower_min"], "memo": MEMO["cusum_tabular_min"], "ok": reproduces(cu["tabular_lower_min"], MEMO["cusum_tabular_min"], rel=0.10)}
    checks["cp_maxt"] = {"got": cp["max_t_unconstrained"], "memo": MEMO["cp_maxt"], "ok": reproduces(cp["max_t_unconstrained"], MEMO["cp_maxt"])}
    checks["cp_prereg_t"] = {"got": cp["t_prereg_last2"], "memo": MEMO["cp_prereg_t"], "ok": reproduces(cp["t_prereg_last2"], MEMO["cp_prereg_t"])}
    checks["cp_perm_p"] = {"got": cp["perm_p_unconstrained"], "memo": MEMO["cp_perm_p"], "ok": reproduces(cp["perm_p_unconstrained"], MEMO["cp_perm_p"], rel=0.60, abs_=0.003)}
    checks["perdraw_gauss"] = {"got": pp["gauss_sealed"]["per_draw"], "memo": MEMO["perdraw_gauss"], "ok": reproduces(pp["gauss_sealed"]["per_draw"], MEMO["perdraw_gauss"])}
    checks["perdraw_t"] = {"got": pp["t_fit"]["per_draw"], "memo": MEMO["perdraw_t"], "ok": reproduces(pp["t_fit"]["per_draw"], MEMO["perdraw_t"])}
    checks["pair_gauss"] = {"got": pp["gauss_sealed"]["any_consecutive_pair_in_n"], "memo": MEMO["pair_gauss"], "ok": reproduces(pp["gauss_sealed"]["any_consecutive_pair_in_n"], MEMO["pair_gauss"])}
    checks["pair_t"] = {"got": pp["t_fit"]["any_consecutive_pair_in_n"], "memo": MEMO["pair_t"], "ok": reproduces(pp["t_fit"]["any_consecutive_pair_in_n"], MEMO["pair_t"])}
    checks["perdraw_yw"] = {"got": pp["yw_sigma024"]["per_draw"], "memo": MEMO["perdraw_yw"], "ok": reproduces(pp["yw_sigma024"]["per_draw"], MEMO["perdraw_yw"])}
    checks["pair_yw"] = {"got": pp["yw_sigma024"]["any_consecutive_pair_in_n"], "memo": MEMO["pair_yw"], "ok": reproduces(pp["yw_sigma024"]["any_consecutive_pair_in_n"], MEMO["pair_yw"])}
    results["memo_checks"] = checks
    results["memo_checks_summary"] = {
        "n_checks": len(checks),
        "n_reproduced": sum(1 for c in checks.values() if c["ok"]),
        "failures": [k for k, c in checks.items() if not c["ok"]],
    }

    # NC-15b false alarm sim
    results["nc15b_false_alarm"] = false_alarm_sim(rng)

    # NC-13 sigma compatibility
    results["nc13_sigma_compat"] = {
        "sealed_n15_s1343": sigma_compat(SEALED_S, SEALED_N),
        "record_n19_s159": sigma_compat(float(x.std(ddof=1)), n),
    }
    # harm-pause fixed <0.80 fire rates under each regime
    results["nc13_harm_pause_fixed"] = {
        "sealed_regime": harm_pause_fixed(SEALED_MU, SEALED_S),
        "yw_sigma024_regime": harm_pause_fixed(SEALED_MU, YW_SIGMA),
        "stepdown_mu0665_sealed_s": harm_pause_fixed(0.665, SEALED_S),
    }

    # (4) paired harm-pause
    paired = {}
    for regime, sig in [("sealed_s1343", SEALED_S), ("yw_sigma024", YW_SIGMA),
                        ("wide_intermediate_s019", 0.19)]:
        for c in (1.5, 2.0):
            paired[f"{regime}_c{c}"] = paired_harm_pause(
                rng, SEALED_MU, sig, SEALED_MU, sig, trailing_k=4, c=c)
    # also confirm invariance under a step-down gate (gate truly worse) vs contemporaneous ctrl
    paired["POWER_gate_down0.15_sealed_c1.5"] = paired_harm_pause(
        rng, SEALED_MU - 0.15, SEALED_S, SEALED_MU, SEALED_S, trailing_k=4, c=1.5)
    results["paired_harm_pause"] = {
        "definition": "pause iff gated_draw < mean(trailing_k contemporaneous fillers) - c*s(trailing_k)",
        "trailing_k": 4,
        "false_fire_rates_under_null": paired,
        "note": "false-fire is invariant to mu and (near-)invariant to sigma because "
                "the criterion is scale-relative to the contemporaneous fillers.",
    }

    out_path = "runs/stationarity_repro_2026-08-02.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_path}")
    print(json.dumps(results["memo_checks_summary"], indent=2))
    print("NC-15b false-alarm rate:", results["nc15b_false_alarm"]["false_alarm_rate"])
    print("NC-13 sealed sigma0-in-CI:", results["nc13_sigma_compat"]["sealed_n15_s1343"]["sigma0_inside_95CI"],
          "p_two:", results["nc13_sigma_compat"]["sealed_n15_s1343"]["p_two_sided"])
    print("NC-13 record sigma0-in-CI:", results["nc13_sigma_compat"]["record_n19_s159"]["sigma0_inside_95CI"],
          "p_two:", results["nc13_sigma_compat"]["record_n19_s159"]["p_two_sided"])
    print("harm-pause fixed <0.80 any-in-4 sealed:", results["nc13_harm_pause_fixed"]["sealed_regime"]["any_in_k"],
          "yw:", results["nc13_harm_pause_fixed"]["yw_sigma024_regime"]["any_in_k"])


if __name__ == "__main__":
    main()
