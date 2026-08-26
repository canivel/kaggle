#!/usr/bin/env python
"""
Stationarity re-check 2026-08-07 — watch-rule fire (two consecutive sub-0.80:
0.77 on 08-06, 0.78 on 08-07).

Adapted copy of the SEALED `scripts/stationarity_repro.py` (2026-08-02, NC-15
discharge). The sealed original is NOT modified. Same pre-registered analysis
constants (CUSUM h=4/5, k=0.5; min-segment >= 3 for the change-point scan;
permutation B=20,000; alpha=0.01 for the change-point call), applied to the
updated n=24 record ledger from `runs/lb_ground_truth.md`.

Key outputs:
  (a) Mann-Kendall trend + Sen slope (drift test the original ran)
  (b) CUSUM vs the frozen n=15 sealed control (h=4 primary / h=5 strict)
  (c) Change-point max-|Welch t| permutation test, min-segment>=3 (primary,
      per NC-15(iii)/directive 9) and unconstrained (diagnostic only)
  (d) Trailing-window z-scores for the two firing draws (0.77, 0.78) vs the
      sealed control, vs prior-record stats, and trailing-4 window
  (e) The two-consecutive-sub-0.80 event probability under the stationary fit:
      P(at least one adjacent pair both < 0.80 somewhere in a 24-draw record)
      under (i) sealed Gaussian N(0.9727, 0.1343), (ii) sealed t-predictive
      (df=14), (iii) the n=24 record-fit Gaussian. Multiple-looks aware.
  (f) sigma-compat of the n=24 record s vs the (already-REJECTED) yw 0.24 regime,
      for continuity of the NC-13 line.

Verdict rule (same as 08-02 discharge): STATIONARY iff MK p NS (>0.05), CUSUM
max|cum z| < h=4 tabular-lower not breaching h, and change-point perm p
(min-segment>=3) >= alpha 0.01. NON-STATIONARY iff change-point p < 0.01 with a
split respecting min-segment. Otherwise INCONCLUSIVE.

Run:  uv run python scripts/stationarity_recheck_20260807.py
Emits: runs/stationarity_recheck_2026-08-07.json
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import numpy as np
from scipy import stats

# ----------------------------------------------------------------------------
# PRE-REGISTERED constants — identical to sealed scripts/stationarity_repro.py
# ----------------------------------------------------------------------------
SEED = 20260807
CUSUM_H_STATE = 4
CUSUM_H_STRICT = 5
CUSUM_K = 0.5
MIN_SEGMENT = 3
PERM_B = 20000
CP_ALPHA = 0.01

# Sealed n=15 control parameters (frozen per prereg §3)
SEALED_MU = 0.9727
SEALED_S = 0.1343
SEALED_N = 15
SEALED_DF = SEALED_N - 1
# t-predictive sd: s * sqrt(1 + 1/n) — matches sealed script's 0.1387
SEALED_SD_PRED = SEALED_S * math.sqrt(1 + 1 / SEALED_N)

YW_SIGMA = 0.24
HARM_FLOOR = 0.80          # the watch-rule threshold (two consecutive < 0.80)

# ----------------------------------------------------------------------------
# The canonical n=24 record ledger (runs/lb_ground_truth.md, 2026-08-07 refresh).
# = sealed n=19 series + 0.99 (08-03), 0.97 (08-04), 1.21 (08-05),
#   0.77 (08-06), 0.78 (08-07).
# ----------------------------------------------------------------------------
LEDGER = [0.82, 0.89, 0.93, 1.02, 0.95, 1.33, 0.92, 0.93, 1.14, 0.82,
          1.05, 0.84, 1.02, 0.90, 1.03, 0.85, 1.10, 0.65, 0.68, 0.99,
          0.97, 1.21, 0.77, 0.78]


# ---------------------------------------------------------------------------
# (a) Mann-Kendall trend test + Sen slope  [verbatim from sealed script]
# ---------------------------------------------------------------------------
def mann_kendall(x):
    x = np.asarray(x, float)
    n = len(x)
    S = 0
    for i in range(n - 1):
        S += np.sum(np.sign(x[i + 1:] - x[i]))
    S = int(S)
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
    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            slopes.append((x[j] - x[i]) / (j - i))
    sen = float(np.median(slopes))
    return {"S": S, "varS": float(varS), "z": float(z), "p": float(p), "sen_slope": sen}


# ---------------------------------------------------------------------------
# (b) CUSUM vs sealed N(mu, s)  [verbatim from sealed script]
# ---------------------------------------------------------------------------
def cusum(x, mu, s, k=CUSUM_K):
    z = (np.asarray(x, float) - mu) / s
    cum = np.cumsum(z)
    final = float(cum[-1])
    maxabs = float(np.max(np.abs(cum)))
    C_lo = np.zeros(len(z))
    prev = 0.0
    for i, zi in enumerate(z):
        prev = min(0.0, prev + zi + k)
        C_lo[i] = prev
    tabular_min = float(np.min(C_lo))
    return {
        "path": [round(float(v), 4) for v in cum],
        "final": final,
        "maxabs": maxabs,
        "tabular_lower_min": tabular_min,
        "crosses_h4": bool(maxabs >= CUSUM_H_STATE),
        "crosses_h5": bool(maxabs >= CUSUM_H_STRICT),
        "tabular_crosses_h4": bool(abs(tabular_min) >= CUSUM_H_STATE),
    }


# ---------------------------------------------------------------------------
# (c) Change-point scan + permutation p  [verbatim from sealed script]
# ---------------------------------------------------------------------------
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
    n = len(x)
    best_t, best_k = 0.0, None
    for k in range(min_seg, n - min_seg + 1):
        t = welch_t(x[:k], x[k:])
        if not np.isnan(t) and abs(t) > best_t:
            best_t, best_k = abs(t), k
    return best_t, best_k


def max_split_t_nomin(x):
    n = len(x)
    best_t, best_k = 0.0, None
    for k in range(2, n - 1):
        t = welch_t(x[:k], x[k:])
        if not np.isnan(t) and abs(t) > best_t:
            best_t, best_k = abs(t), k
    return best_t, best_k


def changepoint(x, rng, min_seg=MIN_SEGMENT, B=PERM_B):
    x = np.asarray(x, float)
    n = len(x)
    obs_nomin, split_nomin = max_split_t_nomin(x)
    obs_min, split_min = max_split_t(x, min_seg)
    # pre-specified split: the two firing draws vs prior record (last-2 vs prior-22)
    t_prereg = abs(welch_t(x[:n - 2], x[n - 2:]))
    count_nomin = 0
    count_min = 0
    for _ in range(B):
        perm = rng.permutation(x)
        tm, _ = max_split_t_nomin(perm)
        if tm >= obs_nomin - 1e-12:
            count_nomin += 1
        tmin, _ = max_split_t(perm, min_seg)
        if tmin >= obs_min - 1e-12:
            count_min += 1
    return {
        "max_t_unconstrained": float(obs_nomin),
        "split_unconstrained": int(split_nomin) if split_nomin else None,
        "max_t_minseg3": float(obs_min),
        "split_minseg3": int(split_min) if split_min else None,
        "t_prereg_last2": float(t_prereg),
        "perm_p_unconstrained": (count_nomin + 1) / (B + 1),
        "perm_p_minseg3": (count_min + 1) / (B + 1),
        "perm_B": B,
    }


# ---------------------------------------------------------------------------
# (e) Consecutive-pair probability under a null  [verbatim from sealed script]
# ---------------------------------------------------------------------------
def per_draw_prob(thresh, dist):
    if dist["kind"] == "gauss":
        return float(stats.norm.cdf(thresh, dist["mu"], dist["sigma"]))
    elif dist["kind"] == "t":
        z = (thresh - dist["mu"]) / dist["sd_pred"]
        return float(stats.t.cdf(z, dist["df"]))
    raise ValueError(dist["kind"])


def prob_at_least_one_consecutive_pair(p, n):
    """P(>=1 adjacent pair of successes) in n iid Bernoulli(p) trials (DP)."""
    q = 1 - p
    s0, s1 = q, p
    for _ in range(n - 1):
        ns0 = (s0 + s1) * q
        ns1 = s0 * p
        s0, s1 = ns0, ns1
    return 1.0 - (s0 + s1)


def pair_prob_block(thresh, n, record_mu, record_s):
    dists = {
        "gauss_sealed": {"kind": "gauss", "mu": SEALED_MU, "sigma": SEALED_S},
        "t_fit_sealed": {"kind": "t", "mu": SEALED_MU, "sd_pred": SEALED_SD_PRED,
                         "df": SEALED_DF},
        "gauss_record_n24_fit": {"kind": "gauss", "mu": record_mu, "sigma": record_s},
    }
    out = {}
    for name, d in dists.items():
        pd = per_draw_prob(thresh, d)
        out[name] = {
            "per_draw": pd,
            "specific_adjacent_pair": pd * pd,
            "any_consecutive_pair_in_n": prob_at_least_one_consecutive_pair(pd, n),
            "expected_count_in_n": pd * n,
        }
    return out


# ---------------------------------------------------------------------------
# (f) sigma-compat  [verbatim from sealed script]
# ---------------------------------------------------------------------------
def sigma_compat(s, n, sigma0=YW_SIGMA):
    df = n - 1
    stat = df * s * s / (sigma0 * sigma0)
    p_lower = float(stats.chi2.cdf(stat, df))
    p_upper = float(stats.chi2.sf(stat, df))
    p_two = 2 * min(p_lower, p_upper)
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


def main():
    rng = np.random.default_rng(SEED)
    x = np.array(LEDGER, float)
    n = len(x)
    record_mu = float(x.mean())
    record_s = float(x.std(ddof=1))

    results = {
        "meta": {
            "generated": datetime.now(timezone.utc).isoformat(),
            "seed": SEED,
            "trigger": "watch-rule FIRED: two consecutive sub-0.80 filler draws "
                       "(08-06: 0.77, 08-07: 0.78)",
            "ledger": LEDGER,
            "n": n,
            "sealed_original": "scripts/stationarity_repro.py (2026-08-02, unmodified)",
            "prereg_constants": {
                "CUSUM_H_STATE": CUSUM_H_STATE,
                "CUSUM_H_STRICT": CUSUM_H_STRICT,
                "CUSUM_K": CUSUM_K,
                "MIN_SEGMENT": MIN_SEGMENT,
                "PERM_B": PERM_B,
                "CP_ALPHA": CP_ALPHA,
            },
            "sealed_control": {"mu": SEALED_MU, "s": SEALED_S, "n": SEALED_N},
        }
    }

    # ledger stats + trailing-window z-scores (d)
    prior23 = x[:-1]
    prior22 = x[:-2]
    results["ledger_stats"] = {
        "mean_n24": record_mu,
        "s_n24": record_s,
        "mean_prior22": float(prior22.mean()),
        "s_prior22": float(prior22.std(ddof=1)),
        "trailing4_mean": float(x[-4:].mean()),
        "z_077_vs_sealed": float((0.77 - SEALED_MU) / SEALED_S),
        "z_078_vs_sealed": float((0.78 - SEALED_MU) / SEALED_S),
        "z_077_vs_prior22": float((0.77 - prior22.mean()) / prior22.std(ddof=1)),
        "z_078_vs_prior23": float((0.78 - prior23.mean()) / prior23.std(ddof=1)),
        "mean_last2": float(x[-2:].mean()),
        "z_last2_mean_vs_sealed": float(
            (x[-2:].mean() - SEALED_MU) / (SEALED_S / math.sqrt(2))),
    }

    # (a) Mann-Kendall
    mk = mann_kendall(x)
    results["mann_kendall"] = mk

    # (b) CUSUM vs sealed control
    cu = cusum(x, SEALED_MU, SEALED_S)
    results["cusum"] = cu

    # (c) change-point
    cp = changepoint(x, rng)
    results["changepoint"] = cp

    # (e) pair probability at the 0.80 watch-rule floor
    pp = pair_prob_block(HARM_FLOOR, n, record_mu, record_s)
    results["pair_prob_080"] = pp

    # (f) sigma compat (continuity of NC-13; yw sigma=0.24 already REJECTED)
    results["sigma_compat_yw024"] = {
        "sealed_n15_s1343": sigma_compat(SEALED_S, SEALED_N),
        "record_n24": sigma_compat(record_s, n),
    }

    # ---- verdict per the 08-02 discharge rule ----
    # NOTE on the CUSUM criterion: the pre-registered alarm statistic is the
    # TABULAR one-sided lower CUSUM with slack k=0.5 against h=4 (the sealed
    # 08-02 check ruled STATIONARY with raw standardized-cumsum maxabs 4.55 > 4
    # but tabular_lower_min -3.58, |.| < 4 -> no alarm). The raw cumulative sum
    # is NOT a valid fixed-threshold alarm: it accumulates ~ -0.23 sigma per
    # draw mechanically because the record mean (0.9413) sits slightly below
    # the frozen sealed control mean (0.9727), a feature already present and
    # adjudicated at the 08-02 discharge. Raw maxabs is kept as a diagnostic.
    mk_ns = mk["p"] > 0.05
    cusum_ok = not cu["tabular_crosses_h4"]
    cp_ns = cp["perm_p_minseg3"] >= CP_ALPHA
    if mk_ns and cusum_ok and cp_ns:
        verdict = "STATIONARY"
    elif not cp_ns:
        verdict = "NON-STATIONARY"
    else:
        verdict = "INCONCLUSIVE"
    results["verdict"] = {
        "verdict": verdict,
        "criteria": {
            "mk_p_gt_0.05": bool(mk_ns),
            "tabular_cusum_below_h4": bool(cusum_ok),
            "changepoint_minseg3_p_ge_0.01": bool(cp_ns),
        },
        "cusum_note": "alarm statistic = tabular lower CUSUM (k=0.5, h=4), per "
                      "08-02 precedent; raw cumsum maxabs is diagnostic only "
                      "(drifts ~-0.23 sigma/draw from the frozen-control mean "
                      "offset, adjudicated stationary at 08-02 with maxabs 4.55)",
        "headline": {
            "mk_p": mk["p"],
            "cusum_maxabs": cu["maxabs"],
            "cusum_tabular_lower_min": cu["tabular_lower_min"],
            "cp_perm_p_minseg3": cp["perm_p_minseg3"],
            "cp_perm_p_unconstrained_diagnostic": cp["perm_p_unconstrained"],
            "pair_prob_any_in_24_sealed_gauss":
                pp["gauss_sealed"]["any_consecutive_pair_in_n"],
            "pair_prob_any_in_24_sealed_t":
                pp["t_fit_sealed"]["any_consecutive_pair_in_n"],
            "pair_prob_any_in_24_record_fit":
                pp["gauss_record_n24_fit"]["any_consecutive_pair_in_n"],
        },
    }

    out_path = "runs/stationarity_recheck_2026-08-07.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_path}")
    print(json.dumps(results["verdict"], indent=2))
    print("ledger_stats:", json.dumps(results["ledger_stats"], indent=2))
    print("changepoint:", json.dumps(cp, indent=2))
    print("pair_prob_080:", json.dumps(pp, indent=2))


if __name__ == "__main__":
    main()
