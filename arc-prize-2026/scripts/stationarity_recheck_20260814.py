#!/usr/bin/env python
"""
Stationarity re-check 2026-08-14 — watch-rule FIRED AGAIN (fresh two-consecutive
sub-0.80 pair: 0.78 on 08-13, 0.70 on 08-14).

Adapted copy of `scripts/stationarity_recheck_20260807.py`, which was itself an
adapted copy of the SEALED `scripts/stationarity_repro.py` (2026-08-02, NC-15
discharge). NEITHER predecessor is modified. All pre-registered analysis constants
are carried over VERBATIM (CUSUM h=4/5, k=0.5; min-segment >= 3 for the change-point
scan; permutation B=20,000; alpha=0.01 for the change-point call).

Applied to the n=31 record ledger from `runs/ledger.json` (re-derived from the
Kaggle API by scripts/ledger.py on 2026-08-14).

WATCH-RULE BOOKKEEPING (the rule as WRITTEN, not as remembered):
  * PRIMARY (stationarity_2026-08-02.md GUARD 2 / recheck memo 08-07): the rule is
    "two CONSECUTIVE sub-0.80 draws" -> fire this battery.
  * ESCALATION: "if a THIRD CONSECUTIVE sub-0.80 lands: promote to NON-STATIONARY,
    re-baseline the control on a fresh in-regime window (n>=8), re-derive the
    promote threshold."
  * The 08-06/08-07 pair (0.77, 0.78) fired and RESOLVED-STATIONARY (p=0.757).
    0.87 on 08-08 BROKE the streak, so the escalation counter RESET (ITERATION_LOG
    08-08: "re-arms only on a future sub-0.80 pair").
  * 08-13 (0.78) + 08-14 (0.70) is therefore a FRESH PRIMARY FIRE of the
    two-consecutive rule, NOT the third-consecutive escalation.

Also carries the 08-12 order-statistics (Blom E[max@k]) reframe forward onto the
updated ledger, including the like-for-like calibration check that the 08-13 log
records as previously botched (mismatched k).

Run:  uv run python scripts/stationarity_recheck_20260814.py
Emits: runs/stationarity_recheck_2026-08-14.json
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import numpy as np
from scipy import stats

# ----------------------------------------------------------------------------
# PRE-REGISTERED constants — identical to the sealed 08-02 script and the 08-07 copy
# ----------------------------------------------------------------------------
SEED = 20260814
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
SEALED_SD_PRED = SEALED_S * math.sqrt(1 + 1 / SEALED_N)

YW_SIGMA = 0.24
HARM_FLOOR = 0.80          # the watch-rule threshold

# ----------------------------------------------------------------------------
# The canonical n=31 record ledger, oldest-first.
# = the 08-07 n=24 series (verbatim, verified elementwise below) + 0.87 (08-08),
#   0.89 (08-09), 1.05 (08-10), 1.09 (08-11), 1.07 (08-12), 0.78 (08-13), 0.70 (08-14).
# ----------------------------------------------------------------------------
LEDGER_N24_0807 = [0.82, 0.89, 0.93, 1.02, 0.95, 1.33, 0.92, 0.93, 1.14, 0.82,
                   1.05, 0.84, 1.02, 0.90, 1.03, 0.85, 1.10, 0.65, 0.68, 0.99,
                   0.97, 1.21, 0.77, 0.78]
LEDGER = LEDGER_N24_0807 + [0.87, 0.89, 1.05, 1.09, 1.07, 0.78, 0.70]

# Order-statistics frame (08-12 reframe): draws remaining to the Nov 2 close.
K_REMAINING = 80
K_REMAINING_ALT = 110
GOLD_LINE = 1.58
PRIZE_LINE = 1.64
OBSERVED_MAX = 1.33


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
    C_lo = np.zeros(len(z))
    prev = 0.0
    for i, zi in enumerate(z):
        prev = min(0.0, prev + zi + k)
        C_lo[i] = prev
    tabular_min = float(np.min(C_lo))
    return {
        "path": [round(float(v), 4) for v in cum],
        "final": float(cum[-1]),
        "maxabs": float(np.max(np.abs(cum))),
        "tabular_lower_min": tabular_min,
        "tabular_lower_argmin": int(np.argmin(C_lo)),
        "crosses_h4": bool(np.max(np.abs(cum)) >= CUSUM_H_STATE),
        "crosses_h5": bool(np.max(np.abs(cum)) >= CUSUM_H_STRICT),
        "tabular_crosses_h4": bool(abs(tabular_min) >= CUSUM_H_STATE),
        "tabular_crosses_h5": bool(abs(tabular_min) >= CUSUM_H_STRICT),
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
    q = 1 - p
    s0, s1 = q, p
    for _ in range(n - 1):
        ns0 = (s0 + s1) * q
        ns1 = s0 * p
        s0, s1 = ns0, ns1
    return 1.0 - (s0 + s1)


def prob_at_least_one_run_of_m(p, n, m):
    """P(>=1 run of m consecutive successes) in n iid Bernoulli(p). DP over run length."""
    # state[j] = prob mass with current trailing run length j (j=0..m-1), no run of m yet
    state = [0.0] * m
    state[0] = 1.0
    hit = 0.0
    for _ in range(n):
        new = [0.0] * m
        for j in range(m):
            if state[j] == 0.0:
                continue
            new[0] += state[j] * (1 - p)          # failure resets run
            if j + 1 >= m:
                hit += state[j] * p               # completes a run of m
            else:
                new[j + 1] += state[j] * p
        state = new
    return hit


def pair_prob_block(thresh, n, record_mu, record_s):
    dists = {
        "gauss_sealed": {"kind": "gauss", "mu": SEALED_MU, "sigma": SEALED_S},
        "t_fit_sealed": {"kind": "t", "mu": SEALED_MU, "sd_pred": SEALED_SD_PRED,
                         "df": SEALED_DF},
        "gauss_record_fit": {"kind": "gauss", "mu": record_mu, "sigma": record_s},
    }
    out = {}
    for name, d in dists.items():
        pd = per_draw_prob(thresh, d)
        out[name] = {
            "per_draw": pd,
            "specific_adjacent_pair": pd * pd,
            "any_consecutive_pair_in_n": prob_at_least_one_consecutive_pair(pd, n),
            "any_run_of_3_in_n": prob_at_least_one_run_of_m(pd, n, 3),
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


# ---------------------------------------------------------------------------
# (g) Order statistics — Blom E[max of k] under a normal fit.
#     E[max@k] ~= mu + sigma * Phi^-1( (k - 0.375) / (k + 0.25) )
# ---------------------------------------------------------------------------
def blom_emax(mu, sigma, k):
    return float(mu + sigma * stats.norm.ppf((k - 0.375) / (k + 0.25)))


def p_max_exceeds(mu, sigma, k, thresh):
    """P(max of k iid N(mu,sigma) >= thresh)."""
    return float(1.0 - stats.norm.cdf(thresh, mu, sigma) ** k)


def main():
    rng = np.random.default_rng(SEED)
    x = np.array(LEDGER, float)
    n = len(x)
    record_mu = float(x.mean())
    record_s = float(x.std(ddof=1))

    # --- integrity: the 08-07 prefix must be preserved verbatim ---
    prefix_ok = LEDGER[:24] == LEDGER_N24_0807

    results = {
        "meta": {
            "generated": datetime.now(timezone.utc).isoformat(),
            "seed": SEED,
            "trigger": "watch-rule FIRED (FRESH primary fire): two consecutive "
                       "sub-0.80 filler draws (08-13: 0.78, 08-14: 0.70)",
            "ledger": LEDGER,
            "n": n,
            "predecessors_unmodified": [
                "scripts/stationarity_repro.py (2026-08-02, sealed)",
                "scripts/stationarity_recheck_20260807.py (2026-08-07)",
            ],
            "prefix_matches_0807_n24": bool(prefix_ok),
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

    # --- watch-rule bookkeeping, computed not remembered ---
    sub80_flags = [bool(v < HARM_FLOOR) for v in LEDGER]
    # longest trailing run of sub-0.80
    trailing_run = 0
    for f in reversed(sub80_flags):
        if f:
            trailing_run += 1
        else:
            break
    # all maximal runs
    runs = []
    cur = 0
    for i, f in enumerate(sub80_flags):
        if f:
            cur += 1
        else:
            if cur:
                runs.append({"end_index_0based": i - 1, "length": cur})
            cur = 0
    if cur:
        runs.append({"end_index_0based": len(sub80_flags) - 1, "length": cur})
    sorted_vals = sorted(LEDGER)
    results["watch_rule"] = {
        "threshold": HARM_FLOOR,
        "rule_primary": "two CONSECUTIVE sub-0.80 draws -> fire this battery",
        "rule_escalation": "a THIRD CONSECUTIVE sub-0.80 -> promote NON-STATIONARY, "
                           "re-baseline control on fresh in-regime window (n>=8)",
        "sub80_count_total": int(sum(sub80_flags)),
        "sub80_values": [v for v in LEDGER if v < HARM_FLOOR],
        "all_sub80_runs": runs,
        "trailing_consecutive_sub80": trailing_run,
        "escalation_triggered": bool(trailing_run >= 3),
        "fire_type": ("ESCALATION (3rd consecutive)" if trailing_run >= 3
                      else "PRIMARY (fresh 2-consecutive pair)" if trailing_run == 2
                      else "NOT FIRED"),
        "latest_draw": LEDGER[-1],
        "latest_is_record_min": bool(LEDGER[-1] == min(LEDGER)),
        "record_min": min(LEDGER),
        "latest_rank_ascending_1based": int(sorted_vals.index(LEDGER[-1]) + 1),
        "three_lowest": sorted_vals[:3],
    }

    # ledger stats + trailing-window z-scores (d)
    prior30 = x[:-1]
    prior29 = x[:-2]
    results["ledger_stats"] = {
        "mean_n31": record_mu,
        "s_n31": record_s,
        "mean_prior29": float(prior29.mean()),
        "s_prior29": float(prior29.std(ddof=1)),
        "trailing4_mean": float(x[-4:].mean()),
        "trailing4_prev_mean": float(x[-5:-1].mean()),
        "z_070_vs_sealed": float((0.70 - SEALED_MU) / SEALED_S),
        "z_078_vs_sealed": float((0.78 - SEALED_MU) / SEALED_S),
        "z_070_vs_prior30": float((0.70 - prior30.mean()) / prior30.std(ddof=1)),
        "z_078_vs_prior29": float((0.78 - prior29.mean()) / prior29.std(ddof=1)),
        "mean_last2": float(x[-2:].mean()),
        "z_last2_mean_vs_sealed": float(
            (x[-2:].mean() - SEALED_MU) / (SEALED_S / math.sqrt(2))),
        "z_last2_mean_vs_record": float(
            (x[-2:].mean() - record_mu) / (record_s / math.sqrt(2))),
        "max": float(x.max()),
        "min": float(x.min()),
    }

    mk = mann_kendall(x)
    results["mann_kendall"] = mk

    cu = cusum(x, SEALED_MU, SEALED_S)
    results["cusum"] = cu

    cp = changepoint(x, rng)
    results["changepoint"] = cp

    pp = pair_prob_block(HARM_FLOOR, n, record_mu, record_s)
    results["pair_prob_080"] = pp

    results["sigma_compat_yw024"] = {
        "sealed_n15_s1343": sigma_compat(SEALED_S, SEALED_N),
        "record_n31": sigma_compat(record_s, n),
    }

    # ---- verdict per the 08-02 discharge rule (identical criteria) ----
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
        "headline": {
            "mk_p": mk["p"],
            "sen_slope": mk["sen_slope"],
            "cusum_maxabs_diagnostic": cu["maxabs"],
            "cusum_tabular_lower_min": cu["tabular_lower_min"],
            "cp_perm_p_minseg3": cp["perm_p_minseg3"],
            "cp_max_t_minseg3": cp["max_t_minseg3"],
            "cp_split_minseg3": cp["split_minseg3"],
            "cp_perm_p_unconstrained_diagnostic": cp["perm_p_unconstrained"],
            "cp_max_t_unconstrained_diagnostic": cp["max_t_unconstrained"],
            "cp_split_unconstrained_diagnostic": cp["split_unconstrained"],
            "pair_prob_any_in_n_sealed_gauss":
                pp["gauss_sealed"]["any_consecutive_pair_in_n"],
            "pair_prob_any_in_n_sealed_t":
                pp["t_fit_sealed"]["any_consecutive_pair_in_n"],
            "pair_prob_any_in_n_record_fit":
                pp["gauss_record_fit"]["any_consecutive_pair_in_n"],
        },
    }

    # ---- (g) order statistics / E[max@k] reframe on the UPDATED ledger ----
    # Like-for-like calibration: E[max@n] vs the OBSERVED best-of-n. The 08-13 log
    # records the earlier claim as WRONG because it compared E[max@80] to best-of-29.
    cal = {}
    for label, (mu_, s_, k_) in {
        "prior_n29_params_at_k29": (0.9503, 0.1513, 29),
        "current_n31_params_at_k31": (record_mu, record_s, n),
    }.items():
        e = blom_emax(mu_, s_, k_)
        cal[label] = {
            "mu": mu_, "sigma": s_, "k": k_,
            "E[max@k]": e,
            "observed_max": OBSERVED_MAX,
            "under_prediction": round(OBSERVED_MAX - e, 4),
        }
    proj = {}
    for mu_label, mu_ in [("current_0.9368", record_mu),
                          ("efficiency_low_1.26", 1.26),
                          ("efficiency_mid_1.31", 1.31),
                          ("efficiency_high_1.36", 1.36)]:
        row = {}
        for k_ in (K_REMAINING, K_REMAINING_ALT):
            e = blom_emax(mu_, record_s, k_)
            row[f"E[max@{k_}]"] = e
            row[f"clears_gold_{GOLD_LINE}@{k_}"] = bool(e >= GOLD_LINE)
            row[f"clears_prize_{PRIZE_LINE}@{k_}"] = bool(e >= PRIZE_LINE)
            row[f"P(max@{k_}>=gold)"] = p_max_exceeds(mu_, record_s, k_, GOLD_LINE)
            row[f"P(max@{k_}>=prize)"] = p_max_exceeds(mu_, record_s, k_, PRIZE_LINE)
        proj[mu_label] = row
    # sensitivity of the CURRENT-mean projection to the ledger update
    delta = {}
    for k_ in (K_REMAINING, K_REMAINING_ALT):
        before = blom_emax(0.9503, 0.1513, k_)
        after = blom_emax(record_mu, record_s, k_)
        delta[f"k={k_}"] = {
            "E[max]_at_n29_params": before,
            "E[max]_at_n31_params": after,
            "delta": round(after - before, 4),
        }
    results["order_statistics"] = {
        "method": "Blom: E[max@k] = mu + sigma * Phi^-1((k-0.375)/(k+0.25))",
        "gold_line": GOLD_LINE,
        "prize_line": PRIZE_LINE,
        "observed_max_banked": OBSERVED_MAX,
        "calibration_like_for_like": cal,
        "projections": proj,
        "effect_of_the_0.70_update": delta,
        "caveat": "Blom-normal UNDER-predicts under right skew; at k=n it already "
                  "under-predicts our own observed max. Direction is conservative. "
                  "Also: the banked 1.33 is already realised, so campaign max is "
                  "max(1.33, max of remaining k) -- these E[max@k] figures are for "
                  "the remaining draws only.",
    }

    out_path = "runs/stationarity_recheck_2026-08-14.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_path}\n")
    print("prefix_matches_0807_n24:", prefix_ok)
    print("\n=== WATCH RULE ===")
    print(json.dumps(results["watch_rule"], indent=2))
    print("\n=== VERDICT ===")
    print(json.dumps(results["verdict"], indent=2))
    print("\n=== LEDGER STATS ===")
    print(json.dumps(results["ledger_stats"], indent=2))
    print("\n=== PAIR PROB @0.80 ===")
    print(json.dumps(pp, indent=2))
    print("\n=== SIGMA COMPAT ===")
    print(json.dumps(results["sigma_compat_yw024"]["record_n31"], indent=2))
    print("\n=== ORDER STATISTICS ===")
    print(json.dumps(results["order_statistics"], indent=2))


if __name__ == "__main__":
    main()
