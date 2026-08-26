#!/usr/bin/env python
"""
K3' FALLOUT — 2026-08-10.

Four deliverables, all offline, $0, zero Kaggle pushes, read-only w.r.t. every
existing artifact:

  D1  Re-screen A22 under K3' (as sealed in learnings/sweeps/gate_recalibration_2026-08-09.md
      section 3.3) using the m>=3 same-config baseline families that exist on disk,
      plus a config-legality audit of WHICH family is the legal comparator.
  D2  Type-II / monotonicity recalibration of K3' against runs/null10
      (R25 methodology objection N1, filed FATAL).
  D3  Numbers needed by duck_eval/SCREEN_PROTOCOL.md.
  D4  Warpack-specific null: what is estimable for free from disk, and what more costs.

Writes runs/k3prime_fallout_2026-08-10.json.
Run:  uv run python duck_eval/r24_prep/k3prime_fallout.py
"""
import glob
import hashlib
import itertools
import json
import math
import os
import re
import statistics
import sys
from datetime import datetime, timezone

import numpy as np
from scipy import stats

ROOT = r"F:\kaggle\arc-prize-2026"
NULL10 = os.path.join(ROOT, "runs", "null10")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from phase1_gate import signflip_p_exact  # noqa: E402

SEEDS = list(range(101, 111))
OLD_K3_MEAN = -0.128
OLD_K3_WORST = -1.0
NGAMES = 25


# ------------------------------------------------------------------ loaders
def load_bench_lc(path):
    """{game_prefix: levels_completed} from a benchmark.json (same keying as the screens)."""
    b = json.load(open(path))
    return {r["game_id"].split("-")[0]: int(r["levels_completed"]) for r in b["game_runs"]}, b


def load_null10():
    lc = {}
    for s in SEEDS:
        d = json.load(open(os.path.join(NULL10, "vanilla_seed%d.json" % s)))
        lc[s] = {g.split("-")[0]: int(r["levels_completed"]) for g, r in d["games"].items()}
    return lc


def sd(xs):
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def t95(df):
    return float(stats.t.ppf(0.95, df))


def quantile(vals, q):
    return float(np.quantile(np.asarray(vals, dtype=float), q, method="linear"))


# ------------------------------------------------------------------ 0. inventory
def inventory():
    """Every 25-game duck-harness run on disk, grouped into same-`label` families,
    with byte-level duplicate detection by (start_time, per-game lc)."""
    rows = []
    for p in sorted(glob.glob(os.path.join(ROOT, "runs", "**", "benchmark.json"), recursive=True)):
        try:
            b = json.load(open(p))
        except Exception:
            continue
        gr = b.get("game_runs")
        if not gr or len(gr) != NGAMES or b.get("solver_label") != "duck-harness":
            continue
        lc = {r["game_id"].split("-")[0]: int(r["levels_completed"]) for r in gr}
        rel = os.path.relpath(p, ROOT).replace("\\", "/")
        rows.append(dict(
            path=rel, label=b.get("label"), start=b.get("start_time"), end=b.get("end_time"),
            lc_total=sum(lc.values()), lc=lc,
            prefix_set_md5=hashlib.md5(",".join(sorted(lc)).encode()).hexdigest()[:8],
            instance_id_md5=hashlib.md5(
                ",".join(sorted(r["game_id"] for r in gr)).encode()).hexdigest()[:8],
        ))
    # duplicate detection
    seen, dups = {}, []
    for r in rows:
        key = (r["start"], r["lc_total"], tuple(sorted(r["lc"].items())))
        if key in seen:
            r["duplicate_of"] = seen[key]
            dups.append((r["path"], seen[key]))
        else:
            seen[key] = r["path"]
            r["duplicate_of"] = None
    fams = {}
    for r in rows:
        if r["duplicate_of"]:
            continue
        # seed-suffixed labels (duck-null-seed101..110, duck-v2-seed201..203,
        # duck-phase1-seedN) are ONE config family with the seed in the label.
        lab = re.sub(r"seed\d+$", "seedNNN", r["label"])
        r["family"] = lab
        fams.setdefault(lab, []).append(r)
    return rows, fams, dups


def family_stats(runs):
    """Run-level mean-lc-per-game stats for one same-label family."""
    means = [r["lc_total"] / float(NGAMES) for r in runs]
    m = len(means)
    s = sd(means)
    ss = sum((x - statistics.fmean(means)) ** 2 for x in means) if m > 1 else 0.0
    return dict(m=m, runs=[r["path"] for r in runs], lc_totals=[r["lc_total"] for r in runs],
                mean_lc_per_game=[round(x, 4) for x in means],
                s_base=s, ss=ss, df=m - 1)


# ------------------------------------------------------------------ K3' as sealed
def k3prime_sealed_line(s_base, m, alpha=0.05):
    """PASS iff mean_dlc >= line.  line = -t(1-alpha, df=m-1) * s_base * sqrt(1+1/m).
    Undefined at m=1 (df=0) -> sealed fixed fallback."""
    if m < 2:
        return None
    tq = float(stats.t.ppf(1 - alpha, m - 1))
    return -tq * s_base * math.sqrt(1.0 + 1.0 / m)


SEALED_FALLBACK_M1 = -0.200        # measured type-I 2.2% on null10 (gate_recalibration 3.1)
SEALED_FALLBACK_M3 = -0.190        # empirical 5th pct of the vanilla null at m=3/4
SEALED_FALLBACK_A10 = -0.160       # alpha=0.10, measured 7.8%


# ------------------------------------------------------------------ D1
def d1(rows, fams):
    out = {}

    # --- config legality audit -------------------------------------------------
    def banner_scan(dirpath):
        hits = {}
        for lg in glob.glob(os.path.join(dirpath, "*.log")):
            if "vllm" in os.path.basename(lg):
                continue
            txt = open(lg, "rb").read().decode("utf-8", "replace")
            hits[os.path.basename(lg)] = dict(
                has_no_warpack_banner=("NO warpack" in txt),
                has_warpack_banking=("warpack: banking" in txt),
                n_COMPACTION_lines=txt.count("COMPACTION "),
            )
        return hits

    audit = {}
    for name, d in [
        ("a22_v1", "runs/a22_compaction_v1"),
        ("a22_v2", "runs/a22_v2_seed1"),
        ("a22_v2_1", "runs/kernel_pulls/a22_v2_1"),
        ("war_eval_v1", "runs/kernel_pulls/war_eval_v1"),
        ("war_eval_v2", "runs/kernel_pulls/war_eval_v2"),
        ("war_eval_v3", "runs/kernel_pulls/war_eval_v3"),
        ("w0_eval_s1", "runs/kernel_pulls/w0_eval_s1"),
        ("w0_cont_eval", "runs/kernel_pulls/w0_cont_eval"),
    ]:
        p = os.path.join(ROOT, d)
        b = json.load(open(os.path.join(p, "benchmark.json")))
        gs = os.path.join(p, "git_status.txt")
        audit[name] = dict(
            dir=d, label=b["label"],
            git_status_md5=hashlib.md5(open(gs, "rb").read()).hexdigest() if os.path.exists(gs) else None,
            log_banners=banner_scan(p),
        )
    gs_hashes = {k: v["git_status_md5"] for k, v in audit.items()}
    out["config_audit"] = dict(
        runs=audit,
        git_status_is_identical_across_all_arms=len(set(gs_hashes.values())) == 1,
        git_status_note=("git_status.txt is byte-identical across ALL of these arms including "
                         "arms of demonstrably DIFFERENT configs (warpack vs no-warpack). It "
                         "records workstation repo commits, not the arm config, so 'byte-identical "
                         "git_status.txt' (r24 minutes 3.3a) is NOT evidence of same-config."),
        verdict=("A22 arms banner 'NO warpack/ledger-graft/sentinel' and carry label "
                 "...-continuation-v1; war_eval_v1/v2/v3 banner 'warpack: banking' and carry "
                 "label ...-warpack-v1. The A22 arms and their as-screened baseline are "
                 "DIFFERENT CONFIGS."),
    )

    # --- the two candidate baseline families -----------------------------------
    warpack = fams.get("duck-harness-kaggle-warpack-v1", [])
    cont = fams.get("duck-harness-kaggle-continuation-v1", [])
    out["baseline_families"] = dict(
        warpack_v1=family_stats(warpack),
        continuation_v1=family_stats(cont),
    )

    # --- arms -------------------------------------------------------------------
    ARMS = [
        ("a22_v1", "runs/a22_compaction_v1/benchmark.json",
         "runs/a22_compaction_v1/m1m2m3_screen.json"),
        ("a22_v2", "runs/a22_v2_seed1/benchmark.json",
         "runs/a22_v2_seed1/m1m2m3_screen.json"),
        ("a22_v2_1", "runs/kernel_pulls/a22_v2_1/benchmark.json",
         "runs/kernel_pulls/a22_v2_1/m1m2m3_screen.json"),
    ]
    arm_lc = {}
    old_verdicts = {}
    for name, bpath, spath in ARMS:
        lc, _ = load_bench_lc(os.path.join(ROOT, bpath))
        arm_lc[name] = lc
        sc = json.load(open(os.path.join(ROOT, spath)))
        old_verdicts[name] = dict(
            source=spath, baseline=sc.get("baseline"),
            mean_dlc=sc["M1"]["mean_dlc"], worst_dlc=sc["M1"]["worst_dlc"],
            signflip_p_exact=sc["M1"]["signflip_p_exact"], verdict=sc["M1"]["verdict"],
            old_K3_mean_leg=">= %.3f" % OLD_K3_MEAN, old_K3_worst_leg=">= %.1f" % OLD_K3_WORST,
        )
        # cross-check the on-disk screen against a recompute from raw benchmarks
        war1, _ = load_bench_lc(os.path.join(ROOT, "runs/kernel_pulls/war_eval_v1/benchmark.json"))
        dl = [lc[g] - war1[g] for g in sorted(lc)]
        old_verdicts[name]["recomputed_mean_dlc_vs_war_v1"] = statistics.fmean(dl)
        old_verdicts[name]["recompute_matches_screen"] = (
            abs(statistics.fmean(dl) - sc["M1"]["mean_dlc"]) < 1e-9)

    def screen(arm_name, base_runs, alpha=0.05):
        lc = arm_lc[arm_name]
        games = sorted(lc)
        base = {g: statistics.fmean([r["lc"][g] for r in base_runs]) for g in games}
        dl = [lc[g] - base[g] for g in games]
        m = len(base_runs)
        s_base = sd([r["lc_total"] / float(NGAMES) for r in base_runs])
        line = k3prime_sealed_line(s_base, m, alpha)
        nz = [v for v in dl if abs(v) > 1e-12]
        return dict(
            arm=arm_name, baseline_family=[r["path"] for r in base_runs], m=m,
            s_base=s_base, df=m - 1, t95=t95(m - 1) if m > 1 else None,
            inflation_sqrt_1_plus_1_over_m=math.sqrt(1 + 1.0 / m),
            k3prime_line=line,
            mean_dlc=statistics.fmean(dl), worst_dlc=min(dl),
            n_le_m2=sum(1 for v in dl if v <= -2 + 1e-12),
            arm_lc_total=sum(lc.values()),
            base_lc_total=sum(base.values()),
            verdict=("PASS" if (line is not None and statistics.fmean(dl) >= line)
                     else ("FAIL" if line is not None else "NOT-SCREENABLE(m<2)")),
            margin=None if line is None else statistics.fmean(dl) - line,
            n_nonzero_games=len(nz),
        )

    res = {}
    for name in arm_lc:
        res[name] = dict(
            old_K3=old_verdicts[name],
            vs_warpack_m3_alpha05=screen(name, warpack, 0.05),
            vs_warpack_m3_alpha10=screen(name, warpack, 0.10),
            vs_continuation_m2_alpha05=screen(name, cont, 0.05),
            vs_war_v1_single_m1_fallback=None,
        )
        # m=1 sealed fallback against the as-screened single baseline
        war1 = [r for r in warpack if r["path"].endswith("war_eval_v1/benchmark.json")]
        lc = arm_lc[name]
        dl = [lc[g] - war1[0]["lc"][g] for g in sorted(lc)]
        mn = statistics.fmean(dl)
        res[name]["vs_war_v1_single_m1_fallback"] = dict(
            baseline="runs/kernel_pulls/war_eval_v1", m=1,
            line=SEALED_FALLBACK_M1, mean_dlc=mn,
            verdict="PASS" if mn >= SEALED_FALLBACK_M1 else "FAIL",
            note="sealed K3' m=1 fixed fallback, measured type-I 2.2% on null10")
    out["arms"] = res

    # --- legality citations ------------------------------------------------------
    seal = json.load(open(os.path.join(ROOT, "runs", "sealed", "r17_thresholds.json")))
    out["legality"] = dict(
        r16_seal=dict(
            source="learnings/panel/r16_circulation.md L417-418 (section 11)",
            claim="control band = 4-run set {war_eval_v1,v2,v3,w0_eval_s1}, n=4, LEGAL"),
        r17_seal=dict(
            source="runs/sealed/r17_thresholds.json -> thresholds.control_band",
            sealed_at=seal["sealed_at"],
            verdict=seal["thresholds"]["control_band"]["verdict"],
            fallback=seal["thresholds"]["control_band"]["fallback"],
            gpu_h=seal["thresholds"]["control_band"]["gpu_h"]),
        r17_reanchor=seal["thresholds"]["legal_control_reanchor"],
        conflict=("R16 section 11 sealed the n=4 warpack-inclusive band as LEGAL on 07-19/20; "
                  "R17 sealed it ILLEGAL on 07-22 (config-diff exceeds the {(f)} envelope) and "
                  "prescribed 2 fresh W0 seeds as the fix. The 08-09 recalibration cited the "
                  "R16 band without noting the R17 override."),
    )
    return out


# ------------------------------------------------------------------ D2
def d2(null_lc):
    games = sorted(null_lc[SEEDS[0]])
    out = {}

    # --- reviewer's arithmetic, reproduced ---------------------------------------
    pair_means = []
    for i, j in itertools.permutations(SEEDS, 2):
        pair_means.append(statistics.fmean([null_lc[i][g] - null_lc[j][g] for g in games]))
    pair_sd = sd(pair_means)
    run_means = [sum(null_lc[s].values()) / float(NGAMES) for s in SEEDS]
    s_run_direct = sd(run_means)

    subs = {}
    for m in (1, 2, 3, 4, 5, 9, 10):
        subs[str(m)] = dict(
            m=m,
            line_from_pair_sd_over_sqrt2=(None if m < 2 else
                                          k3prime_sealed_line(pair_sd / math.sqrt(2), m)),
            line_from_direct_run_sd=(None if m < 2 else k3prime_sealed_line(s_run_direct, m)),
            t95_df_m_minus_1=(None if m < 2 else t95(m - 1)),
            sqrt_1_plus_1_over_m=math.sqrt(1 + 1.0 / m),
        )
    out["reviewer_substitution"] = dict(
        null10_pair_sd_of_mean_dlc=pair_sd,
        s_base_implied_pair_sd_over_sqrt2=pair_sd / math.sqrt(2),
        s_base_direct_run_level_sd=s_run_direct,
        sealed_fallbacks=dict(m1=SEALED_FALLBACK_M1, m_ge_3=SEALED_FALLBACK_M3,
                              alpha_0_10=SEALED_FALLBACK_A10),
        lines_by_m=subs,
        reviewer_claimed=dict(s_base=0.0865, m3=-0.292, m5=-0.202, m10=-0.166),
        reproduces=None,  # filled below
    )
    r = out["reviewer_substitution"]
    r["reproduces"] = dict(
        m3=abs(subs["3"]["line_from_pair_sd_over_sqrt2"] - (-0.292)) < 0.002,
        m5=abs(subs["5"]["line_from_pair_sd_over_sqrt2"] - (-0.202)) < 0.002,
        m10=abs(subs["10"]["line_from_pair_sd_over_sqrt2"] - (-0.166)) < 0.002,
    )
    r["miscalibration_confirmed"] = (
        abs(subs["3"]["line_from_pair_sd_over_sqrt2"]) > abs(SEALED_FALLBACK_M1))
    r["m_ge_3_fallback_reproducible_from_formula"] = (
        abs(abs(subs["3"]["line_from_pair_sd_over_sqrt2"]) - abs(SEALED_FALLBACK_M3)) < 0.01)

    # --- measured operating characteristics on null10 -----------------------------
    # draw = (held-out arm run i, baseline subset S of size m from the other 9)
    DELTAS = [0.0, 0.1, 0.2, 0.3]

    def draws(m):
        for i in SEEDS:
            others = [s for s in SEEDS if s != i]
            for S in itertools.combinations(others, m):
                stat = statistics.fmean(
                    [null_lc[i][g] - statistics.fmean([null_lc[o][g] for o in S]) for g in games])
                s_sub = sd([sum(null_lc[o].values()) / float(NGAMES) for o in S])
                s_out = sd([sum(null_lc[o].values()) / float(NGAMES) for o in others])  # nu=8
                yield stat, s_sub, s_out

    def oc(m, mode, alpha=0.05, clamp=True):
        """mode: 'sealed' (t at df=m-1, s from the m subset; fixed fallback at m=1)
                 'corrected' (t at df=nu_pool, s pooled/leave-one-out; monotone clamp)"""
        recs, lines = [], []
        for stat, s_sub, s_out in draws(m):
            if mode == "sealed":
                line = SEALED_FALLBACK_M1 if m == 1 else k3prime_sealed_line(s_sub, m, alpha)
            else:
                nu = 8  # pooled df available from the 9 non-arm same-config runs
                line = -t95(nu) * s_out * math.sqrt(1 + 1.0 / m) if alpha == 0.05 else \
                       -float(stats.t.ppf(1 - alpha, nu)) * s_out * math.sqrt(1 + 1.0 / m)
                if clamp:
                    line = max(line, SEALED_FALLBACK_M1)  # never wider than the m=1 fallback
            lines.append(line)
            recs.append((stat, line))
        res = dict(m=m, mode=mode, alpha=alpha, n_draws=len(recs),
                   line_mean=statistics.fmean(lines), line_sd=sd(lines),
                   line_min=min(lines), line_max=max(lines),
                   line_p05=quantile(lines, 0.05), line_p95=quantile(lines, 0.95))
        for d in DELTAS:
            fails = sum(1 for stat, line in recs if (stat - d) < line)
            key = "type_I" if d == 0.0 else "power_at_dlc_minus_%.1f" % d
            res[key] = fails / float(len(recs))
        # minimum harm detectable with >=80% power (grid)
        mdd = None
        for d in [x / 100.0 for x in range(0, 121)]:
            p = sum(1 for stat, line in recs if (stat - d) < line) / float(len(recs))
            if p >= 0.80:
                mdd = d
                break
        res["min_detectable_harm_at_80pct_power"] = mdd
        return res

    sealed_oc, corr_oc = {}, {}
    for m in (1, 2, 3, 5, 9):
        sealed_oc[str(m)] = oc(m, "sealed")
        corr_oc[str(m)] = oc(m, "corrected")
    out["measured_operating_characteristics"] = dict(
        harm_model="location shift: every game's arm lc reduced by delta (mean dlc shifts by -delta)",
        null_corpus="runs/null10, 10 same-config vanilla runs x 25 games; exhaustive draws",
        sealed_K3prime=sealed_oc, corrected_K3prime2=corr_oc,
        note=("'corrected_K3prime2' here is an INTERMEDIATE diagnostic (pooled-df t form, "
              "clamped) kept for the record. The RECOMMENDED gate is under the top-level key "
              "corrected_gate_K3pp, whose numbers are the ones published in "
              "duck_eval/SCREEN_PROTOCOL.md. Both instruments are exhaustive-subset draws "
              "(finite pool of 10), which is why their type-I rises with m; the bootstrap "
              "instrument under corrected_gate_K3pp.measured_on_null10 is the primary one."),
    )

    # monotonicity check on the mean line magnitude
    def mono(d):
        ms = sorted(int(k) for k in d)
        mags = [abs(d[str(m)]["line_mean"]) for m in ms]
        return dict(m=ms, mean_line_magnitude=[round(x, 4) for x in mags],
                    monotone_non_increasing=all(mags[i] >= mags[i + 1] - 1e-12
                                                for i in range(len(mags) - 1)))
    out["monotonicity"] = dict(sealed=mono(sealed_oc), corrected=mono(corr_oc))

    # --- diagnosis ---------------------------------------------------------------
    out["diagnosis"] = dict(
        t_multipliers={str(df): t95(df) for df in (1, 2, 4, 6, 8, 9, 20, 50)},
        parametric_line_in_units_of_s_base={
            str(m): (None if m < 2 else t95(m - 1) * math.sqrt(1 + 1.0 / m))
            for m in (1, 2, 3, 4, 5, 9, 10)},
        m1_fallback_in_units_of_s_base=abs(SEALED_FALLBACK_M1) / s_run_direct,
        cause=("The sealed K3' mixes two constructions on two different scales. The m=1 "
               "fallback -0.200 is an EMPIRICAL 5th percentile of the null10 pair distribution "
               "(= 2.33 * s_base). The m>=2 line is a PARAMETRIC t prediction interval whose "
               "multiplier t(0.95,m-1)*sqrt(1+1/m) is 8.93 at m=1(hypothetically), 3.37 at m=3, "
               "2.21 at m=5, 1.93 at m=9. The parametric family is itself monotone; the break is "
               "the splice: at m=3 the parametric multiplier (3.37) is 1.45x the empirical m=1 "
               "multiplier (2.33), so buying two extra baseline runs WIDENS the pass band. The "
               "driver is t(0.95,df=2)=2.920 (1.78x the large-sample 1.645) applied to a df=2 "
               "variance estimate, not the sqrt(1+1/m) inflation (which correctly shrinks "
               "1.414 -> 1.155 -> 1.054)."),
        fix=("Estimate s_base with the largest df available (pool same-harness-family within-family "
             "sums of squares) so t is ~1.65-1.75 at every m, and clamp the line so its magnitude "
             "is monotone non-increasing in m."),
    )

    # --- bootstrap calibration ----------------------------------------------------
    # KEY SIMPLIFICATION, exact: mean_g( lc_arm[g] - mean_i lc_base_i[g] )
    #                          = M_arm - mean_i M_base_i,   M_r = run lc total / 25.
    # So the whole calibration of the mean-dlc statistic depends on the 10 run means only.
    # The exhaustive-subset instrument above draws WITHOUT replacement from a pool of 10,
    # which makes its m>=2 tail quantiles finite-pool artefacts. An i.i.d. bootstrap over
    # the 10 observed run means removes that and gives a clean m-dependence.
    M = np.array([sum(null_lc[s].values()) / float(NGAMES) for s in SEEDS])
    sigma = float(M.std(ddof=1))
    rng = np.random.default_rng(20260810)
    NB = 400000
    boot = {}
    for m in (1, 2, 3, 4, 5, 6, 9, 10):
        arm = rng.choice(M, size=NB)
        base = rng.choice(M, size=(NB, m)).mean(axis=1)
        stat = arm - base
        q05 = float(np.quantile(stat, 0.05))
        boot[str(m)] = dict(
            m=m, n_boot=NB, stat_sd=float(stat.std(ddof=1)),
            theory_sd=sigma * math.sqrt(1 + 1.0 / m),
            q05=q05, q10=float(np.quantile(stat, 0.10)),
            c05_multiplier_of_sigma=abs(q05) / sigma,
            normal_theory_multiplier=1.645 * math.sqrt(1 + 1.0 / m),
        )
    # monotone (non-increasing in m) envelope of the measured multiplier
    ms = [1, 2, 3, 4, 5, 6, 9, 10]
    running = None
    for m in ms:
        c = boot[str(m)]["c05_multiplier_of_sigma"]
        running = c if running is None else min(running, c)
        boot[str(m)]["c05_monotone_envelope"] = running
    out["bootstrap_calibration"] = dict(
        run_means=[round(float(x), 4) for x in M], sigma_run=sigma,
        identity="mean dlc == M_arm - mean(M_baseline); per-game structure cancels exactly",
        by_m=boot,
        note="i.i.d. bootstrap over the 10 null10 run means, 400k draws per m, seed 20260810",
    )

    # --- the corrected gate, K3'' -------------------------------------------------
    # line(m) = -C(m) * s_hat,  C(m) = c1 * sqrt((1+1/m)/2),  c1 = measured m=1 multiplier.
    # Monotone non-increasing in m BY CONSTRUCTION (t and s_hat do not depend on m).
    c1 = boot["1"]["c05_multiplier_of_sigma"]
    # C(m) = the MEASURED 5th-percentile multiplier, monotonised (running min in m).
    # Monotone non-increasing in m by construction; empirically calibrated at every m.
    C_meas = {str(m): boot[str(m)]["c05_monotone_envelope"] for m in (1, 2, 3, 4, 5, 6, 9, 10)}
    # publishable rounded schedule (2 dp), re-checked out-of-sample below
    C = {"1": 2.33, "2": 2.10, "3": 2.02, "4": 1.98, "5": 1.96, "6": 1.94, "9": 1.94, "10": 1.94}
    C_theory_shape = {str(m): c1 * math.sqrt((1 + 1.0 / m) / 2.0)
                      for m in (1, 2, 3, 4, 5, 6, 9, 10)}
    rng_eval = np.random.default_rng(99260810)   # independent evaluation stream

    def oc_corrected(m, s_hat=None, nu=8, deltas=(0.0, 0.1, 0.2, 0.3)):
        s_hat = sigma if s_hat is None else s_hat
        line = -C[str(m)] * s_hat
        arm = rng_eval.choice(M, size=NB)
        base = rng_eval.choice(M, size=(NB, m)).mean(axis=1)
        stat = arm - base
        r = dict(m=m, line=line, C=C[str(m)], s_hat=s_hat)
        for d in deltas:
            key = "type_I" if d == 0 else "power_at_dlc_minus_%.1f" % d
            r[key] = float(np.mean((stat - d) < line))
        # minimum detectable harm at 80% power
        r["min_detectable_harm_at_80pct_power"] = float(
            -line + 0.8416 * float(stat.std(ddof=1)))
        return r

    def oc_sealed_boot(m, deltas=(0.0, 0.1, 0.2, 0.3)):
        arm = rng.choice(M, size=NB)
        bs = rng.choice(M, size=(NB, m))
        base = bs.mean(axis=1)
        stat = arm - base
        if m == 1:
            line = np.full(NB, SEALED_FALLBACK_M1)
        else:
            s_sub = bs.std(axis=1, ddof=1)
            line = -t95(m - 1) * s_sub * math.sqrt(1 + 1.0 / m)
        r = dict(m=m, line_mean=float(line.mean()), line_sd=float(line.std(ddof=1)),
                 line_p05=float(np.quantile(line, 0.05)),
                 frac_degenerate_s_sub_eq_0=float(np.mean(np.isclose(line, 0.0))))
        for d in deltas:
            key = "type_I" if d == 0 else "power_at_dlc_minus_%.1f" % d
            r[key] = float(np.mean((stat - d) < line))
        return r

    out["corrected_gate_K3pp"] = dict(
        form="mean dlc >= -C(m) * s_hat ;  C(m) = published multiplier schedule (below)",
        c1_measured_at_m1=c1,
        c1_normal_theory_1_645_times_sqrt2=1.645 * math.sqrt(2),
        C_by_m=C,
        C_measured_monotone_envelope=C_meas,
        C_normal_theory_shape_for_reference=C_theory_shape,
        monotone_non_increasing_in_m=all(
            C[str(a)] >= C[str(b)] - 1e-12
            for a, b in zip([1, 2, 3, 4, 5, 6, 9], [2, 3, 4, 5, 6, 9, 10])),
        s_hat_rule=("pooled run-to-run sd of per-game mean lc for the comparator's harness "
                    "family, estimated by pooling within-family sums of squares across "
                    "same-rail control families; publish s_hat, its df (require >= 4) and the "
                    "families used"),
        measured_on_null10=dict(
            corrected={str(m): oc_corrected(m) for m in (1, 3, 5, 9)},
            sealed_bootstrap={str(m): oc_sealed_boot(m) for m in (1, 3, 5, 9)},
        ),
        lines_at_vanilla_sigma={str(m): -C[str(m)] * sigma for m in (1, 3, 5, 9)},
        lines_at_warpack_sigma_0_189={str(m): -C[str(m)] * 0.189 for m in (1, 3, 5, 9)},
        lines_at_pooled_buildrail_sigma_0_1417={str(m): -C[str(m)] * 0.14174 for m in (1, 3, 5, 9)},
    )
    return out


# ------------------------------------------------------------------ D4
def d4(rows, fams, null_lc):
    out = {}
    games = sorted(null_lc[SEEDS[0]])

    warpack = fams.get("duck-harness-kaggle-warpack-v1", [])
    W3 = sorted(warpack, key=lambda r: r["start"])
    wtot = [r["lc_total"] for r in W3]
    wmeans = [t / float(NGAMES) for t in wtot]
    s_war = sd(wmeans)

    # (a) what is free: 3 runs -> df=2
    chi_lo = float(stats.chi2.ppf(0.95, 2))
    chi_hi = float(stats.chi2.ppf(0.05, 2))
    out["free_warpack_null"] = dict(
        runs=[r["path"] for r in W3], lc_totals=wtot, mean_lc_per_game=wmeans,
        s_base=s_war, df=2,
        n_independent_pairs=3, n_ordered_pairs=6,
        sigma_90pct_CI=[s_war * math.sqrt(2.0 / chi_lo), s_war * math.sqrt(2.0 / chi_hi)],
        relative_SE_of_sigma_hat=1.0 / math.sqrt(2 * 2),
        precision_note=("df=2. The 90%% CI on sigma spans %.3f-%.3f, a %.1fx range. The "
                        "point estimate 0.189 carries no more information than 'somewhere "
                        "between half and four times the vanilla figure'.")
                       % (s_war * math.sqrt(2.0 / chi_lo), s_war * math.sqrt(2.0 / chi_hi),
                          math.sqrt(chi_lo / chi_hi)),
    )

    # variance ratio vs vanilla, with CI (R25 N5)
    van_means = [sum(null_lc[s].values()) / float(NGAMES) for s in SEEDS]
    s_van = sd(van_means)
    F = (s_war ** 2) / (s_van ** 2)
    out["free_warpack_null"]["variance_ratio_vs_vanilla"] = dict(
        F=F, df1=2, df2=9,
        p_one_sided=float(1 - stats.f.cdf(F, 2, 9)),
        CI_90=[F / float(stats.f.ppf(0.95, 2, 9)), F / float(stats.f.ppf(0.05, 2, 9))],
        CI_95=[F / float(stats.f.ppf(0.975, 2, 9)), F / float(stats.f.ppf(0.025, 2, 9))],
        vanilla_s_base=s_van,
    )

    # (b) the free alternative nobody ran: pool the OTHER same-label families on the
    #     same build rail and the same 25 game instances.
    POOL_LABELS = ["duck-harness-kaggle-warpack-v1", "duck-harness-kaggle",
                   "duck-harness-kaggle-continuation-v1", "duck-harness-kaggle-sentinel-v2"]
    pool, tot_ss, tot_df = {}, 0.0, 0
    for lab in POOL_LABELS:
        fs = family_stats(fams.get(lab, []))
        pool[lab] = fs
        tot_ss += fs["ss"]
        tot_df += fs["df"]
    s_pool = math.sqrt(tot_ss / tot_df) if tot_df else None
    # Bartlett across families with df>=1
    groups = [[r["lc_total"] / float(NGAMES) for r in fams[lab]]
              for lab in POOL_LABELS if len(fams.get(lab, [])) > 1]
    bart = stats.bartlett(*groups) if len(groups) > 1 else None
    out["pooled_buildrail_null"] = dict(
        families=pool, pooled_ss=tot_ss, pooled_df=tot_df, s_pooled=s_pool,
        bartlett=dict(stat=float(bart.statistic), p=float(bart.pvalue)) if bart else None,
        vs_vanilla=dict(
            F=(s_pool ** 2) / (s_van ** 2), df1=tot_df, df2=9,
            p_one_sided=float(1 - stats.f.cdf((s_pool ** 2) / (s_van ** 2), tot_df, 9))),
        sigma_90pct_CI=[s_pool * math.sqrt(tot_df / float(stats.chi2.ppf(0.95, tot_df))),
                        s_pool * math.sqrt(tot_df / float(stats.chi2.ppf(0.05, tot_df)))],
        note=("All four families run the SAME 25 game instance ids on the SAME free build rail; "
              "pooling their within-family sums of squares raises df from 2 to %d at zero cost. "
              "Note the spread ACROSS families: the plain 'duck-harness-kaggle' family (m=3) has "
              "s_base BELOW vanilla's, while warpack's is 2.2x it -- with df=2 each, that is what "
              "a df=2 variance estimate looks like when it is pure noise." % tot_df),
    )

    # (c) precision-vs-m curve and the price of buying it
    curve = {}
    for m in range(2, 13):
        df = m - 1
        curve[str(m)] = dict(
            df=df, t95=t95(df), multiplier=t95(df) * math.sqrt(1 + 1.0 / m),
            rel_SE_sigma=1.0 / math.sqrt(2.0 * df),
            sigma_CI90_width_ratio=math.sqrt(float(stats.chi2.ppf(0.95, df)) /
                                             float(stats.chi2.ppf(0.05, df))),
            line_if_sigma_is_0_189=-t95(df) * 0.189 * math.sqrt(1 + 1.0 / m),
            power_vs_dlc_minus_0_20=float(stats.norm.cdf(
                (-(-t95(df) * 0.189 * math.sqrt(1 + 1.0 / m)) - 0.20) /
                (0.189 * math.sqrt(1 + 1.0 / m)) * -1 + 0)),
        )
    # recompute power properly: P(stat < line | mean = -0.20, sd = sigma*sqrt(1+1/m))
    for k, v in curve.items():
        m = int(k)
        se = 0.189 * math.sqrt(1 + 1.0 / m)
        v["power_vs_dlc_minus_0_20"] = float(stats.norm.cdf(
            (v["line_if_sigma_is_0_189"] + 0.20) / se))
        v["power_vs_dlc_minus_0_30"] = float(stats.norm.cdf(
            (v["line_if_sigma_is_0_189"] + 0.30) / se))
    out["precision_vs_m"] = dict(
        curve=curve,
        note="power columns assume the true sigma equals the warpack point estimate 0.189 and "
             "that the pooled-df correction is NOT applied (i.e. the sealed K3' form).")
    # same curve with a high-df t (the corrected form) for comparison
    curve2 = {}
    for m in range(2, 13):
        se = 0.189 * math.sqrt(1 + 1.0 / m)
        line = -t95(8) * se
        curve2[str(m)] = dict(line=line, power_vs_minus_0_20=float(stats.norm.cdf((line + 0.20) / se)),
                              power_vs_minus_0_30=float(stats.norm.cdf((line + 0.30) / se)))
    out["precision_vs_m_corrected_form"] = curve2

    # (d) the price, in builds
    seal = json.load(open(os.path.join(ROOT, "runs", "sealed", "r17_thresholds.json")))
    durations = []
    for r in rows:
        if r["duplicate_of"]:
            continue
        try:
            a = datetime.fromisoformat(r["start"]); b = datetime.fromisoformat(r["end"])
            durations.append((b - a).total_seconds() / 3600.0)
        except Exception:
            pass
    # (d0) how many runs a POWERED screen would need (not just a precise sigma)
    # SE = sigma * sqrt(1/k + 1/m); need C*SE + z80*SE <= |true harm|
    design = {}
    for lab, sg in [("warpack_sigma_0.189", s_war), ("pooled_sigma_0.1417", None)]:
        pass
    for lab, sg in [("warpack_sigma_0_189", s_war),
                    ("pooled_buildrail_sigma_0_1417", None),
                    ("vanilla_sigma_0_086", s_van)]:
        sg = sg if sg is not None else 0.14173527750312867
        d_lab = {}
        for harm in (0.10, 0.20, 0.30):
            need = harm / ((1.94 + 0.8416) * sg)          # sqrt(1/k + 1/m) bound, C(m>=6)=1.94
            bal = None
            if need > 0:
                inv = need ** 2
                if inv > 0:
                    kk = 2.0 / inv
                    bal = math.ceil(kk) if kk == kk else None
            d_lab["harm_%.2f" % harm] = dict(
                required_sqrt_1overk_plus_1overm=need,
                balanced_k_eq_m=bal,
                total_builds_balanced=(None if bal is None else 2 * bal),
                asymptotic_power_with_infinite_baseline_and_1_arm_seed=float(
                    stats.norm.cdf((-1.94 * sg + harm) / sg)),
            )
        design[lab] = d_lab
    out["power_design"] = dict(
        rule="80% power one-sided at the corrected gate: (C + z_0.80) * sigma * sqrt(1/k + 1/m) <= |harm|, "
             "C = 1.94 (m>=6), z_0.80 = 0.8416; k = arm seeds, m = baseline seeds",
        by_family=design,
        note="the asymptotic column is the ceiling: even an INFINITE baseline with one arm seed "
             "cannot exceed it, because sigma (not df) is the binding constraint",
    )

    out["price"] = dict(
        measured_build_hours=dict(n=len(durations), mean=statistics.fmean(durations),
                                  min=min(durations), max=max(durations)),
        minutes_5_4_budget="12-13 builds/week (30 GPU-h / 2.2-2.4 h) -- learnings/war_room/"
                           "r24_minutes_2026-08-09.md section 5.4",
        r17_sealed_price_of_a_legal_control_band=dict(
            source="runs/sealed/r17_thresholds.json -> thresholds.control_band",
            fallback=seal["thresholds"]["control_band"]["fallback"],
            gpu_h=seal["thresholds"]["control_band"]["gpu_h"]),
    )
    return out


# ------------------------------------------------------------------ main
def main():
    rows, fams, dups = inventory()
    null_lc = load_null10()

    out = dict(provenance=dict(
        generated_utc=datetime.now(timezone.utc).isoformat(),
        script="duck_eval/r24_prep/k3prime_fallout.py",
        purpose="D1 A22 re-screen under K3'; D2 K3' type-II recalibration (R25 methodology N1); "
                "D3 protocol numbers; D4 warpack-specific null scope+price.",
        cost="$0, CPU-only, zero Kaggle pushes, read-only",
        upstream=["learnings/sweeps/gate_recalibration_2026-08-09.md",
                  "runs/gate_recalibration_2026-08-09.json",
                  "learnings/war_room/r24_minutes_2026-08-09.md",
                  "learnings/panel/round25/methodology.md (N1, N4, N5)"],
    ))
    out["inventory"] = dict(
        n_runs=len([r for r in rows if not r["duplicate_of"]]),
        duplicates=dups,
        runs=[{k: v for k, v in r.items() if k != "lc"} for r in rows],
        families={lab: family_stats(rs) for lab, rs in sorted(fams.items())},
        all_share_prefix_set=len({r["prefix_set_md5"] for r in rows}) == 1,
        instance_id_sets=sorted({r["instance_id_md5"] for r in rows}),
    )
    out["D1_a22_rescreen"] = d1(rows, fams)
    out["D2_type_II_recalibration"] = d2(null_lc)
    out["D4_warpack_null"] = d4(rows, fams, null_lc)

    # ---- D1 addendum: the same arms under the CORRECTED gate K3'' ----------------
    C = out["D2_type_II_recalibration"]["corrected_gate_K3pp"]["C_by_m"]
    s_pool = out["D4_warpack_null"]["pooled_buildrail_null"]["s_pooled"]
    s_war = out["D4_warpack_null"]["free_warpack_null"]["s_base"]
    s_cont = out["D1_a22_rescreen"]["baseline_families"]["continuation_v1"]["s_base"]
    add = {}
    for name, arm in out["D1_a22_rescreen"]["arms"].items():
        add[name] = {}
        for tag, mean_key, m, s_hat, df in [
            ("warpack_m3_s_warpack_df2", "vs_warpack_m3_alpha05", 3, s_war, 2),
            ("warpack_m3_s_pooled_df6", "vs_warpack_m3_alpha05", 3, s_pool, 6),
            ("continuation_m2_s_cont_df1", "vs_continuation_m2_alpha05", 2, s_cont, 1),
            ("continuation_m2_s_pooled_df6", "vs_continuation_m2_alpha05", 2, s_pool, 6),
        ]:
            mn = arm[mean_key]["mean_dlc"]
            line = -C[str(m)] * s_hat
            se = s_hat * math.sqrt(1 + 1.0 / m)
            add[name][tag] = dict(
                m=m, s_hat=s_hat, s_hat_df=df, C=C[str(m)], line=line, mean_dlc=mn,
                verdict="PASS" if mn >= line else "FAIL", margin=mn - line,
                SE_of_statistic=se,
                min_detectable_harm_at_80pct_power=-line + 0.8416 * se,
                min_detectable_harm_in_levels_over_25_games=(-line + 0.8416 * se) * NGAMES,
                power_vs_true_harm_0_20=float(stats.norm.cdf((line + 0.20) / se)),
                meets_m_ge_3_precondition=(m >= 3),
            )
    out["D1_a22_rescreen"]["under_corrected_gate"] = add

    # ---- D3: the protocol file, and the numbers it publishes --------------------
    out["D3_protocol"] = dict(
        canonical_file="duck_eval/SCREEN_PROTOCOL.md",
        created_because="no canonical screen-protocol file existed; the gate lived inline in "
                        "runs/kernel_pulls/a22_v2_1/_screen_m1m2.py L116-119 (and v1/v2 twins), "
                        "restated per-arm in each prereg, and numerically in "
                        "runs/sealed/r17_thresholds.json",
        hard_preconditions=dict(
            P1_same_config_legality="baseline family must match the arm's config except the one "
                                    "pre-registered change, verified from run-log mechanism "
                                    "banners; label alone insufficient; git_status.txt is not "
                                    "evidence (byte-identical across all pulls). Warpack band is "
                                    "ILLEGAL as a control for {(f)}-envelope arms per R17 seal.",
            P2_m_ge_3="baseline = per-game mean of m >= 3 same-config runs; m<3 => NOT SCREENABLE",
            P3_sigma_df_ge_4="sigma_hat pooled across same-rail control families, df >= 4",
        ),
        gate="mean dlc >= -C(m) * sigma_hat",
        C_schedule=C,
        standing_sigma_pooled=dict(value=s_pool, df=6),
        struck_legs=["worst-game dlc >= -1.0 (type-I 50%/60%)",
                     "#(dlc<=-2)<=2 is advisory, never gating",
                     "actions_per_level_completed may not be a co-primary"],
        mandatory_seal_items=["baseline family paths + banner evidence", "m", "sigma_hat + df + "
                             "families pooled", "line + measured type-I + corpus",
                             "80%-power detection floor in levels",
                             "power-honesty clause if power < 50%"],
        family_screenability={
            lab: dict(m=f["m"], lc_totals=f["lc_totals"],
                      screenable=f["m"] >= 3)
            for lab, f in out["inventory"]["families"].items() if f["m"] >= 2},
    )

    # ---- rulings ----------------------------------------------------------------
    out["rulings"] = dict(
        D1=dict(
            status="FORMALLY OPEN AND UNWORKED",
            headline="A22 compaction: screened by a broken instrument against an illegal, "
                     "high-outlier, single-run baseline; not shown to help, not shown to harm; "
                     "NOT re-screenable at $0 today.",
            k3prime_as_sealed_warpack_m3_line=-0.6374,
            arms_pass_every_calibrated_line_on_disk=True,
            legal_family="duck-harness-kaggle-continuation-v1 (m=2, FAILS the m>=3 precondition)",
            power_at_affordable_m=dict(warpack_m3=0.202, pooled_m3=0.299),
            min_detectable_harm_levels=dict(warpack_m3=14.1, pooled_m3=10.6),
            recommendation="NO builds. Negative EV vs lane (a), which was ratified on independent "
                           "grounds and keeps the budget.",
            revival_conditions=["R1 lane (a) free instrumentation + R25 N3 discharged",
                                "R2 a surviving mechanism claim (non-harm is not a reason to run)",
                                "R3 continuation-v1 control family reaches m>=3 (1 seed short)",
                                "R4 prereg publishes sigma_hat/df/line/power-floor-in-levels",
                                "R5 power<50% => declared exploratory probe, not a screen"],
        ),
        D2=dict(
            n1_miscalibration_reproduced=True,
            sealed_m3_line=-0.2916, sealed_m1_fallback=-0.200,
            looser_at_m3_than_m1=True,
            m_ge_3_fallback_minus_0_190_reproducible_from_formula=False,
            corrected_gate="K3'' : mean dlc >= -C(m)*sigma_hat, C = 2.33/2.10/2.02/1.98/1.96/1.94",
            monotone=True,
        ),
        D4=dict(
            recommendation="Do NOT spend build-rail runs on a warpack-specific null; adopt the "
                           "free pooled build-rail sigma_hat = 0.1417 (df 6).",
            cost_avoided_builds_to_match_pooling=5,
            cost_avoided_builds_for_a_powered_screen=16,
            weeks_of_rail=16 / 13.0,
            exception="1 W0 continuation seed (2.2 GPU-h) to close the R17-sealed legal control "
                      "band at m=3 -- only when an arm on that harness is actually queued",
        ),
    )
    return out


if __name__ == "__main__":
    res = main()
    dest = os.path.join(ROOT, "runs", "k3prime_fallout_2026-08-10.json")
    json.dump(res, open(dest, "w"), indent=1, default=float)
    print("wrote", dest)
    print("\n--- families ---")
    for lab, f in res["inventory"]["families"].items():
        print("%-58s m=%d lc=%s s_base=%.4f df=%d" % (lab, f["m"], f["lc_totals"], f["s_base"], f["df"]))
    print("\n--- D1 ---")
    for name, a in res["D1_a22_rescreen"]["arms"].items():
        w = a["vs_warpack_m3_alpha05"]
        c = a["vs_continuation_m2_alpha05"]
        print("%-9s oldK3=%-4s | warpack m=3 line %.4f mean %.4f -> %s | cont m=2 line %s mean %.4f"
              % (name, a["old_K3"]["verdict"], w["k3prime_line"], w["mean_dlc"], w["verdict"],
                 ("%.4f" % c["k3prime_line"]) if c["k3prime_line"] is not None else "n/a",
                 c["mean_dlc"]))
    print("\n--- D2 ---")
    r = res["D2_type_II_recalibration"]["reviewer_substitution"]
    print("s_base(pair/sqrt2)=%.4f  s_base(direct)=%.4f" % (
        r["s_base_implied_pair_sd_over_sqrt2"], r["s_base_direct_run_level_sd"]))
    for m in ("2", "3", "5", "9", "10"):
        print("  m=%s sealed line %.4f" % (m, r["lines_by_m"][m]["line_from_pair_sd_over_sqrt2"]))
    print("miscalibration_confirmed:", r["miscalibration_confirmed"])
    oc = res["D2_type_II_recalibration"]["measured_operating_characteristics"]
    for mode in ("sealed_K3prime", "corrected_K3prime2"):
        print(mode)
        for m in ("1", "2", "3", "5", "9"):
            d = oc[mode][m]
            print("   m=%s line %.4f typeI %.4f pow(-0.2) %.4f pow(-0.3) %.4f mdd80 %s"
                  % (m, d["line_mean"], d["type_I"], d["power_at_dlc_minus_0.2"],
                     d["power_at_dlc_minus_0.3"], d["min_detectable_harm_at_80pct_power"]))
    print("\n--- D4 ---")
    f = res["D4_warpack_null"]
    print("warpack s=%.4f df=2 CI90 %s" % (f["free_warpack_null"]["s_base"],
                                           f["free_warpack_null"]["sigma_90pct_CI"]))
    p = f["pooled_buildrail_null"]
    print("pooled s=%.4f df=%d CI90 %s bartlett %s" % (p["s_pooled"], p["pooled_df"],
                                                       p["sigma_90pct_CI"], p["bartlett"]))
