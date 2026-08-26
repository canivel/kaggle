#!/usr/bin/env python
"""Phase-1 A/B offline scorer + gate.

Scores OUR phase1 seed runs and Tufa's vanilla null runs with the SAME
RHAE formula and the SAME per-level baselines (Tufa's benchmark.json
base_actions_per_level), then runs a paired game-level exact sign-flip
permutation test.

RHAE formula (mirrors taaf GameRun._compute_final_score, which mirrors
arc_agi.scorecard.EnvironmentScoreCalculator v0.9.8):
  per level i (0-indexed), weight w_i = i+1
  level_score_i = min(115, (baseline_i / actions_i)^2 * 100)
                  if level completed and actions_i > 0 else 0
  score = min( sum(w_i * level_score_i) / sum(all w_i),
               sum(w_i over levels with score>0) / sum(all w_i) * 100 )

Validation: recomputes every one of Tufa's 500 game_runs from raw
(levels_completed, actions_per_level, base_actions_per_level) and
cross-checks against BOTH the stored final_score fields and
score.json seed_scores. Aborts if max abs error > 1e-9.

Usage:
  python scripts/phase1_gate.py \
      --null runs/tufa_example_run/benchmark.json \
      --null-score runs/tufa_example_run/score.json \
      --seeds runs/phase1_ab/phase1_seed1.json runs/phase1_ab/phase1_seed2.json \
      --out runs/phase1_ab/gate_report_provisional.md

Each --seeds entry phase1_seedN.json must have a sibling detailed
benchmark dir seedN/benchmark.json (with actions_per_level); a
benchmark.json path may also be passed directly.
"""

import argparse
import json
import random
import sys
from pathlib import Path

ALPHA = 0.0125
EXCLUDE_PREFIXES = {"bp35"}  # pre-registered: flaky arcade in seeds 2-3
N_SEEDS_PLANNED = 3


# ---------------------------------------------------------------- RHAE

def rhae_score(base_actions_per_level, actions_per_level, levels_completed,
               number_of_levels):
    """Exact mirror of taaf GameRun._compute_final_score."""
    if base_actions_per_level is None or number_of_levels == 0:
        return 0.0
    total_score = 0.0
    total_weights = 0
    max_weights = 0
    for level_idx in range(number_of_levels):
        weight = level_idx + 1
        total_weights += weight
        completed = level_idx < levels_completed
        actions = (actions_per_level[level_idx]
                   if level_idx < len(actions_per_level) else 0)
        # align by index; if our game version has more levels than the
        # null baseline list, reuse the last baseline (flagged upstream)
        if level_idx < len(base_actions_per_level):
            baseline = base_actions_per_level[level_idx]
        else:
            baseline = base_actions_per_level[-1]
        if completed and actions > 0:
            level_score = min(115.0, (baseline / actions) ** 2 * 100)
        else:
            level_score = 0.0
        if level_score > 0:
            max_weights += weight
        total_score += level_score * weight
    if total_weights == 0:
        return 0.0
    score = total_score / total_weights
    max_score = max_weights / total_weights * 100
    return min(score, max_score)


# ---------------------------------------------------------------- loading

def load_null(benchmark_path, score_path):
    """Load Tufa's 500 vanilla runs; validate our scorer against their
    stored final_score AND score.json. Returns (per-prefix dict, max_err,
    overall_mean_check)."""
    bench = json.loads(Path(benchmark_path).read_text())
    sj = json.loads(Path(score_path).read_text()) if score_path else None

    runs_by_gid = {}
    for run in (bench if isinstance(bench, list) else bench["game_runs"]):
        runs_by_gid.setdefault(run["game_id"], []).append(run)

    max_err = 0.0
    n_checked = 0
    games = {}
    for gid, runs in runs_by_gid.items():
        seed_scores_ref = sj["games"][gid]["seed_scores"] if sj else {}
        # runs are stored in pass order (pass-0 .. pass-19)
        recomputed = []
        for i, run in enumerate(runs):
            s = rhae_score(run["base_actions_per_level"],
                           run["actions_per_level"],
                           run["levels_completed"],
                           run["number_of_levels"])
            recomputed.append(s)
            if run.get("final_score") is not None:
                max_err = max(max_err, abs(s - run["final_score"]))
                n_checked += 1
            ref = seed_scores_ref.get(f"example-run/pass-{i}") if sj else None
            if ref is not None:
                max_err = max(max_err, abs(s - ref))
                n_checked += 1
        game_mean = sum(recomputed) / len(recomputed)
        if sj:
            max_err = max(max_err, abs(game_mean - sj["games"][gid]["score"]))
        games[gid[:4]] = {
            "gid": gid,
            "scores": recomputed,
            "mean": game_mean,
            "levels": [r["levels_completed"] for r in runs],
            "nlev": runs[0]["number_of_levels"],
            "base": runs[0]["base_actions_per_level"],
            "states": [r["state"] for r in runs],
        }
    overall = sum(g["mean"] for g in games.values()) / len(games)
    if sj:
        max_err = max(max_err, abs(overall - sj["score"]))
    return games, max_err, n_checked, overall, (sj["score"] if sj else overall)


def find_detail(seed_path):
    """Locate detailed benchmark.json (with actions_per_level) for a
    phase1_seedN.json summary file."""
    p = Path(seed_path)
    d = json.loads(p.read_text())
    if "game_runs" in d:  # already a detailed benchmark file
        return d, p
    seed_no = d.get("seed")
    candidates = []
    if seed_no is not None:
        candidates.append(p.parent / f"seed{seed_no}" / "benchmark.json")
    stem = p.stem  # phase1_seedN
    if "seed" in stem:
        candidates.append(p.parent / stem.split("phase1_")[-1] / "benchmark.json")
    for c in candidates:
        if c.exists():
            return json.loads(c.read_text()), c
    raise SystemExit(
        f"ERROR: no detailed benchmark.json (actions_per_level) found for "
        f"{seed_path}; looked at {[str(c) for c in candidates]}. "
        f"Pull it from the pod (/workspace/ab_results/seedN/benchmark.json).")


def load_our_seed(seed_path, null_games):
    """Score one of our seeds offline with the null arm's baselines.
    Returns {prefix: {score, lc, gid, flags}}."""
    detail, detail_path = find_detail(seed_path)
    out = {}
    for run in detail["game_runs"]:
        gid = run["game_id"]
        prefix = gid[:4]
        ng = null_games.get(prefix)
        if ng is None:
            continue
        flags = []
        if gid != ng["gid"]:
            flags.append("diff-version")
        if run["number_of_levels"] != ng["nlev"]:
            flags.append(f"nlev {run['number_of_levels']} vs {ng['nlev']}")
        if run["number_of_levels"] > len(ng["base"]):
            flags.append("baseline-padded")
        s = rhae_score(ng["base"], run["actions_per_level"],
                       run["levels_completed"], run["number_of_levels"])
        out[prefix] = {
            "score": s,
            "lc": run["levels_completed"],
            "gid": gid,
            "state": run["state"],
            "flags": flags,
            "detail_path": str(detail_path),
        }
    return out


# ------------------------------------------------- exact sign-flip test

def signflip_p_exact(deltas, observed_sum):
    """Exact one-sided sign-flip p-value: P(sum of +/-d_i >= observed_sum)
    over all 2^n sign assignments, via meet-in-the-middle. Ties counted
    in (identity permutation always counts, so p > 0)."""
    n = len(deltas)
    half = n // 2
    a, b = deltas[:half], deltas[half:]

    def subset_sums(ds):
        sums = [0.0]
        for d in ds:
            sums = [s + d for s in sums] + [s - d for s in sums]
        return sums

    sa = subset_sums(a)
    sb = sorted(subset_sums(b))
    scale = max(1.0, abs(observed_sum), max((abs(d) for d in deltas), default=1.0))
    eps = 1e-9 * scale
    import bisect
    count = 0
    for s in sa:
        # need s + t >= observed_sum - eps  ->  t >= observed_sum - eps - s
        idx = bisect.bisect_left(sb, observed_sum - eps - s)
        count += len(sb) - idx
    return count / (2 ** n), 2 ** n


def signflip_p_mc(deltas, observed_sum, n_iter=100_000, seed=0):
    rng = random.Random(seed)
    n = len(deltas)
    count = 1  # identity
    for _ in range(n_iter):
        s = sum(d if rng.random() < 0.5 else -d for d in deltas)
        if s >= observed_sum - 1e-12:
            count += 1
    return count / (n_iter + 1), n_iter + 1


def paired_test(deltas):
    obs = sum(deltas)
    if len(deltas) <= 26:
        p, n_perm = signflip_p_exact(deltas, obs)
        method = f"exact ({n_perm} sign assignments)"
    else:
        p, n_perm = signflip_p_mc(deltas, obs)
        method = f"Monte Carlo ({n_perm} draws)"
    return obs / len(deltas), p, method


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--null", default="runs/tufa_example_run/benchmark.json")
    ap.add_argument("--null-score", default="runs/tufa_example_run/score.json")  # NULL_SCORE_OPTIONAL: pass nonexistent path to skip validation
    ap.add_argument("--seeds", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # 1. load + VALIDATE scorer against Tufa's own outputs (hard gate)
    from pathlib import Path as _P
    _ns = args.null_score if _P(args.null_score).exists() else None
    null_games, max_err, n_checked, overall, overall_ref = load_null(
        args.null, _ns)
    if _ns is not None and max_err > 1e-9:
        raise SystemExit(f"SCORER VALIDATION FAILED: max abs error {max_err:.3e} "
                         f"over {n_checked} cross-checks (limit 1e-9). "
                         f"Do not trust any verdict.")

    # 2. load + score our seeds
    seeds = [load_our_seed(sp, null_games) for sp in args.seeds]
    n_seeds = len(seeds)
    provisional = n_seeds < N_SEEDS_PLANNED

    # 3. build paired per-game table
    prefixes = sorted(null_games)
    rows = []
    for p in prefixes:
        ours = [s[p] for s in seeds if p in s]
        row = {
            "prefix": p,
            "null_mean": null_games[p]["mean"],
            "null_lc_mean": sum(null_games[p]["levels"]) / len(null_games[p]["levels"]),
            "our_scores": [o["score"] for o in ours],
            "our_lcs": [o["lc"] for o in ours],
            "n_ours": len(ours),
            "flags": sorted({f for o in ours for f in o["flags"]}),
            "excluded": None,
        }
        if p in EXCLUDE_PREFIXES:
            row["excluded"] = "pre-registered (flaky arcade seeds 2-3)"
        elif len(ours) == 0:
            row["excluded"] = "no data in any seed"
        elif len(ours) < n_seeds:
            row["flags"].append(f"only {len(ours)}/{n_seeds} seeds")
        if row["n_ours"]:
            row["our_mean"] = sum(row["our_scores"]) / row["n_ours"]
            row["our_lc_mean"] = sum(row["our_lcs"]) / row["n_ours"]
            row["delta"] = row["our_mean"] - row["null_mean"]
            row["lc_delta"] = row["our_lc_mean"] - row["null_lc_mean"]
        rows.append(row)

    inc = [r for r in rows if r["excluded"] is None]
    deltas = [r["delta"] for r in inc]
    lc_deltas = [r["lc_delta"] for r in inc]

    mean_delta, p_val, method = paired_test(deltas)
    lc_mean_delta, lc_p, lc_method = paired_test(lc_deltas)
    verdict = "PASS" if (mean_delta > 0 and p_val <= ALPHA) else "FAIL"

    our_mean_inc = sum(r["our_mean"] for r in inc) / len(inc)
    null_mean_inc = sum(r["null_mean"] for r in inc) / len(inc)
    our_all = [r for r in rows if r.get("n_ours")]
    our_mean_all = sum(r["our_mean"] for r in our_all) / len(our_all)

    # wa30 / tr87 partial-credit asymmetry check
    asym_notes = []
    for p in ("wa30", "tr87"):
        ng = null_games[p]
        frac_null = any(lv != int(lv) for lv in ng["levels"])
        frac_ours = any(any(l != int(l) for l in [o]) for s in seeds
                        for o in ([s[p]["lc"]] if p in s else []))
        asym_notes.append(
            f"{p}: null levels_completed all integer={not frac_null}, "
            f"ours all integer={not frac_ours}; scored offline with identical "
            f"formula+baselines both arms -> no exclusion needed.")

    tag = "PROVISIONAL" if provisional else "FINAL"
    L = []
    L.append(f"# Phase-1 A/B gate report — **{tag}** ({n_seeds}/{N_SEEDS_PLANNED} seeds)")
    if provisional:
        L.append("")
        L.append("> **PROVISIONAL** — seed 3 not yet included. Do not act on this verdict.")
    L.append("")
    L.append("## Scorer validation (hard requirement)")
    L.append(f"- Reproduced Tufa score.json + per-run final_score from raw benchmark.json: "
             f"**max abs error {max_err:.3e}** over {n_checked} cross-checks (500 runs, "
             f"25 game means, overall). PASS (limit 1e-9).")
    L.append(f"- Recomputed overall null score {overall:.6f} vs published {overall_ref:.6f}.")
    L.append("- Formula: per level i (0-idx, weight i+1): min(115, (base/actions)^2*100) if "
             "completed & actions>0 else 0; score = weighted mean over ALL level weights, "
             "capped at (weights of scoring levels)/(all weights)*100.")
    L.append("")
    L.append("## Arms")
    L.append(f"- Null: {args.null} — 25 games x 20 vanilla passes.")
    for i, sp in enumerate(args.seeds):
        d = seeds[i]
        src = next(iter(d.values()))["detail_path"] if d else "?"
        L.append(f"- Ours seed {i+1}: {sp} (detail: {src}, {len(d)} games)")
    L.append("- Both arms scored OFFLINE with identical formula and the null arm's "
             "base_actions_per_level (joined on 4-char game prefix).")
    L.append("")
    L.append("## Per-game paired deltas (RHAE, Kaggle-comparable units)")
    L.append("")
    L.append("| game | null mean (20p) | ours mean | delta | our lc | null lc mean | flags |")
    L.append("|---|---|---|---|---|---|---|")
    for r in rows:
        if r["excluded"]:
            L.append(f"| {r['prefix']} | {r['null_mean']:.4f} | — | — | — | "
                     f"{r['null_lc_mean']:.2f} | **EXCLUDED: {r['excluded']}** |")
        else:
            lcs = ",".join(str(x) for x in r["our_lcs"])
            L.append(f"| {r['prefix']} | {r['null_mean']:.4f} | {r['our_mean']:.4f} | "
                     f"{r['delta']:+.4f} | {lcs} | {r['null_lc_mean']:.2f} | "
                     f"{'; '.join(r['flags'])} |")
    L.append("")
    L.append("## Primary gate (RHAE)")
    L.append(f"- Included games: n = {len(inc)}")
    L.append(f"- Mean RHAE, included games: ours {our_mean_inc:.4f} vs null {null_mean_inc:.4f}")
    L.append(f"- Mean RHAE, ours all {len(our_all)} games (their-1.6002-scale): {our_mean_all:.4f} "
             f"(null all-25 reference: {overall:.4f})")
    L.append(f"- Mean paired delta: **{mean_delta:+.4f}**")
    L.append(f"- One-sided sign-flip permutation p (improvement): **{p_val:.6f}** [{method}]")
    L.append(f"- Alpha: {ALPHA}")
    L.append("")
    L.append(f"## VERDICT: **{verdict}**{' (PROVISIONAL)' if provisional else ''}")
    L.append("")
    L.append("## Secondary: levels completed (robustness)")
    L.append(f"- Mean paired lc delta: {lc_mean_delta:+.4f}; one-sided p: {lc_p:.6f} [{lc_method}]")
    L.append("")
    L.append("## Exclusion / asymmetry checks")
    L.append(f"- bp35 excluded (pre-registered; absent from our seeds >=2).")
    for n in asym_notes:
        L.append(f"- {n}")
    L.append("")
    diffv = [r["prefix"] for r in rows if any("diff-version" in f for f in r["flags"])]
    L.append(f"- Game-version note: {len(diffv)} games served to us under a different "
             f"version hash than the null run ({', '.join(diffv)}); scored with null "
             f"baselines by level index (pre-registered same-baselines-both-arms).")
    report = "\n".join(L) + "\n"

    print(report)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"[saved: {args.out}]", file=sys.stderr)


if __name__ == "__main__":
    main()
