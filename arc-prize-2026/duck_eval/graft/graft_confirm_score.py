"""Sealed scorer for the GRAFT SCORE-CONFIRMATION arm (v5, 2026-08-20 slot 1).

PREREG: learnings/war_room/graft_confirm_prereg_2026-08-19.md (sealed 2026-08-19 evening,
BEFORE the push). Written and selftested before the kernel runs, per house rule.

THIS ARM IS POST-HOC MOTIVATED AND SAYS SO: seed 1 (v4, exp_id 19) measured mean_score 2.303
at lc_total 18 (lc-matched baseline: 1.427) and that result has been SEEN. The CONFIRM bar is
set such that seed 1 would pass it — this scorer tests REPLICATION, not discovery.

Verdict order (three-state minimum honored):
  INFRA DEATH  any certification assertion fails / no benchmark.json / n_games != 25
  HARM         lc_total <= 12  (the standing K3'' levels guard — capability must not drop)
  CONFIRM      lc_total >= 13 AND mean_score >= lc-matched bar (matched baseline score + 0.5,
               ties in |lc distance| resolved to the HIGHER-score baseline — conservative)
  NULL         otherwise

Certification (banner, armed lines, forbidden flags, stock-fallback signatures) and the
CLI-2.2.3 JSON log decoding are REUSED from graft_score.py — the 2026-08-19 instrument fix
and its regression fixtures are inherited, not reimplemented.

Usage:
    python duck_eval/graft/graft_confirm_score.py <pulled_kernel_dir>
    python duck_eval/graft/graft_confirm_score.py --selftest
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import graft_score as gs  # noqa: E402  (certify_install, _read_log, _infra, constants)

# Baseline family duck-harness-kaggle, m=3 (lc_total, mean_score) — same family as the seal.
BASELINES = [(18, 1.427), (19, 1.939), (21, 3.420)]
HARM_LC_TOTAL = 12          # K3'' guard, identical to the levels seal
CONFIRM_MARGIN = 0.5        # coordinator-ruled replication margin, sealed pre-push
N_GAMES = 25


def confirm_bar(lc_total: int) -> float:
    """Score bar = lc-matched baseline score + margin; distance ties -> higher score (harder)."""
    dmin = min(abs(lc_total - lc) for lc, _ in BASELINES)
    matched = max(s for lc, s in BASELINES if abs(lc_total - lc) == dmin)
    return matched + CONFIRM_MARGIN


def score(run_dir: Path) -> dict:
    log = gs._read_log(run_dir)
    ok, reason, facts = gs.certify_install(log)
    if not ok:
        return gs._infra(f"graft install not certified: {reason}", **facts)

    path = run_dir / "benchmark.json"
    if not path.exists():
        return gs._infra("no benchmark.json in the pull directory", **facts)
    bench = json.loads(path.read_text(encoding="utf-8"))
    runs = bench.get("game_runs") or []
    if len(runs) != N_GAMES:
        return gs._infra(f"n_games={len(runs)}, expected {N_GAMES}", **facts)

    lc_total = sum(int(r.get("levels_completed") or 0) for r in runs)
    mean_score = sum(float(r.get("final_score") or 0.0) for r in runs) / len(runs)
    total_actions = sum(sum(r.get("actions_per_level") or []) for r in runs)
    games_won = sum(1 for r in runs if r.get("state") == "won")
    bar = confirm_bar(lc_total)

    out = {
        "decisive": True,
        "lc_total": lc_total,
        "mean_score": round(mean_score, 6),
        "confirm_bar": bar,
        "total_actions": total_actions,
        "games_won": games_won,
        "n_games": len(runs),
        **facts,
    }
    if lc_total <= HARM_LC_TOTAL:
        out["verdict"] = "HARM"
        out["reason"] = (f"lc_total {lc_total} <= {HARM_LC_TOTAL} (K3'' capability guard) — "
                         "score is not read when capability drops")
        return out
    if mean_score >= bar:
        out["verdict"] = "CONFIRM"
        out["reason"] = (f"mean_score {mean_score:.3f} >= lc-matched bar {bar:.3f} "
                         f"(lc_total {lc_total}); replication screen passed — licenses an A21 "
                         "exploration draw, NOT a promotion claim")
    else:
        out["verdict"] = "NULL"
        out["reason"] = (f"mean_score {mean_score:.3f} < lc-matched bar {bar:.3f} "
                         f"(lc_total {lc_total}); the seed-1 score advantage did not replicate")
    return out


def render(result: dict) -> str:
    lines = ["=" * 78, "GRAFT SCORE-CONFIRMATION ARM — SEALED VERDICT", "=" * 78]
    lines.append(f"  verdict               {result['verdict']}"
                 + ("" if result.get("decisive") else "   (NOT DECISIVE)"))
    lines.append(f"  reason                {result.get('reason', '')}")
    if "lc_total" in result:
        lines.append(f"  lc_total              {result['lc_total']}   (HARM guard <= {HARM_LC_TOTAL})")
        lines.append(f"  mean_score            {result['mean_score']}   (bar {result['confirm_bar']})")
        lines.append(f"  total_actions         {result['total_actions']}   "
                     "(descriptive; seed-1 3257, baselines 4757/4033)")
        lines.append(f"  games reaching 'won'  {result['games_won']}   (predict 0 — banking unreachable)")
    if "features_present" in result:
        lines.append(f"  FEATURES certified    {result['features_present']}")
    lines.append("")
    lines.append("  POST-HOC HONESTY: the motivating seed-1 result was SEEN before this seal;")
    lines.append("  the bar was set so seed 1 would pass. CONFIRM = replication, nothing more.")
    lines.append("=" * 78)
    return "\n".join(lines)


# ---- SELFTEST ---------------------------------------------------------------
def _selftest() -> int:
    import tempfile

    failures = []
    checks_run = [0]

    def expect(name: str, got, want) -> None:
        checks_run[0] += 1
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")

    def make(per_game_lc, per_game_scores, *, log=None, bench=True, n=N_GAMES):
        root = Path(tempfile.mkdtemp())
        (root / "kernel.log").write_text(log if log is not None else gs.GOOD_BANNER,
                                         encoding="utf-8")
        if bench:
            lcs = list(per_game_lc) + [0] * (n - len(per_game_lc))
            scs = list(per_game_scores) + [0.0] * (n - len(per_game_scores))
            runs = [{"game_id": f"g{i:02d}", "levels_completed": lcs[i], "state": "gave_up",
                     "final_score": scs[i], "actions_per_level": [10] * max(lcs[i], 1)}
                    for i in range(n)]
            (root / "benchmark.json").write_text(
                json.dumps({"label": "duck-harness-kaggle", "n_passes": 1, "game_runs": runs}),
                encoding="utf-8")
        return root

    # --- the bar table, verbatim from the prereg
    expect("bar(13) = 1.927", confirm_bar(13), 1.927)
    expect("bar(18) = 1.927", confirm_bar(18), 1.927)
    expect("bar(19) = 2.439", confirm_bar(19), 2.439)
    expect("bar(20) = 3.920 (tie -> higher)", confirm_bar(20), 3.92)
    expect("bar(21) = 3.920", confirm_bar(21), 3.92)
    expect("bar(25) = 3.920", confirm_bar(25), 3.92)

    # --- seed-1 replication values must CONFIRM (the bar is set so they do; stated in prereg)
    r = score(make([18], [2.303 * N_GAMES]))
    expect("seed-1 values -> CONFIRM", r["verdict"], "CONFIRM")

    # --- boundaries
    expect("lc 18, score below bar -> NULL",
           score(make([18], [1.9 * N_GAMES]))["verdict"], "NULL")
    expect("lc 18, score exactly at bar -> CONFIRM",
           score(make([18], [1.927 * N_GAMES]))["verdict"], "CONFIRM")
    expect("lc 19 raises the bar: 2.30 -> NULL",
           score(make([19], [2.30 * N_GAMES]))["verdict"], "NULL")

    # --- the HARM guard beats ANY score (score is never read when capability drops)
    expect("lc 12 with a huge score -> HARM",
           score(make([12], [9.9 * N_GAMES]))["verdict"], "HARM")
    expect("lc 13 restores score reading -> CONFIRM",
           score(make([13], [2.0 * N_GAMES]))["verdict"], "CONFIRM")

    # --- INFRA DEATH beats everything (inherited certification)
    expect("no banner -> INFRA",
           score(make([18], [2.303 * N_GAMES], log="nothing here"))["verdict"], "INFRA DEATH")
    expect("no benchmark.json -> INFRA",
           score(make([18], [2.303 * N_GAMES], bench=False))["verdict"], "INFRA DEATH")
    expect("wrong game count -> INFRA",
           score(make([18], [2.303 * N_GAMES], n=24))["verdict"], "INFRA DEATH")
    for sig in gs.STOCK_FALLBACK_SIGNATURES:
        expect(f"stock fallback {sig!r} -> INFRA",
               score(make([18], [2.303 * N_GAMES], log=gs.GOOD_BANNER + sig))["verdict"],
               "INFRA DEATH")

    # --- the CLI-2.2.3 JSON log format must certify (the 08-19 instrument fix, inherited)
    runtime_banner = gs.GOOD_BANNER.replace("'", '"').replace(" True", " true")
    cli_json = json.dumps([{"stream_name": "stdout", "time": 1.0, "data": line + chr(10)}
                           for line in runtime_banner.splitlines()])
    expect("CLI-JSON log certifies -> CONFIRM",
           score(make([18], [2.303 * N_GAMES], log=cli_json))["verdict"], "CONFIRM")

    # --- wrong arm: banking in FEATURES / [banking] armed -> INFRA
    wrong = gs.GOOD_BANNER.replace("'efficiency': True", "'banking': True, 'efficiency': True")
    expect("banking in FEATURES -> INFRA",
           score(make([18], [2.303 * N_GAMES], log=wrong))["verdict"], "INFRA DEATH")
    expect("[banking] armed line -> INFRA",
           score(make([18], [2.303 * N_GAMES],
                      log=gs.GOOD_BANNER + "[banking] armed" + chr(10)))["verdict"],
           "INFRA DEATH")

    n_checks = checks_run[0]  # counted, never transcribed
    if failures:
        print(f"SELFTEST FAILED ({len(failures)} of {n_checks}):")
        for f in failures:
            print("  " + f)
        return 1
    print(f"selftest OK ({n_checks}/{n_checks} checks, 0 failures)")
    print(f"  HARM     lc_total <= {HARM_LC_TOTAL}")
    print("  CONFIRM  mean_score >= lc-matched baseline + 0.5 (bars: 1.927 / 2.439 / 3.920)")
    print("  NULL     otherwise; INFRA DEATH on any certification failure")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        raise SystemExit(_selftest())
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    result = score(Path(sys.argv[1]))
    print(render(result))
    print(json.dumps(result, indent=2, sort_keys=True))
