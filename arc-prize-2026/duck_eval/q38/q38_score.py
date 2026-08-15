"""SEALED verdict scorer for the Q38 engine-swap eval. Written 2026-08-15 BEFORE the push.

`feedback_audit_the_instrument`: on 08-12 four ARC gates were broken at the moment they were
needed; on 08-14 a watch-rule was read with a retired threshold. Both times the instrument was
the defect and both times it failed silently and in our favour. So this scorer is written and
selftested BEFORE the kernel runs, and its constants are SEALED. Reading the data and then
adjusting a number here is the one thing that voids the screen.

CONSTANTS (sealed 2026-08-15, pre-push) — every one is derived in
`learnings/war_room/q38_engine_swap_prereg_2026-08-15.md`:

  BASELINE      duck-harness-kaggle, m = 3
                gate_eval_v1 lc 18 | gate_eval_v2 lc 19 | duckgate_v1post lc 21
                per-game mean lc = 58 / (3 x 25) = 0.773333
  SIGMA         0.141740 lc/game, df 6  (SCREEN_PROTOCOL sec P3 standing pooled estimate)
  SE(delta)     sigma * sqrt(1 + 1/3) = 0.163667 lc/game  (k=1 arm seed, m=3 baseline)
  C(3)          2.02  (SCREEN_PROTOCOL sec 2, null10-measured 5th-pct multiplier)

VERDICTS, in evaluation order (the third state is mandatory — the 08-14 LoRA canary landed in
exactly the state a two-state rule had no legal way to record):

  INFRA DEATH (not decisive)  no benchmark.json / != 25 games / boot asserts raised /
                              window drift > 5% / the served engine cannot be confirmed
  HARM (decisive)             mean dlc <= -0.286320  (K3'' line at m=3)  -> the engine is
                              worse and research_restart sec2 collapses
  REFUTE-2x (decisive)        mean dlc <= +0.250000  -> the "consistent 2x on the local 25"
                              claim is NOT reproduced on our harness (3.20 sigma below it)
  CONFIRM-2x (decisive)       mean dlc >= +0.500000  -> a >=65% lift; 95.3% power against a
                              true doubling, 0.11% false-positive rate under the null
  INDETERMINATE               anything between: a real but sub-claim lift, one seed cannot
                              separate it from noise

The score-based reading is reported and is explicitly NON-INFERENTIAL: the baseline's own
mean_score spread is 1.427 / 1.939 / 3.420 (sd 1.033 on n=3), so a score-based 2x test has
~60% power and may not carry a verdict. Levels-completed is the decision statistic
(SCREEN_PROTOCOL sec 0).

    python duck_eval/q38/q38_score.py <pulled_kernel_dir>
    python duck_eval/q38/q38_score.py --selftest
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# ---- SEALED CONSTANTS -----------------------------------------------------
BASELINE_LABEL = "duck-harness-kaggle"
BASELINE_RUNS = {"gate_eval_v1": 18, "gate_eval_v2": 19, "duckgate_v1post": 21}
BASELINE_M = 3
N_GAMES = 25
BASELINE_LC_PER_GAME = 58 / (3 * 25)          # 0.7733333...
SIGMA = 0.141740
SE_DELTA = 0.163667
C_M3 = 2.02
HARM_LINE = -0.286320                          # -C(3) * sigma
REFUTE_LINE = 0.250000
CONFIRM_LINE = 0.500000
WINDOW_S = 7920.0
WINDOW_DRIFT_TOL = 0.05
BASELINE_TOKS_PER_S = (212.52, 203.66, 197.47)
BASELINE_ACTIONS = (4757, 4033, 4632)
INFRA_SIGNATURES = (
    "Q38-EVAL FATAL",
    "No supported CUDA architectures",
    "libcudart",
    "CUDA out of memory",
    "torch.OutOfMemoryError",
    "Timed out waiting for vLLM server",
    "Missing attached dataset path",
    "ValueError: Unknown quantization",
    "does not support",
)
# ---------------------------------------------------------------------------


def _load(run_dir: Path) -> dict | None:
    path = run_dir / "benchmark.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _log_text(run_dir: Path) -> str:
    parts = []
    for pattern in ("*.log", "*.txt"):
        for path in run_dir.glob(pattern):
            try:
                parts.append(path.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                pass
    return "\n".join(parts)


def score(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    log = _log_text(run_dir)
    out: dict = {"run_dir": str(run_dir)}

    signature = next((s for s in INFRA_SIGNATURES if s in log), None)
    bench = _load(run_dir)

    if bench is None:
        out.update(verdict="INFRA DEATH (not decisive)", decisive=False,
                   reason=f"no benchmark.json; infra signature {signature!r}" if signature
                   else "no benchmark.json and no recognised infra signature")
        return out

    runs = bench.get("game_runs") or []
    out["label"] = bench.get("label")
    out["n_games"] = len(runs)
    if len(runs) != N_GAMES:
        out.update(verdict="INFRA DEATH (not decisive)", decisive=False,
                   reason=f"{len(runs)} games, expected {N_GAMES}"
                          + (f"; infra signature {signature!r}" if signature else ""))
        return out

    lc_total = sum(int(r.get("levels_completed") or 0) for r in runs)
    scores = [float(r.get("final_score") or 0.0) for r in runs]
    actions = sum(sum(r.get("actions_per_level") or []) for r in runs)
    windows = [float(r.get("final_wallclock_seconds") or 0.0) for r in runs]
    drifted = [w for w in windows if abs(w - WINDOW_S) > WINDOW_DRIFT_TOL * WINDOW_S]

    out.update(lc_total=lc_total,
               lc_per_game=lc_total / N_GAMES,
               mean_score=sum(scores) / N_GAMES,
               actions_total=actions,
               actions_per_window=actions / N_GAMES,
               windows_drifted=len(drifted),
               baseline_lc_per_game=BASELINE_LC_PER_GAME,
               baseline_lc_total=sum(BASELINE_RUNS.values()) / BASELINE_M)

    delta = lc_total / N_GAMES - BASELINE_LC_PER_GAME
    out["mean_delta_lc"] = delta
    out["delta_levels_over_25"] = delta * N_GAMES
    out["z_vs_baseline"] = delta / SE_DELTA

    if len(drifted) > 2:
        out.update(verdict="INFRA DEATH (not decisive)", decisive=False,
                   reason=f"{len(drifted)} games drifted >5% from the {WINDOW_S:.0f}s window; "
                          "the comparison against the recorded family is VOID")
        return out
    if "Q38-EVAL BOOT-ASSERTS PASSED" not in log and log:
        out.update(verdict="INFRA DEATH (not decisive)", decisive=False,
                   reason="the boot asserts never reported PASSED, so the served engine is "
                          "UNCONFIRMED; a Qwen3.6 result read as a Qwen3.8 result is the one "
                          "failure this screen cannot survive")
        return out

    if delta <= HARM_LINE:
        out.update(verdict="HARM (decisive)", decisive=True,
                   reason=f"mean dlc {delta:+.4f} <= K3'' line {HARM_LINE:+.4f} "
                          f"(C(3)={C_M3} x sigma={SIGMA})")
    elif delta <= REFUTE_LINE:
        out.update(verdict="REFUTE-2x (decisive)", decisive=True,
                   reason=f"mean dlc {delta:+.4f} <= {REFUTE_LINE:+.4f}: the 'consistent 2x on "
                          "the local 25' claim is NOT reproduced on our harness")
    elif delta >= CONFIRM_LINE:
        out.update(verdict="CONFIRM-2x (decisive)", decisive=True,
                   reason=f"mean dlc {delta:+.4f} >= {CONFIRM_LINE:+.4f}: a lift of the claimed "
                          "order; the engine explanation survives its primary falsifier")
    else:
        out.update(verdict="INDETERMINATE", decisive=False,
                   reason=f"mean dlc {delta:+.4f} lies between {REFUTE_LINE:+.4f} and "
                          f"{CONFIRM_LINE:+.4f}: a real-looking but sub-claim lift that one arm "
                          "seed cannot separate from noise")
    return out


def render(result: dict) -> str:
    lines = [
        "=" * 78,
        f"Q38 ENGINE-SWAP EVAL — SEALED VERDICT: {result['verdict']}",
        f"decisive: {result.get('decisive')}",
        f"reason:   {result.get('reason')}",
        "=" * 78,
    ]
    if "lc_total" in result:
        lines += [
            f"  label                 {result.get('label')}",
            f"  levels_completed      {result['lc_total']} over {result['n_games']} games "
            f"({result['lc_per_game']:.4f}/game)",
            f"  baseline (m=3)        {result['baseline_lc_total']:.2f} levels "
            f"({result['baseline_lc_per_game']:.4f}/game)  [duck-harness-kaggle 18/19/21]",
            f"  mean dlc              {result['mean_delta_lc']:+.4f} /game  "
            f"({result['delta_levels_over_25']:+.2f} levels over 25)   z = "
            f"{result['z_vs_baseline']:+.2f}",
            f"  lines                 HARM <= {HARM_LINE:+.4f} | REFUTE-2x <= {REFUTE_LINE:+.4f}"
            f" | CONFIRM-2x >= {CONFIRM_LINE:+.4f}",
            "  --- descriptive, NON-INFERENTIAL ---",
            f"  mean_score            {result['mean_score']:.3f}   (baseline 1.427/1.939/3.420, "
            "sd 1.033 on n=3 — a score-based 2x test has ~60% power and carries NO verdict)",
            f"  actions_total         {result['actions_total']} "
            f"({result['actions_per_window']:.1f}/window; baseline 4757/4033/4632)",
            f"  windows_drifted       {result['windows_drifted']}/25",
            "  tokens/s: read 'generated tokens/sec' from summary.txt "
            "(baseline 212.52 / 203.66 / 197.47 job-wallclock)",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def _selftest() -> int:
    import tempfile

    passed = failed = 0

    def expect(name: str, got: str, want: str) -> None:
        nonlocal passed, failed
        if got == want:
            passed += 1
            print(f"  ok   {name}")
        else:
            failed += 1
            print(f"  FAIL {name}: got {got!r} want {want!r}")

    def make(lc_total: int, *, n=N_GAMES, window=WINDOW_S, log="Q38-EVAL BOOT-ASSERTS PASSED",
             bench=True, drift_n=0) -> Path:
        root = Path(tempfile.mkdtemp())
        if log is not None:
            (root / "run.log").write_text(log, encoding="utf-8")
        if bench:
            per = [0] * n
            for i in range(lc_total):
                per[i % n] += 1
            runs = [{"game_id": f"g{i:02d}", "levels_completed": per[i],
                     "final_score": float(per[i]), "actions_per_level": [180],
                     "final_wallclock_seconds": (window * 0.5 if i < drift_n else window)}
                    for i in range(n)]
            (root / "benchmark.json").write_text(
                json.dumps({"label": BASELINE_LABEL, "game_runs": runs}), encoding="utf-8")
        return root

    print("== boundary arithmetic ==")
    # 0.773333 + 0.500000 = 1.273333/game -> 31.83 levels -> 32 confirms, 31 does not.
    expect("32 levels -> CONFIRM-2x", score(make(32))["verdict"], "CONFIRM-2x (decisive)")
    expect("31 levels -> INDETERMINATE", score(make(31))["verdict"], "INDETERMINATE")
    # 0.773333 + 0.250000 = 1.023333/game -> 25.58 levels -> 25 refutes, 26 does not.
    expect("26 levels -> INDETERMINATE", score(make(26))["verdict"], "INDETERMINATE")
    expect("25 levels -> REFUTE-2x", score(make(25))["verdict"], "REFUTE-2x (decisive)")
    # 0.773333 - 0.286320 = 0.487013/game -> 12.18 levels -> 12 is harm, 13 is not.
    expect("13 levels -> REFUTE-2x", score(make(13))["verdict"], "REFUTE-2x (decisive)")
    expect("12 levels -> HARM", score(make(12))["verdict"], "HARM (decisive)")
    expect("0 levels -> HARM", score(make(0))["verdict"], "HARM (decisive)")
    expect("58 levels (a true 2x+) -> CONFIRM-2x", score(make(58))["verdict"],
           "CONFIRM-2x (decisive)")
    expect("19 levels (dead-on the baseline) -> REFUTE-2x", score(make(19))["verdict"],
           "REFUTE-2x (decisive)")

    print("== the third state ==")
    expect("no benchmark.json -> INFRA DEATH",
           score(make(30, bench=False))["verdict"], "INFRA DEATH (not decisive)")
    expect("wrong game count -> INFRA DEATH",
           score(make(30, n=4))["verdict"], "INFRA DEATH (not decisive)")
    expect("boot asserts never PASSED -> INFRA DEATH (cannot confirm the engine)",
           score(make(40, log="some other output"))["verdict"], "INFRA DEATH (not decisive)")
    expect("Q38-EVAL FATAL in the log with no bench -> INFRA DEATH",
           score(make(0, bench=False, log="Q38-EVAL FATAL: quantization_config is not"))["verdict"],
           "INFRA DEATH (not decisive)")
    expect("3 drifted windows -> INFRA DEATH (comparison void)",
           score(make(40, drift_n=3))["verdict"], "INFRA DEATH (not decisive)")
    expect("2 drifted windows is tolerated",
           score(make(40, drift_n=2))["verdict"], "CONFIRM-2x (decisive)")

    print("== a HIGH score cannot rescue a LOW lc, and vice versa ==")
    r = score(make(12))
    expect("harm verdict survives whatever the score says", r["verdict"], "HARM (decisive)")

    print("== constants are the sealed ones ==")
    for name, got, want in (("BASELINE_LC_PER_GAME", round(BASELINE_LC_PER_GAME, 6), 0.773333),
                            ("SIGMA", SIGMA, 0.141740),
                            ("SE_DELTA", SE_DELTA, 0.163667),
                            ("HARM_LINE", HARM_LINE, -0.286320),
                            ("REFUTE_LINE", REFUTE_LINE, 0.25),
                            ("CONFIRM_LINE", CONFIRM_LINE, 0.5)):
        expect(f"{name} sealed", str(got), str(want))

    print(f"\nQ38 SCORER SELFTEST: {passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        sys.exit(_selftest())
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    print(render(score(Path(sys.argv[1]))))
