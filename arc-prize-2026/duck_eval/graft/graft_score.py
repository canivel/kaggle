"""SEALED verdict scorer for the GRAFT FLOOR arm. Written 2026-08-17 BEFORE the push.

`feedback_audit_the_instrument`: on 08-12 four ARC gates were broken at the moment they were
needed; on 08-14 a watch-rule was read with a retired threshold; on 08-17 the Q38 low arm died
because its pin-verification gate still asserted the PREVIOUS arm's signature and read the new
arm's correct behaviour as failure. So this scorer is written and selftested BEFORE the kernel
runs, and its constants are SEALED. Reading the data and then adjusting a number here is the
one thing that voids the screen.

THE 08-17 LESSON, ENCODED. A gate's LOGIC must be invariant across arms, but its EXPECTED
VALUES are a function of the arm. Freezing both is what turned a correct poisoning gate into a
landmine that fired precisely when the arm worked. Here that means:

  * SUCCESS IS NEVER DEFINED AS SILENCE. The arm is certified by the POSITIVE presence of its
    own markers (the FEATURES banner and the per-flag `armed` lines), never by the absence of
    someone else's.
  * The arm is ALSO defined by exclusion, so the forbidden flags are asserted ABSENT — a
    wrong-arm marker is fatal, not ignorable.
  * `install()` NEVER RAISES and degrades silently to stock. A silent stock run and a genuine
    null are byte-indistinguishable in `benchmark.json`. Therefore an unverifiable banner is
    INFRA DEATH, never a NULL/REFUTE. This is the whole reason the assertions exist.

CONSTANTS (sealed 2026-08-17, pre-push) — every one derived in
`learnings/war_room/graft_floor_prereg_2026-08-17.md`, which cites
`duck_eval/SCREEN_PROTOCOL.md` and `duck_eval/graft/bundle_audit_2026-08-17.md`:

  BASELINE      duck-harness-kaggle, m = 3 (P1 legal, P2 satisfied)
                gate_eval_v1 lc 18 | gate_eval_v2 lc 19 | duckgate_v1post lc 21
                per-game mean lc = 58 / (3 x 25) = 0.773333
  SIGMA         0.141740 lc/game, df 6 (SCREEN_PROTOCOL P3 standing pooled estimate)
  C(3)          2.02 (SCREEN_PROTOCOL sec 2, null10-measured 5th-pct multiplier)
  HARM line     -C(3) * sigma = -0.286320   (canonical K3", measured type-I 4.4 % at m=3)
  SIGNAL line   +C(3) * sigma = +0.286320   (the MIRROR of the same envelope; the symmetry is
                                             an ASSUMPTION [INF], not separately measured)

VERDICTS, in evaluation order (the third state is mandatory — the 08-14 LoRA canary landed in
exactly the state a two-state rule had no legal way to record):

  INFRA DEATH (not decisive)  any assertion below fails / no benchmark.json / != 25 games /
                              window drift > 5 % / the graft install cannot be confirmed
  HARM (decisive)             mean dlc <= -0.286320  -> the public floor is harmful to us
  SIGNAL (decisive)           mean dlc >= +0.286320  -> a real capability lift (lc_total >= 27)
  NULL (decisive)             in between: K3" PASS, no signal. Means "not a BIG effect",
                              NOT "no effect" — the bar is +28 % over the best lc ever recorded
                              on this rail (22, war_eval_v1). Must be written that way.

The score-based reading is reported and is explicitly NON-INFERENTIAL: the baseline's own
mean_score spread is 1.427 / 1.939 / 3.420 (sd 1.033 on n=3), so a score-based test has ~60 %
power and CANNOT carry a verdict. Levels-completed is the decision statistic
(SCREEN_PROTOCOL sec 0). A score movement may not be converted into a verdict after the fact.

    python duck_eval/graft/graft_score.py <pulled_kernel_dir>
    python duck_eval/graft/graft_score.py --selftest
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# ---- SEALED CONSTANTS -----------------------------------------------------
BASELINE_LABEL = "duck-harness-kaggle"
BASELINE_RUNS = {"gate_eval_v1": 18, "gate_eval_v2": 19, "duckgate_v1post": 21}
BASELINE_M = 3
N_GAMES = 25
BASELINE_LC_PER_GAME = 58 / (3 * 25)           # 0.7733333...
SIGMA = 0.141740
C_M3 = 2.02
# DERIVED, never transcribed. C(3)*sigma = 2.02 * 0.141740 = 0.2863148. The Q38 prereg sealed
# this as "0.286320", a 6-dp rounding that is NOT equal to the product; the difference is
# 5e-6 lc/game and moves no decision boundary (the lines land at lc_total 12.175 / 26.491
# either way), but a constant that disagrees with its own stated derivation is exactly the
# class of defect that has cost this campaign three lanes. So it is computed here.
HARM_LINE = -C_M3 * SIGMA                      # -0.2863148  canonical K3"
SIGNAL_LINE = +C_M3 * SIGMA                    # +0.2863148  mirror; the symmetry is [INF]
WINDOW_S = 7920.0
WINDOW_DRIFT_TOL = 0.05
BASELINE_SCORE_SPREAD = (1.427, 1.939, 3.420)

# The arm's identity, asserted POSITIVELY.
FLAGS_ON = ("efficiency", "retry_guard", "shortcircuit", "goalkeep", "hudmask")
ARMED_LINES_REQUIRED = ("[goalkeep] armed", "[hudmask] armed")
# The arm is defined by their exclusion; a wrong-arm marker is fatal.
FLAGS_FORBIDDEN = ("banking", "transfer", "clickmap")  # clickmap: v21 flag, NOT part of this arm (2026-08-20)
ARMED_LINES_FORBIDDEN = ("[banking] armed", "[clickmap] armed")
BANNER_RE = re.compile(r"TAAF_GRAFTS FEATURES=(\{.*?\})\s+API_VERSION=(\d+)")
GRAFTS_API_VERSION = 1
SERVED_MODEL = "vrfai/Qwen3.6-27B-FP8"
AUDITED_BUNDLE_MANIFEST_SHA = "df447f61caa181cca68049e28b139e02"

# A silent fall-back to stock is the single most dangerous outcome: it is a normal-looking run.
STOCK_FALLBACK_SIGNATURES = (
    "[taaf_grafts] install failed -> stock",
    "cell-12 graft failed",
)
INFRA_SIGNATURES = (
    "GRAFT-EVAL FATAL",
    "No supported CUDA architectures",
    "libcudart",
    "CUDA out of memory",
    "torch.OutOfMemoryError",
    "Timed out waiting for vLLM server",
    "Missing attached dataset path",
    "TAAF source bundle not found",
    "ModuleNotFoundError",
)


def _decode_cli_json_log(text: str) -> str:
    """Normalise CLI 2.2.3 `kernels logs` output to the runtime's actual stdout.

    INSTRUMENT FIX 2026-08-19 (world-normalisation only; no assertion or threshold touched):
    that CLI writes a JSON array of {stream_name, time, data} entries, so every quote in the
    runtime banner arrives backslash-escaped in the file's raw bytes and the textual flag
    check ('"efficiency"' in banner) can never match -- the v4 run's PERFECT banner was read
    as 'flags missing' and scored INFRA DEATH. The selftest had only ever seen plain-text
    fixtures: internal consistency is not correctness (the q38-low lesson, second instance).
    Decode JSON-array logs to their concatenated data payloads; pass all other formats
    through unchanged, so plain-text logs (and the original selftest checks) behave
    identically."""
    s = text.lstrip()
    if not s.startswith("["):
        return text
    try:
        entries = json.loads(text)
    except Exception:
        return text
    if not isinstance(entries, list):
        return text
    return "".join(e.get("data", "") for e in entries if isinstance(e, dict))


def _read_log(run_dir: Path) -> str:
    """Concatenate every plausible log artifact in the pull directory."""
    parts = []
    for pattern in ("*.log", "*.txt", "log*", "*.out"):
        for p in sorted(run_dir.glob(pattern)):
            try:
                parts.append(_decode_cli_json_log(p.read_text(encoding="utf-8", errors="replace")))
            except OSError:
                continue
    return "\n".join(parts)


def _infra(reason: str, **extra) -> dict:
    out = {"verdict": "INFRA DEATH", "decisive": False, "reason": reason}
    out.update(extra)
    return out


def certify_install(log: str) -> tuple[bool, str, dict]:
    """Certify from the RUNTIME banner that this arm's flags actually installed.

    Returns (ok, reason, facts). Never trusts the notebook source: `install()` silently ignores
    unknown flag names, so a typo produces a stock run that looks like a clean arm.
    """
    facts: dict = {}
    for sig in STOCK_FALLBACK_SIGNATURES:
        if sig in log:
            return False, f"graft install fell back to stock: {sig!r}", facts
    m = BANNER_RE.search(log)
    if not m:
        return False, "no 'TAAF_GRAFTS FEATURES=... API_VERSION=' banner in the log", facts
    raw, api = m.group(1), int(m.group(2))
    facts["banner"] = raw
    facts["api_version"] = api
    if api != GRAFTS_API_VERSION:
        return False, f"GRAFTS_API_VERSION={api}, sealed expectation {GRAFTS_API_VERSION}", facts
    # The banner is a Python dict repr; read flag names out of it textually rather than eval().
    present = {f for f in (*FLAGS_ON, *FLAGS_FORBIDDEN) if f"'{f}'" in raw or f'"{f}"' in raw}
    facts["features_present"] = sorted(present)
    missing = [f for f in FLAGS_ON if f not in present]
    if missing:
        return False, f"flags missing from the FEATURES banner: {missing}", facts
    wrong_arm = [f for f in FLAGS_FORBIDDEN if f in present]
    if wrong_arm:
        return False, f"FORBIDDEN flag present in FEATURES (wrong arm): {wrong_arm}", facts
    for line in ARMED_LINES_REQUIRED:
        if line not in log:
            return False, f"missing required armed line {line!r}", facts
    for line in ARMED_LINES_FORBIDDEN:
        if line in log:
            return False, f"forbidden armed line present (wrong arm): {line!r}", facts
    return True, "install certified", facts


def score(run_dir: Path) -> dict:
    log = _read_log(run_dir)
    path = run_dir / "benchmark.json"
    if not path.exists():
        signature = next((s for s in INFRA_SIGNATURES if s in log), None)
        return _infra(f"no benchmark.json; infra signature {signature!r}" if signature
                      else "no benchmark.json and no recognised infra signature")

    bench = json.loads(path.read_text(encoding="utf-8"))
    runs = bench.get("game_runs") or []
    if len(runs) != N_GAMES:
        return _infra(f"{len(runs)} game_runs, expected {N_GAMES}")

    ok, reason, facts = certify_install(log)
    if not ok:
        # A silent stock run is indistinguishable from a genuine null in benchmark.json. It may
        # never be scored as NULL/HARM/SIGNAL — that is how a lane gets killed by an instrument.
        return _infra(f"graft install not certified: {reason}", **facts)

    if SERVED_MODEL not in log:
        return _infra(f"served model {SERVED_MODEL!r} not confirmed in the log", **facts)

    lc_total = sum(int(r.get("levels_completed") or 0) for r in runs)
    scores = [float(r.get("final_score") or 0.0) for r in runs]
    mean_dlc = lc_total / N_GAMES - BASELINE_LC_PER_GAME
    won = sum(1 for r in runs if str(r.get("state")) == "won")

    if mean_dlc <= HARM_LINE:
        verdict, decisive = "HARM", True
    elif mean_dlc >= SIGNAL_LINE:
        verdict, decisive = "SIGNAL", True
    else:
        verdict, decisive = "NULL", True

    return {
        "verdict": verdict,
        "decisive": decisive,
        "reason": f"mean dlc {mean_dlc:+.6f} vs HARM {HARM_LINE:+.6f} / SIGNAL {SIGNAL_LINE:+.6f}",
        "lc_total": lc_total,
        "n_games": len(runs),
        "mean_dlc": mean_dlc,
        "mean_score": sum(scores) / N_GAMES,
        "games_won": won,
        "label": bench.get("label"),
        **facts,
    }


def render(result: dict) -> str:
    lines = [
        "=" * 78,
        "GRAFT FLOOR ARM — SEALED VERDICT",
        "=" * 78,
        f"  verdict               {result['verdict']}"
        f"{'' if result.get('decisive') else '   (NOT DECISIVE)'}",
        f"  reason                {result['reason']}",
    ]
    if "lc_total" in result:
        lines += [
            f"  levels_completed      {result['lc_total']} over {result['n_games']} games "
            f"(baseline {'/'.join(str(v) for v in BASELINE_RUNS.values())}, m={BASELINE_M})",
            f"  mean dlc / game       {result['mean_dlc']:+.6f}   "
            f"(HARM <= {HARM_LINE:+.6f} | SIGNAL >= {SIGNAL_LINE:+.6f})",
            f"  games reaching 'won'  {result['games_won']}   "
            "(0 in all 470 prior campaign game-runs — banking's unreachable trigger)",
            f"  mean_score            {result['mean_score']:.3f}   "
            f"(baseline {'/'.join(f'{v}' for v in BASELINE_SCORE_SPREAD)}, "
            "NON-INFERENTIAL — carries no verdict)",
        ]
    if "features_present" in result:
        lines.append(f"  FEATURES certified    {result['features_present']}")
    if result["verdict"] == "NULL":
        lines += [
            "",
            "  READ THIS AS: 'not a BIG effect', NOT 'no effect'. The SIGNAL bar (lc_total 27)",
            "  is +28% over the highest lc ever recorded on this rail (22, war_eval_v1).",
        ]
    lines.append("=" * 78)
    return "\n".join(lines)


# ---- SELFTEST -------------------------------------------------------------
GOOD_BANNER = (
    "TAAF_GRAFTS FEATURES={'efficiency': True, 'retry_guard': True, 'shortcircuit': True, "
    "'goalkeep': True, 'hudmask': True} API_VERSION=1\n"
    "[goalkeep] armed\n[hudmask] armed\n"
    f"vLLM server ready: {SERVED_MODEL}\n"
)


def _selftest() -> int:
    import tempfile

    failures = []

    checks_run = [0]

    def expect(name: str, got: str, want: str) -> None:
        checks_run[0] += 1
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")

    def make(per_game_lc, *, log=GOOD_BANNER, bench=True, n=N_GAMES, scores=None):
        root = Path(tempfile.mkdtemp())
        (root / "kernel.log").write_text(log, encoding="utf-8")
        if bench:
            per = list(per_game_lc) + [0] * (n - len(per_game_lc))
            runs = [{"game_id": f"g{i:02d}", "levels_completed": per[i], "state": "gave_up",
                     "final_score": (scores[i] if scores else 0.0)} for i in range(n)]
            (root / "benchmark.json").write_text(
                json.dumps({"label": BASELINE_LABEL, "n_passes": 1, "game_runs": runs}),
                encoding="utf-8")
        return root

    # --- verdict boundaries. mean dlc = lc/25 - 0.773333
    expect("lc_total 12 -> HARM", score(make([12]))["verdict"], "HARM")
    expect("lc_total 13 -> NULL", score(make([13]))["verdict"], "NULL")
    expect("lc_total 19 (= baseline) -> NULL", score(make([19]))["verdict"], "NULL")
    expect("lc_total 26 -> NULL", score(make([26]))["verdict"], "NULL")
    expect("lc_total 27 -> SIGNAL", score(make([27]))["verdict"], "SIGNAL")
    expect("lc_total 22 (best ever) -> NULL", score(make([22]))["verdict"], "NULL")

    # exact boundary arithmetic: 12.175 and 26.491 are the real lines
    assert 12 / 25 - BASELINE_LC_PER_GAME <= HARM_LINE, "12 must be HARM"
    assert 13 / 25 - BASELINE_LC_PER_GAME > HARM_LINE, "13 must not be HARM"
    assert 27 / 25 - BASELINE_LC_PER_GAME >= SIGNAL_LINE, "27 must be SIGNAL"
    assert 26 / 25 - BASELINE_LC_PER_GAME < SIGNAL_LINE, "26 must not be SIGNAL"

    # --- INFRA DEATH beats every levels verdict
    expect("no benchmark.json -> INFRA", score(make([19], bench=False))["verdict"], "INFRA DEATH")
    expect("wrong game count -> INFRA", score(make([19], n=24))["verdict"], "INFRA DEATH")
    expect("no banner -> INFRA", score(make([19], log="nothing here\n"))["verdict"], "INFRA DEATH")

    # --- REGRESSION 2026-08-19: the CLI 2.2.3 JSON log format must certify (v4's near-miss).
    def cli_json(payload: str) -> str:
        return json.dumps([{"stream_name": "stdout", "time": 1.0, "data": line + "\n"}
                           for line in payload.splitlines()])
    RUNTIME_BANNER = (
        'TAAF_GRAFTS FEATURES={"efficiency":true,"goalkeep":true,"hudmask":true,'
        '"retry_guard":true,"shortcircuit":true} API_VERSION=1\n'
        "[goalkeep] armed\n[hudmask] armed\n"
        f"vLLM server ready: {SERVED_MODEL}\n"
    )
    expect("CLI-JSON log, good banner -> NULL",
           score(make([19], log=cli_json(RUNTIME_BANNER)))["verdict"], "NULL")
    bad = RUNTIME_BANNER.replace('"goalkeep":true,', "")
    expect("CLI-JSON log, goalkeep missing -> INFRA",
           score(make([19], log=cli_json(bad)))["verdict"], "INFRA DEATH")
    wrong = RUNTIME_BANNER.replace('"efficiency":true,', '"banking":true,"efficiency":true,')
    expect("CLI-JSON log, banking armed -> INFRA",
           score(make([19], log=cli_json(wrong + "[banking] armed\n")))["verdict"], "INFRA DEATH")

    # --- THE CENTRAL REGRESSION: a silent stock fallback must NEVER be scored as NULL.
    for sig in STOCK_FALLBACK_SIGNATURES:
        r = score(make([19], log=GOOD_BANNER + sig + "\n"))
        expect(f"stock fallback {sig!r} -> INFRA", r["verdict"], "INFRA DEATH")
    # ... including when the levels number looks like a clean SIGNAL.
    r = score(make([30], log="[taaf_grafts] install failed -> stock: boom\n"))
    expect("stock fallback with SIGNAL-looking lc -> INFRA", r["verdict"], "INFRA DEATH")

    # --- wrong-arm markers are fatal (the 08-17 Q38-low lesson, mirrored)
    r = score(make([19], log=GOOD_BANNER + "[banking] armed\n"))
    expect("banking armed -> INFRA", r["verdict"], "INFRA DEATH")
    r = score(make([19], log=GOOD_BANNER.replace(
        "'hudmask': True}", "'hudmask': True, 'banking': True}")))
    expect("banking in FEATURES -> INFRA", r["verdict"], "INFRA DEATH")

    # --- a missing required flag is fatal (silently-ignored typo detection)
    r = score(make([19], log=GOOD_BANNER.replace("'goalkeep': True, ", "")))
    expect("goalkeep missing from FEATURES -> INFRA", r["verdict"], "INFRA DEATH")
    r = score(make([19], log=GOOD_BANNER.replace("[hudmask] armed\n", "")))
    expect("hudmask armed line missing -> INFRA", r["verdict"], "INFRA DEATH")

    # --- API version mismatch fails CLOSED
    r = score(make([19], log=GOOD_BANNER.replace("API_VERSION=1", "API_VERSION=2")))
    expect("API_VERSION=2 -> INFRA", r["verdict"], "INFRA DEATH")

    # --- served engine must be confirmed (no engine confound)
    r = score(make([19], log=GOOD_BANNER.replace(SERVED_MODEL, "Qwen/Qwen3.8-27B-FP8")))
    expect("wrong served model -> INFRA", r["verdict"], "INFRA DEATH")

    # --- score is reported but never decides
    r = score(make([19], scores=[99.0] * N_GAMES))
    expect("huge score with baseline lc -> still NULL", r["verdict"], "NULL")
    r = score(make([12], scores=[99.0] * N_GAMES))
    expect("huge score with harmful lc -> still HARM", r["verdict"], "HARM")

    # --- 'won' is surfaced (banking reachability evidence for a future arm)
    assert score(make([19]))["games_won"] == 0

    # --- sealed constants are internally consistent
    assert abs(HARM_LINE + C_M3 * SIGMA) < 1e-12, "HARM line != -C(3)*sigma"
    assert abs(SIGNAL_LINE - C_M3 * SIGMA) < 1e-12, "SIGNAL line != +C(3)*sigma"
    # ... and it must still agree with SCREEN_PROTOCOL sec 2's published m=3 line of -0.286.
    assert round(HARM_LINE, 3) == -0.286, f"HARM line {HARM_LINE} != protocol's -0.286"
    assert abs(BASELINE_LC_PER_GAME - 58 / 75) < 1e-12
    assert sum(BASELINE_RUNS.values()) == 58 and len(BASELINE_RUNS) == BASELINE_M

    # COUNTED, not transcribed (2026-08-19: the hardcoded '22' survived three new checks
    # silently -- same class as the builder's stale banner print and the prereg's 0.286320).
    n_checks = checks_run[0]
    if failures:
        print(f"SELFTEST FAILED ({len(failures)} of {n_checks}):")
        for f in failures:
            print("  -", f)
        return 1
    print(f"selftest OK ({n_checks}/{n_checks} checks, 0 failures)")
    print(f"  HARM   lc_total <= 12   (mean dlc <= {HARM_LINE:+.6f})")
    print(f"  NULL   lc_total 13..26")
    print(f"  SIGNAL lc_total >= 27   (mean dlc >= {SIGNAL_LINE:+.6f})")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        raise SystemExit(_selftest())
    if len(sys.argv) != 2:
        raise SystemExit(__doc__.strip().splitlines()[-2].strip())
    result = score(Path(sys.argv[1]))
    print(render(result))
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
