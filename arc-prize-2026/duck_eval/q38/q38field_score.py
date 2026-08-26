"""SEALED verdict scorer for the Q38 FIELD-FLOOR ADOPTION arm (`arc3-q38-field-eval` v1).

Written 2026-08-20 ~09:15 EDT, BEFORE the kernel reached COMPLETE. Constants are SEALED:
reading the data and then adjusting a number here is the one thing that voids the screen.

WHY THIS FILE EXISTS (the instrument gap it closes)
---------------------------------------------------
`feedback_audit_the_instrument`: audit the gate BEFORE the data lands. On 2026-08-20 the two
scorers on disk were checked against THIS arm and BOTH would have mis-read it:

  * `duck_eval/graft/graft_score.py` -> **false INFRA DEATH on two independent counts**:
      1. `ARMED_LINES_REQUIRED = ("[goalkeep] armed", "[hudmask] armed")`. The field arm is a
         byte-faithful FOYSAL rebase with **grafts 0 by construction** (build-time byte-audit,
         prereg sec 1). Those lines are ABSENT by design -> "missing required armed line".
      2. `SERVED_MODEL = "vrfai/Qwen3.6-27B-FP8"`. This arm serves `Qwen/Qwen3.8-27B-FP8`.
  * `duck_eval/q38/q38_score.py` -> wrong sealed question. Its bands are REFUTE-2x/CONFIRM-2x
    (the 08-15 engine-swap claim), not the K3" HARM/NULL/SIGNAL lines this arm was sealed on.

Had 14:45 arrived and either been reached for, the arm would have been voided by the
instrument rather than by the evidence -- the 08-12 failure mode, for the fifth time.

SEALED CONSTANTS -- every one transcribed from
`learnings/war_room/q38_field_prereg_2026-08-20.md` sec 2 and sec 3. No new number is invented here.

  BASELINE      duck-harness-kaggle, m = 3; lc 18 | 19 | 21  -> 58 / (3*25) = 0.773333 lc/game
  SIGMA         0.141740 lc/game, df 6   (SCREEN_PROTOCOL sec P3 standing pooled estimate)
  C(3)          2.02                     (SCREEN_PROTOCOL sec 2, null10-measured)
  HARM line     -C(3)*SIGMA = -0.2863148 -> lc_total <= 12
  SIGNAL line   +C(3)*SIGMA = +0.2863148 -> lc_total >= 27
  NULL          lc_total 13..26

  These are the SAME lines as the graft seal (prereg sec 3: "K3'' + mirror"), deliberately, so
  the two arms are read on one ruler. DERIVED from C_M3*SIGMA, never transcribed -- the prereg's
  "0.286320" is a 6-dp rounding that is not equal to the product (5e-6 lc/game; moves no
  boundary, since the lines land at lc_total 12.175 / 26.491 either way).

RUNTIME CERTIFICATION (prereg sec 2) -- any failure => INFRA DEATH, never NULL/ADOPT:
  1. benchmark.json present with n_games = 25
  2. vLLM serve banner shows served model `Qwen/Qwen3.8-27B-FP8` (the repacked Kaggle Model;
     `feedback_kaggle_model_attach` -- attach is the silent-drop trap)
  3. bundle identity: NO `reasoning_effort` anywhere in the log (absent => template xhigh, which
     is the whole point of the arm; a pinned effort means a DIFFERENT config ran)
  4. no stock-fallback / ModuleNotFoundError-class death
  PLUS the inverse-arm guard: this arm has NO grafts. Any TAAF_GRAFTS armed marker in the log
  means a wrong-arm artifact was scored -- fatal, exactly as a missing marker is fatal for the
  graft arm. (The graft scorer guards presence; this one must guard ABSENCE. Same principle,
  mirrored -- that asymmetry is what made a shared scorer impossible.)

QUEUE-HEAD GATE (coordinator-ruled, prereg sec 3): COMPLETE + certification above by 18:00 EDT.
The lc/score bands do NOT gate the queue head -- the board-verified 2.23 carries the draw
decision; our run is its certification, not its audition. `--certify-only` answers exactly that
question and NOTHING else, so the 18:00 call cannot be contaminated by having seen the score.

    python duck_eval/q38/q38field_score.py <pulled_kernel_dir>
    python duck_eval/q38/q38field_score.py <pulled_kernel_dir> --certify-only
    python duck_eval/q38/q38field_score.py --selftest
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "graft"))
# Reuse the 08-19 instrument fix (CLI 2.2.3 writes logs as a JSON array, which made every
# quote in the runtime banner arrive backslash-escaped and silently broke textual checks).
# Importing rather than re-typing it is deliberate: a copy would re-introduce the bug the
# moment that fix is improved. Only world-normalisation is shared -- no threshold, no assertion.
from graft_score import _decode_cli_json_log, _read_log, _infra  # noqa: E402

# ---- SEALED CONSTANTS -----------------------------------------------------
BASELINE_LABEL = "duck-harness-kaggle"
BASELINE_RUNS = {"gate_eval_v1": 18, "gate_eval_v2": 19, "duckgate_v1post": 21}
BASELINE_M = 3
N_GAMES = 25
BASELINE_LC_PER_GAME = 58 / (3 * 25)           # 0.7733333...
SIGMA = 0.141740
C_M3 = 2.02
HARM_LINE = -C_M3 * SIGMA                      # -0.2863148  canonical K3"
SIGNAL_LINE = +C_M3 * SIGMA                    # +0.2863148  mirror

# Non-inferential context (prereg sec 3). Recorded, never used to pick a verdict.
BASELINE_SCORE_SPREAD = (1.427, 1.939, 3.420)
Q38_MEDIUM_SCORE = 2.795
FIELD_SCORE_EXPECTATION = 3.4                  # "clears the spread max" if the class reproduces

# ---- RUNTIME CERTIFICATION (prereg sec 2) ---------------------------------
SERVED_MODEL = "Qwen/Qwen3.8-27B-FP8"
EFFORT_TOKEN = "reasoning_effort"              # MUST be absent => template xhigh
# This arm carries NO grafts. Presence of any of these means a wrong-arm artifact.
GRAFT_MARKERS_FORBIDDEN = (
    "TAAF_GRAFTS FEATURES=",
    "[goalkeep] armed",
    "[hudmask] armed",
    "[banking] armed",
    "[clickmap] armed",
)
STOCK_FALLBACK_SIGNATURES = (
    "[taaf_grafts] install failed -> stock",
    "cell-12 graft failed",
)
INFRA_SIGNATURES = (
    "No supported CUDA architectures",
    "libcudart",
    "CUDA out of memory",
    "torch.OutOfMemoryError",
    "Timed out waiting for vLLM server",
    "Missing attached dataset path",
    "ModuleNotFoundError",
)

# ARM-IDENTITY MARKERS (2026-08-22, local_gate cross-arm finding: this scorer CERTIFIED the
# private-base artifact — a superset composition carrying every marker above — and emitted a
# cross-arm verdict. A reader must refuse other arms' artifacts on IDENTITY, not just config.)
ARM_IDENTITY_REQUIRED = ("anim-20260807",)          # the 08-07 anim bundle, this arm's harness
ARM_IDENTITY_FORBIDDEN = ("PRIVATE-ARM BANNER", "model-20260815-q38-p1")
# 2026-08-22 correction: the solver banner echoes the PICKLED DEFAULT (7920.0) on EVERY
# budget sibling (printed pre-assignment), so echo-string forbids can never fire. The
# budget siblings are excluded by the WALLCLOCK WINDOW below instead (exp-34 rule, done
# with an observable that actually discriminates).
WALL_WINDOW = (7700.0, 8300.0)  # this arm's budget signature (T1 = 7920s)


def _live_signature_hits(text: str, signatures: tuple) -> list:
    """INSTRUMENT FIX 2026-08-22 (local_gate finding; 4th 'internal consistency is not
    correctness' instance, this time OURS): two INFRA_SIGNATURES are raise-message
    LITERALS in the notebook's own setup source, which Kaggle echoes into every log --
    a flat substring scan therefore INFRA-DEATHs every real artifact. A signature only
    counts as LIVE when it appears outside an echoed-source context: not inside a
    quoted literal (quote-parity + adjacent-quote checks) and not on a papermill
    source-echo line (`---->  3 ...` / numbered source lines)."""
    import re as _re
    hits = []
    for sig in signatures:
        for line in text.splitlines():
            i = line.find(sig)
            if i < 0:
                continue
            before, after = line[:i], line[i + len(sig):]
            if before.count('"') % 2 == 1 or before.count("'") % 2 == 1:
                continue  # inside a quoted literal (echoed source)
            if after.lstrip()[:1] in ('"', "'"):
                continue
            if _re.match(r"\s*(-+>)?\s*\d+ ", line):
                continue  # papermill source echo
            hits.append(sig)
            break
    return hits


def certify_runtime(log: str) -> tuple[bool, str, dict]:
    """Prereg sec 2, in order. Returns (ok, reason, facts).

    Never trusts the notebook source: what was pushed and what ran can differ (Kaggle attaches
    the LATEST dataset version, so a bundle content change after seal is invisible at build time
    and only detectable here).
    """
    facts: dict = {
        "served_model_confirmed": SERVED_MODEL in log,
        "effort_token_present": EFFORT_TOKEN in log,
        "graft_markers": [m for m in GRAFT_MARKERS_FORBIDDEN if m in log],
        "stock_fallback": [s for s in STOCK_FALLBACK_SIGNATURES if s in log],
        "infra_signatures": _live_signature_hits(log, INFRA_SIGNATURES),
    }
    if facts["stock_fallback"]:
        return False, f"stock fallback signature present: {facts['stock_fallback'][0]!r}", facts
    if facts["infra_signatures"]:
        return False, f"infra signature present: {facts['infra_signatures'][0]!r}", facts
    if not facts["served_model_confirmed"]:
        return False, f"served model {SERVED_MODEL!r} not confirmed in the log", facts
    if facts["effort_token_present"]:
        return False, (f"{EFFORT_TOKEN!r} present in the log -- effort was PINNED; the unpinned "
                       f"(xhigh) config that defines this arm did NOT run"), facts
    if facts["graft_markers"]:
        return False, (f"graft marker present (wrong arm -- this arm has grafts 0 by "
                       f"construction): {facts['graft_markers'][0]!r}"), facts
    for m in ARM_IDENTITY_REQUIRED:
        if m not in log:
            return False, f"arm-identity marker {m!r} ABSENT -- not this arm's artifact", facts
    for m in ARM_IDENTITY_FORBIDDEN:
        if m in log:
            return False, f"foreign arm-identity marker {m!r} present -- refuse cross-arm read", facts
    return True, "runtime certified", facts


def score(run_dir: Path, certify_only: bool = False) -> dict:
    log = _read_log(run_dir)
    path = run_dir / "benchmark.json"
    if not path.exists():
        sigs = _live_signature_hits(log, INFRA_SIGNATURES)
        signature = sigs[0] if sigs else None
        return _infra(f"no benchmark.json; infra signature {signature!r}" if signature
                      else "no benchmark.json and no recognised infra signature")

    bench = json.loads(path.read_text(encoding="utf-8"))
    _walls = sorted(float(g.get("final_wallclock_seconds") or 0.0)
                    for g in (bench.get("game_runs") or []))
    if _walls:
        _wm = _walls[len(_walls) // 2]
        if not (WALL_WINDOW[0] <= _wm <= WALL_WINDOW[1]):
            return _infra(f"wallclock median {_wm:.0f}s outside this arm's window "
                          f"{WALL_WINDOW} - budget-sibling artifact, refuse cross-arm read")
    runs = bench.get("game_runs") or []
    if len(runs) != N_GAMES:
        return _infra(f"{len(runs)} game_runs, expected {N_GAMES}")

    ok, reason, facts = certify_runtime(log)
    if not ok:
        # A silent stock/wrong-config run is indistinguishable from a genuine null inside
        # benchmark.json. It may never be scored as NULL/HARM/SIGNAL.
        return _infra(f"runtime not certified: {reason}", **facts)

    if certify_only:
        # The 18:00 queue-head call. Deliberately returns BEFORE lc/score are computed so the
        # decision cannot be contaminated by having seen them (prereg sec 3).
        return {"verdict": "CERTIFIED", "decisive": True, "certify_only": True,
                "reason": "COMPLETE + prereg sec 2 certification passed; queue-head gate OPEN",
                **facts}

    lc_total = sum(int(r.get("levels_completed") or 0) for r in runs)
    scores = [float(r.get("final_score") or 0.0) for r in runs]
    mean_score = sum(scores) / len(scores)
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
        "mean_dlc": mean_dlc,
        "won": won,
        "mean_score": mean_score,
        "score_note": (
            f"NON-INFERENTIAL. baseline spread {BASELINE_SCORE_SPREAD} (sd 1.033, n=3); "
            f"Q38-medium {Q38_MEDIUM_SCORE}; field-class expectation >= {FIELD_SCORE_EXPECTATION}. "
            f"No verdict follows from score alone (prereg sec 3)."
        ),
        **facts,
    }


# ---- SELFTEST -------------------------------------------------------------
def _selftest() -> int:
    import tempfile

    GOOD_LOG = (
        "INFO installing wheels from wheelhouse\n"
        "taaf.kaggle: source bundle = /kaggle/input/taaf-kaggle-source-anim-20260807-anim\n"
        f"vLLM server ready: {SERVED_MODEL}\n"
        "MULTIMODAL_CONTEXT=current_grid\n"
        "benchmark complete\n"
    )

    def make(lcs, log=GOOD_LOG, scores=None, states=None):
        d = Path(tempfile.mkdtemp())
        runs = []
        for i, lc in enumerate(lcs):
            runs.append({
                "levels_completed": lc,
                "final_score": (scores[i] if scores else 0.0),
                "state": (states[i] if states else "lost"),
                "final_wallclock_seconds": 7920.0,
            })
        (d / "benchmark.json").write_text(json.dumps({"game_runs": runs}), encoding="utf-8")
        (d / "run.log").write_text(log, encoding="utf-8")
        return d

    def pad(total):
        """25 games summing to `total`, so the band boundaries are exercised exactly."""
        base = [0] * N_GAMES
        i = 0
        while total > 0:
            base[i % N_GAMES] += 1
            total -= 1
            i += 1
        return base

    fails = []

    def expect(label, got, want):
        ok = got == want
        print(f"  {'ok  ' if ok else 'FAIL'} {label}: got {got!r} want {want!r}")
        if not ok:
            fails.append(label)

    print("band boundaries (the sealed lines, exercised on both sides):")
    expect("lc_total 12 -> HARM", score(make(pad(12)))["verdict"], "HARM")
    expect("lc_total 13 -> NULL", score(make(pad(13)))["verdict"], "NULL")
    expect("lc_total 26 -> NULL", score(make(pad(26)))["verdict"], "NULL")
    expect("lc_total 27 -> SIGNAL", score(make(pad(27)))["verdict"], "SIGNAL")

    print("\nNEGATIVE CONTROLS -- every certification gate must actually be able to REFUSE\n"
          "(`feedback_guard_never_fired`: a guard that never fired may be one that CANNOT):")
    expect("wrong served model -> INFRA DEATH",
           score(make(pad(30), log="vLLM server ready: vrfai/Qwen3.6-27B-FP8\n"))["verdict"],
           "INFRA DEATH")
    expect("reasoning_effort PINNED -> INFRA DEATH",
           score(make(pad(30), log=GOOD_LOG + 'reasoning_effort="medium"\n'))["verdict"],
           "INFRA DEATH")
    expect("graft marker present (wrong arm) -> INFRA DEATH",
           score(make(pad(30), log=GOOD_LOG + "[goalkeep] armed\n"))["verdict"],
           "INFRA DEATH")
    expect("clickmap armed (v21 wave) -> INFRA DEATH",
           score(make(pad(30), log=GOOD_LOG + "[clickmap] armed\n"))["verdict"],
           "INFRA DEATH")
    expect("stock fallback -> INFRA DEATH",
           score(make(pad(30), log=GOOD_LOG + "[taaf_grafts] install failed -> stock\n"))["verdict"],
           "INFRA DEATH")
    expect("ModuleNotFoundError -> INFRA DEATH",
           score(make(pad(30), log=GOOD_LOG + "ModuleNotFoundError: no module named x\n"))["verdict"],
           "INFRA DEATH")
    expect("wrong game count -> INFRA DEATH", score(make([1] * 24))["verdict"], "INFRA DEATH")

    print("\nthe good path must NOT be refused (a gate that refuses everything is also broken):")
    expect("clean log + lc 30 -> SIGNAL", score(make(pad(30)))["verdict"], "SIGNAL")

    print("\n--certify-only must answer certification and NOT leak the score:")
    r = score(make(pad(30)), certify_only=True)
    expect("clean -> CERTIFIED", r["verdict"], "CERTIFIED")
    expect("no lc_total leaked", "lc_total" in r, False)
    expect("no mean_score leaked", "mean_score" in r, False)
    expect("uncertified -> INFRA DEATH",
           score(make(pad(30), log="vLLM server ready: other\n"), certify_only=True)["verdict"],
           "INFRA DEATH")

    print("\nJSON-array log decoding (the 08-19 instrument fix) must survive the import:")
    d = Path(tempfile.mkdtemp())
    (d / "benchmark.json").write_text(
        json.dumps({"game_runs": [{"levels_completed": v, "final_score": 0.0, "state": "lost",
                                   "final_wallclock_seconds": 7920.0}
                                  for v in pad(30)]}), encoding="utf-8")
    (d / "run.log").write_text(
        json.dumps([{"stream_name": "stdout", "time": 0, "data": GOOD_LOG}]), encoding="utf-8")
    expect("JSON-array log -> SIGNAL (not INFRA)", score(d)["verdict"], "SIGNAL")

    print("\nderivation check (constants must agree with their stated derivation):")
    expect("HARM line == -C(3)*SIGMA", round(HARM_LINE, 7), round(-2.02 * 0.141740, 7))
    expect("baseline lc/game == 58/75", round(BASELINE_LC_PER_GAME, 6), round(58 / 75, 6))

    print(f"\n{'SELFTEST FAILED: ' + ', '.join(fails) if fails else 'SELFTEST OK (all gates fire, good path passes)'}")
    return 1 if fails else 0


def main() -> int:
    args = [a for a in sys.argv[1:]]
    if "--selftest" in args:
        return _selftest()
    certify_only = "--certify-only" in args
    positional = [a for a in args if not a.startswith("--")]
    if not positional:
        print(__doc__)
        return 2
    run_dir = Path(positional[0])
    if not run_dir.is_dir():
        print(f"not a directory: {run_dir}")
        return 2
    result = score(run_dir, certify_only=certify_only)
    print(json.dumps(result, indent=2, default=str))
    if result["verdict"] == "NULL":
        print("\n  READ THIS AS: 'not a BIG effect', NOT 'no effect'. The SIGNAL bar (lc_total 27)")
        print("  is a >=65% lift; one seed cannot separate a real small lift from noise.")
    if result["verdict"] == "INFRA DEATH":
        print("\n  NOT DECISIVE. This says the instrument could not certify what ran -- it says")
        print("  NOTHING about the arm's capability. Do not fold it into any ledger.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
