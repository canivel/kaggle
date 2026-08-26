"""Sealed scorer - exec-WM arm (execwm_prereg_2026-08-25.md).

Certification (exp-34 rule: identity positive AND negative):
  REQUIRED  "[execwm] armed"            the wrapper constructed on this run
  REQUIRED  served-model + bundle markers of the certified field-floor vehicle
  FORBIDDEN foreign arm markers ([notes], [p2], grafts, private banner)

The arm's delivery instruments come FIRST from the job-dir report files
(execwm/*.json, written by the analyzer wrapper -- immune to the P1 0-byte-log
failure class) and only fall back to log markers when reports are absent.

Screening primary (matches the campaign standard): lc_total + trim1.
Graceful-degradation clause: if fallback rate is 100% (exec-WM never planned),
the verdict must be read as FLOOR-EQUIVALENT, not as an exec-WM result.

Usage:
    python duck_eval/execwm/execwm_score.py <pull_dir>
    python duck_eval/execwm/execwm_score.py --selftest
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SERVED_MODEL = "Qwen/Qwen3.8-27B-FP8"
BUNDLE_MARKER = "anim-20260807"
ARMED_MARKER = "[execwm] armed"
FOREIGN = ("PRIVATE-ARM BANNER", "TAAF_GRAFTS FEATURES", "[goalkeep] armed",
           "[clickmap] armed", "[banking] armed", "[notes] persistent-namespace armed",
           "[p2] reset semantics OK")
N_GAMES = 25


class ScoreError(RuntimeError):
    """Malformed artifact: fail loudly, never return zeros."""


def _read_log(run_dir: Path) -> str:
    parts = []
    for pat in ("*.log", "*.txt"):
        for p in sorted(run_dir.glob(pat)):
            try:
                parts.append(p.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                pass
    log_dir = run_dir / "log"
    if log_dir.is_dir():
        for p in sorted(log_dir.rglob("*")):
            if p.is_file():
                try:
                    parts.append(p.read_text(encoding="utf-8", errors="replace"))
                except OSError:
                    pass
    return "\n".join(parts)


def _read_bench(run_dir: Path) -> dict:
    bp = run_dir / "benchmark.json"
    if not bp.is_file():
        found = sorted(run_dir.rglob("benchmark.json"))
        if not found:
            raise ScoreError(f"no benchmark.json under {run_dir}")
        bp = found[0]
    bench = json.loads(bp.read_text(encoding="utf-8"))
    if not isinstance(bench, dict):
        raise ScoreError("benchmark.json root is not an object")
    runs = bench.get("game_runs")
    if not isinstance(runs, list) or not runs:
        raise ScoreError("benchmark.json has no game_runs")
    for i, r in enumerate(runs):
        if not isinstance(r, dict):
            raise ScoreError(f"game_runs[{i}] is not an object")
        if "levels_completed" not in r:
            raise ScoreError(f"game_runs[{i}] missing levels_completed")
        if "final_score" not in r:
            raise ScoreError(f"game_runs[{i}] missing final_score")
        if not isinstance(r.get("actions_per_level"), list):
            raise ScoreError(f"game_runs[{i}] missing actions_per_level list")
    return bench


def _read_reports(run_dir: Path) -> list[dict]:
    reports = []
    for p in sorted(run_dir.rglob("execwm/*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ScoreError(f"unreadable exec-WM report {p.name}: {exc}")
        if isinstance(data, dict) and data.get("armed"):
            reports.append(data)
    return reports


def certify(run_dir: Path) -> tuple[bool, str, dict]:
    log = _read_log(run_dir)
    facts: dict = {}
    if not log.strip():
        return False, "no log captured (P1 0-byte class) - certification undefined", facts
    if SERVED_MODEL not in log:
        return False, f"served model {SERVED_MODEL!r} absent", facts
    if BUNDLE_MARKER not in log:
        return False, f"bundle marker {BUNDLE_MARKER!r} absent", facts
    if ARMED_MARKER not in log:
        return False, f"required marker {ARMED_MARKER!r} absent - exec-WM never armed", facts
    for m in FOREIGN:
        if m in log:
            return False, f"foreign arm marker {m!r} present", facts
    facts["armed_count"] = log.count(ARMED_MARKER)
    return True, "certified", facts


def delivery(run_dir: Path) -> dict:
    """Per-phase instruments, report-file first, log-markers fallback."""
    reports = _read_reports(run_dir)
    out = {"source": "reports" if reports else "log",
           "games_reported": len(reports),
           "levels_seen": 0, "levels_planned": 0, "levels_cleared_by_plan": 0,
           "levels_cleared_by_explore": 0, "levels_fallback": 0,
           "fallback_reasons": {}, "rules_verified": 0, "rules_rejected": 0,
           "breaks": 0, "llm_calls": 0, "wm_actions": 0, "disabled_games": 0}
    if reports:
        for rep in reports:
            out["llm_calls"] += int(rep.get("llm_calls") or 0)
            out["wm_actions"] += int(rep.get("actions_executed") or 0)
            if rep.get("disabled_reason"):
                out["disabled_games"] += 1
            for lv in (rep.get("levels") or {}).values():
                out["levels_seen"] += 1
                out["breaks"] += int(lv.get("breaks") or 0)
                if lv.get("cleared_via") == "plan":
                    out["levels_cleared_by_plan"] += 1
                if lv.get("cleared_via") == "explore":
                    out["levels_cleared_by_explore"] += 1
                if lv.get("fallback"):
                    out["levels_fallback"] += 1
                    reason = str(lv.get("fallback_reason") or "?")
                    out["fallback_reasons"][reason] = \
                        out["fallback_reasons"].get(reason, 0) + 1
                if lv.get("phase") == "P" or lv.get("plans_run"):
                    out["levels_planned"] += 1
                for rd in (lv.get("rules") or {}).values():
                    if rd.get("verified"):
                        out["rules_verified"] += 1
                    else:
                        out["rules_rejected"] += 1
    else:
        log = _read_log(run_dir)
        out["levels_cleared_by_plan"] = log.count("CLEARED via=plan")
        out["levels_cleared_by_explore"] = log.count("CLEARED via=explore")
        out["levels_fallback"] = log.count("] fallback reason=") + log.count(" fallback reason=")
        out["breaks"] = log.count(" BREAK ")
    seen = max(1, out["levels_seen"]) if reports else None
    out["fallback_rate"] = (out["levels_fallback"] / seen) if seen else None
    return out


def score(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    bench = _read_bench(run_dir)
    runs = bench["game_runs"]
    lc = sum(int(r.get("levels_completed") or 0) for r in runs)
    scores = [float(r.get("final_score") or 0.0) for r in runs]
    mean_score = sum(scores) / len(runs)
    trim1 = (sum(scores) - max(scores)) / (len(runs) - 1) if len(runs) > 1 else 0.0
    actions = sum(sum(r.get("actions_per_level") or []) for r in runs)
    cert_ok, cert_reason, cert_facts = certify(run_dir)
    deliv = delivery(run_dir)
    verdict = "CERTIFIED" if cert_ok else "INFRA DEATH"
    if cert_ok and deliv["levels_cleared_by_plan"] == 0 and \
            (deliv["fallback_rate"] in (None, 1.0)):
        verdict = "CERTIFIED-FLOOR-EQUIVALENT"
    return {
        "n_games": len(runs),
        "lc_total": lc,
        "won": sum(1 for r in runs if str(r.get("state")) == "won"),
        "mean_score": mean_score,
        "trim1": trim1,
        "total_actions": actions,
        "verdict": verdict,
        "reason": cert_reason,
        "cert_facts": cert_facts,
        "delivery": deliv,
    }


# ---------------------------------------------------------------------------
def _selftest() -> int:
    import tempfile
    ok = 0
    fail = 0

    def check(name, cond):
        nonlocal ok, fail
        if cond:
            ok += 1
        else:
            fail += 1
            print(f"SELFTEST FAIL: {name}")

    good_run = {"game_id": "g0", "levels_completed": 2, "final_score": 5.0,
                "actions_per_level": [3, 4], "state": "gave_up"}
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        # healthy synthetic fixture (the arm-mismatch lesson: scorer must
        # accept a HEALTHY artifact before the data lands)
        (d / "benchmark.json").write_text(json.dumps(
            {"label": "x", "game_runs": [good_run] * 3}), encoding="utf-8")
        (d / "kernel.log").write_text(
            f"boot {SERVED_MODEL} bundle {BUNDLE_MARKER}\n"
            f"{ARMED_MARKER} v1 game=g0 llm=on\n", encoding="utf-8")
        rep_dir = d / "execwm"
        rep_dir.mkdir()
        (rep_dir / "g0_p0.json").write_text(json.dumps({
            "armed": True, "game_id": "g0", "llm_calls": 1,
            "actions_executed": 40, "disabled_reason": None,
            "levels": {"1": {"phase": "P", "plans_run": 2, "breaks": 1,
                             "fallback": False, "fallback_reason": "",
                             "cleared_via": "plan",
                             "rules": {"ACTION1": {"verified": True},
                                       "ACTION9": {"verified": False}}}}}),
            encoding="utf-8")
        res = score(d)
        check("healthy certifies", res["verdict"] == "CERTIFIED")
        check("healthy lc", res["lc_total"] == 6)
        check("healthy plan clears", res["delivery"]["levels_cleared_by_plan"] == 1)
        check("healthy verified rules", res["delivery"]["rules_verified"] == 1)

        # floor-equivalent: armed but zero plan clears + full fallback
        (rep_dir / "g0_p0.json").write_text(json.dumps({
            "armed": True, "game_id": "g0", "llm_calls": 0,
            "actions_executed": 10, "disabled_reason": None,
            "levels": {"1": {"phase": "F", "plans_run": 0, "breaks": 0,
                             "fallback": True, "fallback_reason": "no-verified-model",
                             "cleared_via": None, "rules": {}}}}), encoding="utf-8")
        res = score(d)
        check("floor-equivalent verdict",
              res["verdict"] == "CERTIFIED-FLOOR-EQUIVALENT")

        # cross-arm negative: foreign marker refuses
        (d / "kernel.log").write_text(
            f"{SERVED_MODEL} {BUNDLE_MARKER}\n{ARMED_MARKER}\n"
            "[notes] persistent-namespace armed\n", encoding="utf-8")
        res = score(d)
        check("foreign marker refused", res["verdict"] == "INFRA DEATH")

        # missing armed marker refuses
        (d / "kernel.log").write_text(
            f"{SERVED_MODEL} {BUNDLE_MARKER}\nno marker here\n", encoding="utf-8")
        res = score(d)
        check("unarmed refused", res["verdict"] == "INFRA DEATH"
              and "never armed" in res["reason"])

        # 0-byte log refuses (P1 class)
        (d / "kernel.log").write_text("", encoding="utf-8")
        res = score(d)
        check("empty log refused", res["verdict"] == "INFRA DEATH"
              and "no log" in res["reason"])

    # malformed artifacts fail loudly
    cases = {
        "no_game_runs": {"label": "x"},
        "empty_game_runs": {"label": "x", "game_runs": []},
        "run_missing_lc": {"game_runs": [
            {k: v for k, v in good_run.items() if k != "levels_completed"}]},
        "run_missing_score": {"game_runs": [
            {k: v for k, v in good_run.items() if k != "final_score"}]},
        "scalar_actions": {"game_runs": [
            {**{k: v for k, v in good_run.items() if k != "actions_per_level"},
             "total_actions": 7}]},
        "run_is_string": {"game_runs": ["nope"]},
    }
    for name, payload in cases.items():
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "benchmark.json").write_text(json.dumps(payload), encoding="utf-8")
            try:
                score(d)
                check(f"malformed {name} raises", False)
            except ScoreError:
                check(f"malformed {name} raises", True)
    with tempfile.TemporaryDirectory() as td:
        try:
            score(Path(td))
            check("missing benchmark raises", False)
        except ScoreError:
            check("missing benchmark raises", True)

    print(f"execwm_score selftest: {ok} ok / {fail} fail")
    return 1 if fail else 0


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    if args[0] == "--selftest":
        return _selftest()
    res = score(Path(args[0]))
    print(json.dumps(res, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
