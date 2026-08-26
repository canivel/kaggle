"""Sealed scorer -- P2 reset-anchored episodic retry (p2_reset_retry_prereg_2026-08-22.md).

Certification (exp-34 rule: identity POSITIVE and NEGATIVE):
  REQUIRED   "[p2] reset semantics OK"      the boot check ran on this run
  REQUIRED   "[p2] reset-retry armed"       the patch applied on this run
  REQUIRED   served-model + bundle markers of the certified field-floor vehicle
  FORBIDDEN  foreign arm markers ([notes], [execwm], [cadence], grafts, private)

DELIVERY IS READ FROM THE JOB-DIR REPORTS FIRST (``p2/*.json``), log second.
P1 COMPLETED, was pulled TWICE, and its kernel log was 0 BYTES both times, so its
sealed certification -- defined on log markers -- was UNEVALUABLE. exec-WM survived
that class only because its scorer read report files first. This one does the same.

TWO GATES, AND THEY MEASURE DIFFERENT THINGS:
  D1 DELIVERY  -- >=15/25 games ENTER retry_mode (the trigger actually fires).
                  Pre-measured on retained artifacts at 19/25 for this vehicle:
                  learnings/war_room/p2_trigger_fireability_2026-08-26.md
  D2 USE       -- >=25% of armed turns actually CALL attempt().
                  This is the arm's real risk, not D1. feedback_advertise_where_
                  model_reads.md: a schema-only affordance DELIVERED at 96.3% and
                  got 1.3% USE. Delivery is not use; the scorer refuses to conflate
                  them and reports DELIVERED-NOT-USED as its own verdict.

D1 COUNTS ARM-REACHABILITY, NOT REPORT-PRESENCE. Lesson bought this morning on
exec-WM (learnings/war_room/execwm_seed1_read_2026-08-26.md): that arm's D1 failed
18/25 purely because 7 click-only games were CORRECTLY self-disabled and emitted no
report -- the gate counted a correct refusal as a delivery failure. Here, a game
that produced a report but never armed is ARM-REACHED-AND-SILENT, which is a real
(and interesting) outcome, not a missing measurement; it is counted, not dropped.

Usage:
    python duck_eval/p2/p2_score.py <pull_dir>
    python duck_eval/p2/p2_score.py --selftest
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SERVED_MODEL = "Qwen/Qwen3.8-27B-FP8"
BUNDLE_MARKER = "anim-20260807"
BOOT_MARKER = "[p2] reset semantics OK"
ARMED_MARKER = "[p2] reset-retry armed"
FOREIGN = (
    "PRIVATE-ARM BANNER",
    "TAAF_GRAFTS FEATURES",
    "[goalkeep] armed",
    "[clickmap] armed",
    "[banking] armed",
    "[notes] persistent-namespace armed",
    "[execwm] armed",
    "[cadence] effort pin armed",
    "[cadence] max_output armed",
)
N_GAMES = 25

D1_GAMES_BAR = 15          # games that must enter retry_mode
D2_USE_RATE_BAR = 0.25     # fraction of armed turns that must call attempt()


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
    """The per-game D2 reports the patch flushes to <run_root>/p2/<game>.json."""
    reports = []
    for p in sorted(run_dir.rglob("p2/*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise ScoreError(f"unreadable P2 report {p.name}: {exc}")
        if isinstance(data, dict) and "armed_turns" in data:
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
    if BOOT_MARKER not in log:
        return False, f"required marker {BOOT_MARKER!r} absent - boot check never ran", facts
    if ARMED_MARKER not in log:
        return False, f"required marker {ARMED_MARKER!r} absent - P2 never armed", facts
    for m in FOREIGN:
        if m in log:
            return False, f"foreign arm marker {m!r} present", facts
    facts["boot_check_count"] = log.count(BOOT_MARKER)
    facts["armed_count"] = log.count(ARMED_MARKER)
    return True, "certified", facts


def delivery(run_dir: Path) -> dict:
    """D1 (trigger fired) and D2 (affordance used), report-file first."""
    reports = _read_reports(run_dir)
    out: dict = {"source": "reports" if reports else "none",
                 "games_reported": len(reports)}
    if not reports:
        out.update({
            "games_armed": None, "armed_turns": None, "acting_turns": None,
            "turns_calling_attempt": None, "attempt_calls_armed": None,
            "attempt_calls_unarmed": None, "d2_use_rate": None,
            "max_stuck_run_p50": None,
            "d1_pass": None, "d2_pass": None,
        })
        return out

    games_armed = sum(1 for r in reports if r.get("ever_armed") or (r.get("armed_turns") or 0) > 0)
    armed_turns = sum(int(r.get("armed_turns") or 0) for r in reports)
    acting_turns = sum(int(r.get("acting_turns") or 0) for r in reports)
    calling = sum(int(r.get("turns_calling_attempt_armed") or 0) for r in reports)
    calls_a = sum(int(r.get("attempt_calls_armed") or 0) for r in reports)
    calls_u = sum(int(r.get("attempt_calls_unarmed") or 0) for r in reports)
    runs_sorted = sorted(int(r.get("max_stuck_run") or 0) for r in reports)
    p50 = runs_sorted[len(runs_sorted) // 2] if runs_sorted else None

    out.update({
        "games_armed": games_armed,
        "armed_turns": armed_turns,
        "acting_turns": acting_turns,
        "turns_calling_attempt": calling,
        "attempt_calls_armed": calls_a,
        "attempt_calls_unarmed": calls_u,
        "d2_use_rate": (calling / armed_turns) if armed_turns else None,
        "max_stuck_run_p50": p50,
        "d1_pass": games_armed >= D1_GAMES_BAR,
        "d2_pass": (calling / armed_turns >= D2_USE_RATE_BAR) if armed_turns else False,
    })
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
    if cert_ok:
        if deliv["d1_pass"] is False:
            # The trigger did not fire on enough games. Pre-measured at 19/25 on
            # this vehicle, so this reads as an INSTRUMENT problem, not a result.
            verdict = "CERTIFIED-TRIGGER-UNDERFIRED"
        elif deliv["d2_pass"] is False and deliv["armed_turns"]:
            # The affordance was offered and the model did not take it. This is a
            # REAL and reportable outcome -- and it is NOT a mechanism refutation,
            # because the mechanism never ran. P1 died here.
            verdict = "CERTIFIED-DELIVERED-NOT-USED"

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

    def make(root: Path, *, log: str, reports: list[dict] | None = None,
             n=N_GAMES, lc_each=1, score_each=2.0):
        root.mkdir(parents=True, exist_ok=True)
        (root / "run.log").write_text(log, encoding="utf-8")
        bench = {"game_runs": [
            {"game_id": f"g{i:02d}-x", "levels_completed": lc_each,
             "final_score": score_each, "state": "gave_up",
             "actions_per_level": [10, 10]} for i in range(n)]}
        (root / "benchmark.json").write_text(json.dumps(bench), encoding="utf-8")
        if reports is not None:
            d = root / "p2"
            d.mkdir(exist_ok=True)
            for i, rep in enumerate(reports):
                (d / f"g{i:02d}.json").write_text(json.dumps(rep), encoding="utf-8")

    HEALTHY_LOG = (
        f"serving {SERVED_MODEL}\nmounting {BUNDLE_MARKER} bundle\n"
        f"{BOOT_MARKER}\n{ARMED_MARKER} H=4 K=5 cap=40\n"
    )

    def rep(armed_turns, calling, acting=20, max_run=9, calls=None):
        return {"armed_turns": armed_turns, "acting_turns": acting,
                "turns_calling_attempt_armed": calling,
                "attempt_calls_armed": calls if calls is not None else calling,
                "attempt_calls_unarmed": 0, "ever_armed": armed_turns > 0,
                "max_stuck_run": max_run, "H": 4, "K": 5, "cap": 40,
                "d2_use_rate": (calling / armed_turns) if armed_turns else None}

    with tempfile.TemporaryDirectory() as td:
        T = Path(td)

        # --- POSITIVE CONTROL FIRST (the arm-mismatch lesson, 2026-08-20): a
        # scorer that has never been shown a HEALTHY fixture can return INFRA
        # DEATH on a healthy arm and nobody finds out until the data lands.
        healthy = T / "healthy"
        make(healthy, log=HEALTHY_LOG, reports=[rep(10, 4) for _ in range(19)]
             + [rep(0, 0, max_run=2) for _ in range(6)])
        r = score(healthy)
        check("H1 healthy fixture CERTIFIES", r["verdict"] == "CERTIFIED")
        check("H2 healthy D1 passes (19 armed >= 15)", r["delivery"]["d1_pass"] is True)
        check("H3 healthy D2 passes (0.40 >= 0.25)", r["delivery"]["d2_pass"] is True)
        check("H4 games_armed counted", r["delivery"]["games_armed"] == 19)
        check("H5 use rate exact", abs(r["delivery"]["d2_use_rate"] - 0.4) < 1e-9)
        check("H6 lc_total", r["lc_total"] == 25)
        check("H7 source is reports", r["delivery"]["source"] == "reports")

        # --- CERTIFICATION MUST BE ABLE TO REFUSE ---
        d = T / "nolog"
        make(d, log="   ", reports=[rep(10, 4)])
        check("R1 empty log -> INFRA DEATH", score(d)["verdict"] == "INFRA DEATH")

        d = T / "nomodel"
        make(d, log=f"{BUNDLE_MARKER}\n{BOOT_MARKER}\n{ARMED_MARKER}\n")
        check("R2 wrong served model -> INFRA DEATH", score(d)["verdict"] == "INFRA DEATH")

        d = T / "nobundle"
        make(d, log=f"{SERVED_MODEL}\n{BOOT_MARKER}\n{ARMED_MARKER}\n")
        check("R3 missing bundle marker -> INFRA DEATH", score(d)["verdict"] == "INFRA DEATH")

        d = T / "noboot"
        make(d, log=f"{SERVED_MODEL}\n{BUNDLE_MARKER}\n{ARMED_MARKER}\n")
        check("R4 missing boot check -> INFRA DEATH", score(d)["verdict"] == "INFRA DEATH")

        d = T / "noarm"
        make(d, log=f"{SERVED_MODEL}\n{BUNDLE_MARKER}\n{BOOT_MARKER}\n")
        check("R5 missing armed marker -> INFRA DEATH", score(d)["verdict"] == "INFRA DEATH")

        # --- CROSS-ARM NEGATIVE CONTROLS: every sibling arm must be REFUSED ---
        for i, m in enumerate(FOREIGN):
            d = T / f"foreign{i}"
            make(d, log=HEALTHY_LOG + m + "\n", reports=[rep(10, 4)])
            check(f"X{i} foreign marker {m!r} REFUSED",
                  score(d)["verdict"] == "INFRA DEATH")

        # --- D1 / D2 MUST BE ABLE TO REFUSE INDEPENDENTLY ---
        d = T / "d1fail"
        make(d, log=HEALTHY_LOG, reports=[rep(10, 4) for _ in range(14)]
             + [rep(0, 0, max_run=2) for _ in range(11)])
        r = score(d)
        check("D1a 14 armed < 15 -> TRIGGER-UNDERFIRED",
              r["verdict"] == "CERTIFIED-TRIGGER-UNDERFIRED")
        check("D1b d1_pass False", r["delivery"]["d1_pass"] is False)

        d = T / "d2fail"
        make(d, log=HEALTHY_LOG, reports=[rep(10, 1) for _ in range(19)]
             + [rep(0, 0, max_run=2) for _ in range(6)])
        r = score(d)
        check("D2a use 0.10 < 0.25 -> DELIVERED-NOT-USED",
              r["verdict"] == "CERTIFIED-DELIVERED-NOT-USED")
        check("D2b d1 still passed", r["delivery"]["d1_pass"] is True)
        check("D2c the P1 rate (1.3%) would also fail",
              rep(100, 1)["d2_use_rate"] < D2_USE_RATE_BAR)

        # D1 is checked BEFORE D2: a trigger that underfired makes the use rate
        # meaningless, and reporting "not used" would blame the model for an
        # instrument failure.
        d = T / "bothfail"
        make(d, log=HEALTHY_LOG, reports=[rep(10, 0) for _ in range(5)]
             + [rep(0, 0, max_run=2) for _ in range(20)])
        check("D3 D1 failure outranks D2 failure",
              score(d)["verdict"] == "CERTIFIED-TRIGGER-UNDERFIRED")

        # --- MISSING REPORTS: unevaluable, and it must SAY so, not score 0 ---
        d = T / "noreports"
        make(d, log=HEALTHY_LOG)
        r = score(d)
        check("N1 no reports -> delivery unevaluable, not zero",
              r["delivery"]["d1_pass"] is None and r["delivery"]["d2_use_rate"] is None)
        check("N2 ... and it does NOT claim DELIVERED-NOT-USED",
              r["verdict"] == "CERTIFIED")
        check("N3 source says none", r["delivery"]["source"] == "none")

        # --- MALFORMED ARTIFACTS FAIL LOUDLY ---
        d = T / "badbench"
        d.mkdir(parents=True, exist_ok=True)
        (d / "run.log").write_text(HEALTHY_LOG, encoding="utf-8")
        (d / "benchmark.json").write_text(json.dumps({"game_runs": []}), encoding="utf-8")
        try:
            score(d)
            check("M1 empty game_runs raises", False)
        except ScoreError:
            check("M1 empty game_runs raises", True)

        d = T / "badfields"
        d.mkdir(parents=True, exist_ok=True)
        (d / "run.log").write_text(HEALTHY_LOG, encoding="utf-8")
        (d / "benchmark.json").write_text(
            json.dumps({"game_runs": [{"levels_completed": 1}]}), encoding="utf-8")
        try:
            score(d)
            check("M2 missing final_score raises", False)
        except ScoreError:
            check("M2 missing final_score raises", True)

        # --- REAL FOREIGN ARTIFACTS ON DISK MUST ALL BE REFUSED ---
        repo = Path(__file__).resolve().parents[2]
        pulls = repo / "runs" / "kernel_pulls"
        refused = tried = 0
        for name in ("q38_field_v1", "execwm_v1", "p1_notes_v1", "budget_t3_v1",
                     "private_base_v1", "q38graft_v1"):
            p = pulls / name
            if not p.is_dir():
                continue
            tried += 1
            try:
                v = score(p)["verdict"]
            except ScoreError:
                v = "INFRA DEATH"
            if v == "INFRA DEATH":
                refused += 1
            else:
                print(f"SELFTEST FAIL: real foreign artifact {name} was ACCEPTED as {v}")
        check(f"A1 all {tried} real foreign artifacts REFUSED", tried > 0 and refused == tried)

    print(f"p2_score selftest: {ok} ok / {fail} fail")
    return 1 if fail else 0


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    if args[0] == "--selftest":
        return _selftest()
    print(json.dumps(score(Path(args[0])), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
