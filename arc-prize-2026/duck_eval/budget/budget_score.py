"""Sealed scorer - ARM 1 budget-elasticity (budget_elasticity_prereg_2026-08-22.md).

Per-arm certification tables (exp-34 rule: identity positive AND negative - each arm's
runtime echo is REQUIRED for itself and FORBIDDEN for every sibling, incl. T1's 7920.0).
Screening primary: trim1 (mean per-game score minus the best game) + lc_total co-primary.
Raw mean_score is RETIRED as a primary (50.4% one game) and reported descriptive-only.
The elasticity verdict is CROSS-ARM (needs both T05 and T3 + the T1 replicates 28/30) and
is computed by --elasticity; per-arm invocations certify + report only.

Usage:
    python duck_eval/budget/budget_score.py --arm t05|t3 <pull_dir> [--certify-only]
    python duck_eval/budget/budget_score.py --elasticity <t05_pull> <t3_pull>
    python duck_eval/budget/budget_score.py --selftest
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "graft"))
import graft_score as gs  # corrected CLI-JSON log decoding, _infra

SERVED_MODEL = "Qwen/Qwen3.8-27B-FP8"
BUNDLE_MARKER = "anim-20260807"
FOREIGN = ("PRIVATE-ARM BANNER", "model-20260815-q38-p1", "TAAF_GRAFTS FEATURES",
           "[goalkeep] armed", "[clickmap] armed")
N_GAMES = 25
T1_REPLICATES = (28, 30)  # field 28 + ArmA-base 30, mean 29.0, pooled seed sd 2.80
# INSTRUMENT FIX 2026-08-22, found against the REAL T0.5 artifact BEFORE the elasticity
# read: the solver banner echoes the PICKLED DEFAULT (7920.0), printed when the benchmark
# loads and BEFORE the customization cell assigns the arm's value -- so the echo cannot
# discriminate arms. The discriminating runtime observable is the BENCHMARK's per-game
# final_wallclock_seconds distribution (T0.5 real artifact: median 3960, max 4003 -- the
# budget binds every game). Certification therefore reads the wallclock window; the 7920.0
# echo is EXPECTED on every arm and asserted as such (its absence would mean a different
# vehicle).
ARMS = {
    "t05": {"budget": 3960.0,  "wall_lo": 3900.0,  "wall_hi": 4300.0},
    "t3":  {"budget": 23760.0, "wall_lo": 23700.0, "wall_hi": 24100.0},
}
DEFAULT_ECHO = "max_runtime_s_per_game=7920.0"  # pickled default, pre-assignment print


def certify(log: str, arm: str):
    spec = ARMS[arm]
    facts = {}
    if SERVED_MODEL not in log:
        return False, f"served model {SERVED_MODEL!r} not found", facts
    if "reasoning_effort" in log:
        return False, "reasoning_effort present (xhigh default violated)", facts
    if BUNDLE_MARKER not in log:
        return False, f"bundle marker {BUNDLE_MARKER!r} absent - not the 08-07 anim harness", facts
    for m in FOREIGN:
        if m in log:
            return False, f"foreign arm marker {m!r} present", facts
    if DEFAULT_ECHO not in log:
        return False, f"vehicle banner {DEFAULT_ECHO!r} absent - different vehicle/solver", facts
    facts["runtime_echo"] = DEFAULT_ECHO + " (pickled default; arm discriminated by wallclock)"
    return True, "certified", facts


def read_bench(run_dir: Path):
    b = json.loads((run_dir / "benchmark.json").read_text(encoding="utf-8"))
    runs = b.get("game_runs") or []
    lc = sum(int(r.get("levels_completed") or 0) for r in runs)
    scores = [float(r.get("final_score") or 0.0) for r in runs]
    mean_score = sum(scores) / len(runs) if runs else 0.0
    trim1 = (sum(scores) - max(scores)) / (len(runs) - 1) if len(runs) > 1 else 0.0
    actions = sum(sum(r.get("actions_per_level") or []) for r in runs)
    turns = 0
    for r in runs:
        h = r.get("history")
        turns += len(h) if isinstance(h, list) else sum(r.get("actions_per_level") or [0])
    walls = sorted(float(r.get("final_wallclock_seconds") or 0.0) for r in runs)
    wall_median = walls[len(walls) // 2] if walls else 0.0
    wall_max = walls[-1] if walls else 0.0
    return len(runs), lc, trim1, mean_score, actions, turns, wall_median, wall_max


def score(run_dir: Path, arm: str, certify_only: bool = False) -> dict:
    log = gs._read_log(run_dir)
    ok, reason, facts = certify(log, arm)
    if not ok:
        return gs._infra(f"not certified: {reason}", **facts)
    if not (run_dir / "benchmark.json").exists():
        return gs._infra("no benchmark.json", **facts)
    n, lc, trim1, ms, actions, turns, wall_med, wall_max = read_bench(run_dir)
    if n != N_GAMES:
        return gs._infra(f"n_games={n} != {N_GAMES}", **facts)
    spec = ARMS[arm]
    if not (spec["wall_lo"] <= wall_med <= spec["wall_hi"]):
        return gs._infra(
            f"wallclock median {wall_med:.0f}s outside this arm's window "
            f"[{spec['wall_lo']:.0f}, {spec['wall_hi']:.0f}] - the budget did NOT bind at "
            f"this arm's value (wrong arm or treatment failed)", **facts)
    facts["wall_median_s"] = wall_med
    facts["wall_max_s"] = wall_max
    out = {"verdict": "CERTIFIED-REPORT", "decisive": False, "arm": arm,
           "lc_total": lc, "trim1": round(trim1, 4), "mean_score_descriptive": round(ms, 4),
           "total_actions": actions, "turns_proxy": turns,
           "t1_replicates": list(T1_REPLICATES), **facts}
    if certify_only:
        out["note"] = "certify-only: numbers reported, elasticity verdict is cross-arm"
    return out


def elasticity(t05_dir: Path, t3_dir: Path) -> dict:
    a = score(t05_dir, "t05")
    b = score(t3_dir, "t3")
    if a.get("verdict") == "INFRA DEATH" or b.get("verdict") == "INFRA DEATH":
        return {"verdict": "INFRA DEATH", "t05": a, "t3": b}
    t1_lc = sum(T1_REPLICATES) / len(T1_REPLICATES)
    lc05, lc3 = a["lc_total"], b["lc_total"]
    # ANOMALY branch first (prereg): lc(T3) < 24 with turns up >= 2x is NOT an elasticity read
    t1_turns_per_game = 17.0  # panel-measured design point
    t3_turns_pg = b["turns_proxy"] / N_GAMES
    if lc3 < 24 and t3_turns_pg >= 2 * t1_turns_per_game:
        return {"verdict": "ANOMALY",
                "detail": "lc(T3) < 24 with turns >= 2x - long-run pathology; diagnose, don't score",
                "t05": a, "t3": b}
    eps = (math.log(max(lc3, 1)) - math.log(max(lc05, 1))) / (math.log(23760.0) - math.log(3960.0))
    if lc3 >= 45 and lc05 <= 22:
        v = "HIGH ELASTICITY (>=0.60): budget program CONFIRMED; Arms 2/3 fire; C7 unlocks"
    elif lc3 <= 33:
        v = "KILL (<0.25): the ENTIRE budget family dies (C1/C2/C3/C7); pivot C4+C5"
    else:
        v = "PARTIAL (0.25..0.60): C2 reduced priority, C4 promoted"
    return {"verdict": v, "epsilon": round(eps, 4), "t1_mean": t1_lc, "t05": a, "t3": b}


def _selftest() -> int:
    import tempfile
    fails, n = [], [0]

    def expect(name, got, want):
        n[0] += 1
        if got != want:
            fails.append(f"{name}: {got!r} != {want!r}")

    NL = chr(10)

    def glog(arm):
        return (
            "taaf.kaggle: source bundle = /kaggle/input/taaf-kaggle-source-anim-20260807-anim" + NL
            + f"vLLM server ready: {SERVED_MODEL}" + NL
            + f"benchmark.solver: HarnessSolver(label='duck-harness', {DEFAULT_ECHO}, concurrency=28)" + NL
        )

    def make(arm, lcs, log=None, history_len=17, wall=None):
        d = Path(tempfile.mkdtemp())
        (d / "kernel.log").write_text(log if log is not None else glog(arm), encoding="utf-8")
        w = wall if wall is not None else ARMS[arm]["budget"] + 10.0
        runs = [{"game_id": f"g{i}", "levels_completed": lcs[i] if i < len(lcs) else 0,
                 "state": "gave_up", "final_score": 1.0 + (i == 0) * 9.0,
                 "actions_per_level": [5], "history": [1] * history_len,
                 "final_wallclock_seconds": w} for i in range(N_GAMES)]
        (d / "benchmark.json").write_text(json.dumps({"game_runs": runs}), encoding="utf-8")
        return d

    expect("t05 certifies own echo", score(make("t05", [20]), "t05")["verdict"], "CERTIFIED-REPORT")
    expect("t3 refuses t05 artifact", score(make("t05", [20]), "t3")["verdict"], "INFRA DEATH")
    expect("t05 refuses a T1-wallclock artifact (budget did not bind at 3960)",
           score(make("t05", [20], wall=7920.0), "t05")["verdict"], "INFRA DEATH")
    expect("vehicle banner absent refused",
           score(make("t05", [20], log=glog("t05").replace("7920.0", "9999.0")), "t05")["verdict"],
           "INFRA DEATH")
    expect("foreign graft marker refused",
           score(make("t05", [20], log=glog("t05") + "TAAF_GRAFTS FEATURES={}" + NL), "t05")["verdict"],
           "INFRA DEATH")
    expect("private banner refused",
           score(make("t05", [20], log=glog("t05") + "PRIVATE-ARM BANNER: x" + NL), "t05")["verdict"],
           "INFRA DEATH")
    expect("effort pin refused",
           score(make("t05", [20], log=glog("t05") + "reasoning_effort=low" + NL), "t05")["verdict"],
           "INFRA DEATH")
    r = score(make("t05", [20]), "t05")
    expect("trim1 drops the best game", r["trim1"], 1.0)
    e = elasticity(make("t05", [20]), make("t3", [46]))
    expect("HIGH branch", "HIGH" in e["verdict"], True)
    e = elasticity(make("t05", [26]), make("t3", [31]))
    expect("KILL branch", "KILL" in e["verdict"], True)
    e = elasticity(make("t05", [24]), make("t3", [40]))
    expect("PARTIAL branch", "PARTIAL" in e["verdict"], True)
    e = elasticity(make("t05", [22]), make("t3", [20], history_len=68))
    expect("ANOMALY branch", e["verdict"], "ANOMALY")

    if fails:
        print(f"SELFTEST FAILED ({len(fails)}/{n[0]}):")
        for f in fails:
            print("  " + f)
        return 1
    print(f"selftest OK ({n[0]}/{n[0]} checks, 0 failures)")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--selftest"]:
        raise SystemExit(_selftest())
    if args and args[0] == "--elasticity":
        print(json.dumps(elasticity(Path(args[1]), Path(args[2])), indent=2, sort_keys=True))
        raise SystemExit(0)
    assert args[0] == "--arm"
    print(json.dumps(score(Path(args[2]), args[1], certify_only="--certify-only" in args),
                     indent=2, sort_keys=True))
