"""SEALED verdict scorer for the TOKEN-COST arm (reasoning_effort=low). Written 2026-08-16,
BEFORE the arm was built and long before it runs.

Constants derive from prereg `learnings/war_room/q38_engine_swap_prereg_2026-08-15.md` sections
15-17. Nothing here may be edited after data lands; that is the whole point of the seal.

TWO PRIMARIES, reported jointly, NEITHER overriding the other:

  PRIMARY-A  levels vs B1 = duck-harness-kaggle m=3 (Qwen3.6): 18/19/21 -> 0.773333 lc/game
             Same K3'' machinery and the SAME thresholds as the engine arm, deliberately, so
             the two arms are directly comparable.
                HARM   <= -0.286320   (<= 12 levels)
                NO-LIFT<= +0.250000   (<= 25 levels)
                LIFT   >= +0.500000   (>= 32 levels)

  PRIMARY-B  actions vs B2 = the engine arm (effort=medium, n=1): 2,857 actions, 21 levels.
             The mechanism claim. n=1 vs n=1 -> NO ERROR MODEL. This is a pre-registered
             DECISION RULE, not a significance test, and it says so in its own output.
                ACTION-RECOVERY  >= 3,665   (>= 50% of the 1,617-action gap back to B1's 4,474)
                NO-RECOVERY      <= 3,100   (< 15% of the gap)
                PARTIAL          between

The interesting quadrant is NAMED IN ADVANCE: ACTION-RECOVERY + NO-LIFT means the knob bought
actions but the shorter thinking gave back the per-action quality (the engine arm measured
1.70x levels/action at 2.10x tokens/action). That is a real finding and MUST NOT be reported
as a failure.

Third state INFRA DEATH (not decisive) carried forward unchanged.
Score-based reading is NON-INFERENTIAL by prereg section 15.3 and carries no verdict.

    python duck_eval/q38/q38low_score.py <pulled_kernel_dir>
    python duck_eval/q38/q38low_score.py --selftest
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# ---- SEALED CONSTANTS -----------------------------------------------------
N_GAMES = 25
# B1 - the Qwen3.6 baseline family
B1_RUNS = {"gate_eval_v1": 18, "gate_eval_v2": 19, "duckgate_v1post": 21}
B1_LC_PER_GAME = 58 / (3 * 25)      # 0.773333
B1_ACTIONS = 4474
B1_TOKENS_PER_ACTION = 370
# B2 - the engine arm (effort=medium), n=1
B2_LEVELS = 21
B2_ACTIONS = 2857
B2_TOKENS_PER_ACTION = 776
B2_LEVELS_PER_ACTION = 21 / 2857
# PRIMARY-A lines (identical to the engine arm's, on purpose)
SIGMA = 0.141740
SE_DELTA = 0.163667
HARM_LINE = -0.286320
NOLIFT_LINE = 0.250000
LIFT_LINE = 0.500000
# PRIMARY-B lines
ACTION_GAP = B1_ACTIONS - B2_ACTIONS                 # 1617
ACTION_RECOVERY = B2_ACTIONS + int(0.50 * ACTION_GAP)   # 3665
ACTION_NORECOVERY = B2_ACTIONS + int(0.15 * ACTION_GAP)  # 3099 -> stated as 3100
ACTION_NORECOVERY = 3100
WINDOW_S = 7920.0
WINDOW_DRIFT_TOL = 0.05
FT09 = "ft09-0d8bbf25"
FT09_B1_MEAN_LC = 2.3333
INFRA_SIGNATURES = (
    "Q38-EVAL FATAL", "No supported CUDA architectures", "libcudart",
    "CUDA out of memory", "Timed out waiting for vLLM server",
    "Missing attached dataset path",
)
# ---------------------------------------------------------------------------


def _log_text(run_dir: Path) -> str:
    parts = []
    for pat in ("*.log", "*.txt"):
        for path in run_dir.glob(pat):
            try:
                parts.append(path.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                pass
    return "\n".join(parts)


def score(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    log = _log_text(run_dir)
    out: dict = {"run_dir": str(run_dir)}
    path = run_dir / "benchmark.json"
    signature = next((s for s in INFRA_SIGNATURES if s in log), None)

    if not path.is_file():
        out.update(verdict="INFRA DEATH (not decisive)", decisive=False,
                   reason=f"no benchmark.json; infra signature {signature!r}" if signature
                   else "no benchmark.json and no recognised infra signature")
        return out
    bench = json.loads(path.read_text(encoding="utf-8"))
    runs = bench.get("game_runs") or []
    out["label"] = bench.get("label")
    out["n_games"] = len(runs)
    if len(runs) != N_GAMES:
        out.update(verdict="INFRA DEATH (not decisive)", decisive=False,
                   reason=f"{len(runs)} games, expected {N_GAMES}")
        return out

    def acts(r):
        return sum(r.get("actions_per_level") or [])

    lc = sum(int(r.get("levels_completed") or 0) for r in runs)
    actions = sum(acts(r) for r in runs)
    scores = [float(r.get("final_score") or 0.0) for r in runs]
    tokens = 0
    for r in runs:
        note = r.get("solver_note") or ""
        if "tokens=" in note:
            try:
                tokens += int(note.split("tokens=")[-1])
            except ValueError:
                pass
    drifted = [r for r in runs
               if abs(float(r.get("final_wallclock_seconds") or 0.0) - WINDOW_S)
               > WINDOW_DRIFT_TOL * WINDOW_S]
    zero_action = sorted(r["game_id"] for r in runs if acts(r) == 0)
    ft09_actions = next((acts(r) for r in runs if r.get("game_id") == FT09), None)

    out.update(levels=lc, lc_per_game=lc / N_GAMES, actions=actions,
               actions_per_window=actions / N_GAMES, tokens=tokens,
               tokens_per_action=(tokens / actions if actions else None),
               mean_score=sum(scores) / N_GAMES,
               levels_per_action=(lc / actions if actions else None),
               zero_action_games=zero_action, ft09_actions=ft09_actions,
               windows_drifted=len(drifted))

    if len(drifted) > 2:
        out.update(verdict="INFRA DEATH (not decisive)", decisive=False,
                   reason=f"{len(drifted)} windows drifted >5%; comparison VOID")
        return out
    if log and "Q38-EVAL BOOT-ASSERTS PASSED" not in log:
        out.update(verdict="INFRA DEATH (not decisive)", decisive=False,
                   reason="boot asserts never reported PASSED; the served engine and the "
                          "effort pin are UNCONFIRMED")
        return out

    delta = lc / N_GAMES - B1_LC_PER_GAME
    out["mean_delta_lc_vs_B1"] = delta
    out["z_vs_B1"] = delta / SE_DELTA
    if delta <= HARM_LINE:
        a = "HARM"
    elif delta <= NOLIFT_LINE:
        a = "NO-LIFT"
    elif delta >= LIFT_LINE:
        a = "LIFT"
    else:
        a = "INDETERMINATE"
    out["primary_A"] = a

    if actions >= ACTION_RECOVERY:
        b = "ACTION-RECOVERY"
    elif actions <= ACTION_NORECOVERY:
        b = "NO-RECOVERY"
    else:
        b = "PARTIAL"
    out["primary_B"] = b

    out["verdict"] = f"A={a} / B={b}"
    out["decisive"] = a in ("HARM", "NO-LIFT", "LIFT")
    if a == "LIFT" and b == "ACTION-RECOVERY":
        out["reason"] = ("the knob recovered actions AND lifted levels - the mechanism story "
                         "holds end to end")
    elif b == "ACTION-RECOVERY" and a == "NO-LIFT":
        out["reason"] = ("PRE-NAMED QUADRANT: the knob bought actions but shorter thinking gave "
                         "back the per-action quality. A REAL FINDING, NOT A FAILURE - it says "
                         "the 1.70x levels/action was PAID FOR by the 2.10x tokens/action and "
                         "the trade is ~neutral in both directions")
    elif b == "NO-RECOVERY":
        out["reason"] = ("the effort knob did not move the action budget; the token cost is not "
                         "driven by the injected instruction and the next lever is "
                         "LOCAL_ANALYZER_YIELD_SECONDS, not the prompt")
    else:
        out["reason"] = "partial action recovery; read A and B together"
    return out


def render(r: dict) -> str:
    L = ["=" * 78,
         f"Q38 TOKEN-COST ARM (effort=low) - SEALED VERDICT: {r['verdict']}",
         f"decisive (PRIMARY-A): {r.get('decisive')}",
         f"reason: {r.get('reason')}",
         "=" * 78]
    if "levels" not in r:
        return "\n".join(L)
    L += [
        "PRIMARY-A  levels vs B1 (Qwen3.6 m=3, 18/19/21 = 0.7733/game)",
        f"    levels {r['levels']} ({r['lc_per_game']:.4f}/game)   mean dlc "
        f"{r['mean_delta_lc_vs_B1']:+.4f}   z {r['z_vs_B1']:+.2f}",
        f"    lines: HARM <= {HARM_LINE:+.4f} | NO-LIFT <= {NOLIFT_LINE:+.4f} | "
        f"LIFT >= {LIFT_LINE:+.4f}          -> {r['primary_A']}",
        "PRIMARY-B  actions vs B2 (engine arm, effort=medium, n=1: 2857 actions, 21 levels)",
        f"    actions {r['actions']} ({r['actions_per_window']:.1f}/window)   "
        f"B2 {B2_ACTIONS}   B1 {B1_ACTIONS}",
        f"    lines: NO-RECOVERY <= {ACTION_NORECOVERY} | ACTION-RECOVERY >= "
        f"{ACTION_RECOVERY}                 -> {r['primary_B']}",
        "    NOTE: n=1 vs n=1. This is a pre-registered DECISION RULE, not a significance test.",
        "--- mechanism, descriptive ---",
        f"    tokens/action  {r['tokens_per_action']:.0f}  (B2 {B2_TOKENS_PER_ACTION}, "
        f"B1 {B1_TOKENS_PER_ACTION})" if r.get("tokens_per_action") else "    tokens/action  n/a",
        f"    levels/action  {r['levels_per_action']:.5f}  (B2 {B2_LEVELS_PER_ACTION:.5f})"
        if r.get("levels_per_action") else "    levels/action  n/a",
        f"    ft09 actions   {r['ft09_actions']}  (B2: 0 -> ANY non-zero is direct mechanism "
        f"evidence for the yield-budget explanation; worth ~{FT09_B1_MEAN_LC} levels)",
        f"    zero-action games {len(r['zero_action_games'])}: {r['zero_action_games']}",
        f"    windows drifted {r['windows_drifted']}/25",
        "--- NON-INFERENTIAL (prereg 15.3) ---",
        f"    mean_score {r['mean_score']:.3f}  (B1 1.427/1.939/3.420 sd 1.033 n=3; B2 2.795) "
        f"- ~60% power, NO VERDICT",
        "    tokens/s: use the Q38-EVAL DECODE lines, NOT summary.txt's job-wallclock figure",
    ]
    return "\n".join(L)


def _selftest() -> int:
    import tempfile
    p = f = 0

    def expect(name, got, want):
        nonlocal p, f
        if got == want:
            p += 1; print(f"  ok   {name}")
        else:
            f += 1; print(f"  FAIL {name}: got {got!r} want {want!r}")

    def make(levels, actions, *, n=N_GAMES, ft09_act=None, log="Q38-EVAL BOOT-ASSERTS PASSED",
             bench=True, drift=0):
        root = Path(tempfile.mkdtemp())
        if log is not None:
            (root / "r.log").write_text(log, encoding="utf-8")
        if bench:
            lv = [0] * n; ac = [0] * n
            for i in range(levels):
                lv[i % n] += 1
            base, rem = divmod(actions, n)
            for i in range(n):
                ac[i] = base + (1 if i < rem else 0)
            runs = []
            for i in range(n):
                gid = FT09 if i == 0 else f"g{i:02d}"
                a = ft09_act if (gid == FT09 and ft09_act is not None) else ac[i]
                runs.append({"game_id": gid, "levels_completed": lv[i],
                             "final_score": float(lv[i]), "actions_per_level": [a],
                             "solver_note": f"tokens={a * 700}",
                             "final_wallclock_seconds": WINDOW_S * (0.5 if i < drift else 1.0)})
            (root / "benchmark.json").write_text(
                json.dumps({"label": "duck-harness-kaggle", "game_runs": runs}), encoding="utf-8")
        return root

    print("== PRIMARY-A boundaries (identical to the engine arm) ==")
    expect("32 levels -> LIFT", score(make(32, 4000))["primary_A"], "LIFT")
    expect("31 levels -> INDETERMINATE", score(make(31, 4000))["primary_A"], "INDETERMINATE")
    expect("25 levels -> NO-LIFT", score(make(25, 4000))["primary_A"], "NO-LIFT")
    expect("12 levels -> HARM", score(make(12, 4000))["primary_A"], "HARM")
    expect("21 levels (= the engine arm) -> NO-LIFT", score(make(21, 4000))["primary_A"],
           "NO-LIFT")

    print("== PRIMARY-B boundaries ==")
    expect("3665 actions -> ACTION-RECOVERY", score(make(21, 3665))["primary_B"],
           "ACTION-RECOVERY")
    expect("3664 actions -> PARTIAL", score(make(21, 3664))["primary_B"], "PARTIAL")
    expect("3100 actions -> NO-RECOVERY", score(make(21, 3100))["primary_B"], "NO-RECOVERY")
    expect("2857 actions (= the engine arm) -> NO-RECOVERY",
           score(make(21, 2857))["primary_B"], "NO-RECOVERY")
    expect("4474 actions (= B1) -> ACTION-RECOVERY", score(make(21, 4474))["primary_B"],
           "ACTION-RECOVERY")

    print("== the pre-named quadrant is recognised, not treated as failure ==")
    r = score(make(21, 4000))
    expect("A=NO-LIFT / B=ACTION-RECOVERY", r["verdict"], "A=NO-LIFT / B=ACTION-RECOVERY")
    expect("and it is labelled a real finding",
           "NOT A FAILURE" in r["reason"], True)

    print("== third state ==")
    expect("no benchmark.json -> INFRA DEATH",
           score(make(21, 4000, bench=False))["verdict"], "INFRA DEATH (not decisive)")
    expect("boot asserts absent -> INFRA DEATH",
           score(make(21, 4000, log="other"))["verdict"], "INFRA DEATH (not decisive)")
    expect("3 drifted windows -> INFRA DEATH",
           score(make(21, 4000, drift=3))["verdict"], "INFRA DEATH (not decisive)")

    print("== ft09 mechanism read ==")
    expect("ft09 zero actions is surfaced", score(make(21, 4000, ft09_act=0))["ft09_actions"], 0)
    expect("ft09 zero-action listed", FT09 in score(make(21, 4000, ft09_act=0))["zero_action_games"],
           True)

    print("== sealed constants ==")
    for n, got, want in (("B1_LC_PER_GAME", round(B1_LC_PER_GAME, 6), 0.773333),
                         ("ACTION_RECOVERY", ACTION_RECOVERY, 3665),
                         ("ACTION_NORECOVERY", ACTION_NORECOVERY, 3100),
                         ("B2_ACTIONS", B2_ACTIONS, 2857),
                         ("HARM_LINE", HARM_LINE, -0.286320),
                         ("LIFT_LINE", LIFT_LINE, 0.5)):
        expect(f"{n} sealed", str(got), str(want))

    print(f"\nQ38-LOW SCORER SELFTEST: {p} passed, {f} failed")
    return 1 if f else 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        sys.exit(_selftest())
    if len(sys.argv) != 2:
        print(__doc__); sys.exit(2)
    print(render(score(Path(sys.argv[1]))))
