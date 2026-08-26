"""Self-test for ``animation_score.py`` -- CPU only, no GPU, no LLM, no network,
no Kaggle calls, $0.

The real pull does not exist yet (the kernel is still running), so this builds
SYNTHETIC pull directories under ``duck_eval/warpack/_test_fixtures/animation/``
that reproduce the EMITTER's real output format, derived line-for-line from
``duck_eval/warpack/_kaggle_dataset/animation_patch.py``:

  * ``_emit_event``   -> the greppable ``ANIMATION v=1 kind=... game=... action=...``
                         stdout line (bbox and MOUSE(...) action displays both
                         contain spaces -- the fixture keeps them, so the parser
                         is tested against the awkward case).
  * ``canary_report`` -> the single ``ANIMATION CANARY v=1 version=v1 ...`` line,
                         with ``token_fraction=`` EMPTY, because the builder's
                         cell-14 hook calls ``canary_report()`` with no
                         ``total_tokens`` argument.
  * ``apply``         -> the ``animation v1: ACTIVE (4 seams patched) ...`` banner,
                         and ``animation: PATCH FAILED`` from the graft's except.

Logs are written in the real Kaggle shape: a JSON array of
``{"stream_name","time","data"}`` records, so the log loader is exercised on the
same format the pull will hand it.

Cases:
  a healthy        -> CANARIES CLEAR
  b patch-failed   -> VOID (never FAIL)
  c zero-invisible -> VOID + audit-method-under-review
  d token-breach   -> KILL
  e errors-nonzero -> KILL
  f real vanilla runs under runs/kernel_pulls/ (no animation at all) -> must
    degrade gracefully to VOID without crashing; one of them (a22_v2_1) is also
    a P1-illegal run and must be flagged as such.

Run:  uv run python duck_eval/warpack/animation_score_selftest.py
"""
from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIX = HERE / "_test_fixtures" / "animation"

sys.path.insert(0, str(HERE))
import animation_score as A  # noqa: E402

PASS = 0
FAIL = 0

# The real 25-game instance ids (from runs/kernel_pulls/w0_eval_s1/benchmark.json).
GAME_IDS = [
    "sk48-d8078629", "tn36-ef4dde99", "m0r0-492f87ba", "bp35-0a0ad940",
    "cn04-2fe56bfb", "dc22-fdcac232", "tu93-0768757b", "lp85-305b61c3",
    "ka59-38d34dbb", "wa30-ee6fef47", "vc33-5430563c", "lf52-271a04aa",
    "r11l-495a7899", "sc25-635fd71a", "sp80-589a99af", "ar25-0c556536",
    "sb26-7fbdac44", "cd82-fb555c5d", "re86-8af5384d", "s5i5-18d95033",
    "ls20-9607627b", "ft09-0d8bbf25", "su15-1944f8ab", "tr87-cd924810",
    "g50t-5849a774",
]
# audit type per game (runs/animation/frame_audit.json)
TYPE1 = ("ft09", "cd82", "sc25", "ls20")
TYPE2 = ("bp35", "g50t", "ka59", "lf52", "lp85", "r11l", "sb26", "sk48",
         "sp80", "su15", "tn36", "tu93", "vc33")


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}" + (f"  [{detail}]" if detail else ""))


# --------------------------------------------------------------------------- #
# fixture builders -- formats derived from animation_patch.py
# --------------------------------------------------------------------------- #
def event_line(game: str, kind: str, action: str, frames: int, unique: int,
               board_unchanged: int, cells: int, bbox: str,
               run_actions: int, run_multi: int, run_invisible: int) -> str:
    """Byte-shape of animation_patch._emit_event's print()."""
    return (f"ANIMATION v=1 kind={kind} game={game} "
            f"action={action} frames={frames} "
            f"unique={unique} "
            f"board_unchanged={board_unchanged} "
            f"transient_cells={cells} "
            f"bbox={bbox} "
            f"run_actions={run_actions} run_multi={run_multi} "
            f"run_invisible={run_invisible}\n")


def canary_line(actions: int, multi: int, invisible: int, summaries: int,
                errors: int, games_with_events: int, games_with_invisible: int,
                audit_engaged: list[str], tokens_est: int) -> str:
    """Byte-shape of animation_patch.canary_report()'s print(), as the builder
    calls it: canary_report() with NO total_tokens -> token_fraction= is EMPTY."""
    return (f"ANIMATION CANARY v=1 version=v1 "
            f"actions={actions} multi={multi} "
            f"invisible={invisible} "
            f"summaries={summaries} errors={errors} "
            f"games_with_events={games_with_events} "
            f"games_with_invisible={games_with_invisible} "
            f"audit_type1_engaged={','.join(audit_engaged) or 'NONE'} "
            f"tokens_est={tokens_est} "
            f"token_fraction=\n")


BANNER = ("animation v1: ACTIVE (4 seams patched)  -  per-action intermediate-frame "
          "summary from GameState.raw.frame (taaf/game.py:170 discards all but "
          "frame[-1]; zero prior consumers). Fixed scalar schema, NO raw frames, "
          "~45 tok, emitted only on animated actions. only_invisible=OFF (default); "
          "outcome_text=ON (default); NO no-op guard (prereg sec2.2: separately "
          "gated, downstream); zero LLM calls, no locks, game-agnostic\n")
SEED_BANNER = ("animation-eval: SEED=1 animation-awareness ON, NO "
               "warpack/ledger-graft/sentinel/compaction (pairs with the "
               "duck-harness-kaggle-continuation-v1 family); ANIMATION_AWARE=1; "
               "NO no-op guard\n")
GRAFT_BANNER = ("animation v1: graft applied from /kaggle/input/arc-war-kit "
                "(applied=True); NO warpack/ledger-graft/sentinel/compaction/noop-guard\n")
CONT_BANNERS = [
    "continuation v1: game-over-continuation ACTIVE (2 modules patched)\n",
    "continuation v1: (f) game-over-continuation graft applied from "
    "/kaggle/input/datasets/canivel/arc-war-kit (applied=True); NO warpack/ledger\n",
]


def write_log(path: Path, lines: list[str]) -> None:
    """Kaggle build-log shape: JSON array of stream records."""
    recs = [{"stream_name": "stdout", "time": 5.0 + i * 0.01, "data": s}
            for i, s in enumerate(lines)]
    path.write_text(json.dumps(recs, indent=0).replace("},", "},\n"), encoding="utf-8")


def build_pull(name: str, *, patch_failed: bool = False, invisible: bool = True,
               token_breach: bool = False, errors: int = 0) -> Path:
    """One synthetic pull dir: benchmark.json + run log + viewer events."""
    rng = random.Random(f"animation_selftest:{name}")
    out = FIX / name
    if out.exists():
        shutil.rmtree(out)
    (out / "artifacts").mkdir(parents=True)

    # ---- benchmark.json (arm label is stamped by animation_patch.apply) ----
    label = ("duck-harness-kaggle-continuation-v1" if patch_failed
             else "duck-harness-kaggle-continuation-v1-animation-v1")
    game_runs, per_game_actions = [], {}
    for gid in GAME_IDS:
        n_act = rng.randrange(40, 480)
        per_game_actions[gid] = n_act
        tokens = int(n_act * rng.uniform(250, 400))
        hist = [{"action": {"id": "ACTION1", "data": {}},
                 "generated_tokens": tokens // n_act,
                 "uncached_input_tokens": 0,
                 "wallclock_seconds": 7900.0 * (i + 1) / n_act}
                for i in range(n_act)]
        game_runs.append(dict(
            game_id=gid, number_of_levels=8, state="gave_up", history=hist,
            levels_completed=rng.choice([0, 0, 0, 1, 1, 2]),
            solver_note=f"tokens={tokens}", final_wallclock_seconds=7925.3,
            final_generated_tokens=0, final_uncached_input_tokens=0))
    (out / "benchmark.json").write_text(
        json.dumps(dict(label=label, n_passes=1, solver_label="duck-harness",
                        game_runs=game_runs)), encoding="utf-8")
    total_tokens = sum(int(r["solver_note"].split("=")[1]) for r in game_runs)

    # ---- viewer events (vanilla harness artifact; M3 reads these) ----------
    for gid in GAME_IDS:
        rows = []
        prev = None
        for i in range(per_game_actions[gid]):
            repeat = prev is not None and rng.random() < 0.25
            disp = prev if repeat else rng.choice(
                ["UP", "DOWN", "LEFT", "RIGHT",
                 f"MOUSE(row={rng.randrange(64)}, col={rng.randrange(64)})"])
            rows.append(dict(type="action", action_num=i + 1, analysis_step=i // 4,
                             action_display=disp, action_name="ACTION6",
                             board_changed=(not repeat) and rng.random() < 0.6,
                             batch_index=1, batch_size=1, level=0, reward=0.0))
            prev = disp
        with (out / "artifacts" / f"{gid}_p0_events.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

    # ---- run log ----------------------------------------------------------
    lines: list[str] = list(CONT_BANNERS)
    lines.append(SEED_BANNER)
    if patch_failed:
        lines.append("animation: PATCH FAILED - continuing with VANILLA duck harness\n")
        write_log(out / "arc3-duck-animation-eval.log", lines)
        return out
    lines.append(BANNER)
    lines.append(GRAFT_BANNER)

    run_actions = run_multi = run_invisible = 0
    per_game_inv: dict[str, int] = {}
    per_game_multi: dict[str, int] = {}
    for gid in GAME_IDS:
        short = gid.split("-")[0]
        n_act = per_game_actions[gid]
        if short in TYPE1 and invisible:
            n_multi, inv_share = max(3, n_act // 6), 0.7
        elif short in TYPE2:
            n_multi, inv_share = max(3, n_act // 8), 0.0
        else:
            n_multi, inv_share = 0, 0.0
        run_actions += n_act
        for j in range(n_multi):
            is_inv = j < int(n_multi * inv_share)
            run_multi += 1
            if is_inv:
                run_invisible += 1
            per_game_multi[short] = per_game_multi.get(short, 0) + 1
            per_game_inv[short] = per_game_inv.get(short, 0) + (1 if is_inv else 0)
            lines.append(event_line(
                gid, "reject_or_consumed" if is_inv else "motion",
                f"MOUSE(row={rng.randrange(64)}, col={rng.randrange(64)})"
                if j % 3 == 0 else "UP",
                frames=rng.randrange(2, 6), unique=rng.randrange(2, 5),
                board_unchanged=1 if is_inv else 0,
                cells=rng.randrange(1, 400),
                bbox=f"[{rng.randrange(30)}, {rng.randrange(30)}, "
                     f"{rng.randrange(30, 63)}, {rng.randrange(30, 63)}]",
                run_actions=run_actions, run_multi=run_multi,
                run_invisible=run_invisible))

    engaged = sorted(g for g in per_game_inv if g in TYPE1 and per_game_inv[g] > 0)
    tokens_est = (int(total_tokens * 0.03) if token_breach
                  else run_multi * A.TOKENS_PER_SUMMARY)
    lines.append(canary_line(
        actions=run_actions, multi=run_multi, invisible=run_invisible,
        summaries=run_multi, errors=errors,
        games_with_events=len(per_game_multi),
        games_with_invisible=len([g for g, v in per_game_inv.items() if v]),
        audit_engaged=engaged, tokens_est=tokens_est))
    write_log(out / "arc3-duck-animation-eval.log", lines)
    return out


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
BASE = [REPO / "runs" / "kernel_pulls" / "w0_eval_s1",
        REPO / "runs" / "kernel_pulls" / "w0_cont_eval"]


def run(pull: Path, tag: str) -> dict:
    out_dir = FIX / "_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    rc = A.main(["--pull", str(pull), "--baseline", str(BASE[0]),
                 "--baseline", str(BASE[1]), "--date", tag,
                 "--out-dir", str(out_dir), "--quiet"])
    assert rc == 0, f"CLI returned {rc}"
    res = json.loads((out_dir / f"score_{tag}.json").read_text(encoding="utf-8"))
    md = (out_dir / f"score_{tag}.md").read_text(encoding="utf-8")
    res["_md"] = md
    return res


def main() -> int:
    FIX.mkdir(parents=True, exist_ok=True)

    print("\nA. healthy run")
    r = run(build_pull("a_healthy"), "a_healthy")
    c = r["canaries"]
    check("A1 verdict CANARIES CLEAR", r["verdict"]["verdict"] == "CANARIES CLEAR",
          r["verdict"]["verdict"])
    check("A2 K-A0..K-A4 all PASS",
          all(c[k]["status"] == "PASS" for k in ("K-A0", "K-A1", "K-A2", "K-A3", "K-A4")),
          json.dumps({k: c[k]["status"] for k in ("K-A0", "K-A1", "K-A2", "K-A3", "K-A4")}))
    check("A3 every canary carries a raw evidence line",
          all(any(c[k]["evidence"]) for k in ("K-A0", "K-A1", "K-A2", "K-A3", "K-A4")))
    check("A4 event lines parsed on >=5 distinct games",
          c["K-A1"]["distinct_games"] >= 5, str(c["K-A1"]["distinct_games"]))
    check("A5 M0 invisible rate > 0 and per-game rows cover 25 games",
          r["M0"]["invisible_rate"] > 0 and len(r["M0"]["per_game"]) == 25,
          f"{r['M0']['invisible_rate']} / {len(r['M0']['per_game'])}")
    check("A6 M0 expectation check MET (type-1 nonzero, ~0 elsewhere)",
          r["M0"]["expectation_check"] == "MET",
          f"misses={r['M0']['expectation_misses']} surprises={r['M0']['expectation_surprises']}")
    check("A7 M0 per-game rows print the offline-audit expectation next to the observed",
          all(row["audit_invisible_pct_combined"] is not None
              for row in r["M0"]["per_game"]),
          "audit column missing on some row")
    check("A8 event-line counters agree with the canary line",
          r["M0"]["event_lines_consistent_with_canary"] is True)
    check("A9 M1 refuses a verdict: NOT SCREENABLE + only legal string",
          r["M1"]["screenable_statement"] == A.NOT_SCREENABLE
          and r["M1"]["verdict"] == A.M1_LEGAL_VERDICT
          and "PASS" not in r["M1"]["verdict"] and "FAIL" not in r["M1"]["verdict"],
          r["M1"]["verdict"])
    s = r["M1"]["seal_arithmetic"]
    check("A10 seal arithmetic re-derived and matches the sealed values "
          "(sigma 0.14174/df 6, K3'' -0.2977, floor 0.4437 = 11.09 levels)",
          s["seal_arithmetic_match"] is True
          and s["checks"]["k3pp_line_rederived"] == A.SEALED_K3PP_M2
          and s["checks"]["floor_rederived"] == A.SEALED_FLOOR_M2
          and s["checks"]["levels_rederived"] == A.SEALED_FLOOR_LEVELS_M2,
          json.dumps(s["checks"]))
    check("A11 baseline family read from benchmark.json with label verified, m=2",
          r["M1"]["m"] == 2 and r["M1"]["all_baseline_labels_match"] is True
          and [b["lc_total"] for b in r["M1"]["baselines"]] == [16, 10],
          json.dumps([b["lc_total"] for b in r["M1"]["baselines"]]))
    check("A12 family SS re-check from the two pulls reproduces the sealed 0.0288",
          r["M1"]["family_ss_recheck"]["matches"] is True,
          json.dumps(r["M1"]["family_ss_recheck"]))
    check("A13 M2 is the DECIDING metric and reports the +17% external reference",
          r["M2"]["deciding_metric"] is True
          and r["M2"]["tokens_per_action_vs_external"]["his_pct"] == 17.0
          and r["M2"]["tokens_per_action_delta_pct"] is not None,
          json.dumps(r["M2"].get("tokens_per_action_delta_pct")))
    check("A14 M2 wall-clock/actions coupling states fewer-or-more actions explicitly",
          isinstance(r["M2"]["wallclock_actions_coupling"]["arm_executed_fewer_actions"], bool)
          and "actions per game than the family"
          in r["M2"]["wallclock_actions_coupling"]["statement"],
          r["M2"]["wallclock_actions_coupling"]["statement"])
    check("A15 M3 computed on all four type-1 games for arm and both baselines",
          r["M3"]["arm"]["games_available"] == 4
          and all(b["games_available"] == 4 for b in r["M3"]["baselines"].values()),
          json.dumps({k: v["games_available"] for k, v in r["M3"]["baselines"].items()}))
    check("A16 P1 legality PASS on the arm (continuation banner, no forbidden tokens)",
          r["P1"][0]["status"] == "PASS", json.dumps(r["P1"][0].get("forbidden_tokens_found")))
    check("A17 markdown carries the external-prior block and the M1 refusal",
          "EXTERNAL PRIOR" in r["_md"] and "p = 0.92" in r["_md"]
          and A.NOT_SCREENABLE in r["_md"] and A.M1_LEGAL_VERDICT in r["_md"])
    check("A18 markdown puts M2 (DECIDING) before M0",
          r["_md"].index("## 2. M2 (DECIDING") < r["_md"].index("## 3. M0"))
    check("A19 K-A3 threshold untouched at 1% and flagged as externally justified",
          c["K-A3"]["bound"] == 0.01 and c["K-A3"]["threshold_unchanged"] is True
          and "734369" in c["K-A3"]["external_source"])
    check("A20 parser survives spaces inside action=MOUSE(...) and bbox=[a, b, c, d]",
          r["event_lines"] > 0 and r["M0"]["event_line_multi"] == r["canary_line"]["multi"],
          f"{r['event_lines']} lines")

    print("\nB. PATCH FAILED run  ->  must be VOID, never FAIL")
    r = run(build_pull("b_patch_failed", patch_failed=True), "b_patch_failed")
    check("B1 verdict is VOID", r["verdict"]["verdict"] == "VOID", r["verdict"]["verdict"])
    check("B2 verdict is explicitly NOT a FAIL",
          "FAIL" not in r["verdict"]["verdict"] and "NOT a FAIL" in r["verdict"]["why"],
          r["verdict"]["why"])
    check("B3 K-A0 FAIL with the PATCH FAILED line as evidence",
          r["canaries"]["K-A0"]["status"] == "FAIL"
          and r["canaries"]["K-A0"]["patch_failed_line"] is True
          and any("PATCH FAILED" in e for e in r["canaries"]["K-A0"]["evidence"]))
    check("B4 no KILL is raised on a run that never applied the patch",
          r["verdict"]["kill_reasons"] == [], json.dumps(r["verdict"]["kill_reasons"]))
    check("B5 label is unstamped (no -animation-) and reported",
          r["canaries"]["K-A0"]["label_stamped"] is False, str(r["arm_label"]))

    print("\nC. zero-invisible run  ->  VOID + audit method back under review")
    r = run(build_pull("c_zero_invisible", invisible=False), "c_zero_invisible")
    check("C1 verdict is VOID", r["verdict"]["verdict"] == "VOID", r["verdict"]["verdict"])
    check("C2 K-A2 FAIL", r["canaries"]["K-A2"]["status"] == "FAIL")
    check("C3 audit-method-under-review flag raised",
          r["verdict"]["audit_method_under_review"] is True
          and r["canaries"]["K-A2"]["audit_method_under_review"] is True)
    check("C4 markdown carries the audit-under-review flag",
          "audit method itself goes back under review" in r["_md"])
    check("C5 K-A0/K-A1 still PASS (patch ran, events fired) -- the VOID is K-A2's",
          r["canaries"]["K-A0"]["status"] == "PASS"
          and r["canaries"]["K-A1"]["status"] == "PASS")
    check("C6 M0 flags the type-1 games as expectation MISSes",
          sorted(r["M0"]["expectation_misses"]) == sorted(TYPE1),
          json.dumps(r["M0"]["expectation_misses"]))

    print("\nD. token-breach run  ->  KILL")
    r = run(build_pull("d_token_breach", token_breach=True), "d_token_breach")
    check("D1 verdict is KILL", r["verdict"]["verdict"] == "KILL", r["verdict"]["verdict"])
    check("D2 K-A3 FAIL with the fraction over the 1% bound",
          r["canaries"]["K-A3"]["status"] == "FAIL"
          and r["canaries"]["K-A3"]["token_fraction"] >= 0.01,
          str(r["canaries"]["K-A3"]["token_fraction"]))
    check("D3 kill reason names K-A3", any("K-A3" in x for x in r["verdict"]["kill_reasons"]))
    check("D4 K-A3 fraction computed here (the log's token_fraction= is empty)",
          r["canary_line"]["token_fraction_reported"] is None
          and r["canaries"]["K-A3"]["token_fraction"] is not None)

    print("\nE. errors-nonzero run  ->  KILL")
    r = run(build_pull("e_errors", errors=7), "e_errors")
    check("E1 verdict is KILL", r["verdict"]["verdict"] == "KILL", r["verdict"]["verdict"])
    check("E2 K-A4 FAIL with errors=7",
          r["canaries"]["K-A4"]["status"] == "FAIL" and r["canaries"]["K-A4"]["errors"] == 7)
    check("E3 kill reason names K-A4", any("K-A4" in x for x in r["verdict"]["kill_reasons"]))

    print("\nF. REAL vanilla pulls (no animation at all) -> graceful VOID, no crash")
    for d, expect_p1 in ((REPO / "runs" / "kernel_pulls" / "w0_eval_s1", "PASS"),
                         (REPO / "runs" / "kernel_pulls" / "a22_v2_1", "FAIL")):
        if not d.is_dir():
            check(f"F {d.name} present", False, "missing")
            continue
        try:
            r = run(d, f"f_{d.name}")
        except Exception as exc:  # noqa: BLE001
            check(f"F {d.name} runs without crashing", False, repr(exc))
            continue
        check(f"F {d.name} runs without crashing and returns VOID",
              r["verdict"]["verdict"] == "VOID", r["verdict"]["verdict"])
        check(f"F {d.name} K-A0 FAIL (no banner, no ANIMATION_AWARE=1 stamp)",
              r["canaries"]["K-A0"]["status"] == "FAIL"
              and not r["canaries"]["K-A0"]["banner_present"])
        check(f"F {d.name} M0 reports zeros rather than dividing by zero",
              r["M0"]["invisible_actions"] == 0 and r["M0"]["multi_frame_actions"] == 0
              and r["M0"]["invisible_rate"] == 0.0)
        check(f"F {d.name} M1/M2/M3 still computed (no exception, real numbers)",
              r["M1"]["computed"] and r["M2"]["computed"]
              and r["M3"]["arm"]["games_available"] == 4)
        check(f"F {d.name} P1 legality == {expect_p1}",
              r["P1"][0]["status"] == expect_p1,
              json.dumps(list(r["P1"][0]["forbidden_tokens_found"])))
        check(f"F {d.name} markdown renders end to end",
              r["_md"].strip().endswith("_") and "## 6. P1 legality" in r["_md"])

    print(f"\n{'=' * 66}\nanimation_score self-test: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
