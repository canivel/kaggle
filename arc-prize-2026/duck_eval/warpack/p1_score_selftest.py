"""Self-test for ``p1_score.py`` -- CPU only, no GPU, no LLM, no network, no
Kaggle calls, $0.

WHY THIS EXISTS. The animation arm was KILLED on 2026-08-12 by canary K-A3
firing on ``token_fraction``, a field the builder never populated, against a
denominator that could not contain the quantity it proxied. An untested scorer
produced a verdict nobody could defend. ``p1_score.py`` grades the P1 arm under
a SEALED prereg; it is written and tested HERE, BEFORE the kernel output exists,
so no threshold can be tuned to a verdict.

The real pull does not exist yet (``canivel/arc3-duck-p1-eval`` v1 is still
running), so every fixture is SYNTHETIC and is built under
``duck_eval/warpack/_test_fixtures/p1/``. Crucially the canary and event lines
in those fixtures are **produced by the REAL emitter** -- this file imports
``_kaggle_dataset/p1_suppressor_patch.py`` and calls its own ``canary_report()``
and ``_emit()``, capturing stdout. A field the emitter renames or drops
therefore breaks the fixtures and the tests, which is the whole point.

Groups:
  X   SCORER-vs-EMITTER field cross-check (static + live round-trip) --
      every field the scorer parses must be one the emitter actually writes,
      and the banner tokens the scorer greps must be producible by the module
      and the notebook cell that prints them.
  A   healthy run  -> all seven canaries PASS, no kill rule, PROMOTE.
  B   each canary K-P0..K-P6 driven to FAIL individually (the other direction).
  C   missing / empty / malformed evidence -> ERROR + DISCARD, never a silent
      0 and never a PASS.
  D   the five sealed kill rules, each fired and each not-fired; and the
      explicit non-rule: M2 below x1.019 does NOT kill.
  E   M0 band edges: exactly 3%, exactly 30%, just inside, just outside; plus
      the prereg's three replayed expectations (5.9% / 20.0% / 17.6%).
  F   log-format regression: Kaggle JSON-array logs must parse identically to
      raw text (the format the pull actually lands in).
  G   MECH-C block parsing, including the emitter's " (+N more)" suffix.

Run:  .venv/Scripts/python.exe duck_eval/warpack/p1_score_selftest.py
"""
from __future__ import annotations

import io
import json
import os
import re
import shutil
import sys
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIX = HERE / "_test_fixtures" / "p1"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "_kaggle_dataset"))
import p1_score as S            # noqa: E402
import p1_suppressor_patch as P  # noqa: E402

EMITTER_SRC = (HERE / "_kaggle_dataset" / "p1_suppressor_patch.py").read_text(
    encoding="utf-8")
NB_PATH = REPO / "notebooks" / "duckp1-eval" / "arc3-duck-p1-eval.ipynb"

PASS = 0
FAIL = 0
NOTES: list[str] = []

GAME_IDS = [
    "sk48-d8078629", "tn36-ef4dde99", "m0r0-492f87ba", "bp35-0a0ad940",
    "cn04-2fe56bfb", "dc22-fdcac232", "tu93-0768757b", "lp85-305b61c3",
    "ka59-38d34dbb", "wa30-ee6fef47", "vc33-5430563c", "lf52-271a04aa",
    "r11l-495a7899", "sc25-635fd71a", "sp80-589a99af", "ar25-0c556536",
    "sb26-7fbdac44", "cd82-fb555c5d", "re86-8af5384d", "s5i5-18d95033",
    "ls20-9607627b", "ft09-0d8bbf25", "su15-1944f8ab", "tr87-cd924810",
    "g50t-5849a774",
]
BASE_APL = [61, 177, 101, 103, 230, 181, 125, 92]


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}" + (f"  [{detail}]" if detail else ""))


# --------------------------------------------------------------------------- #
# fixture builders -- canary/event lines come from the REAL emitter
# --------------------------------------------------------------------------- #
def emit_canary(*, executed: int, declined: int, aborted: int, dup_exec: int,
                errors: int = 0, ambiguous: tuple[str, ...] = (),
                games: tuple[str, ...] = tuple(GAME_IDS),
                mode: str = "noop", confirm: str = "2",
                abort_revisit: str = "0") -> str:
    """Call the SHIPPED ``canary_report()`` and capture the line it prints."""
    saved_env = {k: os.environ.get(k) for k in
                 ("P1_MEMO_MODE", "P1_CONFIRM", "P1_ABORT_REVISIT")}
    os.environ["P1_MEMO_MODE"] = mode
    os.environ["P1_CONFIRM"] = confirm
    os.environ["P1_ABORT_REVISIT"] = abort_revisit
    old = dict(P.CANARY)
    P.CANARY.clear()
    for i, g in enumerate(games):
        P.CANARY[g] = {
            "actions_executed": executed if i == 0 else 0,
            "declined": declined if i == 0 else 0,
            "aborted": aborted if i == 0 else 0,
            "dup_requests": dup_exec if i == 0 else 0,
            "dup_executed": dup_exec if i == 0 else 0,
            "errors": errors if i == 0 else 0,
            "ambiguous": g in ambiguous,
            "ambiguity_pairs": 7 if g in ambiguous else 0,
        }
    buf = io.StringIO()
    with redirect_stdout(buf):
        P.canary_report()
    P.CANARY.clear()
    P.CANARY.update(old)
    for k, v in saved_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    return buf.getvalue()


def emit_events(games: list[str]) -> str:
    """Call the SHIPPED ``_emit()`` for every event kind, on `games`."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        for i, g in enumerate(games):
            st = P.P1State(g)
            st.level = i % 3
            st.declined, st.aborted = 4 + i, 2 + i
            st.dup_requests, st.dup_executed = 9 + i, 5 + i
            st.ambiguous = i % 5 == 0
            st.ambiguity_pairs = {("h1", "UP"), ("h2", "DOWN")} if st.ambiguous else set()
            P._emit("decline", st, "action=MOUSE(row=12, col=34) n=2")
            P._emit("batch_abort", st, "action=UP bi=2/6 saved=5")
            if st.ambiguous:
                P._emit("latent_state", st, "action=DOWN")
            P._emit("game_end", st, "")
    return buf.getvalue()


BANNER = ("p1 v1: ACTIVE (4 seams patched) - zero-information action suppressor. "
          "A: memo decline mode=noop confirm=2 max_declines=1 (online "
          "latent-state detector disables A per game on first contradictory "
          "outcome; NO game-id list). B: batch abort on no-op + intra-batch "
          "cycle (revisit is DEFAULT OFF: it cuts the level-completing batch on "
          "tu93/sp80/ar25 in the recorded traces). C: non-truncatable memory "
          "block ON (max_dead=8). Zero LLM calls, no locks, no game-id logic, "
          "vanilla fallback.\n")
GRAFT = ("p1 v1: graft applied from /kaggle/input/arc-war-kit (applied=True); "
         "NO warpack/ledger-graft/sentinel/compaction/animation\n")
SEED = ("p1-eval: SEED=1 zero-information action suppressor ON, NO "
        "warpack/ledger-graft/sentinel/compaction/animation (pairs with the "
        "duck-harness-kaggle-continuation-v1 family); P1_SUPPRESS=1; shipped "
        "defaults memo_mode=noop confirm=2 abort_revisit=OFF\n")
CONT = "continuation v1: game-over-continuation ACTIVE (2 modules patched)\n"
PATCH_FAILED = "p1: PATCH FAILED - continuing with VANILLA duck harness\n"

BLOCK = (
    "P1 memory (runner ground truth from the full transition record; never "
    "truncated - trust this over your recollection):\n"
    "- board fingerprint 0a1b2c3d4e5f6071; 4 distinct board(s) seen on this "
    "level; 11 (board,action) pair(s) recorded.\n"
    "- NOT YET TRIED from this exact board: LEFT, SPACE.\n"
    "- CONFIRMED NO EFFECT from this exact board: A1, A2, A3, A4, A5, A6, A7, "
    "UP (+3 more). Re-issuing one of these is not spent and tells you nothing "
    "new.\n"
    "- this game has latent state (7 pair(s) gave different outcomes from the "
    "same board); repeats are NOT suppressed here.\n")


def write_kaggle_log(path: Path, chunks: list[str]) -> None:
    """Kaggle build-log shape: a JSON array of {stream_name,time,data} records."""
    recs = [{"stream_name": "stdout", "time": 5.0 + i * 0.01, "data": s}
            for i, s in enumerate(chunks)]
    path.write_text(json.dumps(recs, indent=0).replace("},", "},\n"),
                    encoding="utf-8")


def level_rows(n_new: int, n_dup: int, completed: bool, step0: int) -> list[dict]:
    """One level's viewer rows over a single hub board: `1 + n_new` fresh
    (board, action) pairs and exactly `n_dup` duplicates of the first one."""
    acts = ["D"] + [f"N{i}" for i in range(n_new)] + ["D"] * n_dup
    rows = []
    for i, a in enumerate(acts):
        rows.append(dict(type="action", board=[[0, 1], [1, 0]], action_display=a,
                         action_name="ACTION1", board_changed=False,
                         level_completed=(completed and i == len(acts) - 1),
                         analysis_step=step0 + i, action_num=i + 1,
                         batch_index=1, batch_size=1, level=1, reward=0.0))
    return rows


def write_game_events(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"type": "initial", "board": [[0, 1], [1, 0]]}) + "\n")
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def build_bench(out: Path, lc_by_game: dict[str, int], apl_first: int = 120,
                n_levels: int = 8) -> None:
    game_runs = []
    for gid in GAME_IDS:
        lc = lc_by_game.get(gid, 0)
        apl = [0] * n_levels
        for i in range(max(1, lc)):
            apl[i] = apl_first
        game_runs.append(dict(
            game_id=gid, number_of_levels=n_levels,
            base_actions_per_level=BASE_APL, actions_per_level=apl,
            levels_completed=lc, state="gave_up", history=[],
            solver_note="tokens=100000", final_wallclock_seconds=7925.3,
            final_generated_tokens=0, final_uncached_input_tokens=0))
    (out / "benchmark.json").write_text(json.dumps(dict(
        label="duck-harness-kaggle-continuation-v1-p1-v1", n_passes=1,
        solver_label="duck-harness", game_runs=game_runs)), encoding="utf-8")


def build_pull(name: str, *, canary_line: str | None, banner: bool = True,
               seams: int = 4, applied: bool = True, patch_failed: bool = False,
               banner_defaults: bool = True, n_event_games: int = 25,
               lc_by_game: dict[str, int] | None = None,
               benchmark: bool = True, artifacts: bool = True,
               transcripts: bool = True, raw_text_log: bool = False,
               apl_first: int = 120, viewer_lc_bug: bool = False) -> Path:
    out = FIX / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    lc_by_game = lc_by_game if lc_by_game is not None else {
        g: (1 if i < 17 else 0) for i, g in enumerate(GAME_IDS)}
    if benchmark:
        build_bench(out, lc_by_game, apl_first=apl_first)

    if artifacts:
        (out / "artifacts").mkdir(exist_ok=True)
        for gid in GAME_IDS:
            lc = lc_by_game.get(gid, 0)
            rows = level_rows(6, 1, lc >= 1 and not viewer_lc_bug, 1)
            rows += level_rows(1, 3, False, 100)
            write_game_events(out / "artifacts" / f"{gid}_p0_events.jsonl", rows)

    if transcripts:
        (out / "transcripts").mkdir(exist_ok=True)
        for gid in GAME_IDS[:2]:
            txt = ""
            for step in (1, 2):
                txt += (f"\n--- analysis_step={step} | action={step} | "
                        f"12:15:29 | tool-agent ---\n[USER PROMPT]\nboard...\n"
                        f"{BLOCK}\n")
            (out / "transcripts" / f"{gid}_p0.txt").write_text(txt, encoding="utf-8")

    chunks = [CONT, SEED]
    if patch_failed:
        chunks.append(PATCH_FAILED)
    if banner:
        b = BANNER if banner_defaults else BANNER.replace(
            "mode=noop confirm=2", "mode=all confirm=2").replace(
            "revisit is DEFAULT OFF", "revisit is ON")
        if seams != 4:
            b = b.replace("(4 seams patched)", f"({seams} seams patched)")
        chunks.append(b)
    chunks.append(GRAFT if applied else GRAFT.replace("applied=True", "applied=False"))
    if n_event_games:
        chunks.append(emit_events(GAME_IDS[:n_event_games]))
    if canary_line:
        chunks.append(canary_line)
    log = out / "arc3-duck-p1-eval.log"
    if raw_text_log:
        log.write_text("".join(chunks), encoding="utf-8")
    else:
        write_kaggle_log(log, chunks)
    return out


def build_family(name: str, *, n_new_cleared: int = 8, n_dup_cleared: int = 1,
                 n_new_open: int = 1, n_dup_open: int = 8) -> Path:
    """Two synthetic comparator-family runs, tiny but real in shape.
    cleared-level dup rate = n_dup_cleared / (1 + n_new_cleared + n_dup_cleared);
    whole-run dup rate mixes in the (much duplicative) open level."""
    dirs = []
    for seed in (1, 2):
        out = FIX / f"{name}_s{seed}"
        if out.exists():
            shutil.rmtree(out)
        (out / "artifacts").mkdir(parents=True)
        build_bench(out, {g: 1 for g in GAME_IDS})
        (out / "benchmark.json").write_text(json.dumps(dict(
            label="duck-harness-kaggle-continuation-v1", n_passes=1,
            game_runs=json.loads((out / "benchmark.json").read_text())["game_runs"],
        )), encoding="utf-8")
        for gid in GAME_IDS:
            rows = level_rows(n_new_cleared, n_dup_cleared, True, 1)
            rows += level_rows(n_new_open, n_dup_open, False, 100)
            write_game_events(out / "artifacts" / f"{gid}_p0_events.jsonl", rows)
        dirs.append(out)
    return dirs[0].parent


FAMILY: list[Path] = []


def run_scorer(pull: Path, tag: str) -> dict:
    out = FIX / "_out"
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{tag}.json"
    rc = S.main(["--run", str(pull), "--out", str(dest), "--no-cache", "--quiet"]
                + [x for d in FAMILY for x in ("--family", str(d))])
    assert rc == 0, f"scorer returned {rc}"
    return json.loads(dest.read_text(encoding="utf-8"))


def healthy_canary(**kw) -> str:
    # the two flagged games are kept inside the lc>=1 block of every fixture, so
    # kill rule 3 only fires where a test deliberately makes it fire (group D3).
    d = dict(executed=4845, declined=200, aborted=106, dup_exec=150, errors=0,
             ambiguous=(GAME_IDS[2], GAME_IDS[5]))
    d.update(kw)
    return emit_canary(**d)


# --------------------------------------------------------------------------- #
def group_x() -> None:
    print("\nX. SCORER-vs-EMITTER field cross-check (the animation defect class)")

    def printed_fields(anchor: str) -> set[str]:
        i = EMITTER_SRC.find(anchor)
        j = EMITTER_SRC.find("flush=True", i)
        return set(re.findall(r"(\w+)=", EMITTER_SRC[i:j]))

    can_fields = printed_fields('f"P1 CANARY v=')
    ev_fields = printed_fields('f"P1 v={_EVENT_V} kind=')
    scorer_can = set(S.CANARY_RE.groupindex)
    scorer_ev = set(S.EVENT_RE.groupindex) - {"detail"}

    missing_can = sorted(scorer_can - can_fields)
    missing_ev = sorted(scorer_ev - ev_fields)
    check("X1 every canary field the scorer parses is EMITTED by canary_report()",
          not missing_can, f"scorer reads but emitter never writes: {missing_can}")
    check("X2 every event field the scorer parses is EMITTED by _emit()",
          not missing_ev, f"scorer reads but emitter never writes: {missing_ev}")
    check("X3 the scorer's declared CANARY_REQUIRED == its own parsed fields",
          set(S.CANARY_REQUIRED) == scorer_can,
          f"{sorted(set(S.CANARY_REQUIRED) ^ scorer_can)}")
    unused = sorted(can_fields - scorer_can)
    NOTES.append(f"canary fields emitted but not parsed by the scorer: {unused}")
    check("X4 no canary field is emitted with an empty value by construction "
          "(the K-A3 token_fraction= defect)",
          "=\"" not in EMITTER_SRC.split('f"P1 CANARY v=')[1].split("flush=True")[0]
          and "token_fraction" not in EMITTER_SRC)

    # live round-trip through the REAL emitter
    line = healthy_canary()
    can, err = S.parse_canary(line)
    check("X5 a canary line produced by the REAL canary_report() parses",
          can is not None and err is None, str(err))
    check("X6 every parsed canary field is non-None on the live round-trip",
          can is not None and all(can.get(k) is not None for k in S.CANARY_REQUIRED),
          json.dumps({k: can.get(k) for k in S.CANARY_REQUIRED} if can else {}))
    check("X7 the live round-trip preserves the counters exactly",
          can and (can["executed"], can["declined"], can["aborted"],
                   can["dup_exec"], can["errors"]) == (4845, 200, 106, 150, 0),
          json.dumps(can))
    check("X8 ambiguous_games round-trips as a NON-EMPTY list of game ids",
          can and can["ambiguous_games"] == sorted([GAME_IDS[2], GAME_IDS[5]]),
          json.dumps(can["ambiguous_games"] if can else None))

    evs = S.parse_events(emit_events(GAME_IDS[:6]))
    kinds = {e["kind"] for e in evs}
    check("X9 event lines from the REAL _emit() parse, incl. spaces in "
          "action=MOUSE(row=12, col=34)",
          len(evs) == 6 * 3 + 2 and kinds == {"decline", "batch_abort",
                                              "latent_state", "game_end"},
          f"{len(evs)} lines, kinds={sorted(kinds)}")
    check("X10 event game ids round-trip in full (hash suffix preserved)",
          {e["game"] for e in evs} == set(GAME_IDS[:6]))

    # banner tokens the scorer greps must be producible
    seams = EMITTER_SRC.count("patched += 1")
    check("X11 the emitter really patches 4 seams (K-P0 greps '(4 seams patched)')",
          seams == 4, f"patched += 1 appears {seams}x")
    check("X12 banner renders mode=noop / confirm=2 under the shipped defaults",
          P.CFG.memo_mode == "noop" and P.CFG.confirm == 2
          and P.CFG.abort_revisit is False
          and 'mode={CFG.memo_mode}' in EMITTER_SRC
          and 'confirm={CFG.confirm}' in EMITTER_SRC,
          f"mode={P.CFG.memo_mode} confirm={P.CFG.confirm} revisit={P.CFG.abort_revisit}")
    check("X13 the literal 'revisit is DEFAULT OFF' K-P2 greps exists in the emitter",
          "revisit is DEFAULT OFF" in EMITTER_SRC)
    check("X14 P1_CONFIRM is clamped to >= 2 in code (prereg sec4 consequence 2)",
          _confirm_with_env("1") == 2 and _confirm_with_env("5") == 5)
    nb = NB_PATH.read_text(encoding="utf-8") if NB_PATH.is_file() else ""
    check("X15 the notebook prints 'applied={applied}' (K-P0 greps applied=True) "
          "and 'p1: PATCH FAILED' (K-P0's negative)",
          "applied={applied}" in nb.replace("\\n", "") and "p1: PATCH FAILED" in nb)
    check("X16 the notebook calls canary_report() with NO arguments and the "
          "emitter takes none (the animation builder's defect was an argument "
          "the caller never passed)",
          "_p1.canary_report()" in nb and "def canary_report() -> dict" in EMITTER_SRC)
    check("X17 MECH-C's block regex matches the emitter's real first line",
          S._BLOCK_RE.search(BLOCK) is not None
          and "P1 memory (runner ground truth" in EMITTER_SRC)


def _confirm_with_env(v: str) -> int:
    old = os.environ.get("P1_CONFIRM")
    os.environ["P1_CONFIRM"] = v
    try:
        return P.CFG.confirm
    finally:
        if old is None:
            os.environ.pop("P1_CONFIRM", None)
        else:
            os.environ["P1_CONFIRM"] = old


def group_a() -> dict:
    print("\nA. healthy run -> 7/7 canaries PASS, no kill rule, PROMOTE")
    r = run_scorer(build_pull("a_healthy", canary_line=healthy_canary()), "a_healthy")
    c = r["canaries"]
    check("A1 verdict PROMOTE", r["verdict"]["verdict"] == "PROMOTE",
          f"{r['verdict']['verdict']}: {r['verdict']['why']}")
    for k in ("K-P0", "K-P1", "K-P2", "K-P3", "K-P4", "K-P5", "K-P6"):
        check(f"A2 {k} PASS", c[k]["status"] == "PASS",
              f"{c[k]['status']} err={c[k].get('error')}")
    check("A3 log parsed as a Kaggle json-array (NOT raw text)",
          r["log"]["source"] == "json-array", r["log"]["source"])
    check("A4 no evidence errors", r["evidence_errors"] == [],
          json.dumps(r["evidence_errors"]))
    check("A5 no kill rule fired and all five are EVALUABLE",
          all(not x["fired"] for x in r["kill_rules"])
          and all(x["evaluable"] for x in r["kill_rules"]),
          json.dumps([(x["rule"], x["fired"], x["evaluable"]) for x in r["kill_rules"]]))
    check("A6 M0 = 306/5151-style ratio computed as saved/requested",
          r["M0"]["saved"] == 306 and r["M0"]["requested"] == 5151,
          f"{r['M0']['saved']}/{r['M0']['requested']}")
    check("A7 M0 in band and reported as the PRIMARY endpoint",
          r["M0"]["in_band"] is True and r["M0"]["primary"] is True)
    check("A8 M1 is flagged NOT SCREENABLE (family m=2)",
          r["M1"]["screenable"] is False and "NOT SCREENABLE" in r["M1"]["note"])
    check("A9 M2 carries the explicit 'not a kill rule' statement",
          "NOT a kill" in r["M2"]["not_a_kill_rule"])
    check("A10 K-P6 compared the arm against a family with m=2 whose labels match",
          c["K-P6"]["family_m"] == 2 and r["family_dup"]["all_labels_match_family"],
          json.dumps(r["family_dup"]["runs"]))
    check("A11 kill rule 4 reports that a violation is NOT directly observable",
          r["kill_rules"][3]["directly_observable"] is False
          and r["kill_rules"][3]["sealed_settings_in_force"] is True)
    check("A12 kill rule 4's artifact-consistency read found no lc mismatch "
          "and actually checked all 25 games",
          r["kill_rules"][3]["viewer_vs_benchmark_lc_games_checked"] == 25
          and r["kill_rules"][3]["viewer_vs_benchmark_lc_mismatches"] == [],
          json.dumps(r["kill_rules"][3]["viewer_vs_benchmark_lc_mismatches"]))
    return r


def group_b() -> None:
    print("\nB. every canary driven to FAIL individually (a canary that cannot "
          "fail is the defect that killed the last arm)")

    r = run_scorer(build_pull("b0_no_banner", canary_line=healthy_canary(),
                              banner=False), "b0_no_banner")
    check("B0a K-P0 FAIL when the ACTIVE banner is absent",
          r["canaries"]["K-P0"]["status"] == "FAIL"
          and r["canaries"]["K-P0"]["banner_present"] is False)
    check("B0a' verdict is KILL via kill rule 1 (prereg sec6.1) and is flagged "
          "INFRA DEATH, not a mechanism result",
          r["verdict"]["verdict"] == "KILL"
          and any("kill rule 1" in x for x in r["verdict"]["kill_reasons"])
          and "INFRA DEATH" in r["kill_rules"][0]["note"])
    r = run_scorer(build_pull("b0_seams", canary_line=healthy_canary(), seams=3),
                   "b0_seams")
    check("B0b K-P0 FAIL when the banner reports 3 seams, not 4",
          r["canaries"]["K-P0"]["status"] == "FAIL"
          and r["canaries"]["K-P0"]["four_seams"] is False
          and r["canaries"]["K-P0"]["banner_present"] is True)
    r = run_scorer(build_pull("b0_applied", canary_line=healthy_canary(),
                              applied=False), "b0_applied")
    check("B0c K-P0 FAIL on applied=False",
          r["canaries"]["K-P0"]["status"] == "FAIL"
          and r["canaries"]["K-P0"]["applied_true"] is False)
    r = run_scorer(build_pull("b0_patchfail", canary_line=healthy_canary(),
                              patch_failed=True), "b0_patchfail")
    check("B0d K-P0 FAIL on the 'p1: PATCH FAILED' line even with a good banner",
          r["canaries"]["K-P0"]["status"] == "FAIL"
          and r["canaries"]["K-P0"]["patch_failed_line"] is True)

    r = run_scorer(build_pull("b1_4games", canary_line=healthy_canary(),
                              n_event_games=4), "b1_4games")
    check("B1 K-P1 FAIL at 4 distinct games, PASS at 5",
          r["canaries"]["K-P1"]["status"] == "FAIL"
          and r["canaries"]["K-P1"]["distinct_games"] == 4,
          str(r["canaries"]["K-P1"]["distinct_games"]))
    r5 = run_scorer(build_pull("b1_5games", canary_line=healthy_canary(),
                               n_event_games=5), "b1_5games")
    check("B1' K-P1 PASS at exactly 5 distinct games (boundary)",
          r5["canaries"]["K-P1"]["status"] == "PASS"
          and r5["canaries"]["K-P1"]["distinct_games"] == 5)
    r0 = run_scorer(build_pull("b1_0games", canary_line=healthy_canary(),
                               n_event_games=0), "b1_0games")
    check("B1'' K-P1 FAIL with zero event lines",
          r0["canaries"]["K-P1"]["status"] == "FAIL"
          and r0["canaries"]["K-P1"]["event_lines"] == 0)

    r = run_scorer(build_pull("b2_defaults", canary_line=healthy_canary(mode="all"),
                              banner_defaults=False), "b2_defaults")
    check("B2 K-P2 FAIL when the banner does not state the sealed defaults",
          r["canaries"]["K-P2"]["status"] == "FAIL"
          and r["canaries"]["K-P2"]["tokens"]["revisit is DEFAULT OFF"] is False)

    r = run_scorer(build_pull("b3_errors", canary_line=healthy_canary(errors=7)),
                   "b3_errors")
    check("B3 K-P3 FAIL on errors=7",
          r["canaries"]["K-P3"]["status"] == "FAIL"
          and r["canaries"]["K-P3"]["errors"] == 7)
    check("B3' errors!=0 fires kill rule 1 -> KILL",
          r["verdict"]["verdict"] == "KILL"
          and any("kill rule 1" in x for x in r["verdict"]["kill_reasons"]))

    # K-P4 both directions are group E; here only the FAIL-high direction
    r = run_scorer(build_pull("b4_high", canary_line=healthy_canary(
        executed=600, declined=300, aborted=100, dup_exec=20)), "b4_high")
    check("B4 K-P4 FAIL above the band (400/1000 = 40%)",
          r["canaries"]["K-P4"]["status"] == "FAIL"
          and r["canaries"]["K-P4"]["above_band"] is True,
          str(r["canaries"]["K-P4"]["rate"]))

    r = run_scorer(build_pull("b5_noamb", canary_line=healthy_canary(ambiguous=())),
                   "b5_noamb")
    check("B5 K-P5 FAIL when ambiguous_games is empty (emitter writes 'NONE')",
          r["canaries"]["K-P5"]["status"] == "FAIL"
          and r["canaries"]["K-P5"]["ambiguous_games"] == [])
    check("B5' an empty ambiguous list is NOT a silent pass and the run is "
          "NO-PROMOTE, not PROMOTE",
          r["verdict"]["verdict"] == "NO-PROMOTE"
          and "K-P5" in " ".join(r["verdict"]["reading_gate_failures"]))
    check("B5'' K-P5 carries the level-vs-game-scoped detector tension",
          "LEVEL-scoped" in r["canaries"]["K-P5"]["note"])

    # K-P6: arm dup_rate above / below / straddling the family definitions.
    # family fixture: cleared-level dup 1/10 = 10%, whole-run dup 9/20 = 45%.
    r = run_scorer(build_pull("b6_high", canary_line=healthy_canary(
        executed=4845, declined=200, aborted=106, dup_exec=3000)), "b6_high")
    check("B6 K-P6 FAIL when the arm dup_rate exceeds the family's",
          r["canaries"]["K-P6"]["status"] == "FAIL",
          f"{r['canaries']['K-P6']['arm_dup_rate']} vs "
          f"{r['canaries']['K-P6']['family_dup_rate_all_actions']}")
    r = run_scorer(build_pull("b6_disputed", canary_line=healthy_canary(
        executed=4845, declined=200, aborted=106, dup_exec=970)), "b6_disputed")
    check("B6' K-P6 DISPUTED when the two family definitions straddle the arm "
          "(escalate, do not execute)",
          r["canaries"]["K-P6"]["status"] == "DISPUTED"
          and r["verdict"]["verdict"] == "NO-PROMOTE",
          f"arm={r['canaries']['K-P6']['arm_dup_rate']} "
          f"all={r['canaries']['K-P6']['family_dup_rate_all_actions']} "
          f"cleared={r['canaries']['K-P6']['family_dup_rate_cleared_levels']}")


def group_c() -> None:
    print("\nC. missing / empty / malformed evidence -> explicit ERROR + DISCARD "
          "(never a silent 0, never a PASS)")

    r = run_scorer(build_pull("c_nocanary", canary_line=None), "c_nocanary")
    c = r["canaries"]
    check("C1 no canary line -> K-P3/K-P4/K-P5/K-P6 all ERROR (not FAIL, not PASS)",
          all(c[k]["status"] == "ERROR" for k in ("K-P3", "K-P4", "K-P5", "K-P6")),
          json.dumps({k: c[k]["status"] for k in ("K-P3", "K-P4", "K-P5", "K-P6")}))
    check("C2 a missing canary does NOT become errors=0 and does NOT become "
          "M0 = 0%",
          c["K-P3"]["errors"] is None and r["M0"]["rate"] is None
          and r["M0"]["saved"] is None)
    check("C3 verdict DISCARD, and kill rule 1 is UNEVALUABLE rather than fired",
          r["verdict"]["verdict"] == "DISCARD"
          and r["kill_rules"][0]["evaluable"] is False
          and not r["kill_rules"][0]["fired"], r["verdict"]["verdict"])
    check("C4 M0 < 3% (kill rule 5) is UNEVALUABLE on a missing canary -- an "
          "absent mechanism reading is not a null-by-delivery result",
          r["kill_rules"][4]["evaluable"] is False
          and not r["kill_rules"][4]["fired"])

    r = run_scorer(build_pull("c_unavail", canary_line=(
        "P1 CANARY unavailable: RuntimeError('boom')\n")), "c_unavail")
    check("C5 the builder's 'P1 CANARY unavailable' path is recognised as an "
          "ERROR with the raw line quoted, not parsed as a canary",
          r["canary"] is None and "unavailable" in (r["canary_error"] or "")
          and r["verdict"]["verdict"] == "DISCARD", str(r["canary_error"]))

    bad = healthy_canary().replace("errors=", "error=")
    r = run_scorer(build_pull("c_malformed", canary_line=bad), "c_malformed")
    check("C6 a canary line with a RENAMED field is reported as MALFORMED and "
          "names the missing field",
          r["canary"] is None and "MALFORMED" in (r["canary_error"] or "")
          and "'errors'" in (r["canary_error"] or ""), str(r["canary_error"]))
    check("C7 a malformed canary yields DISCARD, never a verdict",
          r["verdict"]["verdict"] == "DISCARD" and r["verdict"]["readable"] is False)

    r = run_scorer(build_pull("c_nobench", canary_line=healthy_canary(),
                              benchmark=False), "c_nobench")
    check("C8 missing benchmark.json does NOT become levels_completed=0 -> "
          "kill rule 2; it is UNEVALUABLE + DISCARD",
          r["kill_rules"][1]["evaluable"] is False
          and not r["kill_rules"][1]["fired"]
          and r["M1"]["arm_lc_total"] is None
          and r["M2"]["arm_score"] is None
          and r["verdict"]["verdict"] == "DISCARD",
          json.dumps(r["kill_rules"][1]))

    out = FIX / "c_nolog"
    build_pull("c_nolog", canary_line=healthy_canary())
    (out / "arc3-duck-p1-eval.log").unlink()
    r = run_scorer(out, "c_nolog")
    check("C9 no log at all -> every canary ERROR, verdict DISCARD, no KILL",
          all(r["canaries"][k]["status"] == "ERROR" for k in
              ("K-P0", "K-P1", "K-P2", "K-P3"))
          and r["verdict"]["verdict"] == "DISCARD"
          and r["verdict"]["kill_reasons"] == [],
          json.dumps({k: r["canaries"][k]["status"] for k in r["canaries"]}))
    check("C10 the missing log is named in evidence_errors",
          any("LOG:" in x for x in r["evidence_errors"]),
          json.dumps(r["evidence_errors"]))

    amb_raw = ",".join(sorted([GAME_IDS[2], GAME_IDS[5]]))
    r = run_scorer(build_pull("c_emptyamb", canary_line=healthy_canary().replace(
        f"ambiguous_games={amb_raw}", "ambiguous_games=")), "c_emptyamb")
    check("C11 an EMPTY ambiguous_games= value parses but FAILS K-P5 -- the "
          "exact animation shape (empty field silently satisfying a gate)",
          r["canary"] is not None and r["canary"]["ambiguous_games"] == []
          and r["canaries"]["K-P5"]["status"] == "FAIL")

    r = run_scorer(build_pull("c_zeroactions", canary_line=healthy_canary(
        executed=0, declined=0, aborted=0, dup_exec=0)), "c_zeroactions")
    check("C12 requested=0 makes M0 UNDEFINED (ERROR), not 0% -> DISCARD, and "
          "kill rule 5 does not fire on it",
          r["canaries"]["K-P4"]["status"] == "ERROR"
          and r["M0"]["rate"] is None
          and not r["kill_rules"][4]["fired"]
          and r["verdict"]["verdict"] == "DISCARD",
          json.dumps(r["M0"]))

    r = run_scorer(build_pull("c_viewerbug", canary_line=healthy_canary(),
                              viewer_lc_bug=True), "c_viewerbug")
    check("C13 kill rule 4's artifact-consistency read catches viewer/benchmark "
          "levels_completed disagreement",
          len(r["kill_rules"][3]["viewer_vs_benchmark_lc_mismatches"]) == 17,
          str(len(r["kill_rules"][3]["viewer_vs_benchmark_lc_mismatches"])))


def group_d() -> None:
    print("\nD. the five sealed kill rules (prereg sec6), each in both directions")

    # rule 2: levels_completed <= 15
    lc15 = {g: (1 if i < 15 else 0) for i, g in enumerate(GAME_IDS)}
    lc16 = {g: (1 if i < 16 else 0) for i, g in enumerate(GAME_IDS)}
    r = run_scorer(build_pull("d2_kill", canary_line=healthy_canary(),
                              lc_by_game=lc15), "d2_kill")
    check("D2a kill rule 2 FIRES at levels_completed = 15",
          r["kill_rules"][1]["fired"] and r["verdict"]["verdict"] == "KILL",
          json.dumps(r["kill_rules"][1]["detail"]))
    r = run_scorer(build_pull("d2_ok", canary_line=healthy_canary(),
                              lc_by_game=lc16), "d2_ok")
    check("D2b kill rule 2 does NOT fire at 16 (the sealed line is '<= 15')",
          not r["kill_rules"][1]["fired"] and r["verdict"]["verdict"] == "PROMOTE",
          json.dumps(r["kill_rules"][1]["detail"]))

    # rule 3: an ambiguity-flagged game loses a level vs the family per-game mean
    lc_loss = {g: (1 if i < 17 else 0) for i, g in enumerate(GAME_IDS)}
    lc_loss[GAME_IDS[2]] = 0                      # m0r0 is flagged ambiguous
    lc_loss[GAME_IDS[20]] = 1                     # keep the total above 15
    lc_loss[GAME_IDS[21]] = 1
    r = run_scorer(build_pull("d3_kill", canary_line=healthy_canary(),
                              lc_by_game=lc_loss), "d3_kill")
    check("D3a kill rule 3 FIRES when a flagged game drops below the family "
          "per-game mean",
          r["kill_rules"][3 - 1]["fired"]
          and r["kill_rules"][2]["losers"][0]["game"] == "m0r0"
          and r["verdict"]["verdict"] == "KILL",
          json.dumps(r["kill_rules"][2]))
    r = run_scorer(build_pull("d3_ok", canary_line=healthy_canary()), "d3_ok")
    check("D3b kill rule 3 does NOT fire when the flagged games hold their levels",
          not r["kill_rules"][2]["fired"] and r["kill_rules"][2]["evaluable"],
          json.dumps(r["kill_rules"][2]["detail"]))

    # rule 4: the invariant that guarantees it must be in force
    r = run_scorer(build_pull("d4_kill", canary_line=healthy_canary(
        mode="all", abort_revisit="1")), "d4_kill")
    check("D4a kill rule 4 FIRES when the run did not ship the sealed settings "
          "(mode=all / abort_revisit=1 -- prereg sec4 says these delete "
          "level-completing actions)",
          r["kill_rules"][3]["fired"] and r["verdict"]["verdict"] == "KILL",
          json.dumps(r["kill_rules"][3]["detail"]))
    r = run_scorer(build_pull("d4_ok", canary_line=healthy_canary()), "d4_ok")
    check("D4b kill rule 4 does NOT fire under the sealed settings",
          not r["kill_rules"][3]["fired"]
          and r["kill_rules"][3]["observed_settings"]
          == {"mode": "noop", "confirm": 2, "abort_revisit": 0},
          json.dumps(r["kill_rules"][3]["observed_settings"]))

    # rule 5: M0 < 3%
    r = run_scorer(build_pull("d5_kill", canary_line=healthy_canary(
        executed=9800, declined=100, aborted=100, dup_exec=50)), "d5_kill")
    check("D5a kill rule 5 FIRES at M0 = 2.00% and the verdict is KILL",
          r["kill_rules"][4]["fired"] and r["verdict"]["verdict"] == "KILL"
          and abs(r["M0"]["rate"] - 0.02) < 1e-12,
          json.dumps(r["M0"]["rate"]))
    check("D5b M0 ABOVE the band is explicitly NOT kill rule 5",
          "NOT this kill rule" in r["kill_rules"][4]["note"])
    r = run_scorer(build_pull("d5_high", canary_line=healthy_canary(
        executed=600, declined=300, aborted=100, dup_exec=20)), "d5_high")
    check("D5c M0 = 40% -> DISCARD (inspect before reading), NOT KILL",
          r["verdict"]["verdict"] == "DISCARD"
          and not r["kill_rules"][4]["fired"]
          and any("ABOVE the 30% band" in x for x in r["verdict"]["discard_reasons"]),
          f"{r['verdict']['verdict']} {r['verdict']['why']}")

    # THE explicit non-rule: M2 below x1.019
    r = run_scorer(build_pull("d_m2low", canary_line=healthy_canary(),
                              apl_first=100000), "d_m2low")
    check("D6a a near-zero M2 (arm RHAE score collapsed) fires NO kill rule",
          all(not x["fired"] for x in r["kill_rules"]) and r["M2"]["arm_score"] < 0.01,
          f"score={r['M2']['arm_score']} kills={r['verdict']['kill_reasons']}")
    check("D6b M2 below x1.019 is NOT a kill rule -- verdict stays PROMOTE and "
          "no kill reason mentions M2 or the multiplier",
          r["verdict"]["verdict"] == "PROMOTE"
          and not any(("M2" in x or "multiplier" in x)
                      for x in r["verdict"]["kill_reasons"]),
          f"{r['verdict']['verdict']}: {r['verdict']['why']}")
    check("D6c the sealed multiplier is carried as an EXPECTATION only",
          r["M2"]["sealed_multiplier_expectation"] == 1.019)

    check("D7 exactly five kill rules are evaluated, numbered 1..5",
          [x["rule"] for x in r["kill_rules"]] == [1, 2, 3, 4, 5],
          json.dumps([x["rule"] for x in r["kill_rules"]]))


def group_e() -> None:
    print("\nE. M0 band edges [3%, 30%] (prereg sec3 square brackets = both ends "
          "INCLUSIVE; kill rule 5 is the strict '< 3%')")

    def m0_case(tag, saved, requested, decl_share=0.5):
        declined = int(saved * decl_share)
        aborted = saved - declined
        executed = requested - saved
        # keep dup_rate well under the family's so K-P6 never confounds the
        # band test (this group is about M0 and M0 only)
        return run_scorer(build_pull(tag, canary_line=healthy_canary(
            executed=executed, declined=declined, aborted=aborted,
            dup_exec=int(executed * 0.05))), tag)

    r = m0_case("e_exact3", 3, 100)
    check("E1 EXACTLY 3.00% is IN band (inclusive) -> K-P4 PASS",
          r["M0"]["rate"] == 0.03 and r["canaries"]["K-P4"]["status"] == "PASS",
          str(r["M0"]["rate"]))
    check("E2 EXACTLY 3.00% does NOT fire kill rule 5 (the rule is M0 < 3%)",
          not r["kill_rules"][4]["fired"] and r["verdict"]["verdict"] == "PROMOTE",
          f"{r['verdict']['verdict']}")
    r = m0_case("e_exact30", 30, 100)
    check("E3 EXACTLY 30.00% is IN band (inclusive) -> K-P4 PASS, no DISCARD",
          r["M0"]["rate"] == 0.30 and r["canaries"]["K-P4"]["status"] == "PASS"
          and r["verdict"]["verdict"] == "PROMOTE", str(r["M0"]["rate"]))
    r = m0_case("e_in_low", 301, 10000)
    check("E4 3.01% (just inside) PASSES", r["canaries"]["K-P4"]["status"] == "PASS")
    r = m0_case("e_out_low", 299, 10000)
    check("E5 2.99% (just outside) FAILS K-P4 and FIRES kill rule 5 -> KILL",
          r["canaries"]["K-P4"]["status"] == "FAIL"
          and r["kill_rules"][4]["fired"] and r["verdict"]["verdict"] == "KILL")
    r = m0_case("e_in_high", 2999, 10000)
    check("E6 29.99% (just inside) PASSES", r["canaries"]["K-P4"]["status"] == "PASS")
    r = m0_case("e_out_high", 3001, 10000)
    check("E7 30.01% (just outside) FAILS K-P4, does NOT kill, and DISCARDs "
          "for inspection",
          r["canaries"]["K-P4"]["status"] == "FAIL"
          and not r["kill_rules"][4]["fired"]
          and r["verdict"]["verdict"] == "DISCARD")

    print("   the prereg's three sealed replay expectations")
    for tag, (saved, requested, pct) in {
        "animation_v1": (306, 5151, "5.9"),
        "a22_v2_seed1": (697, 3492, "20.0"),
        "a22_compaction_v1": (841, 4777, "17.6"),
    }.items():
        r = m0_case(f"e_replay_{tag}", saved, requested)
        rate = r["M0"]["rate"]
        check(f"E8 {tag} {saved}/{requested} scores {pct}% and is IN band, "
              f"no kill, PROMOTE",
              f"{rate*100:.1f}" == pct and r["M0"]["in_band"] is True
              and r["verdict"]["verdict"] == "PROMOTE",
              f"{rate*100:.4f}% verdict={r['verdict']['verdict']}")
    check("E9 the sealed replay expectations are recorded in the report",
          set(S.SEALED_M0_REPLAY) == {"animation_v1", "a22_v2_seed1",
                                      "a22_compaction_v1"})


def group_f() -> None:
    print("\nF. log-format regression -- the pull lands as a Kaggle JSON array")
    a = run_scorer(build_pull("f_json", canary_line=healthy_canary()), "f_json")
    b = run_scorer(build_pull("f_raw", canary_line=healthy_canary(),
                              raw_text_log=True), "f_raw")
    check("F1 a JSON-array log yields the SAME canary as the same content in "
          "raw text (reading the array raw made every ^P1 match fail)",
          a["canary"] == b["canary"] and a["canary"] is not None,
          f"{a['log']['source']} vs {b['log']['source']}")
    check("F2 both formats produce the same verdict and the same 7 canary states",
          a["verdict"]["verdict"] == b["verdict"]["verdict"] == "PROMOTE"
          and {k: a["canaries"][k]["status"] for k in a["canaries"]}
          == {k: b["canaries"][k]["status"] for k in b["canaries"]})
    check("F3 the log format actually used is reported in the output",
          a["log"]["source"] == "json-array" and b["log"]["source"] == "raw-text")

    out = FIX / "f_vllm"
    build_pull("f_vllm", canary_line=healthy_canary())
    (out / "vllm-openai-server.log").write_text("noise\n", encoding="utf-8")
    r = run_scorer(out, "f_vllm")
    check("F4 the vLLM server log is never mistaken for the run log",
          "vllm" not in Path(r["log"]["path"] or "x").name.lower()
          and r["verdict"]["verdict"] == "PROMOTE", str(r["log"]["path"]))


def group_g() -> None:
    print("\nG. MECH-C block parsing (diagnostic, not pre-registered)")
    out = FIX / "g_mechc"
    build_pull("g_mechc", canary_line=healthy_canary())
    # the agent re-issues UP, which is the LAST entry of a capped dead list and
    # therefore carries the emitter's " (+3 more)" suffix.
    for gid in GAME_IDS[:2]:
        rows = [dict(type="action", board=[[0, 1], [1, 0]], action_display="UP",
                     action_name="ACTION1", board_changed=False,
                     level_completed=False, analysis_step=1, action_num=1,
                     batch_index=1, batch_size=1, level=1, reward=0.0),
                dict(type="action", board=[[0, 1], [1, 0]], action_display="LEFT",
                     action_name="ACTION2", board_changed=True,
                     level_completed=True, analysis_step=2, action_num=2,
                     batch_index=1, batch_size=1, level=1, reward=0.0)]
        write_game_events(out / "artifacts" / f"{gid}_p0_events.jsonl", rows)
    r = run_scorer(out, "g_mechc")
    mc = r["mechanism_c"]
    check("G1 the block is detected on every transcript turn",
          mc["turns_with_block"] == 4 and mc["turns_seen"] == 4,
          json.dumps({k: mc[k] for k in ("turns_seen", "turns_with_block")}))
    check("G2 untried / dead / latent lines are all recognised",
          mc["blocks_with_untried"] == 4 and mc["blocks_with_dead"] == 4
          and mc["blocks_flagging_latent"] == 4)
    check("G3 the block is under the sealed 900-char bound",
          mc["block_chars_max"] < 900, str(mc["block_chars_max"]))
    check("G4 a dead primitive carrying the emitter's ' (+N more)' suffix is "
          "still matched (it was silently unmatchable before): 2 games x "
          "(UP on step 1, LEFT on step 2) -> 4 issued, 2 dead re-issues",
          mc["dead_reissued"] == 2 and mc["actions_after_dead_turns"] == 4,
          json.dumps({k: mc[k] for k in ("dead_reissued",
                                         "actions_after_dead_turns")}))
    check("G5 MECH-C reports an explicit error when transcripts are absent",
          "error" in S.mechanism_c(FIX / "_out", ""),
          json.dumps(S.mechanism_c(FIX / "_out", "")))


# --------------------------------------------------------------------------- #
def main() -> int:
    FIX.mkdir(parents=True, exist_ok=True)
    fam_root = build_family("family")
    FAMILY.extend(sorted(p for p in fam_root.glob("family_s*") if p.is_dir()))
    print(f"p1_score self-test | fixtures={FIX} | family={[p.name for p in FAMILY]}")

    group_x()
    group_a()
    group_b()
    group_c()
    group_d()
    group_e()
    group_f()
    group_g()

    if NOTES:
        print("\nNOTES")
        for n in NOTES:
            print(f"  - {n}")
    print(f"\n{'=' * 66}\np1_score self-test: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
