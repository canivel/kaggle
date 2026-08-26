"""Self-test for ``effnote_score.py`` -- CPU only, no GPU, no LLM, no network,
no Kaggle calls, $0.

WHY THIS EXISTS, AND WHY IT RUNS BEFORE THE ARM IS SCORED.

  * The animation arm was KILLED on 2026-08-12 by a canary firing on a field the
    builder never populated, against a denominator that could not contain the
    quantity it proxied. An untested scorer produced a verdict nobody could
    defend.
  * ``p1_score.py`` shipped with a log-parsing bug (Kaggle build logs are a JSON
    ARRAY, not text) that would have read "no canary line" off a HEALTHY arm and
    killed it, and it carried a canary that could not fail.

So: every canary in ``effnote_score.py`` is driven to BOTH outcomes here, the
DEAD-arm fixture must produce NO-PROMOTE, the arm-never-ran fixture must produce
VOID (not FAIL), and the missing-evidence fixture must NOT manufacture B1 = 0.0
(which would read as a spectacular PASS).

Fixtures are SYNTHETIC and are built under
``duck_eval/warpack/_test_fixtures/effnote/``. The canary line, the event lines
and the ACTIVE banner in those fixtures are produced by the **REAL emitter** --
this file imports ``_kaggle_dataset/effnote_patch.py`` and calls its own
``canary_report()``, ``_emit()`` and ``apply()``, capturing stdout. A field the
emitter renames or drops therefore breaks the fixtures and the tests, which is
the whole point.

The behavioural traces are real ``*_p0_events.jsonl`` files replayed by the REAL
``effnote_replay.replay_run`` -- the identical code that produced the sealed
control spread. B1 in the fixtures is therefore produced by the same path that
produces B1 on the arm, not by a stub.

Groups:
  X  SCORER-vs-EMITTER field cross-check: every field the scorer parses must be
     one the emitter actually writes, and the sealed constants must match the
     prereg + the shipped module + the on-disk control spread.
  A  healthy arm (B1 well below the line) -> every canary PASS -> PROMOTE.
  B  every canary driven to FAIL individually, with the sealed consequence.
  C  missing / empty / malformed evidence -> ERROR + VOID, never a silent 0,
     and NEVER B1 = 0.0 from an empty replay.
  D  the five sealed kill rules, each fired and each not-fired.
  E  the B1 decision boundary: strictly-below is strict (equality FAILS), and
     the arm may not be rescued by the control mean.
  F  log-format regression: a Kaggle JSON-array log must parse identically to
     raw text.
  G  the DEAD arm end-to-end: delivery perfect, B1 inside the control spread
     -> NO-PROMOTE (a good delivery number must NOT read as a win), and no
     secondary may rescue it.

Run:  .venv/Scripts/python.exe duck_eval/warpack/effnote_score_selftest.py
"""
from __future__ import annotations

import io
import json
import os
import shutil
import sys
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIX = HERE / "_test_fixtures" / "effnote"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "_kaggle_dataset"))
import effnote_score as S       # noqa: E402
import effnote_patch as EN      # noqa: E402

EMITTER_SRC = (HERE / "_kaggle_dataset" / "effnote_patch.py").read_text(
    encoding="utf-8")
PREREG = (REPO / "learnings" / "war_room" / "effnote_prereg_2026-08-13.md").read_text(
    encoding="utf-8")

PASS = 0
FAIL = 0

GAME_IDS = [
    "sk48-d8078629", "tn36-ef4dde99", "m0r0-492f87ba", "bp35-0a0ad940",
    "cn04-2fe56bfb", "dc22-fdcac232", "tu93-0768757b", "lp85-305b61c3",
    "ka59-38d34dbb", "wa30-ee6fef47", "vc33-5430563c", "lf52-271a04aa",
    "r11l-495a7899", "sc25-635fd71a", "sp80-589a99af", "ar25-0c556536",
    "sb26-7fbdac44", "cd82-fb555c5d", "re86-8af5384d", "s5i5-18d95033",
    "ls20-9607627b", "ft09-0d8bbf25", "su15-1944f8ab", "tr87-cd924810",
    "g50t-5849a774",
]


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}" + (f"  [{detail}]" if detail else ""))


# --------------------------------------------------------------------------- #
# fixture builders -- every EFFNOTE line comes from the REAL emitter
# --------------------------------------------------------------------------- #
def emit_banner(seams: int = 2) -> str:
    """Capture the SHIPPED ``apply()`` banner without importing the harness."""
    saved = os.environ.get("EFFNOTE")
    os.environ["EFFNOTE"] = "1"
    orig_apply_patches = EN._apply_patches
    orig_applied = EN._APPLIED
    EN._apply_patches = lambda: seams          # type: ignore[assignment]
    EN._APPLIED = False
    buf = io.StringIO()
    with redirect_stdout(buf):
        EN.apply(None)
    EN._apply_patches = orig_apply_patches     # type: ignore[assignment]
    EN._APPLIED = orig_applied
    if saved is None:
        os.environ.pop("EFFNOTE", None)
    else:
        os.environ["EFFNOTE"] = saved
    return buf.getvalue()


def emit_canary(per_game: dict[str, dict]) -> str:
    """Call the SHIPPED ``canary_report()`` and capture the line it prints."""
    old = dict(EN.CANARY)
    EN.CANARY.clear()
    EN.CANARY.update(per_game)
    buf = io.StringIO()
    with redirect_stdout(buf):
        EN.canary_report()
    EN.CANARY.clear()
    EN.CANARY.update(old)
    return buf.getvalue()


def emit_events(game: str, *, kinds: tuple[str, ...] = ("note", "game_end"),
                turns: int = 12, noted: int = 11, nz: int = 2, stag: int = 0,
                rev: int = 1, chars_max: int = 444, errors: int = 0) -> str:
    """Call the SHIPPED ``_emit()`` and capture the event lines it prints."""
    st = EN.EffNoteState(game)
    st.turns, st.noted, st.chars_max, st.errors = turns, noted, chars_max, errors
    st.fire_net_zero, st.fire_stagnation, st.fire_revisit = nz, stag, rev
    buf = io.StringIO()
    with redirect_stdout(buf):
        for k in kinds:
            EN._emit(k, st, "anum=3 level=1 used=9 target=57 over=0 chars=300 "
                            "fired=rev=5" if k == "note" else "max_actions_seen=14")
    return buf.getvalue()


GRAFT_LINE = ("effnote v1: graft applied from /kaggle/input/datasets/canivel/"
              "arc-war-kit (applied=True); NO warpack/ledger-graft/sentinel/"
              "compaction/animation/p1")
CONTINUATION_LINE = ("continuation v1: (f) game-over-continuation graft applied "
                     "from /kaggle/input/datasets/canivel/arc-war-kit "
                     "(applied=True); NO warpack/ledger")


def canary_counters(*, games: tuple[str, ...], turns: int, noted: int,
                    over: int, nz: int, stag: int, rev: int, fire_any: int,
                    chars_max: int, errors: int,
                    nz_games: int, stag_games: int, rev_games: int) -> dict:
    """Spread the totals over `games` so the emitter's per-game counts (the
    `/Ng` fields) land exactly where the test wants them."""
    out = {}
    for i, g in enumerate(games):
        out[g] = {"turns": turns // len(games), "noted": noted // len(games),
                  "chars_sum": 300 * (noted // len(games)),
                  "chars_max": chars_max if i == 0 else 0,
                  "over_target": over if i == 0 else 0,
                  "fire_net_zero": (nz if i == 0 else 1) if i < nz_games else 0,
                  "fire_stagnation": (stag if i == 0 else 1) if i < stag_games else 0,
                  "fire_revisit": (rev if i == 0 else 1) if i < rev_games else 0,
                  "fire_any": fire_any if i == 0 else 0,
                  "errors": errors if i == 0 else 0}
    # make the totals exact on the first game
    tot = lambda k: sum(v[k] for v in out.values())  # noqa: E731
    first = out[games[0]]
    first["turns"] += turns - tot("turns")
    first["noted"] += noted - tot("noted")
    return out


def write_game_events(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


B0 = [[0, 1], [1, 0]]
_FRESH = [0]


def fresh_board(_n: int = 0) -> list[list[int]]:
    """A board that has NEVER been seen before (globally unique). It must be
    genuinely unique: a small recycled palette makes the SHIPPED net-zero and
    revisit detectors fire on 'progress', which silently turns the healthy
    fixture into a stalling one."""
    _FRESH[0] += 1
    n = _FRESH[0]
    return [[n, n + 1], [n + 2, n + 3]]


def game_trace(*, revisit_after_stall: bool, n_tail_turns: int = 10,
               complete_level: bool = True) -> list[dict]:
    """A synthetic game with exactly one engineered STALL turn.

    turn 1  : 9 no-op actions on B0 -> at the start of turn 2 the SHIPPED
              detect_stagnation (>=8) and count_recent_revisits (>=4) both fire.
    turn 2  : the STALL turn. `revisit_after_stall` decides whether its actions
              return to B0 (already visited -> B1 = 1.0 on this turn) or move to
              never-seen boards (B1 = 0.0 on this turn).
    turn 3+ : fresh boards, so the stall RATE stays far under the 40% nag line.
    """
    rows: list[dict] = [dict(type="initial", board=B0, level=1)]
    step = 1
    anum = 0

    def act(board, changed, step_, completed=False):
        nonlocal anum
        anum += 1
        return dict(type="action", board=board, action_name=f"ACTION{anum % 5 + 1}",
                    action_display="UP", board_changed=changed,
                    level_completed=completed, analysis_step=step_,
                    action_num=anum, level=1)

    for _ in range(9):                       # turn 1 -- builds the stagnation run
        rows.append(act(B0, False, step))
    step += 1
    for _ in range(3):                       # turn 2 -- THE STALL TURN
        if revisit_after_stall:
            rows.append(act(B0, False, step))
        else:
            rows.append(act(fresh_board(anum + 100), True, step))
    for t in range(n_tail_turns):            # turns 3+ -- ordinary progress
        step += 1
        for _ in range(3):
            rows.append(act(fresh_board(anum + 500), True, step))
    if complete_level:
        rows[-1]["level_completed"] = True
    return rows


def build_pull(name: str, *, revisit_after_stall: bool,
               banner: str | None = None, graft: str | None = GRAFT_LINE,
               canary_line: str | None = None, event_blocks: str | None = None,
               levels_completed: int = 17, n_games: int = 25,
               extra_log: str = "", artifacts: bool = True,
               log_format: str = "json-array", log_name: str = "effnote.log",
               write_log: bool = True) -> Path:
    out = FIX / name
    if out.exists():
        shutil.rmtree(out)
    (out / "artifacts").mkdir(parents=True, exist_ok=True)

    games = GAME_IDS[:n_games]
    rows_bench = []
    for g in games:
        trace = game_trace(revisit_after_stall=revisit_after_stall,
                           complete_level=True)
        if artifacts:
            write_game_events(out / "artifacts" / f"{g}_p0_events.jsonl", trace)
        n_actions = sum(1 for r in trace if r["type"] == "action")
        rows_bench.append(dict(
            game_id=g, base_actions_per_level=[40] * 6,
            actions_per_level=[n_actions] + [0] * 5,
            levels_completed=1, number_of_levels=6))
    # trim/pad levels_completed to the requested run total
    total = sum(r["levels_completed"] for r in rows_bench)
    i = 0
    while total > levels_completed and i < len(rows_bench):
        if rows_bench[i]["levels_completed"] > 0:
            rows_bench[i]["levels_completed"] = 0
            total -= 1
        i += 1
    (out / "benchmark.json").write_text(json.dumps(
        {"label": "duck-harness-kaggle-effnote-v1-continuation-v1",
         "game_runs": rows_bench}), encoding="utf-8")

    body = (banner if banner is not None else emit_banner())
    if graft:
        body += graft + "\n"
    body += CONTINUATION_LINE + "\n"
    body += (event_blocks if event_blocks is not None
             else "".join(emit_events(g) for g in games))
    body += extra_log
    if canary_line is not None:
        body += canary_line
    if write_log:
        p = out / log_name
        if log_format == "json-array":
            p.write_text(json.dumps(
                [{"stream_name": "stdout", "time": 0.0, "data": chunk + "\n"}
                 for chunk in body.split("\n")]), encoding="utf-8")
        else:
            p.write_text(body, encoding="utf-8")
    return out


def healthy_canary(**over) -> str:
    kw = dict(games=tuple(GAME_IDS), turns=1200, noted=1160, over=300,
              nz=100, stag=4, rev=30, fire_any=100, chars_max=603, errors=0,
              nz_games=17, stag_games=1, rev_games=6)
    kw.update(over)
    return emit_canary(canary_counters(**kw))


def run(out: Path) -> dict:
    return S.score(out)


# --------------------------------------------------------------------------- #
# X -- scorer vs emitter, and scorer vs the SEAL
# --------------------------------------------------------------------------- #
def group_x() -> None:
    print("\nX  scorer-vs-emitter field cross-check + seal verification")
    missing = [f for f in S.CANARY_REQUIRED
               if f not in ("v",) and f"{f}=" not in EMITTER_SRC]
    check("X1 every canary field the scorer parses is written by the emitter",
          not missing, str(missing))
    line = healthy_canary()
    parsed, err = S.parse_canary(line)
    check("X2 the scorer's strict regex parses the REAL emitter's canary line",
          parsed is not None and err is None, f"{err} :: {line[:200]}")
    check("X3 the parsed fields round-trip to the emitter's own numbers",
          parsed and parsed["chars_max"] == 603 and parsed["errors"] == 0
          and parsed["nz_games"] == 17 and parsed["stag_games"] == 1
          and parsed["rev_games"] == 6 and parsed["bound"] == 700,
          json.dumps({k: parsed[k] for k in ("chars_max", "errors", "nz_games",
                                             "stag_games", "rev_games", "bound")}
                     if parsed else {}))
    evs = S.parse_events(emit_events("ft09-0d8bbf25"))
    check("X4 the scorer's event regex parses the REAL emitter's event lines",
          len(evs) == 2 and {e["kind"] for e in evs} == {"note", "game_end"},
          str(evs))
    merged = emit_events("ft09-0d8bbf25").replace("\n", "", 1)   # records merge
    mevs = S.parse_events(merged)
    check("X4b two EFFNOTE event lines MERGED by the kernel log format (no "
          "newline between records) are BOTH still parsed -- a greedy `detail` "
          "silently lost one of the 25 kind=game_end lines on the real pull",
          len(mevs) == 2 and {e["kind"] for e in mevs} == {"note", "game_end"},
          str([e["kind"] for e in mevs]))
    check("X5 the ACTIVE banner tokens the scorer greps are producible by apply()",
          S.BANNER_TOKEN in emit_banner() and S.SEAMS_TOKEN in emit_banner()
          and S.REPORT_ONLY_TOKEN in emit_banner())
    _, seal_err = S.verify_seal()
    check("X6 the sealed B1 line matches runs/effnote_replay/control_spread.json",
          seal_err is None, str(seal_err))
    check("X7 the sealed B1 line is the one written in the prereg",
          "0.3986" in PREREG and "PASS** requires the arm to fall **strictly below"
          in PREREG.replace("\n", " ").replace("  ", " ") or "0.3986" in PREREG,
          "prereg does not carry 0.3986")
    check("X8 the sealed character bound matches the SHIPPED module",
          S.CHAR_BOUND == EN.CFG.max_chars == 700,
          f"{S.CHAR_BOUND} vs {EN.CFG.max_chars}")
    check("X9 K-E1 floors are the RE-PRE-REGISTERED ones (stagnation >= 1)",
          S.K_E1_FLOORS == {"net_zero": 3, "revisit": 3, "stagnation": 1},
          str(S.K_E1_FLOORS))
    check("X10 the barred-statistics list names the first/second-half contrast "
          "and states it is NOT COMPUTED",
          any("first-half" in b["statistic"] and "NOT COMPUTED" in b["status"]
              for b in S.BARRED_STATISTICS))
    src = (HERE / "effnote_score.py").read_text(encoding="utf-8")
    check("X11 the scorer contains NO first-half/second-half computation at all",
          "first_half" not in src and "second_half" not in src,
          "a barred statistic is being computed")
    check("X12 the scorer reuses effnote_replay (the control-spread code), it "
          "does not re-implement the metrics",
          "import effnote_replay as RP" in src and "RP.replay_run" in src
          and "detect_net_zero_cycle" not in src)


# --------------------------------------------------------------------------- #
# A -- the healthy arm
# --------------------------------------------------------------------------- #
def group_a() -> None:
    print("\nA  healthy arm (B1 far below the line) -> PROMOTE")
    out = build_pull("a_healthy", revisit_after_stall=False,
                     canary_line=healthy_canary(), levels_completed=17)
    r = run(out)
    for k in S.CANARY_ORDER:
        check(f"A[{k}] PASS", r["canaries"][k]["status"] == "PASS",
              json.dumps(r["canaries"][k], default=str)[:400])
    b1 = r["PRIMARY"]["part2_B1"]
    check("A1 B1 is computed by the REAL replay and is below the sealed line",
          b1["value"] is not None and b1["value"] < S.B1_PASS_LINE,
          str(b1["value"]))
    check("A2 no kill rule fires", not any(x["fired"] for x in r["kill_rules"]),
          str([x["rule"] for x in r["kill_rules"] if x["fired"]]))
    check("A3 VERDICT = PROMOTE", r["verdict"]["verdict"] == "PROMOTE",
          r["verdict"]["why"])
    check("A4 a PROMOTE licenses only a second seed, never an LB claim",
          "SECOND SEED" in r["verdict"]["licenses"])
    check("A5 the instruments agree (replay vs live canary vs benchmark)",
          r["instrument_agreement"]["ok"] is True
          or r["instrument_agreement"]["errors"] == [],
          str(r["instrument_agreement"]["errors"]))


# --------------------------------------------------------------------------- #
# B -- every canary driven to FAIL
# --------------------------------------------------------------------------- #
def group_b() -> None:
    print("\nB  every canary driven to FAIL individually (the other direction)")

    # K-E0: no banner
    out = build_pull("b_ke0", revisit_after_stall=False, banner="",
                     graft=None, canary_line=healthy_canary())
    r = run(out)
    check("B[K-E0] FAIL when the ACTIVE banner / graft line is absent",
          r["canaries"]["K-E0"]["status"] == "FAIL")
    check("B[K-E0] its failure is VOID (the arm never installed), not a result",
          r["verdict"]["verdict"] == "VOID", r["verdict"]["why"][:200])

    # K-E0b leg 1: note_rate below 0.80
    out = build_pull("b_ke0b1", revisit_after_stall=False,
                     canary_line=healthy_canary(turns=1200, noted=600))
    r = run(out)
    check("B[K-E0b] FAIL when note_rate < 0.80",
          r["canaries"]["K-E0b"]["status"] == "FAIL"
          and r["canaries"]["K-E0b"]["leg1_pass"] is False)
    check("B[K-E0b] a leg-1 failure is INFRA DEATH -> VOID (kill rule 4)",
          r["verdict"]["verdict"] == "VOID"
          and any(x["rule"] == 4 and x["fired"] for x in r["kill_rules"]),
          r["verdict"]["why"][:200])

    # K-E0b leg 2: too few games carrying a kind=note line
    out = build_pull("b_ke0b2", revisit_after_stall=False,
                     canary_line=healthy_canary(),
                     event_blocks="".join(
                         emit_events(g, kinds=("note", "game_end"))
                         if i < 5 else emit_events(g, kinds=("game_end",))
                         for i, g in enumerate(GAME_IDS)))
    r = run(out)
    check("B[K-E0b] FAIL when <20 games carry a `kind=note` line",
          r["canaries"]["K-E0b"]["status"] == "FAIL"
          and r["canaries"]["K-E0b"]["leg2_pass"] is False,
          str(r["canaries"]["K-E0b"]["leg2_games_with_note_event"]))
    check("B[K-E0b] a leg-2-only failure is NO-PROMOTE, not VOID (the note did "
          "reach the model)",
          r["verdict"]["verdict"] == "NO-PROMOTE", r["verdict"]["why"][:200])

    # K-E1: stagnation on 0 games
    out = build_pull("b_ke1", revisit_after_stall=False,
                     canary_line=healthy_canary(stag=0, stag_games=0))
    r = run(out)
    check("B[K-E1] FAIL when a detector fires on too few games",
          r["canaries"]["K-E1"]["status"] == "FAIL",
          str(r["canaries"]["K-E1"]["detector_games"]))
    check("B[K-E1] a detector-sanity failure blocks PROMOTE and cannot rescue",
          r["verdict"]["verdict"] == "NO-PROMOTE", r["verdict"]["why"][:200])
    out = build_pull("b_ke1_orig", revisit_after_stall=False,
                     canary_line=healthy_canary(stag=2, stag_games=2))
    r = run(out)
    check("B[K-E1] the RE-PRE-REGISTRATION is disclosed in the output when the "
          "original >=3 floor would have failed",
          r["canaries"]["K-E1"]["status"] == "PASS"
          and r["canaries"]["K-E1"]["would_fail_at_original_floor"] is True)

    # K-E1': nagging
    out = build_pull("b_ke1p", revisit_after_stall=False,
                     canary_line=healthy_canary(turns=1200, noted=1160, nz=900,
                                                fire_any=900))
    r = run(out)
    check("B[K-E1'] FAIL when a detector fires on >40% of turns",
          r["canaries"]["K-E1'"]["status"] == "FAIL",
          str(r["canaries"]["K-E1'"]["worst_detector_rate"]))
    check("B[K-E1'] nagging is a KILL (rule 3)",
          r["verdict"]["verdict"] == "KILL"
          and any(x["rule"] == 3 and x["fired"] for x in r["kill_rules"]),
          r["verdict"]["why"][:200])

    # K-E2: levels_completed below the control minimum
    out = build_pull("b_ke2", revisit_after_stall=False,
                     canary_line=healthy_canary(), levels_completed=13)
    r = run(out)
    check("B[K-E2] FAIL when levels_completed < 14",
          r["canaries"]["K-E2"]["status"] == "FAIL",
          str(r["canaries"]["K-E2"]["levels_completed"]))
    check("B[K-E2] non-harm breach is a KILL (rule 1)",
          r["verdict"]["verdict"] == "KILL"
          and any(x["rule"] == 1 and x["fired"] for x in r["kill_rules"]))
    out = build_pull("b_ke2_edge", revisit_after_stall=False,
                     canary_line=healthy_canary(), levels_completed=14)
    r = run(out)
    check("B[K-E2] lc == 14 exactly is NOT a kill (the rule is lc < 14)",
          r["canaries"]["K-E2"]["status"] == "PASS"
          and not any(x["rule"] == 1 and x["fired"] for x in r["kill_rules"]))

    # K-E3: the character bound leaked
    out = build_pull("b_ke3", revisit_after_stall=False,
                     canary_line=healthy_canary(chars_max=1200))
    r = run(out)
    check("B[K-E3] FAIL when chars_max > 700",
          r["canaries"]["K-E3"]["status"] == "FAIL",
          str(r["canaries"]["K-E3"]["chars_max"]))
    check("B[K-E3] a leaked character bound is a KILL (rule 5)",
          r["verdict"]["verdict"] == "KILL"
          and any(x["rule"] == 5 and x["fired"] for x in r["kill_rules"]))
    out = build_pull("b_ke3_tok", revisit_after_stall=False,
                     canary_line=healthy_canary().rstrip("\n")
                                 + " token_fraction=0.031\n")
    r = run(out)
    check("B[K-E3] FAIL when a TOKEN-VALUED FIELD is appended to the CANARY line "
          "(the animation-arm defect: K-A3 fired on `token_fraction`)",
          r["canaries"]["K-E3"]["status"] == "FAIL"
          and r["canaries"]["K-E3"]["token_fields_in_canary"],
          str(r["canaries"]["K-E3"]["token_metric_hits"]))
    out = build_pull("b_ke3_prose", revisit_after_stall=False,
                     canary_line=healthy_canary())
    r = run(out)
    check("B[K-E3] the SHIPPED banner's own prose 'never a token fraction' does "
          "NOT trip the token check (a naive grep would kill a healthy arm)",
          r["canaries"]["K-E3"]["status"] == "PASS"
          and "never a token fraction" in emit_banner(),
          str(r["canaries"]["K-E3"]["token_metric_hits"]))
    check("B[K-E3] the SHIPPED module source carries NO token metric (the seal's "
          "own claim, asserted here so it can fail if the module changes)",
          not r["canaries"]["K-E3"]["token_fields_in_module"],
          str(r["canaries"]["K-E3"]["token_fields_in_module"]))
    # THE regression that produced a FALSE KILL on the first real run of this
    # scorer: Kaggle log records do not all end in a newline, so an unrelated
    # `[finished] ... tokens=N` print merges onto the end of an EFFNOTE line.
    out = build_pull("b_ke3_merged", revisit_after_stall=False,
                     canary_line=healthy_canary(),
                     extra_log="EFFNOTE v=1 kind=game_end game=lp85-305b61c3 "
                               "turns=28 noted=24 over=0 nz=6 stag=0 rev=2 "
                               "chars_max=542 errors=0 max_actions_seen=26"
                               "[finished] ar25-0c556536 state=gave_up level=0/8 "
                               "score=0.00 actions=18 tokens=68697\n")
    r = run(out)
    check("B[K-E3] an unrelated `tokens=` MERGED onto an EFFNOTE line by the "
          "kernel log format does NOT trip the token check (FALSE KILL "
          "regression, caught before the verdict was written)",
          r["canaries"]["K-E3"]["status"] == "PASS",
          str(r["canaries"]["K-E3"]["token_metric_hits"]))

    # K-E4: errors / PATCH FAILED
    out = build_pull("b_ke4", revisit_after_stall=False,
                     canary_line=healthy_canary(errors=7))
    r = run(out)
    check("B[K-E4] FAIL when errors > 0",
          r["canaries"]["K-E4"]["status"] == "FAIL",
          str(r["canaries"]["K-E4"]["errors"]))
    check("B[K-E4] errors > 0 is INFRA DEATH -> VOID, not a mechanism result",
          r["verdict"]["verdict"] == "VOID")
    out = build_pull("b_ke4_pf", revisit_after_stall=False,
                     canary_line=healthy_canary(),
                     extra_log="effnote: PATCH FAILED - continuing with VANILLA "
                               "duck harness\n")
    r = run(out)
    check("B[K-E4] FAIL on a `PATCH FAILED` line -> VOID",
          r["canaries"]["K-E4"]["status"] == "FAIL"
          and r["verdict"]["verdict"] == "VOID")

    # K-E5: another graft in the log
    out = build_pull("b_ke5", revisit_after_stall=False,
                     canary_line=healthy_canary(),
                     extra_log="compaction v3: ACTIVE (5 seams patched)\n")
    r = run(out)
    check("B[K-E5] FAIL when a FOREIGN graft banner is present",
          r["canaries"]["K-E5"]["status"] == "FAIL"
          and "compaction" in r["canaries"]["K-E5"]["foreign_banners"],
          str(r["canaries"]["K-E5"]["foreign_banners"]))
    check("B[K-E5] a companion graft means this is not the sealed arm -> VOID",
          r["verdict"]["verdict"] == "VOID")
    out = build_pull("b_ke5_ok", revisit_after_stall=False,
                     canary_line=healthy_canary())
    r = run(out)
    check("B[K-E5] the arm's OWN 'NO warpack/ledger-graft/sentinel/compaction/"
          "animation/p1' disclaimer does NOT trip the foreign-banner check",
          r["canaries"]["K-E5"]["status"] == "PASS",
          str(r["canaries"]["K-E5"]["foreign_banners"]))


# --------------------------------------------------------------------------- #
# C -- missing / unreadable evidence
# --------------------------------------------------------------------------- #
def group_c() -> None:
    print("\nC  missing / empty / malformed evidence -> ERROR + VOID, never a "
          "silent 0 and never B1 = 0.0")

    out = build_pull("c_nolog", revisit_after_stall=False,
                     canary_line=healthy_canary(), write_log=False)
    r = run(out)
    check("C1 a missing log is an ERROR on every log-derived canary",
          all(r["canaries"][k]["status"] == "ERROR"
              for k in ("K-E0", "K-E4", "K-E5")),
          str({k: r["canaries"][k]["status"] for k in S.CANARY_ORDER}))
    check("C2 a missing log is VOID, never a FAIL/KILL",
          r["verdict"]["verdict"] == "VOID", r["verdict"]["why"][:200])

    out = build_pull("c_nocanary", revisit_after_stall=False, canary_line=None)
    r = run(out)
    check("C3 an absent canary line is an ERROR, not errors=0 / note_rate=0",
          r["canaries"]["K-E4"]["status"] == "ERROR"
          and r["canaries"]["K-E4"]["errors"] is None
          and r["canaries"]["K-E0b"]["status"] == "ERROR")
    check("C4 an absent canary line is VOID", r["verdict"]["verdict"] == "VOID")

    out = build_pull("c_malformed", revisit_after_stall=False,
                     canary_line="EFFNOTE CANARY v=1 version=v1 games=25 "
                                 "turns=1200 noted=1160 errors=0\n")
    r = run(out)
    check("C5 a MALFORMED canary line names the missing fields and is an ERROR",
          r["canary"] is None and "MALFORMED" in (r["canary_error"] or ""),
          str(r["canary_error"])[:200])

    out = build_pull("c_unavailable", revisit_after_stall=False,
                     canary_line="EFFNOTE CANARY unavailable: RuntimeError('x')\n")
    r = run(out)
    check("C6 `EFFNOTE CANARY unavailable` is detected and reported verbatim",
          r["canary"] is None and "unavailable" in (r["canary_error"] or ""),
          str(r["canary_error"])[:200])

    # THE critical negative: no traces at all must NOT produce B1 = 0.0 = PASS
    out = build_pull("c_noartifacts", revisit_after_stall=False,
                     canary_line=healthy_canary(), artifacts=False)
    r = run(out)
    b1 = r["PRIMARY"]["part2_B1"]
    check("C7 an unreplayable arm has B1 = UNDEFINED, NOT 0.0 (which would read "
          "as a spectacular PASS)",
          b1["value"] is None and b1["status"] == "ERROR", str(b1))
    check("C8 an unreplayable arm is VOID, never PROMOTE",
          r["verdict"]["verdict"] == "VOID", r["verdict"]["why"][:200])

    # the seal itself must be verifiable
    payload, err = S.verify_seal(FIX / "does_not_exist.json")
    check("C9 a missing control spread makes the sealed line unverifiable "
          "(ERROR), it does not fall back to a hardcoded number",
          payload is None and "unverifiable" in (err or ""), str(err))
    drift = FIX / "drifted_spread.json"
    drift.parent.mkdir(parents=True, exist_ok=True)
    drift.write_text(json.dumps({"control_spread": {
        "B1_post_stall_revisit_rate": {"min": 0.55, "max": 0.6,
                                       "values": [0.55, 0.57, 0.6]}}}),
        encoding="utf-8")
    payload, err = S.verify_seal(drift)
    check("C10 a DRIFTED control spread is refused (a threshold that can move "
          "after the data lands is not a threshold)",
          payload is None and "DRIFT" in (err or ""), str(err))


# --------------------------------------------------------------------------- #
# D -- the five kill rules, fired and not-fired
# --------------------------------------------------------------------------- #
def group_d() -> None:
    print("\nD  the five sealed kill rules, each fired and each not-fired")
    healthy = run(build_pull("d_clean", revisit_after_stall=False,
                             canary_line=healthy_canary(), levels_completed=17))
    for i in range(1, 6):
        rule = [x for x in healthy["kill_rules"] if x["rule"] == i][0]
        check(f"D{i}a rule {i} does NOT fire on a healthy arm, and is EVALUABLE",
              not rule["fired"] and rule["evaluable"], rule["detail"])
    dead = run(build_pull("d_dead", revisit_after_stall=True,
                          canary_line=healthy_canary(), levels_completed=17))
    r2 = [x for x in dead["kill_rules"] if x["rule"] == 2][0]
    check("D2b rule 2 FIRES when B1 >= the control-spread minimum",
          r2["fired"] and r2["consequence"] == "NO-PROMOTE", r2["detail"])
    check("D-consequences are the sealed ones (1/3/5 KILL, 2 NO-PROMOTE, 4 VOID)",
          [x["consequence"].split()[0] for x in healthy["kill_rules"]]
          == ["KILL", "NO-PROMOTE", "KILL", "VOID", "KILL"],
          str([x["consequence"] for x in healthy["kill_rules"]]))


# --------------------------------------------------------------------------- #
# E -- the B1 decision boundary
# --------------------------------------------------------------------------- #
def group_e() -> None:
    print("\nE  the B1 decision boundary is STRICT and cannot be re-read")
    can, _ = S.parse_canary(healthy_canary())
    evs = S.parse_events("".join(emit_events(g) for g in GAME_IDS))
    cans = S.run_canaries("effnote v1: ACTIVE (2 seams patched) REPORT-ONLY\n"
                          + GRAFT_LINE, {"source": "json-array"}, can, None,
                          evs, [], None, None)
    for val, want in ((S.B1_PASS_LINE - 1e-12, "PASS"),
                      (S.B1_PASS_LINE, "FAIL"),
                      (S.B1_PASS_LINE + 1e-12, "FAIL"),
                      (0.4741420591980128, "FAIL"),   # the control MEAN
                      (0.4750542299349241, "FAIL"),   # a single control
                      (0.0, "PASS")):
        fake = {"B1_post_stall_revisit_rate": val,
                "B1c_nonstall_revisit_rate": 0.2}
        p = S.primary(can, fake, None, cans)
        check(f"E B1 = {val!r} -> {want}", p["part2_B1"]["status"] == want,
              str(p["part2_B1"]["status"]))
    check("E the control MEAN is reported but is NOT the gate",
          S.primary(can, {"B1_post_stall_revisit_rate": 0.45,
                          "B1c_nonstall_revisit_rate": 0.2}, None,
                    cans)["part2_B1"]["control_mean"] == 0.4741420591980128)


# --------------------------------------------------------------------------- #
# F -- log-format regression
# --------------------------------------------------------------------------- #
def group_f() -> None:
    print("\nF  log-format regression (the p1_score bug that would have killed a "
          "healthy arm)")
    a = run(build_pull("f_json", revisit_after_stall=False,
                       canary_line=healthy_canary(), levels_completed=17,
                       log_format="json-array"))
    b = run(build_pull("f_raw", revisit_after_stall=False,
                       canary_line=healthy_canary(), levels_completed=17,
                       log_format="raw"))
    check("F1 a Kaggle JSON-array log parses at all (canary found)",
          a["canary"] is not None and a["log"]["source"] == "json-array",
          str(a["canary_error"]))
    check("F2 json-array and raw text give the SAME verdict and the same canary "
          "statuses",
          a["verdict"]["verdict"] == b["verdict"]["verdict"]
          and {k: a["canaries"][k]["status"] for k in S.CANARY_ORDER}
          == {k: b["canaries"][k]["status"] for k in S.CANARY_ORDER},
          f"{a['verdict']['verdict']} vs {b['verdict']['verdict']}")
    check("F3 both are PROMOTE on the healthy fixture (no format-induced kill)",
          a["verdict"]["verdict"] == "PROMOTE" == b["verdict"]["verdict"])


# --------------------------------------------------------------------------- #
# G -- the DEAD arm, end to end
# --------------------------------------------------------------------------- #
def group_g() -> None:
    print("\nG  the DEAD arm: perfect delivery, B1 inside the control spread")
    out = build_pull("g_dead", revisit_after_stall=True,
                     canary_line=healthy_canary(), levels_completed=17)
    r = run(out)
    b1 = r["PRIMARY"]["part2_B1"]
    check("G1 delivery is PERFECT on the dead arm (this is the trap)",
          r["canaries"]["K-E0b"]["status"] == "PASS"
          and r["PRIMARY"]["part1_delivery"]["status"] == "PASS")
    check("G2 B1 is at or above the sealed line",
          b1["value"] >= S.B1_PASS_LINE, str(b1["value"]))
    check("G3 VERDICT = NO-PROMOTE (a good delivery number is NOT a win)",
          r["verdict"]["verdict"] == "NO-PROMOTE", r["verdict"]["why"][:200])
    check("G4 the primary is FAIL even though every canary passed",
          r["PRIMARY"]["status"] == "FAIL"
          and all(r["canaries"][k]["status"] == "PASS" for k in S.CANARY_ORDER))
    check("G5 a NO-PROMOTE licenses nothing",
          "nothing" in r["verdict"]["licenses"].lower())
    check("G6 the report renders and states the FAIL plainly",
          "NO-PROMOTE" in S.render_markdown(r)
          and "did not beat the *minimum*" in S.render_markdown(r))
    # a secondary may not rescue the dead primary
    check("G7 B3 (over-target burn) is present but is labelled supporting-only "
          "and does not appear in the verdict reasons",
          "B3" not in r["verdict"]["why"]
          and any("B3" in b["statistic"] for b in S.BARRED_STATISTICS))
    check("G8 lc is present but never cited as a win",
          any("Delta levels_completed" in b["statistic"] for b in S.BARRED_STATISTICS)
          and "may NEVER be cited as a win" in r["canaries"]["K-E2"]["barred"])


# --------------------------------------------------------------------------- #
def main() -> int:
    FIX.mkdir(parents=True, exist_ok=True)
    print(f"effnote_score self-test | fixtures={FIX}")
    group_x()
    group_a()
    group_b()
    group_c()
    group_d()
    group_e()
    group_f()
    group_g()
    print(f"\n{'=' * 66}\neffnote_score self-test: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
