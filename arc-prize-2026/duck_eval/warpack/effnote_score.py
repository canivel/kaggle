"""EFFNOTE arm scorer -- reads a pulled `canivel/arc3-duck-effnote-eval` output
and grades it against the SEALED prereg
`learnings/war_room/effnote_prereg_2026-08-13.md`.

Everything graded here is pre-registered. Nothing is invented after the fact and
no threshold is computed from the arm's own data.

  K-E0 .. K-E5   canaries (prereg sec4), each with its sealed consequence.
  M0 part 1      PRIMARY, mechanism DELIVERY (note_rate, chars_max, errors,
                 per-game coverage). Necessary, NOT sufficient -- P1 mechanism C
                 delivered on 96.3% of turns and was dead.
  M0 part 2      PRIMARY, the BEHAVIOURAL test: B1 (post-stall revisit rate)
                 against the CONTROL SPREAD computed BEFORE the arm existed.
                 PASS requires B1 STRICTLY BELOW 0.39862542955326463, the
                 MINIMUM of the three block-free controls -- not the mean, not
                 any single control, not the arm's own first half.
  M1 / M2        DESCRIPTIVE ONLY (family m = 2 => Delta-lc NOT SCREENABLE;
                 RHAE and the M0-median are descriptive by seal).

TWO DEFECTS OF THE PREVIOUS SCORER THAT ARE FIXED HERE BY CONSTRUCTION.

 1. `p1_score.py` shipped with a log-parsing bug (Kaggle build logs are a JSON
    ARRAY of {stream_name,time,data} records, not text) that would have read
    "no canary line" off a perfectly healthy arm and killed it. The proven
    loader is IMPORTED from `p1_score`, not re-implemented, and the self-test
    drives both formats.
 2. `animation_score` carried a canary that could not fail. Every canary here is
    driven to BOTH outcomes in `effnote_score_selftest.py`, and every
    STRUCTURALLY DEGENERATE check is labelled as such in its own output rather
    than being reported as a pass (see K-E0b leg 1).

EVIDENCE DISCIPLINE. A missing, empty or unparseable field NEVER degrades to
0 / "" / PASS. In particular: an arm whose traces cannot be replayed has
B1 = UNDEFINED, never B1 = 0.0 (which would read as a spectacular PASS). That
path produces VOID.

THE ARM'S OWN FIRST-HALF/SECOND-HALF CONTRAST IS BARRED AND IS NOT COMPUTED BY
THIS FILE AT ALL. It is the statistic that made P1 mechanism C look like a 4.4x
win when it was regression to the mean. See `BARRED_STATISTICS`.

Verdict states (resolution order documented in `resolve_verdict`):
  VOID        the arm did not run, or the evidence cannot be read. No verdict is
              recorded in EITHER direction; rebuild or abandon.
  KILL        a sealed kill rule that indicates real harm fired (1, 3, 5).
  NO-PROMOTE  the arm ran and was readable, and the primary behavioural test
              did not clear the control-spread minimum (kill rule 2), or a
              reading canary failed.
  PROMOTE     delivery clears, no kill rule fires, every canary passes, and
              B1 < 0.39862542955326463.

Usage:
  .venv/Scripts/python.exe duck_eval/warpack/effnote_score.py \
      --run runs/kernel_pulls/effnote_v1 \
      --json runs/effnote/score_2026-08-13.json \
      --md   runs/effnote/score_2026-08-13.md
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "_kaggle_dataset"))

import effnote_replay as RP          # noqa: E402 - THE control-spread code
import effnote_patch as EN           # noqa: E402 - THE shipped module
from p1_score import load_log_text   # noqa: E402 - proven json-array log loader

rhae_score = __import__("p1_score").rhae_score

CONTROL_SPREAD = REPO / "runs" / "effnote_replay" / "control_spread.json"

# --------------------------------------------------------------------------- #
# SEALED CONSTANTS -- every one of these is copied from the prereg, which was
# sealed 2026-08-13 BEFORE the eval kernel was pushed. `verify_seal()` re-reads
# the control spread from disk and refuses to score if any of them has drifted.
# --------------------------------------------------------------------------- #
B1_PASS_LINE = 0.39862542955326463   # prereg sec3 M0 part 2 -- PASS is STRICT <
LC_KILL_FLOOR = 14                   # prereg sec5.1 -- lc < 14 => KILL
NAG_RATE = 0.40                      # prereg sec5.3 -- any detector > 40% => KILL
NOTE_RATE_FLOOR = 0.80               # prereg sec3 M0 part 1 / sec5.4
NOTE_GAMES_FLOOR = 20                # prereg sec3 M0 part 1: >=20 of 25 games
CHAR_BOUND = 700                     # prereg sec5.5 -- CHARACTERS, never tokens
K_E1_FLOORS = {"net_zero": 3, "revisit": 3, "stagnation": 1}  # sec1.1 item 1
N_GAMES_EXPECTED = 25

# Descriptive-only control ranges (prereg sec1 table). Reported, never gating.
CONTROL_RANGES = {
    "D1_note_rate": (0.9608826479438315, 0.9672),
    "D2_stall_rate": (0.0712, 0.0995850622406639),
    "D3_over_rate": (0.1968, 0.2921161825726141),
    "D4_chars_mean": (273.0, 297.5399828030954),
    "D4_chars_max": (602, 603),
    "B1_post_stall_revisit_rate": (0.39862542955326463, 0.5487465181058496),
    "B1c_nonstall_revisit_rate": (0.11056568489948575, 0.1991701244813278),
    "B2_post_stall_noop_rate": (0.07789232531500573, 0.34540389972144847),
    "B3_over_target_burn_total": (1267, 2301),
    "B3_over_target_burn_cleared": (32, 307),
    "B4_stall_turn_size": (3.9450549450549453, 7.275),
    "M0_median_actions_per_cleared_level": (24, 49),
    "levels_cleared": (14, 17),
}

BARRED_STATISTICS = [
    {"statistic": "the arm's own first-half -> second-half contrast on B1 (or on "
                  "any behavioural metric)",
     "why": "prereg sec3 M0 part 2: 'The arm's within-run first-half->second-half "
            "contrast MAY NOT BE CITED.' It is the statistic that made P1 "
            "mechanism C look like a 4.4x win when it was regression to the mean.",
     "status": "NOT COMPUTED BY THIS SCORER AT ALL"},
    {"statistic": "Delta levels_completed vs the family, in the GAIN direction",
     "why": "prereg sec3 power statement: family duck-harness-kaggle-continuation-v1 "
            "is m = 2; SCREEN_PROTOCOL sec1 P2 => NOT SCREENABLE. lc is read ONLY "
            "in the non-harm direction of kill rule 1 (lc < 14 => KILL).",
     "status": "computed for kill rule 1 ONLY; may never be cited as a win"},
    {"statistic": "B1 against the control MEAN (0.4741) or against any SINGLE "
                  "control run",
     "why": "prereg sec3/sec5.2: the line is the MINIMUM of the spread, 0.3986. "
            "Reading against the mean is a post-hoc threshold move.",
     "status": "the mean is reported for context; the GATE uses the minimum only"},
    {"statistic": "any token-fraction / token-cost reading of the note",
     "why": "prereg sec2.1 divergence 2 + sec5.5: the note is an INPUT cost and "
            "the rail reports GENERATED tokens. That denominator mismatch fired "
            "K-A3 and killed the animation arm. The bound is 700 CHARACTERS.",
     "status": "NOT COMPUTED; K-E3 asserts no token metric exists"},
    {"statistic": "a B3 (over-target burn) win carrying the verdict",
     "why": "prereg sec3 M0 part 2: B3 is 'supporting only'; its spread is 9.6x "
            "wide on three draws. 'a B3 win with B1 inside the spread is NOT a "
            "pass.'",
     "status": "reported as supporting evidence only"},
    {"statistic": "RHAE / local-25 score and the M0 median-actions-per-cleared-level",
     "why": "prereg sec3 M2: DESCRIPTIVE ONLY, one draw against an m = 2 family. "
            "'Attributing either to EFFNOTE would be exactly the error this "
            "prereg exists to prevent.'",
     "status": "reported, never gating"},
    {"statistic": "K-E1 (detector sanity) as evidence of efficacy",
     "why": "prereg sec1.1 item 1: K-E1 was re-pre-registered pre-data at "
            "stagnation >= 1 game (from >= 3), with the incentive on the record. "
            "It is a DETECTOR-SANITY canary and CANNOT RESCUE A DEAD ARM.",
     "status": "reported as a canary; never contributes to a PASS"},
]


# --------------------------------------------------------------------------- #
# log loading + emitter-shaped parsers
# Byte-shapes taken from `_kaggle_dataset/effnote_patch.py`:
#   canary_report():
#     EFFNOTE CANARY v=1 version=v1 games=N turns=N noted=N note_rate=0.0000
#     chars_mean=0.0 chars_max=N bound=N over_target=N over_rate=0.0000
#     stall_turns=N stall_rate=0.0000 nz=N/Ng stag=N/Ng rev=N/Ng errors=N
#     target=proxy-only
#   _emit(kind, st, detail):
#     EFFNOTE v=1 kind=K game=G turns=N noted=N over=N nz=N stag=N rev=N
#     chars_max=N errors=N <detail>
#   apply() banner:      effnote v1: ACTIVE (2 seams patched) - ...
#   notebook cell 12:    effnote v1: graft applied from <dir> (applied=True); NO ...
#                        effnote: PATCH FAILED - continuing with VANILLA duck harness
#   notebook cell 14:    EFFNOTE CANARY unavailable: <repr>
# --------------------------------------------------------------------------- #
CANARY_RE = re.compile(
    r"EFFNOTE CANARY v=(?P<v>\S+) version=(?P<version>\S+) games=(?P<games>\d+) "
    r"turns=(?P<turns>\d+) noted=(?P<noted>\d+) note_rate=(?P<note_rate>[0-9.eE+-]+) "
    r"chars_mean=(?P<chars_mean>[0-9.eE+-]+) chars_max=(?P<chars_max>\d+) "
    r"bound=(?P<bound>\d+) over_target=(?P<over_target>\d+) "
    r"over_rate=(?P<over_rate>[0-9.eE+-]+) stall_turns=(?P<stall_turns>\d+) "
    r"stall_rate=(?P<stall_rate>[0-9.eE+-]+) "
    r"nz=(?P<nz>\d+)/(?P<nz_games>\d+)g stag=(?P<stag>\d+)/(?P<stag_games>\d+)g "
    r"rev=(?P<rev>\d+)/(?P<rev_games>\d+)g errors=(?P<errors>\d+) "
    r"target=(?P<target>\S+)")
CANARY_LOOSE_RE = re.compile(r"EFFNOTE CANARY (?!unavailable)(?P<body>.+)")
CANARY_UNAVAILABLE_TOKEN = "EFFNOTE CANARY unavailable"
EVENT_RE = re.compile(
    r"EFFNOTE v=(?P<v>\S+) kind=(?P<kind>\S+) game=(?P<game>\S+) "
    r"turns=(?P<turns>\d+) noted=(?P<noted>\d+) over=(?P<over>\d+) "
    r"nz=(?P<nz>\d+) stag=(?P<stag>\d+) rev=(?P<rev>\d+) "
    # `detail` stops at the next `EFFNOTE v=` marker as well as at a newline:
    # Kaggle log records do not all end in a newline, so a following event line
    # can be MERGED onto this one. A plain `[^\n]*` detail swallows it and
    # finditer never sees it -- that silently lost one of the 25 `kind=game_end`
    # lines on the real pull, which is a delivery-coverage number.
    r"chars_max=(?P<chars_max>\d+) errors=(?P<errors>\d+)"
    r"(?P<detail>(?:(?!EFFNOTE v=)[^\n])*)")

# Every canary field the scorer READS. Cross-checked field-by-field against the
# emitter's own print() in `effnote_score_selftest.py` (group X) -- the animation
# arm died because the scorer read a field the builder never wrote.
CANARY_REQUIRED = ("v", "version", "games", "turns", "noted", "note_rate",
                   "chars_mean", "chars_max", "bound", "over_target", "over_rate",
                   "stall_turns", "stall_rate", "nz", "stag", "rev", "errors",
                   "target")

BANNER_TOKEN = "effnote v1: ACTIVE"
SEAMS_TOKEN = "ACTIVE (2 seams patched)"
REPORT_ONLY_TOKEN = "REPORT-ONLY"
GRAFT_RE = re.compile(r"effnote (?P<version>\S+): graft applied from (?P<dir>\S+) "
                      r"\(applied=(?P<applied>\w+)\)[^\n]*")
PATCH_FAILED_TOKEN = "effnote: PATCH FAILED"

# K-E5: the arm must be ALONE. These are the OTHER grafts' own banners, taken
# from their shipped modules. None of them matches the effnote banner's own
# "NO warpack/ledger-graft/sentinel/compaction/animation/p1" disclaimer text.
FOREIGN_BANNERS = {
    "warpack": re.compile(r"warpack \S+ applied:|warpack banking:"),
    "ledger": re.compile(r"ledger \S+: (patches applied|store keying)"),
    "sentinel": re.compile(r"budget sentinel ACTIVE"),
    "compaction": re.compile(r"compaction \S+: ACTIVE"),
    "animation": re.compile(r"animation \S+: ACTIVE|ANIMATION CANARY v="),
    "p1": re.compile(r"p1 \S+: ACTIVE|P1 CANARY v="),
}
# K-E3's other leg: NO TOKEN METRIC "in the module or the canary" (prereg
# sec2.1 divergence 2, asserted pre-seal by smoke L6 / I6b). TWO deliberate
# scoping decisions, both learned from instruments that fired falsely:
#
#  * it must match a token-valued FIELD (`token_fraction=0.031`), NOT the word
#    "token" in prose -- the SHIPPED banner itself reads "cost bound = 700
#    CHARACTERS (never a token fraction)", and a naive substring grep flags the
#    arm's own disclaimer as a violation and KILLS a healthy arm;
#  * it is scoped to the SHIPPED MODULE SOURCE and the CANARY LINE, which is
#    what the prereg says, NOT to the whole log. Kaggle log records do not all
#    end in a newline, so harness prints merge into EFFNOTE "lines"; a
#    whole-log line-scoped grep matched `tokens=68697` from an unrelated
#    `[finished]` line and produced a FALSE KILL on the first run of this
#    scorer. Caught by the self-test, fixed before the verdict was written.
TOKEN_FIELD_RE = re.compile(r"\b[A-Za-z_]*token[A-Za-z_]*\s*=")
MODULE_SRC = (HERE / "_kaggle_dataset" / "effnote_patch.py").read_text(
    encoding="utf-8")
CANARY_TAIL_CHARS = 80


def find_log(run: Path) -> Path | None:
    cands = [p for p in sorted(run.glob("*.log")) + sorted(run.glob("**/*.log"))
             if "vllm" not in p.name.lower()]
    if not cands:
        return None
    named = [p for p in cands if "effnote" in p.name.lower()]
    return (named or cands)[0]


def read_log(run: Path) -> tuple[str, dict]:
    p = find_log(run)
    if p is None:
        return "", {"path": None, "source": "MISSING",
                    "error": f"no non-vllm *.log under {run}"}
    txt, fmt = load_log_text(p)
    meta = {"path": str(p), "source": fmt, "chars": len(txt)}
    if not txt.strip():
        meta["error"] = f"{p} decoded to an empty blob"
    return txt, meta


def line_with(log: str, token: str) -> str | None:
    i = log.find(token)
    if i < 0:
        return None
    s = log.rfind("\n", 0, i) + 1
    e = log.find("\n", i)
    return log[s:(e if e > 0 else min(len(log), i + 400))].strip()[:600] or None


def parse_canary(log: str) -> tuple[dict | None, str | None]:
    """Strict parse of the single `EFFNOTE CANARY` line.

    A malformed / partial / absent line is an ERROR. It is NEVER returned as an
    empty dict that later reads as errors=0, note_rate=0 or chars_max=0."""
    if not log:
        return None, "no log text"
    if CANARY_UNAVAILABLE_TOKEN in log:
        return None, ("the builder printed "
                      f"{line_with(log, CANARY_UNAVAILABLE_TOKEN)!r} -- "
                      "canary_report() raised, so there is no canary line")
    m = None
    for m in CANARY_RE.finditer(log):
        pass
    if m is None:
        loose = CANARY_LOOSE_RE.search(log)
        if loose:
            body = loose.group("body")
            present = {kv.split("=", 1)[0] for kv in body.split() if "=" in kv}
            missing = [k for k in CANARY_REQUIRED if k not in present]
            return None, (f"EFFNOTE CANARY line present but MALFORMED; "
                          f"missing/renamed field(s) {missing}; "
                          f"raw={body.strip()[:300]!r}")
        return None, "no `EFFNOTE CANARY` line in the log"
    d = m.groupdict()
    out = {k: d[k] for k in ("v", "version", "target")}
    for k in ("games", "turns", "noted", "chars_max", "bound", "over_target",
              "stall_turns", "nz", "nz_games", "stag", "stag_games", "rev",
              "rev_games", "errors"):
        out[k] = int(d[k])
    for k in ("note_rate", "chars_mean", "over_rate", "stall_rate"):
        out[k] = float(d[k])
    out["raw"] = m.group(0)
    # A bounded tail so K-E3 can see a token field appended to the canary line
    # without reading unrelated harness output (see TOKEN_FIELD_RE).
    out["raw_tail"] = log[m.start():m.end() + CANARY_TAIL_CHARS].split("\n")[0]
    return out, None


def parse_events(log: str) -> list[dict]:
    evs = []
    for m in EVENT_RE.finditer(log):
        d = m.groupdict()
        evs.append(dict(
            v=d["v"], kind=d["kind"], game=d["game"],
            game_short=d["game"].split("-")[0],
            turns=int(d["turns"]), noted=int(d["noted"]), over=int(d["over"]),
            nz=int(d["nz"]), stag=int(d["stag"]), rev=int(d["rev"]),
            chars_max=int(d["chars_max"]), errors=int(d["errors"]),
            detail=d["detail"].strip(), raw=m.group(0),
            # The emitter-formatted prefix (through `errors=N`). `detail` is
            # free text and, in a Kaggle log, can carry an unrelated print that
            # merged onto the end of the line -- so field-level checks read the
            # STRUCTURED prefix only.
            structured=m.group(0)[:m.start("detail") - m.start(0)]))
    return evs


def bench_rows(run: Path) -> tuple[list[dict], str | None]:
    p = run / "benchmark.json"
    if not p.is_file():
        return [], f"benchmark.json absent under {run}"
    try:
        b = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return [], f"benchmark.json unreadable: {exc!r}"
    rows = b if isinstance(b, list) else b.get("game_runs")
    if not rows:
        return [], "benchmark.json carries no game_runs"
    need = ("game_id", "base_actions_per_level", "actions_per_level",
            "levels_completed", "number_of_levels")
    missing = sorted({k for r in rows for k in need if k not in r})
    if missing:
        return [], f"benchmark rows missing field(s) {missing}"
    return rows, None


def bench_label(run: Path) -> str | None:
    p = run / "benchmark.json"
    if not p.is_file():
        return None
    try:
        b = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    return b.get("label") if isinstance(b, dict) else None


# --------------------------------------------------------------------------- #
# the seal check + the replay (the SAME code that produced the control spread)
# --------------------------------------------------------------------------- #
def verify_seal(path: Path = CONTROL_SPREAD) -> tuple[dict | None, str | None]:
    """Re-read the pre-computed control spread and refuse to score if the sealed
    B1 line has drifted. A threshold that can move after the data lands is not a
    threshold."""
    if not path.is_file():
        return None, f"control spread absent at {path} -- the sealed B1 line is unverifiable"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return None, f"control spread unreadable: {exc!r}"
    spread = payload.get("control_spread") or {}
    b1 = spread.get("B1_post_stall_revisit_rate")
    if not b1 or "min" not in b1:
        return None, "control spread carries no B1_post_stall_revisit_rate.min"
    if b1["min"] != B1_PASS_LINE:
        return None, (f"SEALED B1 LINE DRIFT: control_spread.json says "
                      f"{b1['min']!r}, this scorer is sealed at {B1_PASS_LINE!r}")
    if len(b1.get("values") or []) < 3:
        return None, ("control spread has fewer than 3 control runs -- the "
                      "'minimum of the spread' line is not the sealed one")
    return payload, None


def replay_arm(run: Path) -> tuple[dict | None, str | None]:
    """Replay the arm's own traces with `effnote_replay.replay_run` -- the
    IDENTICAL code, on the IDENTICAL definitions, that produced the control
    spread before this arm existed. That symmetry is the whole point.

    Never returns a metric derived from an empty replay: 0 games or 0 stall
    actions means B1 is UNDEFINED, not 0.0."""
    rel: str | Path = run
    try:
        rel = run.relative_to(REPO)
    except ValueError:
        pass
    if not (run / "benchmark.json").is_file():
        return None, f"no benchmark.json under {run} -- nothing to replay"
    art = run / "artifacts"
    if not art.is_dir() or not any(art.glob("*_events.jsonl")):
        return None, (f"no artifacts/*_events.jsonl under {run} -- the arm's "
                      "behaviour CANNOT be replayed; B1 is UNDEFINED (NOT 0.0)")
    try:
        res = RP.replay_run(str(rel))
    except Exception as exc:  # noqa: BLE001
        return None, f"replay raised: {exc!r}"
    if not res.get("games"):
        return None, "replay produced 0 games -- B1 is UNDEFINED (NOT 0.0)"
    if not res.get("stall_actions"):
        return None, ("replay produced 0 post-stall actions -- B1 has an empty "
                      "denominator and is UNDEFINED (NOT 0.0)")
    return res, None


def instrument_agreement(canary: dict | None, replay: dict | None,
                         rows: list[dict]) -> dict:
    """Cross-check the OFFLINE replay (which supplies B1) against the LIVE
    in-kernel module (which supplies the canary) and against the benchmark.

    This is the check that says the instrument measuring the primary endpoint
    saw the same run the harness did. The per-turn denominators legitimately
    differ -- the live module counts every `_build_user_prompt` call, including
    turns that issued no action, while the replay's TURN is 'an analysis step
    carrying >=1 action' (prereg sec1 definition, identical for control and arm)
    -- so turn-level counts are REPORTED, not asserted."""
    out: dict = {"checks": [], "ok": True, "errors": []}

    def add(name, ok, detail, hard=True):
        out["checks"].append({"check": name, "ok": bool(ok), "detail": detail,
                              "hard": hard})
        if hard and not ok:
            out["ok"] = False
            out["errors"].append(f"{name}: {detail}")

    bench_actions = (sum(sum(r["actions_per_level"]) for r in rows) if rows else None)
    if replay is not None and bench_actions is not None:
        add("replay reproduces the benchmark action total exactly (smoke R2)",
            replay["actions"] == bench_actions,
            f"replay={replay['actions']} benchmark={bench_actions}")
    if replay is not None and rows:
        add("replay covers every benchmark game",
            replay["games"] == len(rows),
            f"replayed={replay['games']} benchmark_games={len(rows)}")
    if replay is not None and canary is not None:
        # These two are CORROBORATION, not invariants: the live module sees the
        # real frames and the replay sees board digests off the recorded trace,
        # so they are expected to agree and their agreement is strong evidence
        # that the instrument measuring the PRIMARY saw the same run -- but a
        # benign divergence must NOT void a healthy arm (that is the false-kill
        # failure mode this campaign has already paid for). Reported, soft.
        dg = replay["detector_games"]
        add("per-detector DISTINCT-GAME counts, live module vs offline replay "
            "(corroboration)",
            (dg["net_zero"] == canary["nz_games"]
             and dg["stagnation"] == canary["stag_games"]
             and dg["revisit"] == canary["rev_games"]),
            f"replay nz/stag/rev games = {dg['net_zero']}/{dg['stagnation']}/"
            f"{dg['revisit']}; live canary = {canary['nz_games']}/"
            f"{canary['stag_games']}/{canary['rev_games']}", hard=False)
        add("chars_max, live module vs offline replay (corroboration)",
            replay["D4_chars_max"] == canary["chars_max"],
            f"replay={replay['D4_chars_max']} live={canary['chars_max']}",
            hard=False)
        add("turn denominators (REPORTED, not asserted -- see docstring)", True,
            f"live turns={canary['turns']} (every _build_user_prompt call) vs "
            f"replay turns={replay['turns']} (analysis steps carrying >=1 action)",
            hard=False)
    if canary is not None and rows:
        add("canary game count equals the benchmark game count",
            canary["games"] == len(rows),
            f"canary={canary['games']} benchmark={len(rows)}")
    return out


# --------------------------------------------------------------------------- #
# canaries (prereg sec4). Each carries the CONSEQUENCE of its own failure.
# --------------------------------------------------------------------------- #
CANARY_ORDER = ("K-E0", "K-E0b", "K-E1", "K-E1'", "K-E2", "K-E3", "K-E4", "K-E5")


def run_canaries(log: str, log_meta: dict, canary: dict | None,
                 canary_err: str | None, events: list[dict], rows: list[dict],
                 rows_err: str | None, replay: dict | None) -> dict:
    """Every canary returns PASS / FAIL / ERROR. ERROR == the evidence is
    missing or unreadable; it is never a PASS and never a silent 0.

    `on_fail` is the SEALED consequence, using the prereg's own mapping:
      VOID       the arm never ran / is not the sealed arm / cannot be read.
                 No verdict is recorded in either direction (prereg sec5.4
                 'INFRA DEATH, re-run or abandon; the behavioural numbers may
                 not be read at all').
      KILL       real harm (prereg kill rules 1, 3, 5).
      NO-PROMOTE the arm ran but a reading condition did not hold.
    """
    out: dict = {}
    log_missing = log_meta.get("source") == "MISSING" or bool(log_meta.get("error"))
    ngames = len(rows) if rows else (canary["games"] if canary else 0)

    # ---- K-E0: graft installed, 2 seams, report-only banner ----------------
    banner = BANNER_TOKEN in log
    seams = SEAMS_TOKEN in log
    ronly = REPORT_ONLY_TOKEN in log
    gm = GRAFT_RE.search(log)
    applied = bool(gm) and gm.group("applied") == "True"
    st = "ERROR" if log_missing else (
        "PASS" if (banner and seams and ronly and applied) else "FAIL")
    out["K-E0"] = dict(
        name="graft installed: `effnote v1: ACTIVE (2 seams patched)`, REPORT-ONLY "
             "banner, `graft applied ... (applied=True)`",
        status=st, on_fail="VOID",
        banner_present=banner, two_seams=seams, report_only=ronly,
        graft_applied_true=applied,
        consequence="the arm never installed -> the run is not a mechanism "
                    "result in either direction (VOID, rebuild)",
        error=log_meta.get("error"),
        evidence=[x for x in (line_with(log, BANNER_TOKEN),
                              gm.group(0) if gm else None) if x])

    # ---- K-E0b: DELIVERY ---------------------------------------------------
    # Leg 1 as written ("note delivered on >=80% of STALL-OR-OVER-TARGET turns")
    # is STRUCTURALLY DEGENERATE: build_efficiency_note() returns "" only when
    # used <= 0 AND no detector fired, so on a stall-or-over turn the note is
    # non-empty by construction and the rate is 1.0 unless the seam raised
    # (which K-E4 already covers). A check that cannot fail is a defect -- this
    # is the exact bug that shipped twice. So the number the gate READS is the
    # OVERALL note_rate from the canary line, which CAN fall below 0.80 and is
    # the CONSERVATIVE reading (overall <= stall-or-over-target).
    note_rate = canary["note_rate"] if canary else None
    games_with_note_events = len({e["game"] for e in events if e["kind"] == "note"})
    games_delivered = len({e["game"] for e in events
                           if e["kind"] == "game_end" and e["noted"] > 0})
    leg1 = (note_rate is not None and note_rate >= NOTE_RATE_FLOOR)
    leg2 = games_with_note_events >= NOTE_GAMES_FLOOR
    if canary is None:
        st = "ERROR"
    else:
        st = "PASS" if (leg1 and leg2) else "FAIL"
    out["K-E0b"] = dict(
        name=f"note delivered: note_rate >= {NOTE_RATE_FLOOR} and >=1 "
             f"`EFFNOTE v=1 kind=note` line on >= {NOTE_GAMES_FLOOR} of "
             f"{N_GAMES_EXPECTED} games",
        status=st, on_fail="NO-PROMOTE",
        leg1_note_rate=note_rate, leg1_floor=NOTE_RATE_FLOOR, leg1_pass=leg1,
        leg2_games_with_note_event=games_with_note_events,
        leg2_floor=NOTE_GAMES_FLOOR, leg2_pass=leg2,
        games_with_note_delivered=games_delivered,
        degeneracy_disclosure=(
            "the prereg's literal leg-1 denominator (stall-or-over-target turns) "
            "is 1.0 BY CONSTRUCTION and CANNOT FAIL; the OVERALL note_rate is "
            "read instead, which is strictly the harsher reading"),
        leg2_note=(
            "`kind=note` is emitted only on a stall-or-over-target turn, so leg 2 "
            "counts games that produced an EVENT LINE, not games that received "
            "the note. Games where the note was actually delivered (noted > 0 on "
            "the kind=game_end line) is reported alongside and is the reason a "
            "leg-2 miss is NOT read as 'the arm never ran'."),
        consequence="delivery is NECESSARY AND NOT SUFFICIENT (prereg sec3: P1 "
                    "mechanism C delivered on 96.3% of turns and was dead). A "
                    "leg-1 failure is INFRA DEATH (VOID) via kill rule 4; a "
                    "leg-2-only failure blocks PROMOTE (resolved AGAINST the arm) "
                    "but is not VOID while the log shows the note reaching the "
                    "model on the games it ran",
        error=canary_err,
        evidence=[canary["raw"]] if canary else [])
    if canary is not None and not leg1:
        out["K-E0b"]["on_fail"] = "VOID"   # kill rule 4: INFRA DEATH

    # ---- K-E1: detector sanity (re-pre-registered, sec1.1 item 1) ----------
    if canary is None:
        st = "ERROR"
        detg = None
    else:
        detg = {"net_zero": canary["nz_games"], "stagnation": canary["stag_games"],
                "revisit": canary["rev_games"]}
        st = "PASS" if all(detg[k] >= v for k, v in K_E1_FLOORS.items()) else "FAIL"
    out["K-E1"] = dict(
        name="detector sanity: net-zero >= 3 games, revisit >= 3 games, "
             "stagnation >= 1 game (RE-PRE-REGISTERED pre-data, sec1.1 item 1)",
        status=st, on_fail="NO-PROMOTE",
        floors=K_E1_FLOORS, detector_games=detg,
        would_fail_at_original_floor=(
            None if detg is None else detg["stagnation"] < 3),
        disclosure=(
            "the original harness_diff draft asked for stagnation on >= 3 games; "
            "it was relaxed to >= 1 BEFORE the data, with the incentive on the "
            "record, because the three block-free controls fire stagnation on "
            "1-2 games each -- a property of the rail, not of the arm"),
        consequence="DETECTOR-SANITY ONLY. It cannot rescue a dead arm and never "
                    "contributes to a PASS (prereg sec1.1 item 1).",
        error=canary_err,
        evidence=[canary["raw"]] if canary else [])

    # ---- K-E1': nagging -----------------------------------------------------
    rates = detector_rates(canary, replay)
    worst = max((v for v in rates["max_by_detector"].values()), default=None)
    if worst is None:
        st = "ERROR"
    else:
        st = "PASS" if worst <= NAG_RATE else "FAIL"
    out["K-E1'"] = dict(
        name=f"no detector fires on > {NAG_RATE:.0%} of turns (nagging => ignored)",
        status=st, on_fail="KILL",
        worst_detector_rate=worst, threshold=NAG_RATE, rates=rates,
        consequence="prereg kill rule 3: the note is noise and the agent will "
                    "learn to skip it => KILL",
        error=canary_err,
        evidence=[canary["raw"]] if canary else [])

    # ---- K-E2: non-harm on levels_completed --------------------------------
    lc = sum(r["levels_completed"] for r in rows) if rows else None
    st = "ERROR" if lc is None else ("PASS" if lc >= LC_KILL_FLOOR else "FAIL")
    out["K-E2"] = dict(
        name=f"levels_completed >= {LC_KILL_FLOOR} (the minimum of the three "
             f"block-free controls)",
        status=st, on_fail="KILL",
        levels_completed=lc, n_games=(len(rows) if rows else None),
        control_range=list(CONTROL_RANGES["levels_cleared"]),
        consequence="prereg kill rule 1: any trade of levels for efficiency kills "
                    "this arm outright",
        barred="lc is read ONLY in this non-harm direction. Delta-lc is NOT "
               "SCREENABLE (family m = 2) and may NEVER be cited as a win.",
        error=rows_err,
        evidence=[f"benchmark.json: sum(levels_completed) = {lc} over "
                  f"{len(rows) if rows else 0} games"] if lc is not None else [])

    # ---- K-E3: the CHARACTER cost bound ------------------------------------
    chars_max = canary["chars_max"] if canary else None
    bound = canary["bound"] if canary else None
    module_token_fields = [m.group(0) for m in TOKEN_FIELD_RE.finditer(MODULE_SRC)]
    canary_token_fields = ([m.group(0) for m in
                            TOKEN_FIELD_RE.finditer(canary["raw_tail"])]
                           if canary else [])
    event_token_fields = sorted({m.group(0) for e in events
                                 for m in TOKEN_FIELD_RE.finditer(e["structured"])})
    token_hits = module_token_fields + canary_token_fields + event_token_fields
    if canary is None:
        st = "ERROR"
    else:
        st = "PASS" if (chars_max <= CHAR_BOUND and bound == CHAR_BOUND
                        and not token_hits) else "FAIL"
    out["K-E3"] = dict(
        name=f"chars_max <= {CHAR_BOUND} (a CHARACTER bound) and NO token metric "
             f"anywhere in the EFFNOTE surface",
        status=st, on_fail="KILL",
        chars_max=chars_max, bound_reported=bound, bound_sealed=CHAR_BOUND,
        token_metric_hits=token_hits,
        token_fields_in_module=module_token_fields,
        token_fields_in_canary=canary_token_fields,
        token_fields_in_events=event_token_fields,
        token_check_note=(
            "scoped to the SHIPPED MODULE SOURCE and the EFFNOTE CANARY/event "
            "lines (prereg sec2.1: 'no token metric of any kind exists in the "
            "module or the canary'), and matching a token-VALUED FIELD, not the "
            "word 'token' in prose. A whole-log line-scoped grep produced a "
            "FALSE KILL here (Kaggle log records do not all end in a newline, so "
            "an unrelated `[finished] ... tokens=68697` merged into an EFFNOTE "
            "line); the self-test carries that exact regression."),
        consequence="prereg kill rule 5: the static bound leaked => KILL and fix "
                    "the clamp. A TOKEN-FRACTION reading is FORBIDDEN (the rail's "
                    "denominator is generated tokens; the note is an input cost).",
        error=canary_err,
        evidence=[canary["raw"]] if canary else [])

    # ---- K-E4: errors / PATCH FAILED / traceback ---------------------------
    errs = canary["errors"] if canary else None
    patch_failed = PATCH_FAILED_TOKEN in log
    tb = "Traceback (most recent call last)" in log
    if canary is None or log_missing:
        st = "ERROR"
    else:
        st = "PASS" if (errs == 0 and not patch_failed and not tb) else "FAIL"
    out["K-E4"] = dict(
        name="errors = 0, no `effnote: PATCH FAILED`, no traceback",
        status=st, on_fail="VOID",
        errors=errs, patch_failed_line=patch_failed, traceback_present=tb,
        consequence="prereg kill rule 4: INFRA DEATH -- re-run or abandon; the "
                    "behavioural numbers may not be read at all",
        error=canary_err or log_meta.get("error"),
        evidence=[x for x in (canary["raw"] if canary else None,
                              line_with(log, PATCH_FAILED_TOKEN)) if x])

    # ---- K-E5: the graft is ALONE ------------------------------------------
    foreign_present = {}
    for k, rx in FOREIGN_BANNERS.items():
        fm = rx.search(log)
        if fm:
            foreign_present[k] = fm.group(0)
    st = "ERROR" if log_missing else ("PASS" if not foreign_present else "FAIL")
    out["K-E5"] = dict(
        name="the graft is ALONE: no warpack / ledger / sentinel / compaction / "
             "animation / p1 banner in the log",
        status=st, on_fail="VOID",
        foreign_banners=foreign_present,
        checked=sorted(FOREIGN_BANNERS),
        benchmark_label=None,   # filled in by score() from benchmark.json
        consequence="another graft in the log means this is not the sealed arm; "
                    "no verdict may be recorded in either direction",
        error=log_meta.get("error"),
        evidence=[gm.group(0)] if gm else [])
    return out


def detector_rates(canary: dict | None, replay: dict | None) -> dict:
    """Per-detector turn rates from BOTH instruments; the gate reads the MAX
    (i.e. resolves against the arm)."""
    live = {}
    if canary and canary["turns"]:
        t = canary["turns"]
        live = {"net_zero": canary["nz"] / t, "stagnation": canary["stag"] / t,
                "revisit": canary["rev"] / t, "any": canary["stall_rate"]}
    rep = {}
    if replay:
        rep = dict(replay["detector_turn_rate"])
        rep["any"] = replay["D2_stall_rate"]
    keys = sorted(set(live) | set(rep))
    mx = {k: max([x for x in (live.get(k), rep.get(k)) if x is not None],
                 default=None) for k in keys}
    return {"live_canary": live, "offline_replay": rep,
            "max_by_detector": {k: v for k, v in mx.items() if v is not None}}


# --------------------------------------------------------------------------- #
# the PRIMARY endpoint
# --------------------------------------------------------------------------- #
def primary(canary: dict | None, replay: dict | None, replay_err: str | None,
            canaries: dict) -> dict:
    """M0 = mechanism DELIVERY (part 1) + the behavioural test vs the CONTROL
    SPREAD (part 2). Both parts must clear; part 1 licenses nothing on its own."""
    delivery_status = canaries["K-E0b"]["status"]
    b1 = replay["B1_post_stall_revisit_rate"] if replay else None
    if b1 is None:
        b1_status = "ERROR"
    else:
        b1_status = "PASS" if b1 < B1_PASS_LINE else "FAIL"
    lo, hi = CONTROL_RANGES["B1_post_stall_revisit_rate"]
    return dict(
        part1_delivery=dict(
            status=delivery_status,
            note_rate=(canary["note_rate"] if canary else None),
            floor=NOTE_RATE_FLOOR,
            chars_max=(canary["chars_max"] if canary else None),
            errors=(canary["errors"] if canary else None),
            sufficiency="NECESSARY AND NOT SUFFICIENT -- P1 mechanism C "
                        "delivered on 96.3% of turns and was dead (prereg sec3)"),
        part2_B1=dict(
            status=b1_status,
            value=b1,
            pass_line=B1_PASS_LINE,
            rule="PASS requires B1 STRICTLY BELOW the MINIMUM of the control "
                 "spread (0.39862542955326463). Not the mean. Not any single "
                 "control. Not the arm's own first half.",
            control_spread=[lo, hi],
            control_values=[0.39862542955326463, 0.5487465181058496,
                            0.4750542299349241],
            control_mean=0.4741420591980128,
            inside_control_spread=(None if b1 is None else lo <= b1 <= hi),
            margin_vs_line=(None if b1 is None else b1 - B1_PASS_LINE),
            nonstall_counterpart=(replay["B1c_nonstall_revisit_rate"]
                                  if replay else None),
            error=replay_err),
        status=("ERROR" if "ERROR" in (delivery_status, b1_status) else
                ("PASS" if delivery_status == "PASS" and b1_status == "PASS"
                 else "FAIL")))


# --------------------------------------------------------------------------- #
# kill rules (prereg sec5) -- all five, each individually evaluable
# --------------------------------------------------------------------------- #
def kill_rules(canaries: dict, rows: list[dict], rows_err: str | None,
               canary: dict | None, prim: dict) -> list[dict]:
    R: list[dict] = []
    lc = sum(r["levels_completed"] for r in rows) if rows else None
    R.append(dict(
        rule=1, name=f"non-harm: levels_completed < {LC_KILL_FLOOR}",
        consequence="KILL",
        fired=(lc is not None and lc < LC_KILL_FLOOR),
        evaluable=(lc is not None),
        detail=(f"levels_completed = {lc} over {len(rows)} games "
                f"(control range 14-17)" if lc is not None
                else f"UNEVALUABLE: {rows_err}")))

    b1 = prim["part2_B1"]["value"]
    R.append(dict(
        rule=2, name=f"the behavioural test: B1 >= {B1_PASS_LINE}",
        consequence="NO-PROMOTE",
        fired=(b1 is not None and b1 >= B1_PASS_LINE),
        evaluable=(b1 is not None),
        detail=(f"B1 = {b1:.6f} vs the sealed line {B1_PASS_LINE:.6f} "
                f"(control spread minimum)" if b1 is not None
                else f"UNEVALUABLE: {prim['part2_B1'].get('error')}"),
        note="No re-reading against the control mean, against the arm's own "
             "first half, or against a subset chosen after seeing the data."))

    rates = canaries["K-E1'"]["rates"]["max_by_detector"]
    worst = canaries["K-E1'"]["worst_detector_rate"]
    R.append(dict(
        rule=3, name=f"nagging: any detector fires on > {NAG_RATE:.0%} of turns",
        consequence="KILL",
        fired=(worst is not None and worst > NAG_RATE),
        evaluable=(worst is not None),
        detail=(f"worst per-detector turn rate = {worst:.4f} "
                f"({ {k: round(v, 4) for k, v in rates.items()} })"
                if worst is not None else "UNEVALUABLE: no canary and no replay")))

    nr = canary["note_rate"] if canary else None
    errs = canary["errors"] if canary else None
    pf = canaries["K-E4"]["patch_failed_line"]
    fired4 = ((nr is not None and nr < NOTE_RATE_FLOOR)
              or (errs is not None and errs > 0) or bool(pf))
    R.append(dict(
        rule=4, name=f"delivery: note_rate < {NOTE_RATE_FLOOR}, or errors > 0, "
                     f"or any PATCH FAILED",
        consequence="VOID (INFRA DEATH -- re-run or abandon; the behavioural "
                    "numbers may not be read at all)",
        fired=fired4, evaluable=(canary is not None),
        detail=(f"note_rate = {nr}, errors = {errs}, PATCH FAILED = {pf}"
                if canary is not None else "UNEVALUABLE: no canary line")))

    cm = canary["chars_max"] if canary else None
    R.append(dict(
        rule=5, name=f"cost: chars_max > {CHAR_BOUND} CHARACTERS",
        consequence="KILL",
        fired=(cm is not None and cm > CHAR_BOUND),
        evaluable=(cm is not None),
        detail=(f"chars_max = {cm} vs bound {CHAR_BOUND} characters "
                f"(a TOKEN-FRACTION reading is forbidden)" if cm is not None
                else "UNEVALUABLE: no canary line")))
    return R


# --------------------------------------------------------------------------- #
# verdict
# --------------------------------------------------------------------------- #
def resolve_verdict(canaries: dict, rules: list[dict], prim: dict,
                    evidence_errors: list[str], instruments: dict) -> dict:
    """VOID > KILL > NO-PROMOTE > PROMOTE.

    VOID first because an arm that did not run, or a run that cannot be read, is
    NOT a mechanism result in either direction (prereg sec5.4). It is recorded
    as 'no verdict', never as a negative.
    KILL before NO-PROMOTE because the sealed kill rules are unconditional.
    PROMOTE requires: delivery, every canary PASS, no kill rule fired, and
    B1 STRICTLY below the control-spread minimum."""
    void: list[str] = list(evidence_errors)
    void += [f"instrument disagreement -- {e}" for e in instruments.get("errors", [])]
    kill: list[str] = []
    noprom: list[str] = []

    for k in CANARY_ORDER:
        c = canaries[k]
        if c["status"] == "PASS":
            continue
        tag = f"{k} {c['status']}: {c['name']}"
        if c["status"] == "ERROR":
            void.append(f"{tag} -- evidence missing/unreadable, so nothing was "
                        f"observed in either direction")
        elif c["on_fail"] == "VOID":
            void.append(tag)
        elif c["on_fail"] == "KILL":
            kill.append(tag)
        else:
            noprom.append(tag)

    for r in rules:
        if r["fired"]:
            (void if r["consequence"].startswith("VOID") else
             kill if r["consequence"] == "KILL" else noprom).append(
                f"kill rule {r['rule']} ({r['name']}) -- {r['detail']}")
        elif not r["evaluable"]:
            void.append(f"kill rule {r['rule']} UNEVALUABLE: {r['detail']}")

    if prim["part2_B1"]["status"] == "ERROR":
        void.append("PRIMARY B1 is UNDEFINED -- the arm's behaviour could not be "
                    "replayed. B1 is NEVER defaulted to 0.0.")

    if void:
        verdict, why = "VOID", "; ".join(dict.fromkeys(void))
    elif kill:
        verdict, why = "KILL", "; ".join(dict.fromkeys(kill))
    elif noprom:
        verdict, why = "NO-PROMOTE", "; ".join(dict.fromkeys(noprom))
    else:
        verdict, why = "PROMOTE", (
            "delivery clears, every canary passes, no kill rule fired, and "
            f"B1 = {prim['part2_B1']['value']:.6f} is STRICTLY BELOW the "
            f"control-spread minimum {B1_PASS_LINE:.6f}")
    return dict(verdict=verdict, why=why, void_reasons=void, kill_reasons=kill,
                no_promote_reasons=noprom,
                readable=(verdict != "VOID"),
                licenses=("a SECOND SEED of the same kernel, and only then a "
                          "promotion discussion (prereg sec6) -- nothing more"
                          if verdict == "PROMOTE" else
                          "nothing. A delivered mechanism is not an efficacy "
                          "claim."))


# --------------------------------------------------------------------------- #
# descriptive block (never gating)
# --------------------------------------------------------------------------- #
def descriptive(replay: dict | None, rows: list[dict]) -> dict:
    out: dict = {"note": "DESCRIPTIVE ONLY. None of this may be cited as "
                         "evidence of gain (prereg sec3 M1/M2, family m = 2)."}
    if replay:
        table = {}
        for k, (lo, hi) in CONTROL_RANGES.items():
            v = replay.get(k)
            if v is None:
                continue
            table[k] = {"arm": v, "control_min": lo, "control_max": hi,
                        "inside_spread": lo <= v <= hi,
                        "position": ("below" if v < lo else
                                     "above" if v > hi else "inside")}
        out["vs_control_spread"] = table
        out["B4_stall_turn_size"] = replay["B4_stall_turn_size"]
        out["B4_nonstall_turn_size"] = replay["B4_nonstall_turn_size"]
        out["M0_median_actions_per_cleared_level"] = replay[
            "M0_median_actions_per_cleared_level"]
    if rows:
        try:
            out["RHAE_local25"] = sum(
                rhae_score(r["base_actions_per_level"], r["actions_per_level"],
                           r["levels_completed"], r["number_of_levels"])
                for r in rows) / len(rows)
        except Exception as exc:  # noqa: BLE001
            out["RHAE_local25"] = None
            out["RHAE_error"] = repr(exc)
        out["levels_completed"] = sum(r["levels_completed"] for r in rows)
        out["actions_total"] = sum(sum(r["actions_per_level"]) for r in rows)
        out["per_game_levels"] = {r["game_id"].split("-")[0]: r["levels_completed"]
                                  for r in rows}
    return out


# --------------------------------------------------------------------------- #
def score(run: Path) -> dict:
    log, log_meta = read_log(run)
    canary, canary_err = parse_canary(log)
    events = parse_events(log)
    rows, rows_err = bench_rows(run)
    seal, seal_err = verify_seal()
    replay, replay_err = replay_arm(run)

    evidence_errors = [x for x in (
        log_meta.get("error") and f"LOG: {log_meta['error']}",
        canary_err and f"CANARY: {canary_err}",
        rows_err and f"BENCHMARK: {rows_err}",
        seal_err and f"SEAL: {seal_err}",
        replay_err and f"REPLAY: {replay_err}",
    ) if x]

    canaries = run_canaries(log, log_meta, canary, canary_err, events, rows,
                            rows_err, replay)
    canaries["K-E5"]["benchmark_label"] = bench_label(run)
    instruments = instrument_agreement(canary, replay, rows)
    prim = primary(canary, replay, replay_err, canaries)
    rules = kill_rules(canaries, rows, rows_err, canary, prim)
    verdict = resolve_verdict(canaries, rules, prim, evidence_errors, instruments)

    return dict(
        scorer="effnote_score.py",
        scored_at="2026-08-13",
        prereg="learnings/war_room/effnote_prereg_2026-08-13.md (SEALED 2026-08-13, "
               "before the eval kernel was pushed)",
        run=str(run), effnote_module_version=EN.VERSION,
        module_char_bound=EN.CFG.max_chars,
        log=log_meta, canary=canary, canary_error=canary_err,
        n_event_lines=len(events),
        event_kinds={k: sum(1 for e in events if e["kind"] == k)
                     for k in sorted({e["kind"] for e in events})},
        benchmark_error=rows_err, benchmark_label=bench_label(run),
        seal_error=seal_err,
        control_spread_source=str(CONTROL_SPREAD),
        replay=({k: v for k, v in replay.items() if k != "per_game"}
                if replay else None),
        replay_error=replay_err,
        instrument_agreement=instruments,
        canaries=canaries,
        PRIMARY=prim,
        kill_rules=rules,
        descriptive=descriptive(replay, rows),
        barred_statistics=BARRED_STATISTICS,
        verdict=verdict,
    )


# --------------------------------------------------------------------------- #
def render_markdown(rep: dict) -> str:
    v = rep["verdict"]
    p = rep["PRIMARY"]
    b1 = p["part2_B1"]
    L: list[str] = []
    A = L.append
    A(f"# EFFNOTE arm — VERDICT: **{v['verdict']}**")
    A("")
    A(f"*Scored {rep.get('scored_at', '2026-08-13')} by "
      f"`duck_eval/warpack/effnote_score.py` (self-tested by "
      f"`effnote_score_selftest.py`) against the SEALED prereg "
      f"`{rep['prereg']}`.*")
    A("")
    A(f"**Run:** `{rep['run']}` · label `{rep.get('benchmark_label')}` · "
      f"module `effnote {rep['effnote_module_version']}` · "
      f"log `{rep['log'].get('path')}` ({rep['log'].get('source')}, "
      f"{rep['log'].get('chars')} chars)")
    A("")
    A(f"> **{v['verdict']}** — {v['why']}")
    A("")
    A(f"> **This licenses:** {v['licenses']}")
    A("")
    A("---")
    A("")
    A("## 1. THE PRIMARY VERDICT — B1 vs the control spread")
    A("")
    A("**B1 = post-stall revisit rate.** Of the actions the agent issues on a "
      "turn whose note fired at least one stall detector, the fraction that "
      "land on a board state already visited on that level.")
    A("")
    if b1["value"] is None:
        A(f"**B1 is UNDEFINED** — {b1.get('error')}. It is NOT read as 0.0.")
    else:
        A(f"| | value |")
        A(f"|---|---|")
        A(f"| **arm B1** | **{b1['value']:.6f}** |")
        A(f"| **PASS line (control-spread MINIMUM, strict <)** | "
          f"**{b1['pass_line']:.6f}** |")
        A(f"| control spread | {b1['control_spread'][0]:.4f} – "
          f"{b1['control_spread'][1]:.4f} |")
        A(f"| control values | " +
          ", ".join(f"{x:.4f}" for x in b1["control_values"]) + " |")
        A(f"| margin vs the line | {b1['margin_vs_line']:+.6f} |")
        A(f"| inside the control spread? | "
          f"{'YES' if b1['inside_control_spread'] else 'no'} |")
        A(f"| arm's own non-stall counterpart (B1c) | "
          f"{b1['nonstall_counterpart']:.4f} |")
        A("")
        if b1["status"] == "PASS":
            A(f"**PASS.** {b1['value']:.4f} is strictly below "
              f"{b1['pass_line']:.4f}.")
        else:
            A(f"**FAIL, plainly stated: {b1['value']:.4f} ≥ "
              f"{b1['pass_line']:.4f}.** The arm did not beat the *minimum* of "
              f"the control spread — it sits "
              f"{'INSIDE' if b1['inside_control_spread'] else 'outside'} the "
              f"spread computed on three block-free runs BEFORE this arm "
              f"existed. Sealed kill rule 2 fires: **NO-PROMOTE**.")
    A("")
    A(f"*The rule, verbatim from the seal:* {b1['rule']}")
    A("")
    A("**Delivery (M0 part 1)** — "
      f"status **{p['part1_delivery']['status']}**, note_rate "
      f"{p['part1_delivery']['note_rate']}, chars_max "
      f"{p['part1_delivery']['chars_max']}, errors "
      f"{p['part1_delivery']['errors']}. {p['part1_delivery']['sufficiency']}.")
    A("")
    d0 = rep["canaries"]["K-E0b"]
    A(f"Delivery detail: overall note_rate **{d0['leg1_note_rate']}** "
      f"(floor {d0['leg1_floor']}) — leg 1 "
      f"{'PASS' if d0['leg1_pass'] else 'FAIL'}. Games carrying at least one "
      f"`kind=note` event line: **{d0['leg2_games_with_note_event']}** "
      f"(floor {d0['leg2_floor']}) — leg 2 "
      f"{'PASS' if d0['leg2_pass'] else 'FAIL'}. Games where the note was "
      f"actually delivered (`noted > 0` on the per-game `kind=game_end` line): "
      f"**{d0['games_with_note_delivered']}**. Where the prereg is ambiguous "
      "about which of these two the ≥20-games clause means, it is resolved "
      "**AGAINST the arm**: the clause is scored on the literal `kind=note` "
      "count and therefore FAILS. It is not read as VOID, because the log shows "
      "the note reaching the model on every game that ran, so the primary is "
      "genuinely readable.")
    A("")
    A("---")
    A("")
    A("## 2. CANARY TABLE")
    A("")
    A("| id | status | check | evidence line it was decided from | consequence of failure |")
    A("|---|---|---|---|---|")
    def esc(s: str) -> str:
        return str(s).replace("|", "\\|").replace("\n", " ")

    for k in CANARY_ORDER:
        c = rep["canaries"][k]
        ev = (c.get("evidence") or [None])[0]
        ev = esc(ev or c.get("error") or "-")
        if len(ev) > 260:
            ev = ev[:257] + "..."
        A(f"| **{k}** | **{c['status']}** | {esc(c['name'])} | "
          f"`{ev}` | {esc(c['consequence'])} |")
    A("")
    for k in CANARY_ORDER:
        c = rep["canaries"][k]
        extra = [c[f] for f in ("degeneracy_disclosure", "leg2_note", "disclosure",
                                "barred") if c.get(f)]
        if extra:
            A(f"- **{k}** — " + " / ".join(extra))
    A("")
    A("### Instrument agreement (does the thing measuring the primary see the "
      "same run the harness did?)")
    A("")
    for ch in rep["instrument_agreement"]["checks"]:
        A(f"- {'OK  ' if ch['ok'] else 'FAIL'} — {ch['check']}: {ch['detail']}"
          + ("" if ch["hard"] else "  *(reported, not asserted)*"))
    A("")
    A("---")
    A("")
    A("## 3. KILL RULES (sealed, prereg §5)")
    A("")
    A("| # | rule | state | consequence | detail |")
    A("|---|---|---|---|---|")
    for r in rep["kill_rules"]:
        state = "**FIRED**" if r["fired"] else ("ok" if r["evaluable"]
                                                else "UNEVALUABLE")
        A(f"| {r['rule']} | {r['name']} | {state} | {r['consequence']} | "
          f"{r['detail']} |")
    A("")
    A("---")
    A("")
    A("## 4. BARRED STATISTICS — explicitly listed, none of them touched the verdict")
    A("")
    for b in rep["barred_statistics"]:
        A(f"- **{b['statistic']}** — {b['status']}. *Why:* {b['why']}")
    A("")
    A("---")
    A("")
    A("## 5. DESCRIPTIVE ONLY (never gating)")
    A("")
    d = rep["descriptive"]
    if d.get("vs_control_spread"):
        A("| metric | arm | control min | control max | position |")
        A("|---|---|---|---|---|")
        for k, x in d["vs_control_spread"].items():
            fmt = (lambda z: f"{z:.4f}") if isinstance(x["arm"], float) else str
            A(f"| {k} | {fmt(x['arm'])} | {fmt(x['control_min'])} | "
              f"{fmt(x['control_max'])} | {x['position']} |")
        A("")
    A(f"- local-25 RHAE: **{d.get('RHAE_local25')}** · levels_completed "
      f"**{d.get('levels_completed')}** · actions **{d.get('actions_total')}**")
    A(f"- {d['note']}")
    A("")
    A("### 5.1 What this says about the MECHANISM (descriptive, non-inferential)")
    A("")
    A("Every one of these is a single draw against an m = 2 family and none of "
      "them may gate anything. They are recorded because the arm bought a "
      "**mechanism reading**, and this is the reading:")
    A("")
    A("- The note was **built, bounded and delivered exactly as designed.** "
      "D1/D2/D3/D4 all land INSIDE the control spread, the offline replay and "
      "the live in-kernel module agree on which games each detector fired on "
      "(17/1/6 both), and the character clamp held at 603 of 700. Nothing about "
      "the implementation failed.")
    A("- **The detectors kept selecting real waste.** The arm's post-stall "
      "revisit rate is 2.45× its own non-stall rate (0.4971 vs 0.2030), the "
      "same 2.5–3.6× ratio the controls show. The note fired at genuinely "
      "wasteful moments — and the agent went on revisiting anyway.")
    A("- **The one movement outside the spread is in the WRONG direction.** "
      "Mean actions per stall turn is **11.11** against a control range of "
      "3.95–7.28, and B3 over-target burn totals **2470** against 1267–2301. On "
      "a turn where the note said 'you are cycling', the agent issued MORE "
      "actions than any control run did, not fewer. This is descriptive and "
      "n = 1; it is reported because it is the opposite of the intended effect, "
      "not as a harm finding.")
    A("- **The one flattering number is not evidence.** Median actions per "
      "cleared level is 23.5 against a control range of 24–49. It is "
      "DESCRIPTIVE ONLY by seal (M2), it is confounded by which levels this "
      "draw happened to clear, and per the prereg 'attributing either to "
      "EFFNOTE would be exactly the error this prereg exists to prevent'. It is "
      "NOT cited.")
    A("")
    A("---")
    A("")
    A("## 6. HONEST EVIDENCE CLASS")
    A("")
    A("This mechanism appears in exactly **one** ≥1.40 kernel "
      "(`caoyupeng/arc3-duck-v12-1d7d88`, Tara Labs #37 @ 1.46, behind "
      "`install(bm, flags={\"efficiency\": True, ...})` over "
      "`thtennant/taaf-kaggle-source-share-fork`). The graft's author, "
      "thtennant, is at **1.28**. It ranks #1 on mechanism-to-diagnosis fit and "
      "**carries no efficacy evidence whatsoever** — that was stated in the "
      "prereg before the push and it is unchanged by this run. **A delivered "
      "mechanism is NOT an efficacy claim.** This arm delivered the note "
      "cleanly, on all 25 games, inside its character bound, with zero errors — "
      "and the behaviour it was built to change did not change. The delivery "
      "number is not a win; it is the precondition that made the null readable. "
      "The standing counter-evidence sealed with the prereg still stands: "
      "efficiency_diagnosis §3 found **vc33** cleared at 3.00× the human count "
      "with zero duplicates, zero no-ops, zero revisits and a provably minimal "
      "path (100% capability, untouchable by any note), §2.1 puts **40%** of the "
      "gap in that class, and the efficiency lane's ceiling is "
      "**~1.26–1.36 LB — short of the 1.48–1.58 gold line** even if it worked.")
    A("")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--md", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    run = Path(args.run)
    if not run.is_absolute():
        run = REPO / run

    rep = score(run)

    def P(*a):
        if not args.quiet:
            print(*a)

    v = rep["verdict"]
    b1 = rep["PRIMARY"]["part2_B1"]
    P("=" * 78)
    P(f"EFFNOTE SCORER  run={run}")
    P(f"  log: {rep['log'].get('path')}  format={rep['log'].get('source')}"
      f"  chars={rep['log'].get('chars')}")
    if rep["log"].get("error") or rep["canary_error"] or rep["replay_error"]:
        P(f"  EVIDENCE: log={rep['log'].get('error')} canary={rep['canary_error']} "
          f"replay={rep['replay_error']}")
    P("=" * 78)
    P("CANARIES (prereg sec4)")
    for k in CANARY_ORDER:
        c = rep["canaries"][k]
        P(f"  {k:6s} {c['status']:6s} {c['name'][:88]}")
    P("=" * 78)
    P("PRIMARY")
    P(f"  part 1 delivery : {rep['PRIMARY']['part1_delivery']['status']}"
      f"  note_rate={rep['PRIMARY']['part1_delivery']['note_rate']}"
      f"  chars_max={rep['PRIMARY']['part1_delivery']['chars_max']}")
    if b1["value"] is None:
        P(f"  part 2 B1       : UNDEFINED -- {b1.get('error')}")
    else:
        P(f"  part 2 B1       : {b1['status']}  B1={b1['value']:.6f} vs line "
          f"{b1['pass_line']:.6f} (strict <)  spread "
          f"{b1['control_spread'][0]:.4f}-{b1['control_spread'][1]:.4f}")
    P("=" * 78)
    P("KILL RULES")
    for r in rep["kill_rules"]:
        state = "FIRED" if r["fired"] else ("ok" if r["evaluable"] else "UNEVALUABLE")
        P(f"  {r['rule']}. {state:11s} {r['name']}")
        P(f"       {r['detail']}")
    P("=" * 78)
    P(f"  VERDICT: {v['verdict']}")
    P(f"  {v['why']}")

    if args.json:
        outp = Path(args.json)
        if not outp.is_absolute():
            outp = REPO / outp
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
        P(f"\nwrote {outp}")
    if args.md:
        outm = Path(args.md)
        if not outm.is_absolute():
            outm = REPO / outm
        outm.parent.mkdir(parents=True, exist_ok=True)
        outm.write_text(render_markdown(rep), encoding="utf-8")
        P(f"wrote {outm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
