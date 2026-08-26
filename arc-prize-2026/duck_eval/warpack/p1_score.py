"""P1 arm scorer -- reads a pulled `canivel/arc3-duck-p1-eval` output and grades
it against the SEALED prereg `learnings/war_room/p1_prereg_2026-08-12.md`.

Everything graded here is pre-registered. Nothing is invented after the fact.

  K-P0..K-P6  canaries (prereg sec5). K-P0..K-P3 hard; K-P4..K-P6 reading gates.
  M0          PRIMARY, mechanism delivery: saved/requested, band [3%, 30%]
              (square brackets == INCLUSIVE both ends; kill rule 5 is the
              strict `M0 < 3%`, so exactly 3.00% is in band and is NOT a kill).
  M1          Delta-lc vs family -- DESCRIPTIVE ONLY (family m=2 => NOT
              SCREENABLE, SCREEN_PROTOCOL sec1 P2 / sec4.6 power-honesty).
  M2          RHAE score + multiplier. Sealed replay expectation x1.019 mean.
              **M2 below x1.019 is explicitly NOT a kill rule (prereg sec6).**
  M3          duplicate re-execution rate (the K-P6 quantity).
  MECH-C      diagnostic, NOT pre-registered: did the non-truncatable memory
              block land in the LIVE prompt, and did behaviour change?

EVIDENCE DISCIPLINE (the animation post-mortem, 2026-08-12). A field that is
missing, empty or unparseable NEVER degrades to 0 / "" / PASS. It produces an
explicit ERROR on that canary and a DISCARD verdict, because prereg sec5 says a
hard-canary failure means "nothing may be read from it" -- and a scorer that
cannot read the evidence has not observed a mechanism result either way.
Every canary here can both PASS and FAIL on the same code path; see
`p1_score_selftest.py`, which drives all seven in both directions.

Verdict states (resolution order documented in `resolve_verdict`):
  KILL        one of the five sealed kill rules (prereg sec6) fired.
  DISCARD     the run is discard-grade: a hard canary K-P0..K-P3 failed or
              could not be evaluated, evidence is missing/unparseable, or M0
              is ABOVE the band (prereg sec3: "must be inspected before any
              reading" -- not a kill, but not readable either).
  NO-PROMOTE  canaries clear and no kill rule fired, but a reading gate
              (K-P4..K-P6) did not pass: the mechanism did not deliver and
              M1/M2 may not be read as evidence of anything.
  PROMOTE     hard canaries pass, no kill rule fires, all reading gates pass.

Usage:
  .venv/Scripts/python.exe duck_eval/warpack/p1_score.py --run runs/kernel_pulls/p1_v1
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

import p1_replay_validate as V  # noqa: E402  (brings rhae_score + the trace reader)

rhae_score = V.rhae_score

# Family: duck + (f) continuation, NO warpack. m = 2 => NOT SCREENABLE.
# (These are the two runs whose benchmark.json carries the bare
# `duck-harness-kaggle-continuation-v1` label -- w0_eval_s1 lc 16, w0_cont_eval
# lc 10, exactly the "m = 2 (lc 10, 16)" the prereg sec3 M0 names.)
FAMILY_RUNS = ["runs/kernel_pulls/w0_eval_s1", "runs/kernel_pulls/w0_cont_eval"]
FAMILY_LABEL = "duck-harness-kaggle-continuation-v1"
M0_BAND = (0.03, 0.30)          # prereg sec3 M0: INCLUSIVE both ends
SEALED_MULTIPLIER = 1.019       # prereg sec3 M2 -- NOT a kill line (sec6)
SEALED_M0_REPLAY = {"animation_v1": (306, 5151), "a22_v2_seed1": (697, 3492),
                    "a22_compaction_v1": (841, 4777)}
KILL_LC_FLOOR = 15              # prereg sec6.2: levels_completed <= 15 kills
SEALED_MEMO_MODE = "noop"       # prereg sec4 consequence 1
SEALED_ABORT_REVISIT = 0        # prereg sec4 consequence 1
SEALED_CONFIRM_FLOOR = 2        # prereg sec4 consequence 2 (a floor, not a flag)
FAMILY_CACHE = REPO / "runs" / "p1_replay" / "family_dup_cache.json"


# --------------------------------------------------------------------------- #
# log loading -- Kaggle build logs are a JSON ARRAY of {stream_name,time,data}
# records, NOT plain text. Reading them raw makes every `^P1 ...` match fail and
# silently manufactures "no canary / no events", which is a KILL verdict from an
# infra bug. Mirrors animation_score.load_log_text.
# --------------------------------------------------------------------------- #
def load_log_text(path: Path) -> tuple[str, str]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        recs = json.loads(raw)
        if isinstance(recs, list):
            return ("".join(str(r.get("data", "")) for r in recs
                            if isinstance(r, dict)), "json-array")
    except Exception:  # noqa: BLE001 - truncated log: salvage record by record
        pass
    parts: list[str] = []
    salvaged = 0
    for line in raw.splitlines():
        s = line.strip().lstrip(",").rstrip(",")
        if not s.startswith("{"):
            continue
        try:
            rec = json.loads(s)
        except Exception:  # noqa: BLE001
            continue
        parts.append(str(rec.get("data", "")))
        salvaged += 1
    if salvaged:
        return "".join(parts), f"json-salvage({salvaged} records)"
    return raw, "raw-text"


def find_log(run: Path) -> Path | None:
    cands = [p for p in sorted(run.glob("*.log")) + sorted(run.glob("**/*.log"))
             if "vllm" not in p.name.lower()]
    if not cands:
        return None
    named = [p for p in cands if "p1" in p.name.lower()]
    return (named or cands)[0]


def read_log(run: Path) -> tuple[str, dict]:
    """Returns (text, meta). meta.source is 'MISSING' when there is no log at
    all -- the caller must treat that as an ERROR, never as an empty run."""
    p = find_log(run)
    if p is None:
        return "", {"path": None, "source": "MISSING",
                    "error": f"no non-vllm *.log under {run}"}
    txt, fmt = load_log_text(p)
    meta = {"path": str(p), "source": fmt, "chars": len(txt)}
    if not txt.strip():
        meta["error"] = f"{p} decoded to an empty blob"
    return txt, meta


# --------------------------------------------------------------------------- #
# emitter-derived parsers.
# Byte-shapes taken from `_kaggle_dataset/p1_suppressor_patch.py`:
#   canary_report():
#     P1 CANARY v=1 version=v1 games=N executed=N declined=N aborted=N
#     dup_exec=N dup_rate=0.0000 errors=N ambiguous_games=a,b|NONE mode=noop
#     confirm=2 abort_revisit=0
#   _emit(kind, st, detail):
#     P1 v=1 kind=K game=G level=N declined=N aborted=N dup_req=N dup_exec=N
#     amb=0|1 amb_pairs=N <detail>
#   notebook cell 14 failure path:
#     P1 CANARY unavailable: <repr>
# --------------------------------------------------------------------------- #
CANARY_RE = re.compile(
    r"P1 CANARY v=(?P<v>\S+) version=(?P<version>\S+) games=(?P<games>\d+) "
    r"executed=(?P<executed>\d+) declined=(?P<declined>\d+) "
    r"aborted=(?P<aborted>\d+) dup_exec=(?P<dup_exec>\d+) "
    r"dup_rate=(?P<dup_rate>[0-9.eE+-]+) errors=(?P<errors>\d+) "
    r"ambiguous_games=(?P<ambiguous_games>\S*) mode=(?P<mode>\S+) "
    r"confirm=(?P<confirm>\d+) abort_revisit=(?P<abort_revisit>\d+)")
CANARY_LOOSE_RE = re.compile(r"P1 CANARY (?!unavailable)(?P<body>.+)")
CANARY_UNAVAILABLE_TOKEN = "P1 CANARY unavailable"
EVENT_RE = re.compile(
    r"P1 v=(?P<v>\S+) kind=(?P<kind>\S+) game=(?P<game>\S+) level=(?P<level>\d+) "
    r"declined=(?P<declined>\d+) aborted=(?P<aborted>\d+) "
    r"dup_req=(?P<dup_req>\d+) dup_exec=(?P<dup_exec>\d+) "
    r"amb=(?P<amb>[01]) amb_pairs=(?P<amb_pairs>\d+)(?P<detail>[^\n]*)")

# Every canary field the scorer READS. Cross-checked field-by-field against the
# emitter's print in `p1_score_selftest.py` (test group X) -- the animation arm
# died because the scorer read `token_fraction=`, which the builder never
# populated.
CANARY_REQUIRED = ("v", "version", "games", "executed", "declined", "aborted",
                   "dup_exec", "dup_rate", "errors", "ambiguous_games",
                   "mode", "confirm", "abort_revisit")


def _short(g: str) -> str:
    return g.split("-")[0]


def parse_canary(log: str) -> tuple[dict | None, str | None]:
    """Strict parse of the single `P1 CANARY` line.

    Returns (canary, error). A malformed / partial / absent line is an ERROR --
    it is NEVER returned as an empty dict that later reads as errors=0."""
    if not log:
        return None, "no log text"
    if CANARY_UNAVAILABLE_TOKEN in log:
        i = log.find(CANARY_UNAVAILABLE_TOKEN)
        return None, ("the builder printed "
                      f"{log[i:log.find(chr(10), i) if log.find(chr(10), i) > 0 else i + 200].strip()!r}"
                      " -- canary_report() raised, so there is no canary line")
    m = None
    for m in CANARY_RE.finditer(log):
        pass
    if m is None:
        loose = CANARY_LOOSE_RE.search(log)
        if loose:
            body = loose.group("body")
            present = {kv.split("=", 1)[0] for kv in body.split() if "=" in kv}
            missing = [k for k in CANARY_REQUIRED if k not in present]
            return None, (f"P1 CANARY line present but MALFORMED; missing/renamed "
                          f"field(s) {missing}; raw={body.strip()[:300]!r}")
        return None, "no `P1 CANARY` line in the log"
    d = m.groupdict()
    amb_raw = d["ambiguous_games"]
    ambiguous = [] if amb_raw in ("NONE", "", "0") else [
        x for x in amb_raw.split(",") if x]
    return dict(
        v=d["v"], version=d["version"], games=int(d["games"]),
        executed=int(d["executed"]), declined=int(d["declined"]),
        aborted=int(d["aborted"]), dup_exec=int(d["dup_exec"]),
        dup_rate=float(d["dup_rate"]), errors=int(d["errors"]),
        ambiguous_games=ambiguous, ambiguous_games_raw=amb_raw,
        mode=d["mode"], confirm=int(d["confirm"]),
        abort_revisit=int(d["abort_revisit"]), raw=m.group(0),
    ), None


def parse_events(log: str) -> list[dict]:
    evs = []
    for m in EVENT_RE.finditer(log):
        d = m.groupdict()
        detail = d["detail"].strip()
        evs.append(dict(
            v=d["v"], kind=d["kind"], game=d["game"], game_short=_short(d["game"]),
            level=int(d["level"]), declined=int(d["declined"]),
            aborted=int(d["aborted"]), dup_req=int(d["dup_req"]),
            dup_exec=int(d["dup_exec"]), amb=d["amb"] == "1",
            amb_pairs=int(d["amb_pairs"]), detail=detail, raw=m.group(0)))
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


def score_bench(rows: list[dict]) -> float | None:
    if not rows:
        return None
    return sum(rhae_score(r["base_actions_per_level"], r["actions_per_level"],
                          r["levels_completed"], r["number_of_levels"])
               for r in rows) / len(rows)


# --------------------------------------------------------------------------- #
# MECH-C (diagnostic; not pre-registered)
# --------------------------------------------------------------------------- #
_TURN_RE = re.compile(r"^--- analysis_step=(\d+)", re.M)
_BLOCK_RE = re.compile(r"P1 memory \(runner ground truth.*?(?=\n\n|\n\[|\Z)", re.S)
_DEAD_RE = re.compile(r"CONFIRMED NO EFFECT from this exact board: ([^.]+)\.")
_MORE_RE = re.compile(r"\s*\(\+\d+ more\)\s*$")


def _turns(txt: str) -> list[tuple[int, str]]:
    """Split a per-turn transcript into (analysis_step, text) chunks."""
    marks = [(int(m.group(1)), m.start()) for m in _TURN_RE.finditer(txt)]
    out = []
    for i, (step, s) in enumerate(marks):
        e = marks[i + 1][1] if i + 1 < len(marks) else len(txt)
        out.append((step, txt[s:e]))
    return out


def mechanism_c(run: Path, log: str) -> dict:
    """Did the memory block reach the LIVE prompt every turn, and did the agent
    ACT on it?

    `transcripts/*.txt` records the exact `[USER PROMPT]` for EVERY analysis
    step (not just the last), so block presence is directly observable -- no
    inference. The behavioural test is `dead_reissue`: of the actions the agent
    issued on a turn, how many were on that same turn's "CONFIRMED NO EFFECT"
    list? If the block is being read, this trends to zero.
    """
    out: dict = {"source": None, "games": 0, "turns_seen": 0,
                 "turns_with_block": 0, "blocks_with_untried": 0,
                 "blocks_with_dead": 0, "blocks_flagging_latent": 0,
                 "games_with_block": [], "dead_listed_turns": 0,
                 "dead_reissued": 0, "actions_after_dead_turns": 0,
                 "first_half_reissue": 0, "second_half_reissue": 0,
                 "block_chars_mean": None, "block_chars_max": None,
                 "block_coverage": None, "dead_reissue_rate": None}
    src = run / "transcripts"
    if not src.is_dir():
        src = run / "prompts"
    if not src.is_dir():
        out["error"] = f"no transcripts/ or prompts/ dir under {run}"
        return out
    out["source"] = src.name
    chars: list[int] = []
    games: set[str] = set()

    # per-game action_display by analysis_step, from the event logs
    by_step: dict[str, dict[int, list[str]]] = {}
    art = run / "artifacts"
    if art.is_dir():
        for p in art.glob("*_events.jsonl"):
            gid = p.name.split("_p0_events")[0]
            d: dict[int, list[str]] = defaultdict(list)
            with p.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or '"type": "action"' not in line.replace('"type":"action"', '"type": "action"'):
                        continue
                    try:
                        ev = json.loads(line)
                    except Exception:  # noqa: BLE001
                        continue
                    if ev.get("type") != "action":
                        continue
                    st = ev.get("analysis_step")
                    if st is None:
                        continue
                    d[int(st)].append(str(ev.get("action_display")
                                          or ev.get("action_name") or ""))
            by_step[gid] = d

    for p in sorted(src.glob("*")):
        if p.suffix not in (".txt", ".log"):
            continue
        out["games"] += 1
        gid = p.stem.replace("_p0", "")
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            continue
        turns = _turns(txt)
        if not turns:
            turns = [(0, txt)]
        out["turns_seen"] += len(turns)
        steps = by_step.get(f"{gid}", by_step.get(gid, {}))
        n_turns = len(turns)
        for idx, (step, chunk) in enumerate(turns):
            m = _BLOCK_RE.search(chunk)
            if not m:
                continue
            blk = m.group(0)
            out["turns_with_block"] += 1
            games.add(gid[:4])
            chars.append(len(blk))
            out["blocks_with_untried"] += "NOT YET TRIED" in blk
            out["blocks_flagging_latent"] += "latent state" in blk
            dm = _DEAD_RE.search(blk)
            if not dm:
                continue
            out["blocks_with_dead"] += 1
            # the emitter appends " (+N more)" to the LAST listed primitive when
            # the list is capped at P1_BLOCK_MAX_DEAD -- strip it, or the last
            # dead primitive of every capped block never matches an issued
            # action and dead_reissued silently undercounts.
            dead = {_MORE_RE.sub("", a.strip()) for a in dm.group(1).split(",")
                    if a.strip()}
            dead = {a for a in dead if a}
            issued = steps.get(step, [])
            if not issued:
                continue
            out["dead_listed_turns"] += 1
            out["actions_after_dead_turns"] += len(issued)
            hits = sum(1 for a in issued if a in dead)
            out["dead_reissued"] += hits
            if idx < n_turns / 2:
                out["first_half_reissue"] += hits
            else:
                out["second_half_reissue"] += hits

    out["games_with_block"] = sorted(games)
    out["block_chars_mean"] = round(sum(chars) / len(chars), 1) if chars else None
    out["block_chars_max"] = max(chars) if chars else None
    out["block_coverage"] = (out["turns_with_block"] / out["turns_seen"]
                             if out["turns_seen"] else None)
    out["dead_reissue_rate"] = (out["dead_reissued"] / out["actions_after_dead_turns"]
                                if out["actions_after_dead_turns"] else None)
    return out


# --------------------------------------------------------------------------- #
# behaviour, from a run's OWN event logs (diagnosis definitions)
# --------------------------------------------------------------------------- #
def behaviour(run: Path) -> dict:
    """Re-exploration behaviour using exactly the definitions the diagnosis
    used (so arm and family are comparable).

    `dup_all`      -- duplicate (board, action) over ALL actions, memo scoped per
                      level. This is the like-for-like partner of the canary's
                      `dup_exec / executed`, which is also whole-game.
    `dup_cleared`  -- the same, restricted to CLEARED levels: the diagnosis's
                      "as-run on cleared levels 10.5% / 4.9% / 5.0%" quantity.
    """
    art = run / "artifacts"
    if not art.is_dir():
        return {}
    tot = Counter()
    per_game = {}
    rows_list, _err = bench_rows(run)
    rows = {r["game_id"]: r for r in rows_list}
    for p in sorted(art.glob("*_events.jsonl")):
        gid = p.name.split("_p0_events")[0]
        acts = V._replay_events(p)
        if not acts:
            continue
        r = rows.get(gid)
        cleared = r["levels_completed"] if r else 0
        seen: dict[int, set] = {}
        vis: dict[int, set] = {}
        dup = blind = n = 0
        dup_c = n_c = 0
        dead = False
        cur = None
        steps = set()
        for a in acts:
            lv = a["level"]
            sp = seen.setdefault(lv, set())
            vs = vis.setdefault(lv, set())
            if not vs:
                vs.add(a["prev"])
            if (a["step"], a["bs"]) != cur:
                cur = (a["step"], a["bs"])
                dead = False
            n += 1
            steps.add(a["step"])
            key = (a["prev"], a["act"])
            is_dup = key in sp
            dup += is_dup
            blind += dead
            if lv < cleared:
                n_c += 1
                dup_c += is_dup
            sp.add(key)
            rev = a["out"] in vs
            vs.add(a["out"])
            if a["bs"] > 1 and not dead and ((not a["bc"]) or rev):
                dead = True
        tot["actions"] += n
        tot["dup"] += dup
        tot["blind"] += blind
        tot["steps"] += len(steps)
        tot["actions_cleared"] += n_c
        tot["dup_cleared"] += dup_c
        per_game[gid[:4]] = {
            "actions": n, "dup": dup, "blind": blind,
            "dup_rate": dup / n if n else 0.0,
            "blind_rate": blind / n if n else 0.0,
            "actions_per_step": n / len(steps) if steps else 0.0,
            "levels_completed": cleared,
        }
    if not per_game:
        return {}
    return {"totals": dict(tot), "per_game": per_game,
            "dup_rate": tot["dup"] / max(1, tot["actions"]),
            "dup_rate_cleared": (tot["dup_cleared"] / tot["actions_cleared"]
                                 if tot["actions_cleared"] else None),
            "blind_rate": tot["blind"] / max(1, tot["actions"]),
            "actions_per_step": tot["actions"] / max(1, tot["steps"])}


def family_dup(dirs: list[Path], use_cache: bool = True) -> dict:
    """Pooled as-run duplicate rate of the m=2 comparator family (K-P6's
    reference). Cached: parsing the family's ~230 MB of event logs takes
    minutes and never changes."""
    key = "|".join(f"{d}:{sum(p.stat().st_size for p in sorted((d / 'artifacts').glob('*_events.jsonl')))}"
                   if (d / "artifacts").is_dir() else f"{d}:MISSING" for d in dirs)
    if use_cache and FAMILY_CACHE.is_file():
        try:
            cached = json.loads(FAMILY_CACHE.read_text(encoding="utf-8"))
            if cached.get("key") == key:
                return cached["value"]
        except Exception:  # noqa: BLE001
            pass
    runs, missing = {}, []
    dup = act = dup_c = act_c = 0
    for d in dirs:
        if not d.is_dir():
            missing.append(str(d))
            continue
        b = behaviour(d)
        if not b:
            missing.append(f"{d} (no parseable artifacts/*_events.jsonl)")
            continue
        t = b["totals"]
        runs[d.name] = {"label": bench_label(d),
                        "label_matches_family": bench_label(d) == FAMILY_LABEL,
                        "actions": t["actions"], "dup": t["dup"],
                        "dup_rate": b["dup_rate"],
                        "actions_cleared": t.get("actions_cleared", 0),
                        "dup_cleared": t.get("dup_cleared", 0),
                        "dup_rate_cleared": b["dup_rate_cleared"]}
        dup += t["dup"]
        act += t["actions"]
        dup_c += t.get("dup_cleared", 0)
        act_c += t.get("actions_cleared", 0)
    value = {"runs": runs, "m": len(runs), "missing": missing,
             "pooled_dup_rate_all_actions": (dup / act) if act else None,
             "pooled_dup_rate_cleared_levels": (dup_c / act_c) if act_c else None,
             "error": (f"family runs unavailable: {missing}" if missing or not runs
                       else None),
             "all_labels_match_family": bool(runs) and all(
                 r["label_matches_family"] for r in runs.values())}
    if use_cache and not value["error"]:
        try:
            FAMILY_CACHE.parent.mkdir(parents=True, exist_ok=True)
            FAMILY_CACHE.write_text(json.dumps({"key": key, "value": value},
                                               indent=2), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    return value


def family_lc(dirs: list[Path]) -> dict:
    """Per-game levels_completed mean over the family (kill rule 3's reference)."""
    per: dict[str, list[int]] = defaultdict(list)
    runs, missing = {}, []
    for d in dirs:
        rows, err = bench_rows(d)
        if err:
            missing.append(f"{d}: {err}")
            continue
        runs[d.name] = {"label": bench_label(d),
                        "lc_total": sum(r["levels_completed"] for r in rows)}
        for r in rows:
            per[_short(r["game_id"])].append(r["levels_completed"])
    return {"runs": runs, "m": len(runs), "missing": missing,
            "per_game_mean": {g: sum(v) / len(v) for g, v in per.items()},
            "per_game_values": dict(per),
            "error": (f"family benchmarks unavailable: {missing}"
                      if missing or not runs else None)}


# --------------------------------------------------------------------------- #
# canaries (prereg sec5)
# --------------------------------------------------------------------------- #
HARD_CANARIES = ("K-P0", "K-P1", "K-P2", "K-P3")
READING_GATES = ("K-P4", "K-P5", "K-P6")


def run_canaries(log: str, log_meta: dict, events: list[dict],
                 canary: dict | None, canary_err: str | None,
                 fam_dup: dict) -> dict:
    """Every canary returns PASS / FAIL / ERROR (+DISPUTED for K-P6).
    ERROR == the evidence is missing or unreadable; it is never a PASS and never
    a silent 0."""
    out: dict = {}
    log_missing = log_meta.get("source") == "MISSING" or bool(log_meta.get("error"))

    # ---- K-P0 (HARD): banner + 4 seams + applied=True + no PATCH FAILED -----
    banner = "p1 v1: ACTIVE" in log
    seams4 = "ACTIVE (4 seams patched)" in log
    applied = "applied=True" in log
    patch_failed = "p1: PATCH FAILED" in log
    if log_missing:
        st0 = "ERROR"
    else:
        st0 = "PASS" if (banner and seams4 and applied and not patch_failed) else "FAIL"
    out["K-P0"] = dict(
        name="`p1 v1: ACTIVE (4 seams patched)` + applied=True + no PATCH FAILED",
        hard=True, status=st0,
        banner_present=banner, four_seams=seams4, applied_true=applied,
        patch_failed_line=patch_failed,
        error=log_meta.get("error"),
        evidence=[_line_with(log, "p1 v1: ACTIVE"), _line_with(log, "applied="),
                  _line_with(log, "p1: PATCH FAILED")])

    # ---- K-P1 (HARD): >=1 `P1 ` event line on >=5 distinct games ------------
    games = sorted({e["game"] for e in events})
    st1 = "ERROR" if log_missing else ("PASS" if len(events) >= 1 and len(games) >= 5
                                       else "FAIL")
    out["K-P1"] = dict(
        name=">=1 `P1 v=1 kind=...` event line on >=5 distinct games",
        hard=True, status=st1, event_lines=len(events),
        distinct_games=len(games), games=games,
        kinds=dict(Counter(e["kind"] for e in events)),
        error=log_meta.get("error"),
        note=("`kind=game_end` is emitted once per game by seam 1 regardless of "
              "whether any suppression happened, so K-P1 as sealed tests that "
              "the seams ran, NOT that the mechanism engaged (that is K-P4). "
              "Reported, not reinterpreted."),
        evidence=[events[0]["raw"]] if events else [])

    # ---- K-P2 (HARD): the banner states the safe defaults -------------------
    tok = {t: (t in log) for t in ("mode=noop", "confirm=2", "revisit is DEFAULT OFF")}
    st2 = "ERROR" if log_missing else ("PASS" if all(tok.values()) else "FAIL")
    out["K-P2"] = dict(name="banner states mode=noop, confirm=2, revisit is DEFAULT OFF",
                       hard=True, status=st2, tokens=tok,
                       error=log_meta.get("error"),
                       evidence=[_line_with(log, "revisit is DEFAULT OFF")])

    # ---- K-P3 (HARD): errors=0 on the canary line ---------------------------
    if canary is None:
        st3, errs = "ERROR", None
    else:
        errs = canary["errors"]
        st3 = "PASS" if errs == 0 else "FAIL"
    out["K-P3"] = dict(name="errors=0 on the `P1 CANARY` line", hard=True,
                       status=st3, errors=errs, error=canary_err,
                       evidence=[canary["raw"]] if canary else [])

    # ---- K-P4 (READING GATE): M0 in [3%, 30%] -------------------------------
    m0 = m0_from(canary)
    if canary is None:
        st4 = "ERROR"
    elif m0["requested"] == 0:
        st4 = "ERROR"
        m0["error"] = ("requested = declined + aborted + executed = 0: the arm "
                       "took no actions, so M0 is undefined (NOT 0%)")
    else:
        st4 = "PASS" if m0["in_band"] else "FAIL"
    out["K-P4"] = dict(name="M0 saved/requested in [3%, 30%] (both ends INCLUSIVE)",
                       hard=False, status=st4, error=canary_err or m0.get("error"),
                       **{k: m0[k] for k in ("saved", "requested", "rate",
                                             "band", "in_band", "below_band",
                                             "above_band")},
                       above_band_consequence=("prereg sec3: above 30% the arm "
                                               "'must be inspected before any "
                                               "reading' -- not a kill rule"),
                       evidence=[canary["raw"]] if canary else [])

    # ---- K-P5 (READING GATE): the online detector flagged >=1 game ----------
    if canary is None:
        st5, amb = "ERROR", None
    else:
        amb = canary["ambiguous_games"]
        st5 = "PASS" if amb else "FAIL"
    out["K-P5"] = dict(
        name="online latent-state detector flagged >=1 game (ambiguous_games non-empty)",
        hard=False, status=st5, ambiguous_games=amb,
        n_ambiguous=(len(amb) if amb is not None else None), error=canary_err,
        note=("the shipped detector is LEVEL-scoped (P1State.sync_level clears "
              "the memo), while the replay certification that produced the "
              "published 8-game set was GAME-scoped. A level-scoped detector "
              "sees strictly fewer contradictions, so K-P5 can fail while the "
              "safety rule is intact. FLAGGED, not reinterpreted."),
        evidence=[canary["raw"]] if canary else [])

    # ---- K-P6 (READING GATE): dup_rate below the family's as-run rate -------
    arm_dr = canary["dup_rate"] if canary else None
    fam_all = fam_dup.get("pooled_dup_rate_all_actions")
    fam_cl = fam_dup.get("pooled_dup_rate_cleared_levels")
    if arm_dr is None or fam_all is None:
        st6 = "ERROR"
    else:
        below_all = arm_dr < fam_all
        below_cl = (arm_dr < fam_cl) if fam_cl is not None else below_all
        if below_all != below_cl:
            st6 = "DISPUTED"
        else:
            st6 = "PASS" if below_all else "FAIL"
    out["K-P6"] = dict(
        name="M3 dup_rate below the family's as-run duplicate rate",
        hard=False, status=st6, arm_dup_rate=arm_dr,
        family_dup_rate_all_actions=fam_all,
        family_dup_rate_cleared_levels=fam_cl,
        family=fam_dup.get("runs"), family_m=fam_dup.get("m"),
        error=canary_err or fam_dup.get("error"),
        comparator_note=(
            "PRIMARY comparator is the family's whole-run rate, which is the "
            "like-for-like partner of the canary's dup_exec/executed (also "
            "whole-run). The cleared-levels-only rate (the diagnosis's 10.5% "
            "column) is carried as the second reading; the prereg does not say "
            "which it means, so if the two straddle the arm's value the gate is "
            "DISPUTED and must be escalated, not executed."),
        evidence=[canary["raw"]] if canary else [])
    return out


def _line_with(log: str, token: str) -> str | None:
    i = log.find(token)
    if i < 0:
        return None
    s = log.rfind("\n", 0, i) + 1
    e = log.find("\n", i)
    return log[s:(e if e > 0 else min(len(log), i + 300))].strip()[:400] or None


def m0_from(canary: dict | None) -> dict:
    """M0 = saved/requested, saved = declined + aborted, requested = saved +
    executed. Band [3%, 30%] INCLUSIVE (prereg sec3 writes square brackets, and
    kill rule 5 is the strict `< 3%`, so exactly 3.00% is in band and alive)."""
    if canary is None:
        return dict(saved=None, requested=None, rate=None, band=list(M0_BAND),
                    in_band=None, below_band=None, above_band=None,
                    error="no canary line -- M0 is UNDEFINED, not 0%")
    saved = canary["declined"] + canary["aborted"]
    requested = saved + canary["executed"]
    if requested == 0:
        return dict(saved=saved, requested=0, rate=None, band=list(M0_BAND),
                    in_band=None, below_band=None, above_band=None)
    rate = saved / requested
    return dict(saved=saved, requested=requested, rate=rate, band=list(M0_BAND),
                in_band=M0_BAND[0] <= rate <= M0_BAND[1],
                below_band=rate < M0_BAND[0], above_band=rate > M0_BAND[1])


# --------------------------------------------------------------------------- #
# kill rules (prereg sec6) -- all five, each individually evaluable
# --------------------------------------------------------------------------- #
def kill_rules(canaries: dict, canary: dict | None, m0: dict,
               rows: list[dict], rows_err: str | None, arm_run: Path,
               fam_lc: dict) -> list[dict]:
    R: list[dict] = []

    # 1. K-P0 or K-P3 fails.
    k0, k3 = canaries["K-P0"]["status"], canaries["K-P3"]["status"]
    R.append(dict(
        rule=1, name="K-P0 or K-P3 fails (patch did not install, or the action "
                     "path raised)",
        fired=(k0 == "FAIL" or k3 == "FAIL"),
        evaluable=(k0 != "ERROR" and k3 != "ERROR"),
        detail=f"K-P0={k0} K-P3={k3}",
        note="INFRA DEATH when K-P0 fails: not a mechanism result."))

    # 2. levels_completed <= 15 on the local 25.
    lc = sum(r["levels_completed"] for r in rows) if rows else None
    R.append(dict(
        rule=2, name=f"levels_completed <= {KILL_LC_FLOOR} on the local 25",
        fired=(lc is not None and lc <= KILL_LC_FLOOR),
        evaluable=(lc is not None),
        detail=(f"levels_completed={lc} over {len(rows)} games" if lc is not None
                else f"UNEVALUABLE: {rows_err}"),
        levels_completed=lc, n_games=len(rows)))

    # 3. any ambiguity-flagged game loses a level vs the family per-game mean.
    losers: list[dict] = []
    ev3 = canary is not None and not fam_lc.get("error") and bool(rows)
    if ev3:
        arm_lc = {_short(r["game_id"]): r["levels_completed"] for r in rows}
        means = fam_lc["per_game_mean"]
        for g in canary["ambiguous_games"]:
            gs = _short(g)
            if gs not in arm_lc or gs not in means:
                ev3 = False
                losers.append(dict(game=gs, error="game absent from the arm "
                                                  "benchmark or the family"))
                continue
            if arm_lc[gs] < means[gs]:
                losers.append(dict(game=gs, arm_lc=arm_lc[gs],
                                   family_mean_lc=means[gs],
                                   delta=arm_lc[gs] - means[gs],
                                   full_level_loss=arm_lc[gs] <= means[gs] - 1))
    R.append(dict(
        rule=3, name="any game whose ambiguity flag fired loses a level vs the "
                     "family per-game mean",
        fired=bool([x for x in losers if "error" not in x]), evaluable=ev3,
        detail=(f"{len(canary['ambiguous_games'])} flagged game(s); "
                f"{len([x for x in losers if 'error' not in x])} below the family mean"
                if canary else "UNEVALUABLE: no canary line"),
        flagged_games=(canary["ambiguous_games"] if canary else None),
        losers=losers,
        interpretation=("'loses a level' is read as arm_lc < family per-game "
                        "mean (any loss). `full_level_loss` reports the "
                        "stricter arm_lc <= mean - 1 reading alongside it; the "
                        "prereg does not disambiguate and the safety-first "
                        "reading is used for the gate.")))

    # 4. any level-completing action is declined or aborted.
    R.append(kill_rule_4(canary, rows, rows_err, arm_run))

    # 5. M0 < 3%.
    R.append(dict(
        rule=5, name="M0 < 3% (mechanism did not engage -- null by delivery)",
        fired=bool(m0.get("below_band")), evaluable=(m0.get("rate") is not None),
        detail=(f"M0={m0['rate']*100:.2f}%" if m0.get("rate") is not None
                else f"UNEVALUABLE: {m0.get('error') or 'no M0'}"),
        note="M0 ABOVE 30% is NOT this kill rule (prereg sec3: inspect first)."))
    return R


def kill_rule_4(canary: dict | None, rows: list[dict], rows_err: str | None,
                arm_run: Path) -> dict:
    """Prereg sec6.4. A declined/aborted action can never CARRY
    `level_completed`, which is why the prereg calls this a design invariant
    rather than a statistic. Two consequences the scorer must be honest about:

      * a direct positive observation is IMPOSSIBLE from the artifacts -- if an
        abort had cut a level-completing action, that action would simply be
        absent and the level would not have completed. FLAGGED as a tension.
      * what IS checkable is whether the invariant is IN FORCE: the guarantee
        (verified 0/0/0 in replay on all three recorded runs) holds only under
        the sealed settings mode=noop, abort_revisit=0, confirm>=2. If the
        canary reports anything else, the run is not the sealed arm and the
        invariant that makes rule 4 safe was not installed.
      * plus an artifact-consistency read: per game, the viewer's recorded
        level completions must equal benchmark levels_completed.
    """
    settings = {}
    if canary is not None:
        settings = dict(mode=canary["mode"], confirm=canary["confirm"],
                        abort_revisit=canary["abort_revisit"])
    breached = []
    if canary is not None:
        if canary["mode"] != SEALED_MEMO_MODE:
            breached.append(f"mode={canary['mode']} (sealed {SEALED_MEMO_MODE})")
        if canary["abort_revisit"] != SEALED_ABORT_REVISIT:
            breached.append(f"abort_revisit={canary['abort_revisit']} "
                            f"(sealed {SEALED_ABORT_REVISIT})")
        if canary["confirm"] < SEALED_CONFIRM_FLOOR:
            breached.append(f"confirm={canary['confirm']} "
                            f"(sealed floor {SEALED_CONFIRM_FLOOR})")

    # artifact consistency: viewer level completions vs benchmark lc
    mismatches, checked = [], 0
    art = arm_run / "artifacts"
    if rows and art.is_dir():
        for r in rows:
            p = art / f"{r['game_id']}_p0_events.jsonl"
            if not p.is_file():
                continue
            n = 0
            with p.open(encoding="utf-8") as fh:
                for line in fh:
                    if '"level_completed": true' in line or '"level_completed":true' in line:
                        n += 1
            checked += 1
            if n != r["levels_completed"]:
                mismatches.append(dict(game=r["game_id"], viewer_lc=n,
                                       benchmark_lc=r["levels_completed"]))
    return dict(
        rule=4, name="any level-completing action is declined or aborted",
        fired=bool(breached), evaluable=(canary is not None),
        detail=("; ".join(breached) if breached else
                ("sealed settings in force (mode=noop, abort_revisit=0, "
                 "confirm>=2); replay verified 0/0/0 on all three recorded runs"
                 if canary else f"UNEVALUABLE: no canary line ({rows_err or ''})")),
        sealed_settings_in_force=(not breached) if canary is not None else None,
        observed_settings=settings,
        directly_observable=False,
        observability_note=(
            "FLAGGED TENSION (prereg sec6.4): a violation cannot be observed "
            "post hoc -- a declined/aborted action leaves no record and the "
            "level simply does not complete. What is checked here is that the "
            "settings under which the 0/0/0 replay guarantee holds were "
            "actually in force, plus artifact consistency."),
        viewer_vs_benchmark_lc_games_checked=checked,
        viewer_vs_benchmark_lc_mismatches=mismatches)


# --------------------------------------------------------------------------- #
# verdict
# --------------------------------------------------------------------------- #
def resolve_verdict(canaries: dict, rules: list[dict], m0: dict,
                    evidence_errors: list[str]) -> dict:
    """KILL > DISCARD > NO-PROMOTE > PROMOTE.

    KILL first because prereg sec6 kills are unconditional ("module reverted,
    nothing promoted") and sec6.1 kills on K-P0/K-P3 explicitly, even though
    sec5 also calls those runs discard-grade -- both facts are reported.
    DISCARD covers everything the scorer could not read, which is the state the
    animation post-mortem says must never be silently rendered as a result."""
    kills = [f"kill rule {r['rule']}: {r['name']} -- {r['detail']}"
             for r in rules if r["fired"]]
    unevaluable = [f"kill rule {r['rule']} UNEVALUABLE: {r['detail']}"
                   for r in rules if not r["evaluable"] and not r["fired"]]
    hard_fail = [f"{k} {canaries[k]['status']}" for k in HARD_CANARIES
                 if canaries[k]["status"] != "PASS"]
    gate_fail = [f"{k} {canaries[k]['status']}" for k in READING_GATES
                 if canaries[k]["status"] != "PASS"]
    discard = list(evidence_errors) + unevaluable
    discard += [f"hard canary {x} (prereg sec5: discard-grade, nothing may be read)"
                for x in hard_fail]
    if m0.get("above_band"):
        discard.append(f"M0={m0['rate']*100:.2f}% ABOVE the 30% band -- prereg "
                       "sec3 requires inspection before any reading (not a kill)")
    if kills:
        verdict, why = "KILL", "; ".join(kills)
    elif discard:
        verdict, why = "DISCARD", "; ".join(discard)
    elif gate_fail:
        verdict, why = "NO-PROMOTE", ("reading gate(s) " + ", ".join(gate_fail)
                                      + " -- the mechanism did not deliver; M1/M2 "
                                      "may not be read as evidence of anything")
    else:
        verdict, why = "PROMOTE", ("hard canaries pass, no kill rule fired, all "
                                   "reading gates pass (prereg sec7)")
    return dict(verdict=verdict, why=why, kill_reasons=kills,
                discard_reasons=discard, reading_gate_failures=gate_fail,
                hard_canary_failures=hard_fail,
                discard_grade=bool(discard) or bool(hard_fail),
                readable=(verdict in ("PROMOTE", "NO-PROMOTE")),
                m2_note="M2 below x1.019 is NOT a kill rule (prereg sec6).")


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--family", action="append", default=None,
                    help="comparator family run dir (repeatable); defaults to "
                         "the two sealed continuation-v1 runs")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    run = (REPO / args.run) if not Path(args.run).is_absolute() else Path(args.run)
    fam_dirs = [Path(f) if Path(f).is_absolute() else REPO / f
                for f in (args.family or FAMILY_RUNS)]

    def out_print(*a):
        if not args.quiet:
            print(*a)

    log, log_meta = read_log(run)
    canary, canary_err = parse_canary(log)
    events = parse_events(log)
    rows, rows_err = bench_rows(run)
    fam_dup = family_dup(fam_dirs, use_cache=not args.no_cache)
    fam_lcs = family_lc(fam_dirs)

    evidence_errors = [x for x in (
        log_meta.get("error") and f"LOG: {log_meta['error']}",
        canary_err and f"CANARY: {canary_err}",
        rows_err and f"BENCHMARK: {rows_err}",
        fam_dup.get("error") and f"FAMILY(dup): {fam_dup['error']}",
        fam_lcs.get("error") and f"FAMILY(lc): {fam_lcs['error']}",
    ) if x]

    rep: dict = {"run": str(run), "log": log_meta, "canary": canary,
                 "canary_error": canary_err, "benchmark_error": rows_err,
                 "n_event_lines": len(events),
                 "games_with_events": sorted({e["game"] for e in events}),
                 "evidence_errors": evidence_errors,
                 "family_dup": fam_dup, "family_lc": fam_lcs}

    canaries = run_canaries(log, log_meta, events, canary, canary_err, fam_dup)
    rep["canaries"] = canaries
    m0 = m0_from(canary)
    rep["M0"] = dict(primary=True,
                     definition="saved/requested, saved=declined+aborted, "
                                "requested=saved+executed (prereg sec3 M0)",
                     band_inclusive=True,
                     sealed_replay={k: f"{v[0]}/{v[1]} = {v[0]/v[1]*100:.1f}%"
                                    for k, v in SEALED_M0_REPLAY.items()}, **m0)

    out_print("=" * 78)
    out_print(f"P1 SCORER  run={run}")
    out_print(f"  log: {log_meta.get('path')}  format={log_meta.get('source')}"
              f"  chars={log_meta.get('chars')}")
    if evidence_errors:
        out_print("  EVIDENCE ERRORS: " + " | ".join(evidence_errors))
    out_print("=" * 78)
    out_print("CANARIES (prereg sec5)  [K-P0..K-P3 HARD, K-P4..K-P6 reading gates]")
    for k in HARD_CANARIES + READING_GATES:
        c = canaries[k]
        extra = ""
        if k == "K-P1":
            extra = f"  ({c['event_lines']} lines, {c['distinct_games']} games)"
        elif k == "K-P3":
            extra = f"  (errors={c['errors']})"
        elif k == "K-P4":
            extra = (f"  ({c['saved']}/{c['requested']} = "
                     f"{c['rate']*100:.2f}%)" if c["rate"] is not None else "  (n/a)")
        elif k == "K-P5":
            extra = f"  ({c['ambiguous_games']})"
        elif k == "K-P6":
            extra = (f"  (arm {c['arm_dup_rate']} vs family "
                     f"{c['family_dup_rate_all_actions']})")
        out_print(f"  {k} {c['status']:9s} {c['name']}{extra}")
        if c.get("error"):
            out_print(f"        error: {c['error']}")

    out_print("=" * 78)
    out_print("M0 PRIMARY (mechanism delivery) -- the ONLY endpoint readable as evidence")
    if m0.get("rate") is None:
        out_print(f"  UNDEFINED -- {m0.get('error') or 'no canary'}")
    else:
        out_print(f"  saved/requested = {m0['saved']}/{m0['requested']} = "
                  f"{m0['rate']*100:.2f}%   band [3%,30%] inclusive   "
                  f"replayed 5.9% / 20.0% / 17.6%")

    out_print("=" * 78)
    out_print("MECH-C (diagnostic, NOT pre-registered)")
    mc = mechanism_c(run, log)
    rep["mechanism_c"] = mc
    cov = mc["block_coverage"]
    out_print(f"  source={mc['source']}  games {mc['games']}  turns {mc['turns_seen']}"
              f"  turns carrying the block {mc['turns_with_block']} "
              f"(coverage {'n/a' if cov is None else f'{cov*100:.1f}%'})")
    out_print(f"  blocks naming untried {mc['blocks_with_untried']}, dead "
              f"{mc['blocks_with_dead']}, latent {mc['blocks_flagging_latent']}; "
              f"size mean {mc['block_chars_mean']} / max {mc['block_chars_max']} "
              f"(sealed bound 900)")
    dr = mc["dead_reissue_rate"]
    out_print(f"  dead-reissue {mc['dead_reissued']}/{mc['actions_after_dead_turns']}"
              f" ({'n/a' if dr is None else f'{dr*100:.2f}%'})"
              f"  first half {mc['first_half_reissue']} vs second "
              f"{mc['second_half_reissue']}")

    out_print("=" * 78)
    out_print("BEHAVIOUR / M3 (arm's own traces, diagnosis definitions)")
    beh = behaviour(run)
    rep["behaviour"] = beh
    rep["M3"] = dict(
        arm_dup_rate_canary=(canary["dup_rate"] if canary else None),
        arm_dup_rate_traces=beh.get("dup_rate"),
        arm_dup_rate_traces_cleared=beh.get("dup_rate_cleared"),
        family_dup_rate_all_actions=fam_dup.get("pooled_dup_rate_all_actions"),
        family_dup_rate_cleared_levels=fam_dup.get("pooled_dup_rate_cleared_levels"),
        sealed_replay_expectation="6.9% / 3.4% / 4.1% (as-run 10.5% / 4.9% / 5.0%)")
    if beh:
        out_print(f"  dup {beh['dup_rate']*100:.2f}%   blind-tail "
                  f"{beh['blind_rate']*100:.2f}%   actions/step "
                  f"{beh['actions_per_step']:.2f}")
    else:
        out_print("  no parseable artifacts/*_events.jsonl in the arm pull")

    out_print("=" * 78)
    out_print("M1 (DESCRIPTIVE ONLY -- family m=2, NOT SCREENABLE) and M2")
    lc = sum(r["levels_completed"] for r in rows) if rows else None
    rep["M1"] = {"arm_lc_total": lc, "n_games": len(rows), "screenable": False,
                 "error": rows_err,
                 "family_lc_totals": [v["lc_total"] for v in fam_lcs["runs"].values()],
                 "note": "family duck-harness-kaggle-continuation-v1 has m=2; "
                         "SCREEN_PROTOCOL sec1 P2 -> NOT SCREENABLE. This may "
                         "NOT be reported as non-harm."}
    s = score_bench(rows)
    rep["M2"] = {"arm_score": s, "sealed_multiplier_expectation": SEALED_MULTIPLIER,
                 "error": rows_err,
                 "not_a_kill_rule": "M2 below x1.019 is explicitly NOT a kill "
                                    "rule (prereg sec6)"}
    out_print(f"  arm levels_completed total = {lc} over {len(rows)} games "
              f"(family lc totals 10, 16; animation composition reached 17)")
    out_print(f"  arm local-25 RHAE score    = "
              f"{'n/a' if s is None else f'{s:.4f}'}")
    out_print("  M1 is DESCRIPTIVE ONLY. M2 must NOT be read against x1.10, and "
              "M2 < x1.019 is NOT a kill.")

    out_print("=" * 78)
    out_print("KILL RULES (prereg sec6)")
    rules = kill_rules(canaries, canary, m0, rows, rows_err, run, fam_lcs)
    rep["kill_rules"] = rules
    for r in rules:
        state = "FIRED" if r["fired"] else ("ok" if r["evaluable"] else "UNEVALUABLE")
        out_print(f"  {r['rule']}. {state:11s} {r['name']}")
        out_print(f"       {r['detail']}")

    verdict = resolve_verdict(canaries, rules, m0, evidence_errors)
    rep["verdict"] = verdict
    rep["kills"] = verdict["kill_reasons"]      # back-compat
    out_print("=" * 78)
    out_print(f"  VERDICT: {verdict['verdict']}")
    out_print(f"  {verdict['why']}")

    out = Path(args.out) if args.out else run / "p1_score.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    out_print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
