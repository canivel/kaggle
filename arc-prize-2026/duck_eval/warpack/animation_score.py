"""Animation-awareness scorer -- reads a pulled `arc3-duck-animation-eval` run and
emits the prereg's canaries + M0/M1/M2/M3 exactly as sealed in
``learnings/war_room/animation_prereg_2026-08-11.md`` (sec3 canaries, sec4
metrics, sec4.1 seal arithmetic, sec5 kill rules) and
``duck_eval/SCREEN_PROTOCOL.md`` (P1/P2/P3, sec4 incl. sec4.6 power honesty).

NOTHING here is reinterpreted and no endpoint is added. In particular:

  * CANARIES GATE EVERYTHING. K-A0/K-A1/K-A2 fail => **VOID** (the arm did not
    run / did not engage) -- explicitly NOT a FAIL. K-A3/K-A4 fail => **KILL**.
  * M2 is reported FIRST among the metrics and is flagged as the **DECIDING**
    metric of this run, on an EXTERNAL PRE-RESULT PRIOR filed after the seal
    (Kaggle discussion 734369, Jakob Bruggen / Helmut AGI, 2026-08-11 07:55Z;
    `learnings/sweeps/discussion_sweep_2026-08-11.md` sec1.1): the author of the
    feature we ported publishes an efficacy NULL (+1.4% mean, p = 0.92) and a
    harm mechanism -- +17% tokens/action, paid one-for-one in moves under a
    wall-clock cap. M0 remains the PRIMARY pre-registered mechanism endpoint and
    is untouched; the prereg is NOT amended. K-A3 keeps its sealed threshold.
  * M1 is DESCRIPTIVE ONLY. The comparator family
    ``duck-harness-kaggle-continuation-v1`` has m = 2, so the arm is
    **NOT SCREENABLE (SCREEN_PROTOCOL sec1 P2)** and the only legal M1 verdict
    string is "uninformative in both directions". No PASS/FAIL is emitted on M1
    in either direction, ever. The K3'' line and the 80%-power floor are printed
    as ADVISORY and are re-derived here from sigma-hat / C(m) rather than
    hardcoded (the sealed values are asserted against the derivation).

The parser is derived from the EMITTER, ``duck_eval/warpack/_kaggle_dataset/
animation_patch.py``: ``_emit_event`` (greppable ``ANIMATION `` stdout line +
best-effort per-game jsonl sidecar) and ``canary_report`` (the single
``ANIMATION CANARY `` line). Field order/spelling below mirrors those prints.

Usage (pull time):

    uv run python duck_eval/warpack/animation_score.py \
        --pull runs/kernel_pulls/animation_v1

Optional: ``--log`` (defaults to the pull dir's non-vllm ``*.log``),
``--baseline`` (repeatable; defaults to the two sealed continuation-v1 runs),
``--date`` (defaults to today), ``--out-dir`` (defaults to ``runs/animation``).
Writes ``runs/animation/score_<date>.json`` and ``score_<date>.md``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import math
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

# --------------------------------------------------------------------------- #
# sealed constants (prereg sec4.1 + SCREEN_PROTOCOL sec1-2)
# --------------------------------------------------------------------------- #
BASELINE_FAMILY = "duck-harness-kaggle-continuation-v1"
BASELINE_DIRS_DEFAULT = (
    REPO / "runs" / "kernel_pulls" / "w0_eval_s1",
    REPO / "runs" / "kernel_pulls" / "w0_cont_eval",
)
AUDIT_JSON = REPO / "runs" / "animation" / "frame_audit.json"

# prereg sec3 K-A2 / animation_patch._AUDIT_TYPE1_GAMES
AUDIT_TYPE1_GAMES = ("ft09", "cd82", "sc25", "ls20")

# animation_patch._TOKENS_PER_SUMMARY (mirrored; used only if the canary line
# is missing tokens_est).
TOKENS_PER_SUMMARY = 45
K_A3_BOUND = 0.01          # prereg sec3 K-A3: tokens_est/total_tokens < 1%
K_A1_MIN_GAMES = 5         # prereg sec3 K-A1: event lines on >= 5 distinct games

# SCREEN_PROTOCOL sec2 C(m) -- m=1,2 "for advisory arithmetic only" (they fail P2).
C_M = {1: 2.33, 2: 2.10, 3: 2.02, 4: 1.98, 5: 1.96, 6: 1.94}
# SCREEN_PROTOCOL sec1 P3 pooled build-rail estimate: family -> (SS, df)
SIGMA_POOL = {
    "duck-harness-kaggle-warpack-v1": (0.071467, 2),
    "duck-harness-kaggle": (0.007467, 2),
    "duck-harness-kaggle-continuation-v1": (0.028800, 1),
    "duck-harness-kaggle-sentinel-v2": (0.012800, 1),
}
SEALED_SIGMA = 0.14174
SEALED_DF = 6
SEALED_K3PP_M2 = -0.2977      # prereg sec4.1.4
SEALED_FLOOR_M2 = 0.4437      # prereg sec4.1.5 (lc/game)
SEALED_FLOOR_LEVELS_M2 = 11.09
Z80 = 0.8416
N_GAMES = 25

M1_LEGAL_VERDICT = "uninformative in both directions"
NOT_SCREENABLE = "NOT SCREENABLE (SCREEN_PROTOCOL §1 P2)"

# EXTERNAL PRE-RESULT PRIOR (filed alongside the seal, NOT an amendment to it).
# Source: Kaggle discussion topic 734369 "Write Up: Taaf Anim Agent",
# Jakob Bruggen (Helmut AGI, #8 @ 1.61), 2026-08-11 07:55Z, swept in
# learnings/sweeps/discussion_sweep_2026-08-11.md sec1.1 -- the author of the
# feature we ported, publishing his own results on it.
EXTERNAL_PRIOR = dict(
    source="Kaggle discussion 734369 'Write Up: Taaf Anim Agent'",
    author="Jakob Bruggen (Helmut AGI, #8 @ 1.61)",
    posted="2026-08-11T07:55Z",
    swept_in="learnings/sweeps/discussion_sweep_2026-08-11.md §1.1",
    efficacy="NULL: +1.4% mean score, p = 0.92 over 6 games x 4 passes",
    harm_mechanism=("'Tokens are the real currency, not actions.' His animation "
                    "arm went 384 -> 449 tokens/action (+17%), and 'in every "
                    "single game, more tokens per action meant fewer actions'. "
                    "Every run in both arms hit the 132-min wall-clock cap."),
    tool_use="animation() called in 21/24 runs; 2 of 181 calls landed on an informative animation",
    why_not_falsifying=("his flag carries all three stages; ours is stage 1 only "
                        "(fixed ~45-token scalar summary, emitted only on animated "
                        "actions). The retrieval tool and the proactive hint -- "
                        "where his token inflation lives -- are pre-registered as "
                        "explicitly OUT (prereg §2.1/§2.2). Locally measured token "
                        "fraction: 0.00243."),
    consequence=("M2 (tokens/action, tokens/lc, wall-clock/action) is the metric "
                 "that decides whether this arm can ever pay. M0 remains the "
                 "PRIMARY pre-registered mechanism endpoint; a good M0 is "
                 "mechanism DELIVERY, never an efficacy claim -- the best "
                 "available efficacy evidence is his, and it is null."),
    k_a3_note=("K-A3 keeps its sealed threshold (<1% => else KILL) unchanged; "
               "this prior is external pre-result justification for treating a "
               "breach as fatal rather than advisory."),
)
EXTERNAL_TOK_PER_ACTION_DELTA_PCT = 17.0   # his arm: 384 -> 449 tokens/action
EXTERNAL_TOK_PER_ACTION = (384, 449)
# Our own rail is wall-clock bound: the scored rail projects 32,267 s against a
# 32,400 s cap (R24 §3.1: all 25 games ended at 7920.2-7939.9 s against
# max_actions_per_game=None), so his tokens -> fewer actions coupling transfers.
RAIL_WALLCLOCK_PROJECTED_S = 32267
RAIL_WALLCLOCK_CAP_S = 32400

# prereg sec4.1.1 -- the arm must carry the continuation-v1 banner and NONE of
# these.
FORBIDDEN_BANNER_TOKENS = ("warpack:", "LEDGER", "SENTINEL", "COMPACTION ")
CONTINUATION_BANNER_TOKEN = "continuation v1:"

# --------------------------------------------------------------------------- #
# emitter-derived parsers (animation_patch._emit_event / canary_report)
# --------------------------------------------------------------------------- #
EVENT_RE = re.compile(
    r"ANIMATION v=(?P<v>\S+) kind=(?P<kind>\S+) game=(?P<game>\S+) "
    r"action=(?P<action>.*?) frames=(?P<frames>\S+) unique=(?P<unique>\S+) "
    r"board_unchanged=(?P<board_unchanged>[01]) "
    r"transient_cells=(?P<transient_cells>\S+) "
    r"bbox=(?P<bbox>\[[^\]]*\]|None) "
    r"run_actions=(?P<run_actions>\d+) run_multi=(?P<run_multi>\d+) "
    r"run_invisible=(?P<run_invisible>\d+)"
)
CANARY_RE = re.compile(
    r"ANIMATION CANARY v=(?P<v>\S+) version=(?P<version>\S+) "
    r"actions=(?P<actions>\d+) multi=(?P<multi>\d+) invisible=(?P<invisible>\d+) "
    r"summaries=(?P<summaries>\d+) errors=(?P<errors>\d+) "
    r"games_with_events=(?P<games_with_events>\d+) "
    r"games_with_invisible=(?P<games_with_invisible>\d+) "
    r"audit_type1_engaged=(?P<audit_type1_engaged>\S+) "
    r"tokens_est=(?P<tokens_est>\d+) "
    r"token_fraction=(?P<token_fraction>\S*)"
)
BANNER_RE = re.compile(r"animation (?P<version>v\S+): ACTIVE \((?P<seams>\d+) seams patched\)")
GRAFT_RE = re.compile(r"animation \S+: graft applied from (?P<dir>\S+) \(applied=(?P<applied>\w+)\)")
PATCH_FAILED_TOKEN = "animation: PATCH FAILED"
FLAG_STAMP_TOKEN = "ANIMATION_AWARE=1"
CANARY_UNAVAILABLE_TOKEN = "ANIMATION CANARY unavailable"


def _short(game: str) -> str:
    """`ft09-0d8bbf25` -> `ft09` (the audit's key). Emitter labels are engine
    game ids (`env.environment_info.game_id`), which carry the hash suffix."""
    return game.split("-")[0]


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #
def load_log_text(path: Path) -> tuple[str, str]:
    """Kaggle build logs are a JSON array of {stream_name,time,data} records.
    Concatenate every `data` (order preserved) so a print that arrives in
    several chunks is still one searchable blob. Degrades to raw text."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        recs = json.loads(raw)
        if isinstance(recs, list):
            return "".join(str(r.get("data", "")) for r in recs if isinstance(r, dict)), "json-array"
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


def find_log(pull: Path) -> Path | None:
    cands = [p for p in sorted(pull.glob("*.log")) if "vllm" not in p.name.lower()]
    if not cands:
        return None
    named = [p for p in cands if "animation" in p.name.lower()]
    return (named or cands)[0]


def load_bench(pull: Path) -> dict[str, Any] | None:
    p = pull / "benchmark.json"
    if not p.is_file():
        return None
    bench = json.loads(p.read_text(encoding="utf-8"))
    games: dict[str, dict[str, Any]] = {}
    for r in bench.get("game_runs", []):
        gid = r["game_id"]
        m = re.search(r"tokens=(\d+)", r.get("solver_note") or "")
        games[_short(gid)] = dict(
            game_id=gid,
            lc=r.get("levels_completed", 0),
            actions=len(r.get("history") or []),
            gen_tokens=int(m.group(1)) if m else sum(
                int(h.get("generated_tokens") or 0) for h in (r.get("history") or [])),
            wallclock_s=r.get("final_wallclock_seconds"),
            state=r.get("state"),
        )
    return dict(label=bench.get("label"), path=str(p), games=games,
                lc_total=sum(g["lc"] for g in games.values()),
                actions_total=sum(g["actions"] for g in games.values()),
                gen_tokens_total=sum(g["gen_tokens"] for g in games.values()),
                wallclock_total=sum(g["wallclock_s"] or 0.0 for g in games.values()))


def load_viewer_actions(pull: Path) -> dict[str, list[dict[str, Any]]]:
    """Vanilla harness per-game viewer events. `type=action` rows carry
    `action_display` + `board_changed` and are 1:1 with benchmark history --
    the only on-disk ground truth for M3 (no-op = board_changed False)."""
    out: dict[str, list[dict[str, Any]]] = {}
    for f in sorted((pull / "artifacts").glob("*_events.jsonl")):
        if "_animation_events" in f.name or "_compaction_events" in f.name:
            continue
        g = _short(f.name)
        rows: list[dict[str, Any]] = []
        with f.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if rec.get("type") == "action" and "board_changed" in rec:
                    rows.append(rec)
        if rows:
            out.setdefault(g, []).extend(rows)
    return out


def load_sidecars(pull: Path) -> dict[str, list[dict[str, Any]]]:
    """animation_patch._emit_event's best-effort per-game jsonl sidecars.
    NOTE (defect, see the report): the emitter looks for
    `session._animation_state_path`, which is only ever set on the ToolAgent,
    so these files are expected to be ABSENT. K-A1 is decided from the stdout
    event lines (prereg sec3), not from these."""
    out: dict[str, list[dict[str, Any]]] = {}
    for f in sorted(pull.rglob("*_animation_events.jsonl")):
        rows = []
        with f.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except Exception:  # noqa: BLE001
                        pass
        if rows:
            out[_short(f.name)] = rows
    return out


VLLM_THROUGHPUT_RE = re.compile(
    r"INFO (?P<mo>\d\d)-(?P<dy>\d\d) (?P<h>\d\d):(?P<mi>\d\d):(?P<s>\d\d) .*?"
    r"Avg prompt throughput: (?P<prompt>[0-9.]+) tokens/s, "
    r"Avg generation throughput: (?P<gen>[0-9.]+) tokens/s")


def estimate_vllm_tokens(pull: Path, bench_gen_tokens: int | None) -> dict[str, Any]:
    """Integrate vLLM's periodic throughput samples to get the run's PROMPT
    token total, which `benchmark.json` does not record (`uncached_input_tokens`
    is 0 there). Self-validating: the same integration over the *generation*
    channel is compared against the benchmark's own generated-token total, so
    the estimate is only used when it reproduces a number we already know."""
    log = pull / "vllm-openai-server.log"
    if not log.is_file():
        return dict(available=False, reason="vllm-openai-server.log absent")
    prompt = gen = 0.0
    prev = None
    n = 0
    for line in log.open(encoding="utf-8", errors="replace"):
        m = VLLM_THROUGHPUT_RE.search(line)
        if not m:
            continue
        d = m.groupdict()
        t = _dt.datetime(2000, int(d["mo"]), int(d["dy"]), int(d["h"]), int(d["mi"]), int(d["s"]))
        dt = (t - prev).total_seconds() if prev else 10.0
        if dt <= 0 or dt > 120:          # log rollover / gap -> vLLM's 10 s default
            dt = 10.0
        prompt += float(d["prompt"]) * dt
        gen += float(d["gen"]) * dt
        prev = t
        n += 1
    if not n:
        return dict(available=False, reason="no throughput samples in the vllm log")
    ratio = (gen / bench_gen_tokens) if bench_gen_tokens else None
    ok = ratio is not None and 0.8 <= ratio <= 1.25
    return dict(available=True, samples=n, prompt_tokens_est=round(prompt),
                generated_tokens_est=round(gen),
                total_tokens_est=round(prompt + gen),
                validation_ratio_gen_est_over_benchmark=ratio,
                validated=ok,
                note=("integration of 'Avg prompt/generation throughput' samples; "
                      "validated by reproducing the benchmark's generated-token "
                      "total on the generation channel (accept 0.80-1.25x)"))


def load_audit() -> dict[str, Any]:
    if not AUDIT_JSON.is_file():
        return {}
    a = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    return dict(totals=a.get("totals", {}),
                games={g["game"]: g for g in a.get("games", [])},
                path=str(AUDIT_JSON))


# --------------------------------------------------------------------------- #
# canaries (prereg sec3) -- these gate EVERYTHING
# --------------------------------------------------------------------------- #
def _evidence(blob: str, token: str, width: int = 200) -> str | None:
    i = blob.find(token)
    if i < 0:
        return None
    start = blob.rfind("\n", 0, i) + 1
    end = blob.find("\n", i)
    line = blob[start:(end if end > 0 else min(len(blob), i + width))]
    return line.strip()[:400] or None


def parse_events(blob: str) -> list[dict[str, Any]]:
    events = []
    for m in EVENT_RE.finditer(blob):
        d = m.groupdict()
        events.append(dict(
            v=d["v"], kind=d["kind"], game=d["game"], game_short=_short(d["game"]),
            action=d["action"].strip(),
            frames=_int(d["frames"]), unique=_int(d["unique"]),
            board_unchanged=d["board_unchanged"] == "1",
            transient_cells=_int(d["transient_cells"]), bbox=d["bbox"],
            run_actions=int(d["run_actions"]), run_multi=int(d["run_multi"]),
            run_invisible=int(d["run_invisible"]),
        ))
    return events


def _int(s: str) -> int | None:
    try:
        return int(s)
    except Exception:  # noqa: BLE001
        return None


def parse_canary(blob: str) -> dict[str, Any] | None:
    m = CANARY_RE.search(blob)
    if not m:
        return None
    d = m.groupdict()
    engaged = [] if d["audit_type1_engaged"] == "NONE" else [
        x for x in d["audit_type1_engaged"].split(",") if x]
    frac = None
    if d["token_fraction"]:
        try:
            frac = float(d["token_fraction"])
        except Exception:  # noqa: BLE001
            frac = None
    return dict(
        v=d["v"], version=d["version"], actions=int(d["actions"]),
        multi=int(d["multi"]), invisible=int(d["invisible"]),
        summaries=int(d["summaries"]), errors=int(d["errors"]),
        games_with_events=int(d["games_with_events"]),
        games_with_invisible=int(d["games_with_invisible"]),
        audit_type1_engaged=engaged, tokens_est=int(d["tokens_est"]),
        token_fraction_reported=frac, raw=m.group(0),
    )


def run_canaries(blob: str, events: list[dict[str, Any]],
                 canary: dict[str, Any] | None, arm_bench: dict[str, Any] | None,
                 sidecars: dict[str, Any], pull_dir: Path | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}

    # ---- K-A0: banner + ANIMATION_AWARE=1 stamp; absent/PATCH FAILED => VOID.
    banner = BANNER_RE.search(blob)
    graft = GRAFT_RE.search(blob)
    patch_failed = PATCH_FAILED_TOKEN in blob
    stamp = FLAG_STAMP_TOKEN in blob
    label = (arm_bench or {}).get("label")
    ka0_ok = bool(banner) and stamp and not patch_failed
    out["K-A0"] = dict(
        name="banner + ANIMATION_AWARE=1 stamp",
        status="PASS" if ka0_ok else "FAIL",
        outcome_if_fail="VOID (ran VANILLA -- explicitly NOT a FAIL)",
        banner_present=bool(banner),
        banner_version=(banner.group("version") if banner else None),
        seams_patched=(int(banner.group("seams")) if banner else None),
        flag_stamp_present=stamp,
        patch_failed_line=patch_failed,
        label_stamped=bool(label and "-animation-" in label),
        label=label,
        evidence=[e for e in (
            _evidence(blob, "animation v") if banner else None,
            _evidence(blob, FLAG_STAMP_TOKEN),
            _evidence(blob, PATCH_FAILED_TOKEN) if patch_failed else None,
            (graft.group(0) if graft else None),
        ) if e],
    )

    # ---- K-A1: >=1 `ANIMATION ` event line on >= 5 distinct games.
    games_with_events = sorted({e["game_short"] for e in events})
    ka1_ok = len(events) >= 1 and len(games_with_events) >= K_A1_MIN_GAMES
    out["K-A1"] = dict(
        name=f">=1 ANIMATION event line on >={K_A1_MIN_GAMES} distinct games",
        status="PASS" if ka1_ok else "FAIL",
        outcome_if_fail="VOID",
        event_lines=len(events),
        distinct_games=len(games_with_events),
        games=games_with_events,
        sidecar_games=sorted(sidecars),
        sidecar_note=("per-game jsonl sidecars absent -- EXPECTED, see the "
                      "emitter defect note; K-A1 is decided on the stdout event "
                      "lines per prereg §3" if not sidecars else "sidecars present"),
        evidence=[events[0]["_raw"] if "_raw" in events[0] else _evidence(blob, "ANIMATION v=")] if events else [],
    )

    # ---- K-A2: nonzero `invisible` on >=1 of {ft09,cd82,sc25,ls20}.
    inv_by_game: dict[str, int] = {}
    for e in events:
        if e["board_unchanged"]:
            inv_by_game[e["game_short"]] = inv_by_game.get(e["game_short"], 0) + 1
    engaged_from_events = sorted(g for g in inv_by_game if g in AUDIT_TYPE1_GAMES)
    engaged_from_canary = sorted({_short(g) for g in (canary or {}).get("audit_type1_engaged", [])})
    engaged = sorted(set(engaged_from_events) | set(engaged_from_canary))
    ka2_ok = bool(engaged)
    out["K-A2"] = dict(
        name="nonzero invisible on >=1 of ft09/cd82/sc25/ls20",
        status="PASS" if ka2_ok else "FAIL",
        outcome_if_fail="VOID + audit method back under review",
        type1_games=list(AUDIT_TYPE1_GAMES),
        invisible_by_type1_game={g: inv_by_game.get(g, 0) for g in AUDIT_TYPE1_GAMES},
        engaged_from_event_lines=engaged_from_events,
        engaged_from_canary_line=engaged_from_canary,
        audit_method_under_review=not ka2_ok,
        evidence=([_evidence(blob, "audit_type1_engaged=")] if canary else [])
        + [e["action"] and f"{e['game']} action={e['action']} frames={e['frames']} "
           f"board_unchanged=1 transient_cells={e['transient_cells']}"
           for e in events if e["board_unchanged"] and e["game_short"] in AUDIT_TYPE1_GAMES][:1],
    )

    # ---- K-A3: animation_tokens_est / total_tokens < 1%  (else KILL).
    # DENOMINATOR AMBIGUITY (reported, not resolved by fiat -- see the analyzer
    # report): the prereg says "total_tokens" and the emitter's summary tokens
    # are PROMPT tokens, but benchmark.json only records GENERATED tokens
    # (uncached_input_tokens is 0). The two denominators straddle the 1% bound by
    # roughly 20x. Primary = the run's real total (prompt+generation, integrated
    # from the vLLM log and self-validated); the generated-only fraction is
    # carried as the conservative bound. If they straddle the bound the canary is
    # marked DISPUTED and must be escalated, not executed.
    tokens_est = (canary or {}).get("tokens_est")
    if tokens_est is None and events:
        tokens_est = len(events) * TOKENS_PER_SUMMARY
    gen_tokens = (arm_bench or {}).get("gen_tokens_total")
    vllm = estimate_vllm_tokens(pull_dir, gen_tokens) if pull_dir else dict(available=False)
    total_tokens = (vllm.get("total_tokens_est")
                    if vllm.get("available") and vllm.get("validated") else None)
    denom_src = "vllm prompt+generation integration (self-validated)"
    if total_tokens is None:
        total_tokens, denom_src = gen_tokens, (
            "benchmark.json sum of solver_note tokens= (GENERATED tokens only -- "
            "conservative: it can only over-fire this KILL rule, never under-fire)")
    frac = (tokens_est / total_tokens) if (tokens_est is not None and total_tokens) else None
    frac_gen = (tokens_est / gen_tokens) if (tokens_est is not None and gen_tokens) else None
    if frac is None:
        ka3_status, ka3_out = "UNDETERMINED", "VOID (cannot evaluate the bound)"
    elif frac < K_A3_BOUND:
        ka3_status, ka3_out = "PASS", ""
    else:
        ka3_status, ka3_out = "FAIL", "KILL"
    disputed = (frac is not None and frac_gen is not None
                and (frac < K_A3_BOUND) != (frac_gen < K_A3_BOUND))
    out["K-A3"] = dict(
        name="animation_tokens_est / total_tokens < 1%",
        status=ka3_status, outcome_if_fail="KILL (arm killed, module reverted)",
        outcome=ka3_out,
        animation_tokens_est=tokens_est,
        total_tokens=total_tokens,
        total_tokens_source=denom_src,
        generated_tokens=gen_tokens,
        vllm_token_estimate=vllm,
        token_fraction=frac,
        token_fraction_generated_only=frac_gen,
        denominator_disputed=disputed,
        denominator_note=(
            "DISPUTED: the prompt-inclusive and generated-only denominators fall "
            "on opposite sides of the 1% bound. The prereg does not define which "
            "'total_tokens' it means and the builder never passes one to "
            "canary_report(). ESCALATE -- do not execute the KILL/PASS on this leg "
            "alone." if disputed else
            "both candidate denominators agree on this leg"),
        bound=K_A3_BOUND,
        canary_line_token_fraction=(canary or {}).get("token_fraction_reported"),
        note=("canary_report() is invoked without total_tokens by the builder, so "
              "the log's token_fraction field is empty by construction; the "
              "fraction above is computed here from the run's own token total"),
        threshold_unchanged=True,
        external_justification=EXTERNAL_PRIOR["k_a3_note"],
        external_source=EXTERNAL_PRIOR["source"],
        see_also="M2.tokens_per_action_delta_pct (directly comparable to his +17%)",
        evidence=[(canary or {}).get("raw")] if canary else [],
    )

    # ---- K-A4: animation_errors == 0  (else KILL).
    errors = (canary or {}).get("errors")
    if errors is None:
        ka4_status, ka4_out = "UNDETERMINED", "VOID (cannot evaluate)"
    elif errors == 0:
        ka4_status, ka4_out = "PASS", ""
    else:
        ka4_status, ka4_out = "FAIL", "KILL"
    out["K-A4"] = dict(
        name="animation_errors == 0",
        status=ka4_status, outcome_if_fail="KILL (a perception patch that raises "
                                           "in the action path is not shippable)",
        outcome=ka4_out, errors=errors,
        evidence=[(canary or {}).get("raw")] if canary else [],
    )

    out["canary_line_present"] = canary is not None
    out["canary_unavailable_line"] = CANARY_UNAVAILABLE_TOKEN in blob
    return out


def verdict_from_canaries(canaries: dict[str, Any]) -> dict[str, Any]:
    """prereg sec5. K-A0/1/2 fail => VOID (NOT a FAIL). K-A3/K-A4 fail => KILL.

    Ordering note (the prereg does not order a simultaneous VOID+KILL): if K-A0
    failed the arm never ran, so the KILL rules are not evaluable and the run is
    VOID; otherwise a KILL condition outranks a VOID condition, because sec5
    says K-A3/K-A4 kill "regardless of M1" and a patch that throws is not
    shippable "even if it scores". Both reason lists are always reported."""
    void_reasons, kill_reasons = [], []
    for k in ("K-A0", "K-A1", "K-A2"):
        if canaries[k]["status"] != "PASS":
            void_reasons.append(f"{k} {canaries[k]['name']} -> {canaries[k]['status']}")
    for k in ("K-A3", "K-A4"):
        st = canaries[k]["status"]
        if st == "FAIL":
            kill_reasons.append(f"{k} {canaries[k]['name']} -> FAIL")
        elif st == "UNDETERMINED":
            void_reasons.append(f"{k} {canaries[k]['name']} -> UNDETERMINED "
                                "(no evaluable evidence)")
    if not canaries["canary_line_present"]:
        void_reasons.append("ANIMATION CANARY line absent from the log "
                            "(prereg §3: absent => the run did not test the arm)")
    if canaries["K-A0"]["status"] != "PASS":
        verdict = "VOID"
        why = "K-A0 failed: the arm did not run (VANILLA fallback). VOID is NOT a FAIL."
    elif kill_reasons:
        verdict = "KILL"
        why = "; ".join(kill_reasons)
    elif void_reasons:
        verdict = "VOID"
        why = "; ".join(void_reasons)
    else:
        verdict = "CANARIES CLEAR"
        why = ("mechanism engaged and is free + exception-clean; the arm's result "
               "is M0 + the canaries (prereg §4.1.6)")
    return dict(verdict=verdict, why=why, void_reasons=void_reasons,
                kill_reasons=kill_reasons,
                audit_method_under_review=canaries["K-A2"]["audit_method_under_review"],
                note="VOID != FAIL: rebuild, and do not record a verdict in either direction.")


# --------------------------------------------------------------------------- #
# M0 (PRIMARY -- mechanism)
# --------------------------------------------------------------------------- #
def compute_m0(events: list[dict[str, Any]], canary: dict[str, Any] | None,
               arm_bench: dict[str, Any] | None, audit: dict[str, Any]) -> dict[str, Any]:
    per_game_multi: dict[str, int] = {}
    per_game_inv: dict[str, int] = {}
    for e in events:
        g = e["game_short"]
        per_game_multi[g] = per_game_multi.get(g, 0) + 1
        if e["board_unchanged"]:
            per_game_inv[g] = per_game_inv.get(g, 0) + 1

    bench_games = (arm_bench or {}).get("games", {})
    games = sorted(set(bench_games) | set(per_game_multi) | set(audit.get("games", {})))

    rows = []
    for g in games:
        executed = bench_games.get(g, {}).get("actions")
        multi = per_game_multi.get(g, 0)
        inv = per_game_inv.get(g, 0)
        ag = audit.get("games", {}).get(g, {})
        comb = ag.get("combined", {})
        rec = ag.get("recorded", {})
        exp_type = ag.get("type")
        expectation = "nonzero" if g in AUDIT_TYPE1_GAMES else "~0"
        if g in AUDIT_TYPE1_GAMES:
            status = "MATCH" if inv > 0 else "MISS (type-1 game returned 0 invisible)"
        else:
            status = "MATCH" if inv == 0 else "SURPRISE (non-type-1 game returned invisible)"
        rows.append(dict(
            game=g, audit_type=exp_type, executed_actions=executed,
            multi_frame_actions=multi, invisible_actions=inv,
            invisible_rate=(inv / executed) if executed else None,
            multi_frame_rate=(multi / executed) if executed else None,
            expectation=expectation, expectation_status=status,
            audit_invisible_pct_combined=comb.get("invisible_pct_of_actions"),
            audit_multi_frame_pct_combined=comb.get("multi_frame_pct"),
            audit_invisible_pct_recorded=rec.get("invisible_pct_of_actions"),
            audit_invisible_combined=comb.get("invisible"),
            audit_actions_combined=comb.get("actions"),
        ))

    executed_total = (canary or {}).get("actions") or (arm_bench or {}).get("actions_total")
    inv_total = (canary or {}).get("invisible", sum(per_game_inv.values()))
    multi_total = (canary or {}).get("multi", sum(per_game_multi.values()))
    at = audit.get("totals", {})
    misses = [r["game"] for r in rows if r["expectation_status"].startswith("MISS")]
    surprises = [r["game"] for r in rows if r["expectation_status"].startswith("SURPRISE")]
    return dict(
        primary=True,
        definition="invisible_actions / executed_actions (prereg §4 M0)",
        executed_actions=executed_total,
        executed_actions_from_canary=(canary or {}).get("actions"),
        executed_actions_from_benchmark=(arm_bench or {}).get("actions_total"),
        invisible_actions=inv_total, multi_frame_actions=multi_total,
        invisible_rate=(inv_total / executed_total) if executed_total else None,
        multi_frame_rate=(multi_total / executed_total) if executed_total else None,
        event_line_invisible=sum(per_game_inv.values()),
        event_line_multi=sum(per_game_multi.values()),
        event_lines_consistent_with_canary=(
            None if canary is None else
            (sum(per_game_multi.values()) == canary["multi"]
             and sum(per_game_inv.values()) == canary["invisible"])),
        log_truncation_suspected=(
            None if canary is None else sum(per_game_multi.values()) < canary["multi"]),
        offline_audit_expectation=dict(
            source=audit.get("path"),
            games_multi_frame=at.get("games_multi_frame"),
            games_type1=at.get("games_type1"),
            actions=at.get("actions"), invisible=at.get("invisible"),
            invisible_pct_of_actions=at.get("invisible_pct_of_actions"),
            multi_frame_pct=at.get("multi_frame_pct"),
            registered="nonzero on ft09/cd82/sc25/ls20, ~0 elsewhere (prereg §4 M0)",
        ),
        expectation_misses=misses, expectation_surprises=surprises,
        expectation_check=("MET" if not misses and not surprises
                           else "DEVIATION (see misses/surprises)"),
        per_game=rows,
    )


# --------------------------------------------------------------------------- #
# M1 (SECONDARY -- DESCRIPTIVE ONLY, NOT A SCREEN)
# --------------------------------------------------------------------------- #
def derive_seal_arithmetic(m: int) -> dict[str, Any]:
    """Re-derive sigma-hat, the advisory K3'' line and the 80%-power floor from
    the protocol's own inputs, then assert they match the sealed values."""
    ss = sum(v[0] for v in SIGMA_POOL.values())
    df = sum(v[1] for v in SIGMA_POOL.values())
    sigma = math.sqrt(ss / df)
    c = C_M.get(m)
    line = -c * SEALED_SIGMA if c else None
    floor = (c * SEALED_SIGMA + Z80 * SEALED_SIGMA * math.sqrt(1 + 1 / m)) if c else None
    levels = floor * N_GAMES if floor else None
    checks = dict(
        sigma_rederived=round(sigma, 5),
        sigma_matches_sealed=abs(round(sigma, 5) - SEALED_SIGMA) < 5e-6,
        df_rederived=df, df_matches_sealed=(df == SEALED_DF),
        k3pp_line_rederived=(round(line, 4) if line is not None else None),
        k3pp_matches_sealed=(line is not None and abs(round(line, 4) - SEALED_K3PP_M2) < 1e-9
                             if m == 2 else None),
        floor_rederived=(round(floor, 4) if floor is not None else None),
        floor_matches_sealed=(floor is not None and abs(round(floor, 4) - SEALED_FLOOR_M2) < 1e-9
                              if m == 2 else None),
        levels_rederived=(round(levels, 2) if levels is not None else None),
        levels_matches_sealed=(levels is not None and abs(round(levels, 2) - SEALED_FLOOR_LEVELS_M2) < 1e-9
                               if m == 2 else None),
    )
    return dict(
        m=m, C_m=c, sigma_hat=SEALED_SIGMA, df=SEALED_DF,
        pooled_families={k: dict(SS=v[0], df=v[1]) for k, v in SIGMA_POOL.items()},
        k3pp_line_advisory=line, power80_floor_lc_per_game=floor,
        power80_floor_levels=levels, n_games=N_GAMES,
        advisory_label=("ADVISORY ONLY -- C(1) and C(2) are listed in "
                        "SCREEN_PROTOCOL §2 'for advisory arithmetic only'; C(2) "
                        "was never measured (interpolated between m=1 2.0% and "
                        "m=3 4.4% type-I). Not a gate."),
        checks=checks,
        seal_arithmetic_match=all(v for k, v in checks.items()
                                  if k.endswith("_matches_sealed") and v is not None),
    )


def compute_m1(arm_bench: dict[str, Any] | None,
               baselines: list[dict[str, Any]]) -> dict[str, Any]:
    m = len(baselines)
    seal = derive_seal_arithmetic(m)
    legal_labels = [b.get("label") == BASELINE_FAMILY for b in baselines]
    out: dict[str, Any] = dict(
        descriptive_only=True,
        screenable=False if m < 3 else None,
        screenable_statement=NOT_SCREENABLE if m < 3 else "screenable (m>=3)",
        verdict=M1_LEGAL_VERDICT,
        verdict_note=("prereg §4.1.6 / SCREEN_PROTOCOL §4.6: power at m=2 is far "
                      "below 50%; this run is an exploratory mechanism probe, not "
                      "a screen. No PASS may be reported as non-harm; no FAIL may "
                      "be reported as harm. The ONLY legal M1 verdict string is "
                      f"'{M1_LEGAL_VERDICT}'."),
        family=BASELINE_FAMILY, m=m,
        baselines=[dict(path=b.get("path"), label=b.get("label"),
                        label_matches_family=(b.get("label") == BASELINE_FAMILY),
                        lc_total=b.get("lc_total")) for b in baselines],
        all_baseline_labels_match=all(legal_labels) if baselines else False,
        seal_arithmetic=seal,
    )
    if not arm_bench or not baselines:
        out["computed"] = False
        out["reason"] = "arm benchmark.json and/or baseline runs unavailable"
        return out

    arm_games = arm_bench["games"]
    common = sorted(set(arm_games) & set.intersection(*[set(b["games"]) for b in baselines]))
    out["n_games_paired"] = len(common)
    out["game_set_identical"] = (len(common) == len(arm_games)
                                 == min(len(b["games"]) for b in baselines))
    per_game = []
    for g in common:
        base_lcs = [b["games"][g]["lc"] for b in baselines]
        base_mean = sum(base_lcs) / len(base_lcs)
        per_game.append(dict(game=g, arm_lc=arm_games[g]["lc"], baseline_lcs=base_lcs,
                             baseline_mean_lc=base_mean,
                             dlc=arm_games[g]["lc"] - base_mean))
    deltas = [r["dlc"] for r in per_game]
    mean_dlc = sum(deltas) / len(deltas) if deltas else None
    nz = [d for d in deltas if d != 0]
    p_two = None
    try:
        sys.path.insert(0, str(REPO / "scripts"))
        from phase1_gate import signflip_p_exact  # noqa: PLC0415
        if nz:
            p_two = min(1.0, 2 * signflip_p_exact(nz, abs(sum(nz)))[0])
    except Exception as exc:  # noqa: BLE001 - descriptive statistic only
        out["signflip_error"] = repr(exc)
    out.update(
        computed=True,
        arm_lc_total=arm_bench["lc_total"],
        baseline_lc_totals=[b["lc_total"] for b in baselines],
        baseline_family_mean_levels=sum(b["lc_total"] for b in baselines) / len(baselines),
        mean_dlc=mean_dlc,
        sd_games=(statistics.stdev(deltas) if len(deltas) > 1 else None),
        wins=sum(1 for d in deltas if d > 0), losses=sum(1 for d in deltas if d < 0),
        nonzero_games=len(nz),
        signflip_p_exact_two_sided=p_two,
        signflip_note="descriptive significance only (SCREEN_PROTOCOL §3: not a gate)",
        vs_advisory_k3pp_line=(
            None if mean_dlc is None or seal["k3pp_line_advisory"] is None
            else dict(line=seal["k3pp_line_advisory"],
                      mean_dlc_above_line=mean_dlc >= seal["k3pp_line_advisory"],
                      label="ADVISORY ONLY -- may not be reported as PASS or FAIL")),
        per_game=per_game,
    )
    # live re-check of the continuation family's own SS contribution to sigma-hat
    if all(b.get("label") == BASELINE_FAMILY for b in baselines) and len(baselines) == 2:
        means = [b["lc_total"] / N_GAMES for b in baselines]
        mu = sum(means) / len(means)
        ss = sum((x - mu) ** 2 for x in means)
        out["family_ss_recheck"] = dict(
            per_game_means=means, ss=round(ss, 6), df=len(means) - 1,
            sealed_ss=SIGMA_POOL[BASELINE_FAMILY][0],
            matches=abs(ss - SIGMA_POOL[BASELINE_FAMILY][0]) < 1e-6)
    return out


# --------------------------------------------------------------------------- #
# M2 (DECIDING -- external prior 734369) -- tokens/action, tokens/lc,
# wall-clock/action, and the tokens -> fewer-actions coupling
# --------------------------------------------------------------------------- #
def _m2_row(b: dict[str, Any]) -> dict[str, Any]:
    act, lc = b["actions_total"], b["lc_total"]
    tok, wall = b["gen_tokens_total"], b["wallclock_total"]
    n = len(b["games"]) or 1
    return dict(label=b.get("label"), path=b.get("path"), actions=act, lc=lc,
                gen_tokens=tok, wallclock_s=wall, n_games=len(b["games"]),
                actions_per_game=act / n, wallclock_per_game=wall / n,
                tokens_per_action=(tok / act) if act else None,
                tokens_per_lc=(tok / lc) if lc else None,
                wallclock_per_action=(wall / act) if act else None)


def compute_m2(arm_bench: dict[str, Any] | None,
               baselines: list[dict[str, Any]]) -> dict[str, Any]:
    head = dict(
        deciding_metric=True,
        deciding_because=EXTERNAL_PRIOR["consequence"],
        external_prior=EXTERNAL_PRIOR,
        external_reference=dict(
            tokens_per_action_before=EXTERNAL_TOK_PER_ACTION[0],
            tokens_per_action_after=EXTERNAL_TOK_PER_ACTION[1],
            delta_pct=EXTERNAL_TOK_PER_ACTION_DELTA_PCT,
            label="his stage-1+2+3 arm; ours is stage 1 only"),
    )
    if not arm_bench or not baselines:
        head.update(computed=False,
                    reason="arm benchmark.json and/or baseline runs unavailable")
        return head
    arm = _m2_row(arm_bench)
    rows = [_m2_row(b) for b in baselines]
    n_fam_games = sum(r["n_games"] for r in rows) or 1
    fam = dict(
        label=f"{BASELINE_FAMILY} (m={len(baselines)}, pooled)",
        actions=sum(r["actions"] for r in rows), lc=sum(r["lc"] for r in rows),
        gen_tokens=sum(r["gen_tokens"] for r in rows),
        wallclock_s=sum(r["wallclock_s"] for r in rows),
        n_games=n_fam_games)
    fam.update(actions_per_game=fam["actions"] / n_fam_games,
               wallclock_per_game=fam["wallclock_s"] / n_fam_games,
               tokens_per_action=fam["gen_tokens"] / fam["actions"] if fam["actions"] else None,
               tokens_per_lc=fam["gen_tokens"] / fam["lc"] if fam["lc"] else None,
               wallclock_per_action=fam["wallclock_s"] / fam["actions"] if fam["actions"] else None)
    ratios = {k: (arm[k] / fam[k]) if (arm[k] and fam[k]) else None
              for k in ("tokens_per_action", "tokens_per_lc", "wallclock_per_action",
                        "actions_per_game")}
    tpa_delta_pct = ((ratios["tokens_per_action"] - 1.0) * 100.0
                     if ratios["tokens_per_action"] else None)
    apg_delta_pct = ((ratios["actions_per_game"] - 1.0) * 100.0
                     if ratios["actions_per_game"] else None)

    # per-game actions coupling: did the arm buy tokens with moves?
    base_actions_per_game: dict[str, float] = {}
    for g in arm_bench["games"]:
        vals = [b["games"][g]["actions"] for b in baselines if g in b["games"]]
        if vals:
            base_actions_per_game[g] = sum(vals) / len(vals)
    coupling_rows = []
    for g, base in sorted(base_actions_per_game.items()):
        a = arm_bench["games"][g]["actions"]
        coupling_rows.append(dict(game=g, arm_actions=a, baseline_mean_actions=base,
                                  delta_actions=a - base,
                                  delta_pct=((a / base - 1.0) * 100.0) if base else None))
    fewer = [r["game"] for r in coupling_rows if r["delta_actions"] < 0]

    head.update(
        computed=True, arm=arm, family=fam, baselines=rows, arm_over_family=ratios,
        tokens_per_action_delta_pct=tpa_delta_pct,
        tokens_per_action_vs_external=dict(
            ours_pct=tpa_delta_pct, his_pct=EXTERNAL_TOK_PER_ACTION_DELTA_PCT,
            ratio_to_his=(tpa_delta_pct / EXTERNAL_TOK_PER_ACTION_DELTA_PCT)
            if tpa_delta_pct is not None else None,
            statement=(None if tpa_delta_pct is None else
                       f"arm tokens/action is {tpa_delta_pct:+.2f}% vs the "
                       f"continuation-v1 family; his stage-1+2+3 arm was "
                       f"{EXTERNAL_TOK_PER_ACTION_DELTA_PCT:+.1f}% "
                       f"({EXTERNAL_TOK_PER_ACTION[0]} -> {EXTERNAL_TOK_PER_ACTION[1]})")),
        wallclock_actions_coupling=dict(
            arm_actions_per_game=arm["actions_per_game"],
            family_actions_per_game=fam["actions_per_game"],
            actions_per_game_delta_pct=apg_delta_pct,
            arm_executed_fewer_actions=(arm["actions_per_game"] < fam["actions_per_game"]),
            statement=(f"the arm executed "
                       f"{'FEWER' if arm['actions_per_game'] < fam['actions_per_game'] else 'MORE OR EQUAL'} "
                       f"actions per game than the family "
                       f"({arm['actions_per_game']:.1f} vs {fam['actions_per_game']:.1f}, "
                       f"{apg_delta_pct:+.2f}%)" if apg_delta_pct is not None else None),
            games_with_fewer_actions=fewer,
            n_games_with_fewer_actions=len(fewer),
            arm_wallclock_per_game=arm["wallclock_per_game"],
            family_wallclock_per_game=fam["wallclock_per_game"],
            rail_is_wallclock_bound=dict(
                projected_s=RAIL_WALLCLOCK_PROJECTED_S, cap_s=RAIL_WALLCLOCK_CAP_S,
                note=("our scored rail projects 32,267 s against a 32,400 s cap and "
                      "all 25 games end on wall clock, not on an action limit "
                      "(max_actions_per_game=None) -- so his tokens -> fewer "
                      "actions -> fewer levels path transfers to us")),
            per_game=coupling_rows,
        ),
        note="tokens are generated tokens (uncached_input_tokens is 0 on "
             "this rail); wall clock is final_wallclock_seconds per game")
    return head


# --------------------------------------------------------------------------- #
# M3 (descriptive) -- repeated identical no-op actions on the type-1 games
# --------------------------------------------------------------------------- #
def compute_m3_for(actions_by_game: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    per_game = {}
    for g in AUDIT_TYPE1_GAMES:
        rows = actions_by_game.get(g)
        if not rows:
            per_game[g] = dict(available=False)
            continue
        rows = sorted(rows, key=lambda r: r.get("action_num", 0))
        n = len(rows)
        noops = sum(1 for r in rows if not r.get("board_changed"))
        repeats = 0
        longest = cur = 0
        for i in range(1, n):
            same = rows[i].get("action_display") == rows[i - 1].get("action_display")
            if same and not rows[i].get("board_changed"):
                repeats += 1
                cur += 1
                longest = max(longest, cur)
            else:
                cur = 0
        per_game[g] = dict(available=True, actions=n, noop_actions=noops,
                           noop_rate=noops / n if n else None,
                           repeated_identical_noops=repeats,
                           repeated_identical_noop_rate=repeats / n if n else None,
                           longest_repeat_run=longest)
    avail = [v for v in per_game.values() if v.get("available")]
    tot_a = sum(v["actions"] for v in avail)
    tot_r = sum(v["repeated_identical_noops"] for v in avail)
    tot_n = sum(v["noop_actions"] for v in avail)
    return dict(per_game=per_game, games_available=len(avail),
                actions=tot_a, repeated_identical_noops=tot_r, noop_actions=tot_n,
                repeated_identical_noop_rate=(tot_r / tot_a) if tot_a else None,
                noop_rate=(tot_n / tot_a) if tot_a else None)


def compute_m3(pull: Path, baseline_dirs: list[Path]) -> dict[str, Any]:
    arm = compute_m3_for(load_viewer_actions(pull))
    base = {}
    for d in baseline_dirs:
        base[d.name] = compute_m3_for(load_viewer_actions(d))
    pooled_a = sum(v["actions"] for v in base.values())
    pooled_r = sum(v["repeated_identical_noops"] for v in base.values())
    return dict(
        descriptive_only=True,
        definition=("action i counts as a repeated identical no-op iff "
                    "board_changed is False AND action_display == action_display[i-1]; "
                    "read from the vanilla harness per-game viewer events "
                    "(artifacts/*_events.jsonl, type=action, 1:1 with benchmark history)"),
        games=list(AUDIT_TYPE1_GAMES), arm=arm, baselines=base,
        baseline_family_pooled=dict(
            actions=pooled_a, repeated_identical_noops=pooled_r,
            repeated_identical_noop_rate=(pooled_r / pooled_a) if pooled_a else None),
        arm_over_family=((arm["repeated_identical_noop_rate"] /
                          (pooled_r / pooled_a))
                         if (arm["repeated_identical_noop_rate"] and pooled_a and pooled_r)
                         else None),
    )


# --------------------------------------------------------------------------- #
# P1 legality (prereg §4.1.1)
# --------------------------------------------------------------------------- #
def p1_legality(blob: str | None, name: str) -> dict[str, Any]:
    if blob is None:
        return dict(run=name, available=False,
                    note="no log available -- P1 cannot be verified from banners "
                         "(SCREEN_PROTOCOL §1 P1: label alone is insufficient; "
                         "git_status.txt is NOT evidence)")
    carries_cont = CONTINUATION_BANNER_TOKEN in blob
    found = {}
    for tok in FORBIDDEN_BANNER_TOKENS:
        n = blob.count(tok)
        if n:
            found[tok] = dict(count=n, first_line=_evidence(blob, tok))
    return dict(
        run=name, available=True,
        continuation_v1_banner=carries_cont,
        continuation_banner_evidence=_evidence(blob, CONTINUATION_BANNER_TOKEN),
        forbidden_tokens_found=found,
        status="PASS" if (carries_cont and not found) else "FAIL",
        rule=("prereg §4.1.1: the arm carries the continuation-v1 banner PLUS the "
              "animation banner, and NO warpack: / LEDGER / SENTINEL / 'COMPACTION ' "
              "lines. The warpack band is ILLEGAL as a control here "
              "(runs/sealed/r17_thresholds.json → thresholds.control_band)."),
    )


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def _f(x: Any, spec: str = ".4f") -> str:
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int,)) and spec.endswith("d"):
        return format(x, spec)
    try:
        return format(float(x), spec)
    except Exception:  # noqa: BLE001
        return str(x)


def render_md(res: dict[str, Any]) -> str:
    v = res["verdict"]
    c = res["canaries"]
    m0, m1, m2, m3 = res["M0"], res["M1"], res["M2"], res["M3"]
    L: list[str] = []
    L.append(f"# animation-awareness v1 — score {res['date']}")
    L.append("")
    L.append(f"**VERDICT: {v['verdict']}** — {v['why']}")
    L.append("")
    L.append(f"- pull: `{res['pull']}`")
    L.append(f"- log: `{res['log']}` ({res['log_mode']})")
    L.append(f"- arm label: `{res['arm_label']}`")
    L.append(f"- prereg: `learnings/war_room/animation_prereg_2026-08-11.md` (SEALED) — "
             f"canaries §3, metrics §4, seal arithmetic §4.1, kill rules §5")
    L.append(f"- {v['note']}")
    if v["audit_method_under_review"]:
        L.append("- **FLAG: the audit method itself goes back under review** (prereg §3 K-A2).")
    L.append("")
    ep = EXTERNAL_PRIOR
    L.append("## 0. EXTERNAL PRIOR — the feature's own author published a NULL")
    L.append("")
    L.append(f"**{ep['source']}** — {ep['author']}, {ep['posted']} "
             f"(swept in `{ep['swept_in']}`). Filed as a **pre-result external "
             f"prior**; the prereg is NOT amended and K-A3 keeps its sealed threshold.")
    L.append("")
    L.append(f"- **Efficacy: {ep['efficacy']}.**")
    L.append(f"- Harm mechanism: {ep['harm_mechanism']}")
    L.append(f"- Tool use: {ep['tool_use']}")
    L.append(f"- Why this does not falsify our arm: {ep['why_not_falsifying']}")
    L.append(f"- Consequence for this run: {ep['consequence']}")
    L.append("")
    L.append("> **M0 measures mechanism DELIVERY, not efficacy.** A good M0 may not "
             "be read as an efficacy claim: the best available efficacy evidence is "
             "his, and it is null.")
    L.append("")
    L.append("## 1. Canaries (gate everything — prereg §3/§5)")
    L.append("")
    L.append("| canary | status | outcome if fail | key numbers |")
    L.append("|---|---|---|---|")
    rows = [
        ("K-A0", c["K-A0"], f"banner={_f(c['K-A0']['banner_present'])} "
                            f"stamp={_f(c['K-A0']['flag_stamp_present'])} "
                            f"PATCH FAILED={_f(c['K-A0']['patch_failed_line'])} "
                            f"seams={c['K-A0']['seams_patched']}"),
        ("K-A1", c["K-A1"], f"{c['K-A1']['event_lines']} event lines on "
                            f"{c['K-A1']['distinct_games']} distinct games "
                            f"(need >= {K_A1_MIN_GAMES})"),
        ("K-A2", c["K-A2"], f"invisible by type-1 game: "
                            f"{json.dumps(c['K-A2']['invisible_by_type1_game'])}"),
        ("K-A3", c["K-A3"], f"tokens_est={c['K-A3']['animation_tokens_est']} / "
                            f"total={c['K-A3']['total_tokens']} = "
                            f"{_f(c['K-A3']['token_fraction'], '.6f')} (bound < 1%)"),
        ("K-A4", c["K-A4"], f"animation_errors={c['K-A4']['errors']}"),
    ]
    for k, d, nums in rows:
        L.append(f"| {k} {d['name']} | **{d['status']}** | {d['outcome_if_fail']} | {nums} |")
    L.append("")
    L.append("Raw evidence lines each canary was decided from:")
    L.append("")
    for k, d, _ in rows:
        ev = [e for e in (d.get("evidence") or []) if e]
        if ev:
            for e in ev:
                L.append(f"- `{k}` — `{e}`")
        else:
            L.append(f"- `{k}` — *(no matching line found in the log)*")
    if not c["canary_line_present"]:
        L.append("- `ANIMATION CANARY` — **line ABSENT from the log**"
                 + (" (`ANIMATION CANARY unavailable: ...` present instead)"
                    if c["canary_unavailable_line"] else ""))
    L.append("")
    L.append(f"- per-game jsonl sidecars: {c['K-A1']['sidecar_note']}")
    L.append("")

    L.append("## 2. M2 (DECIDING — external prior 734369) — tokens, and what they cost in moves")
    L.append("")
    if m2.get("computed"):
        L.append("| run | actions | actions/game | lc | gen tokens | **tok/action** | tok/lc | "
                 "wall-clock s/action |")
        L.append("|---|---|---|---|---|---|---|---|")
        for r in [m2["arm"]] + m2["baselines"] + [m2["family"]]:
            L.append(f"| {r['label']} | {r['actions']} | {_f(r['actions_per_game'], '.1f')} | "
                     f"{r['lc']} | {r['gen_tokens']} | "
                     f"**{_f(r['tokens_per_action'], '.1f')}** | {_f(r['tokens_per_lc'], '.0f')} | "
                     f"{_f(r['wallclock_per_action'], '.2f')} |")
        L.append("")
        tv = m2["tokens_per_action_vs_external"]
        L.append(f"- **tokens/action delta: {_f(m2['tokens_per_action_delta_pct'], '+.2f')}%** "
                 f"vs the `{BASELINE_FAMILY}` family. External reference: his "
                 f"stage-1+2+3 arm was **+17.0%** "
                 f"({EXTERNAL_TOK_PER_ACTION[0]} → {EXTERNAL_TOK_PER_ACTION[1]} tok/action). "
                 f"Ratio to his: {_f(tv['ratio_to_his'], '.3f')}×.")
        L.append(f"- arm / family ratios: tok/action "
                 f"{_f(m2['arm_over_family']['tokens_per_action'], '.4f')}, "
                 f"tok/lc {_f(m2['arm_over_family']['tokens_per_lc'], '.4f')}, "
                 f"wall-clock/action {_f(m2['arm_over_family']['wallclock_per_action'], '.4f')}")
        c2 = m2["wallclock_actions_coupling"]
        L.append("")
        L.append("**Wall-clock / actions coupling** (his causal path from tokens to lost levels):")
        L.append("")
        L.append(f"- **{c2['statement']}**")
        L.append(f"- arm executed fewer actions than the family: "
                 f"**{_f(c2['arm_executed_fewer_actions'])}** "
                 f"({c2['n_games_with_fewer_actions']}/{len(c2['per_game'])} games individually "
                 f"below the family mean)")
        L.append(f"- wall clock per game: arm {_f(c2['arm_wallclock_per_game'], '.1f')} s vs "
                 f"family {_f(c2['family_wallclock_per_game'], '.1f')} s")
        L.append(f"- {c2['rail_is_wallclock_bound']['note']}")
        L.append(f"- {m2['note']}")
    else:
        L.append(f"- not computed: {m2.get('reason')}")
    L.append("")

    L.append("## 3. M0 (PRIMARY pre-registered mechanism endpoint — DELIVERY, not efficacy)")
    L.append("")
    L.append(f"`invisible_actions / executed_actions` = "
             f"{m0['invisible_actions']}/{m0['executed_actions']} = "
             f"**{_f(m0['invisible_rate'], '.5f')}**; "
             f"`multi_frame_actions / executed_actions` = "
             f"{m0['multi_frame_actions']}/{m0['executed_actions']} = "
             f"{_f(m0['multi_frame_rate'], '.5f')}.")
    exp = m0["offline_audit_expectation"]
    L.append("")
    L.append(f"Offline pre-build audit (`{exp['source']}`, the pre-registered "
             f"expectation): {exp['games_multi_frame']}/25 games multi-frame, "
             f"{exp['invisible']}/{exp['actions']} = {exp['invisible_pct_of_actions']}% "
             f"INVISIBLE, multi-frame {exp['multi_frame_pct']}%. "
             f"Registered expectation: {exp['registered']}.")
    L.append(f"**Expectation check: {m0['expectation_check']}** "
             f"(misses: {m0['expectation_misses'] or 'none'}; "
             f"surprises: {m0['expectation_surprises'] or 'none'})")
    L.append("")
    L.append("| game | audit type | exec | multi | invis | invis rate | multi rate | "
             "audit invis% (comb / probe-A) | audit multi% | expectation | status |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in m0["per_game"]:
        L.append(
            f"| {r['game']} | {r['audit_type']} | {r['executed_actions']} | "
            f"{r['multi_frame_actions']} | {r['invisible_actions']} | "
            f"{_f(r['invisible_rate'], '.4f')} | {_f(r['multi_frame_rate'], '.4f')} | "
            f"{r['audit_invisible_pct_combined']}% / {r['audit_invisible_pct_recorded']}% | "
            f"{r['audit_multi_frame_pct_combined']}% | {r['expectation']} | "
            f"{r['expectation_status']} |")
    L.append("")
    L.append(f"- executed actions: canary={m0['executed_actions_from_canary']}, "
             f"benchmark={m0['executed_actions_from_benchmark']}")
    L.append(f"- event lines vs canary counters consistent: "
             f"{_f(m0['event_lines_consistent_with_canary'])} "
             f"(event lines: {m0['event_line_multi']} multi / "
             f"{m0['event_line_invisible']} invisible)"
             + ("  **(log truncation suspected — event lines < canary counter)**"
                if m0.get("log_truncation_suspected") else ""))
    L.append("")

    L.append("## 4. M1 (SECONDARY — DESCRIPTIVE ONLY, NOT A SCREEN)")
    L.append("")
    L.append(f"**{m1['screenable_statement']}** — family `{m1['family']}`, m = {m1['m']}.")
    L.append(f"**M1 verdict: {m1['verdict']}.** {m1['verdict_note']}")
    L.append("")
    for b in m1["baselines"]:
        L.append(f"- baseline `{b['path']}` — label `{b['label']}` "
                 f"(matches family: {_f(b['label_matches_family'])}), lc total {b['lc_total']}")
    if m1.get("family_ss_recheck"):
        rc = m1["family_ss_recheck"]
        L.append(f"- family SS re-check from those two benchmark.json files: "
                 f"SS={rc['ss']} df={rc['df']} vs sealed {rc['sealed_ss']} "
                 f"→ matches: {_f(rc['matches'])}")
    s = m1["seal_arithmetic"]
    L.append("")
    L.append(f"- σ̂ re-derived from the SCREEN_PROTOCOL §1 P3 pooled SS table = "
             f"{s['checks']['sigma_rederived']} (sealed {s['sigma_hat']}, "
             f"match {_f(s['checks']['sigma_matches_sealed'])}); "
             f"df = {s['checks']['df_rederived']} (sealed {SEALED_DF}, "
             f"match {_f(s['checks']['df_matches_sealed'])})")
    L.append(f"- **ADVISORY** K3″ line at m={s['m']}: −C({s['m']})·σ̂ = "
             f"−{s['C_m']}×{s['sigma_hat']} = **{_f(s['k3pp_line_advisory'], '.5f')}** "
             f"lc/game (sealed {SEALED_K3PP_M2}, match "
             f"{_f(s['checks']['k3pp_matches_sealed'])})")
    L.append(f"- **ADVISORY** 80%-power floor at m={s['m']}: "
             f"C({s['m']})·σ̂ + 0.8416·σ̂·√(1+1/{s['m']}) = "
             f"**{_f(s['power80_floor_lc_per_game'], '.5f')}** lc/game = "
             f"**{_f(s['power80_floor_levels'], '.2f')} levels** over "
             f"{s['n_games']} games (sealed {SEALED_FLOOR_M2} / "
             f"{SEALED_FLOOR_LEVELS_M2}, match "
             f"{_f(s['checks']['floor_matches_sealed'])}/"
             f"{_f(s['checks']['levels_matches_sealed'])})")
    L.append(f"- {s['advisory_label']}")
    L.append(f"- seal arithmetic all-match: **{_f(s['seal_arithmetic_match'])}**")
    if m1.get("computed"):
        L.append("")
        L.append(f"- arm lc total {m1['arm_lc_total']} vs baseline totals "
                 f"{m1['baseline_lc_totals']} (family mean "
                 f"{_f(m1['baseline_family_mean_levels'], '.1f')} levels)")
        L.append(f"- paired mean Δlc = **{_f(m1['mean_dlc'], '.5f')}** lc/game over "
                 f"{m1['n_games_paired']} games (sd {_f(m1['sd_games'], '.4f')}, "
                 f"{m1['wins']}W/{m1['losses']}L, sign-flip p = "
                 f"{_f(m1['signflip_p_exact_two_sided'], '.4f')} — "
                 f"{m1['signflip_note']})")
        if m1.get("vs_advisory_k3pp_line"):
            k = m1["vs_advisory_k3pp_line"]
            L.append(f"- vs the ADVISORY line {_f(k['line'], '.5f')}: mean Δlc is "
                     f"{'above' if k['mean_dlc_above_line'] else 'below'} it — "
                     f"{k['label']}")
        L.append("")
        L.append("| game | arm lc | baseline lcs | baseline mean | Δlc |")
        L.append("|---|---|---|---|---|")
        for r in m1["per_game"]:
            L.append(f"| {r['game']} | {r['arm_lc']} | {r['baseline_lcs']} | "
                     f"{_f(r['baseline_mean_lc'], '.1f')} | {_f(r['dlc'], '.1f')} |")
    else:
        L.append(f"- not computed: {m1.get('reason')}")
    L.append("")

    L.append("## 5. M3 (descriptive) — repeated identical no-ops on the type-1 games")
    L.append("")
    L.append(f"Definition: {m3['definition']}")
    L.append("")
    L.append("| run | games | actions | no-ops | repeated identical no-ops | rate |")
    L.append("|---|---|---|---|---|---|")
    a = m3["arm"]
    L.append(f"| ARM | {a['games_available']}/4 | {a['actions']} | {a['noop_actions']} | "
             f"{a['repeated_identical_noops']} | "
             f"{_f(a['repeated_identical_noop_rate'], '.4f')} |")
    for name, b in m3["baselines"].items():
        L.append(f"| {name} | {b['games_available']}/4 | {b['actions']} | {b['noop_actions']} | "
                 f"{b['repeated_identical_noops']} | "
                 f"{_f(b['repeated_identical_noop_rate'], '.4f')} |")
    fp = m3["baseline_family_pooled"]
    L.append(f"| family pooled | — | {fp['actions']} | — | {fp['repeated_identical_noops']} | "
             f"{_f(fp['repeated_identical_noop_rate'], '.4f')} |")
    L.append("")
    L.append(f"- arm / family rate ratio: {_f(m3['arm_over_family'], '.4f')}")
    L.append("")

    L.append("## 6. P1 legality (prereg §4.1.1, SCREEN_PROTOCOL §1 P1)")
    L.append("")
    L.append("| run | continuation-v1 banner | forbidden tokens | status |")
    L.append("|---|---|---|---|")
    for p in res["P1"]:
        if not p.get("available"):
            L.append(f"| {p['run']} | n/a | n/a | NOT VERIFIABLE |")
            continue
        L.append(f"| {p['run']} | {_f(p['continuation_v1_banner'])} | "
                 f"{', '.join(p['forbidden_tokens_found']) or 'none'} | "
                 f"**{p['status']}** |")
    L.append("")
    for p in res["P1"]:
        if p.get("available") and p.get("forbidden_tokens_found"):
            for tok, d in p["forbidden_tokens_found"].items():
                L.append(f"- `{p['run']}` illegal `{tok}` ×{d['count']} — `{d['first_line']}`")
    L.append("")
    L.append(f"_Generated by `duck_eval/warpack/animation_score.py` at "
             f"{res['generated']}._")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def score(pull: Path, log_path: Path | None, baseline_dirs: list[Path],
          date: str) -> dict[str, Any]:
    arm_bench = load_bench(pull)
    log_path = log_path or find_log(pull)
    blob, log_mode = (load_log_text(log_path) if log_path and log_path.is_file()
                      else ("", "MISSING"))
    events = parse_events(blob)
    canary = parse_canary(blob)
    sidecars = load_sidecars(pull)
    audit = load_audit()

    baselines = []
    for d in baseline_dirs:
        b = load_bench(d)
        if b:
            b["dir"] = str(d)
            baselines.append(b)

    canaries = run_canaries(blob, events, canary, arm_bench, sidecars)
    verdict = verdict_from_canaries(canaries)
    m0 = compute_m0(events, canary, arm_bench, audit)
    m1 = compute_m1(arm_bench, baselines)
    m2 = compute_m2(arm_bench, baselines)
    m3 = compute_m3(pull, baseline_dirs)

    p1 = [p1_legality(blob if log_path else None, f"ARM {pull.name}")]
    for d in baseline_dirs:
        bl = find_log(d)
        bblob = load_log_text(bl)[0] if bl and bl.is_file() else None
        p1.append(p1_legality(bblob, f"baseline {d.name}"))

    return dict(
        schema="animation_score/1",
        generated=_dt.datetime.now().isoformat(timespec="seconds"),
        date=date, pull=str(pull), log=str(log_path) if log_path else None,
        log_mode=log_mode, arm_label=(arm_bench or {}).get("label"),
        prereg="learnings/war_room/animation_prereg_2026-08-11.md",
        protocol="duck_eval/SCREEN_PROTOCOL.md",
        external_prior=EXTERNAL_PRIOR,
        metric_roles=dict(
            M0="PRIMARY pre-registered mechanism endpoint (delivery, not efficacy)",
            M2="DECIDING metric of this run (external prior 734369)",
            M1="descriptive only; NOT SCREENABLE at m=2; no PASS/FAIL ever",
            M3="descriptive"),
        verdict=verdict, canaries=canaries, canary_line=canary,
        M2=m2, M0=m0, M1=m1, M3=m3, P1=p1,
        event_lines=len(events),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Score a pulled arc3-duck-animation-eval run against the "
                    "SEALED prereg animation_prereg_2026-08-11.md.")
    ap.add_argument("--pull", required=True, help="pulled kernel output directory")
    ap.add_argument("--log", default=None, help="run log (default: the pull dir's non-vllm *.log)")
    ap.add_argument("--baseline", action="append", default=None,
                    help=f"baseline run dir (repeatable; default: the two sealed "
                         f"{BASELINE_FAMILY} runs)")
    ap.add_argument("--date", default=None, help="date tag for the outputs (default: today)")
    ap.add_argument("--out-dir", default=None, help="default: runs/animation")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    pull = Path(args.pull).resolve()
    if not pull.is_dir():
        print(f"ERROR: --pull {pull} is not a directory", file=sys.stderr)
        return 2
    log_path = Path(args.log).resolve() if args.log else None
    baseline_dirs = ([Path(b).resolve() for b in args.baseline] if args.baseline
                     else [Path(p) for p in BASELINE_DIRS_DEFAULT])
    date = args.date or _dt.date.today().isoformat()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (REPO / "runs" / "animation")
    out_dir.mkdir(parents=True, exist_ok=True)

    res = score(pull, log_path, baseline_dirs, date)
    js = out_dir / f"score_{date}.json"
    md = out_dir / f"score_{date}.md"
    js.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    md.write_text(render_md(res), encoding="utf-8")

    if not args.quiet:
        v = res["verdict"]
        print(f"VERDICT: {v['verdict']} -- {v['why']}")
        for k in ("K-A0", "K-A1", "K-A2", "K-A3", "K-A4"):
            print(f"  {k}: {res['canaries'][k]['status']}")
        m2 = res["M2"]
        if m2.get("computed"):
            print(f"  M2 (DECIDING) tok/action arm {m2['arm']['tokens_per_action']:.1f} vs "
                  f"family {m2['family']['tokens_per_action']:.1f} "
                  f"= {m2['tokens_per_action_delta_pct']:+.2f}% (his +17.0%); "
                  f"{m2['wallclock_actions_coupling']['statement']}")
        else:
            print(f"  M2 (DECIDING): not computed -- {m2.get('reason')}")
        print(f"  M0 invisible/executed = {res['M0']['invisible_actions']}/"
              f"{res['M0']['executed_actions']} "
              f"({_f(res['M0']['invisible_rate'], '.5f')}); "
              f"expectation {res['M0']['expectation_check']}")
        print(f"  M1: {res['M1']['screenable_statement']} -> {res['M1']['verdict']}")
        print(f"  wrote {js}")
        print(f"  wrote {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
