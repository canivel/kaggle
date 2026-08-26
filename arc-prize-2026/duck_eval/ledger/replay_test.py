"""Stage-1 unit/replay test (protocol step 1) — local, no LLM.

Replays the 13 recorded runs through the ledger extractor:
  * 3 seed-1 text transcripts (sb26/su15/lp85, runs/phase1_ab/seed1/) via the
    legacy HeuristicExtractor, turn by turn, exactly as the live tap would run;
  * 10 null benchmark action histories (runs/null10/seed101..110) via the
    harness-side FACT feed logic (repeat-coordinate no-ops, per-action replay).

Pass gates (intervention_plan.md, protocol step 1):
  G1  sb26 seed1: ledger accumulates >=20 refuted ordering-family variants
      AND escalation would have fired by action ~60.
  G2  su15 seed1: both self-disproved goals (move-magentas-into-blob,
      align-with-top-bar) reach status refuted with the agent's own
      arithmetic as evidence.
  G3  sb26 SPACE=timer-only FACT: recorded early, still present in the final
      digest with >14 analyzer turns elapsed since (i.e. it outlived the
      14-message window) and it survives the action-140 GAME_OVER restart.
  G4  digest <=600 tokens at EVERY replayed turn across all 13 runs.
  G5  GOAL:/RESULT:/FACT: prompt-field regex extraction round-trips (the new
      contract the live graft relies on).

Run:  uv run python duck_eval/ledger/replay_test.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

import ledger_core as core  # noqa: E402

TRANSCRIPTS = REPO / "runs" / "phase1_ab" / "seed1" / "transcripts"
NULL10 = REPO / "runs" / "null10"

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}" + (f"  [{detail}]" if detail else ""))
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


# ---------------------------------------------------------------- transcript replay
_BLOCK_RE = re.compile(r"^--- analysis_step=(\d+) \| action=(\d+) \|.*?---$", re.M)
_THINK_RE = re.compile(
    r"\[(THINKING|ASSISTANT)\]\n(.*?)(?=\n\[(?:TOOL CALL|TOOL RESULT|ANALYZER|"
    r"USER|SYSTEM|THINKING|ASSISTANT)|\Z)", re.S)


def parse_transcript(path: Path) -> list[tuple[int, int, str, str]]:
    """-> ordered [(analysis_step, action_num, model_text, full_block)]."""
    text = path.read_text(encoding="utf-8", errors="replace")
    parts = _BLOCK_RE.split(text)
    turns: list[tuple[int, int, str, str]] = []
    for i in range(1, len(parts), 3):
        step, action = int(parts[i]), int(parts[i + 1])
        body = parts[i + 2]
        chunks = [m.group(2) for m in _THINK_RE.finditer(body)]
        turns.append((step, action, "\n".join(chunks), body))
    return turns


def replay_transcript(path: Path):
    """Feed a recorded transcript through Ledger + HeuristicExtractor turn by
    turn, tracking per-turn digest sizes and level/game-over events (from the
    recorded action results). Returns (ledger, extractor, stats)."""
    ledger = core.Ledger()
    extractor = core.HeuristicExtractor(ledger)
    turns = parse_transcript(path)
    max_digest_tokens = 0
    digests_checked = 0
    seen_events: set[str] = set()
    last_step = 0
    for step, action, model_text, body in turns:
        # replicate env events visible in the recorded results
        for m in re.finditer(r"\{[^{}]*'action_num': (\d+)[^{}]*\}", body):
            blob = m.group(0)
            a_num = int(m.group(1))
            if "'level_completed': True" in blob:
                key = f"lvl@{a_num}"
                if key not in seen_events:
                    seen_events.add(key)
                    extractor.observe_level_up(a_num)
            if "'game_over': True" in blob:
                key = f"go@{a_num}"
                if key not in seen_events:
                    seen_events.add(key)
                    extractor.observe_game_over(a_num)
        extractor.process_turn(model_text, step=step, action=action)
        digest = ledger.render_digest()
        max_digest_tokens = max(max_digest_tokens, core.estimate_tokens(digest))
        digests_checked += 1
        last_step = max(last_step, step)
    stats = {
        "turns": digests_checked,
        "max_digest_tokens": max_digest_tokens,
        "last_step": last_step,
        "final_digest": ledger.render_digest(),
    }
    return ledger, extractor, stats


# ---------------------------------------------------------------- null10 replay
def replay_null_history(game_run: dict) -> tuple[core.Ledger, int]:
    """Replay a benchmark action history through the harness-side FACT logic
    (repeat-coordinate no-op detection uses coordinate recurrence, since the
    recorded histories carry no board_changed flag)."""
    ledger = core.Ledger()
    counts: dict[str, int] = {}
    max_digest_tokens = 0
    for i, entry in enumerate(game_run.get("history") or []):
        action = entry.get("action") or {}
        display = action.get("id", "")
        data = action.get("data") or {}
        if display == "ACTION6":
            display = f"MOUSE(row={data.get('y')}, col={data.get('x')})"
        counts[display] = counts.get(display, 0) + 1
        if counts[display] in (3, 10, 30):  # heavy re-click -> candidate no-op
            ledger.add_fact(
                f"repeated action: {display} issued {counts[display]}x "
                "with no level progress; likely no-op", action=i)
        max_digest_tokens = max(
            max_digest_tokens, core.estimate_tokens(ledger.render_digest()))
    return ledger, max_digest_tokens


# ---------------------------------------------------------------- gates
def main() -> int:
    print(f"ledger stage-1 replay test | repo={REPO}")
    all_digests_ok = True
    runs_replayed = 0

    # --- seed1 text transcripts (3 runs) --------------------------------
    print("\n[seed1 transcripts]")
    results = {}
    for game in ("sb26", "su15", "lp85"):
        path = next(TRANSCRIPTS.glob(f"{game}-*_p0.txt"))
        ledger, extractor, stats = replay_transcript(path)
        results[game] = (ledger, extractor, stats)
        runs_replayed += 1
        ok = stats["max_digest_tokens"] <= core.DIGEST_TOKEN_CAP
        all_digests_ok &= ok
        print(f"  {game}: turns={stats['turns']} hyps={len(ledger.hypotheses)} "
              f"refuted={ledger.refuted_count()} facts={len(ledger.facts)} "
              f"max_digest_tokens={stats['max_digest_tokens']}")

    # G1: sb26 — >=20 refuted ordering variants + escalation by ~action 60
    sb_ledger, _, sb_stats = results["sb26"]
    n_ord = sb_ledger.refuted_count("ordering")
    check("G1a sb26 >=20 refuted ordering-family variants", n_ord >= 20,
          f"refuted ordering variants = {n_ord}")
    trig = sb_ledger.escalation_trigger_action
    check("G1b sb26 escalation would fire by ~action 60",
          trig is not None and trig <= 60, f"trigger action = {trig}")

    # G2: su15 — both self-disproved goals refuted w/ own arithmetic
    su_ledger, _, _ = results["su15"]
    arith = re.compile(r"-\d|\d+\s*[-+]\s*\d+\s*=|off the board")

    def refuted_goal(keyword_re: str):
        pat = re.compile(keyword_re, re.I)
        for h in su_ledger.hypotheses:
            if h["status"] == "refuted" and pat.search(h["statement"]):
                if any(arith.search(e) for e in h["evidence"]):
                    return h
        return None

    blob_goal = refuted_goal(r"blob")
    bar_goal = refuted_goal(r"top bar|top row|align|position")
    check("G2a su15 'move magentas into blob' goal refuted w/ arithmetic",
          blob_goal is not None,
          (blob_goal["id"] + ": " + blob_goal["evidence"][0][:60])
          if blob_goal else "not found")
    check("G2b su15 'align with top bar' goal refuted w/ arithmetic",
          bar_goal is not None and (blob_goal is None
                                    or bar_goal["id"] != blob_goal["id"]),
          (bar_goal["id"] + ": " + bar_goal["evidence"][0][:60])
          if bar_goal else "not found")

    # G3: sb26 SPACE=timer FACT survives eviction + GAME_OVER
    space_fact = next(
        (f for f in sb_ledger.facts
         if re.search(r"(?i)space", f["statement"])
         and re.search(r"(?i)timer|decrement", f["statement"])), None)
    final_digest = sb_stats["final_digest"]
    survives = (space_fact is not None
                and space_fact["statement"][:60] in final_digest
                and (sb_stats["last_step"] - space_fact["born_step"]) > 14)
    check("G3a sb26 SPACE=timer FACT recorded and in final digest past msg 14",
          survives,
          f"fact={space_fact['statement'][:60] if space_fact else None!r} "
          f"born_step={space_fact['born_step'] if space_fact else '-'} "
          f"last_step={sb_stats['last_step']}")
    check("G3b sb26 ledger survived the GAME_OVER restart (entries kept)",
          sb_ledger.game_overs >= 1 and sb_ledger.refuted_count() >= 20
          and space_fact is not None,
          f"game_overs={sb_ledger.game_overs}")

    # --- null10 histories (10 runs) --------------------------------------
    print("\n[null10 action histories]")
    for seed_dir in sorted(NULL10.glob("seed1??")):
        bench = json.loads((seed_dir / "benchmark.json").read_text(encoding="utf-8"))
        worst = 0
        for game_run in bench["game_runs"]:
            if game_run["game_id"].split("-")[0] in ("sb26", "su15", "lp85"):
                _, max_tok = replay_null_history(game_run)
                worst = max(worst, max_tok)
        runs_replayed += 1
        ok = worst <= core.DIGEST_TOKEN_CAP
        all_digests_ok &= ok
        print(f"  {seed_dir.name}: max_digest_tokens={worst} {'ok' if ok else 'OVER'}")

    check("G4 digest <=600 tokens at every turn across all 13 runs",
          all_digests_ok and runs_replayed == 13,
          f"runs={runs_replayed}")

    # G5: new-contract GOAL:/RESULT:/FACT: extraction round-trip
    sample = (
        "GOAL: [ordering] fill main frame slots 0,1,3 and frame-14 slots with "
        "the flattened call order\n"
        "some analysis text...\n"
        "FACT: SPACE only decrements the timer bar; it never submits\n"
        "RESULT: refuted - flattened call order run failed, fail flash at slot 2\n")
    records = core.extract_goal_result(sample)
    led = core.Ledger()
    led.ingest(records, step=3, action=12)
    ok = (len(records) == 3
          and records[0]["kind"] == "goal" and records[0]["family"] == "ordering"
          and records[1]["kind"] == "fact"
          and records[2]["kind"] == "result" and records[2]["verdict"] == "refuted"
          and led.refuted_count("ordering") == 1 and len(led.facts) == 1
          and led.hypotheses[0]["evidence"])
    check("G5 GOAL:/RESULT:/FACT: prompt-field extraction round-trip", ok,
          str(records)[:120])
    # escalation one-shot semantics on the same path
    for i in range(3):
        led.ingest(core.extract_goal_result(
            f"GOAL: [ordering] variant {i} place items in reverse order\n"
            f"RESULT: refuted - run failed again variant {i}"), action=20 + i)
    fired_once = led.consume_escalation()
    fired_twice = led.consume_escalation()
    check("G5b escalation arms at 3 same-family refutations and is one-shot",
          fired_once is not None and "4 mechanically distinct" in fired_once
          and fired_twice is None,
          f"armed_at_action={led.escalation_trigger_action}")

    print(f"\nRESULT: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
