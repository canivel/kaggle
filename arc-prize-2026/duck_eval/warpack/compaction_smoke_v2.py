"""A22 compaction v2.1 (pure eviction, digest-OFF) CPU smoke test -- no GPU,
no LLM.

Run from the repo root:
    uv run python duck_eval/warpack/compaction_smoke_v2.py

Covers every sealed v2 design point (a22_compaction_v2_prereg_2026-08-04.md
sec2) PLUS the v2.1 digest-OFF arm (a22_compaction_v2_1_prereg_2026-08-06.md
sec1: digest injection disabled by default behind COMPACTION_DIGEST=0;
reserve_applied=0 and digest_tokens=0 on every event; =1 restores v2).
feedback_test_before_submit: ALWAYS runtime-test agent code before any
Kaggle push. The v1 smoke (compaction_smoke.py) tested the retired v1
mechanism and is kept for history; THIS file is the pre-push gate for v2.1.

Sections V1-V4, V9-V11 run flag-independent unit checks. Section D runs the
v2.1 DEFAULT (COMPACTION_DIGEST unset => digest OFF). Sections V5-V8 run
under COMPACTION_DIGEST=1 as the flag-restores-v2 REGRESSION (proving the
flag works both ways); the env var is removed again before V9.

  V0  flag gate / kill switch / idempotency / v2.1 identity (banner, label,
      VERSION, digest=OFF default) / RETAIN default OFF (no _chat_completion
      wrapper)
  V1  FACT hygiene gate accept/reject unit cases (hedge prefixes incl.
      case-insensitivity + punctuation-prefixed, mid-sentence truncation,
      questions, trailing quotes/brackets)
  V2  anti-self-ingestion: digest-shaped lines stripped before extraction
  V3  region model: pin selection (scientist-note carrier >=2 labels, latest
      reasoning block) + deterministic eviction ordering classes 1..5 incl.
      head-user span expansion, pinned-span skip, last-resort pin yield,
      preserve-tail never evicted
  V4  stuck rubric: K=5 default, fewer-than-K => not stuck, any change in the
      last K => not stuck, non-tool/malformed messages ignored, env override
  D   digest-OFF (v2.1 default, COMPACTION_DIGEST unset): eviction still
      fires under token pressure, pins respected, capture-into-store
      unchanged, NO digest message EVER in the outbound history (even with
      earned records on a later trim), digest_tokens=0 AND reserve_applied=0
      on every emitted event (stdout + sidecar), event/sidecar v=2.1
  V5  end-to-end trim UNDER COMPACTION_DIGEST=1 (v2-restore regression):
      region-aware eviction, reserve-only-when-earned
      (records first appearing during a no-reserve trim are deferred to the
      next trim), digest header softened + non-quotable, refuted rendered,
      NO ACTIVE/CONFIRMED lines, event line + sidecar with the NEW v2 fields,
      stale-digest hygiene (one digest, never ingested, tokens <= reserve),
      episodes increment only on evicting trims
  V6  refuted NEVER elided (no "+N more"), newest-first, budget overflow
      drops the OLDEST lines; FACT lines newest-first; hedged FACT gated out
  V7  empty gate => inject NOTHING (no digest message, reserve_applied=0)
  V8  while-stuck: (a) persistence cap DEFERRED (no cut, no ingest,
      stuck_suppressed counts), (b) budget-forced eviction still happens but
      NO digest + NO event, (c) counters flush into the next non-stuck event
  V9  RETAIN sub-arm: _mirror_reasoning unit + wrapper installed ONLY under
      COMPACTION_RETAIN=1 (subprocess) + banner says mirroring=ON
  V10 vanilla fallback: a poisoned store makes the trim wrapper fall back to
      stock behavior (no crash, no digest, no event)
  V11 source discipline: no threading/locks, no HTTP client in the eviction
      path, COMPACTION_DIGEST defaults to 0 in source, ledger_core.py
      byte-identical to the canonical twin
  V12 builder --compaction: graft cell + COMPACTION=1 stamp + v2.1 patch
      greppable in the dataset (VERSION v2.1 + ACTIVE banner)
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BUNDLE = REPO / "duck_eval" / "taaf_bundle" / "src"
DATASET = HERE / "_kaggle_dataset"

sys.path.insert(0, str(DATASET))
sys.path.insert(0, str(BUNDLE / "ARC3-Inference"))

# Small window so trims actually evict; set BEFORE tool_agent import (the
# module reads env at import time).
os.environ["LOCAL_ANALYZER_CONTEXT_WINDOW"] = "4096"
os.environ["LOCAL_ANALYZER_MODEL_ID"] = "smoke-test-model"
os.environ["LOCAL_ANALYZER_BASE_URL"] = "http://127.0.0.1:9/v1"
os.environ["COMPACTION_RESERVE_TOKENS"] = "400"
for _k in ("COMPACTION", "COMPACTION_DISABLE", "COMPACTION_RETAIN",
           "COMPACTION_STUCK_K", "COMPACTION_DIGEST"):
    os.environ.pop(_k, None)

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


from inference.agent.tool_agent import ToolAgent  # noqa: E402

import compaction_patch  # noqa: E402
import ledger_core  # noqa: E402


def big(text: str, chars: int = 9000) -> str:
    reps = (chars // (len(text) + 1)) + 1
    return "\n".join([text] * reps)[:chars]


def tool_msg(action_num: int, level: int, *, board_changed: bool,
             action: str = "MOUSE", level_completed: bool = False,
             game_over: bool = False, pad: int = 0) -> dict:
    payload = {
        "tool": "python",
        "returncode": 0,
        "result": {
            "action_result": {
                "executed": True,
                "action_num": action_num,
                "level": level,
                "board_changed": board_changed,
                "executed_actions": [action],
                "level_completed": level_completed,
                "game_over": game_over,
            }
        },
    }
    if pad:
        payload["stdout"] = "x" * pad
    return {"role": "tool", "tool_call_id": f"call-{action_num}",
            "content": json.dumps(payload)}


def asst(content: str, *, reasoning: str = "", tool_calls: bool = False,
         call_id: str = "call-x") -> dict:
    m: dict = {"role": "assistant", "content": content}
    if reasoning:
        m["reasoning"] = reasoning
    if tool_calls:
        m["tool_calls"] = [{"id": call_id, "type": "function",
                            "function": {"name": "python", "arguments": "{}"}}]
    return m


def parse_event(line: str) -> dict:
    return dict(tok.split("=", 1) for tok in line.split("COMPACTION ", 1)[1].split()
                if "=" in tok)


def run() -> int:
    print("== V0 flag gate / kill switch / v2.1 identity ==")
    stock_trim = ToolAgent._trim_messages_for_context
    stock_chat = ToolAgent._chat_completion
    check("VERSION is v2.1", compaction_patch.VERSION == "v2.1",
          compaction_patch.VERSION)
    check("digest default is OFF (COMPACTION_DIGEST unset => 0)",
          compaction_patch._digest_enabled() is False)
    check("apply() no-op without COMPACTION=1", compaction_patch.apply() is False)
    check("methods unpatched without flag",
          ToolAgent._trim_messages_for_context is stock_trim)
    os.environ["COMPACTION"] = "1"
    os.environ["COMPACTION_DISABLE"] = "1"
    check("kill switch beats flag", compaction_patch.apply() is False)
    check("methods unpatched under kill switch",
          ToolAgent._trim_messages_for_context is stock_trim)
    del os.environ["COMPACTION_DISABLE"]

    class FakeBm:
        label = "duck"

    bm = FakeBm()
    banner_io = io.StringIO()
    with contextlib.redirect_stdout(banner_io):
        applied = compaction_patch.apply(bm)
    banner = banner_io.getvalue()
    check("apply() True with COMPACTION=1", applied is True)
    check("bm.label stamped v2.1", bm.label == "duck-compaction-v2.1", bm.label)
    check("banner is 'compaction v2.1: ACTIVE'",
          "compaction v2.1: ACTIVE" in banner, banner[:120])
    check("banner shows digest=OFF (v2.1 default)", "digest=OFF" in banner,
          banner[:400])
    check("banner shows RETAIN mirroring OFF (v2 default)",
          "mirroring=OFF (v2 default)" in banner)
    check("banner shows stuck-suppress K=5", "stuck-suppress K=5" in banner)
    check("methods patched", ToolAgent._trim_messages_for_context is not stock_trim)
    check("idempotent re-apply", compaction_patch.apply() is True)
    check("RETAIN OFF: _chat_completion NOT wrapped",
          ToolAgent._chat_completion is stock_chat
          and ToolAgent._chat_completion.__name__ != "compaction_chat_completion")
    check("RETAIN default is 0", compaction_patch._retain_enabled() is False)

    print("== V1 FACT hygiene gate ==")
    ok = compaction_patch._fact_hygiene_ok
    check("accepts declarative sentence", ok("The key opens the door."))
    check("accepts exclamation", ok("SPACE only decrements the timer!"))
    check("accepts trailing quote", ok('The rule is "match colors".'))
    check("accepts trailing bracket", ok("The exit is on the right (top row)."))
    check("rejects hedge 'Maybe'", not ok("Maybe the slots fill left to right."))
    check("rejects hedge 'Wait,' (case-insensitive)", not ok("Wait, the timer resets."))
    check("rejects hedge 'actually'", not ok("actually the door is open."))
    check("rejects hedge 'I think'", not ok("I think the key is red."))
    check("rejects punctuation-prefixed hedge", not ok('"Maybe this works."'))
    check("rejects mid-sentence truncation",
          not ok("The board has seven slots and the"))
    check("rejects question", not ok("Is the timer a countdown?"))
    check("rejects empty/short", not ok("") and not ok("Ok."))
    check("hedge word mid-sentence is fine",
          ok("The agent should not wait here."))

    print("== V2 anti-self-ingestion strip ==")
    strip = compaction_patch._strip_digest_echoes
    echo_text = ("The timer counts down every turn.\n"
                 "FACT F3: SPACE only advances the timer without changing the board.\n"
                 "  REFUTED H2: filling slots left to right wins.\n"
                 "CONFIRMED H4: the door needs a key.\n"
                 "ACTIVE H5: maybe the colors cycle.\n"
                 "=== COMPACTED HISTORY v2 (echo of the digest header) ===\n"
                 "The door needs the red key first.")
    stripped = strip(echo_text)
    check("genuine lines survive", "timer counts down" in stripped
          and "red key first" in stripped)
    check("FACT echo stripped", "FACT F3" not in stripped)
    check("REFUTED echo stripped (indented)", "REFUTED H2" not in stripped)
    check("CONFIRMED echo stripped", "CONFIRMED H4" not in stripped)
    check("ACTIVE echo stripped", "ACTIVE H5" not in stripped)
    check("digest-marker echo stripped",
          compaction_patch.DIGEST_MARKER not in stripped)
    # end-to-end: an evicted assistant echoing a digest line never re-seeds
    # the ledger (the sc25 F5-quotes-F3 round trip)
    s_probe = compaction_patch._CompactionStore(None)
    s_probe.ingest_message(asst(echo_text))
    check("echoed FACT not re-harvested into the ledger",
          not any("advances the timer" in f["statement"]
                  for f in s_probe.ledger.facts),
          repr(s_probe.ledger.facts))

    print("== V3 region model: pins + eviction ordering ==")
    sci_msg = asst("World model: grid of 7 slots\nGoal model: fill in order\n"
                   "Plan: test SPACE next")
    check("scientist-note carrier detected (>=2 labels)",
          compaction_patch._is_scientist_note(sci_msg))
    check("single label is NOT a carrier",
          not compaction_patch._is_scientist_note(asst("Plan: test SPACE next")))
    check("reasoning block detected",
          compaction_patch._has_reasoning(asst("x", reasoning="deep")))
    check("empty reasoning is not a reasoning block",
          not compaction_patch._has_reasoning(asst("x")))

    hist = [
        {"role": "user", "content": "turn 1 dump"},                     # 0
        asst("ep A musing", tool_calls=True, call_id="c1"),             # 1
        tool_msg(11, 1, board_changed=True),                            # 2
        sci_msg,                                                        # 3 pinned
        {"role": "user", "content": "turn 2 dump"},                     # 4
        asst("ep B refute", tool_calls=True, call_id="c2"),             # 5
        tool_msg(12, 1, board_changed=False),                           # 6
        asst("long musing text"),                                       # 7
        asst("final note", reasoning="deep tail thought"),              # 8 pinned
        {"role": "user", "content": "current frame"},                   # 9
    ]
    sel = compaction_patch._select_evictable_block
    s1 = sel(hist, 1)
    check("class 1: oldest stale episode first", s1 == ("episode", 1, 2), repr(s1))
    del hist[1:3]
    s2 = sel(hist, 1)
    check("class 2: older user block next (head span with pin skipped)",
          s2 == ("user", 2, 2), repr(s2))
    del hist[2]
    s3 = sel(hist, 1)
    check("class 3: non-pinned assistant text block", s3 == ("reasoning", 4, 4),
          repr(s3))
    del hist[4]
    s4 = sel(hist, 1)
    check("class 4: newest episode yields last of the classes",
          s4 == ("episode", 2, 3), repr(s4))
    del hist[2:4]
    s5 = sel(hist, 1)
    check("class 5: last resort -- pins yield (head span incl. sci+reasoning)",
          s5 == ("fallback", 0, 2), repr(s5))
    del hist[0:3]
    check("preserve tail never evicted (no candidate left)",
          sel(hist, 1) is None, repr(hist))

    print("== V4 stuck rubric ==")
    is_stuck = compaction_patch._is_stuck
    nostep = [tool_msg(i, 1, board_changed=False) for i in range(1, 6)]
    step = [tool_msg(9, 1, board_changed=True)]
    check("K=5 default", compaction_patch._stuck_k() == 5)
    check("5 no-change actions => stuck", is_stuck(list(nostep), 5))
    check("4 no-change actions => NOT stuck (fewer than K)",
          not is_stuck(nostep[:4], 5))
    check("change within last K => NOT stuck",
          not is_stuck(nostep[:4] + step, 5))
    check("older change + 5 trailing no-change => stuck",
          is_stuck(step + nostep, 5))
    check("non-tool messages ignored by the rubric",
          is_stuck([{"role": "user", "content": "x"}] + nostep
                   + [asst("thinking")], 5))
    check("malformed tool payload ignored",
          not is_stuck(nostep[:4] + [{"role": "tool", "content": "not json"}], 5))
    os.environ["COMPACTION_STUCK_K"] = "3"
    check("COMPACTION_STUCK_K env override", compaction_patch._stuck_k() == 3)
    del os.environ["COMPACTION_STUCK_K"]
    check("K back to default after unset", compaction_patch._stuck_k() == 5)

    tmp = Path(tempfile.mkdtemp(prefix="compaction_smoke_v2_"))

    def fresh_messages() -> list[dict]:
        return [
            {"role": "system", "content": "You are a coding agent solving a grid-based puzzle game."},
            {"role": "user", "content": big("Turn 1 state dump.")},
            asst("Maybe the goal is to fill the seven slots with the top-row colors in left-to-right order.",
                 reasoning=big("Thinking about slot ordering hypotheses at length.", 4000),
                 tool_calls=True, call_id="call-11"),
            tool_msg(11, 1, board_changed=True, action="MOUSE", pad=3000),
            asst("That didn't work - nothing changed after filling all the slots in order.\n"
                 "SPACE only decrements the timer without changing the board at all.",
                 reasoning=big("Re-deriving the layout again from ascii.", 4000),
                 tool_calls=True, call_id="call-12"),
            tool_msg(12, 1, board_changed=False, action="SPACE", game_over=True, pad=3000),
            {"role": "user", "content": "Turn 3 state dump (small)." + " z" * 200},
            asst(big("Long derivation notes about the board layout.", 6000)),
            asst("Final note before acting.", reasoning="deep tail thinking"),
            {"role": "user", "content": "Recent turn: pick the next action."},
        ]

    from inference.agent.tool_agent import _estimate_tokens

    print("== D digest-OFF (v2.1 default): pure eviction, zero injection ==")
    agent_d = ToolAgent(model="smoke-test-model")
    agent_d._compaction_state_path = tmp / "sc25-dddd_p0_runtime_state.json"
    d_io = io.StringIO()
    with contextlib.redirect_stdout(d_io):
        trimmed_d = agent_d._trim_messages_for_context(fresh_messages(), tools=None)
    store_d = agent_d._compaction_store
    ev_d = [ln for ln in d_io.getvalue().splitlines()
            if compaction_patch.EVENT_ANCHOR in ln]
    check("digest-OFF: eviction still fires under token pressure",
          store_d.evicted_msgs > 0, f"evicted={store_d.evicted_msgs}")
    check("digest-OFF: request within stock budget (no reserve subtracted)",
          _estimate_tokens({"messages": trimmed_d}) <= agent_d._context_budget_tokens)
    check("digest-OFF: NO digest message in the outbound history",
          not any(compaction_patch._is_digest_message(m) for m in trimmed_d))
    check("digest-OFF: system prompt pinned", trimmed_d[0]["role"] == "system")
    check("digest-OFF: latest-reasoning block pinned (survives the trim)",
          any(m.get("reasoning") == "deep tail thinking" for m in trimmed_d),
          repr([m.get("role") for m in trimmed_d]))
    check("digest-OFF: current-frame carrier pinned (preserve tail)",
          trimmed_d[-1].get("content") == "Recent turn: pick the next action.")
    check("digest-OFF: capture-into-store unchanged (refuted harvested)",
          store_d.ledger.refuted_count() >= 1,
          f"hyps={store_d.ledger.hypotheses}")
    check("digest-OFF: exactly ONE event emitted", len(ev_d) == 1, repr(ev_d))
    rec_d = parse_event(ev_d[0]) if ev_d else {}
    check("digest-OFF: event v=2.1", rec_d.get("v") == "2.1", repr(rec_d))
    check("digest-OFF: digest_tokens=0 on the event",
          rec_d.get("digest_tokens") == "0", repr(rec_d))
    check("digest-OFF: reserve_applied=0 on the event",
          rec_d.get("reserve_applied") == "0", repr(rec_d))
    check("digest-OFF: retained_reasoning_msgs=0 (RETAIN canary holds)",
          rec_d.get("retained_reasoning_msgs") == "0", repr(rec_d))
    check("digest-OFF: eviction class order engaged (ev_episode >= 1)",
          int(rec_d.get("ev_episode", 0)) >= 1, repr(rec_d))
    # The store HAS earned records now (refuted >= 1): v2 would inject a
    # digest on the next trim; v2.1 must NOT — evicting or not.
    d2_io = io.StringIO()
    with contextlib.redirect_stdout(d2_io):
        again_d = agent_d._trim_messages_for_context(list(trimmed_d), tools=None)
    check("digest-OFF: NO digest even with earned records on a later trim",
          not any(compaction_patch._is_digest_message(m) for m in again_d))
    ev_d2 = [ln for ln in d2_io.getvalue().splitlines()
             if compaction_patch.EVENT_ANCHOR in ln]
    check("digest-OFF: EVERY stdout event digest_tokens=0 AND reserve_applied=0",
          all(parse_event(ln).get("digest_tokens") == "0"
              and parse_event(ln).get("reserve_applied") == "0"
              for ln in ev_d + ev_d2))
    sidecar_d = tmp / "sc25-dddd_p0_compaction_events.jsonl"
    d_recs = ([json.loads(ln) for ln in
               sidecar_d.read_text(encoding="utf-8").splitlines()]
              if sidecar_d.is_file() else [])
    check("digest-OFF: sidecar events all digest_tokens=0 reserve_applied=0 v=2.1",
          bool(d_recs) and all(r["digest_tokens"] == 0 and r["reserve_applied"] == 0
                               and r["v"] == "2.1" for r in d_recs),
          repr(d_recs[:1]))

    print("== V5 end-to-end trim UNDER COMPACTION_DIGEST=1 "
          "(v2-restore regression) ==")
    os.environ["COMPACTION_DIGEST"] = "1"
    check("COMPACTION_DIGEST=1 enables the digest channel",
          compaction_patch._digest_enabled() is True)
    on_io = io.StringIO()
    with contextlib.redirect_stdout(on_io):
        compaction_patch.apply()  # idempotent; re-prints the runtime banner
    check("banner shows digest=ON under COMPACTION_DIGEST=1",
          "digest=ON" in on_io.getvalue(), on_io.getvalue()[:400])
    agent = ToolAgent(model="smoke-test-model")
    agent._compaction_state_path = tmp / "sb26-xxxx_p0_runtime_state.json"

    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        trimmed = agent._trim_messages_for_context(fresh_messages(), tools=None)
    event_lines = [ln for ln in stdout.getvalue().splitlines()
                   if compaction_patch.EVENT_ANCHOR in ln]
    store = agent._compaction_store
    check("eviction happened", store.evicted_msgs > 0, f"evicted={store.evicted_msgs}")
    check("system prompt untouched", trimmed[0]["role"] == "system")
    check("NO digest on the first trim (reserve only when earned; records "
          "appeared during a no-reserve trim => deferred)",
          not any(compaction_patch._is_digest_message(m) for m in trimmed[1:]))
    from inference.agent.tool_agent import _estimate_tokens
    total_tokens = _estimate_tokens({"messages": trimmed})
    check("request within stock budget", total_tokens <= agent._context_budget_tokens,
          f"{total_tokens} > {agent._context_budget_tokens}")
    check("ledger refuted >= 1 (region eviction reached the refuting turn)",
          store.ledger.refuted_count() >= 1, f"hyps={store.ledger.hypotheses}")
    check("exactly one COMPACTION event line", len(event_lines) == 1, repr(event_lines))
    ev = parse_event(event_lines[0]) if event_lines else {}
    check("event v=2.1", ev.get("v") == "2.1", repr(ev))
    check("event carries game stem", ev.get("game") == "sb26-xxxx_p0")
    check("event reserve_applied=0 on the earning trim", ev.get("reserve_applied") == "0")
    check("event retain=0", ev.get("retain") == "0")
    check("event retained_reasoning_msgs=0 (inverted RETAIN canary)",
          ev.get("retained_reasoning_msgs") == "0")
    for field in ("ev_episode", "ev_user", "ev_reasoning", "ev_fallback",
                  "stuck_suppressed", "gated_facts"):
        check(f"event field {field} present", field in ev, repr(ev))
    check("event ev_episode >= 1 (stale episode evicted first)",
          int(ev.get("ev_episode", 0)) >= 1, repr(ev))
    check("event ev_user >= 1 (user span evicted)",
          int(ev.get("ev_user", 0)) >= 1, repr(ev))
    check("episodes incremented on evicting trim", store.episodes == 1,
          f"episodes={store.episodes}")
    check("pending reset after event", store.pending_msgs == 0)
    sidecar = tmp / "sb26-xxxx_p0_compaction_events.jsonl"
    check("sidecar written", sidecar.is_file())
    if sidecar.is_file():
        rec = json.loads(sidecar.read_text(encoding="utf-8").splitlines()[0])
        check("sidecar record sane (v=2.1, all new fields)",
              rec["kind"] == "evict_compact" and rec["v"] == "2.1"
              and rec["evicted_msgs"] > 0
              and all(k in rec for k in ("ev_episode", "ev_user", "ev_reasoning",
                                         "ev_fallback", "stuck_suppressed",
                                         "reserve_applied", "gated_facts",
                                         "retain", "retained_reasoning_msgs")),
              repr(rec))

    # second trim: the store has earned the reserve -> digest injected
    stdout2 = io.StringIO()
    with contextlib.redirect_stdout(stdout2):
        again = agent._trim_messages_for_context(list(trimmed), tools=None)
    check("digest injected on the NEXT trim (deferred injection lands)",
          len(again) > 1 and compaction_patch._is_digest_message(again[1]))
    digest = again[1]["content"] if len(again) > 1 else ""
    check("digest header v2 + softened (prior, not proof)",
          "v2" in digest.splitlines()[0] and "treat as prior, not proof" in digest
          and "re-testing is allowed" in digest)
    check("digest header non-quotable memo", "do not quote or restate" in digest)
    check("digest has NO 'do NOT re-verify' directive", "re-verify" not in digest)
    check("digest carries REFUTED line", "REFUTED" in digest)
    check("digest has NO ACTIVE/CONFIRMED hypothesis lines",
          not re.search(r"^(ACTIVE|CONFIRMED) H\d", digest, re.M), digest[:400])
    check("digest never elides (+N more absent)",
          not re.search(r"\+\s*\d+\s+more", digest))
    check("digest has PROGRESS meta line", "PROGRESS:" in digest and "game_overs=" in digest)
    check("digest tokens <= reserve",
          ledger_core.estimate_tokens(digest) <= 400)
    ev2_lines = [ln for ln in stdout2.getvalue().splitlines()
                 if compaction_patch.EVENT_ANCHOR in ln]
    if ev2_lines:
        check("second-trim event has reserve_applied=1",
              parse_event(ev2_lines[0]).get("reserve_applied") == "1",
              repr(ev2_lines))
    else:
        check("second trim evicted nothing -> correctly no event", True)

    # stale-digest hygiene + non-evicting-trim discipline
    evicted_before = store.evicted_msgs
    episodes_before = store.episodes
    with contextlib.redirect_stdout(io.StringIO()):
        third = agent._trim_messages_for_context(list(again), tools=None)
    n_digests = sum(1 for m in third[1:] if compaction_patch._is_digest_message(m))
    check("exactly one digest after re-trim", n_digests == 1, f"n={n_digests}")
    check("digest itself never ingested", store.evicted_msgs == evicted_before,
          f"{store.evicted_msgs} != {evicted_before}")
    tiny_io = io.StringIO()
    with contextlib.redirect_stdout(tiny_io):
        tiny = agent._trim_messages_for_context(
            [{"role": "system", "content": "sys"},
             {"role": "user", "content": "tiny frame"}], tools=None)
    check("non-evicting trim: no event, episodes unchanged",
          compaction_patch.EVENT_ANCHOR not in tiny_io.getvalue()
          and store.episodes == episodes_before)
    check("digest still injected on a non-evicting trim (store persists)",
          len(tiny) > 1 and compaction_patch._is_digest_message(tiny[1]))

    print("== V6 refuted never elided / render order / FACT gating ==")
    s6 = compaction_patch._CompactionStore(None)
    s6.ingest_message({"role": "user", "content": "old dump"})  # has_content
    for i in range(1, 31):
        h = s6.ledger.add_hypothesis(
            f"Hypothesis number {i:02d} about the layout wins the level.",
            "other", step=i, action=i)
        s6.ledger.refute(h, f"observed failure {i:02d}")
    s6.ledger.add_fact("The first durable fact is about the door.")
    s6.ledger.add_fact("The second fact is newer than the first.")
    s6.ledger.add_fact("Maybe the hedged fact should be gated out.")
    os.environ["COMPACTION_RESERVE_TOKENS"] = "300"
    d6 = s6.render_digest()
    os.environ["COMPACTION_RESERVE_TOKENS"] = "400"
    ref_lines = [ln for ln in d6.splitlines() if ln.startswith("REFUTED")]
    check("refuted overflow: one line per record, no count line",
          0 < len(ref_lines) < 30 and not re.search(r"\+\s*\d+\s+more", d6),
          f"n={len(ref_lines)}")
    check("refuted newest-first (H30 renders first)",
          ref_lines and ref_lines[0].startswith("REFUTED H30"), repr(ref_lines[:2]))
    check("refuted overflow drops the OLDEST lines",
          "REFUTED H1 " not in d6 and "REFUTED H2 " not in d6)
    rendered_ids = [int(ln.split()[1][1:]) for ln in ref_lines]
    check("rendered refuted ids are a contiguous newest tail",
          rendered_ids == list(range(30, 30 - len(rendered_ids), -1)),
          repr(rendered_ids))
    check("budget priority: REFUTED starves FACT under overflow",
          "FACT F" not in d6)
    check("gated_fact_count excludes the hedged fact",
          s6.gated_fact_count() == 2, str(s6.gated_fact_count()))
    s6b = compaction_patch._CompactionStore(None)
    s6b.ingest_message({"role": "user", "content": "old dump"})
    s6b.ledger.add_fact("The first durable fact is about the door.")
    s6b.ledger.add_fact("The second fact is newer than the first.")
    s6b.ledger.add_fact("Maybe the hedged fact should be gated out.")
    d6b = s6b.render_digest()  # ample reserve (400), no refuted
    check("hedged FACT gated out of the digest", "hedged fact" not in d6b)
    if "FACT F1" in d6b and "FACT F2" in d6b:
        check("FACT lines newest-first",
              d6b.index("FACT F2") < d6b.index("FACT F1"))
    else:
        check("FACT lines newest-first", False, d6b[-400:])

    print("== V7 empty gate => inject NOTHING ==")
    s7 = compaction_patch._CompactionStore(None)
    check("no content => empty digest", s7.render_digest() == "")
    s7.ingest_message({"role": "user", "content": big("state dump only.", 4000)})
    check("content but zero records => empty digest (no header-only digest)",
          s7.render_digest() == "")
    agent7 = ToolAgent(model="smoke-test-model")
    agent7._compaction_state_path = tmp / "ft09-zzzz_p0_runtime_state.json"
    msgs7 = ([{"role": "system", "content": "sys"}]
             + [{"role": "user", "content": big(f"turn {i} dump.", 6000)}
                for i in range(6)]
             + [{"role": "user", "content": "current frame"}])
    out7_io = io.StringIO()
    with contextlib.redirect_stdout(out7_io):
        out7 = agent7._trim_messages_for_context(msgs7, tools=None)
    check("record-free eviction injects NO digest",
          not any(compaction_patch._is_digest_message(m) for m in out7[1:]))
    ev7 = [ln for ln in out7_io.getvalue().splitlines()
           if compaction_patch.EVENT_ANCHOR in ln]
    check("record-free event has reserve_applied=0 digest_tokens=0",
          len(ev7) == 1 and parse_event(ev7[0]).get("reserve_applied") == "0"
          and parse_event(ev7[0]).get("digest_tokens") == "0", repr(ev7))

    print("== V8 while-stuck behaviors ==")
    # (a) persistence cap DEFERRED outright
    agent8 = ToolAgent(model="smoke-test-model")
    longh: list[dict] = []
    for i in range(35):
        longh.append({"role": "user", "content": f"turn {i} state"})
        longh.append(asst(f"Move block {i} into the container on the right side."))
    longh.append(asst("Trying again.", tool_calls=True, call_id="c-stuck"))
    longh.extend(tool_msg(100 + i, 1, board_changed=False) for i in range(5))
    with contextlib.redirect_stdout(io.StringIO()):
        kept = agent8._keep_recent_history_turns(list(longh), max_turns=30)
    s8 = agent8._compaction_store
    check("stuck: 30-turn cap DEFERRED (no cut)", len(kept) == len(longh),
          f"{len(kept)} != {len(longh)}")
    check("stuck: cap-defer counted in stuck_suppressed", s8.stuck_suppressed == 1)
    check("stuck: nothing ingested by the deferred cap", s8.evicted_msgs == 0)
    longh.append(tool_msg(200, 1, board_changed=True))  # break the streak
    with contextlib.redirect_stdout(io.StringIO()):
        kept2 = agent8._keep_recent_history_turns(list(longh), max_turns=30)
    check("not stuck: cap enforced again", len(kept2) < len(longh))
    check("not stuck: cap evictions captured", s8.evicted_msgs > 0)
    check("cap still bounded",
          sum(1 for m in kept2 if m.get("role") == "assistant") <= 30)

    # (b) budget-forced eviction while stuck: region-aware, captured, NO event,
    # NO digest; (c) counters flush into the next non-stuck event
    agent9 = ToolAgent(model="smoke-test-model")
    agent9._compaction_state_path = tmp / "lp85-yyyy_p0_runtime_state.json"
    s9 = compaction_patch._get_store(agent9)
    s9.ingest_message({"role": "user", "content": "seed old dump"})
    h9 = s9.ledger.add_hypothesis(
        "The lever on the left opens the gate immediately.", "other",
        step=1, action=1)
    s9.ledger.refute(h9, "gate stayed shut")
    check("pre-stuck store renders a digest (would have earned the reserve)",
          s9.render_digest() != "")
    stuck_msgs = (
        [{"role": "system", "content": "sys"},
         {"role": "user", "content": big("old turn dump.", 9000)},
         asst(big("derivation loop musing.", 5000), tool_calls=True, call_id="c9")]
        + [tool_msg(300 + i, 1, board_changed=False, pad=2000) for i in range(5)]
        + [{"role": "user", "content": "current frame"}])
    pend_before = s9.pending_msgs
    out9_io = io.StringIO()
    with contextlib.redirect_stdout(out9_io):
        out9 = agent9._trim_messages_for_context(stuck_msgs, tools=None)
    check("stuck trim: budget-forced eviction still happened (physics)",
          s9.pending_msgs > pend_before, f"pending={s9.pending_msgs}")
    check("stuck trim: NO event emitted",
          compaction_patch.EVENT_ANCHOR not in out9_io.getvalue(),
          out9_io.getvalue()[:200])
    check("stuck trim: NO digest injected (reserve suppressed while stuck)",
          not any(compaction_patch._is_digest_message(m) for m in out9[1:]))
    check("stuck trim: stuck_suppressed counted", s9.stuck_suppressed >= 1)
    check("stuck trim: request still within stock budget",
          _estimate_tokens({"messages": out9}) <= agent9._context_budget_tokens)
    pend_flush = s9.pending_msgs
    out10_io = io.StringIO()
    with contextlib.redirect_stdout(out10_io):
        out10 = agent9._trim_messages_for_context(
            [{"role": "system", "content": "sys"},
             {"role": "user", "content": "fresh frame, actions moved again"}],
            tools=None)
    ev10 = [ln for ln in out10_io.getvalue().splitlines()
            if compaction_patch.EVENT_ANCHOR in ln]
    check("next non-stuck trim: ONE flush event", len(ev10) == 1, repr(ev10))
    if ev10:
        rec10 = parse_event(ev10[0])
        check("flush event carries the stuck-accumulated counts",
              int(rec10.get("evicted_msgs", 0)) == pend_flush, repr(rec10))
        check("flush event reserve_applied=1 (digest now earned + not stuck)",
              rec10.get("reserve_applied") == "1", repr(rec10))
        check("flush event reports stuck_suppressed",
              int(rec10.get("stuck_suppressed", 0)) >= 1, repr(rec10))
    check("digest injected once un-stuck",
          len(out10) > 1 and compaction_patch._is_digest_message(out10[1]))
    check("pending reset after flush", s9.pending_msgs == 0)

    # back to the v2.1 default (digest OFF) for the remaining sections
    del os.environ["COMPACTION_DIGEST"]
    check("digest OFF again after unset", compaction_patch._digest_enabled() is False)

    print("== V9 RETAIN sub-arm ==")
    msgs = [{"role": "assistant", "content": "c", "reasoning": "deep thought"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "d"}]
    mirrored = compaction_patch._mirror_reasoning(msgs)
    check("reasoning mirrored to reasoning_content",
          msgs[0].get("reasoning_content") == "deep thought")
    check("mirror count correct", mirrored == 1)
    check("no mirror without reasoning", "reasoning_content" not in msgs[2])
    sub_code = (
        "import os, sys\n"
        f"sys.path.insert(0, {str(DATASET)!r})\n"
        f"sys.path.insert(0, {str(BUNDLE / 'ARC3-Inference')!r})\n"
        "os.environ['LOCAL_ANALYZER_CONTEXT_WINDOW'] = '4096'\n"
        "os.environ['LOCAL_ANALYZER_MODEL_ID'] = 'smoke-test-model'\n"
        "os.environ['LOCAL_ANALYZER_BASE_URL'] = 'http://127.0.0.1:9/v1'\n"
        "os.environ['COMPACTION'] = '1'\n"
        "os.environ['COMPACTION_RETAIN'] = '1'\n"
        "from inference.agent.tool_agent import ToolAgent\n"
        "import compaction_patch\n"
        "assert compaction_patch.apply() is True\n"
        "assert ToolAgent._chat_completion.__name__ == 'compaction_chat_completion'\n"
        "print('RETAIN_SUBPROC_OK')\n"
    )
    sub = subprocess.run([sys.executable, "-c", sub_code],
                         capture_output=True, text=True, cwd=str(REPO))
    check("COMPACTION_RETAIN=1 installs the chat wrapper (subprocess)",
          sub.returncode == 0 and "RETAIN_SUBPROC_OK" in sub.stdout,
          (sub.stderr or sub.stdout)[-500:])
    check("RETAIN=1 banner says mirroring=ON", "mirroring=ON" in sub.stdout,
          sub.stdout[-500:])
    check("RETAIN subprocess banner still digest=OFF (flags independent)",
          "digest=OFF" in sub.stdout, sub.stdout[-500:])

    print("== V10 vanilla fallback on wrapper failure ==")
    agent_bad = ToolAgent(model="smoke-test-model")
    agent_bad._compaction_store = object()  # poisons _get_store -> AttributeError
    bad_io = io.StringIO()
    with contextlib.redirect_stdout(bad_io):
        out_bad = agent_bad._trim_messages_for_context(fresh_messages(), tools=None)
    check("poisoned store: trim falls back to stock (no crash, system first)",
          out_bad and out_bad[0]["role"] == "system")
    check("poisoned store: no digest, no event",
          not any(compaction_patch._is_digest_message(m) for m in out_bad[1:])
          and compaction_patch.EVENT_ANCHOR not in bad_io.getvalue())
    check("poisoned store: vanilla trim still fits budget",
          _estimate_tokens({"messages": out_bad}) <= agent_bad._context_budget_tokens)

    print("== V11 source discipline ==")
    src = (DATASET / "compaction_patch.py").read_text(encoding="utf-8")
    code_only = "\n".join(ln for ln in src.splitlines()
                          if not ln.lstrip().startswith("#"))
    check("no threading/locks in the patch",
          "import threading" not in code_only and "Lock(" not in code_only)
    check("no HTTP client in the eviction path (zero LLM calls)",
          not re.search(r"^\s*(import|from)\s+(requests|urllib|http\b|httpx|openai|anthropic)",
                        code_only, re.M))
    check("COMPACTION_DIGEST defaults to 0 in source",
          'os.environ.get("COMPACTION_DIGEST", "0")' in src)
    import hashlib
    sha_ds = hashlib.sha256((DATASET / "ledger_core.py").read_bytes()).hexdigest()
    sha_twin = hashlib.sha256(
        (REPO / "duck_eval" / "ledger" / "ledger_core.py").read_bytes()).hexdigest()
    check("ledger_core.py NOT modified (byte-identical to canonical twin)",
          sha_ds == sha_twin, f"{sha_ds[:12]} != {sha_twin[:12]}")

    print("== V12 builder --compaction ==")
    result = subprocess.run(
        [sys.executable, str(HERE / "build_eval_notebook.py"), "--compaction"],
        capture_output=True, text=True, cwd=str(REPO))
    check("builder exits 0", result.returncode == 0,
          result.stderr[-800:] if result.returncode else "")
    out_nb = REPO / "notebooks" / "duckcompaction-eval" / "arc3-duck-compaction-eval.ipynb"
    check("eval notebook written", out_nb.is_file())
    if out_nb.is_file():
        nb = json.loads(out_nb.read_text(encoding="utf-8"))
        cell2 = "".join(nb["cells"][2]["source"])
        cell12 = "".join(nb["cells"][12]["source"])
        check("cell 2 forces offline bench", "WARPACK_FORCE_OFFLINE_BENCH" in cell2)
        check("cell 2 sets COMPACTION=1 (stamp cell)",
              'os.environ["COMPACTION"] = "1"' in cell2)
        check("cell 2 seed banner", "COMPACTION_EVAL_SEED" in cell2)
        check("cell 12 imports compaction_patch", "import compaction_patch" in cell12)
        check("cell 12 has NO warpack graft", "import warpack_patch" not in cell12)
        check("(f) continuation default rides", "import continuation_patch" in cell12)
        meta = json.loads((out_nb.parent / "kernel-metadata.json").read_text(encoding="utf-8"))
        check("kernel id correct", meta["id"] == "canivel/arc3-duck-compaction-eval")
    check("dataset patch is v2.1 (VERSION stamp)", 'VERSION = "v2.1"' in src)
    check("dataset patch prints the v2.1 ACTIVE banner",
          'f"compaction {VERSION}: ACTIVE "' in src)

    print(f"\ncompaction v2.1 smoke: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(run())
