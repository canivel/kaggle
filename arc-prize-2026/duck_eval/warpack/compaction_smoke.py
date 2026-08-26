"""A22 compaction + retained-reasoning CPU smoke test -- no GPU, no LLM.

Run from the repo root:
    .venv/Scripts/python.exe duck_eval/warpack/compaction_smoke.py

Covers (feedback_test_before_submit: ALWAYS runtime-test agent code before
any Kaggle push):
  C0  flag gate: apply() is a no-op without COMPACTION=1; kill switch
      COMPACTION_DISABLE=1 wins over the flag; idempotent re-apply
  C1  token-budget eviction -> mechanical digest injected as messages[1]
      (role user, DIGEST_MARKER prefix), request stays within the stock
      context budget (reserve discipline)
  C2  digester content: ledger_core picked up goal->refuted + FACT lines from
      evicted assistant text (reasoning included); action-effect counts +
      level/game-over progress from evicted tool payloads
  C3  COMPACTION event: greppable stdout line + per-game jsonl sidecar;
      episodes increment; pending counters reset
  C4  stale-digest hygiene: exactly ONE digest per request, digests are never
      ingested (no feedback loop), digest tokens <= reserve
  C5  persistence-cap eviction (_persistent_history_messages 30-turn cap) is
      captured too
  C6  RETAIN: assistant `reasoning` mirrored to `reasoning_content` outbound
      (_mirror_reasoning) + _chat_completion wrapper installed
  C7  builder: --compaction eval notebook carries the graft, the COMPACTION=1
      flag line, and the (f) continuation default block
"""
from __future__ import annotations

import contextlib
import io
import json
import os
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
os.environ.pop("COMPACTION", None)
os.environ.pop("COMPACTION_DISABLE", None)

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


def fresh_messages() -> list[dict]:
    return [
        {"role": "system", "content": "You are a coding agent solving a grid-based puzzle game."},
        {"role": "user", "content": big("Turn 1 state dump.")},
        {"role": "assistant",
         "content": "Maybe the goal is to fill the seven slots with the top-row colors in left-to-right order.",
         "reasoning": big("Thinking about slot ordering hypotheses at length.", 4000),
         "tool_calls": [{"id": "call-11", "type": "function",
                         "function": {"name": "python", "arguments": "{}"}}]},
        tool_msg(11, 1, board_changed=True, action="MOUSE", pad=3000),
        {"role": "assistant",
         "content": ("That didn't work - nothing changed after filling all the slots in order.\n"
                     "SPACE only decrements the timer without changing the board at all."),
         "reasoning": big("Re-deriving the layout again from ascii.", 4000),
         "tool_calls": [{"id": "call-12", "type": "function",
                         "function": {"name": "python", "arguments": "{}"}}]},
        tool_msg(12, 1, board_changed=False, action="SPACE", game_over=True, pad=3000),
        {"role": "user", "content": "Recent turn: pick the next action."},
    ]


def run() -> int:
    print("== C0 flag gate / kill switch ==")
    stock_trim = ToolAgent._trim_messages_for_context
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
    check("apply() True with COMPACTION=1", compaction_patch.apply(bm) is True)
    check("bm.label stamped", bm.label == "duck-compaction-v1")
    check("methods patched", ToolAgent._trim_messages_for_context is not stock_trim)
    check("idempotent re-apply", compaction_patch.apply() is True)

    print("== C1/C2/C3 eviction -> digest + event ==")
    agent = ToolAgent(model="smoke-test-model")
    tmp = Path(tempfile.mkdtemp(prefix="compaction_smoke_"))
    agent._compaction_state_path = tmp / "sb26-xxxx_p0_runtime_state.json"

    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        trimmed = agent._trim_messages_for_context(fresh_messages(), tools=None)
    event_lines = [ln for ln in stdout.getvalue().splitlines()
                   if compaction_patch.EVENT_ANCHOR in ln]

    store = agent._compaction_store
    check("eviction happened", store.evicted_msgs > 0,
          f"evicted={store.evicted_msgs}")
    check("digest injected at messages[1]",
          len(trimmed) > 1 and compaction_patch._is_digest_message(trimmed[1]))
    digest = trimmed[1]["content"] if len(trimmed) > 1 else ""
    check("system prompt untouched",
          trimmed[0]["role"] == "system")
    from inference.agent.tool_agent import _estimate_tokens
    total_tokens = _estimate_tokens({"messages": trimmed})
    check("request within stock budget",
          total_tokens <= agent._context_budget_tokens,
          f"{total_tokens} > {agent._context_budget_tokens}")

    check("ledger refuted >= 1", store.ledger.refuted_count() >= 1,
          f"hyps={store.ledger.hypotheses}")
    check("ledger fact captured (SPACE=timer)",
          any("SPACE" in f["statement"] for f in store.ledger.facts),
          f"facts={store.ledger.facts}")
    check("digest carries REFUTED line", "REFUTED" in digest)
    check("action-effect counts folded",
          "MOUSE" in store.action_counts and store.action_counts["MOUSE"][0] >= 1,
          f"counts={store.action_counts}")
    check("game_over folded", store.game_overs >= 1)
    check("digest has PROGRESS line", "PROGRESS:" in digest and "game_overs=" in digest)

    check("exactly one COMPACTION event line", len(event_lines) == 1,
          repr(event_lines))
    check("event carries game stem", event_lines and "game=sb26-xxxx_p0" in event_lines[0])
    sidecar = tmp / "sb26-xxxx_p0_compaction_events.jsonl"
    check("sidecar written", sidecar.is_file())
    if sidecar.is_file():
        rec = json.loads(sidecar.read_text(encoding="utf-8").splitlines()[0])
        check("sidecar record sane",
              rec["kind"] == "evict_compact" and rec["evicted_msgs"] > 0)
    check("pending reset after event", store.pending_msgs == 0)

    print("== C4 stale-digest hygiene ==")
    evicted_before = store.evicted_msgs
    with contextlib.redirect_stdout(io.StringIO()):
        again = agent._trim_messages_for_context(list(trimmed), tools=None)
    n_digests = sum(1 for m in again[1:] if compaction_patch._is_digest_message(m))
    check("exactly one digest after re-trim", n_digests == 1, f"n={n_digests}")
    check("digest itself never ingested", store.evicted_msgs == evicted_before,
          f"{store.evicted_msgs} != {evicted_before}")
    import ledger_core
    check("digest tokens <= reserve",
          ledger_core.estimate_tokens(again[1]["content"]) <= 400)

    print("== C5 persistence-cap eviction captured ==")
    agent2 = ToolAgent(model="smoke-test-model")
    long_history: list[dict] = [{"role": "system", "content": "sys"}]
    for i in range(40):
        long_history.append({"role": "user", "content": f"turn {i} state"})
        long_history.append({"role": "assistant",
                             "content": f"Maybe the goal is to move block {i} into the container on the right side.",
                             "reasoning": f"thinking {i}"})
    with contextlib.redirect_stdout(io.StringIO()):
        persisted = agent2._persistent_history_messages(long_history, tools=None)
    store2 = agent2._compaction_store
    check("30-turn-cap evictions captured", store2.evicted_msgs >= 10,
          f"evicted={store2.evicted_msgs}")
    assistant_kept = sum(1 for m in persisted if m.get("role") == "assistant")
    check("persistence cap still enforced", assistant_kept <= 30,
          f"kept={assistant_kept}")

    print("== C6 retained reasoning ==")
    msgs = [{"role": "assistant", "content": "c", "reasoning": "deep thought"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "d"}]
    mirrored = compaction_patch._mirror_reasoning(msgs)
    check("reasoning mirrored to reasoning_content",
          msgs[0].get("reasoning_content") == "deep thought")
    check("mirror count correct", mirrored == 1)
    check("no mirror without reasoning", "reasoning_content" not in msgs[2])
    check("_chat_completion wrapper installed",
          ToolAgent._chat_completion.__name__ == "compaction_chat_completion")

    print("== C7 builder --compaction ==")
    result = subprocess.run(
        [sys.executable, str(HERE / "build_eval_notebook.py"), "--compaction"],
        capture_output=True, text=True, cwd=str(REPO))
    check("builder exits 0", result.returncode == 0,
          result.stderr[-800:] if result.returncode else "")
    out_nb = REPO / "notebooks" / "duckcompaction-eval" / "arc3-duck-compaction-eval.ipynb"
    check("eval notebook written", out_nb.is_file())
    if out_nb.is_file():
        nb_text = out_nb.read_text(encoding="utf-8")
        nb = json.loads(nb_text)
        cell2 = "".join(nb["cells"][2]["source"])
        cell12 = "".join(nb["cells"][12]["source"])
        check("cell 2 forces offline bench", "WARPACK_FORCE_OFFLINE_BENCH" in cell2)
        check("cell 2 sets COMPACTION=1", 'os.environ["COMPACTION"] = "1"' in cell2)
        check("cell 2 seed banner", "COMPACTION_EVAL_SEED" in cell2)
        check("cell 12 imports compaction_patch", "import compaction_patch" in cell12)
        check("cell 12 has NO warpack graft", "import warpack_patch" not in cell12)
        check("(f) continuation default rides", "import continuation_patch" in cell12)
        meta = json.loads((out_nb.parent / "kernel-metadata.json").read_text(encoding="utf-8"))
        check("kernel id correct", meta["id"] == "canivel/arc3-duck-compaction-eval")

    print(f"\ncompaction smoke: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(run())
