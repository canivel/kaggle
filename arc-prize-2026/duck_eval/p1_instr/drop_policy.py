"""ITEM 4 (R24 §3.4 / §5.3) — §6.1 restated as a DROP-POLICY invariant, with a checker.

WHY §6.1 AS WRITTEN IS SELF-VOIDING
-----------------------------------
Proposal §6.1 demands that "`evicted_chars` and the trimmed-message sequence
emitted by `_trim_messages_for_context()` be byte-identical to baseline on every
event". But that function takes `tools` and feeds them into the token estimate:

    tool_agent.py:1749-1767  _trim_messages_for_context(messages, *, tools=...)
        budget_tokens = max(1, self._context_budget_tokens - max(0, extra))
        while history and self._estimate_request_input_tokens(
                [system_message, *history], tools=tools) > budget_tokens: ...
    tool_agent.py:1672-1682  _estimate_request_input_tokens -> payload includes tools
    tool_agent.py:537-542    _estimate_tokens = (len(json.dumps(payload))+2)//3

P1 necessarily edits the system prompt and the tool schema (item 3). Both are
inside the estimated payload, so the estimate moves, so the eviction boundary
moves, so the trimmed byte sequence differs — on event 1. The arm voids itself
before it measures anything.

§6.1' — THE DROP-POLICY INVARIANT (restatement)
-----------------------------------------------
P1 may change the CONTENT the eviction policy operates on; it may not change the
POLICY. Formally, the arm holds iff all five clauses hold:

  (a) POLICY CODE FROZEN. `_trim_messages_for_context`, `_drop_oldest_history_block`
      and `_drop_until_first_user_message` are byte-identical to baseline
      (sha256 of each function's source, pinned in this module and audited).
  (b) POLICY PARAMETERS FROZEN. `preserve_recent`, `extra_safety_tokens`,
      `_reply_reserve_tokens`, `_request_safety_margin_tokens`,
      `LOCAL_ANALYZER_CONTEXT_WINDOW` and `LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS`
      equal baseline.
  (c) COMPENSATED BUDGET. The arm's `_context_budget_tokens` is baseline's plus
      the PRE-REGISTERED constant `delta_tokens` = est(P1 system + P1 tools)
      − est(baseline system + baseline tools), computed OFFLINE from the patch
      surface manifest and published in the prereg. This is the only permitted
      numeric change and its sole purpose is to leave the HISTORY budget
      identical.
  (d) DROP-TRACE IDENTITY. The observed drop trace — the ordered list of
      (turn_ordinal, drop_event_ordinal, dropped_roles, n_dropped, kept_len) —
      is identical to baseline's. Not the bytes of the messages: the DECISIONS.
  (e) NO NEW REMOVAL PATH. P1 adds no code that removes, rewrites, summarises,
      reorders or compacts any message. P1 adds a store OUTSIDE the window.

  Residual, declared: `_estimate_tokens` floors at //3, so a compensated budget
  reproduces the baseline boundary up to +-1 token. Clause (d) is therefore
  VERIFIED empirically from the recorded traces rather than assumed; a single
  divergence VOIDS the arm for §0 exactly as §6.1 intended.

Usage:
  uv run python duck_eval/p1_instr/drop_policy.py
"""
from __future__ import annotations

import ast
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_surface import MANIFEST_PATH  # noqa: E402
from pull_io import iter_sections, load_pull  # noqa: E402
from sandbox_facts import PROMPTS_SRC, TOOL_AGENT_SRC  # noqa: E402

# Baseline env, read from the pull's own taaf_setup_env.json in `main`.
DEFAULT_CONTEXT_WINDOW = 32768
REQUEST_SAFETY_MARGIN_TOKENS = 512   # tool_agent.py::_REQUEST_SAFETY_MARGIN_TOKENS
REPLY_RESERVE_DEFAULT = 512          # self._max_output_tokens or 512

POLICY_FUNCTIONS = (
    "_trim_messages_for_context",
    "_drop_oldest_history_block",
    "_drop_until_first_user_message",
    "_estimate_request_input_tokens",
)


# --------------------------------------------------------------------------
# (a) policy-code freeze: hash the fork's own function sources
# --------------------------------------------------------------------------
def policy_source_hashes() -> dict[str, str]:
    text = TOOL_AGENT_SRC.read_text(encoding="utf-8")
    tree = ast.parse(text)
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in POLICY_FUNCTIONS:
            src = ast.get_source_segment(text, node) or ""
            out[node.name] = hashlib.sha256(src.encode("utf-8")).hexdigest()
    # module-level _estimate_tokens too
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_estimate_tokens":
            src = ast.get_source_segment(text, node) or ""
            out["_estimate_tokens"] = hashlib.sha256(src.encode("utf-8")).hexdigest()
    return out


# --------------------------------------------------------------------------
# verbatim re-implementations (kept honest by the hashes above)
# --------------------------------------------------------------------------
def estimate_tokens(value: Any) -> int:
    """tool_agent.py:537-542, verbatim."""
    try:
        rendered = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    except TypeError:
        rendered = str(value)
    return max(1, (len(rendered) + 2) // 3)


def estimate_request_input_tokens(messages: list[dict[str, Any]], *,
                                  tools: list[dict[str, Any]] | None = None) -> int:
    payload: dict[str, Any] = {"messages": messages}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    return estimate_tokens(payload)


def drop_oldest_history_block(history: list[dict[str, Any]], *, preserve_recent: int
                              ) -> tuple[bool, list[str]]:
    """tool_agent.py:1683-1698, verbatim, plus a record of what it removed."""
    removed: list[str] = []
    removable = len(history) - preserve_recent
    if removable <= 0:
        return False, removed
    first = history.pop(0)
    removed.append(str(first.get("role", "")).strip())
    first_role = str(first.get("role", "")).strip()
    if first_role in {"assistant", "tool"}:
        while history and history[0].get("role") == "tool" and len(history) > preserve_recent:
            removed.append(str(history.pop(0).get("role", "")).strip())
        return True, removed
    while history and history[0].get("role") == "tool" and len(history) > preserve_recent:
        removed.append(str(history.pop(0).get("role", "")).strip())
    while history and history[0].get("role") != "user" and len(history) > preserve_recent:
        removed.append(str(history.pop(0).get("role", "")).strip())
    return True, removed


def drop_until_first_user_message(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trimmed = list(history)
    while trimmed and str(trimmed[0].get("role", "")).strip() != "user":
        trimmed.pop(0)
    return trimmed


@dataclass(frozen=True)
class DropEvent:
    turn: int
    event: int
    roles: tuple[str, ...]
    kept_len: int


def trim_with_trace(messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None,
                    budget_tokens: int, turn: int, preserve_recent: int = 1
                    ) -> tuple[list[dict[str, Any]], list[DropEvent]]:
    """tool_agent.py:1749-1767 with a drop trace attached."""
    trace: list[DropEvent] = []
    if not messages:
        return [], trace
    system_message = messages[0]
    history = list(messages[1:])
    event = 0
    while history and estimate_request_input_tokens(
            [system_message, *history], tools=tools) > budget_tokens:
        moved, removed = drop_oldest_history_block(history, preserve_recent=preserve_recent)
        if not moved:
            break
        trace.append(DropEvent(turn, event, tuple(removed), len(history)))
        event += 1
    history = drop_until_first_user_message(history)
    return [system_message, *history], trace


def compare_traces(base: list[DropEvent], arm: list[DropEvent]) -> dict[str, object]:
    if base == arm:
        return {"identical": True, "n_events": len(base), "verdict": "HOLDS"}
    for i, (b, a) in enumerate(zip(base, arm)):
        if b != a:
            return {"identical": False, "first_divergence_index": i,
                    "baseline": b.__dict__, "arm": a.__dict__,
                    "verdict": "VOID"}
    return {"identical": False, "first_divergence_index": min(len(base), len(arm)),
            "n_base": len(base), "n_arm": len(arm), "verdict": "VOID"}


# --------------------------------------------------------------------------
# (c) the compensating token delta, computed from the patch surface
# --------------------------------------------------------------------------
def _literal_of_source_line(line: str) -> str:
    return ast.literal_eval(line.strip().rstrip(","))


def rendered_surface_deltas() -> dict[str, list[tuple[str, str]]]:
    """(baseline_text, p1_text) pairs, rendered (not source-escaped), by channel."""
    man = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    out: dict[str, list[tuple[str, str]]] = {"system_prompt": [], "tool_schema": []}
    for e in man["entries"]:
        if e["action"] != "PATCH":
            continue
        try:
            b = _literal_of_source_line(str(e["baseline"]))
            p = _literal_of_source_line(str(e["p1"]))
        except (SyntaxError, ValueError):
            continue
        out.setdefault(str(e["channel"]), []).append((b, p))
    return out


def baseline_tools_payload() -> list[dict[str, Any]]:
    """Reconstruct the exact `_tools()` payload from the fork source."""
    text = TOOL_AGENT_SRC.read_text(encoding="utf-8")
    tree = ast.parse(text)
    desc = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "_PYTHON_TOOL_DESCRIPTION"
                for t in node.targets):
            desc = ast.literal_eval(node.value)
    code_desc = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_tools":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str) \
                        and sub.value.startswith("Python code to run."):
                    code_desc = sub.value
    if desc is None or code_desc is None:
        raise RuntimeError("could not reconstruct _tools() payload from the fork")
    return [{
        "type": "function",
        "function": {
            "name": "python",
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": code_desc}},
                "required": ["code"],
            },
        },
    }]


def p1_tools_payload() -> list[dict[str, Any]]:
    tools = json.loads(json.dumps(baseline_tools_payload()))
    fn = tools[0]["function"]
    for b, p in rendered_surface_deltas().get("tool_schema", []):
        if b in fn["description"]:
            fn["description"] = fn["description"].replace(b, p, 1)
        cd = fn["parameters"]["properties"]["code"]["description"]
        if b in cd:
            fn["parameters"]["properties"]["code"]["description"] = cd.replace(b, p, 1)
    return tools


def p1_system_prompt(baseline_prompt: str) -> tuple[str, list[str]]:
    applied: list[str] = []
    out = baseline_prompt
    for b, p in rendered_surface_deltas().get("system_prompt", []):
        if b in out:
            out = out.replace(b, p, 1)
            applied.append(b.strip()[:48])
        elif b.rstrip("\n") in out:
            # transcript sections are `.strip()`ed, so the LAST addendum line
            # loses its trailing newline (tool_agent.py:596)
            out = out.replace(b.rstrip("\n"), p.rstrip("\n"), 1)
            applied.append(b.strip()[:48])
    return out, applied


def token_delta(baseline_prompt: str) -> dict[str, int]:
    base_tools = baseline_tools_payload()
    arm_tools = p1_tools_payload()
    arm_prompt, _ = p1_system_prompt(baseline_prompt)
    base = estimate_request_input_tokens(
        [{"role": "system", "content": baseline_prompt}], tools=base_tools)
    arm = estimate_request_input_tokens(
        [{"role": "system", "content": arm_prompt}], tools=arm_tools)
    return {
        "baseline_tokens": base,
        "p1_tokens": arm,
        "delta_tokens": arm - base,
        "baseline_prompt_chars": len(baseline_prompt),
        "p1_prompt_chars": len(arm_prompt),
    }


# --------------------------------------------------------------------------
# runtime recorder the screen installs (real code, emitted for the patch bundle)
# --------------------------------------------------------------------------
RECORDER_PATCH_SOURCE = '''\
# --- P1 drop-policy recorder (install BEFORE the P1 namespace patch) ---------
# Wraps ToolAgent._trim_messages_for_context to emit the §6.1' drop trace.
# Writes one JSONL record per trim event to $P1_DROP_TRACE (per game).
import json, os, threading
from inference.agent import tool_agent as _ta

_P1_TRACE_LOCK = threading.Lock()
_P1_TRACE_PATH = os.environ.get("P1_DROP_TRACE", "")
_P1_TURN = {"n": 0}


def _p1_install_drop_recorder():
    original_trim = _ta.ToolAgent._trim_messages_for_context
    original_drop = _ta.ToolAgent._drop_oldest_history_block

    def traced_drop(self, history, *, preserve_recent):
        before = len(history)
        head = [str(m.get("role", "")) for m in history[:4]]
        moved = original_drop(self, history, preserve_recent=preserve_recent)
        if moved and _P1_TRACE_PATH:
            rec = {"turn": _P1_TURN["n"], "n_dropped": before - len(history),
                   "head_roles": head, "kept_len": len(history)}
            with _P1_TRACE_LOCK:
                with open(_P1_TRACE_PATH, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\\n")
        return moved

    def traced_trim(self, messages, **kwargs):
        _P1_TURN["n"] += 1
        return original_trim(self, messages, **kwargs)

    _ta.ToolAgent._drop_oldest_history_block = traced_drop
    _ta.ToolAgent._trim_messages_for_context = traced_trim
    print("p1_instr: drop-policy recorder installed ->", _P1_TRACE_PATH or "(disabled)")


_p1_install_drop_recorder()
'''


# --------------------------------------------------------------------------
# offline validation on real transcript content
# --------------------------------------------------------------------------
def reconstruct_messages(transcript_path: Path, *, max_messages: int = 400
                         ) -> tuple[str, list[dict[str, Any]]]:
    """Length-faithful message reconstruction from a real transcript."""
    text = transcript_path.read_text(encoding="utf-8", errors="replace")
    system = ""
    msgs: list[dict[str, Any]] = []
    for label, body in iter_sections(text):
        if label == "SYSTEM PROMPT" and not system:
            system = body.strip()
        elif label == "USER PROMPT":
            msgs.append({"role": "user", "content": body.strip()})
        elif label.startswith("TOOL CALL: python"):
            msgs.append({"role": "assistant", "content": None, "tool_calls": [
                {"id": "python", "type": "function",
                 "function": {"name": "python",
                              "arguments": json.dumps({"code": body.strip()})}}]})
        elif label.startswith("TOOL RESULT: python"):
            msgs.append({"role": "tool", "tool_call_id": "python",
                         "content": body.strip()})
        if len(msgs) >= max_messages:
            break
    return system, msgs


def validate_on_pull(pull_name: str = "war_eval_v1", n_games: int = 5) -> dict[str, object]:
    pull = load_pull(pull_name)
    results = []
    for i, (gid, tpath) in enumerate(pull.iter_transcripts()):
        if i >= n_games:
            break
        system, msgs = reconstruct_messages(tpath)
        if not system or len(msgs) < 10:
            continue
        base_tools = baseline_tools_payload()
        arm_tools = p1_tools_payload()
        arm_system, applied = p1_system_prompt(system)
        td = token_delta(system)
        budget = DEFAULT_CONTEXT_WINDOW - REPLY_RESERVE_DEFAULT - REQUEST_SAFETY_MARGIN_TOKENS

        base_trace: list[DropEvent] = []
        naive_trace: list[DropEvent] = []
        comp_trace: list[DropEvent] = []
        # replay the conversation turn by turn, exactly as `analyze` does
        for turn in range(1, len(msgs) + 1):
            window = msgs[:turn]
            _, tb = trim_with_trace([{"role": "system", "content": system}, *window],
                                    tools=base_tools, budget_tokens=budget, turn=turn)
            _, tn = trim_with_trace([{"role": "system", "content": arm_system}, *window],
                                    tools=arm_tools, budget_tokens=budget, turn=turn)
            _, tc = trim_with_trace([{"role": "system", "content": arm_system}, *window],
                                    tools=arm_tools,
                                    budget_tokens=budget + td["delta_tokens"], turn=turn)
            base_trace += tb
            naive_trace += tn
            comp_trace += tc
        results.append({
            "game": gid,
            "n_messages": len(msgs),
            "surface_strings_applied": len(applied),
            "delta_tokens": td["delta_tokens"],
            "n_drop_events_baseline": len(base_trace),
            "naive_6_1": compare_traces(base_trace, naive_trace),
            "compensated_6_1prime": compare_traces(base_trace, comp_trace),
        })
    return {"pull": pull_name, "games": results}


def main(argv: list[str]) -> int:
    print("=== §6.1' drop-policy invariant ===")
    print("(a) policy source hashes (pin these in the S2 prereg):")
    for k, v in sorted(policy_source_hashes().items()):
        print(f"    {k:34} {v}")

    pull = load_pull("war_eval_v1")
    gid, tpath = next(pull.iter_transcripts())
    system, _ = reconstruct_messages(tpath)
    td = token_delta(system)
    print(f"\n(c) compensating delta, from {tpath.name}:")
    print(f"    {json.dumps(td)}")

    print("\n(d) drop-trace identity, replayed on real transcript content:")
    rep = validate_on_pull("war_eval_v1", n_games=5)
    naive_void = 0
    comp_hold = 0
    for r in rep["games"]:  # type: ignore[index]
        print(f"    {r['game']:16} msgs={r['n_messages']:4d} "
              f"drops={r['n_drop_events_baseline']:4d} "
              f"naive §6.1 -> {r['naive_6_1']['verdict']:5}  "
              f"§6.1'(compensated) -> {r['compensated_6_1prime']['verdict']}")
        naive_void += r["naive_6_1"]["verdict"] == "VOID"
        comp_hold += r["compensated_6_1prime"]["verdict"] == "HOLDS"
    n = len(rep["games"])  # type: ignore[arg-type]
    print(f"\n    naive §6.1 self-voids on {naive_void}/{n} games; "
          f"§6.1' compensated holds on {comp_hold}/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
