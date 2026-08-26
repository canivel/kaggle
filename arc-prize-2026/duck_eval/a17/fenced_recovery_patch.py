"""Fenced-python tool-call recovery — A17 canary v4 harness graft.

Design refs:
  - runs/kernel_pulls/a17_canary_v3/analysis.md: Qwen2.5-VL-72B-AWQ under the
    real ~31k-token duck context emitted markdown-fenced ```python blocks
    instead of hermes <tool_call> markup on 1200/1200 LLM responses ->
    tool_call_count=0 everywhere -> 0 actions, all games gave_up at 0.00.
  - runs/a17_recovery_replay/recovery_report.json: extracting the fenced block
    as a python tool call recovers 434/436 analysis turns (99.5%) on the
    recorded v3 traffic; every recovered turn is exactly one ast-valid block.
  - learnings/war_room/research_2026-07-26.md: known Qwen tool-format
    pathology (bare hermes collapses to fenced code under verbose prompts).

THE FIX: extend the harness's existing markup-recovery path (tool_agent
``_recover_tool_calls_from_markup`` and its ``_contains_tool_call_markup``
gate) so that when a response carries NO native tool calls and NO tool-call
markup but DOES carry a fenced python block, the concatenated fenced code is
recovered as a single ``python`` tool call. The recovered call flows through
the stock validation/execution path unchanged; transcripts show it as
``tool_calls_recovered_from_markup: yes``.

Recovery rule (identical to the offline replay that produced 99.5%):
  1. collect ```python blocks, plus untagged ``` blocks that ast-parse;
  2. concatenate in order -> candidate body;
  3. recover ONLY if the body ast-parses (a non-parsing body is left alone ->
     stock no-tool-call turn, exactly as v3 behaved).

House pattern (mirrors duck_eval/continuation/continuation_patch.py): VERSION
marker, env kill switch, blanket-guarded apply, runtime banner, vanilla
fallback on ANY failure (worst case: stock v3 behavior, never worse). No
threading, no locks (feedback_lock_deadlock.md).
"""
from __future__ import annotations

import ast
import json
import logging
import os
import re

log = logging.getLogger("fenced_recovery")

VERSION = "v1"

_FENCE_RE = re.compile(r"```(python)?[ \t]*\n(.*?)```", re.S)

# Running count of recovered turns; printed every _HITS_LOG_EVERY recoveries so
# the kernel log carries post-run evidence (grep 'fenced-recovery').
HITS = 0
_HITS_LOG_EVERY = 25


def _fenced_python_body(*chunks: str) -> str:
    """Concatenated fenced-python body per the replay rule, or '' if none/invalid."""
    blocks: list[str] = []
    for chunk in chunks:
        if not chunk or not chunk.strip():
            continue
        for lang, code in _FENCE_RE.findall(chunk):
            if lang == "python":
                blocks.append(code)
            else:
                try:
                    ast.parse(code)
                except SyntaxError:
                    continue
                blocks.append(code)
    if not blocks:
        return ""
    body = "\n".join(blocks)
    try:
        ast.parse(body)
    except SyntaxError:
        return ""
    return body


def _strip_fenced_python(text: str) -> str:
    """Remove exactly the fenced blocks the recovery rule would consume."""
    if not text or not _fenced_python_body(text):
        return text

    def _drop(match: re.Match) -> str:
        lang, code = match.group(1), match.group(2)
        if lang == "python":
            return ""
        try:
            ast.parse(code)
        except SyntaxError:
            return match.group(0)
        return ""

    return _FENCE_RE.sub(_drop, text).strip()


def apply() -> bool:
    """Wrap tool_agent's markup-recovery trio to also handle fenced python.

    Returns True on success (or if already applied), False on any failure —
    in which case NOTHING is changed (vanilla fallback: stock v3 behavior).

    Kill switch: FENCED_RECOVERY_DISABLE=1 -> no-op, returns False.
    Idempotent: re-apply detects the wrapper marker and returns True.
    """
    if os.environ.get("FENCED_RECOVERY_DISABLE") == "1":
        log.info("fenced-recovery %s: FENCED_RECOVERY_DISABLE=1 -> no-op", VERSION)
        return False

    try:
        import importlib

        ta = importlib.import_module("inference.agent.tool_agent")

        required = (
            "_contains_tool_call_markup",
            "_recover_tool_calls_from_markup",
            "_strip_tool_call_markup",
        )
        for name in required:
            if not callable(getattr(ta, name, None)):
                log.warning(
                    "fenced-recovery %s: tool_agent.%s missing/non-callable "
                    "(source drift) -> no-op (vanilla)", VERSION, name)
                return False

        if getattr(ta._contains_tool_call_markup, "_fenced_recovery", None):
            log.info("fenced-recovery %s: already applied (no-op)", VERSION)
            return True

        orig_contains = ta._contains_tool_call_markup
        orig_recover = ta._recover_tool_calls_from_markup
        orig_strip = ta._strip_tool_call_markup

        def contains_patched(*chunks: str) -> bool:
            if orig_contains(*chunks):
                return True
            return bool(_fenced_python_body(*(c or "" for c in chunks)))

        def recover_patched(*chunks: str) -> list:
            recovered = orig_recover(*chunks)
            if recovered:
                return recovered
            body = _fenced_python_body(*(c or "" for c in chunks))
            if not body:
                return []
            global HITS
            HITS += 1
            if HITS == 1 or HITS % _HITS_LOG_EVERY == 0:
                print(f"fenced-recovery {VERSION} hits={HITS}", flush=True)
            return [
                {
                    "id": f"fenced-call-{HITS}",
                    "type": "function",
                    "function": {
                        "name": "python",
                        "arguments": json.dumps({"code": body}, ensure_ascii=True),
                    },
                }
            ]

        def strip_patched(text: str) -> str:
            return _strip_fenced_python(orig_strip(text))

        contains_patched._fenced_recovery = VERSION  # idempotency marker
        ta._contains_tool_call_markup = contains_patched
        ta._recover_tool_calls_from_markup = recover_patched
        ta._strip_tool_call_markup = strip_patched

        print(
            f"A17-CANARY fenced-recovery={VERSION} ACTIVE "
            f"(python-fence -> python tool call; replay-validated 434/436; "
            f"kill=FENCED_RECOVERY_DISABLE)",
            flush=True,
        )
        log.info("fenced-recovery %s installed on inference.agent.tool_agent", VERSION)
        return True
    except Exception:  # noqa: BLE001 - any failure -> vanilla fallback
        log.exception("[fenced-recovery] apply failed -> stock v3 behavior")
        return False
