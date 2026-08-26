"""Smoke tests for fenced_recovery_patch (A17 canary v4 graft).

Runs against the REAL bundled harness module
(duck_eval/taaf_bundle/src/ARC3-Inference inference.agent.tool_agent), then
replays the recorded canary-v3 transcripts through the patched functions.

Usage:  uv run python duck_eval/a17/fenced_recovery_smoke.py
Exit 0 = all pass.
"""
from __future__ import annotations

import ast
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BUNDLE_SRC = REPO / "duck_eval" / "taaf_bundle" / "src" / "ARC3-Inference"
TRANSCRIPTS = REPO / "runs" / "kernel_pulls" / "a17_canary_v3" / "transcripts"

sys.path.insert(0, str(BUNDLE_SRC))
sys.path.insert(0, str(HERE))

PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f" — {detail}" if detail and not ok else ""))
    PASS += ok
    FAIL += not ok


HERMES_MARKUP = (
    "<tool_call><function=python>"
    "<parameter=code>print('hi')</parameter>"
    "</function></tool_call>"
)
FENCED = "Let me inspect.\n```python\nseg = current_frame.segmentation\naction(actions=[{'action': 'MOUSE', 'row': 1, 'col': 2}])\n```\nDone."
FENCED_BAD = "```python\ndef broken(:\n```"
PLAIN = "I think the answer is to click the red square."


def main() -> int:
    import importlib

    ta = importlib.import_module("inference.agent.tool_agent")
    import fenced_recovery_patch as frp

    orig_contains = ta._contains_tool_call_markup
    orig_recover = ta._recover_tool_calls_from_markup

    # --- S1 kill switch (before any wrap) ---
    os.environ["FENCED_RECOVERY_DISABLE"] = "1"
    check("S1 kill-switch apply()->False", frp.apply() is False)
    check("S1b kill-switch left module untouched",
          ta._contains_tool_call_markup is orig_contains)
    del os.environ["FENCED_RECOVERY_DISABLE"]

    # --- S2 apply + idempotency ---
    check("S2 apply()->True", frp.apply() is True)
    check("S2b re-apply idempotent ->True", frp.apply() is True)
    wrapped_recover = ta._recover_tool_calls_from_markup
    check("S2c module actually wrapped",
          ta._contains_tool_call_markup is not orig_contains)

    # --- S3 hermes markup regression: orig path still first ---
    check("S3 markup still detected", ta._contains_tool_call_markup(HERMES_MARKUP))
    r_orig = orig_recover(HERMES_MARKUP)
    r_new = wrapped_recover(HERMES_MARKUP)
    check("S3b markup recovery unchanged",
          json.dumps(r_orig, sort_keys=True) == json.dumps(r_new, sort_keys=True),
          f"orig={r_orig!r} new={r_new!r}")

    # --- S4 fenced python recovered as python tool call ---
    check("S4 fence detected by contains()", ta._contains_tool_call_markup(FENCED))
    calls = wrapped_recover(FENCED)
    ok = (
        len(calls) == 1
        and calls[0]["function"]["name"] == "python"
    )
    check("S4b one python tool call", ok, repr(calls))
    if ok:
        args = json.loads(calls[0]["function"]["arguments"])
        code = args.get("code", "")
        parse_ok = True
        try:
            ast.parse(code)
        except SyntaxError:
            parse_ok = False
        check("S4c arguments JSON + ast-valid code",
              parse_ok and "action(" in code)

    # --- S5 markup wins over fence when both present ---
    both = HERMES_MARKUP + "\n" + FENCED
    r_both = wrapped_recover(both)
    check("S5 markup takes precedence",
          bool(r_both) and r_both[0]["function"]["name"] != "python"
          or (bool(r_orig) and json.dumps(r_both, sort_keys=True)
              == json.dumps(orig_recover(both), sort_keys=True)),
          repr(r_both))

    # --- S6/S7 negatives ---
    check("S6 plain prose -> no detect, no recover",
          not ta._contains_tool_call_markup(PLAIN) and wrapped_recover(PLAIN) == [])
    check("S7 malformed fence -> no detect, no recover (stock v3 behavior)",
          not ta._contains_tool_call_markup(FENCED_BAD)
          and wrapped_recover(FENCED_BAD) == [])

    # --- S8 strip removes consumed fence, keeps prose ---
    stripped = ta._strip_tool_call_markup(FENCED)
    check("S8 strip removes fenced code, keeps prose",
          "```" not in stripped and "Let me inspect." in stripped and "Done." in stripped,
          repr(stripped))

    # --- S9 TRANSCRIPT REPLAY on real recorded v3 traffic ---
    turn_re = re.compile(r"^--- analysis_step=\d+ \| action=\d+ \| ", re.M)
    asst_re = re.compile(r"^\[ASSISTANT\]\n(.*?)(?=^\[[A-Z ]+\]$|\Z)", re.S | re.M)
    total = recovered = 0
    for path in sorted(TRANSCRIPTS.glob("*_p0.txt")):
        text = path.read_text(encoding="utf-8", errors="replace")
        starts = [m.start() for m in turn_re.finditer(text)]
        for i, s in enumerate(starts):
            turn = text[s : starts[i + 1] if i + 1 < len(starts) else len(text)]
            m = asst_re.search(turn)
            if not m:
                continue
            total += 1
            calls = wrapped_recover(m.group(1))
            if not calls:
                continue
            args = json.loads(calls[0]["function"]["arguments"])
            ast.parse(args["code"])  # raises -> test file fails loudly
            recovered += 1
    check(f"S9 replay: recovered {recovered}/{total} recorded turns (need >=430)",
          total >= 430 and recovered >= 430 and recovered / max(total, 1) >= 0.99)

    # --- S10 drift guard: missing symbol -> apply refuses, changes nothing ---
    saved = ta._strip_tool_call_markup
    # fresh module state for a clean apply attempt
    ta._contains_tool_call_markup = orig_contains
    ta._recover_tool_calls_from_markup = orig_recover
    del ta._strip_tool_call_markup
    check("S10 drift guard apply()->False", frp.apply() is False)
    check("S10b drift guard left module vanilla",
          ta._contains_tool_call_markup is orig_contains)
    ta._strip_tool_call_markup = saved
    check("S10c re-apply after restore ->True", frp.apply() is True)

    print(f"\n{PASS} PASS, {FAIL} FAIL")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
