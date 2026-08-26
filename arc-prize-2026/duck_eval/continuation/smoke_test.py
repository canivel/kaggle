"""Smoke test for the game-over-continuation patch (A12 hygiene window).

Run:  uv run python duck_eval/continuation/smoke_test.py

Tests (all against the real bundled harness's _build_system_prompt):
  (a) pre-patch: output contains "stop acting immediately".
  (b) post-apply(): output contains BOTH new lines, NOT "stop acting immediately".
  (c) byte-identity outside the replaced region: reversing NEW_LINES back to
      OLD_LINE in the post prompt reproduces the pre prompt exactly.
  (d) idempotency: second apply() returns True, prompt unchanged, no dup lines.
  (e) kill switch: fresh importlib-reload with CONTINUATION_DISABLE=1 leaves the
      prompt untouched.
  (f) tampered-source: if OLD_LINE is absent (pre-mutate the attr), apply()
      returns False and the prompt is left exactly as found.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "duck_eval" / "taaf_bundle" / "src" / "ARC3-Inference"
CONT_DIR = Path(__file__).resolve().parent

for p in (str(HARNESS), str(CONT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

TOOL_OUT = 1024


def _prompt() -> str:
    import inference.agent.tool_agent as tool_agent
    return tool_agent._build_system_prompt(tool_output_tokens=TOOL_OUT)


def _fresh_import():
    """Reload prompts, tool_agent, and continuation_patch from scratch so the
    module-global rebind and any kill-switch env are re-read.

    Popping the submodules from sys.modules is not enough: the parent package
    ``inference.agent`` keeps stale attribute references to the previously
    patched submodules, and ``from inference.agent import prompts`` would return
    those. Re-import via ``importlib.import_module`` (which rebinds the parent
    attribute) so callers get genuinely clean, source-fresh modules."""
    for name in ("inference.agent.prompts", "inference.agent.tool_agent",
                 "continuation_patch"):
        sys.modules.pop(name, None)
    importlib.import_module("inference.agent.prompts")
    importlib.import_module("inference.agent.tool_agent")
    return importlib.import_module("continuation_patch")


def main() -> int:
    results: list[tuple[str, bool, str]] = []

    def check(label: str, cond: bool, detail: str = "") -> None:
        results.append((label, bool(cond), detail))
        print(f"[{'PASS' if cond else 'FAIL'}] {label}"
              + (f" -- {detail}" if detail and not cond else ""))

    OLD_PHRASE = "stop acting immediately"

    # Clean baseline: ensure no stale kill switch, fresh modules.
    os.environ.pop("CONTINUATION_DISABLE", None)
    cont = _fresh_import()

    # (a) pre-patch prompt carries the OLD phrase.
    pre = _prompt()
    check("(a) pre-patch prompt contains 'stop acting immediately'",
          OLD_PHRASE in pre, "OLD phrase absent before patch")

    # (b) apply -> both NEW lines present, OLD phrase gone.
    ok = cont.apply()
    post = _prompt()
    both_new = cont.NEW_LINES in post
    no_old = OLD_PHRASE not in post
    check("(b) apply() returned True", ok is True, f"apply()={ok!r}")
    check("(b) post prompt contains BOTH new lines", both_new,
          "NEW_LINES block not found verbatim")
    check("(b) post prompt no longer contains OLD phrase", no_old,
          "OLD phrase still present")

    # (c) byte-identity outside the replaced region: reverse NEW->OLD == pre.
    reconstructed = post.replace(cont.NEW_LINES, cont.OLD_LINE)
    check("(c) prompt outside replaced region is byte-identical pre/post",
          reconstructed == pre,
          "reversing NEW->OLD did not reproduce the pre-patch prompt")

    # (d) idempotency: second apply() True, prompt unchanged, no dup lines.
    ok2 = cont.apply()
    post2 = _prompt()
    dup = post2.count(cont.NEW_LINES)
    check("(d) second apply() returned True", ok2 is True, f"apply()={ok2!r}")
    check("(d) prompt unchanged after second apply()", post2 == post,
          "prompt drifted on second apply()")
    check("(d) NEW_LINES block appears exactly once (no duplication)", dup == 1,
          f"NEW_LINES occurs {dup}x")

    # (e) kill switch: fresh reload with CONTINUATION_DISABLE=1 -> untouched.
    os.environ["CONTINUATION_DISABLE"] = "1"
    try:
        cont_ks = _fresh_import()
        pre_ks = _prompt()
        ok_ks = cont_ks.apply()
        post_ks = _prompt()
        check("(e) apply() returned False under CONTINUATION_DISABLE=1",
              ok_ks is False, f"apply()={ok_ks!r}")
        check("(e) prompt untouched under kill switch",
              post_ks == pre_ks and OLD_PHRASE in post_ks,
              "kill switch did not preserve the vanilla prompt")
    finally:
        os.environ.pop("CONTINUATION_DISABLE", None)

    # (f) tampered-source: OLD_LINE absent -> apply() False, prompt left as found.
    cont_t = _fresh_import()
    from inference.agent import prompts as prompts_mod
    from inference.agent import tool_agent as tool_agent_mod
    # Simulate drift/tamper: strip the target line so OLD_LINE is not a substring.
    tampered = prompts_mod.PYTHON_ADDENDUM.replace(cont_t.OLD_LINE,
                                                   "- (game-over guidance removed)\n")
    prompts_mod.PYTHON_ADDENDUM = tampered
    tool_agent_mod.PYTHON_ADDENDUM = tampered
    pre_t = _prompt()
    ok_t = cont_t.apply()
    post_t = _prompt()
    check("(f) apply() returned False on tampered source (OLD_LINE absent)",
          ok_t is False, f"apply()={ok_t!r}")
    check("(f) prompt left exactly as found on tampered source",
          post_t == pre_t and cont_t.NEW_LINES not in post_t,
          "tampered-source path mutated the prompt")

    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"\nSMOKE {passed}/{total} PASS")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
