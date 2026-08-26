"""Game-over-continuation fix — amendment A12 standalone prompt hygiene.

Design refs:
  - learnings/war_room/grinder_cracking_design.md §(f) ("Game-over-continuation
    fix — counting bound ~= 0.00, ships anyway"); ships FIRST as the A12
    standalone hygiene window precisely because it claims nothing.
  - learnings/war_room/gpt56_distill_ft09_su15.md (the "Game-over-continuation"
    paragraph): the "stop acting immediately" system-prompt line cost GPT-5.6 56
    of 60 minutes on su15 — the environment auto-resets the board but keeps
    ``state=GAME_OVER`` until the next action (``valid_actions`` stays
    populated), so a rule-following frontier model concluded the run was
    terminal and stopped playing. Qwen survives only by disobedience.

THE FIX: rewrite exactly one line at the end of ``PYTHON_ADDENDUM`` so that
``game_over`` is described as non-terminal for the run (fresh attempt on the same
level, actions remain valid). NOTHING else changes: no reach-probe line, no
ledger, no diff summarizer, no feature bundling (A12 is a hygiene-only window).

CONSUMPTION PATH (verified against the bundled harness):
  - ``inference.agent.tool_agent`` does ``from .prompts import PYTHON_ADDENDUM``
    (line 21), binding a module-global name in the tool_agent namespace.
  - ``inference.agent.tool_agent._build_system_prompt`` (line 425) reads that
    tool_agent module global at CALL time.
  - sessions build the system prompt at init (solver, line ~1016).
So a patch applied before session construction must rewrite BOTH
``inference.agent.prompts.PYTHON_ADDENDUM`` AND
``inference.agent.tool_agent.PYTHON_ADDENDUM``.

House pattern (mirrors duck_eval/ledger/ledger_patch.py): VERSION marker, env
kill switch, blanket-guarded apply, runtime banner print, vanilla fallback on ANY
failure (worst case: the stock duck prompt, unchanged). No threading, no locks,
no game-id logic.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("continuation")

VERSION = "v1"

# Exact line to replace: prompts.py PYTHON_ADDENDUM final entry (verified against
# the bundled source 2026-07-18). Includes the leading "- " and trailing "\n".
OLD_LINE = (
    "- If an action result reports `game_over`, `run_complete`, "
    "`level_completed`, or `done`, stop acting immediately and re-ground on the "
    "next turn.\n"
)

# Replacement: two lines. First softens "stop acting immediately" to "stop the
# current batch of actions"; second states game_over is NOT terminal for the run.
NEW_LINES = (
    "- If an action result reports `game_over`, `run_complete`, "
    "`level_completed`, or `done`, stop the current batch of actions and "
    "re-ground on the next turn.\n"
    "- `game_over` is NOT terminal for the run: the environment starts a fresh "
    "attempt on the same level and actions remain valid. On the turn after a "
    "`game_over`, re-ground on the new frame and keep playing immediately; "
    "everything you have learned about the game still applies.\n"
)

# Modules whose PYTHON_ADDENDUM attribute must be rewritten (both point at the
# same string object today, but tool_agent binds its own name at import time).
_TARGET_MODULES = ("inference.agent.prompts", "inference.agent.tool_agent")


def apply() -> bool:
    """Rewrite the game-over line in PYTHON_ADDENDUM across both consuming
    modules. Returns True on success (or if already applied), False on any
    failure — in which case NOTHING is changed (vanilla fallback: stock prompt).

    Kill switch: CONTINUATION_DISABLE=1 -> no-op, returns False.
    Idempotent: a second call detects NEW text already present and returns True
    without duplicating.
    """
    if os.environ.get("CONTINUATION_DISABLE") == "1":
        log.info("continuation %s: CONTINUATION_DISABLE=1 -> no-op", VERSION)
        return False

    try:
        import importlib

        mods = {name: importlib.import_module(name) for name in _TARGET_MODULES}

        # Inspect current state per module.
        has_old: dict[str, bool] = {}
        has_new: dict[str, bool] = {}
        for name, mod in mods.items():
            text = getattr(mod, "PYTHON_ADDENDUM", None)
            if not isinstance(text, str):
                log.warning(
                    "continuation %s: %s.PYTHON_ADDENDUM missing/non-str -> "
                    "no-op (vanilla)", VERSION, name)
                return False
            has_old[name] = OLD_LINE in text
            has_new[name] = NEW_LINES in text

        # Idempotency: every target already carries the NEW text and none the
        # OLD -> already applied, do not duplicate.
        if all(has_new.values()) and not any(has_old.values()):
            log.info("continuation %s: already applied (no-op)", VERSION)
            return True

        # Tampered-source / drift guard: the exact OLD line must be present in
        # EVERY target that is not already migrated. If it is missing anywhere,
        # change NOTHING and fall back to vanilla.
        for name in _TARGET_MODULES:
            if not has_old[name] and not has_new[name]:
                log.warning(
                    "continuation %s: OLD_LINE not found in %s.PYTHON_ADDENDUM "
                    "(source drift/tamper) -> no-op (vanilla), nothing patched",
                    VERSION, name)
                return False

        # All targets accounted for -> perform the exact-substring replacement.
        for name, mod in mods.items():
            text = getattr(mod, "PYTHON_ADDENDUM")
            if OLD_LINE in text:
                setattr(mod, "PYTHON_ADDENDUM", text.replace(OLD_LINE, NEW_LINES))

        print(
            f"continuation {VERSION}: game-over-continuation ACTIVE "
            f"(2 modules patched)",
            flush=True,
        )
        log.info("continuation %s installed across %s", VERSION, _TARGET_MODULES)
        return True
    except Exception:  # noqa: BLE001 - any failure -> vanilla fallback
        log.exception("[continuation] apply failed -> stock prompt (vanilla)")
        return False
