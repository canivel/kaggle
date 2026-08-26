"""Hypothesis Ledger + Goal-Family Escalation graft (R2, intervention_plan.md).

taaf_grafts-style flagged module. ``install(bm, flags)`` is the single entry
point (``apply()`` is the env-var convenience used by the hook cell). ALL
flags default OFF; ``install(bm, {})`` is a proven no-op. Any install error
restores stock behaviour (worst case: vanilla duck).

Flags:
  ledger      -- persistent per-game HYPOTHESIS/FACT store outside the
                 14-message window; GOAL:/RESULT:/FACT: prompt fields with
                 regex extraction; <=600-token digest injected every turn;
                 survives GAME_OVER restarts and level transitions (store is
                 keyed PER GAME by the runtime-state filename stem -- the
                 harness puts every game's ``<artifact_stem>_p<N>_runtime_
                 state.json`` in ONE shared artifacts dir, so v1's
                 parent-dir keying silently shared one store across all
                 concurrent games -- and mirrored to
                 ``<runtime_dir>/ledger_<stem>.json``, never to messages).
  escalation  -- implies ledger. After N=3 same-family fully-executed
                 refutations, injects the ONE-SHOT 4-family enumeration
                 prompt (execution-order/program, transfer-between-structures,
                 merge/physics, spatial-alignment); fires once per trigger.

What gets patched (see PATCH_NOTES.md):
  1. ToolAgent.analyze                                -> bind per-game ledger,
                                                         persist after turn
  2. ToolAgent._update_summarized_knowledge_from_assistant
                                                      -> GOAL:/RESULT:/FACT:
                                                         regex extraction tap
  3. ToolAgent._build_user_prompt                     -> protocol lines +
                                                         digest + one-shot
                                                         escalation injection
  4. _HarnessGameSession._execute_action              -> harness-side FACTs
                                                         (level-up, GAME_OVER,
                                                         no-op coordinates)

No game-id logic anywhere. No solver replacement; analyzer-side patches are
class-level monkeypatches so the pickled ``bm.solver`` picks them up at call
time (same mechanism as phase1_patch).
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

try:  # package-relative first (dataset layout), then flat (sys.path insert)
    from . import ledger_core as core
except ImportError:  # pragma: no cover
    import ledger_core as core

log = logging.getLogger("ledger")

VERSION = "v2"  # v2: per-game store keying (v1 keyed by state_path.parent =
                # the SHARED artifacts dir in the live layout -> one store for
                # all concurrent games; panel R12 N6 fix)
GRAFTS_API_VERSION = 1
STORE_KEYING = "per-game:runtime-state-stem"

_FLAG_NAMES = ("ledger", "escalation")

PROTOCOL_LINES = (
    "Ledger protocol (mandatory): state your current goal hypothesis on its own "
    "line as `GOAL: [ordering|transfer|merge|alignment|other] <one sentence>`. "
    "After the evidence for it arrives, report on its own line "
    "`RESULT: confirmed|refuted|unclear - <one-line evidence>`. Record durable "
    "action-effect observations as `FACT: <one sentence>` lines. GOAL/RESULT/"
    "FACT lines are stored in a permanent per-game ledger (re-injected above "
    "every turn, surviving context eviction, GAME_OVER restarts and level "
    "changes). Never re-execute a REFUTED goal or re-probe a recorded FACT."
)

# -------------------------------------------------------------------------
# per-game ledger registry, keyed by (runtime dir, game stem). The stem is the
# runtime-state filename up to "runtime_state" -- on the live harness that is
# "<artifact_stem(game_id)>_p<pass>" (solver.py builds
# "<stem>_runtime_state.json" for every game inside ONE shared artifacts dir);
# test rigs that use a bare "runtime_state.json" in a per-game dir fall back
# to the dir name, which is unique there. Persistence is one file per game
# ("ledger_<stem>.json") so concurrent games never share a store or a file.
# -------------------------------------------------------------------------
_LEDGERS: dict[str, core.Ledger] = {}
_LEDGERS_LOCK = threading.Lock()
_CFG: dict[str, bool] = {"ledger": False, "escalation": False}
_APPLIED = False


def _ledger_key(state_path: Path) -> tuple[str, Path]:
    """(registry key, persistence path) for the game that owns state_path."""
    p = Path(state_path)
    name = p.name
    i = name.find("runtime_state")
    stem = name[:i].rstrip("_.") if i >= 0 else p.stem
    if not stem:
        stem = p.parent.name or "game"
    parent = p.parent.resolve()
    return f"{parent}::{stem}", parent / f"ledger_{stem}.json"


def get_ledger(state_path: Path) -> core.Ledger:
    key, path = _ledger_key(state_path)
    with _LEDGERS_LOCK:  # dict ops only under the lock (never call out)
        led = _LEDGERS.get(key)
    if led is None:
        loaded = core.Ledger.load(path)  # I/O outside the lock
        with _LEDGERS_LOCK:
            led = _LEDGERS.setdefault(key, loaded)
    return led


def save_ledger(state_path: Path) -> None:
    key, path = _ledger_key(state_path)
    with _LEDGERS_LOCK:  # dict ops only under the lock
        led = _LEDGERS.get(key)
    if led is not None:
        try:
            led.save(path)  # I/O outside the lock
        except Exception as exc:  # noqa: BLE001 - persistence is best-effort
            log.debug("ledger save failed: %s", exc)


def _normalize_flags(flags: Any) -> dict[str, bool]:
    resolved = {name: False for name in _FLAG_NAMES}
    if isinstance(flags, dict):
        for name in _FLAG_NAMES:
            resolved[name] = bool(flags.get(name))
    elif isinstance(flags, str):
        wanted = {token.strip().lower() for token in flags.split(",") if token.strip()}
        for name in _FLAG_NAMES:
            resolved[name] = name in wanted
    if resolved["escalation"]:
        resolved["ledger"] = True  # escalation implies ledger
    return resolved


# -------------------------------------------------------------------------
# patches
# -------------------------------------------------------------------------
def _apply_patches() -> None:
    global _APPLIED
    if _APPLIED:
        return

    import inference.framework.solver as solver_mod
    from inference.agent.tool_agent import ToolAgent

    # (1) analyze: bind the per-game ledger + persist after the turn --------
    _orig_analyze = ToolAgent.analyze

    def ledger_analyze(self, state_path, action_num, valid_actions=None,
                       step_env=None, **kwargs):
        bound_path = None
        if _CFG["ledger"]:
            try:
                bound_path = Path(state_path)
                led = get_ledger(bound_path)  # per-game key from the filename
                self._ledger_state = led
                self._ledger_action_num = int(action_num or 0)
                self._ledger_step = int(kwargs.get("analysis_step") or 0)
            except Exception as exc:  # noqa: BLE001 - never break the turn
                log.warning("ledger pre-analyze hook failed: %s", exc)
        try:
            return _orig_analyze(self, state_path, action_num,
                                 valid_actions=valid_actions,
                                 step_env=step_env, **kwargs)
        finally:
            if _CFG["ledger"] and bound_path is not None:
                save_ledger(bound_path)

    ToolAgent.analyze = ledger_analyze

    # (2) assistant-content tap: GOAL:/RESULT:/FACT: extraction -------------
    _orig_knowledge = ToolAgent._update_summarized_knowledge_from_assistant

    def ledger_update_knowledge(self, content):
        _orig_knowledge(self, content)
        if not _CFG["ledger"]:
            return
        led: core.Ledger | None = getattr(self, "_ledger_state", None)
        if led is None:
            return
        try:
            records = core.extract_goal_result(content or "")
            if records:
                led.ingest(records,
                           step=getattr(self, "_ledger_step", 0),
                           action=getattr(self, "_ledger_action_num", 0))
        except Exception as exc:  # noqa: BLE001
            log.debug("ledger extraction failed: %s", exc)

    ToolAgent._update_summarized_knowledge_from_assistant = ledger_update_knowledge

    # (3) user prompt: protocol + digest + one-shot escalation --------------
    _orig_build_prompt = ToolAgent._build_user_prompt

    def ledger_build_user_prompt(self, action_num, **kwargs):
        prompt = _orig_build_prompt(self, action_num, **kwargs)
        if not _CFG["ledger"]:
            return prompt
        led: core.Ledger | None = getattr(self, "_ledger_state", None)
        if led is None:
            return prompt
        try:
            extra = [led.render_digest(), PROTOCOL_LINES]
            if _CFG["escalation"]:
                escalation = led.consume_escalation()  # one-shot per trigger
                if escalation:
                    extra.append(escalation)
            return prompt + "\n" + "\n".join(extra)
        except Exception as exc:  # noqa: BLE001
            log.debug("ledger prompt injection failed: %s", exc)
            return prompt

    ToolAgent._build_user_prompt = ledger_build_user_prompt

    # (4) harness-side FACT feed (works even with non-LLM analyzers) --------
    _orig_execute_action = solver_mod._HarnessGameSession._execute_action

    def ledger_execute_action(self, action, *, batch_index, batch_size,
                              generated_tokens=None, flush_viewer_payload=True):
        payload = _orig_execute_action(
            self, action, batch_index=batch_index, batch_size=batch_size,
            generated_tokens=generated_tokens,
            flush_viewer_payload=flush_viewer_payload)
        if not _CFG["ledger"]:
            return payload
        try:
            session_state_path = Path(self.state_path)
            led = get_ledger(session_state_path)  # same per-game key as analyze
            action_num = int(payload.get("action_num") or 0)
            display = str(payload.get("action_display") or "")
            if payload.get("level_completed"):
                led.add_fact(
                    f"level completed at action {action_num} (after {display})",
                    action=action_num)
                led.levels_seen += 1
            if payload.get("game_over"):
                led.game_overs += 1
                led.add_fact(
                    f"GAME_OVER at action {action_num} (after {display}); "
                    "this ledger persists across the restart",
                    action=action_num)
            if (not payload.get("board_changed")
                    and display.startswith("MOUSE")
                    and not payload.get("game_over")):
                counts = getattr(self, "_ledger_noop_counts", None)
                if counts is None:
                    counts = self._ledger_noop_counts = {}
                counts[display] = counts.get(display, 0) + 1
                if counts[display] >= 2:  # known no-op coordinate
                    led.add_fact(
                        f"no-op: {display} changed nothing "
                        f"{counts[display]}x; do not click it again",
                        action=action_num)
            save_ledger(session_state_path)
        except Exception as exc:  # noqa: BLE001 - never break the action path
            log.debug("ledger execute_action hook failed: %s", exc)
        return payload

    solver_mod._HarnessGameSession._execute_action = ledger_execute_action

    _APPLIED = True


# -------------------------------------------------------------------------
# entry points
# -------------------------------------------------------------------------
def install(bm: Any = None, flags: Any = None) -> dict[str, bool]:
    """Composite-style installer. All flags default OFF -> proven no-op.
    On ANY error the flags are reset to OFF (stock behaviour)."""
    resolved = _normalize_flags(flags)
    if not any(resolved.values()):
        log.info("ledger %s: all flags off -> no-op", VERSION)
        _CFG.update(resolved)
        return dict(_CFG)
    previous = dict(_CFG)
    try:
        _apply_patches()
        _CFG.update(resolved)
        if bm is not None and hasattr(bm, "label"):
            suffix = "+".join(n for n in _FLAG_NAMES if resolved[n])
            bm.label = f"{bm.label}-ledger-{VERSION}-{suffix}"
        # Runtime banner on stdout: the build log is the only proof of which
        # dataset version actually ran (feedback_kaggle_dataset_code_sync).
        print(f"ledger {VERSION}: store keying = {STORE_KEYING}; "
              f"flags={resolved}", flush=True)
        log.info("ledger %s installed: %s", VERSION, resolved)
    except Exception:
        _CFG.update(previous)
        log.exception("[ledger] install failed -> stock")
        raise
    return dict(_CFG)


def apply(bm: Any = None) -> dict[str, bool]:
    """Env-var driven install (hook cell): LEDGER_FLAGS='ledger,escalation'."""
    import os

    return install(bm, os.environ.get("LEDGER_FLAGS", ""))
