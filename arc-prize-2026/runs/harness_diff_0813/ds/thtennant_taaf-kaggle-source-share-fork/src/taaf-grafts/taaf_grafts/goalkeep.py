"""Goal-inference graft: stop erasing what the agent worked out, and hand it
back the objective evidence the harness already collected.

MOTIVATION (measured on the v12 kernel-v2 commit run, 4 games / 481 turns —
see ``experiments/mine_transcripts.py``):

The stock ``ToolAgent`` carries a "working world model" across turns. Its ONLY
writer is ``_update_summarized_knowledge_from_assistant``, which parses labelled
blocks (``Goal model:``, ``Action model:`` ...) out of *assistant prose*. But a
27B under a tool-calling grammar emits prose in only **2-9% of turns** — it goes
straight to the tool call. And ``_update_summarized_knowledge_from_step_summary``
then **blanket-wipes** ``world_model``/``goal_model``/``action_model``/
``recent_findings``/``open_questions``/``current_plan`` on ANY
``level_transition`` OR ``game_over``.

The two combine into a ratchet that only ever turns one way. Measured carry of a
non-empty working model:

    m0r0     1/118 turns      sk48      4/132 turns
    sk48-dup 10/114 turns     tn36     18/117 turns
    ------------------------------------------------
    total   33/481 turns = 6.9%

In tn36 the model wrote its single best goal hypothesis at step 20 ("match slot
patterns to the yellow shape column projections") *in direct response to a game
over* — and that same turn's ``game_over`` wiped it before it was ever shown
back. The agent then spent 96 more turns re-deriving the objective from raw
segmentation, every turn, from nothing.

Two fixes, both game-agnostic (they touch only harness result keys — no game
identifiers, no layout assumptions, nothing that could overfit the public 25):

1. SELECTIVE RETENTION. A game over falsifies your *plan*, not your knowledge of
   what the game wants. A level transition falsifies the *layout*, not the goal
   or the action semantics — the stock system prompt itself says "levels often
   build on earlier mechanics". So retain by kind instead of wiping by event.

2. A MECHANICAL GOAL LEDGER. The agent is told nothing about its objective, but
   the harness observes the two channels that define it: ``level_completed``
   (what counts as progress) and ``game_over`` (what ends the attempt), plus
   ``board_changed`` (which actions do anything at all). Accumulate those over
   the whole game and inject a compact digest EVERY turn. It is recall of
   measured fact, never inference, so it cannot be wrong in the way a
   hallucinated world model can; and it is written by the harness, so unlike the
   prose channel it is present on 100% of turns rather than 7%.

The ledger costs zero environment actions and zero LLM turns. That matters more
here than better action choice: the same commit run shows games ending on the
7920s wallclock at ~400 actions, i.e. roughly one LLM turn per action, so the
binding constraint is turns spent re-deriving what was already known.

INVARIANTS:
- ``apply_goalkeep()`` returns a revert thunk; composite's install-failure path
  calls it, so a broken patch can never persist into the run.
- Patches are idempotent: re-applying is a no-op that still returns a thunk
  restoring the ORIGINAL stock methods.
- Every patched method is individually try/except'd around its own logic and
  falls back to the stock behaviour it wrapped, so a ledger bug degrades to
  stock prompt text rather than taking the game down.
- The ledger is per-game: it re-keys on ``_session_runtime_dir``, which is what
  stock ``_ensure_session`` uses to decide a session is new.
"""

from __future__ import annotations

from typing import Any, Callable

# Knowledge keys that survive each event kind. Everything not listed is cleared.
# A game over kills the PLAN only. A level transition additionally kills the
# layout-specific world model and the now-stale recent findings.
_KEEP_ON_GAME_OVER = (
    "world_model",
    "goal_model",
    "action_model",
    "recent_findings",
    "open_questions",
    "cross_level_notes",
)
_KEEP_ON_LEVEL_TRANSITION = (
    "goal_model",
    "action_model",
    "open_questions",
    "cross_level_notes",
)
_ALL_KEYS = (
    "world_model",
    "goal_model",
    "action_model",
    "recent_findings",
    "open_questions",
    "current_plan",
    "cross_level_notes",
)

_MAX_TRACKED_ACTIONS = 12
_MAX_EVENTS_SHOWN = 3
_TAIL_ACTIONS_SHOWN = 4


class _GoalLedger:
    """Per-game accumulation of the objective evidence the harness observes.

    Nothing in here is a hypothesis. Every field is a count or a timestamp of
    something the environment actually reported.
    """

    __slots__ = (
        "effect",
        "progress",
        "deaths",
        "total_actions",
        "best_level",
        "recent_actions",
        "_in_game_over",
    )

    def __init__(self) -> None:
        # action name -> [times it changed the board, times it was executed]
        self.effect: dict[str, list[int]] = {}
        self.progress: list[dict[str, Any]] = []
        self.deaths: list[dict[str, Any]] = []
        self.total_actions = 0
        self.best_level = 0
        self.recent_actions: list[str] = []
        self._in_game_over = False

    # -- accumulation -------------------------------------------------------

    def observe(self, action_results: list[dict[str, Any]]) -> None:
        for item in action_results:
            if not isinstance(item, dict) or not item.get("executed"):
                continue
            names = _executed_names(item)
            changed = bool(item.get("board_changed"))
            for name in names:
                # Aggregate by BASE action: a click carries its coordinates, so
                # keying the rate table on the display string would give one
                # bucket per pixel and say nothing. Verified against the real
                # tn36 run, where 433 clicks produced ~200 distinct displays.
                slot = self.effect.setdefault(_base_name(name), [0, 0])
                slot[1] += 1
                if changed:
                    slot[0] += 1
            self.total_actions += len(names)
            self.recent_actions.extend(names)
            del self.recent_actions[:-16]

            level = _as_int(item.get("level"))
            if level is not None:
                self.best_level = max(self.best_level, level)

            if item.get("level_completed") or item.get("run_complete"):
                self.progress.append(
                    {
                        "at": _as_int(item.get("action_num")) or self.total_actions,
                        "level": level,
                        "tail": list(self.recent_actions[-_TAIL_ACTIONS_SHOWN:]),
                    }
                )
            # ``game_over`` reflects the environment STATE, which stays set
            # until the harness resets, so consecutive results repeat it. Count
            # the edge into game-over, not every result that observes it --
            # otherwise the real tn36 run reads as 373 deaths instead of ~7 and
            # the attempt-length estimate degrades to "about 1 action".
            over = bool(item.get("game_over"))
            if over and not self._in_game_over:
                self.deaths.append(
                    {
                        "at": _as_int(item.get("action_num")) or self.total_actions,
                        "level": level,
                        "tail": list(self.recent_actions[-_TAIL_ACTIONS_SHOWN:]),
                    }
                )
            self._in_game_over = over

    # -- rendering ----------------------------------------------------------

    def lines(self) -> list[str]:
        if not self.effect and not self.progress and not self.deaths:
            return []
        out = ["Goal evidence (measured by the harness from actual outcomes, not inferred):"]

        # Which actions do anything at all. Ordered by how often they were
        # tried so the digest is stable turn to turn rather than reshuffling.
        ranked = sorted(self.effect.items(), key=lambda kv: (-kv[1][1], kv[0]))
        shown = ranked[:_MAX_TRACKED_ACTIONS]
        table = ", ".join(f"{name} {hit}/{tot}" for name, (hit, tot) in shown)
        if len(ranked) > len(shown):
            table += f", (+{len(ranked) - len(shown)} more)"
        out.append(f"- Board-changing rate per action so far: {table}.")
        dead = [name for name, (hit, tot) in ranked if tot >= 3 and hit == 0]
        if dead:
            out.append(
                f"- Never changed the board in {'/'.join(str(self.effect[n][1]) for n in dead[:4])} "
                f"tries: {', '.join(dead[:4])}. Treat as inert unless state changed."
            )

        # What counts as progress -- the only ground truth about the objective.
        if self.progress:
            for ev in self.progress[-_MAX_EVENTS_SHOWN:]:
                tail = ", ".join(ev["tail"]) or "unknown"
                out.append(
                    f"- PROGRESS: level {ev['level']} completed at action {ev['at']}. "
                    f"Immediately preceding actions: {tail}."
                )
        else:
            out.append(
                f"- No level has been completed in {self.total_actions} actions. "
                "Nothing done so far has been scored as progress; if the current "
                "approach has not changed that, it is the approach that is wrong."
            )

        # What ends the attempt.
        if self.deaths:
            ats = [str(ev["at"]) for ev in self.deaths[-_MAX_EVENTS_SHOWN:]]
            out.append(
                f"- Attempt ended (game over) {len(self.deaths)}x, most recently at "
                f"action {', '.join(ats)}."
            )
            tail = ", ".join(self.deaths[-1]["tail"])
            if tail:
                out.append(f"- Actions just before the latest game over: {tail}.")
            if len(self.deaths) >= 2:
                spans = [ev["at"] for ev in self.deaths if ev["at"]]
                if len(spans) >= 2:
                    gaps = [b - a for a, b in zip(spans, spans[1:]) if b > a]
                    if gaps:
                        out.append(
                            f"- Attempts last about {sum(gaps) // len(gaps)} actions. "
                            "Budget the plan to finish inside that."
                        )
        out.append(f"- Highest level reached so far: {self.best_level}.")
        return out


def _base_name(name: str) -> str:
    """``ACTION6(26,42)`` -> ``ACTION6``; leave plain names untouched."""
    head = name.split("(", 1)[0].strip()
    return head or name


def _executed_names(item: dict[str, Any]) -> list[str]:
    names = item.get("executed_actions")
    if isinstance(names, list):
        cleaned = [str(n).strip() for n in names if str(n).strip()]
        if cleaned:
            return cleaned
    fallback = str(item.get("action_display") or "").strip()
    return [fallback] if fallback else []


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _ledger_for(agent: Any) -> _GoalLedger:
    """Return this game's ledger, re-created when the session changes.

    Keyed on ``_session_runtime_dir`` because that is exactly what stock
    ``_ensure_session`` uses to decide the session is new (and where it resets
    ``_summarized_knowledge``), so the ledger's lifetime matches the knowledge
    it augments without patching a third method.
    """
    key = getattr(agent, "_session_runtime_dir", None)
    ledger = getattr(agent, "_goalkeep_ledger", None)
    if ledger is None or getattr(agent, "_goalkeep_session", "?") != key:
        ledger = _GoalLedger()
        agent._goalkeep_ledger = ledger  # noqa: SLF001
        agent._goalkeep_session = key  # noqa: SLF001
    return ledger


def apply_goalkeep() -> Callable[[], None]:
    """Patch ``ToolAgent`` for goal retention + the measured goal ledger.

    Returns a thunk restoring the stock methods. Safe to call twice: the second
    call detects the existing patch and returns a thunk that still restores the
    original stock methods rather than the first patch's wrappers.
    """
    import inference.agent.tool_agent as ta

    cls = ta.ToolAgent
    stock_wipe = getattr(
        cls, "_goalkeep_stock_wipe", cls._update_summarized_knowledge_from_step_summary
    )
    stock_summarize = getattr(cls, "_goalkeep_stock_summarize", cls._summarize_step_sequence)
    stock_lines = getattr(cls, "_goalkeep_stock_lines", cls._summarized_knowledge_lines)

    def _selective_retention(self: Any) -> None:
        """Retain knowledge by KIND instead of wiping it by EVENT."""
        try:
            summary = getattr(self, "_last_step_summary", None)
            if not summary:
                return
            if summary.get("run_complete"):
                return
            if summary.get("level_transition"):
                keep = _KEEP_ON_LEVEL_TRANSITION
            elif summary.get("game_over"):
                keep = _KEEP_ON_GAME_OVER
            else:
                return
            for key in _ALL_KEYS:
                if key not in keep:
                    self._summarized_knowledge[key] = ""  # noqa: SLF001
        except Exception:  # noqa: BLE001 — never take the run down; fall back to stock
            try:
                stock_wipe(self)
            except Exception:  # noqa: BLE001
                pass

    def _summarize_and_record(self: Any, action_results: list[dict[str, Any]]) -> Any:
        summary = stock_summarize(self, action_results)
        try:
            if isinstance(action_results, list):
                _ledger_for(self).observe(action_results)
        except Exception:  # noqa: BLE001 — the ledger is an add-on; never fatal
            pass
        return summary

    def _lines_with_ledger(self: Any) -> list[str]:
        base = stock_lines(self)
        try:
            evidence = _ledger_for(self).lines()
        except Exception:  # noqa: BLE001
            evidence = []
        if not evidence:
            return base
        if not base:
            # Stock returns [] when every knowledge slot is empty -- which is
            # 93% of turns. The ledger alone still needs its framing lines.
            return [
                *evidence,
                "- Revise any item above immediately if `current_frame` or `history` contradicts it.",
            ]
        # Insert before stock's trailing "Revise any item above..." line so the
        # revise instruction still governs the whole block.
        return [*base[:-1], *evidence, base[-1]]

    cls._goalkeep_stock_wipe = stock_wipe  # noqa: SLF001
    cls._goalkeep_stock_summarize = stock_summarize  # noqa: SLF001
    cls._goalkeep_stock_lines = stock_lines  # noqa: SLF001
    cls._update_summarized_knowledge_from_step_summary = _selective_retention
    cls._summarize_step_sequence = _summarize_and_record
    cls._summarized_knowledge_lines = _lines_with_ledger

    def revert() -> None:
        cls._update_summarized_knowledge_from_step_summary = stock_wipe
        cls._summarize_step_sequence = stock_summarize
        cls._summarized_knowledge_lines = stock_lines
        for attr in (
            "_goalkeep_stock_wipe",
            "_goalkeep_stock_summarize",
            "_goalkeep_stock_lines",
        ):
            if attr in cls.__dict__:
                delattr(cls, attr)

    return revert
