"""Render a verified action plan into duck-harness-EXACT SFT examples.

The whole point of this module is that it does NOT re-implement the prompt.
It subclasses the harness's own `inference.agent.tool_agent.ToolAgent` and
overrides exactly one method -- `_chat_completion` -- so every byte of the
system prompt, the user prompt, the tool schema, the tool result JSON, the
image part and the history-eviction policy is produced by the same code that
runs inside the scored Kaggle kernel. The "model response" is synthesized from
the oracle plan instead of coming off the wire; nothing is dialled.

Each `_chat_completion` call yields one training example:

    {"messages": [system, (user, assistant, tool)*, user],
     "target":   {assistant message with the `python` tool call}}

Policy the rendered assistant turns are designed to teach (and ONLY this):

  P1  commit -- when a sequence is verified, emit it as ONE batched
      `action([...])`, do not dribble one action per turn.
  P2  do not re-probe settled state -- the note never re-asks a question the
      transcript has already answered; there is no observe-only turn after the
      first.
  P3  quadratic cost awareness -- the note states the action budget in the
      scoring rule's own terms.
  P4  derive, don't memorize -- MOUSE targets are emitted as an index into the
      segmentation the model can see, whenever the coordinate is recoverable
      that way (see `derived_click_fraction` in the datasheet).
"""
from __future__ import annotations

import copy
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from harness_env import bootstrap

bootstrap()

from inference.agent.tool_agent import ToolAgent, _ChatCompletionResult  # noqa: E402

_CLICK = 6
_ENGINE_TO_MODEL = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT", 5: "SPACE", 6: "MOUSE", 7: "UNDO"}


# --------------------------------------------------------------------------
# turn planning
# --------------------------------------------------------------------------
@dataclass
class Turn:
    actions: list[dict[str, Any]]
    level_completing: bool = False
    # A turn holding exactly one action the searcher PROVED is a no-op at this
    # state. It gets its own turn on purpose: `solver.step_env` aggregates
    # `board_changed` with `any()` across a batch, so a no-op mixed in with real
    # actions reports True and teaches nothing. Alone, the tool result reads
    # `board_changed: False` -- the only way this corpus can contain a positive
    # example of "that did nothing" (war-room note §7.2).
    is_noop_probe: bool = False


def split_into_turns(
    plan_actions: list[dict[str, Any]],
    level_boundaries: set[int],
    *,
    first_batch: int = 3,
    batch_size: int = 8,
) -> list[Turn]:
    """Chunk a verified plan into as few turns as the harness allows.

    Splits are forced at level completions (the engine stops a batch there
    anyway -- `solver.step_env` breaks on `level_completed`), so a batch never
    straddles a level. Within a level the first chunk is deliberately short:
    that is the one turn where the agent legitimately does not yet know the
    action model. Everything after it is a commit.
    """
    turns: list[Turn] = []
    buf: list[dict[str, Any]] = []
    # CONDITIONAL length, not unconditional. P2 found the level-completing action is the
    # last of a long batch -- but a batch that was STALL-GATED and opened with
    # re-traversal, i.e. length earned by context. EFFNOTE failed today by lengthening
    # unconditionally (B4 = 11.11 actions/stall-turn vs a control max of 7.28; B1 post-stall
    # revisit 0.4971 vs a <0.3986 threshold, NO-PROMOTE). A flat cap teaches LENGTH; the ramp
    # teaches "short until this level proves resistant, then commit long".
    ramp = [first_batch, first_batch, max(first_batch, batch_size // 4), batch_size]
    turn_in_level = 0

    def limit_now() -> int:
        return ramp[min(turn_in_level, len(ramp) - 1)]

    for index, action in enumerate(plan_actions):
        if action.get("_probe"):
            if buf:
                turns.append(Turn(actions=buf))
                buf = []
                turn_in_level += 1
            turns.append(Turn(actions=[action], is_noop_probe=True))
            turn_in_level += 1
            continue
        buf.append(action)
        boundary = index in level_boundaries
        if boundary or len(buf) >= limit_now():
            turns.append(Turn(actions=buf, level_completing=boundary))
            buf = []
            turn_in_level = 0 if boundary else turn_in_level + 1
    if buf:
        turns.append(Turn(actions=buf))
    return turns


# --------------------------------------------------------------------------
# assistant-turn rendering
# --------------------------------------------------------------------------
_SEG_PREAMBLE = """seg = current_frame.segmentation
nodes = sorted(seg['nodes'], key=lambda n: n['pixels'])
def _center(n):
    rs = [p[0] for p in n['boundary']]; cs = [p[1] for p in n['boundary']]
    return ((min(rs) + max(rs)) // 2, (min(cs) + max(cs)) // 2)
"""


def _node_centroids(grid: tuple) -> list[tuple[int, int]]:
    """Same ordering the oracle used to propose clicks: objects sorted by pixel
    count ascending, centroid of the boundary box."""
    from inference.utils.segmentation import segment_layer

    try:
        seg = segment_layer(grid, "WwgGcBMPRbSYOrNp")
    except Exception:
        return []
    out: list[tuple[int, int]] = []
    for node in sorted(seg.get("nodes", []), key=lambda n: n.get("pixels", 0)):
        boundary = node.get("boundary") or []
        if not boundary:
            out.append((-1, -1))
            continue
        rows = [int(p[0]) for p in boundary]
        cols = [int(p[1]) for p in boundary]
        out.append((
            max(0, min(63, (min(cols) + max(cols)) // 2)),
            max(0, min(63, (min(rows) + max(rows)) // 2)),
        ))
    return out


def render_action_expr(
    actions: list[dict[str, Any]], grid: tuple
) -> tuple[str, bool, int, int]:
    """Return (python expression for the action list, uses_segmentation,
    n_derived_clicks, n_literal_clicks)."""
    centroids = _node_centroids(grid)
    index_of = {}
    for i, xy in enumerate(centroids):
        index_of.setdefault(xy, i)

    parts: list[str] = []
    derived = literal = 0
    uses_seg = False
    for action in actions:
        aid = int(action["id"])
        name = _ENGINE_TO_MODEL.get(aid, f"ACTION{aid}")
        if aid != _CLICK:
            parts.append(f"{name!r}")
            continue
        x, y = int(action["x"]), int(action["y"])
        idx = index_of.get((x, y))
        if idx is not None:
            derived += 1
            uses_seg = True
            parts.append(
                "{'action': 'MOUSE', 'row': _center(nodes[%d])[0], 'col': _center(nodes[%d])[1]}"
                % (idx, idx)
            )
        else:
            literal += 1
            parts.append("{'action': 'MOUSE', 'row': %d, 'col': %d}" % (y, x))
    return "[" + ", ".join(parts) + "]", uses_seg, derived, literal


def render_code(
    actions: list[dict[str, Any]],
    grid: tuple,
    *,
    used_this_level: int,
    human_target: int,
) -> tuple[str, dict[str, int]]:
    expr, uses_seg, derived, literal = render_action_expr(actions, grid)
    lines: list[str] = []
    if uses_seg:
        lines.append(_SEG_PREAMBLE.rstrip())
    lines.append(f"plan = {expr}")
    lines.append(
        "res = action(plan)\n"
        "print({k: res[k] for k in "
        "('executed_count', 'level', 'board_changed', 'level_completed', 'game_over') if k in res})"
    )
    return "\n".join(lines), {"derived_clicks": derived, "literal_clicks": literal}


def render_note(
    *,
    turn_index: int,
    actions: list[dict[str, Any]],
    used_this_level: int,
    human_target: int,
    level: int,
    level_completing: bool,
    last_batch_changed: bool | None,
    is_noop_probe: bool = False,
    prior_noop: str | None = None,
) -> str:
    names = [_ENGINE_TO_MODEL.get(int(a["id"]), "?") for a in actions]
    budget_left = max(0, human_target - used_this_level)
    lines: list[str] = []
    if is_noop_probe:
        return "\n".join([
            f"World model: level {level}, I do not yet know whether {names[0]} does anything "
            f"from this state.",
            f"Plan: spend exactly one action to settle it. One probe is cheap; re-asking the same "
            f"question every turn is what makes a run expensive under (human/mine)^2.",
        ])
    if prior_noop:
        lines.append(
            f"World model: level {level}, {prior_noop} changed nothing from that state. That "
            f"question is settled, so it is off the table and I will not spend another action "
            f"re-testing it."
        )
    elif turn_index == 0:
        lines.append(
            f"World model: level {level}, unknown mechanics. One short probe of "
            f"{len(actions)} action(s) to fix the action model, then commit."
        )
    else:
        changed = "the board moved" if last_batch_changed else "the board did not move"
        lines.append(
            f"World model: level {level}, {changed} on the last batch; that question is "
            f"settled, so I do not re-test it."
        )
    lines.append(
        f"Action model: {', '.join(names)} -- the shortest sequence consistent with the "
        f"evidence so far. I commit it rather than re-testing one action at a time."
    )
    lines.append(
        f"Plan: score is (human_actions / my_actions)^2, so every wasted action costs "
        f"quadratically. Used {used_this_level} of ~{human_target} on this level "
        f"({budget_left} left). Committing all {len(actions)} action(s) in one batched call "
        f"instead of one per turn."
    )
    if level_completing:
        lines.append("Open questions: whether this batch clears the level. If it does I stop "
                     "and re-read on the next level rather than probing here.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# recording agent
# --------------------------------------------------------------------------
@dataclass
class Example:
    messages: list[dict[str, Any]]
    target: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)


class TeacherAgent(ToolAgent):
    """ToolAgent with the network replaced by an oracle plan."""

    def __init__(
        self,
        turns: list[Turn],
        *,
        game_id: str,
        baseline_actions: list[int],
        stop_event: threading.Event,
        grid_provider: Callable[[], tuple],
        level_provider: Callable[[], int],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._turns = list(turns)
        self._cursor = 0
        self._game_id = game_id
        self._baseline = list(baseline_actions)
        self._stop_event = stop_event
        self._grid_provider = grid_provider
        self._level_provider = level_provider
        self._used_this_level = 0
        self._last_level = 1
        self._last_batch_changed: bool | None = None
        self._prior_noop: str | None = None
        self.examples: list[Example] = []
        self.render_stats = {"derived_clicks": 0, "literal_clicks": 0}

    def _human_target(self, level: int) -> int:
        idx = max(0, level - 1)
        if idx < len(self._baseline):
            return int(self._baseline[idx])
        return int(self._baseline[-1]) if self._baseline else 40

    def _chat_completion(self, messages, *, tools=None, request_timeout_seconds=None):
        if self._cursor >= len(self._turns):
            self._stop_event.set()
            return _ChatCompletionResult(
                message={"role": "assistant", "content": "Plan complete."},
                finish_reason="stop",
            )
        turn = self._turns[self._cursor]
        self._cursor += 1

        level = int(self._level_provider())
        if level != self._last_level:
            self._last_level = level
            self._used_this_level = 0
        grid = self._grid_provider()
        human_target = self._human_target(level)

        note = render_note(
            turn_index=self._cursor - 1,
            actions=turn.actions,
            used_this_level=self._used_this_level,
            human_target=human_target,
            level=level,
            level_completing=turn.level_completing,
            last_batch_changed=self._last_batch_changed,
            is_noop_probe=turn.is_noop_probe,
            prior_noop=self._prior_noop,
        )
        code, stats = render_code(
            turn.actions,
            grid,
            used_this_level=self._used_this_level,
            human_target=human_target,
        )
        for key, value in stats.items():
            self.render_stats[key] += value

        assistant = {
            "role": "assistant",
            "content": note,
            "tool_calls": [
                {
                    "id": f"call_{self._game_id}_{self._cursor:03d}",
                    "type": "function",
                    "function": {"name": "python", "arguments": json.dumps({"code": code})},
                }
            ],
        }
        self.examples.append(
            Example(
                messages=copy.deepcopy(messages),
                target=copy.deepcopy(assistant),
                meta={
                    "game_id": self._game_id,
                    "turn_index": self._cursor - 1,
                    "level": level,
                    "batch_size": len(turn.actions),
                    "used_this_level_before": self._used_this_level,
                    "human_target_level": human_target,
                    "level_completing": turn.level_completing,
                    "is_noop_probe": turn.is_noop_probe,
                    **stats,
                },
            )
        )
        self._used_this_level += len(turn.actions)
        self._last_batch_changed = not turn.is_noop_probe
        self._prior_noop = (
            _ENGINE_TO_MODEL.get(int(turn.actions[0]['id']), 'that action')
            if turn.is_noop_probe else None
        )
        self._session_generated_tokens += max(1, (len(note) + len(code)) // 4)
        return _ChatCompletionResult(message=assistant, finish_reason="tool_calls")
