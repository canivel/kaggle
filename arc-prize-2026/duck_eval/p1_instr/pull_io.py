"""Shared readers for kernel-pull artifacts (benchmark.json + transcripts).

Every P1 instrument in this package reads the SAME two artifacts the free build
rail already produces, so all five §5.3 items are offline, $0, zero-push:

  runs/kernel_pulls/<pull>/benchmark.json   -- per-game run records
  runs/kernel_pulls/<pull>/transcripts/*.txt -- per-game analyzer transcripts
  runs/kernel_pulls/<pull>/*.log            -- solver banner (concurrency, caps)

Verified structure (war_eval_v1, 2026-07-14 pull):
  benchmark.json["game_runs"][i] = {
      game_id, number_of_levels, base_actions_per_level, actions_per_level,
      levels_completed, final_score, final_wallclock_seconds, started_at,
      history: [ {action:{id,data}, generated_tokens, uncached_input_tokens,
                  wallclock_seconds}, ... ]   # wallclock_seconds is CUMULATIVE
  }
  Invariant checked in `load_pull`: sum(actions_per_level) == len(history).

Transcript sections are emitted by
`inference/agent/tool_agent.py::_append_transcript_section` as::

    [LABEL]
    <content>
    <blank>

with turn headers `--- analysis_step=N | action=M | HH:MM:SS | tool-agent ---`.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[2]
PULLS = ROOT / "runs" / "kernel_pulls"

# --------------------------------------------------------------------------
# Sandbox fault strings, copied verbatim from
# duck_eval/taaf_bundle/src/ARC3-Inference/inference/agent/python_tool_sandbox.py
# These are the ONLY host-side outcomes that destroy the child process, i.e.
# under P1 the only outcomes that destroy the persistent namespace.
#   :506  f"Tool timed out after {timeout_seconds}s"   <- _kill_process_group
#   :404  "Sandbox process exited unexpectedly."
#   :533  "Sandbox process returned an invalid response."
#   :575  "Sandbox process returned an unknown message type."
#   :471  "Sandbox process could not start."
# --------------------------------------------------------------------------
FAULT_PATTERNS: dict[str, re.Pattern[str]] = {
    "timeout": re.compile(r"Tool timed out after \d+s"),
    "exit": re.compile(r"Sandbox process exited unexpectedly\."),
    "invalid_response": re.compile(r"Sandbox process returned an invalid response\."),
    "unknown_message": re.compile(r"Sandbox process returned an unknown message type\."),
    "no_start": re.compile(r"Sandbox process could not start\."),
}

_HEADER_RE = re.compile(
    r"^--- analysis_step=(?P<step>\d+) \| action=(?P<action>[-\w]+) \| "
    r"(?P<clock>\d\d:\d\d:\d\d) \| tool-agent ---$",
    re.M,
)
_SECTION_RE = re.compile(r"^\[(?P<label>[A-Z][A-Z0-9 ]*(?::[^\]\n]*)?)\]$", re.M)
_CODE_MARKUP_RE = re.compile(
    r"<parameter=code>\n?(?P<code>.*?)\n?</parameter>", re.S
)
_BANNER_RE = re.compile(
    r"max_actions_per_game=(?P<max_actions>[\w.]+), "
    r"max_runtime_s_per_game=(?P<max_runtime>[\w.]+), "
    r"concurrency=(?P<concurrency>\d+)"
)


@dataclass
class ToolCall:
    """One `python` tool call == one sandbox child process at baseline."""

    game_id: str
    call_index: int          # 0-based, per game, in transcript order
    analysis_step: int       # harness turn this call belongs to
    action_num: str          # display action number at turn start
    clock: str               # HH:MM:SS of the turn header
    code: str                # raw code string handed to the sandbox
    result: str              # rendered [TOOL RESULT: python] text
    fault: str | None        # None, or one of FAULT_PATTERNS keys

    @property
    def is_fault(self) -> bool:
        return self.fault is not None


@dataclass
class GameRun:
    game_id: str
    levels_completed: int
    number_of_levels: int
    actions_per_level: list[int]
    final_score: float
    final_wallclock_seconds: float
    cum_wallclock: list[float] = field(default_factory=list)
    generated_tokens: list[int] = field(default_factory=list)

    @property
    def n_actions(self) -> int:
        return len(self.cum_wallclock)

    def lc_at_action_prefix(self, k: int) -> int:
        """Levels completed after the first `k` scored actions.

        Level j (0-based) is completed at cumulative action
        sum(actions_per_level[0..j]); only the first `levels_completed`
        levels were actually completed (the next entry is the partial one).
        """
        done = 0
        cum = 0
        for j, spent in enumerate(self.actions_per_level):
            if j >= self.levels_completed:
                break
            cum += spent
            if cum <= k:
                done += 1
            else:
                break
        return done

    def wallclock_at_action_prefix(self, k: int) -> float:
        if k <= 0:
            return 0.0
        return self.cum_wallclock[min(k, self.n_actions) - 1]


@dataclass
class Pull:
    name: str
    path: Path
    label: str
    solver_label: str
    games: dict[str, GameRun]
    banner: dict[str, Any]

    @property
    def total_lc(self) -> int:
        return sum(g.levels_completed for g in self.games.values())

    def transcript_path(self, game_id: str) -> Path | None:
        hits = sorted((self.path / "transcripts").glob(f"{game_id}*.txt"))
        return hits[0] if hits else None

    def iter_transcripts(self) -> Iterator[tuple[str, Path]]:
        for gid in sorted(self.games):
            p = self.transcript_path(gid)
            if p is not None:
                yield gid, p


def parse_solver_banner(pull_dir: Path) -> dict[str, Any]:
    """Read concurrency / action cap / runtime cap out of the run's own log.

    §5.4 requires the concurrency constant be read from the run banner, not
    hard-coded (the 16 in proposal §6.2 is the dataclass default, not the rail).
    """
    out: dict[str, Any] = {}
    for log in sorted(pull_dir.glob("*.log")):
        try:
            text = log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        m = _BANNER_RE.search(text)
        if m:
            raw_actions = m.group("max_actions")
            out = {
                "source": f"{log.name}",
                "max_actions_per_game": None if raw_actions == "None" else float(raw_actions),
                "max_runtime_s_per_game": float(m.group("max_runtime")),
                "concurrency": int(m.group("concurrency")),
            }
            break
    return out


def load_pull(name: str, *, strict: bool = True) -> Pull:
    path = PULLS / name
    data = json.loads((path / "benchmark.json").read_text(encoding="utf-8"))
    games: dict[str, GameRun] = {}
    for run in data["game_runs"]:
        hist = run.get("history") or []
        gr = GameRun(
            game_id=run["game_id"],
            levels_completed=int(run["levels_completed"]),
            number_of_levels=int(run["number_of_levels"]),
            actions_per_level=[int(v) for v in run["actions_per_level"]],
            final_score=float(run["final_score"]),
            final_wallclock_seconds=float(run["final_wallclock_seconds"]),
            cum_wallclock=[float(h["wallclock_seconds"]) for h in hist],
            generated_tokens=[int(h.get("generated_tokens") or 0) for h in hist],
        )
        if strict and sum(gr.actions_per_level) != gr.n_actions:
            raise ValueError(
                f"{name}/{gr.game_id}: sum(actions_per_level)="
                f"{sum(gr.actions_per_level)} != len(history)={gr.n_actions}"
            )
        games[gr.game_id] = gr
    return Pull(
        name=name,
        path=path,
        label=str(data.get("label", "")),
        solver_label=str(data.get("solver_label", "")),
        games=games,
        banner=parse_solver_banner(path),
    )


def iter_sections(text: str) -> Iterator[tuple[str, str]]:
    """Yield (label, body) for every `[LABEL]` section of a transcript, in order."""
    marks = list(_SECTION_RE.finditer(text))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        yield m.group("label"), text[m.end() + 1:end]


def _classify_fault(result_text: str) -> str | None:
    for kind, pat in FAULT_PATTERNS.items():
        if pat.search(result_text):
            return kind
    return None


def parse_transcript(path: Path, game_id: str) -> list[ToolCall]:
    """Extract the ordered `python` tool calls (and their results) for one game."""
    text = path.read_text(encoding="utf-8", errors="replace")

    # Turn headers, so each tool call can be attributed to an analysis_step.
    headers = [(m.start(), m.group("step"), m.group("action"), m.group("clock"))
               for m in _HEADER_RE.finditer(text)]

    # Section boundaries.
    sections: list[tuple[int, int, str]] = []  # (start_of_label, start_of_body, label)
    marks = list(_SECTION_RE.finditer(text))
    for i, m in enumerate(marks):
        body_start = m.end() + 1
        body_end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        sections.append((m.start(), body_start, m.group("label")))
        # store end separately below via lookup
    ends = [marks[i + 1].start() if i + 1 < len(marks) else len(text)
            for i in range(len(marks))]

    calls: list[ToolCall] = []
    pending: dict[str, Any] | None = None
    for idx, (label_pos, body_start, label) in enumerate(sections):
        body = text[body_start:ends[idx]]
        # attribute to the last header before this section
        step, action, clock = 0, "?", "?"
        for hpos, hstep, haction, hclock in headers:
            if hpos <= label_pos:
                step, action, clock = int(hstep), haction, hclock
            else:
                break

        if label.startswith("TOOL CALL: python"):
            code = None
            cm = _CODE_MARKUP_RE.search(body)
            if cm:
                code = cm.group("code")
            else:
                # fall back to the JSON-arguments rendering
                try:
                    code = json.loads(body.strip()).get("code")
                except Exception:  # noqa: BLE001
                    code = None
            if pending is not None:  # tool call with no result section
                calls.append(_finish(pending, ""))
                pending = None
            pending = {
                "game_id": game_id,
                "analysis_step": step,
                "action_num": action,
                "clock": clock,
                "code": code or "",
            }
        elif label.startswith("TOOL RESULT: python") and pending is not None:
            calls.append(_finish(pending, body))
            pending = None

    if pending is not None:
        calls.append(_finish(pending, ""))

    for i, c in enumerate(calls):
        c.call_index = i
    return calls


def _finish(pending: dict[str, Any], result_body: str) -> ToolCall:
    return ToolCall(
        game_id=pending["game_id"],
        call_index=-1,
        analysis_step=pending["analysis_step"],
        action_num=pending["action_num"],
        clock=pending["clock"],
        code=pending["code"],
        result=result_body,
        fault=_classify_fault(result_body),
    )


def load_calls(pull: Pull) -> dict[str, list[ToolCall]]:
    out: dict[str, list[ToolCall]] = {}
    for gid, tpath in pull.iter_transcripts():
        out[gid] = parse_transcript(tpath, gid)
    return out
