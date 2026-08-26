"""Single source of truth about the sandbox, read from the frozen fork's own source.

Nothing here is hand-copied: `SAFE_MODULES`, `SAFE_BUILTINS` and the harness
pre-loaded globals are parsed out of
`duck_eval/taaf_bundle/src/ARC3-Inference/inference/agent/python_tool_sandbox.py`
so that if the fork moves, every P1 instrument moves with it (or fails loudly).
"""
from __future__ import annotations

import ast
import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SANDBOX_SRC = (ROOT / "duck_eval" / "taaf_bundle" / "src" / "ARC3-Inference"
               / "inference" / "agent" / "python_tool_sandbox.py")
TOOL_AGENT_SRC = (ROOT / "duck_eval" / "taaf_bundle" / "src" / "ARC3-Inference"
                  / "inference" / "agent" / "tool_agent.py")
PROMPTS_SRC = (ROOT / "duck_eval" / "taaf_bundle" / "src" / "ARC3-Inference"
               / "inference" / "agent" / "prompts.py")

# Names the sandbox bootstrap injects into `runtime_globals` before exec.
# python_tool_sandbox.py:322-370 (`runtime_globals`, `_refresh_state`, `action`).
HARNESS_GLOBALS = frozenset({
    "result",
    "current_frame",
    "latest_frame",
    "history",
    "transitions",
    "last_transition",
    "previous_frame",
    "last_action_frame",
    "last_action",
    "valid_actions",
    "last_action_result",
    "action",
    "__builtins__",
    "__name__",
    "__doc__",
})


def _bootstrap_source() -> str:
    """The dedented sandbox child program (the value of _SANDBOX_BOOTSTRAP)."""
    text = SANDBOX_SRC.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_SANDBOX_BOOTSTRAP" for t in node.targets
        ):
            # _SANDBOX_BOOTSTRAP = textwrap.dedent(r"""...""").replace(...)
            for sub in ast.walk(node.value):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str) \
                        and "SAFE_MODULES" in sub.value:
                    import textwrap
                    return textwrap.dedent(sub.value)
    raise RuntimeError(f"_SANDBOX_BOOTSTRAP literal not found in {SANDBOX_SRC}")


def _literal_set(name: str) -> frozenset[str]:
    src = _bootstrap_source()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return frozenset(ast.literal_eval(node.value))
    raise RuntimeError(f"{name} not found in sandbox bootstrap")


@lru_cache(maxsize=None)
def safe_modules() -> frozenset[str]:
    return _literal_set("SAFE_MODULES")


@lru_cache(maxsize=None)
def safe_builtins() -> frozenset[str]:
    return _literal_set("SAFE_BUILTINS")


@lru_cache(maxsize=None)
def sandbox_line_index() -> dict[str, int]:
    """1-based line numbers of the load-bearing sandbox constructs, for citation."""
    text = SANDBOX_SRC.read_text(encoding="utf-8").splitlines()
    idx: dict[str, int] = {}
    for i, line in enumerate(text, start=1):
        s = line.strip()
        if s.startswith("SAFE_MODULES = {"):
            idx.setdefault("SAFE_MODULES", i)
        if s.startswith("def _set_limits"):
            idx.setdefault("_set_limits", i)
        if s.startswith("def _kill_process_group"):
            idx.setdefault("_kill_process_group", i)
        if "Tool timed out after" in s:
            idx.setdefault("timeout_string", i)
        if s.startswith("def run_sandboxed_python"):
            idx.setdefault("run_sandboxed_python", i)
        if "RLIMIT_CPU" in s:
            idx.setdefault("RLIMIT_CPU", i)
    return idx


def rlimit_kinds() -> set[str]:
    """Which resource limits `_set_limits` actually sets.

    §5.4: there is NO memory rlimit. With persistent children x 25 concurrent
    games that is an OOM path to a whole-notebook ERROR, so P1 must add one.
    """
    src = _bootstrap_source()
    tree = ast.parse(src)
    kinds: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_set_limits":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str) \
                        and sub.value.startswith("RLIMIT_"):
                    kinds.add(sub.value)
    return kinds


TYCHO_STATE_PROBE = """\
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class Cell(Enum):
    EMPTY = 0
    WALL = 1

@dataclass
class State:
    level: int = 0
    grid: tuple = ()
    cursor: Optional[tuple] = None
    facts: dict = field(default_factory=dict)

s = State(level=1, grid=((0, 1), (1, 0)), cursor=(0, 0))
result = {"level": s.level, "cell": Cell.WALL.value, "ok": True}
print(result)
"""


def tycho_state_importable(modules: frozenset[str] | None = None) -> tuple[bool, list[str]]:
    """Does the Tycho `State` dataclass probe import-resolve under SAFE_MODULES?"""
    mods = modules if modules is not None else safe_modules()
    needed = sorted({
        m.split(".", 1)[0]
        for m in re.findall(r"^(?:from|import)\s+([\w.]+)", TYCHO_STATE_PROBE, re.M)
    })
    missing = [m for m in needed if m not in mods]
    return (not missing), missing
