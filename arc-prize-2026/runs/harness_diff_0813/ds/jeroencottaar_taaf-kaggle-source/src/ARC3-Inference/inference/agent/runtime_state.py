"""In-process state shared between the analyzer host and python_tool host paths."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from inference.utils.grid_utils import format_grid_ascii


RUNTIME_STATE_FILENAME = "tool_runtime_state.json"

_CACHE: dict[str, tuple["Frame | None", list["HistoryEntry"]]] = {}


@dataclass(frozen=True)
class Frame:
    grid: tuple[tuple[int, ...], ...]
    step: int
    level: int

    @property
    def shape(self) -> tuple[int, int]:
        rows = len(self.grid)
        cols = max((len(row) for row in self.grid), default=0)
        return rows, cols

    @property
    def ascii(self) -> str:
        return format_grid_ascii(self.grid)

    def __str__(self) -> str:
        rows, cols = self.shape
        return (
            f"Level: {self.level}\n"
            f"Step: {self.step}\n"
            f"Grid shape: {rows} x {cols}\n"
            f"Grid contents:\n{self.ascii}"
        )


@dataclass(frozen=True)
class HistoryEntry:
    action: str
    frame: Frame


def load_runtime_state(path: Path) -> tuple[Frame | None, list[HistoryEntry]]:
    return _CACHE.get(str(path), (None, []))


def write_runtime_state(
    path: Path,
    *,
    current_frame: Frame | None,
    history: list[HistoryEntry],
) -> None:
    # Shallow-copy the history list so later caller-side appends don't
    # mutate the cached entry; Frame/HistoryEntry are frozen.
    _CACHE[str(path)] = (current_frame, list(history))


def forget_runtime_state(path: Path) -> None:
    _CACHE.pop(str(path), None)
