"""Replay agent for official games, driven by recorded human action lists.

The per-game action lists in re_arc/dsl/official_human_replays/ hold the
fastest public human replay segments per level (see
scripts/fetch_official_human_replays.py for provenance and verification).
"""

from __future__ import annotations

import json
from pathlib import Path

from ..core import DslAgent

REPLAY_DIR = Path(__file__).resolve().parent.parent / "official_human_replays"


class OfficialHumanReplayAgent(DslAgent):
    def __init__(self, game_id: str):
        base = game_id.split("-", 1)[0].lower()
        path = REPLAY_DIR / f"{base}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        recorded_game_id = str(payload["game_id"]).strip().lower()
        if recorded_game_id != game_id.strip().lower():
            raise ValueError(f"{path} holds a replay for {recorded_game_id!r}, not {game_id!r}.")
        levels = payload["levels"]
        super().__init__(game_id=game_id, total_levels=len(levels))
        self._actions = [
            (int(action_id), {str(key): int(value) for key, value in (data or {}).items()})
            for segment in levels
            for action_id, data in segment
        ]
        self._cursor = 0

    def reset_episode(self):
        super().reset_episode()
        self._cursor = 0

    def next_action(self, _env, _observation):
        if self._cursor >= len(self._actions):
            raise RuntimeError(f"{self.game_id}: human replay exhausted after {self._cursor} actions without WIN.")
        action_id, action_data = self._actions[self._cursor]
        self._cursor += 1
        return action_id, action_data
