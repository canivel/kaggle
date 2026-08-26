"""Record the engine-verified scripted L2 policies as flat action sequences.

Ports of the war-room scratchpad policies (`sb26_policy.py`, `drive_su15.py`,
`lp85_policy.py` — see learnings/war_room/*_mechanics.md) refactored to RECORD
each real env action as a model-style request dict:
    {"action": "MOUSE", "row": y, "col": x}   or   {"action": "SPACE"}
so the sequence can be replayed byte-identically through the (grafted)
`_HarnessGameSession.step_env` path. The engines are deterministic, so a
recorded sequence reproduces the recorder's outcome on a fresh instance.

Each recorder drives the RAW engine module directly (same as the verified
scratchpad scripts) and asserts levels_completed/level_index == 2 before
returning, so a recording failure can never masquerade as a harness bug.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
ENV_FILES = REPO / "kaggle-data" / "environment_files"


def _load_game_module(game: str):
    game_dir = next((ENV_FILES / game).iterdir())
    path = game_dir / f"{game}.py"
    spec = importlib.util.spec_from_file_location(game, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _game_class(mod, game: str):
    name = game.capitalize()
    if hasattr(mod, name):
        return getattr(mod, name)
    # fallback: first class defined in the module with perform_action
    for value in vars(mod).values():
        if isinstance(value, type) and hasattr(value, "perform_action"):
            return value
    raise RuntimeError(f"no game class in {game}")


class _Recorder:
    """Wraps a raw engine game; records model-style requests as it clicks."""

    def __init__(self, game: str, resets: int = 1) -> None:
        from arcengine import ActionInput
        from arcengine.enums import GameAction

        self._ActionInput = ActionInput
        self._GameAction = GameAction
        mod = _load_game_module(game)
        self.game = _game_class(mod, game)()
        self.sequence: list[dict[str, Any]] = []
        self.frame = None
        for _ in range(resets):
            self.frame = self.game.perform_action(
                ActionInput(id=GameAction.RESET), raw=True)

    def click(self, x: int, y: int):
        self.frame = self.game.perform_action(
            self._ActionInput(id=self._GameAction.ACTION6,
                              data={"x": int(x), "y": int(y)}), raw=True)
        self.sequence.append({"action": "MOUSE", "row": int(y), "col": int(x)})
        return self.frame

    def space(self):
        self.frame = self.game.perform_action(
            self._ActionInput(id=self._GameAction.ACTION5), raw=True)
        self.sequence.append({"action": "SPACE"})
        return self.frame


# ---------------------------------------------------------------- sb26
def record_sb26() -> list[dict[str, Any]]:
    """Program-the-color-sequence game; L2 = CALL/subroutine mechanic.
    Static winning assignment from sb26_mechanics.md (engine-verified)."""
    rec = _Recorder("sb26", resets=1)

    def place(item_xy, spot_xy):
        rec.click(item_xy[0] + 1, item_xy[1] + 1)   # select tray item
        rec.click(spot_xy[0] + 1, spot_xy[1] + 1)   # drop on slot spot

    # Level 1: targets 9,14,11,15
    for item, spot in [((33, 56), (20, 27)), ((17, 56), (26, 27)),
                       ((41, 56), (32, 27)), ((25, 56), (38, 27))]:
        place(item, spot)
    rec.space()  # RUN (ACTION5)
    assert rec.game.level_index >= 1, f"sb26 L1 failed: {rec.game.level_index}"

    # Level 2: flattened main0,main1,f14[0..3],main3 = 12,15,8,9,14,11,6
    for item, spot in [((29, 56), (20, 20)), ((15, 56), (26, 20)),
                       ((36, 56), (38, 20)), ((8, 56), (20, 34)),
                       ((43, 56), (26, 34)), ((22, 56), (32, 34)),
                       ((50, 56), (38, 34))]:
        place(item, spot)
    rec.space()
    assert rec.game.level_index >= 2, f"sb26 L2 failed: {rec.game.level_index}"
    return rec.sequence


# ---------------------------------------------------------------- su15
def record_su15() -> list[dict[str, Any]]:
    """Suika-vacuum game; drag/merge fruits into the goal (drive_su15.py port)."""
    rec = _Recorder("su15", resets=2)
    game = rec.game

    def centers():
        return {i: game.qmecbepbyz(s) for i, s in enumerate(game.hmeulfxgy)}

    def drag_merge(ia: int, ib: int) -> None:
        while True:
            sa, sb = game.hmeulfxgy[ia], game.hmeulfxgy[ib]
            ax, ay = game.qmecbepbyz(sa)
            bx, by = game.qmecbepbyz(sb)
            d = math.hypot(bx - ax, by - ay)
            if d <= 11:
                mx, my = (ax + bx) // 2, (ay + by) // 2
                my = max(11, min(61, my))
                rec.click(mx, my)
                return
            ux, uy = (bx - ax) / d, (by - ay) / d
            px = py = None
            for hop in (7, 6, 5, 4, 3):
                px, py = int(round(ax + ux * hop)), int(round(ay + uy * hop))
                py = max(11, min(61, py))
                px = max(1, min(62, px))
                others = [s for j, s in enumerate(game.hmeulfxgy) if j != ia
                          and game.yrufkxnmou(px, py, game.qjlubdgly, s)]
                if not others:
                    break
            rec.click(px, py)

    def drag_to(goal, level_index_done: int, max_clicks: int = 45) -> None:
        n = 0
        while game.level_index == level_index_done:
            s0 = game.hmeulfxgy[0]
            cx, cy = game.qmecbepbyz(s0)
            d = math.hypot(goal[0] - cx, goal[1] - cy)
            if d <= 7:
                rec.click(goal[0], goal[1])
                break
            ux, uy = (goal[0] - cx) / d, (goal[1] - cy) / d
            px, py = int(round(cx + ux * 7)), int(round(cy + uy * 7))
            py = max(11, min(61, py))
            rec.click(px, py)
            n += 1
            assert n < max_clicks, "su15 drag stuck"

    # Level 1: drag the level-2 fruit into the goal center (48, 15)
    drag_to((48, 15), level_index_done=0)
    assert game.level_index == 1, f"su15 L1 failed: {game.level_index}"

    def idx_at(px, py, lvl):
        for i, s in enumerate(game.hmeulfxgy):
            if game.amnmgwpkeb.get(s, 0) == lvl:
                cx, cy = game.qmecbepbyz(s)
                if abs(cx - px) <= 3 and abs(cy - py) <= 3:
                    return i
        raise KeyError((px, py, lvl))

    # Level 2: 8x L0 -> 4x L1 -> 2x L2 -> 1x L3 into goal center (33, 27)
    rec.click(39, 38)
    rec.click(17, 39)
    rec.click(15, 56)
    rec.click(48, 55)
    assert sorted(game.amnmgwpkeb.get(s, 0) for s in game.hmeulfxgy) == [1, 1, 1, 1]
    drag_merge(idx_at(17, 39, 1), idx_at(39, 38, 1))
    drag_merge(idx_at(15, 56, 1), idx_at(48, 55, 1))
    assert sorted(game.amnmgwpkeb.get(s, 0) for s in game.hmeulfxgy) == [2, 2]
    drag_merge(0, 1)
    assert sorted(game.amnmgwpkeb.get(s, 0) for s in game.hmeulfxgy) == [3]
    drag_to((33, 27), level_index_done=1)
    assert game.level_index == 2, f"su15 L2 failed: {game.level_index}"
    return rec.sequence


# ---------------------------------------------------------------- lp85
def record_lp85() -> list[dict[str, Any]]:
    """Rotating-chain game; L2 = park-and-return (lp85_policy.py port)."""
    rec = _Recorder("lp85", resets=1)
    g = rec.game

    def find_button(tag):
        for s in g.current_level._sprites:
            if s.tags and s.tags[0] == tag:
                return s
        raise RuntimeError("no button " + tag)

    def display_coords_for(sprite):
        for dy in range(64):
            for dx in range(64):
                r = g.camera.display_to_grid(dx, dy)
                if r:
                    gx, gy = r
                    if (sprite.x <= gx < sprite.x + sprite.width
                            and sprite.y <= gy < sprite.y + sprite.height):
                        if sprite.pixels[gy - sprite.y][gx - sprite.x] != -1:
                            return dx, dy
        raise RuntimeError("no display coord for " + str(sprite.tags))

    def click(tag):
        dx, dy = display_coords_for(find_button(tag))
        return rec.click(dx, dy)

    def slot_of(pos, chain_slots):
        for k, p in chain_slots.items():
            if (p.x * 3, p.y * 3) == pos:
                return k
        return None

    # Level 1: generic single-chain solve
    lvl_name = g.current_level.get_data("level_name")
    chain = g.uopmnplcnv[lvl_name]["A"]
    slots, n_slots = chain["qcmzcjocmj"], chain["oxbwsencfv"]
    brackets = g.current_level.get_sprites_by_tag("bghvgbtwcb")
    goals = [s for s in g.current_level._sprites if s.tags and s.tags[0] == "goal"]
    assert len(brackets) == 1 and len(goals) == 1
    tslot = slot_of((brackets[0].x + 1, brackets[0].y + 1), slots)
    gslot = slot_of((goals[0].x, goals[0].y), slots)
    r = (tslot - gslot) % n_slots
    tag, n = ("button_A_R", r) if r <= n_slots - r else ("button_A_L", n_slots - r)
    for _ in range(n):
        click(tag)
    assert g.level_index == 1, f"lp85 L1 failed: {g.level_index}"

    # Level 2: park-and-return
    for tag in (["button_A_R"] * 1 + ["button_C_R"] * 1
                + ["button_A_R"] * 3 + ["button_C_R"] * 3):
        click(tag)
    assert g.level_index == 2, f"lp85 L2 failed: {g.level_index}"
    return rec.sequence


RECORDERS = {"sb26": record_sb26, "su15": record_su15, "lp85": record_lp85}


def record_all() -> dict[str, list[dict[str, Any]]]:
    return {game: recorder() for game, recorder in RECORDERS.items()}


if __name__ == "__main__":
    for game, sequence in record_all().items():
        print(f"{game}: {len(sequence)} actions recorded, L2 cleared")
