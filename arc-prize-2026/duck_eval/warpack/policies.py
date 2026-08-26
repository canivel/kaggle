"""Scripted (no-LLM) analyzers replaying the war-room winning policies through
the duck harness session (``_HarnessGameSession.step_env``), so the warpack
trace recorder sees exactly what a real run would record.

Ports of the engine-verified scratchpad scripts referenced by
learnings/war_room/{sb26,su15,lp85}_mechanics.md. Each analyzer clears
Level 1 then Level 2 deterministically, then sets the session stop event.

Coordinates go through the model action space ({"action": "MOUSE", "row": y,
"col": x}) exactly like the LLM's tool calls would.
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any


def _result(payload: dict[str, Any] | None) -> SimpleNamespace:
    return SimpleNamespace(
        retryable_failure=False,
        yielded_control=False,
        step_executed=bool(payload and payload.get("executed")),
    )


class _ScriptedAnalyzer:
    """Base: one step_env request per analyze() turn, stop at 2 levels."""

    generated_tokens = 0
    _timeout = 5.0
    target_levels = 2

    def __init__(self, game: Any, stop_event: Any) -> None:
        self.game = game
        self.stop_event = stop_event
        self.turns = 0
        self.last_payload: dict[str, Any] | None = None

    @property
    def raw(self) -> Any:
        """The live arcengine game instance behind GameAPI."""
        return self.game.env._game

    def click(self, x: int, y: int) -> dict[str, Any]:
        return {"action": "MOUSE", "row": int(y), "col": int(x)}

    def analyze(self, state_path, action_num, valid_actions=None, step_env=None, **kwargs):
        self.turns += 1
        if int(self.game.current_state.levels_completed) >= self.target_levels or self.turns > 300:
            self.stop_event.set()
            return _result(None)
        request = self.next_request()
        if request is None:
            self.stop_event.set()
            return _result(None)
        payload = step_env(request)
        self.last_payload = payload
        return _result(payload)

    def next_request(self) -> dict[str, Any] | None:  # pragma: no cover
        raise NotImplementedError


class Sb26Policy(_ScriptedAnalyzer):
    """sb26 'program the color sequence': L1 = 4 placements + RUN,
    L2 = CALL-aware 7 placements + RUN (24 actions total).
    Fixed coordinates from sb26_mechanics.md (click = sprite.x+1, sprite.y+1)."""

    def __init__(self, game, stop_event):
        super().__init__(game, stop_event)
        script: list[dict[str, Any]] = []

        def place(item_xy, spot_xy):
            script.append(self.click(item_xy[0] + 1, item_xy[1] + 1))
            script.append(self.click(spot_xy[0] + 1, spot_xy[1] + 1))

        # Level 1: targets 9,14,11,15; tray y=56; slots y=27
        for item, spot in [((33, 56), (20, 27)), ((17, 56), (26, 27)),
                           ((41, 56), (32, 27)), ((25, 56), (38, 27))]:
            place(item, spot)
        script.append({"action": "SPACE"})  # ACTION5 = RUN
        # Level 2: main = 12,15,CALL,6 ; frame14 = 8,9,14,11
        for item, spot in [((29, 56), (20, 20)), ((15, 56), (26, 20)),
                           ((36, 56), (38, 20)), ((8, 56), (20, 34)),
                           ((43, 56), (26, 34)), ((22, 56), (32, 34)),
                           ((50, 56), (38, 34))]:
            place(item, spot)
        script.append({"action": "SPACE"})
        self.script = script
        self.pos = 0

    def next_request(self):
        if self.pos >= len(self.script):
            return None
        request = self.script[self.pos]
        self.pos += 1
        return request


class Su15Policy(_ScriptedAnalyzer):
    """su15 'Suika Vacuum': L1 = drag the single fruit into the ring at
    (48,15); L2 = 4 pair-merges, drag-merge to two L2s, one L3, drag into the
    goal ring center (33,27). State recomputed from the live engine each turn
    (port of scratchpad drive_su15.py)."""

    L2_PAIR_CLICKS = [(39, 38), (17, 39), (15, 56), (48, 55)]

    def __init__(self, game, stop_event):
        super().__init__(game, stop_event)
        self.pair_pos = 0

    # --- engine introspection helpers (deobfuscated names in mechanics doc)
    def _fruits(self):
        raw = self.raw
        return [(i, s, raw.amnmgwpkeb.get(s, 0), raw.qmecbepbyz(s))
                for i, s in enumerate(raw.hmeulfxgy)]

    def _hop_toward(self, ia: int, target: tuple[float, float], check_catch: bool) -> dict[str, Any]:
        raw = self.raw
        sa = raw.hmeulfxgy[ia]
        ax, ay = raw.qmecbepbyz(sa)
        tx, ty = target
        d = math.hypot(tx - ax, ty - ay)
        ux, uy = (tx - ax) / d, (ty - ay) / d
        px = py = None
        for hop in (7, 6, 5, 4, 3):
            px = max(1, min(62, int(round(ax + ux * hop))))
            py = max(11, min(61, int(round(ay + uy * hop))))
            if not check_catch:
                break
            others = [s for j, s in enumerate(raw.hmeulfxgy) if j != ia
                      and raw.yrufkxnmou(px, py, raw.qjlubdgly, s)]
            if not others:
                break
        return self.click(px, py)

    def next_request(self):
        raw = self.raw
        fruits = self._fruits()
        if not fruits:
            return None
        if raw.level_index == 0:
            # Level 1: drag the single fruit into the goal center (48,15).
            goal = (48, 15)
            _, _, _, (cx, cy) = fruits[0]
            if math.hypot(goal[0] - cx, goal[1] - cy) <= 7:
                return self.click(*goal)
            return self._hop_toward(0, goal, check_catch=False)
        if raw.level_index != 1:
            return None
        # Level 2.
        zeros = [f for f in fruits if f[2] == 0]
        if zeros and self.pair_pos < len(self.L2_PAIR_CLICKS):
            request = self.click(*self.L2_PAIR_CLICKS[self.pair_pos])
            self.pair_pos += 1
            return request
        if len(fruits) == 1:
            # One L3 fruit: drag its center into the goal ring at (33,27).
            goal = (33, 27)
            ia, _, _, (cx, cy) = fruits[0]
            if math.hypot(goal[0] - cx, goal[1] - cy) <= 7:
                return self.click(*goal)
            return self._hop_toward(ia, goal, check_catch=False)
        # Merge the closest same-level pair at the highest level with >=2.
        best = None
        for level in sorted({f[2] for f in fruits}, reverse=True):
            group = [f for f in fruits if f[2] == level]
            if len(group) < 2:
                continue
            for a in range(len(group)):
                for b in range(a + 1, len(group)):
                    (ia, _, _, (ax, ay)) = group[a]
                    (ib, _, _, (bx, by)) = group[b]
                    d = math.hypot(bx - ax, by - ay)
                    if best is None or d < best[0]:
                        best = (d, ia, ib, (ax, ay), (bx, by))
            break
        if best is None:
            return None
        d, ia, ib, (ax, ay), (bx, by) = best
        if d <= 11:
            mx = (int(ax) + int(bx)) // 2
            my = max(11, min(61, (int(ay) + int(by)) // 2))
            return self.click(mx, my)
        return self._hop_toward(ia, (bx, by), check_catch=True)


class Lp85Policy(_ScriptedAnalyzer):
    """lp85 'looping_chains': L1 = generic single-chain rotation; L2 =
    park-and-return via the chain crossings (1x A_R, 1x C_R, 3x A_R, 3x C_R).
    Button display coords brute-forced through camera.display_to_grid
    (port of scratchpad lp85_policy.py)."""

    L2_SEQ = ["button_A_R"] + ["button_C_R"] + ["button_A_R"] * 3 + ["button_C_R"] * 3

    def __init__(self, game, stop_event):
        super().__init__(game, stop_event)
        self.l1_plan: list[str] | None = None
        self.l1_pos = 0
        self.l2_pos = 0
        self._coord_cache: dict[tuple[int, str], tuple[int, int]] = {}

    def _button_click(self, tag: str) -> dict[str, Any]:
        raw = self.raw
        key = (raw.level_index, tag)
        if key not in self._coord_cache:
            sprite = next(s for s in raw.current_level._sprites
                          if s.tags and s.tags[0] == tag)
            found = None
            for dy in range(64):
                for dx in range(64):
                    r = raw.camera.display_to_grid(dx, dy)
                    if not r:
                        continue
                    gx, gy = r
                    if (sprite.x <= gx < sprite.x + sprite.width
                            and sprite.y <= gy < sprite.y + sprite.height
                            and sprite.pixels[gy - sprite.y][gx - sprite.x] != -1):
                        found = (dx, dy)
                        break
                if found:
                    break
            if found is None:
                raise RuntimeError(f"no display coord for {tag}")
            self._coord_cache[key] = found
        dx, dy = self._coord_cache[key]
        return self.click(dx, dy)

    def _plan_l1(self) -> list[str]:
        raw = self.raw
        lvl_name = raw.current_level.get_data("level_name")
        chain = raw.uopmnplcnv[lvl_name]["A"]
        slots, n = chain["qcmzcjocmj"], chain["oxbwsencfv"]

        def slot_of(pos):
            for k, p in slots.items():
                if (p.x * 3, p.y * 3) == pos:
                    return k
            return None

        bracket = raw.current_level.get_sprites_by_tag("bghvgbtwcb")[0]
        goal = next(s for s in raw.current_level._sprites
                    if s.tags and s.tags[0] == "goal")
        tslot = slot_of((bracket.x + 1, bracket.y + 1))
        gslot = slot_of((goal.x, goal.y))
        r = (tslot - gslot) % n
        tag, count = ("button_A_R", r) if r <= n - r else ("button_A_L", n - r)
        return [tag] * count

    def next_request(self):
        raw = self.raw
        if raw.level_index == 0:
            if self.l1_plan is None:
                self.l1_plan = self._plan_l1()
            if self.l1_pos >= len(self.l1_plan):
                return None
            tag = self.l1_plan[self.l1_pos]
            self.l1_pos += 1
            return self._button_click(tag)
        if raw.level_index == 1:
            if self.l2_pos >= len(self.L2_SEQ):
                return None
            tag = self.L2_SEQ[self.l2_pos]
            self.l2_pos += 1
            return self._button_click(tag)
        return None
