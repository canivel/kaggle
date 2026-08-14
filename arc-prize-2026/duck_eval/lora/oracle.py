"""Expert-plan search against the local engines (CPU only, no LLM, no network).

Produces, per environment, the SHORTEST verified action sequence we can find
that clears one or more levels, then prunes it and re-verifies the pruned
sequence from a fresh game. Only plans that (a) replay to the same
levels_completed and (b) use no more actions than the human baseline for those
levels are eligible as training targets.

Why greedy-with-lookahead rather than BFS: the engines run at ~7-10k actions/s
but the branching factor with click games is ~O(#objects), so a full BFS is
wasteful. A 1-ply lookahead scored by (level_completed > new-state > board
changed) is Go-Explore-lite and is enough to crack the levels whose mechanics
are reachable at all; anything it cannot crack is simply not a training source.

Nothing here reads a game id to decide what to do -- the candidate set comes
only from `available_actions` and the segmentation of the current board.
"""
from __future__ import annotations

import copy
import hashlib
import random
from dataclasses import dataclass, field
from typing import Any

from harness_env import bootstrap

bootstrap()

import arcengine  # noqa: E402
from taaf.game_api import GameAPI  # noqa: E402

import engine_compat  # noqa: E402

engine_compat.apply()

_ID2ACTION = {a.value: a for a in arcengine.GameAction}
_CLICK = 6
_RESET = 0


@dataclass
class Plan:
    game_id: str
    actions: list[dict[str, Any]]          # [{"id": 1} | {"id": 6, "x":.., "y":..}]
    levels_cleared: int
    baseline_actions: list[int]            # human baseline per level
    raw_action_count: int                  # before pruning
    verified: bool = False
    notes: list[str] = field(default_factory=list)
    # Candidates the 1-ply lookahead PROVED are no-ops at a real board state:
    # (level, action). The searcher gets these free by cloning; a real agent
    # would pay one action each. They are the only honest source of
    # "that did nothing" examples (war-room note 7.2/7.4) -- the committed
    # trace has almost none, because greedy never commits a candidate it just
    # saw do nothing.
    noop_candidates: list[tuple[int, dict[str, Any]]] = field(default_factory=list)

    @property
    def action_count(self) -> int:
        return len(self.actions)

    @property
    def human_actions(self) -> int:
        return sum(self.baseline_actions[: self.levels_cleared])

    @property
    def rhae(self) -> float:
        """Sum of per-level (human/agent)^2 is not recoverable post-hoc from a
        single number, so we report the whole-plan ratio: human/agent."""
        if not self.action_count:
            return 0.0
        return self.human_actions / float(self.action_count)


def _grid(state) -> tuple:
    data = state.frame.data
    rows = data.tolist() if hasattr(data, "tolist") else data
    return tuple(tuple(int(cell) for cell in row) for row in rows)


def _terminal(state) -> bool:
    """`state.game_over` is the taaf-level flag; arcengine can also sit in
    GAME_OVER independently, and issuing a non-RESET action there raises."""
    if bool(state.game_over):
        return True
    raw = getattr(state, "raw", None)
    raw_state = getattr(raw, "state", None)
    return raw_state is not None and str(getattr(raw_state, "name", raw_state)) == "GAME_OVER"


def _grid_hash(state) -> str:
    raw = repr(_grid(state)).encode("utf-8")
    return hashlib.blake2b(raw, digest_size=12).hexdigest()


def _click_candidates(state, limit: int = 24) -> list[tuple[int, int]]:
    """Click targets derived from the board itself: one per segmented object,
    at the object's centroid (snapped into the object when the centroid falls
    outside it). Game-agnostic by construction."""
    from inference.utils.segmentation import segment_layer

    grid = _grid(state)
    chars = "WwgGcBMPRbSYOrNp"
    try:
        seg = segment_layer(grid, chars)
    except Exception:
        seg = {"nodes": []}
    out: list[tuple[int, int]] = []
    nodes = seg.get("nodes", [])
    # Smallest objects first: interactive tokens are usually small, background
    # slabs are usually huge and useless as click targets.
    for node in sorted(nodes, key=lambda n: n.get("pixels", 0)):
        boundary = node.get("boundary") or []
        if not boundary:
            continue
        rows = [int(p[0]) for p in boundary]
        cols = [int(p[1]) for p in boundary]
        y = max(0, min(63, (min(rows) + max(rows)) // 2))
        x = max(0, min(63, (min(cols) + max(cols)) // 2))
        if (x, y) not in out:
            out.append((x, y))
        if len(out) >= limit:
            break
    return out


def _candidates(state, rnd: random.Random, click_limit: int) -> list[dict[str, Any]]:
    avail = [v for v in state.available_actions if v not in (_RESET,)]
    out: list[dict[str, Any]] = [{"id": v} for v in avail if v != _CLICK]
    if _CLICK in avail:
        clicks = _click_candidates(state, limit=click_limit)
        rnd.shuffle(clicks)
        out.extend({"id": _CLICK, "x": x, "y": y} for x, y in clicks)
    return out


def _to_input(action: dict[str, Any]) -> "arcengine.ActionInput":
    aid = int(action["id"])
    if aid == _CLICK:
        return arcengine.ActionInput(
            id=_ID2ACTION[aid], data={"x": int(action["x"]), "y": int(action["y"])}
        )
    return arcengine.ActionInput(id=_ID2ACTION[aid], data={})


def open_game(game_id: str, spec, *, allow_deepcopy: bool = True) -> GameAPI:
    game = GameAPI(env_name=game_id, arcade_spec=spec, allow_deepcopy=allow_deepcopy)
    game.start_game()
    game._finish_game = lambda: None  # keep the engine scorecard open
    return game


def replay(game_id: str, spec, actions: list[dict[str, Any]]) -> tuple[int, bool]:
    """Fresh game, apply `actions`, return (levels_completed, game_over)."""
    game = open_game(game_id, spec, allow_deepcopy=False)
    state = game.current_state
    for action in actions:
        if _terminal(state):
            break
        if int(action["id"]) not in state.available_actions:
            return -1, bool(state.game_over)
        try:
            state = game.execute_action(_to_input(action))
        except Exception:
            # The engine refuses non-RESET actions once arcengine reports
            # GAME_OVER; treat that as "this trial does not replay".
            return -1, True
    return int(state.levels_completed), bool(state.game_over)


def greedy_search(
    game_id: str,
    spec,
    *,
    max_actions: int = 260,
    target_levels: int = 2,
    click_limit: int = 24,
    seed: int = 0,
) -> Plan | None:
    """One greedy pass with 1-ply lookahead on cloned engines."""
    rnd = random.Random(seed)
    try:
        game = open_game(game_id, spec)
    except Exception:
        return None
    baseline = list(game.base_actions_per_level or [])
    state = game.current_state
    seen: set[str] = {_grid_hash(state)}
    trace: list[dict[str, Any]] = []
    noop_candidates: list[tuple[int, dict[str, Any]]] = []
    start_levels = int(state.levels_completed)

    while len(trace) < max_actions:
        if _terminal(state) or int(state.levels_completed) - start_levels >= target_levels:
            break
        cands = _candidates(state, rnd, click_limit)
        if not cands:
            break
        best = None
        here = _grid_hash(state)
        level_now = int(state.levels_completed)
        for action in cands:
            try:
                clone = copy.deepcopy(game)
                nxt = clone.execute_action(_to_input(action))
            except Exception:
                continue
            score = 0.0
            if int(nxt.levels_completed) > int(state.levels_completed):
                score += 1000.0
            if nxt.game_over:
                score -= 500.0
            h = _grid_hash(nxt)
            if h == here and int(nxt.levels_completed) == level_now and not nxt.game_over:
                # Proven dead at THIS state, without spending a real action.
                if len(noop_candidates) < 64:
                    noop_candidates.append((level_now, action))
            if h not in seen:
                score += 100.0
            if h != _grid_hash(state):
                score += 10.0
            score += rnd.random() * 0.01
            if best is None or score > best[0]:
                best = (score, action, h)
            if score >= 1000.0:
                break  # a level clear ends the search for this step
        if best is None or best[0] <= 0.0:
            break
        _, action, h = best
        try:
            state = game.execute_action(_to_input(action))
        except Exception:
            break
        trace.append(action)
        seen.add(h)

    cleared = int(state.levels_completed) - start_levels
    if cleared <= 0:
        return None
    return Plan(
        game_id=game_id,
        actions=trace,
        levels_cleared=cleared,
        baseline_actions=baseline,
        raw_action_count=len(trace),
        noop_candidates=noop_candidates,
    )


def find_noops(game_id: str, spec, actions: list[dict[str, Any]]) -> list[tuple[int, int, dict[str, Any]]]:
    """Replay the RAW trace and return the actions that changed nothing.

    These are the informative failures: a real, agent-observable "that did
    nothing" at a real board state. `prune()` deletes all of them, which is why
    the v0 corpus contains 334 tool results and **zero** `board_changed: False`
    (§7.2) -- the behaviour we most want to teach has no positive examples.

    Returns `(index_in_raw_trace, level_at_the_time, action)`. A no-op is
    state-neutral by definition, so re-inserting one at the same level is
    provably safe: the level still clears, and the harness-replay check still
    proves it.
    """
    game = open_game(game_id, spec, allow_deepcopy=False)
    state = game.current_state
    out: list[tuple[int, int, dict[str, Any]]] = []
    for index, action in enumerate(actions):
        if _terminal(state):
            break
        before = _grid_hash(state)
        level = int(state.levels_completed)
        try:
            state = game.execute_action(_to_input(action))
        except Exception:
            break
        if _grid_hash(state) == before and int(state.levels_completed) == level:
            out.append((index, level, action))
    return out


def prune(plan: Plan, spec) -> Plan:
    """Drop every action that is not needed to reach the same levels_completed.

    Two passes, both verified by replay so the result is provably equivalent:
      1. trailing trim -- everything after the last level completion.
      2. greedy leave-one-out -- walk the plan and delete any single action
         whose removal still replays to the same levels_completed.
    """
    actions = list(plan.actions)

    # 1. trailing trim.
    target = plan.levels_cleared
    lo, hi = 1, len(actions)
    while lo < hi:
        mid = (lo + hi) // 2
        lc, _ = replay(plan.game_id, spec, actions[:mid])
        if lc >= target:
            hi = mid
        else:
            lo = mid + 1
    actions = actions[:lo]

    # 2. leave-one-out (single pass, back to front so indices stay valid).
    i = len(actions) - 1
    while i >= 0:
        trial = actions[:i] + actions[i + 1 :]
        lc, _ = replay(plan.game_id, spec, trial)
        if lc >= target:
            actions = trial
        i -= 1

    lc, game_over = replay(plan.game_id, spec, actions)
    pruned = Plan(
        game_id=plan.game_id,
        actions=actions,
        levels_cleared=min(target, lc),
        baseline_actions=plan.baseline_actions,
        raw_action_count=plan.raw_action_count,
        verified=(lc >= target),
        noop_candidates=list(plan.noop_candidates),
    )
    if not pruned.verified:
        pruned.notes.append(f"replay mismatch: got lc={lc}, wanted {target}")
    if game_over:
        pruned.notes.append("game_over at end of pruned plan")
    return pruned
