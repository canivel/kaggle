"""Phase-1 exploration substrate — pure-CPU core (stdlib only).

Everything in this module is harness-agnostic and testable without a GPU:

  - ``state_signature``      : segmentation-graph signature of a frame (dedup hash)
  - ``DedupArchive``         : deduped state archive with cost-aware frontier scoring
  - ``ProgressTracker``      : no-progress turn counter (explore trigger)
  - ``summarize_animation``  : pixel-delta text summary across an action's
                               intermediate frames (attacks sb26/tn36 blindness)
  - ``plan_probe_actions``   : scripted explore() action selection
  - ``run_explore``          : scripted explore loop driven by an executor callback
  - ``render_explore_summary``: curated <=500-token report for the next user message

Design constraints honoured (Tufa writeup review 2026-07-08):
  * explore() is HARNESS-SIDE and scripted; nothing here is an LLM tool.
  * Frontier score = novelty / (1 + return_cost); the only reset primitive is a
    full-episode RESET, so return_cost is the prefix replay length.
  * Summaries are text only; NO multi-image injection.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

Grid = Sequence[Sequence[int]]

_ORTH = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Character labels matching duck's ARC_COLOR_CHARS ordering (0-15).
_COLOR_CHARS = "0123456789ABCDEF"


# ---------------------------------------------------------------------------
# Segmentation-graph signature
# ---------------------------------------------------------------------------

def connected_components(grid: Grid) -> list[dict[str, Any]]:
    """4-connected same-value components. Returns dicts with color/size/cells/
    min_r/min_c/centroid, ordered by top-most-left-most cell (reading order)."""
    height = len(grid)
    width = len(grid[0]) if height else 0
    comp_id = [[-1] * width for _ in range(height)]
    components: list[dict[str, Any]] = []
    for sr in range(height):
        row = grid[sr]
        for sc in range(width):
            if comp_id[sr][sc] != -1:
                continue
            value = row[sc]
            cid = len(components)
            cells: list[tuple[int, int]] = []
            stack = [(sr, sc)]
            comp_id[sr][sc] = cid
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in _ORTH:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and comp_id[nr][nc] == -1 and grid[nr][nc] == value:
                        comp_id[nr][nc] = cid
                        stack.append((nr, nc))
            rs = [r for r, _ in cells]
            cs = [c for _, c in cells]
            components.append(
                {
                    "id": cid,
                    "color": int(value),
                    "size": len(cells),
                    "cells": cells,
                    "min_r": min(rs),
                    "min_c": min(cs),
                    "max_r": max(rs),
                    "max_c": max(cs),
                    "centroid": (sum(rs) // len(rs), sum(cs) // len(cs)),
                }
            )
    # component adjacency (4-connectivity across component borders)
    adj: set[tuple[int, int]] = set()
    for r in range(height):
        for c in range(width):
            cid = comp_id[r][c]
            if r + 1 < height and comp_id[r + 1][c] != cid:
                other = comp_id[r + 1][c]
                adj.add((min(cid, other), max(cid, other)))
            if c + 1 < width and comp_id[r][c + 1] != cid:
                other = comp_id[r][c + 1]
                adj.add((min(cid, other), max(cid, other)))
    for comp in components:
        comp["adjacent"] = sorted(
            {b if a == comp["id"] else a for a, b in adj if comp["id"] in (a, b)}
        )
    return components


def shape_hash(cells: Sequence[tuple[int, int]]) -> str:
    """Translation-invariant shape signature (duck's object-hash idiom)."""
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm = sorted((r - min_r, c - min_c) for r, c in cells)
    return hashlib.sha1(repr(norm).encode()).hexdigest()[:12]


def state_signature(grid: Grid, *, noise_floor: int = 1) -> str:
    """Segmentation-graph signature of a frame.

    The hash covers, for every component with >= ``noise_floor`` cells:
    (color, size, translation-invariant shape hash, position), plus the
    component adjacency structure. Components below the noise floor (e.g.
    1-px animation sparkles) are excluded, so the signature is stable under
    cosmetic flicker while still distinguishing any real object move.
    """
    comps = connected_components(grid)
    if noise_floor > 1 and any(c["size"] < noise_floor for c in comps):
        # Erase sub-floor components (animation sparkles) into their dominant
        # neighbour colour, then re-segment, so a 1-px flicker changes neither
        # the small-object set nor the surrounding component's shape hash.
        cleaned = [list(row) for row in grid]
        by_id = {c["id"]: c for c in comps}
        for comp in comps:
            if comp["size"] >= noise_floor:
                continue
            neighbour_colors = [
                by_id[o]["color"] for o in comp["adjacent"] if by_id[o]["size"] >= noise_floor
            ]
            if not neighbour_colors:
                continue
            fill = max(set(neighbour_colors), key=neighbour_colors.count)
            for r, c in comp["cells"]:
                cleaned[r][c] = fill
        comps = connected_components(cleaned)
    kept = [c for c in comps if c["size"] >= noise_floor]
    kept_ids = {c["id"] for c in kept}
    items = [
        (c["color"], c["size"], shape_hash(c["cells"]), c["min_r"], c["min_c"])
        for c in kept
    ]
    adjacency = sorted(
        (c["id"], other)
        for c in kept
        for other in c["adjacent"]
        if other in kept_ids and other > c["id"]
    )
    payload = repr((sorted(items), adjacency)).encode()
    return hashlib.sha1(payload).hexdigest()[:16]


def grid_delta(before: Grid, after: Grid) -> dict[str, Any]:
    """Pixel-delta description between two frames (counts, bbox, transitions)."""
    changed: list[tuple[int, int]] = []
    transitions: dict[tuple[int, int], int] = {}
    height = min(len(before), len(after))
    for r in range(height):
        brow, arow = before[r], after[r]
        width = min(len(brow), len(arow))
        for c in range(width):
            if brow[c] != arow[c]:
                changed.append((r, c))
                key = (int(brow[c]), int(arow[c]))
                transitions[key] = transitions.get(key, 0) + 1
    if not changed:
        return {"count": 0, "bbox": None, "transitions": []}
    rs = [r for r, _ in changed]
    cs = [c for _, c in changed]
    top = sorted(transitions.items(), key=lambda kv: -kv[1])[:2]
    return {
        "count": len(changed),
        "bbox": (min(rs), min(cs), max(rs), max(cs)),
        "transitions": [
            {"from": _COLOR_CHARS[max(0, min(15, a))], "to": _COLOR_CHARS[max(0, min(15, b))], "px": n}
            for (a, b), n in top
        ],
    }


def describe_delta(delta: dict[str, Any]) -> str:
    if not delta or not delta.get("count"):
        return "no pixel change"
    bbox = delta["bbox"]
    trans = ", ".join(f"{t['from']}->{t['to']}({t['px']}px)" for t in delta["transitions"])
    return (
        f"{delta['count']}px changed in rows {bbox[0]}-{bbox[2]}, cols {bbox[1]}-{bbox[3]}"
        + (f"; colors {trans}" if trans else "")
    )


# ---------------------------------------------------------------------------
# Dedup archive + frontier
# ---------------------------------------------------------------------------

@dataclass
class ArchiveEntry:
    sig: str
    level: int
    first_step: int
    prefix_len: int
    visits: int = 1
    untried: set[str] = field(default_factory=set)
    tried: set[str] = field(default_factory=set)


class DedupArchive:
    """Deduped state archive keyed by segmentation-graph signature.

    Frontier economics (pre-registered): the only reset primitive is a full-
    episode RESET, so returning to a state costs a prefix replay of length d.
    frontier score = novelty / (1 + return_cost), where novelty = number of
    untried valid actions at that state and return_cost = prefix length.
    """

    def __init__(self, *, noise_floor: int = 1) -> None:
        self.noise_floor = max(1, int(noise_floor))
        self.states: dict[str, ArchiveEntry] = {}
        self.total_observations = 0
        self.best_level = 0

    # -- signatures ---------------------------------------------------------
    def signature(self, grid: Grid) -> str:
        return state_signature(grid, noise_floor=self.noise_floor)

    # -- observation --------------------------------------------------------
    def observe(
        self,
        grid: Grid,
        *,
        level: int,
        step: int,
        available_actions: Sequence[str] = (),
    ) -> dict[str, Any]:
        """Record the current state; returns {"sig": str, "new": bool}."""
        sig = self.signature(grid)
        self.total_observations += 1
        level = max(0, int(level or 0))
        self.best_level = max(self.best_level, level)
        actions = {str(a).strip().upper() for a in available_actions if str(a).strip()}
        actions.discard("RESET")
        entry = self.states.get(sig)
        if entry is None:
            self.states[sig] = ArchiveEntry(
                sig=sig,
                level=level,
                first_step=max(0, int(step or 0)),
                prefix_len=max(0, int(step or 0)),
                untried=set(actions),
            )
            return {"sig": sig, "new": True}
        entry.visits += 1
        if actions:
            # keep only actions still valid; newly valid actions become untried
            entry.untried = (entry.untried | (actions - entry.tried)) & actions
        return {"sig": sig, "new": False}

    def mark_tried(self, sig: str, action: str) -> None:
        entry = self.states.get(sig)
        if entry is None:
            return
        name = str(action or "").strip().upper()
        base = name.split("(", 1)[0]
        for candidate in {name, base}:
            entry.untried.discard(candidate)
            if candidate:
                entry.tried.add(candidate)

    # -- frontier -----------------------------------------------------------
    def frontier_score(self, entry: ArchiveEntry) -> float:
        return len(entry.untried) / (1.0 + entry.prefix_len)

    def frontier(self, top_k: int = 5) -> list[dict[str, Any]]:
        scored = sorted(
            (e for e in self.states.values() if e.untried),
            key=lambda e: (-self.frontier_score(e), e.prefix_len),
        )
        return [
            {
                "sig": e.sig[:8],
                "score": round(self.frontier_score(e), 4),
                "untried": sorted(e.untried),
                "prefix_len": e.prefix_len,
                "level": e.level,
                "visits": e.visits,
            }
            for e in scored[: max(0, top_k)]
        ]

    def untried_for(self, sig: str) -> list[str]:
        entry = self.states.get(sig)
        return sorted(entry.untried) if entry is not None else []

    # -- snapshot (REPL variable payload) ------------------------------------
    def snapshot(
        self,
        *,
        top_k: int = 5,
        current_sig: str | None = None,
        no_progress_turns: int = 0,
        last_probes: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        snap = {
            "unique_states": len(self.states),
            "total_observations": self.total_observations,
            "best_level": self.best_level,
            "no_progress_turns": int(no_progress_turns),
            "current_state_sig": (current_sig or "")[:8],
            "current_state_untried": self.untried_for(current_sig) if current_sig else [],
            "frontier": self.frontier(top_k=top_k),
        }
        if last_probes:
            snap["last_explore_probes"] = last_probes[-12:]
        # keep JSON-safe
        return json.loads(json.dumps(snap))


class ProgressTracker:
    """Counts analyzer turns without progress.

    progress := a new deduped archive state appeared since the previous turn
    OR the level/score incremented (pre-registered Phase-1 definition).

    v2 additions (v2_gating_design_2026-07-11 patch spec):
      * ``turns_since_levelup`` — analyzer turns since the last *real* level-up
        (reset in ``update()`` when ``level > _last_level``; the very first
        nonzero level observed is the baseline, not a level-up).
      * ``actions_on_current_level`` — actions taken since the last level-up,
        tracked via level-up action indices (``action_num`` at the analyzer
        turn where the level increase was observed).
      * ``levelups`` — count of real level-ups (0 = "no level-up yet").
    """

    def __init__(self) -> None:
        self.turns_without_progress = 0
        self._last_state_count = 0
        self._last_level = 0
        # v2: level-up recency + per-level action accounting
        self.turns_since_levelup = 0
        self.levelups = 0
        self.last_levelup_action = 0
        self.last_action_num = 0

    def update(
        self, *, state_count: int, level: int, action_num: int | None = None
    ) -> bool:
        progressed = state_count > self._last_state_count or level > self._last_level
        if action_num is not None:
            self.last_action_num = max(self.last_action_num, int(action_num))
        # Real level-up = increase beyond an already-observed nonzero level;
        # the initial 0 -> first-level transition is the baseline.
        if level > self._last_level and self._last_level > 0:
            self.levelups += 1
            self.turns_since_levelup = 0
            self.last_levelup_action = self.last_action_num
        else:
            self.turns_since_levelup += 1
        self._last_state_count = max(self._last_state_count, state_count)
        self._last_level = max(self._last_level, level)
        if progressed:
            self.turns_without_progress = 0
        else:
            self.turns_without_progress += 1
        return progressed

    @property
    def actions_on_current_level(self) -> int:
        """Actions taken since the last real level-up (all actions when none)."""
        return max(0, self.last_action_num - self.last_levelup_action)

    def reset(self) -> None:
        self.turns_without_progress = 0


# ---------------------------------------------------------------------------
# Animation-diff summarizer
# ---------------------------------------------------------------------------

def summarize_animation(frames: Sequence[Grid], *, char_cap: int = 240) -> str:
    """Text summary of pixel deltas across an action's intermediate frames.

    ``frames`` is the ordered list [pre-action, *intermediate, final]. Returns
    "" when there is no animation (<= 2 frames total, i.e. no intermediates).
    Text only — multi-image injection was shown by Tufa to fail on 27B models.
    """
    if len(frames) < 3:
        return ""
    step_deltas = [grid_delta(frames[i], frames[i + 1]) for i in range(len(frames) - 1)]
    nonzero = [d for d in step_deltas if d["count"]]
    if not nonzero:
        return ""
    nonzero_counts = sorted(d["count"] for d in nonzero)
    median = nonzero_counts[len(nonzero_counts) // 2]
    peak = nonzero_counts[-1]
    first_bbox = nonzero[0]["bbox"]
    last_bbox = nonzero[-1]["bbox"]

    def _center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
        return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

    c0, c1 = _center(first_bbox), _center(last_bbox)
    drift = ""
    if c0 != c1:
        drift = f"; change region drifted (r{c0[0]},c{c0[1]})->(r{c1[0]},c{c1[1]})"
    trans_totals: dict[str, int] = {}
    for d in step_deltas:
        for t in d["transitions"]:
            key = f"{t['from']}->{t['to']}"
            trans_totals[key] = trans_totals.get(key, 0) + t["px"]
    top_trans = sorted(trans_totals.items(), key=lambda kv: -kv[1])[:2]
    trans_text = ", ".join(f"{k}({v}px)" for k, v in top_trans)
    net = grid_delta(frames[0], frames[-1])
    text = (
        f"ANIMATION ({len(frames) - 2} intermediate frames): per-frame delta median {median}px,"
        f" peak {peak}px{drift}; top color flips {trans_text};"
        f" net vs pre-action: {describe_delta(net)}."
    )
    if len(text) > char_cap:
        text = text[: max(0, char_cap - 3)].rstrip() + "..."
    return text


# ---------------------------------------------------------------------------
# Scripted explore()
# ---------------------------------------------------------------------------

_SIMPLE_ACTION_ORDER = ["UP", "DOWN", "LEFT", "RIGHT", "SPACE"]


def _mouse_candidates(grid: Grid, *, limit: int) -> list[tuple[int, int, str]]:
    """Candidate MOUSE targets: centroids of salient components.

    Salience: rare shape/color combos first, small objects before large,
    background (largest component / dominant color) excluded.
    """
    comps = connected_components(grid)
    if not comps:
        return []
    background = max(comps, key=lambda c: c["size"])
    shape_counts: dict[tuple[int, str], int] = {}
    hashes: dict[int, str] = {}
    for comp in comps:
        h = shape_hash(comp["cells"])
        hashes[comp["id"]] = h
        key = (comp["color"], h)
        shape_counts[key] = shape_counts.get(key, 0) + 1
    scored = []
    for comp in comps:
        if comp["id"] == background["id"] or comp["size"] > 1024:
            continue
        rarity = shape_counts[(comp["color"], hashes[comp["id"]])]
        scored.append((rarity, comp["size"], comp["min_r"], comp["min_c"], comp))
    scored.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
    out: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int]] = set()
    for _, _, _, _, comp in scored:
        point = comp["centroid"]
        if point in seen:
            continue
        seen.add(point)
        out.append((point[0], point[1], _COLOR_CHARS[max(0, min(15, comp["color"]))]))
        if len(out) >= limit:
            break
    return out


def plan_probe_actions(
    valid_actions: Sequence[str],
    *,
    grid: Grid,
    untried: Sequence[str] = (),
    budget: int = 8,
    mouse_candidates: int = 4,
    rotation: int = 0,
) -> list[dict[str, Any]]:
    """Ordered probe plan: untried simple actions, then MOUSE clicks on salient
    objects, then remaining (already-tried) simple actions, capped at budget.
    ``rotation`` rotates the simple-action order so successive replans (after a
    state change) do not keep re-probing the same first action."""
    valid = {str(a).strip().upper() for a in valid_actions if str(a).strip()}
    valid.discard("RESET")
    untried_set = {str(a).strip().upper() for a in untried}
    order = list(_SIMPLE_ACTION_ORDER)
    if order:
        shift = int(rotation) % len(order)
        order = order[shift:] + order[:shift]
    plan: list[dict[str, Any]] = []
    for name in order:
        if name in valid and (not untried_set or name in untried_set):
            plan.append({"action": name})
    if "MOUSE" in valid:
        for row, col, color in _mouse_candidates(grid, limit=mouse_candidates):
            plan.append({"action": "MOUSE", "row": int(row), "col": int(col), "_target_color": color})
    for name in order:
        if name in valid and not any(p["action"] == name for p in plan):
            plan.append({"action": name})
    return plan[: max(0, int(budget))]


def _display(action: dict[str, Any]) -> str:
    if action.get("action") == "MOUSE":
        return f"MOUSE(row={action.get('row')}, col={action.get('col')})"
    return str(action.get("action", ""))


def run_explore(
    *,
    execute: Callable[[dict[str, Any]], dict[str, Any]],
    get_state: Callable[[], tuple[Grid, int, int]],
    valid_actions: Sequence[str],
    archive: DedupArchive,
    budget: int = 8,
    mouse_candidates: int = 4,
    min_time_remaining: float = 300.0,
    should_stop: Callable[[], bool] | None = None,
) -> list[dict[str, Any]]:
    """Scripted harness-side explore. Executes up to ``budget`` real actions
    via ``execute`` (a step_env-style callback taking one action dict) and
    returns a list of probe records. The environment genuinely moves — there
    is no save-state — so probes are recorded as a chain, replanning the
    probe list whenever the state signature changes."""
    probes: list[dict[str, Any]] = []
    executed = 0
    while executed < budget:
        if should_stop is not None and should_stop():
            break
        grid0, level0, step0 = get_state()
        sig0 = archive.signature(grid0)
        archive.observe(grid0, level=level0, step=step0, available_actions=valid_actions)
        untried = archive.untried_for(sig0)
        plan = plan_probe_actions(
            valid_actions,
            grid=grid0,
            untried=untried,
            budget=budget - executed,
            mouse_candidates=mouse_candidates,
            rotation=executed,
        )
        if not plan:
            break
        advanced = False
        for action in plan:
            if should_stop is not None and should_stop():
                break
            payload = {k: v for k, v in action.items() if not k.startswith("_")}
            result = execute(payload) or {}
            executed += 1
            display = _display(action)
            record: dict[str, Any] = {
                "action": display,
                "executed": bool(result.get("executed")),
            }
            if action.get("_target_color"):
                record["target_color"] = action["_target_color"]
            if not result.get("executed"):
                record["error"] = str(result.get("error", ""))[:120]
                probes.append(record)
                if executed >= budget:
                    break
                continue
            grid1, level1, step1 = get_state()
            obs = archive.observe(
                grid1,
                level=level1,
                step=step1,
                available_actions=result.get("valid_actions") or [],
            )
            archive.mark_tried(sig0, display)
            delta = grid_delta(grid0, grid1)
            record.update(
                {
                    "board_changed": bool(delta["count"]),
                    "new_state": bool(obs["new"]),
                    "delta": describe_delta(delta),
                    "level": int(level1),
                    "level_up": level1 > level0,
                }
            )
            if result.get("animation_summary"):
                record["animation"] = str(result["animation_summary"])
            for flag in ("level_completed", "game_over", "run_complete"):
                if result.get(flag):
                    record[flag] = True
            probes.append(record)
            remaining = result.get("time_remaining_seconds")
            out_of_time = (
                isinstance(remaining, (int, float)) and remaining < min_time_remaining
            )
            terminal = any(
                result.get(flag)
                for flag in ("level_completed", "game_over", "run_complete", "done")
            )
            if terminal or out_of_time or executed >= budget:
                return probes
            if obs["sig"] != sig0:
                advanced = True
                break  # replan from the new state
            grid0, level0, step0, sig0 = grid1, level1, step1, obs["sig"]
        if not advanced and executed >= budget:
            break
        if not advanced and not plan:
            break
        if not advanced:
            # exhausted the plan without changing state signature
            break
    return probes


def render_explore_summary(
    probes: list[dict[str, Any]],
    archive: DedupArchive,
    *,
    trigger_turns: int,
    char_cap: int = 1500,
) -> str:
    """Curated <=500-token exploration report injected into the next user
    message (500 tokens ~ 1500 chars at duck's conservative 3 chars/token)."""
    lines = [
        "[HARNESS EXPLORATION REPORT]",
        (
            f"No progress for {trigger_turns} turns, so the harness ran a scripted probe"
            f" of {len(probes)} real actions (these actions already happened; the current"
            " frame reflects them). Results:"
        ),
    ]
    for probe in probes:
        if not probe.get("executed"):
            lines.append(f"- {probe['action']}: NOT executed ({probe.get('error', 'invalid')})")
            continue
        bits = []
        if probe.get("run_complete"):
            bits.append("RUN COMPLETE")
        elif probe.get("level_up") or probe.get("level_completed"):
            bits.append(f"LEVEL UP -> level {probe.get('level')}")
        elif probe.get("game_over"):
            bits.append("GAME OVER (auto-reset follows)")
        if probe.get("board_changed"):
            bits.append(("NEW state; " if probe.get("new_state") else "seen-before state; ") + probe.get("delta", ""))
        else:
            bits.append("no board change")
        if probe.get("target_color"):
            bits.append(f"clicked color-{probe['target_color']} object")
        if probe.get("animation"):
            bits.append(probe["animation"])
        lines.append(f"- {probe['action']}: " + "; ".join(bits))
    frontier = archive.frontier(top_k=3)
    lines.append(
        f"Archive: {len(archive.states)} unique states over {archive.total_observations} observations;"
        f" best level {archive.best_level}."
    )
    if frontier:
        best = frontier[0]
        lines.append(
            "Most promising frontier state: "
            f"sig {best['sig']} (score {best['score']}, untried: {', '.join(best['untried'][:6])},"
            f" {best['prefix_len']} actions deep)."
        )
    lines.append(
        "Full details are in the `explore_archive` variable inside the python tool."
        " Prioritize actions marked NEW state or LEVEL UP."
    )
    text = "\n".join(lines)
    if len(text) > char_cap:
        text = text[: max(0, char_cap - 3)].rstrip() + "..."
    return text
