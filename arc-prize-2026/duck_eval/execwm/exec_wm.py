"""exec-WM: executable world model controller for the duck harness (arm: execwm).

Mechanism class (three independent ~99-100% systems + arXiv 2605.05138): write each
game's mechanics as an executable program, VERIFY it against recorded history, and
PLAN inside it with search. Adapted for the offline Qwen3.8-27B rail, where tokens
are the binding constraint and actions are nearly free (eps=0.17):

  PHASE E  scripted exploration (deterministic, zero LLM tokens)
  PHASE I  induction: deterministic object-delta rule MINING first; the LLM is
           called only for actions mining cannot explain (budget-capped, lean
           prompt, constrained JSON fill -- never trusted unverified)
  PHASE V  mechanical prequential verification against ALL recorded transitions
           (interior-masked; lawbook-style run gating). Unverified => never used.
  PHASE P  BFS inside the verified program; the plan executes in the real game
           one action at a time with a per-step settled-frame prediction check.
           A break aborts the plan, feeds the counterexample back to the miner.
  FALLBACK per level: if no verified program or the plan budget is exhausted the
           stock ToolAgent runs UNTOUCHED for that level. Fallback == the
           certified floor behaviour. exec-WM can add levels, never remove them.

Object-centric by construction (lawbook, 647 real-27B actions: board-keyed
memoization DEAD at 2.6%/47%, global effect signatures DEAD, object-level laws
83.7-92.4% precision with run gating).

HUD strips are excluded from every comparison (P0.3: the full-grid signature
provably cannot fire; 18/25 games have a border strip ticking on >=50% of steps).

stdlib only. No LLM-router package, no vendor SDK -- raw HTTP via requests, lazily
imported, optional. Deployed into the bundle as inference/agent/exec_wm.py by the patch
cell; this file in duck_eval/execwm/ is the single source of truth.
"""
from __future__ import annotations

import json
import os
import threading
import time

EXECWM_VERSION = "v1"

# ---- sealed parameters (prereg execwm_prereg_2026-08-25.md; not tunable) ----
MOVE_ACTIONS = ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5")
E_REPEATS = 4            # times each candidate action is pressed in PHASE E
E_MAX_TURNS = 4          # controller turns PHASE E may consume per level
PROBE_BATCH = 10         # env actions executed per controller turn in PHASE E
PLAN_BATCH = 24          # plan steps executed per controller turn in PHASE P
MASK_MIN_PAIRS = 8       # frame pairs before the HUD mask converges
MASK_RATE = 0.5          # change rate for a row/col to count as HUD
MASK_BORDER = 6          # a HUD run must touch within this of the border
MAX_DELTA = 8            # translation search window (|dr|,|dc| <= MAX_DELTA)
MIN_SPRITE_CELLS = 2     # a moving component must have >= this many cells
VERIFY_MIN_N = 3         # prequential checks required before a rule is trusted
VERIFY_PRECISION = 0.90  # acceptance threshold
VERIFY_TAIL_OK = 2       # the last k checks must all be exact (run gating)
MIN_VERIFIED_MOVES = 2   # BFS needs at least this many verified move rules
MAX_BREAKS_PER_LEVEL = 3 # prediction breaks before the level falls back
MAX_GOAL_COLORS = 6      # candidate goal colors ranked by rarity
GOAL_MAX_CELLS = 30      # a goal color must be this rare on the interior
MAX_SWEEP_PLANS = 48     # coverage-sweep plans after goals are exhausted
MAX_PLANS_PER_LEVEL = 96 # total executed plans per level before fallback
LLM_CALLS_PER_GAME = 2   # PHASE I LLM budget (0 disables)
LLM_TIMEOUT_S = 180.0


def _now() -> float:
    return time.monotonic()


# ===========================================================================
# grid helpers
# ===========================================================================
def grid_of(frame_payload):
    """Normalize a runtime-state frame payload's grid to tuple-of-tuples."""
    raw = frame_payload.get("grid") if isinstance(frame_payload, dict) else None
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(tuple(int(c) for c in row) for row in raw if isinstance(row, (list, tuple)))


def color_counts(grid):
    counts = {}
    for row in grid:
        for c in row:
            counts[c] = counts.get(c, 0) + 1
    return counts


# ===========================================================================
# HUD mask (P0.4 lineage: border-strip detection; empty before convergence)
# ===========================================================================
class HudMask:
    """Rows/cols that change on >= MASK_RATE of same-level consecutive frame
    pairs, in maximal runs touching within MASK_BORDER of the border. Degrades
    to an empty mask before MASK_MIN_PAIRS pairs have been seen."""

    def __init__(self):
        self.pairs = 0
        self._row_hits = {}
        self._col_hits = {}
        self._rows = 0
        self._cols = 0

    def observe(self, before, after):
        if not before or not after or len(before) != len(after):
            return
        self._rows = len(before)
        self._cols = max(len(r) for r in before)
        self.pairs += 1
        changed_rows, changed_cols = set(), set()
        for r, (rb, ra) in enumerate(zip(before, after)):
            if rb == ra:
                continue
            for c, (b, a) in enumerate(zip(rb, ra)):
                if b != a:
                    changed_rows.add(r)
                    changed_cols.add(c)
        for r in changed_rows:
            self._row_hits[r] = self._row_hits.get(r, 0) + 1
        for c in changed_cols:
            self._col_hits[c] = self._col_hits.get(c, 0) + 1

    def _runs(self, hits, size):
        hot = sorted(i for i, n in hits.items() if self.pairs and n / self.pairs >= MASK_RATE)
        out = set()
        run = []
        for i in hot + [None]:
            if run and (i is None or i != run[-1] + 1):
                if min(run) < MASK_BORDER or max(run) >= size - MASK_BORDER:
                    out.update(run)
                run = []
            if i is not None:
                run.append(i)
        return out

    def masked_rows(self):
        if self.pairs < MASK_MIN_PAIRS:
            return set()
        return self._runs(self._row_hits, self._rows)

    def masked_cols(self):
        if self.pairs < MASK_MIN_PAIRS:
            return set()
        return self._runs(self._col_hits, self._cols)

    def excluded(self, r, c):
        return r in self.masked_rows() or c in self.masked_cols()

    def interior_cells(self, grid):
        mr, mc = self.masked_rows(), self.masked_cols()
        for r, row in enumerate(grid):
            if r in mr:
                continue
            for c in range(len(row)):
                if c not in mc:
                    yield r, c


def interior_diff(before, after, mask: HudMask):
    """Changed interior cells between two settled frames."""
    mr, mc = mask.masked_rows(), mask.masked_cols()
    out = []
    for r, (rb, ra) in enumerate(zip(before, after)):
        if r in mr or rb == ra:
            continue
        for c, (b, a) in enumerate(zip(rb, ra)):
            if b != a and c not in mc:
                out.append((r, c))
    return out


# ===========================================================================
# transitions (rebuilt from the harness runtime-state history every turn)
# ===========================================================================
class Transition:
    __slots__ = ("action", "before", "after", "level_before", "level_after")

    def __init__(self, action, before, after, level_before, level_after):
        self.action = action
        self.before = before
        self.after = after
        self.level_before = level_before
        self.level_after = level_after


def read_state(state_path):
    """Parse tool_runtime_state.json with stdlib only. Returns (current, history)
    where current is {'grid':..., 'level':..., 'step':...} and history is a list
    of {'action', 'grid', 'level'}."""
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return None, []
    cur = payload.get("current_frame") or {}
    current = {"grid": grid_of(cur), "level": int(cur.get("level", 1) or 1),
               "step": int(cur.get("step", 0) or 0)}
    history = []
    for entry in payload.get("history") or []:
        if not isinstance(entry, dict):
            continue
        fr = entry.get("frame") or {}
        history.append({"action": str(entry.get("action", "")).strip(),
                        "grid": grid_of(fr),
                        "level": int(fr.get("level", 1) or 1)})
    return current, history


# model-facing display labels <-> engine action ids (mirror of the bundle's
# inference/agent/action_names.py; history entries store the MODEL labels)
ENGINE_TO_MODEL = {"ACTION1": "UP", "ACTION2": "DOWN", "ACTION3": "LEFT",
                   "ACTION4": "RIGHT", "ACTION5": "SPACE", "ACTION6": "MOUSE",
                   "RESET": "RESET"}
MODEL_TO_ENGINE = {v: k for k, v in ENGINE_TO_MODEL.items()}


def action_token(display: str) -> str:
    """Engine action id from a history display label: 'UP' -> 'ACTION1',
    'MOUSE(row=3, col=4)' -> 'ACTION6', 'ACTION1' -> 'ACTION1'."""
    raw = display.split(" ", 1)[0].split("(", 1)[0].strip().upper()
    if raw in ENGINE_TO_MODEL:
        return raw
    return MODEL_TO_ENGINE.get(raw, raw)


def transitions_from_history(history):
    out = []
    for prev, cur in zip(history, history[1:]):
        act = action_token(cur["action"])
        if not act or not prev["grid"] or not cur["grid"]:
            continue
        out.append(Transition(act, prev["grid"], cur["grid"],
                              prev["level"], cur["level"]))
    return out


# ===========================================================================
# translation mining (PHASE I, deterministic half)
# ===========================================================================
def detect_translation(before, after, mask: HudMask):
    """Classify one interior transition.

    Returns one of
      ("noop", None)
      ("move", (dr, dc, departures:set, arrivals:set))
      ("unexplained", None)
    A move must explain EVERY interior diff cell as departure or arrival.
    """
    diff = interior_diff(before, after, mask)
    if not diff:
        return "noop", None
    diffset = set(diff)
    rows, cols = len(before), max(len(r) for r in before)
    counts = color_counts(before)
    best = None
    for dr in range(-MAX_DELTA, MAX_DELTA + 1):
        for dc in range(-MAX_DELTA, MAX_DELTA + 1):
            if dr == 0 and dc == 0:
                continue
            departures, arrivals = set(), set()
            for (r, c) in diff:
                r2, c2 = r + dr, c + dc
                if 0 <= r2 < rows and 0 <= c2 < cols and (r2, c2) in diffset \
                        and after[r2][c2] == before[r][c]:
                    departures.add((r, c))
                    arrivals.add((r2, c2))
            if len(departures) < MIN_SPRITE_CELLS:
                continue
            if diffset - departures - arrivals:
                continue
            # Symmetric ambiguity ("the background moved the other way") is
            # broken by RARITY: the true mover is the component whose colors
            # are rarest on the board (the sprite), never the background.
            rarity = sum(counts.get(before[r][c], 0) for (r, c) in departures)
            key = (rarity, abs(dr) + abs(dc), abs(dr), abs(dc))
            if best is None or key < best[0]:
                best = (key, dr, dc, departures, arrivals)
    if best is None:
        return "unexplained", None
    _, dr, dc, departures, arrivals = best
    return "move", (dr, dc, departures, arrivals)


class SpritePattern:
    """Relative cell->color pattern of the moving component, anchored at its
    top-left. Also remembers which color is rarest (the search anchor)."""

    def __init__(self, cells_to_color):
        r0 = min(r for r, _ in cells_to_color)
        c0 = min(c for _, c in cells_to_color)
        self.rel = {(r - r0, c - c0): col for (r, c), col in cells_to_color.items()}
        self.colors = frozenset(self.rel.values())
        counts = {}
        for col in self.rel.values():
            counts[col] = counts.get(col, 0) + 1
        self.anchor_color = min(counts, key=lambda k: (counts[k], k))
        self.anchor_offsets = [rc for rc, col in self.rel.items() if col == self.anchor_color]

    def find(self, grid):
        """All positions (top-left) where the full pattern matches. Uses the
        rarest color as anchor to keep the scan cheap."""
        rows, cols = len(grid), max((len(r) for r in grid), default=0)
        hits = []
        seen = set()
        for r in range(rows):
            row = grid[r]
            for c in range(len(row)):
                if row[c] != self.anchor_color:
                    continue
                for (ar, ac) in self.anchor_offsets:
                    pr, pc = r - ar, c - ac
                    if (pr, pc) in seen:
                        continue
                    seen.add((pr, pc))
                    ok = True
                    for (dr, dc), col in self.rel.items():
                        rr, cc = pr + dr, pc + dc
                        if not (0 <= rr < rows and 0 <= cc < len(grid[rr])) or grid[rr][cc] != col:
                            ok = False
                            break
                    if ok:
                        hits.append((pr, pc))
        return sorted(set(hits))


class Rule:
    __slots__ = ("kind", "delta", "n", "ok", "tail", "verified")

    def __init__(self, kind, delta=None):
        self.kind = kind          # "noop" | "move"
        self.delta = delta        # (dr, dc) for "move"
        self.n = 0                # prequential checks
        self.ok = 0               # exact matches
        self.tail = 0             # consecutive exact matches (run gating)
        self.verified = False

    def precision(self):
        return self.ok / self.n if self.n else 0.0

    def as_dict(self):
        return {"kind": self.kind, "delta": self.delta, "n": self.n,
                "ok": self.ok, "precision": round(self.precision(), 4),
                "verified": self.verified}


class WorldModel:
    """The per-level executable program: sprite pattern + per-action rules +
    learned underlay + permeable-color set. Everything mechanical."""

    def __init__(self):
        self.sprite: SpritePattern | None = None
        self.rules: dict[str, Rule] = {}
        self.underlay: dict[tuple, int] = {}      # cell -> revealed color
        self.permeable: set[int] = set()          # colors the sprite overwrote
        self.blockers: set[int] = set()           # colors observed to refuse a move
        self.unexplained = 0                       # instances the miner cannot explain
        self.mined_from = 0                        # transitions consumed

    # ---- induction (deterministic) ----
    def mine(self, transitions, mask: HudMask, llm_hints=None):
        per_action = {}
        order = []           # classification per transition, in replay order
        pattern_votes = {}   # normalized rel-pattern -> [count, rarity, cells]
        for t in transitions:
            if t.level_before != t.level_after or t.action == "RESET":
                continue
            if t.action not in MOVE_ACTIONS:
                continue
            kind, info = detect_translation(t.before, t.after, mask)
            per_action.setdefault(t.action, []).append((kind, info, t, len(order)))
            order.append(kind)
            if kind == "move":
                dr, dc, departures, arrivals = info
                counts = color_counts(t.before)
                cells = {cell: t.before[cell[0]][cell[1]] for cell in departures}
                rarity = sum(counts.get(v, 0) for v in cells.values())
                r0 = min(r for r, _ in cells)
                c0 = min(c for _, c in cells)
                key = frozenset(((r - r0, c - c0), col)
                                for (r, c), col in cells.items())
                vote = pattern_votes.setdefault(key, [0, rarity, cells])
                vote[0] += 1
                for cell in departures - arrivals:
                    self.underlay[cell] = t.after[cell[0]][cell[1]]
                for cell in arrivals - departures:
                    self.permeable.add(t.before[cell[0]][cell[1]])
        if pattern_votes:
            # CONSENSUS: the sprite is the pattern that recurs across the most
            # move instances; rarity only breaks ties. A one-off translation
            # artifact (an animation, a pickup) can never outvote the sprite.
            _, _, cells = max(pattern_votes.values(),
                              key=lambda v: (v[0], -v[1]))
            self.sprite = SpritePattern(cells)
        # An unexplained event (pickup, toggle, spawn) invalidates blocker
        # evidence recorded before it -- the world is stateful and a door can
        # open. Blockers therefore re-mine from POST-EVENT no-ops only.
        last_event = max((i for i, k in enumerate(order) if k == "unexplained"),
                         default=-1)
        self.blockers = set()
        # blocker mining: an action with move consensus that ALSO produced
        # no-ops reveals which colors refuse the move -- the colors standing in
        # the would-be destination footprint (minus known-permeable ones).
        if self.sprite is not None:
            for action, instances in per_action.items():
                deltas = {info[:2] for k, info, _, _ in instances if k == "move"}
                if len(deltas) != 1:
                    continue
                dr, dc = next(iter(deltas))
                for k, _info, t, idx in instances:
                    if k != "noop" or idx <= last_event:
                        continue
                    hits = self.sprite.find(t.before)
                    if len(hits) != 1:
                        continue
                    pr, pc = hits[0][0] + dr, hits[0][1] + dc
                    cur = {(hits[0][0] + rr, hits[0][1] + cc)
                           for (rr, cc) in self.sprite.rel}
                    nonperm = set()
                    off_board = False
                    for (rr, cc) in self.sprite.rel:
                        r2, c2 = pr + rr, pc + cc
                        if (r2, c2) in cur:
                            continue
                        if 0 <= r2 < len(t.before) and 0 <= c2 < len(t.before[r2]):
                            col = t.before[r2][c2]
                            if col not in self.permeable:
                                nonperm.add(col)
                        else:
                            off_board = True
                    # credit assignment must be UNAMBIGUOUS: only a footprint
                    # whose non-permeable colors are a singleton names its
                    # blocker (a {wall, lane} footprint blames neither).
                    if len(nonperm) == 1 and not off_board:
                        self.blockers.add(next(iter(nonperm)))
        self.rules = {}
        for action, instances in per_action.items():
            kinds = [k for k, _, _, _ in instances]
            moves = [info for k, info, _, _ in instances if k == "move"]
            if moves:
                deltas = {(dr, dc) for dr, dc, _, _ in moves}
                if len(deltas) == 1:
                    self.rules[action] = Rule("move", next(iter(deltas)))
                continue
            if kinds and all(k == "noop" for k in kinds):
                self.rules[action] = Rule("noop")
        # LLM-filled hypotheses enter here on the same footing -- as CANDIDATES
        # that PHASE V must pass before use. Mining always wins a conflict.
        for action, hint in (llm_hints or {}).items():
            if action in self.rules or action not in MOVE_ACTIONS:
                continue
            if isinstance(hint, dict) and hint.get("type") == "translate":
                try:
                    dr, dc = int(hint["dr"]), int(hint["dc"])
                except Exception:
                    continue
                if (dr or dc) and abs(dr) <= MAX_DELTA and abs(dc) <= MAX_DELTA:
                    self.rules[action] = Rule("move", (dr, dc))
        self.blockers -= self.permeable
        self.unexplained = sum(1 for k in order if k == "unexplained")
        self.mined_from = len(transitions)
        return self.rules

    # ---- prediction (used by V and by P's per-step check) ----
    def occupancy(self, grid, pos):
        """Static view of the board with the sprite removed (underlay where
        known, else None = unknown)."""
        cells = {}
        for (dr, dc) in self.sprite.rel:
            cell = (pos[0] + dr, pos[1] + dc)
            cells[cell] = self.underlay.get(cell)
        return cells

    def predict(self, grid, pos, action, mask: HudMask):
        """Predict (next_pos, predicted_cells) for a verified/candidate rule.
        predicted_cells maps cell -> color for every cell whose post-action
        value the model claims to know; None values mean 'unknown, don't
        check'. Returns None if the model cannot predict this action."""
        rule = self.rules.get(action)
        if rule is None or self.sprite is None or pos is None:
            return None
        if rule.kind == "noop":
            return pos, {}
        dr, dc = rule.delta
        rows, cols = len(grid), max(len(r) for r in grid)
        new_pos = (pos[0] + dr, pos[1] + dc)
        target_cells = {}
        for (rr, cc), col in self.sprite.rel.items():
            r2, c2 = new_pos[0] + rr, new_pos[1] + cc
            if not (0 <= r2 < rows and 0 <= c2 < cols):
                return pos, {}  # off-board => predict blocked no-op
            target_cells[(r2, c2)] = col
        current_cells = {(pos[0] + rr, pos[1] + cc) for (rr, cc) in self.sprite.rel}
        blocked = False
        for cell in target_cells:
            if cell in current_cells:
                continue
            col = grid[cell[0]][cell[1]]
            if col not in self.permeable:
                blocked = True
                break
        if blocked:
            return pos, {}      # predict exact no-op
        pred = {}
        for cell, col in target_cells.items():
            pred[cell] = col
        for cell in current_cells - set(target_cells):
            pred[cell] = self.underlay.get(cell)   # None = unknown, don't check
        return new_pos, pred

    def check_prediction(self, before, after, pos, action, mask: HudMask):
        """Compare a prediction against the settled real frame. Returns
        (exact:bool, definite_checked:int, new_pos) or None if no prediction."""
        out = self.predict(before, pos, action, mask)
        if out is None:
            return None
        new_pos, pred = out
        checked = 0
        exact = True
        for r, c in mask.interior_cells(after):
            want = pred.get((r, c), before[r][c] if r < len(before) and c < len(before[r]) else None)
            if want is None:
                # unknown underlay: learn it instead of judging it
                if (r, c) in pred:
                    self.underlay[(r, c)] = after[r][c]
                continue
            checked += 1
            if after[r][c] != want:
                exact = False
        return exact, checked, new_pos

    # ---- verification (PHASE V) ----
    def verify(self, transitions, mask: HudMask):
        """Prequential replay over the recorded history. Resets and refills
        every rule's counters; sets rule.verified by threshold + run gating."""
        for rule in self.rules.values():
            rule.n = rule.ok = rule.tail = 0
            rule.verified = False
        if self.sprite is None:
            return {}
        for t in transitions:
            if t.level_before != t.level_after or t.action not in self.rules:
                continue
            hits = self.sprite.find(t.before)
            if len(hits) != 1:
                continue
            res = self.check_prediction(t.before, t.after, hits[0], t.action, mask)
            if res is None:
                continue
            exact, checked, _ = res
            rule = self.rules[t.action]
            if not exact:
                # A miss that the miner itself cannot explain as any clean
                # translation/no-op is a WORLD EVENT (pickup, teleport,
                # spawn), not evidence against the rule -- it neither
                # confirms nor refutes, so it stays out of the count. A miss
                # that IS a clean noop/other-translation is a real failure.
                kind, _ = detect_translation(t.before, t.after, mask)
                if kind == "unexplained":
                    continue
            rule.n += 1
            if exact:
                rule.ok += 1
                rule.tail += 1
            else:
                rule.tail = 0
        for rule in self.rules.values():
            # Acceptance = sample size + precision. A consecutive-tail demand
            # was tried and REJECTED on the real ls20 rail: the game's ~1/43
            # counter-teleport made one terminal anomaly permanently kill a
            # 30-for-31 rule and halve the board's connectivity. Rare model
            # noise is priced by the precision threshold and by the per-level
            # break budget, not by a sudden-death tail.
            rule.verified = (rule.n >= VERIFY_MIN_N
                            and rule.precision() >= VERIFY_PRECISION)
        return {a: r.as_dict() for a, r in self.rules.items()}

    def verified_moves(self):
        return {a: r for a, r in self.rules.items()
                if r.verified and r.kind == "move"}


# ===========================================================================
# planner (PHASE P): BFS over sprite positions inside the verified program
# ===========================================================================
def bfs_reachable(model: WorldModel, grid, start):
    """BFS over sprite positions using only VERIFIED move rules and the
    conservative permeability model. Returns {pos: (prev_pos, action)}."""
    moves = model.verified_moves()
    sprite = model.sprite
    rows, cols = len(grid), max(len(r) for r in grid)
    parents = {start: (None, None)}
    frontier = [start]
    while frontier:
        nxt = []
        for pos in frontier:
            cur_cells = {(pos[0] + rr, pos[1] + cc) for (rr, cc) in sprite.rel}
            for action, rule in moves.items():
                dr, dc = rule.delta
                np_ = (pos[0] + dr, pos[1] + dc)
                if np_ in parents:
                    continue
                ok = True
                for (rr, cc) in sprite.rel:
                    r2, c2 = np_[0] + rr, np_[1] + cc
                    if not (0 <= r2 < rows and 0 <= c2 < cols):
                        ok = False
                        break
                    if (r2, c2) in cur_cells:
                        continue
                    col = model.underlay.get((r2, c2), grid[r2][c2])
                    # cells currently under the sprite footprint use underlay
                    if col not in model.permeable:
                        ok = False
                        break
                if ok:
                    parents[np_] = (pos, action)
                    nxt.append(np_)
        frontier = nxt
    return parents


def path_to(parents, goal):
    if goal not in parents:
        return None
    actions = []
    pos = goal
    while parents[pos][0] is not None:
        prev, action = parents[pos]
        actions.append(action)
        pos = prev
    actions.reverse()
    return actions


def goal_targets(model: WorldModel, grid, mask: HudMask, pos):
    """Ranked candidate goal positions: for each rare interior color, the
    reachable sprite positions whose footprint touches (or lands adjacent to)
    a cell of that color. Rarest colors first, then nearest cell."""
    counts = {}
    mr, mc = mask.masked_rows(), mask.masked_cols()
    cur_cells = {(pos[0] + rr, pos[1] + cc) for (rr, cc) in model.sprite.rel}
    for r, row in enumerate(grid):
        if r in mr:
            continue
        for c in range(len(row)):
            if c in mc or (r, c) in cur_cells:
                continue
            counts.setdefault(row[c], []).append((r, c))
    bg = max(counts, key=lambda k: len(counts[k])) if counts else None
    boring = {bg} | model.sprite.colors | model.permeable | set(model.underlay.values())
    ranked = sorted((k for k in counts
                     if k not in boring and len(counts[k]) <= GOAL_MAX_CELLS),
                    key=lambda k: (len(counts[k]), k))
    out = []
    for color in ranked[:MAX_GOAL_COLORS]:
        out.append((color, counts[color]))
    return out


def frontier_probes(model: WorldModel, grid, parents):
    """Cheap experiments that EXPAND the verified model: from a reachable
    position, one move whose destination footprint contains colors of unknown
    permeability (not yet walked on, not yet observed to block). Executing it
    either unlocks a new region (the miner adds the color to `permeable`) or
    records a blocker -- both are progress, and both cost one action."""
    rows, cols = len(grid), max(len(r) for r in grid)
    moves = model.verified_moves()
    out = []
    for pos in parents:
        cur = {(pos[0] + rr, pos[1] + cc) for (rr, cc) in model.sprite.rel}
        for action, rule in moves.items():
            dr, dc = rule.delta
            np_ = (pos[0] + dr, pos[1] + dc)
            if np_ in parents:
                continue
            unknown = set()
            ok = True
            for (rr, cc) in model.sprite.rel:
                r2, c2 = np_[0] + rr, np_[1] + cc
                if not (0 <= r2 < rows and 0 <= c2 < cols):
                    ok = False
                    break
                if (r2, c2) in cur:
                    continue
                col = model.underlay.get((r2, c2), grid[r2][c2])
                if col in model.permeable:
                    continue
                if col in model.blockers:
                    ok = False
                    break
                unknown.add(col)
            if ok and unknown:
                out.append((len(unknown), pos, action, tuple(sorted(unknown))))
    counts = color_counts(grid)
    # rarest unknown color first: a 12-cell door outranks a 2600-cell wall
    out.sort(key=lambda x: (min(counts.get(c, 0) for c in x[3]), x[0], x[1], x[2]))
    return out


def positions_touching(model: WorldModel, cells, parents, dilate=1):
    """Reachable sprite positions whose footprint covers any of `cells` (or a
    cell within `dilate` of one -- goals of non-permeable colors can only be
    stood NEXT TO), nearest (by BFS insertion order) first."""
    cellset = set(cells)
    if dilate:
        for (r, c) in list(cellset):
            for dr in range(-dilate, dilate + 1):
                for dc in range(-dilate, dilate + 1):
                    cellset.add((r + dr, c + dc))
    touching = []
    for p in parents:
        for (rr, cc) in model.sprite.rel:
            if (p[0] + rr, p[1] + cc) in cellset:
                touching.append(p)
                break
    return touching


# ===========================================================================
# PHASE I, LLM half (optional, budget-capped, verified before use)
# ===========================================================================
def llm_fill(unexplained, evidence_lines, llm_cfg):
    """One lean chat call asking for constrained JSON rules for the actions
    mining could not explain. Any failure returns {}. Never raises."""
    if not llm_cfg or not unexplained:
        return {}, 0
    try:
        import requests  # lazy; the bundle already depends on it
    except Exception:
        return {}, 0
    prompt = (
        "Recorded effects per action in a 64x64 grid game:\n"
        + "\n".join(evidence_lines[:24])
        + "\nFor each of these actions, answer what it does: "
        + ", ".join(sorted(unexplained))
        + '\nReply ONLY with JSON like {"ACTION3": {"type": "translate", "dr": 0, "dc": 1}} '
          'or {"ACTION3": {"type": "unknown"}}.'
    )
    try:
        resp = requests.post(
            llm_cfg["base_url"].rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {llm_cfg.get('api_key') or 'none'}",
                     "Content-Type": "application/json"},
            json={"model": llm_cfg["model"], "temperature": 0.0, "max_tokens": 512,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=LLM_TIMEOUT_S,
        )
        data = resp.json()
        text = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        tokens = int((data.get("usage") or {}).get("completion_tokens") or 0)
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return {}, tokens
        parsed = json.loads(text[start:end + 1])
        return (parsed if isinstance(parsed, dict) else {}), tokens
    except Exception:
        return {}, 0


# ===========================================================================
# the per-game controller
# ===========================================================================
class _LevelState:
    def __init__(self, level):
        self.level = level
        self.phase = "E"
        self.e_turns = 0
        self.probe_queue = []
        self.probes_done = 0
        self.model = WorldModel()
        self.plans_run = 0
        self.novelty_run = 0
        self.breaks = 0
        self.tried_targets = []
        self.visited = set()      # sprite positions already stood on
        self.events_mark = -1     # unexplained+breaks count at last probe re-arm
        self.refused_probes = []  # probe keys that came back as no-ops (locked doors)
        self.pending_probe = None # key of the probe currently executing
        self.active_plan = None   # (desc, remaining_actions) of a truncated plan
        self.fallback = False
        self.fallback_reason = ""
        self.cleared_via = None
        self.rules_report = {}


class TurnResult:
    """Duck-typed stand-in for AnalyzerTurnResult (the solver only reads these
    attributes)."""
    def __init__(self, step_executed, retryable_failure=False, reasoning="",
                 yielded_control=False):
        self.step_executed = step_executed
        self.retryable_failure = retryable_failure
        self.reasoning = reasoning
        self.yielded_control = yielded_control


class ExecWMController:
    def __init__(self, game_id, log, report_cb, llm_cfg=None):
        self.game_id = game_id
        self.log = log
        self.report_cb = report_cb
        self.llm_cfg = llm_cfg
        self.mask = HudMask()
        self.mask_fed = 0
        self.levels: dict[int, _LevelState] = {}
        self.llm_calls = 0
        self.llm_tokens = 0
        self.actions_executed = 0
        self.disabled_reason = None   # game-level triage (e.g. mouse-only)

    # ------------------------------------------------------------------
    def _feed_mask(self, history):
        for i in range(max(1, self.mask_fed), len(history)):
            prev, cur = history[i - 1], history[i]
            if prev["level"] == cur["level"] and action_token(cur["action"]) != "RESET":
                self.mask.observe(prev["grid"], cur["grid"])
        self.mask_fed = len(history)

    def _lvl(self, level) -> _LevelState:
        if level not in self.levels:
            self.levels[level] = _LevelState(level)
            self.log(f"level={level} begin phase=E")
        return self.levels[level]

    def _move_candidates(self, valid_actions):
        va = set()
        for a in (valid_actions or []):
            raw = str(a).strip().upper()
            va.add(MODEL_TO_ENGINE.get(raw, raw))
        return [a for a in MOVE_ACTIONS if a in va]

    # ------------------------------------------------------------------
    def report(self):
        return {
            "version": EXECWM_VERSION,
            "game_id": self.game_id,
            "armed": True,
            "disabled_reason": self.disabled_reason,
            "llm_calls": self.llm_calls,
            "llm_tokens": self.llm_tokens,
            "actions_executed": self.actions_executed,
            "mask_rows": sorted(self.mask.masked_rows()),
            "mask_cols": sorted(self.mask.masked_cols()),
            "levels": {
                str(l.level): {
                    "phase": l.phase,
                    "probes": l.probes_done,
                    "plans_run": l.plans_run,
                    "breaks": l.breaks,
                    "fallback": l.fallback,
                    "fallback_reason": l.fallback_reason,
                    "cleared_via": l.cleared_via,
                    "rules": l.rules_report,
                } for l in self.levels.values()
            },
        }

    # ------------------------------------------------------------------
    def wants_turn(self, current, valid_actions):
        """Does exec-WM claim this turn, or should the stock agent run?"""
        if self.disabled_reason:
            return False
        cands = self._move_candidates(valid_actions)
        if not cands:
            self.disabled_reason = "no-keyboard-actions"
            self.log(f"disabled reason={self.disabled_reason} valid={sorted(valid_actions or [])}")
            return False
        lvl = self._lvl(current["level"])
        return not lvl.fallback

    def _mark_fallback(self, lvl: _LevelState, reason):
        lvl.fallback = True
        lvl.fallback_reason = reason
        lvl.phase = "F"
        self.log(f"level={lvl.level} fallback reason={reason}")

    # ------------------------------------------------------------------
    # a controller turn: execute deterministic work through step_env
    # ------------------------------------------------------------------
    def run_turn(self, state_path, step_env, should_stop, valid_actions):
        current, history = read_state(state_path)
        if current is None or not current["grid"]:
            return TurnResult(step_executed=False)
        self._feed_mask(history)
        lvl = self._lvl(current["level"])
        transitions = [t for t in transitions_from_history(history)
                       if t.level_before == lvl.level]
        cands = self._move_candidates(valid_actions)

        if lvl.phase == "E":
            return self._turn_explore(lvl, cands, state_path, step_env, should_stop)
        if lvl.phase == "P":
            return self._turn_plan(lvl, current, transitions, state_path, step_env, should_stop)
        return TurnResult(step_executed=False)

    # ---- PHASE E ----
    def _turn_explore(self, lvl, cands, state_path, step_env, should_stop):
        if not lvl.probe_queue:
            need = []
            for rep in range(E_REPEATS):
                for a in cands:
                    need.append(a)
            lvl.probe_queue = need[lvl.probes_done:]
        lvl.e_turns += 1
        executed = 0
        while lvl.probe_queue and executed < PROBE_BATCH:
            if should_stop and should_stop():
                break
            action = lvl.probe_queue.pop(0)
            payload = step_env({"action": action})
            executed += 1
            lvl.probes_done += 1
            self.actions_executed += 1
            if not payload.get("executed"):
                continue
            if payload.get("level_completed") or payload.get("run_complete"):
                lvl.cleared_via = "explore"
                self.log(f"level={lvl.level} CLEARED via=explore probes={lvl.probes_done}")
                self._flush(state_path)
                return TurnResult(step_executed=True)
            if payload.get("game_over"):
                self.log(f"level={lvl.level} game_over during explore (auto-reset)")
                self._flush(state_path)
                return TurnResult(step_executed=True)
        self.log(f"level={lvl.level} explore turn={lvl.e_turns} probes={lvl.probes_done} "
                 f"mask_pairs={self.mask.pairs}")
        if not lvl.probe_queue:
            self._induce(lvl, state_path)
        elif lvl.e_turns >= E_MAX_TURNS:
            self._mark_fallback(lvl, "explore-budget-exhausted")
        self._flush(state_path)
        return TurnResult(step_executed=executed > 0)

    # ---- PHASE I + V ----
    def _induce(self, lvl, state_path):
        current, history = read_state(state_path)
        transitions = [t for t in transitions_from_history(history)
                       if t.level_before == lvl.level]
        lvl.model.mine(transitions, self.mask)
        lvl.rules_report = lvl.model.verify(transitions, self.mask)
        mined = sorted(lvl.model.rules)
        unexplained = [a for a in self._probe_actions(transitions) if a not in lvl.model.rules]
        self.log(f"level={lvl.level} induce mined={mined} unexplained={unexplained} "
                 f"rules={json.dumps(lvl.rules_report, sort_keys=True)}")
        if unexplained and self.llm_cfg and self.llm_calls < LLM_CALLS_PER_GAME:
            evidence = self._evidence_lines(transitions)
            hints, tokens = llm_fill(unexplained, evidence, self.llm_cfg)
            self.llm_calls += 1
            self.llm_tokens += tokens
            if hints:
                lvl.model.mine(transitions, self.mask, llm_hints=hints)
                lvl.rules_report = lvl.model.verify(transitions, self.mask)
                self.log(f"level={lvl.level} induce llm_hints={sorted(hints)} "
                         f"rules={json.dumps(lvl.rules_report, sort_keys=True)}")
        n_moves = len(lvl.model.verified_moves())
        if n_moves >= MIN_VERIFIED_MOVES and lvl.model.sprite is not None:
            lvl.phase = "P"
            self.log(f"level={lvl.level} VERIFIED moves={n_moves} -> plan "
                     f"permeable={sorted(lvl.model.permeable)} "
                     f"blockers={sorted(lvl.model.blockers)}")
        else:
            self._mark_fallback(lvl, f"no-verified-model(moves={n_moves})")

    def _probe_actions(self, transitions):
        return sorted({t.action for t in transitions if t.action in MOVE_ACTIONS})

    def _evidence_lines(self, transitions):
        lines = []
        for t in transitions[-24:]:
            diff = interior_diff(t.before, t.after, self.mask)
            if not diff:
                lines.append(f"{t.action}: no visible change")
            else:
                r0 = min(r for r, _ in diff); r1 = max(r for r, _ in diff)
                c0 = min(c for _, c in diff); c1 = max(c for _, c in diff)
                lines.append(f"{t.action}: {len(diff)} cells changed in rows {r0}-{r1} cols {c0}-{c1}")
        return lines

    # ---- PHASE P ----
    def _turn_plan(self, lvl, current, transitions, state_path, step_env, should_stop):
        grid = current["grid"]
        model = lvl.model
        hits = model.sprite.find(grid) if model.sprite else []
        if len(hits) != 1:
            lvl.breaks += 1
            self.log(f"level={lvl.level} sprite-lost hits={len(hits)} breaks={lvl.breaks}")
            if lvl.breaks >= MAX_BREAKS_PER_LEVEL:
                self._mark_fallback(lvl, "sprite-lost")
            else:
                lvl.phase = "E"
                lvl.probe_queue = list(self._move_candidates(MOVE_ACTIONS))[:2]
            self._flush(state_path)
            return TurnResult(step_executed=False)
        pos = hits[0]
        lvl.visited.add(pos)
        if lvl.active_plan is not None:
            target_desc, actions, probe_last = lvl.active_plan
            lvl.active_plan = None
        else:
            parents = bfs_reachable(model, grid, pos)
            plan = self._next_plan(lvl, grid, pos, parents)
            if plan is None:
                probes = frontier_probes(model, grid, parents)
                self.log(f"level={lvl.level} exhausted-diag parents={len(parents)} "
                         f"probes={len(probes)} tried={len(lvl.tried_targets)} "
                         f"moves={sorted(model.verified_moves())} "
                         f"first_probes={[(p[1], p[2], p[3]) for p in probes[:4]]}")
                self._mark_fallback(lvl, "plan-targets-exhausted")
                self._flush(state_path)
                return TurnResult(step_executed=False)
            target_desc, actions, probe_last = plan
            lvl.plans_run += 1
            self.log(f"level={lvl.level} plan#{lvl.plans_run} target={target_desc} "
                     f"len={len(actions)}")
        executed = 0
        clean = True
        for step_i, action in enumerate(actions[:PLAN_BATCH]):
            is_probe_step = probe_last and step_i == len(actions) - 1
            if should_stop and should_stop():
                break
            cur_state, _ = read_state(state_path)
            before = cur_state["grid"]
            bhits = model.sprite.find(before)
            bpos = bhits[0] if len(bhits) == 1 else None
            payload = step_env({"action": action})
            executed += 1
            self.actions_executed += 1
            if not payload.get("executed"):
                lvl.breaks += 1
                clean = False
                self.log(f"level={lvl.level} plan-step-refused action={action} breaks={lvl.breaks}")
                break
            if payload.get("level_completed") or payload.get("run_complete"):
                lvl.cleared_via = "plan"
                self.log(f"level={lvl.level} CLEARED via=plan plan#{lvl.plans_run} "
                         f"actions={self.actions_executed}")
                self._flush(state_path)
                return TurnResult(step_executed=True)
            if payload.get("game_over"):
                self.log(f"level={lvl.level} game_over during plan (auto-reset)")
                self._flush(state_path)
                return TurnResult(step_executed=True)
            after_state, _ = read_state(state_path)
            after = after_state["grid"]
            if bpos is not None and not is_probe_step:
                res = model.check_prediction(before, after, bpos, action, self.mask)
                if res is not None and res[0]:
                    lvl.visited.add(res[2])
                if res is not None and not res[0]:
                    clean = False
                    kind, _ = detect_translation(before, after, self.mask)
                    if kind == "unexplained":
                        # a world event, not a model error: no break charged;
                        # it feeds the event counter that re-arms probes.
                        self.log(f"level={lvl.level} EVENT action={action} "
                                 f"step={executed} (plan aborted, no break)")
                    else:
                        lvl.breaks += 1
                        self.log(f"level={lvl.level} BREAK action={action} "
                                 f"step={executed} breaks={lvl.breaks}")
                    # counterexample feeds the next mine cycle
                    cur, history = read_state(state_path)
                    self._feed_mask(history)
                    transitions = [t for t in transitions_from_history(history)
                                   if t.level_before == lvl.level]
                    lvl.model.mine(transitions, self.mask)
                    lvl.rules_report = lvl.model.verify(transitions, self.mask)
                    if lvl.breaks >= MAX_BREAKS_PER_LEVEL or \
                            len(lvl.model.verified_moves()) < MIN_VERIFIED_MOVES:
                        self._mark_fallback(lvl, "prediction-breaks")
                    self._flush(state_path)
                    return TurnResult(step_executed=True)
        if clean and len(actions) > executed and executed >= PLAN_BATCH:
            lvl.active_plan = (target_desc, actions[executed:], probe_last)
        elif clean and probe_last and executed == len(actions):
            # probe executed: absorb its outcome into the model right away so
            # the next BFS sees the new permeable color or the new blocker.
            if lvl.pending_probe is not None and not payload.get("board_changed"):
                if lvl.pending_probe not in lvl.refused_probes:
                    lvl.refused_probes.append(lvl.pending_probe)
            lvl.pending_probe = None
            cur, history = read_state(state_path)
            self._feed_mask(history)
            transitions = [t for t in transitions_from_history(history)
                           if t.level_before == lvl.level]
            lvl.model.mine(transitions, self.mask)
            lvl.rules_report = lvl.model.verify(transitions, self.mask)
            events = lvl.model.unexplained + lvl.breaks
            if events != lvl.events_mark:
                # a world event just happened: re-arm probes NOW (not at
                # exhaustion) so the refused-first retry lands inside the
                # same game round as the event that may have unlocked it.
                lvl.events_mark = events
                lvl.tried_targets = [k for k in lvl.tried_targets
                                     if k[0] != "probe"]
                self.log(f"level={lvl.level} probes-rearmed-now events={events}")
        if lvl.plans_run >= MAX_PLANS_PER_LEVEL:
            self._mark_fallback(lvl, "plan-budget-exhausted")
        self._flush(state_path)
        return TurnResult(step_executed=executed > 0)

    def _next_plan(self, lvl, grid, pos, parents):
        """Pick the next untried goal; then coverage sweep; then a frontier
        probe. Returns (desc, actions, probe_last) or None."""
        model = lvl.model
        for color, cells in goal_targets(model, grid, self.mask, pos):
            touching = positions_touching(model, cells, parents)
            for goal in touching:
                key = ("color", color, goal)
                if key in lvl.tried_targets:
                    continue
                actions = path_to(parents, goal)
                if actions:
                    lvl.tried_targets.append(key)
                    return f"color={color}@{goal}", actions, False
        # coverage sweep: visit the NEAREST reachable position never stood on.
        # BFS insertion order == distance order, so the first unvisited entry
        # is the nearest; repeating this tours the whole reachable set. With
        # eps=0.17 actions are nearly free, so the tour costs ~nothing and
        # guarantees the sprite touches every reachable cell of the level.
        if lvl.novelty_run < MAX_SWEEP_PLANS:
            for goal in parents:
                if goal == pos or goal in lvl.visited:
                    continue
                actions = path_to(parents, goal)
                if actions:
                    lvl.novelty_run += 1
                    return f"sweep@{goal}", actions, False
        # frontier probe: one cheap experiment past the verified boundary.
        # Previously REFUSED probes come first: after a world event they are
        # the locked doors most likely to have just opened, and retrying them
        # immediately beats re-touring unprobed cells (the ls20 key/door pair
        # is lost to the round timer otherwise).
        cands = frontier_probes(model, grid, parents)
        refused = [c for c in cands if ("probe", c[1], c[2]) in lvl.refused_probes]
        fresh = [c for c in cands if ("probe", c[1], c[2]) not in lvl.refused_probes]
        for _, ppos, action, colors in refused + fresh:
            key = ("probe", ppos, action)
            if key in lvl.tried_targets:
                continue
            base = path_to(parents, ppos)
            if base is None:
                continue
            lvl.tried_targets.append(key)
            lvl.pending_probe = key
            return f"probe{list(colors)}@{ppos}+{action}", base + [action], True
        # The world is STATEFUL: an unexplained event (a pickup, a toggle, a
        # break) can change what a refused move now does -- the ls20 key/door
        # is exactly this shape. Each NEW unexplained event re-arms every
        # probe once; with no new events, exhaustion is final.
        events = lvl.model.unexplained + lvl.breaks
        if events != lvl.events_mark:
            lvl.events_mark = events
            before = len(lvl.tried_targets)
            lvl.tried_targets = [k for k in lvl.tried_targets if k[0] != "probe"]
            self.log(f"level={lvl.level} probes-rearmed events={events} "
                     f"(cleared {before - len(lvl.tried_targets)})")
            return self._next_plan(lvl, grid, pos, parents)
        return None

    def _flush(self, state_path):
        try:
            self.report_cb(self.report())
        except Exception:
            pass


# ===========================================================================
# the analyzer wrapper (what the patched solver constructs)
# ===========================================================================
class ExecWMAnalyzer:
    """Wraps the stock ToolAgent. Deterministic exec-WM turns run with ZERO
    LLM round-trips; fallback turns delegate to the stock agent untouched."""

    def __init__(self, inner, game=None, index=0, solver=None):
        self.inner = inner
        self.game = game
        self.index = index
        self.solver = solver
        game_id = getattr(getattr(game, "game_run", None), "game_id", None) \
            or getattr(game, "game_id", None) or getattr(game, "env_name", None) \
            or f"game{index}"
        self._log_lock = threading.Lock()
        self._transcript_path = None
        llm_cfg = self._llm_config(inner)
        self.controller = ExecWMController(
            str(game_id), self._log, self._write_report, llm_cfg=llm_cfg)
        self._log(f"armed {EXECWM_VERSION} game={game_id} "
                  f"llm={'on' if llm_cfg else 'off'}")

    # ---- token accounting: the solver reads .generated_tokens ----
    @property
    def generated_tokens(self):
        inner_tokens = getattr(self.inner, "generated_tokens", None)
        if inner_tokens is None:
            inner_tokens = getattr(self.inner, "total_tokens", 0)
        return int(inner_tokens or 0) + int(self.controller.llm_tokens)

    def _llm_config(self, inner):
        if int(os.environ.get("ARC3_EXECWM_LLM", "1") or "1") == 0:
            return None
        model = getattr(inner, "_model", None)
        base_url = getattr(model, "base_url", None)
        model_id = getattr(model, "model_id", None)
        if not base_url or not model_id:
            return None
        return {"base_url": str(base_url), "model": str(model_id),
                "api_key": getattr(inner, "_api_key", "") or ""}

    def _log(self, msg):
        line = f"[execwm] {msg}"
        with self._log_lock:
            print(line, flush=True)
            tp = self._transcript_path
            if tp is not None:
                try:
                    with open(tp, "a", encoding="utf-8") as f:
                        f.write(line + "\n")
                except Exception:
                    pass

    def _write_report(self, report):
        job_dir = getattr(self.solver, "job_dir", None)
        if not job_dir:
            return
        try:
            from pathlib import Path
            out_dir = Path(job_dir) / "execwm"
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = "".join(ch if ch.isalnum() or ch in "-_" else "_"
                           for ch in report["game_id"])
            out = out_dir / f"{stem}_p{self.index}.json"
            tmp = out.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(report, indent=1), encoding="utf-8")
            tmp.replace(out)
        except Exception:
            pass

    # ---- the analyzer protocol ----
    def analyze(self, state_path, action_num, valid_actions=None, step_env=None,
                transcript_path=None, analysis_step=None, transcript_updated=None,
                request_timeout_seconds=None, should_stop=None, **kwargs):
        self._transcript_path = transcript_path
        current, _ = read_state(state_path) if state_path.exists() else (None, [])
        use_wm = (current is not None and current.get("grid")
                  and step_env is not None
                  and self.controller.wants_turn(current, valid_actions))
        if use_wm:
            try:
                result = self.controller.run_turn(
                    state_path, step_env, should_stop, valid_actions)
                if result.step_executed:
                    return result
                # controller did no work this turn (e.g. just fell back):
                # fall through to the stock agent in the same turn.
                current, _ = read_state(state_path)
                lvl = self.controller.levels.get(current["level"]) if current else None
                if lvl is not None and not lvl.fallback:
                    return result
            except Exception as exc:  # never let exec-WM kill the run
                self._log(f"controller-error {type(exc).__name__}: {exc} -> fallback")
                try:
                    lvl = self.controller._lvl(current["level"])
                    self.controller._mark_fallback(lvl, f"controller-error:{type(exc).__name__}")
                except Exception:
                    self.controller.disabled_reason = "controller-error"
        return self.inner.analyze(
            state_path, action_num, valid_actions=valid_actions, step_env=step_env,
            transcript_path=transcript_path, analysis_step=analysis_step,
            transcript_updated=transcript_updated,
            request_timeout_seconds=request_timeout_seconds,
            should_stop=should_stop, **kwargs)


def maybe_wrap_analyzer(inner, game=None, index=0, solver=None):
    """Called from the patched HarnessSolver._make_analyzer. Honors the arm
    kill-switch: ARC3_EXECWM=0 leaves the stock analyzer untouched."""
    if int(os.environ.get("ARC3_EXECWM", "1") or "1") == 0:
        return inner
    return ExecWMAnalyzer(inner, game=game, index=index, solver=solver)
