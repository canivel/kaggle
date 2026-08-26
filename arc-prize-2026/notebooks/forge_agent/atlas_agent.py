# =====================================================================
# ATLAS — Adaptive Tactical Learning Agent for Strategy
#
# Novel approach: instead of brute-force BFS or blind CNN exploration,
# DISCOVER game rules through structured hypothesis testing, then
# PLAN efficient action sequences using the learned model.
#
# Architecture:
#   1. BFS Solver (when game source available — keeps proven L0 solving)
#   2. Object Tracker (connected components, persistent object IDs)
#   3. Action-Effect Model (what each action does to each object)
#   4. Game Classifier (click-toggle? movement? sequence? puzzle?)
#   5. Strategy Planner (uses learned model for efficient action sequences)
#   6. Fallback Explorer (state graph + BFS-to-frontier when model fails)
#
# Key insight: RHAE squares inefficiency. 10x more actions = 1% score.
# The ONLY way to score well is to understand the game and act efficiently.
# =====================================================================
import copy
import glob
import hashlib
import heapq
import importlib.util
import logging
import os
import random
import re
import time
import traceback
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState, ActionInput

logger = logging.getLogger(__name__)

# ==================== OBJECT TRACKER ====================

class TrackedObject:
    """A persistent object tracked across frames."""
    __slots__ = ('oid', 'color', 'pixels', 'cx', 'cy', 'size', 'bbox', 'shape_hash')

    def __init__(self, oid, color, pixels):
        self.oid = oid
        self.color = color
        self.pixels = pixels  # set of (y, x) tuples
        self.size = len(pixels)
        ys = [p[0] for p in pixels]
        xs = [p[1] for p in pixels]
        self.cy = sum(ys) / self.size
        self.cx = sum(xs) / self.size
        self.bbox = (min(ys), min(xs), max(ys), max(xs))
        # Shape hash: normalize position and hash the relative pixel positions
        min_y, min_x = min(ys), min(xs)
        relative = frozenset((y - min_y, x - min_x) for y, x in pixels)
        self.shape_hash = hash(relative)


def extract_objects(frame, bg_color=0):
    """Extract connected components as TrackedObjects using flood fill."""
    h, w = frame.shape
    visited = np.zeros((h, w), dtype=bool)
    objects = []
    oid = 0

    for y in range(h):
        for x in range(w):
            if visited[y, x] or frame[y, x] == bg_color:
                continue
            # Flood fill
            color = frame[y, x]
            pixels = set()
            stack = [(y, x)]
            while stack:
                cy, cx = stack.pop()
                if cy < 0 or cy >= h or cx < 0 or cx >= w:
                    continue
                if visited[cy, cx] or frame[cy, cx] != color:
                    continue
                visited[cy, cx] = True
                pixels.add((cy, cx))
                stack.extend([(cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)])
            if len(pixels) >= 2:  # ignore single pixels
                objects.append(TrackedObject(oid, int(color), pixels))
                oid += 1

    return objects


def match_objects(prev_objects, curr_objects):
    """Match objects between frames by color + position proximity."""
    matched = {}  # curr_oid -> prev_oid
    used_prev = set()

    for co in curr_objects:
        best_prev = None
        best_dist = float('inf')
        for po in prev_objects:
            if po.oid in used_prev:
                continue
            if po.color == co.color and abs(po.size - co.size) < max(po.size, co.size) * 0.5:
                dist = abs(co.cx - po.cx) + abs(co.cy - po.cy)
                if dist < best_dist:
                    best_dist = dist
                    best_prev = po
        if best_prev and best_dist < 20:
            matched[co.oid] = best_prev.oid
            used_prev.add(best_prev.oid)

    return matched


# ==================== ACTION-EFFECT MODEL ====================

class ActionEffect:
    """Records what an action did."""
    __slots__ = ('action_val', 'data', 'frame_changed', 'objects_moved',
                 'objects_color_changed', 'objects_appeared', 'objects_disappeared',
                 'pixel_diff_count')

    def __init__(self, action_val, data=None):
        self.action_val = action_val
        self.data = data
        self.frame_changed = False
        self.objects_moved = []       # list of (oid, dx, dy)
        self.objects_color_changed = []  # list of (oid, old_color, new_color)
        self.objects_appeared = []    # list of oid
        self.objects_disappeared = [] # list of oid
        self.pixel_diff_count = 0


class ActionEffectModel:
    """Learns what each action type does from observed effects."""

    def __init__(self):
        self.effects = []  # list of ActionEffect
        self.action_type_effects = defaultdict(list)  # action_val -> [ActionEffect]
        # Aggregated knowledge
        self.movement_actions = set()     # action_vals that move objects
        self.toggle_actions = set()       # action_vals that change colors
        self.productive_actions = set()   # action_vals that ever changed frame
        self.click_effects = {}           # (approx_y, approx_x) -> ActionEffect
        self.n_observations = 0

    def observe(self, effect: ActionEffect):
        self.effects.append(effect)
        self.action_type_effects[effect.action_val].append(effect)
        self.n_observations += 1

        if effect.frame_changed:
            self.productive_actions.add(effect.action_val)
        if effect.objects_moved:
            self.movement_actions.add(effect.action_val)
        if effect.objects_color_changed:
            self.toggle_actions.add(effect.action_val)
        if effect.action_val == 6 and effect.data:
            key = (effect.data.get('y', 0) // 4, effect.data.get('x', 0) // 4)
            self.click_effects[key] = effect

    def classify_game(self):
        """Classify game type based on observed effects."""
        has_movement = len(self.movement_actions) > 0
        has_toggle = len(self.toggle_actions) > 0
        has_click_effects = any(e.action_val == 6 and e.frame_changed for e in self.effects)
        has_keyboard_effects = any(e.action_val <= 5 and e.frame_changed for e in self.effects)

        if has_keyboard_effects and not has_click_effects:
            return 'MOVEMENT'
        elif has_click_effects and not has_keyboard_effects:
            if has_toggle:
                return 'CLICK_TOGGLE'
            else:
                return 'CLICK_SEQUENCE'
        elif has_click_effects and has_keyboard_effects:
            return 'MIXED'
        else:
            return 'UNKNOWN'

    def suggest_action(self, frame, objects, available_actions, state_hash, tested_here):
        """Suggest the most promising action based on learned model."""
        game_type = self.classify_game()

        if game_type == 'CLICK_TOGGLE' or game_type == 'CLICK_SEQUENCE':
            return self._suggest_click(frame, objects, available_actions, tested_here)
        elif game_type == 'MOVEMENT':
            return self._suggest_movement(available_actions, tested_here)
        else:
            return self._suggest_explore(frame, objects, available_actions, tested_here)

    def _suggest_click(self, frame, objects, available_actions, tested_here):
        """For click games: click untested objects, prioritize by size."""
        if 6 not in available_actions:
            return None
        # Sort objects by size (small first — likely buttons)
        sorted_objs = sorted(objects, key=lambda o: o.size)
        for obj in sorted_objs:
            x, y = int(obj.cx), int(obj.cy)
            # Check if we've clicked near this object
            key = (y // 4, x // 4)
            if key not in self.click_effects:
                return 6, {'x': x, 'y': y}
        # All objects tested — try untested coordinates
        for obj in sorted_objs:
            for py, px in list(obj.pixels)[:3]:  # try a few pixels per object
                key = (py // 4, px // 4)
                if key not in self.click_effects:
                    return 6, {'x': int(px), 'y': int(py)}
        return None

    def _suggest_movement(self, available_actions, tested_here):
        """For movement games: try productive directional actions first."""
        # Prefer actions we KNOW move objects
        for a in self.movement_actions:
            if a in available_actions and a not in tested_here:
                return a, None
        # Then try any untested keyboard action
        for a in range(1, 6):
            if a in available_actions and a not in tested_here:
                return a, None
        return None

    def _suggest_explore(self, frame, objects, available_actions, tested_here):
        """General exploration: try untested actions, prefer productive ones."""
        # Untested keyboard actions first
        for a in range(1, 6):
            if a in available_actions and a not in tested_here:
                if a in self.productive_actions:
                    return a, None
        for a in range(1, 6):
            if a in available_actions and a not in tested_here:
                return a, None
        # Then clicks on untested objects
        if 6 in available_actions and objects:
            return self._suggest_click(frame, objects, available_actions, tested_here)
        return None

    def reset(self):
        self.__init__()


# ==================== STATE GRAPH ====================

class StateNode:
    __slots__ = ('hash', 'objects', 'tested_actions', 'transitions',
                 'visit_count', 'novelty_score', 'frame')

    def __init__(self, hash_val, frame=None):
        self.hash = hash_val
        self.objects = []
        self.tested_actions = set()
        self.transitions = {}  # action_key -> next_hash
        self.visit_count = 0
        self.novelty_score = 1.0
        self.frame = frame


class StateGraph:
    """Directed graph of game states with BFS-to-frontier navigation."""

    def __init__(self):
        self.nodes = {}  # hash -> StateNode
        self.current_hash = None

    def observe(self, state_hash, frame, objects):
        if state_hash not in self.nodes:
            self.nodes[state_hash] = StateNode(state_hash, frame)
        node = self.nodes[state_hash]
        node.objects = objects
        node.visit_count += 1
        node.novelty_score = max(0.01, 1.0 / node.visit_count)
        self.current_hash = state_hash

    def record_transition(self, from_hash, action_key, to_hash, changed):
        if from_hash in self.nodes:
            self.nodes[from_hash].tested_actions.add(action_key)
            if changed:
                self.nodes[from_hash].transitions[action_key] = to_hash

    def untested_count(self, state_hash, available_actions):
        """How many available actions haven't been tested from this state."""
        node = self.nodes.get(state_hash)
        if not node:
            return len(available_actions)
        return len([a for a in available_actions if a not in node.tested_actions])

    def bfs_to_frontier(self, start_hash, available_actions, max_depth=50):
        """Find shortest path to a state with untested actions."""
        if self.untested_count(start_hash, available_actions) > 0:
            return []  # already at frontier

        visited = {start_hash}
        queue = deque([(start_hash, [])])
        while queue:
            h, path = queue.popleft()
            if len(path) > max_depth:
                continue
            node = self.nodes.get(h)
            if not node:
                continue
            for action_key, next_h in node.transitions.items():
                if next_h in visited:
                    continue
                new_path = path + [action_key]
                if self.untested_count(next_h, available_actions) > 0:
                    return new_path
                visited.add(next_h)
                queue.append((next_h, new_path))
        return None  # no frontier reachable

    def most_novel_state(self):
        """Find the state with highest novelty that's been visited."""
        best = None
        best_score = -1
        for h, node in self.nodes.items():
            if node.novelty_score > best_score:
                best_score = node.novelty_score
                best = h
        return best

    def reset(self):
        self.__init__()


# ==================== BFS SOLVER (from FORGE v19) ====================

class BFSSolver:
    """Offline BFS solver using direct game class instantiation."""

    def __init__(self, game_path, game_class_name, scan_timeout=5, bfs_timeout=180):
        self.game_path = game_path
        self.class_name = game_class_name
        self.scan_timeout = scan_timeout
        self.bfs_timeout = bfs_timeout
        self.game_cls = None
        self.solutions = {}

    def load(self):
        try:
            spec = importlib.util.spec_from_file_location('game_mod', self.game_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.game_cls = getattr(mod, self.class_name)
            return True
        except Exception as e:
            logger.warning(f"BFS load failed: {e}")
            return False

    def _state_hash(self, g, frame, hidden_fields=None):
        fh = hashlib.md5(frame.tobytes()).hexdigest()[:16]
        if hidden_fields:
            extras = []
            for fn in hidden_fields:
                try:
                    v = getattr(g, fn, None)
                    if v is not None:
                        extras.append(f"{fn}={v}")
                except: pass
            if extras:
                return fh + "|" + "|".join(extras)
        return fh

    def _probe_hidden_fields(self, game, actions):
        if not actions:
            return []
        initial = {k: v for k, v in game.__dict__.items()
                   if isinstance(v, (int, float, bool)) and not k.startswith('__')}
        changing = set()
        for act_id, data in actions[:10]:
            g = copy.deepcopy(game)
            try:
                ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                g.perform_action(ai, raw=True)
            except: continue
            for k, v in g.__dict__.items():
                if isinstance(v, (int, float, bool)) and not k.startswith('__'):
                    if k in initial and v != initial[k]:
                        if k not in ('_action_count', '_full_reset', '_action_complete'):
                            changing.add(k)
        return sorted([f for f in changing
                       if not f.startswith('_') or f in ('_current_level_index', '_score')])

    def _scan_actions(self, game, f0, bg):
        avail = game._available_actions
        actions = []
        for a in [a for a in avail if a <= 5]:
            g = copy.deepcopy(game)
            try:
                r = g.perform_action(ActionInput(id=GameAction.from_id(a)), raw=True)
                if r.frame and np.sum(f0 != np.array(r.frame[-1])) > 0:
                    actions.append((a, None))
            except: pass
        # Click scan WITHOUT dedup (v15 fix: dedup kills cd82/sp80 L1)
        if 6 in avail:
            t0 = time.time()
            for y in range(0, 64, 2):
                if time.time() - t0 > self.scan_timeout:
                    break
                for x in range(0, 64, 2):
                    if f0[y, x] == bg:
                        continue
                    g = copy.deepcopy(game)
                    try:
                        r = g.perform_action(
                            ActionInput(id=GameAction.ACTION6,
                                        data={'x': x, 'y': y, 'game_id': 'bfs'}),
                            raw=True)
                        if r.frame and np.sum(f0 != np.array(r.frame[-1])) > 0:
                            actions.append((6, {'x': x, 'y': y, 'game_id': 'bfs'}))
                    except: pass
        return actions

    def solve_level(self, level_idx, max_states=500000, prev_solution=None):
        if not self.game_cls:
            return None
        deadline = time.time() + self.bfs_timeout

        game = self.game_cls()
        game.set_level(level_idx)
        game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        r0 = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        if not r0.frame:
            return None
        f0 = np.array(r0.frame[-1])
        bg = int(np.bincount(f0.flatten(), minlength=16).argmax())

        # Transfer from prev level
        if prev_solution and level_idx > 0:
            g = copy.deepcopy(game)
            for i, (act_id, data) in enumerate(prev_solution):
                try:
                    ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                    r = g.perform_action(ai, raw=True)
                    if r.levels_completed > level_idx or g._current_level_index > level_idx:
                        sol = prev_solution[:i+1]
                        self.solutions[level_idx] = sol
                        return sol
                except: break

        actions = self._scan_actions(game, f0, bg)
        if not actions:
            # Warmup unlock
            for wid in [a for a in game._available_actions if a <= 4]:
                gw = copy.deepcopy(game)
                try:
                    gw.perform_action(ActionInput(id=GameAction.from_id(wid)), raw=True)
                    fa = np.array(gw.get_pixels(0, 0, 64, 64))
                    wa = self._scan_actions(gw, fa, bg)
                    if wa:
                        game = gw; f0 = fa; actions = wa; break
                except: pass

        if not actions:
            return None

        # Adaptive depth
        max_depth = 50 if len(actions) <= 4 else (40 if len(actions) <= 8 else 30)

        # BFS
        visited = set()
        queue = deque()
        h0 = self._state_hash(game, f0)
        visited.add(h0)
        queue.append((copy.deepcopy(game), [], 0))
        explored = 0

        while queue and explored < max_states and time.time() < deadline:
            g, hist, depth = queue.popleft()
            for act_id, data in actions:
                g2 = copy.deepcopy(g)
                try:
                    ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                    r = g2.perform_action(ai, raw=True)
                except: continue
                explored += 1
                if not r.frame: continue
                f = np.array(r.frame[-1])
                h = self._state_hash(g2, f)
                if h in visited: continue
                visited.add(h)
                new_hist = hist + [(act_id, data)]
                if r.levels_completed > level_idx or g2._current_level_index > level_idx:
                    self.solutions[level_idx] = new_hist
                    return new_hist
                if depth < max_depth:
                    queue.append((g2, new_hist, depth + 1))

        # Hidden state retry
        if len(visited) < 50 and time.time() < deadline - 30:
            hidden = self._probe_hidden_fields(game, actions)
            if hidden:
                game2 = self.game_cls()
                game2.set_level(level_idx)
                game2.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                game2.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                f0_2 = np.array(game2.perform_action(ActionInput(id=GameAction.RESET), raw=True).frame[-1])
                v2 = {self._state_hash(game2, f0_2, hidden)}
                q2 = deque([(copy.deepcopy(game2), [], 0)])
                while q2 and time.time() < deadline:
                    g, hist, depth = q2.popleft()
                    for act_id, data in actions:
                        g2 = copy.deepcopy(g)
                        try:
                            ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                            r = g2.perform_action(ai, raw=True)
                        except: continue
                        if not r.frame: continue
                        f = np.array(r.frame[-1])
                        h = self._state_hash(g2, f, hidden)
                        if h in v2: continue
                        v2.add(h)
                        if r.levels_completed > level_idx or g2._current_level_index > level_idx:
                            self.solutions[level_idx] = hist + [(act_id, data)]
                            return self.solutions[level_idx]
                        if depth < max_depth:
                            q2.append((g2, hist + [(act_id, data)], depth + 1))

        # IDDFS for deep directional games
        if len(actions) <= 6 and time.time() < deadline - 30:
            game3 = self.game_cls()
            game3.set_level(level_idx)
            game3.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            game3.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            for max_d in range(10, 60):
                if time.time() >= deadline:
                    break
                stack = [(copy.deepcopy(game3), [], set())]
                while stack and time.time() < deadline:
                    g, hist, path_h = stack.pop()
                    if len(hist) >= max_d:
                        continue
                    for act_id, data in actions:
                        g2 = copy.deepcopy(g)
                        try:
                            ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                            r = g2.perform_action(ai, raw=True)
                        except: continue
                        if not r.frame: continue
                        f = np.array(r.frame[-1])
                        h = hashlib.md5(f.tobytes()).hexdigest()[:16]
                        if h in path_h: continue
                        new_hist = hist + [(act_id, data)]
                        if r.levels_completed > level_idx or g2._current_level_index > level_idx:
                            self.solutions[level_idx] = new_hist
                            return new_hist
                        stack.append((g2, new_hist, path_h | {h}))
        return None


def find_game_source(game_id, arc_env=None):
    gid = game_id.split('-')[0]
    cls_name = gid[0].upper() + gid[1:]
    src = None
    if arc_env and hasattr(arc_env, 'environment_info'):
        ei = arc_env.environment_info
        if hasattr(ei, 'local_dir') and ei.local_dir:
            from pathlib import Path
            ld = Path(ei.local_dir)
            for cand in [ld / f"{gid}.py", ld / f"{gid.lower()}.py"]:
                if cand.exists():
                    src = str(cand)
                    content = cand.read_text()[:2000]
                    m = re.search(r'class\s+(\w+)\s*\(\s*ARCBaseGame', content)
                    if m: cls_name = m.group(1)
                    break
    if not src:
        for pat in [f"environment_files/{gid}/**/{gid}.py",
                    f"/kaggle/working/environment_files/{gid}/**/{gid}.py",
                    f"/tmp/*/{gid}/*/{gid}.py",
                    f"/kaggle/input/**/{gid}*/**/{gid}.py"]:
            matches = glob.glob(pat, recursive=True)
            if matches:
                src = matches[0]
                content = open(src).read()[:2000]
                m = re.search(r'class\s+(\w+)\s*\(\s*ARCBaseGame', content)
                if m: cls_name = m.group(1)
                break
    return src, cls_name


# ==================== ATLAS AGENT ====================

ACTION_LIST = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
               GameAction.ACTION4, GameAction.ACTION5]
GA_MAP = {a.value: a for a in GameAction}


class MyAgent(Agent):
    MAX_ACTIONS = float('inf')
    _MAX_FRAMES = 10

    def __init__(s, *a, **kw):
        super().__init__(*a, **kw)
        seed = int(time.time()*1e6) + hash(s.game_id) % 1000000
        random.seed(seed); np.random.seed(seed%(2**32-1))
        s.start_time = time.time()
        s.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # BFS solver
        s._bfs = None; s._bfs_solution = None; s._bfs_step = 0; s._bfs_tried = False

        # ATLAS components
        s._model = ActionEffectModel()
        s._graph = StateGraph()
        s._prev_frame = None; s._prev_hash = None; s._prev_action = None
        s._prev_data = None; s._prev_objects = []
        s._bg = 0
        s._cl = -1  # current level
        s._phase = 'PROBE'  # PROBE -> MODEL -> PLAN -> EXPLORE
        s._probe_step = 0
        s._probe_actions_tried = set()
        s._planned_path = []

    def append_frame(s, f):
        s.frames.append(f)
        if len(s.frames) > s._MAX_FRAMES: s.frames = s.frames[-s._MAX_FRAMES:]
        if f.guid: s.guid = f.guid
        if hasattr(s, "recorder") and not s.is_playback:
            import json; s.recorder.record(json.loads(f.model_dump_json()))

    def _lvl(s, f): return getattr(f, 'score', None) or f.levels_completed
    def _raw(s, fd): return np.array(fd.frame, dtype=np.int64)[-1]

    def _init_bfs(s):
        src, cls = find_game_source(s.game_id, s.arc_env)
        if src:
            s._bfs = BFSSolver(src, cls, scan_timeout=5, bfs_timeout=180)
            if s._bfs.load():
                logger.info(f"BFS: loaded {cls} from {src}")
            else:
                s._bfs = None
        else:
            logger.warning(f"BFS: no source for {s.game_id}")

    def _try_bfs(s, level_idx):
        if not s._bfs: return None
        elapsed = time.time() - s.start_time
        remaining = max(60, 8*3600 - 600 - elapsed)
        budget = min(remaining * 0.3, 600) if level_idx == 0 else min(remaining * 0.1, 300)
        s._bfs.bfs_timeout = max(30, budget)
        prev_sol = s._bfs.solutions.get(level_idx - 1) if level_idx > 0 else None
        sol = s._bfs.solve_level(level_idx, prev_solution=prev_sol)
        if sol: s._bfs_solution = sol; s._bfs_step = 0
        return sol

    def _reset_level(s):
        s._model.reset()
        s._graph.reset()
        s._prev_frame = None; s._prev_hash = None
        s._prev_action = None; s._prev_data = None
        s._prev_objects = []
        s._phase = 'PROBE'
        s._probe_step = 0
        s._probe_actions_tried = set()
        s._planned_path = []

    def _compute_effect(s, prev_frame, prev_objects, curr_frame, curr_objects,
                        action_val, data):
        """Compute what an action did by comparing frames and objects."""
        effect = ActionEffect(action_val, data)
        effect.pixel_diff_count = int(np.sum(prev_frame != curr_frame))
        effect.frame_changed = effect.pixel_diff_count > 0

        if not effect.frame_changed:
            return effect

        # Match objects
        matched = match_objects(prev_objects, curr_objects)
        curr_by_oid = {o.oid: o for o in curr_objects}
        prev_by_oid = {o.oid: o for o in prev_objects}

        # Check movements
        for curr_oid, prev_oid in matched.items():
            co = curr_by_oid[curr_oid]
            po = prev_by_oid[prev_oid]
            dx = co.cx - po.cx
            dy = co.cy - po.cy
            if abs(dx) > 0.5 or abs(dy) > 0.5:
                effect.objects_moved.append((prev_oid, dx, dy))
            if co.color != po.color:
                effect.objects_color_changed.append((prev_oid, po.color, co.color))

        # Check appearances/disappearances
        matched_curr = set(matched.keys())
        matched_prev = set(matched.values())
        for o in curr_objects:
            if o.oid not in matched_curr:
                effect.objects_appeared.append(o.oid)
        for o in prev_objects:
            if o.oid not in matched_prev:
                effect.objects_disappeared.append(o.oid)

        return effect

    def _action_key(s, action_val, data):
        """Create a hashable key for an action."""
        if data and 'x' in data:
            return (action_val, data.get('x', 0) // 4, data.get('y', 0) // 4)
        return (action_val,)

    def is_done(s, frames, lf):
        try: return lf.state is GameState.WIN or (time.time()-s.start_time) >= 8*3600-300
        except: return True

    def choose_action(s, frames, lf):
        try:
            lvl = s._lvl(lf)

            # ===== LEVEL CHANGE =====
            if lvl != s._cl:
                if not s._bfs_tried:
                    s._bfs_tried = True
                    s._init_bfs()
                s._bfs_solution = None; s._bfs_step = 0
                if s._bfs:
                    s._try_bfs(lvl)
                s._reset_level()
                s._cl = lvl

            # ===== RESET =====
            if lf.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
                s._prev_frame = None; s._prev_hash = None
                s._prev_action = None; s._prev_data = None
                a = GameAction.RESET; a.reasoning = "reset"; return a

            # ===== BFS SOLUTION REPLAY =====
            if s._bfs_solution and s._bfs_step < len(s._bfs_solution):
                act_id, data = s._bfs_solution[s._bfs_step]
                s._bfs_step += 1
                sel = GameAction.from_id(act_id)
                if data: sel.set_data(data)
                sel.reasoning = f"bfs:{s._bfs_step}/{len(s._bfs_solution)}"
                return sel

            # ===== ATLAS: OBSERVE + LEARN + ACT =====
            raw = s._raw(lf)
            cnt = np.bincount(raw.flatten(), minlength=16)
            s._bg = int(cnt.argmax())
            frame_hash = hashlib.md5(raw.tobytes()).hexdigest()[:16]

            # Extract objects
            objects = extract_objects(raw, s._bg)

            # Record transition from previous action
            if s._prev_frame is not None and s._prev_action is not None:
                effect = s._compute_effect(s._prev_frame, s._prev_objects,
                                           raw, objects, s._prev_action, s._prev_data)
                s._model.observe(effect)
                ak = s._action_key(s._prev_action, s._prev_data)
                s._graph.record_transition(s._prev_hash, ak, frame_hash,
                                           effect.frame_changed)

            # Update state graph
            s._graph.observe(frame_hash, raw, objects)

            # Get available actions
            avail = getattr(lf, 'available_actions', None) or []
            avail_vals = set()
            for a in avail:
                aid = a.value if hasattr(a, 'value') else int(a)
                avail_vals.add(aid)
            avail_vals.discard(0)  # remove RESET

            tested_here = set()
            node = s._graph.nodes.get(frame_hash)
            if node:
                tested_here = node.tested_actions

            # ===== PHASE: PROBE (first 20 actions) =====
            if s._phase == 'PROBE' and s._probe_step < 20:
                s._probe_step += 1
                action_val = None; data = None

                # First 5: try directional actions
                if s._probe_step <= 5:
                    for a in range(1, 6):
                        if a in avail_vals and a not in s._probe_actions_tried:
                            action_val = a
                            s._probe_actions_tried.add(a)
                            break

                # Next 15: click on objects (small first)
                if action_val is None and 6 in avail_vals:
                    sorted_objs = sorted(objects, key=lambda o: o.size)
                    for obj in sorted_objs:
                        x, y = int(obj.cx), int(obj.cy)
                        key = (6, x // 4, y // 4)
                        if key not in s._probe_actions_tried:
                            action_val = 6
                            data = {'x': x, 'y': y}
                            s._probe_actions_tried.add(key)
                            break

                if action_val is None:
                    # Exhausted probe — switch to MODEL phase
                    s._phase = 'MODEL'
                else:
                    s._prev_frame = raw.copy()
                    s._prev_hash = frame_hash
                    s._prev_action = action_val
                    s._prev_data = data
                    s._prev_objects = objects

                    sel = GA_MAP.get(action_val, GameAction.ACTION1)
                    if data: sel.set_data(data)
                    sel.reasoning = f"probe:{s._probe_step}"
                    return sel

            # Switch to MODEL phase after probe
            if s._phase == 'PROBE':
                s._phase = 'MODEL'
                game_type = s._model.classify_game()
                logger.info(f"ATLAS: game classified as {game_type} after {s._model.n_observations} observations")

            # ===== PHASE: MODEL (use learned action-effect model) =====
            if s._phase in ('MODEL', 'PLAN'):
                # Ask the model for a suggestion
                suggestion = s._model.suggest_action(raw, objects, avail_vals,
                                                     frame_hash, tested_here)
                if suggestion:
                    action_val, data = suggestion
                else:
                    s._phase = 'EXPLORE'

            # ===== PHASE: EXPLORE (state graph + frontier navigation) =====
            if s._phase == 'EXPLORE':
                # Follow planned path if we have one
                if s._planned_path:
                    ak = s._planned_path.pop(0)
                    if isinstance(ak, tuple) and len(ak) == 3:
                        action_val, gy, gx = ak[0], ak[1] * 4 + 2, ak[2] * 4 + 2
                        data = {'x': gx, 'y': gy} if action_val == 6 else None
                    elif isinstance(ak, tuple) and len(ak) == 1:
                        action_val = ak[0]; data = None
                    else:
                        action_val = ak if isinstance(ak, int) else 1; data = None
                else:
                    # BFS to frontier
                    avail_action_keys = set()
                    for a in avail_vals:
                        if a <= 5:
                            avail_action_keys.add((a,))
                        elif a == 6:
                            for obj in objects[:10]:
                                avail_action_keys.add((6, int(obj.cy) // 4, int(obj.cx) // 4))
                    path = s._graph.bfs_to_frontier(frame_hash, avail_action_keys)
                    if path and len(path) > 0:
                        s._planned_path = path[1:]
                        ak = path[0]
                        if isinstance(ak, tuple) and len(ak) == 3:
                            action_val = ak[0]
                            data = {'x': ak[2]*4+2, 'y': ak[1]*4+2} if ak[0] == 6 else None
                        else:
                            action_val = ak[0] if isinstance(ak, tuple) else ak
                            data = None
                    else:
                        # Complete fallback: random productive action
                        productive = list(s._model.productive_actions & avail_vals)
                        if productive:
                            action_val = random.choice(productive)
                        else:
                            action_val = random.choice(list(avail_vals)) if avail_vals else 1
                        data = None
                        if action_val == 6 and objects:
                            obj = random.choice(objects)
                            data = {'x': int(obj.cx), 'y': int(obj.cy)}
                        elif action_val == 6:
                            nz = np.argwhere(raw != s._bg)
                            if len(nz):
                                idx = random.randint(0, len(nz)-1)
                                data = {'x': int(nz[idx][1]), 'y': int(nz[idx][0])}
                            else:
                                data = {'x': 32, 'y': 32}

            # Record state for next transition
            s._prev_frame = raw.copy()
            s._prev_hash = frame_hash
            s._prev_action = action_val
            s._prev_data = data
            s._prev_objects = objects

            sel = GA_MAP.get(action_val, GameAction.ACTION1)
            if data: sel.set_data(data)
            game_type = s._model.classify_game() if s._model.n_observations > 5 else '?'
            sel.reasoning = f"atlas:{s._phase}:{game_type}"
            return sel

        except Exception as e:
            traceback.print_exc()
            a = random.choice(ACTION_LIST); a.reasoning = f"err:{e}"; return a
