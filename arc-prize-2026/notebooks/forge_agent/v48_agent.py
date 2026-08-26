# =====================================================================
# v48 = BFS + SG-CNN hybrid
#   - v39 full BFS infra (BFSSolver, _fast_deepcopy, find_game_source_and_class)
#   - StochasticGoose ActionModel (16-ch one-hot, 4-conv backbone, 5+4096 action head)
#   - choose_action: if BFS solves the level execute BFS plan; else SG-CNN online learning
#   - NO GraphExplorer (replaces v35 GE fallback with SG-CNN)
# =====================================================================
# =====================================================================
# MASTER BASELINE v10 — mynotebook_5 + MCTS Phase 7 (30s fixed cap, beam UNCHANGED)
#
# Built by merging the best parts of 6 top public notebooks:
#
# CORE: FORGE v19 (op_2) — most advanced BFS engine:
#   - A* search with game introspection heuristic (indicator sprites)
#   - Transient field detection (avoids state explosion from counters)
#   - _get_valid_actions() for correct click coordinate detection
#   - Dynamic action rescan BFS (for flood-fill games)
#   - Object model tracking (static/dynamic classification)
#   - _fast_deepcopy (skips camera for 2-3x faster copying)
#   - Level advancement by action replay (correct for multi-level)
#
# ADDITIONS from FORGE v17 (op_3):
#   - Beam search fallback (width 20-200, depth 60)
#   - Sprite permutation for click-only games ≤8 sprites
#   - Stride-1 neighbor click probing (catch odd-coordinate sprites)
#   - Prioritized experience replay (recent + high-reward weighted)
#   - Adaptive BFS time budget
#
# ADDITIONS from MCTS notebook (op_5):
#   - Click masking during CNN inference (only predict known-effective positions)
#   - Novelty-guided action selection during exploration phase
#
# ALL v19 BUG FIXES:
#   - _visited_hashes properly initialized in __init__
#   - 2 RESET calls (not 3) in BFS hidden retry
#   - Epsilon only resets when BFS actually failed
#   - FIX: frame extraction uses perform_action result throughout
# =====================================================================
import copy
import glob
import hashlib
import heapq
import importlib.util
import logging
import math
import os
import pickle
import random
import time
import traceback
from collections import defaultdict, deque
from itertools import permutations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState, ActionInput

logger = logging.getLogger(__name__)


# ==================== FAST DEEPCOPY ====================

def _fast_deepcopy(game):
    """Deepcopy game object, skipping the camera (rendering-only, never mutates)."""
    camera = getattr(game, '_camera', None)
    if camera is not None:
        game._camera = None
    try:
        g = pickle.loads(pickle.dumps(game, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:
        # v39: pickle fails when the game module isn't importable by name
        # (dynamic-load harnesses / some Kaggle paths). copy.deepcopy is
        # slower but always works -> correctness over speed.
        g = copy.deepcopy(game)
    if camera is not None:
        game._camera = camera
        g._camera = camera
    return g


# ==================== BFS SOLVER ====================

class BFSSolver:
    """Hybrid search engine: A* + dynamic rescan + IDDFS + beam + sprite permutation."""

    def __init__(self, game_path, game_class_name, scan_timeout=4, bfs_timeout=180):
        self.game_path = game_path
        self.class_name = game_class_name
        self.scan_timeout = scan_timeout
        self.bfs_timeout = bfs_timeout
        self.game_cls = None
        self.solutions = {}
        self.timed_out_levels = set()

    def load(self):
        try:
            spec = importlib.util.spec_from_file_location('game_mod', self.game_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.game_cls = getattr(mod, self.class_name)
            return True
        except Exception as e:
            logger.warning(f"BFS: Failed to load game class: {e}")
            return False

    # ---- state hashing ----

    def _state_hash(self, g, frame, hidden_fields=None, transient_fields=None):
        fh = str(hash(frame.tobytes()))
        ignore = {'_action_count', '_full_reset', '_action_complete', '_debug', '_seed'}
        if transient_fields:
            ignore.update(transient_fields)
        extras = []
        for k, v in g.__dict__.items():
            if k.startswith('__') or k in ignore:
                continue
            if isinstance(v, (int, float, bool)):
                extras.append(f"{k}={v}")
            elif isinstance(v, (set, frozenset)) and len(v) < 50:
                extras.append(f"{k}={sorted(str(i) for i in v)}")
        if extras:
            eh = str(hash("|".join(sorted(extras))))
            return fh + "|" + eh
        return fh

    # ---- hidden / transient field detection ----

    def _probe_hidden_fields(self, game, actions):
        if not actions:
            return []
        initial = {k: v for k, v in game.__dict__.items()
                   if isinstance(v, (int, float, bool)) and not k.startswith('__')}
        changing = set()
        frame0 = game.get_pixels(0, 0, 64, 64)
        for act_id, data in actions[:10]:
            g = _fast_deepcopy(game)
            try:
                ai = (ActionInput(id=GameAction.from_id(act_id), data=data)
                      if data else ActionInput(id=GameAction.from_id(act_id)))
                g.perform_action(ai, raw=True)
            except:
                continue
            for k, v in g.__dict__.items():
                if isinstance(v, (int, float, bool)) and not k.startswith('__'):
                    if k in initial and v != initial[k]:
                        if k not in ('_action_count', '_full_reset', '_action_complete'):
                            changing.add(k)
        return sorted(f for f in changing
                      if not f.startswith('_') or f in ('_current_level_index', '_score'))

    def _detect_transient_fields(self, game, actions):
        """Fields that change on EVERY action — e.g. budget counters. Exclude from hash."""
        if not actions:
            return set()
        ignore = {'_action_count', '_full_reset', '_action_complete'}
        initial = {k: v for k, v in game.__dict__.items()
                   if isinstance(v, (int, float, bool)) and not k.startswith('__')
                   and k not in ignore}
        changed_count = defaultdict(int)
        n_sampled = 0
        for act_id, data in actions[:min(12, len(actions))]:
            g = _fast_deepcopy(game)
            try:
                ai = (ActionInput(id=GameAction.from_id(act_id), data=data)
                      if data else ActionInput(id=GameAction.from_id(act_id)))
                g.perform_action(ai, raw=True)
            except:
                continue
            n_sampled += 1
            for k in initial:
                if getattr(g, k, initial[k]) != initial[k]:
                    changed_count[k] += 1
        if n_sampled == 0:
            return set()
        transient = set()
        for k, cnt in changed_count.items():
            if cnt != n_sampled:
                continue
            if isinstance(initial[k], bool):
                continue  # boolean flags encode meaningful state
            transient.add(k)
        if transient:
            logger.info(f"BFS: transient fields (excluded from hash): {transient}")
        return transient

    # ---- goal heuristic (indicator introspection) ----

    def _build_goal_heuristic(self, f_init, f_prev_win, demo_model=None):
        def count_indicators(game):
            try:
                total, satisfied = 0, 0
                for av in game.__dict__.values():
                    if not isinstance(av, dict):
                        continue
                    for v in av.values():
                        if not isinstance(v, list):
                            continue
                        for item in v:
                            if hasattr(item, 'is_visible') and hasattr(item, 'pixels'):
                                total += 1
                                if item.is_visible:
                                    satisfied += 1
                return total, satisfied
            except:
                return 0, 0

        if self.game_cls:
            try:
                test = self.game_cls()
                test.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                test.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                total, _ = count_indicators(test)
                if total > 0:
                    logger.info(f"BFS heuristic: introspection found {total} indicators")
                    def introspection_heuristic(f, game=None):
                        if game is None:
                            return 0
                        t, s = count_indicators(game)
                        return max(0, t - s)
                    return introspection_heuristic
            except:
                pass

        logger.info("BFS heuristic: uniform cost (no indicators found)")
        return lambda f, game=None: 0

    # ---- action scanning ----

    def _scan_actions(self, game, f0, bg):
        """Scan for effective actions. Uses _get_valid_actions() when available (fast + precise)."""
        avail = game._available_actions
        actions = []

        # Directional / interact
        for a in [a for a in avail if a <= 5]:
            g = _fast_deepcopy(game)
            try:
                r = g.perform_action(ActionInput(id=GameAction.from_id(a)), raw=True)
                if r.frame and np.sum(f0 != np.array(r.frame[-1])) > 0:
                    actions.append((a, None))
            except:
                pass

        if 6 not in avail:
            return actions

        seen_effects = set()

        # Primary: use game's own valid action list (exact click coords, much faster)
        if hasattr(game, '_get_valid_actions'):
            try:
                for ai_obj in game._get_valid_actions():
                    act_id = ai_obj.id._value_ if hasattr(ai_obj.id, '_value_') else int(ai_obj.id)
                    if act_id != 6:
                        continue
                    g = _fast_deepcopy(game)
                    try:
                        r = g.perform_action(ai_obj, raw=True)
                        if r.frame:
                            f = np.array(r.frame[-1])
                            if np.sum(f0 != f) > 0:
                                eh = hashlib.md5(f.tobytes()).hexdigest()[:12]
                                if eh not in seen_effects:
                                    seen_effects.add(eh)
                                    actions.append((6, ai_obj.data))
                    except:
                        pass
            except:
                pass

        # Fallback: pixel scan (stride 2) if _get_valid_actions unavailable
        if not seen_effects:
            t0 = time.time()
            hit_positions = []
            for y in range(0, 64, 2):
                if time.time() - t0 > self.scan_timeout:
                    break
                for x in range(0, 64, 2):
                    if f0[y, x] == bg:
                        continue
                    g = _fast_deepcopy(game)
                    try:
                        r = g.perform_action(
                            ActionInput(id=GameAction.ACTION6,
                                        data={'x': x, 'y': y, 'game_id': 'bfs'}),
                            raw=True)
                        if not r.frame:
                            continue
                        f = np.array(r.frame[-1])
                        if np.sum(f0 != f) > 0:
                            eh = hashlib.md5(f.tobytes()).hexdigest()[:12]
                            if eh not in seen_effects:
                                seen_effects.add(eh)
                                actions.append((6, {'x': x, 'y': y, 'game_id': 'bfs'}))
                                hit_positions.append((x, y))
                    except:
                        pass

            # Stride-1 neighbors of hit positions (catch odd-coordinate sprites)
            tried = {(x, y) for x, y in hit_positions}
            for hx, hy in hit_positions:
                if time.time() - t0 > self.scan_timeout * 1.5:
                    break
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = hx + dx, hy + dy
                    if (nx, ny) in tried or not (0 <= nx < 64 and 0 <= ny < 64):
                        continue
                    tried.add((nx, ny))
                    if f0[ny, nx] == bg:
                        continue
                    g = _fast_deepcopy(game)
                    try:
                        r = g.perform_action(
                            ActionInput(id=GameAction.ACTION6,
                                        data={'x': nx, 'y': ny, 'game_id': 'bfs'}),
                            raw=True)
                        if r.frame:
                            f = np.array(r.frame[-1])
                            if np.sum(f0 != f) > 0:
                                eh = hashlib.md5(f.tobytes()).hexdigest()[:12]
                                if eh not in seen_effects:
                                    seen_effects.add(eh)
                                    actions.append((6, {'x': nx, 'y': ny, 'game_id': 'bfs'}))
                    except:
                        pass

        return actions

    def _probe_mover_target_colors(self, game):
        g = _fast_deepcopy(game)
        avail = [a for a in game._available_actions if 1 <= a <= 4]
        if not avail:
            return set(), set()
        try:
            r0 = g.perform_action(ActionInput(id=GameAction.from_id(avail[0])), raw=True)
            if not r0.frame:
                return set(), set()
            f0 = np.array(r0.frame[-1])
            bg = int(np.bincount(f0.flatten(), minlength=16).argmax())

            def get_centroids(frame):
                result = {}
                for c in range(16):
                    if c == bg:
                        continue
                    mask = (frame == c)
                    n = int(np.sum(mask))
                    if n < 2:
                        continue
                    ys, xs = np.where(mask)
                    result[c] = (float(np.mean(xs)), float(np.mean(ys)))
                return result

            movement = {}
            prev_c = get_centroids(f0)
            for _ in range(20):
                act = random.choice(avail)
                r2 = g.perform_action(ActionInput(id=GameAction.from_id(act)), raw=True)
                if not r2.frame:
                    break
                curr_c = get_centroids(np.array(r2.frame[-1]))
                for c in prev_c:
                    if c in curr_c:
                        movement[c] = (movement.get(c, 0.0)
                                       + abs(curr_c[c][0] - prev_c[c][0])
                                       + abs(curr_c[c][1] - prev_c[c][1]))
                prev_c = curr_c

            mover_colors = {c for c, m in movement.items() if m > 5}
            target_colors = {c for c, m in movement.items() if m == 0}
            return mover_colors, target_colors
        except:
            return set(), set()

    # ---- main solver ----

    def _init_game_at_level(self, level_idx):
        """Return (game, last_r) positioned at level_idx. Uses set_level() if available, else action replay."""
        game = self.game_cls()
        game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        last_r = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        if level_idx == 0:
            return game, last_r
        if hasattr(game, 'set_level'):
            try:
                game.set_level(level_idx)
                game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                last_r = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                return game, last_r
            except Exception:
                pass
        # Fallback: action replay
        game = self.game_cls()
        game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        last_r = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        for prev_idx in range(level_idx):
            prev_sol = self.solutions.get(prev_idx)
            if not prev_sol:
                return None, None
            for act_id, data in prev_sol:
                ai = (ActionInput(id=GameAction.from_id(act_id), data=data)
                      if data else ActionInput(id=GameAction.from_id(act_id)))
                last_r = game.perform_action(ai, raw=True)
        return game, last_r

    def solve_level(self, level_idx, max_states=500000,
                    prev_solution=None, goal_heuristic=None):
        if not self.game_cls:
            return None

        # Advance to target level using set_level() or action replay
        game, last_r = self._init_game_at_level(level_idx)
        if game is None:
            return None

        if not last_r.frame:
            return None
        f0 = np.array(last_r.frame[-1])
        bg = int(np.bincount(f0.flatten(), minlength=16).argmax())

        # Try solution transfer from previous level
        if prev_solution and level_idx > 0:
            transfer = self._try_transfer(game, level_idx, prev_solution, f0)
            if transfer:
                return transfer

        # Scan actions
        actions = self._scan_actions(game, f0, bg)

        # Warm-up unlock for frozen initial states
        if not actions:
            logger.info(f"BFS L{level_idx}: 0 actions found, trying warm-up unlock")
            avail = game._available_actions
            # Try click warm-up via _get_valid_actions if available
            if 6 in avail and hasattr(game, '_get_valid_actions'):
                try:
                    for va in game._get_valid_actions():
                        act_id = va.id._value_ if hasattr(va.id, '_value_') else int(va.id)
                        if act_id == 6:
                            g_warmup = _fast_deepcopy(game)
                            g_warmup.perform_action(va, raw=True)
                            r_after = g_warmup.perform_action(
                                ActionInput(id=GameAction.ACTION1), raw=True)
                            if r_after.frame:
                                f_after = np.array(r_after.frame[-1])
                                warmup_actions = self._scan_actions(g_warmup, f_after, bg)
                                if warmup_actions:
                                    logger.info(f"BFS L{level_idx}: UNLOCKED with click!")
                                    game = g_warmup
                                    f0 = f_after
                                    actions = warmup_actions
                                    break
                except:
                    pass
            if not actions:
                for warmup_id in [a for a in avail if a <= 4]:
                    g_warmup = _fast_deepcopy(game)
                    try:
                        g_warmup.perform_action(
                            ActionInput(id=GameAction.from_id(warmup_id)), raw=True)
                        f_after = np.array(g_warmup.get_pixels(0, 0, 64, 64))
                        warmup_actions = self._scan_actions(g_warmup, f_after, bg)
                        if warmup_actions:
                            logger.info(f"BFS L{level_idx}: UNLOCKED with ACTION{warmup_id}!")
                            game = g_warmup
                            f0 = f_after
                            actions = warmup_actions
                            break
                    except:
                        pass

        logger.info(f"BFS L{level_idx}: {len(actions)} effective actions")
        if not actions:
            return None

        transient_fields = self._detect_transient_fields(game, actions)
        hfn = goal_heuristic if goal_heuristic is not None else (lambda f, game=None: 0)
        _hfn_uses_game = goal_heuristic is not None

        # ---- Phase 1: A* search ----
        visited = set()
        base_game = _fast_deepcopy(game)
        h0 = self._state_hash(game, f0, transient_fields=transient_fields)
        visited.add(h0)
        counter = 0
        pq = [(hfn(f0, game) * 10, 0, counter, [], base_game)]
        t0 = time.time()
        explored = 0

        while pq and explored < max_states and (time.time() - t0) < self.bfs_timeout:
            f_score, g_score, _, hist, node_game = heapq.heappop(pq)
            for act_id, data in actions:
                g2 = _fast_deepcopy(node_game)
                try:
                    ai = (ActionInput(id=GameAction.from_id(act_id), data=data)
                          if data else ActionInput(id=GameAction.from_id(act_id)))
                    r = g2.perform_action(ai, raw=True)
                except:
                    continue
                explored += 1
                if not r.frame:
                    continue
                f = np.array(r.frame[-1])
                h = self._state_hash(g2, f, transient_fields=transient_fields)
                if h in visited:
                    continue
                visited.add(h)
                new_hist = hist + [(act_id, data)]
                new_g = g_score + 1
                if (r.levels_completed > level_idx
                        or g2._current_level_index > level_idx):
                    elapsed = time.time() - t0
                    logger.info(f"BFS L{level_idx}: SOLVED (A*) in {len(new_hist)} actions "
                                f"({explored} explored, {elapsed:.1f}s)")
                    self.solutions[level_idx] = new_hist
                    return new_hist
                h_val = hfn(f, g2 if _hfn_uses_game else None) * 10
                counter += 1
                heapq.heappush(pq, (new_g + h_val, new_g, counter, new_hist, g2))

        elapsed_first = time.time() - t0
        logger.info(f"BFS L{level_idx}: A* timeout ({explored} explored, "
                    f"{len(visited)} unique, {elapsed_first:.1f}s)")
        self.timed_out_levels.add(level_idx)

        # ---- Phase 2: Dynamic rescan (flood-fill games) ----
        exhausted_quickly = len(pq) == 0 and elapsed_first < self.bfs_timeout * 0.5
        if exhausted_quickly:
            logger.info(f"BFS L{level_idx}: queue exhausted early — dynamic rescan")
            visited_d = {self._state_hash(base_game, f0, transient_fields=transient_fields)}
            queue_d = deque([([], 0, base_game)])
            current_actions = list(actions)
            t0_d = time.time()
            explored_d = 0
            remaining_d = max(30, self.bfs_timeout - elapsed_first)

            while queue_d and explored_d < max_states * 10 and (time.time() - t0_d) < remaining_d:
                hist_d, depth_d, node_game_d = queue_d.popleft()
                for act_id, data in current_actions:
                    g2_d = _fast_deepcopy(node_game_d)
                    try:
                        ai = (ActionInput(id=GameAction.from_id(act_id), data=data)
                              if data else ActionInput(id=GameAction.from_id(act_id)))
                        r = g2_d.perform_action(ai, raw=True)
                    except:
                        continue
                    explored_d += 1
                    if not r.frame:
                        continue
                    f2 = np.array(r.frame[-1])
                    h_d = self._state_hash(g2_d, f2, transient_fields=transient_fields)
                    if h_d in visited_d:
                        continue
                    visited_d.add(h_d)
                    # Rescan from child for newly unlocked actions
                    try:
                        new_acts = self._scan_actions(g2_d, f2, bg)
                        added = [a for a in new_acts if a not in current_actions]
                        if added:
                            logger.info(f"BFS L{level_idx}: rescan found {len(added)} new actions")
                            current_actions.extend(added)
                    except:
                        pass
                    new_hist_d = hist_d + [(act_id, data)]
                    if (r.levels_completed > level_idx
                            or g2_d._current_level_index > level_idx):
                        logger.info(f"BFS L{level_idx}: SOLVED (dynamic rescan) in "
                                    f"{len(new_hist_d)} actions")
                        self.solutions[level_idx] = new_hist_d
                        return new_hist_d
                    if depth_d < 30:
                        queue_d.append((new_hist_d, depth_d + 1, g2_d))

        # ---- Phase 3: Hidden fields retry ----
        elapsed_p2 = time.time() - t0
        if (explored > 0 and (len(visited) < 200 or explored / len(visited) > 5)
                and elapsed_p2 < self.bfs_timeout * 0.8):
            hidden_fields = self._probe_hidden_fields(game, actions)
            if hidden_fields:
                logger.info(f"BFS L{level_idx}: RETRY with hidden fields: {hidden_fields}")
                game2, last_r2 = self._init_game_at_level(level_idx)
                if game2 is None or not last_r2.frame:
                    return None
                f0_2 = np.array(last_r2.frame[-1])
                visited2 = {self._state_hash(game2, f0_2, hidden_fields,
                                              transient_fields=transient_fields)}
                queue2 = deque([([], 0, _fast_deepcopy(game2))])
                t0_2 = time.time()
                explored2 = 0
                remaining2 = max(30, self.bfs_timeout - elapsed_p2)

                while queue2 and explored2 < max_states and (time.time() - t0_2) < remaining2:
                    hist, depth, node_game2 = queue2.popleft()
                    for act_id, data in actions:
                        g2 = _fast_deepcopy(node_game2)
                        try:
                            ai = (ActionInput(id=GameAction.from_id(act_id), data=data)
                                  if data else ActionInput(id=GameAction.from_id(act_id)))
                            r = g2.perform_action(ai, raw=True)
                        except:
                            continue
                        explored2 += 1
                        if not r.frame:
                            continue
                        f = np.array(r.frame[-1])
                        h = self._state_hash(g2, f, hidden_fields,
                                             transient_fields=transient_fields)
                        if h in visited2:
                            continue
                        visited2.add(h)
                        new_hist = hist + [(act_id, data)]
                        if (r.levels_completed > level_idx
                                or g2._current_level_index > level_idx):
                            logger.info(f"BFS L{level_idx}: SOLVED (hidden retry) in "
                                        f"{len(new_hist)} actions")
                            self.solutions[level_idx] = new_hist
                            return new_hist
                        if depth < 30:
                            queue2.append((new_hist, depth + 1, g2))

        # ---- Phase 4: IDDFS (deep directional games, low branching) ----
        elapsed_p3 = time.time() - t0
        remaining_iddfs = max(30, self.bfs_timeout - elapsed_p3)
        if len(actions) <= 6 and remaining_iddfs > 30:
            logger.info(f"BFS L{level_idx}: trying IDDFS (branching={len(actions)}, "
                        f"{remaining_iddfs:.0f}s remaining)")
            game3, _ = self._init_game_at_level(level_idx)
            if game3 is None:
                game3 = _fast_deepcopy(game)
            t0_iddfs = time.time()
            for max_depth in range(10, 60):
                if time.time() - t0_iddfs > remaining_iddfs:
                    break
                stack = [(_fast_deepcopy(game3), [], set())]
                while stack and (time.time() - t0_iddfs) < remaining_iddfs:
                    g, hist, path_hashes = stack.pop()
                    if len(hist) >= max_depth:
                        continue
                    for act_id, data in actions:
                        g2 = _fast_deepcopy(g)
                        try:
                            ai = (ActionInput(id=GameAction.from_id(act_id), data=data)
                                  if data else ActionInput(id=GameAction.from_id(act_id)))
                            r = g2.perform_action(ai, raw=True)
                        except:
                            continue
                        if not r.frame:
                            continue
                        f = np.array(r.frame[-1])
                        fh = hash(f.tobytes())
                        if fh in path_hashes:
                            continue
                        new_hist = hist + [(act_id, data)]
                        if (r.levels_completed > level_idx
                                or g2._current_level_index > level_idx):
                            logger.info(f"BFS L{level_idx}: SOLVED (IDDFS d={max_depth}) "
                                        f"in {len(new_hist)} actions")
                            self.solutions[level_idx] = new_hist
                            return new_hist
                        stack.append((g2, new_hist, path_hashes | {fh}))
            logger.info(f"BFS L{level_idx}: IDDFS exhausted")

        # ---- Phase 5: Sprite permutation (click-only games with ≤8 targets) ----
        elapsed_p4 = time.time() - t0
        remaining_perm = max(20, self.bfs_timeout - elapsed_p4)
        click_actions = [a for a in actions if a[0] == 6]
        non_click = [a for a in actions if a[0] != 6]
        if not non_click and 1 <= len(click_actions) <= 8 and remaining_perm > 10:
            logger.info(f"BFS L{level_idx}: trying sprite permutation "
                        f"({len(click_actions)} clicks)")
            t0_perm = time.time()
            perm_timeout = min(60, remaining_perm)
            for perm in permutations(range(len(click_actions))):
                if time.time() - t0_perm > perm_timeout:
                    break
                g_perm = _fast_deepcopy(game)
                hist_perm = []
                for idx in perm:
                    act_id, data = click_actions[idx]
                    try:
                        ai = (ActionInput(id=GameAction.from_id(act_id), data=data)
                              if data else ActionInput(id=GameAction.from_id(act_id)))
                        r = g_perm.perform_action(ai, raw=True)
                        hist_perm.append((act_id, data))
                        if (r.levels_completed > level_idx
                                or g_perm._current_level_index > level_idx):
                            logger.info(f"BFS L{level_idx}: SOLVED (permutation) "
                                        f"in {len(hist_perm)} actions")
                            self.solutions[level_idx] = hist_perm
                            return hist_perm
                    except:
                        break
            logger.info(f"BFS L{level_idx}: permutation exhausted")

        # ---- Phase 5.5: Random click ordering for medium click-only games (9-20 targets) ----
        elapsed_p55 = time.time() - t0
        remaining_p55 = max(20, self.bfs_timeout - elapsed_p55)
        if not non_click and 9 <= len(click_actions) <= 20 and remaining_p55 > 10:
            logger.info(f"BFS L{level_idx}: random click ordering "
                        f"({len(click_actions)} clicks, {remaining_p55:.0f}s)")
            t0_rand = time.time()
            rand_timeout = min(30, remaining_p55)
            tried_perms = set()
            while time.time() - t0_rand < rand_timeout:
                perm = tuple(random.sample(range(len(click_actions)), len(click_actions)))
                if perm in tried_perms:
                    continue
                tried_perms.add(perm)
                g_rand = _fast_deepcopy(game)
                hist_rand = []
                solved = False
                for idx in perm:
                    act_id, data = click_actions[idx]
                    try:
                        ai = (ActionInput(id=GameAction.from_id(act_id), data=data)
                              if data else ActionInput(id=GameAction.from_id(act_id)))
                        r = g_rand.perform_action(ai, raw=True)
                        hist_rand.append((act_id, data))
                        if (r.levels_completed > level_idx
                                or g_rand._current_level_index > level_idx):
                            logger.info(f"BFS L{level_idx}: SOLVED (random ordering, "
                                        f"{len(tried_perms)} tries) in {len(hist_rand)} actions")
                            self.solutions[level_idx] = hist_rand
                            solved = True
                            return hist_rand
                    except:
                        break
                if solved:
                    break
            logger.info(f"BFS L{level_idx}: random ordering exhausted "
                        f"({len(tried_perms)} tries)")

        # ---- Phase 6: Beam search (medium branching, medium depth) ----
        elapsed_p5 = time.time() - t0
        remaining_bs = max(20, self.bfs_timeout - elapsed_p5)
        if 2 <= len(actions) <= 20 and remaining_bs > 20:
            logger.info(f"BFS L{level_idx}: trying beam search "
                        f"(branching={len(actions)}, {remaining_bs:.0f}s)")
            bw = min(200, max(20, max_states // (len(actions) * 50)))
            game_b = _fast_deepcopy(game)
            f0_b = f0
            beam = [(_fast_deepcopy(game_b), [])]
            vis_b = {self._state_hash(game_b, f0_b, transient_fields=transient_fields)}
            t0_b = time.time()

            for bd in range(60):
                if time.time() - t0_b > remaining_bs or not beam:
                    break
                cands = []
                for g_b, hist_b in beam:
                    for act_id, data in actions:
                        g2 = _fast_deepcopy(g_b)
                        try:
                            ai = (ActionInput(id=GameAction.from_id(act_id), data=data)
                                  if data else ActionInput(id=GameAction.from_id(act_id)))
                            r = g2.perform_action(ai, raw=True)
                        except:
                            continue
                        if not r.frame:
                            continue
                        f = np.array(r.frame[-1])
                        h = self._state_hash(g2, f, transient_fields=transient_fields)
                        if h in vis_b:
                            continue
                        vis_b.add(h)
                        nh = hist_b + [(act_id, data)]
                        if (r.levels_completed > level_idx
                                or g2._current_level_index > level_idx):
                            logger.info(f"BFS L{level_idx}: SOLVED (beam d={bd}) "
                                        f"in {len(nh)} actions")
                            self.solutions[level_idx] = nh
                            return nh
                        pdiff = float(np.sum(f != f0_b)) / 4096.0
                        h_val = hfn(f, g2)
                        score = pdiff + 1.0 / (1.0 + h_val)
                        cands.append((score, g2, nh))
                if not cands:
                    break
                cands.sort(key=lambda x: x[0], reverse=True)
                beam = [(g_b, h_b) for _, g_b, h_b in cands[:bw]]

            logger.info(f"BFS L{level_idx}: beam done ({len(vis_b)} unique, "
                        f"{time.time()-t0_b:.1f}s)")

        # ---- Phase 7: MCTS (fixed 30s budget, non-monotonic games) ----
        elapsed_p7 = time.time() - t0
        mcts_budget = min(30, max(0, self.bfs_timeout - elapsed_p7))
        if 2 <= len(actions) <= 30 and mcts_budget >= 15:
            logger.info(f"BFS L{level_idx}: MCTS ({len(actions)} actions, {mcts_budget:.0f}s)")
            sol = self._mcts_phase(game, f0, actions, level_idx, mcts_budget, hfn)
            if sol:
                return sol
            logger.info(f"BFS L{level_idx}: MCTS exhausted")

        return None

    def _try_transfer(self, game, level_idx, prev_solution, f1):
        try:
            # Direct replay
            g = _fast_deepcopy(game)
            for i, (act_id, data) in enumerate(prev_solution):
                ai = (ActionInput(id=GameAction.from_id(act_id), data=data)
                      if data else ActionInput(id=GameAction.from_id(act_id)))
                try:
                    r = g.perform_action(ai, raw=True)
                    if r.levels_completed > level_idx or g._current_level_index > level_idx:
                        sol = prev_solution[:i + 1]
                        self.solutions[level_idx] = sol
                        logger.info(f"BFS L{level_idx}: TRANSFER (direct replay, {i+1} actions)")
                        return sol
                except:
                    break

            # Object-relative offset transfer
            prev_game = self.game_cls()
            prev_game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            r_prev = prev_game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            if not r_prev.frame:
                return None
            f0 = np.array(r_prev.frame[-1])
            bg = int(np.bincount(f0.flatten(), minlength=16).argmax())

            def get_objects(frame, bg_c):
                objs = []
                for c in range(16):
                    if c == bg_c:
                        continue
                    mask = (frame == c)
                    n = int(np.sum(mask))
                    if n < 2:
                        continue
                    ys, xs = np.where(mask)
                    objs.append({'color': c, 'cx': float(np.mean(xs)),
                                 'cy': float(np.mean(ys)), 'n': n})
                return sorted(objs, key=lambda o: (o['color'], -o['n']))

            objs_prev = get_objects(f0, bg)
            objs_curr = get_objects(f1, bg)
            if not objs_prev or not objs_curr:
                return None

            matched = []
            for op in objs_prev:
                best, best_dist = None, float('inf')
                for oc in objs_curr:
                    if (oc['color'] == op['color']
                            and abs(oc['n'] - op['n']) < max(op['n'], oc['n']) * 0.5):
                        d = abs(oc['cx'] - op['cx']) + abs(oc['cy'] - op['cy'])
                        if d < best_dist:
                            best_dist = d
                            best = oc
                if best:
                    matched.append((op, best))
            if not matched:
                return None

            dx = float(np.mean([m[1]['cx'] - m[0]['cx'] for m in matched]))
            dy = float(np.mean([m[1]['cy'] - m[0]['cy'] for m in matched]))

            transferred = []
            for act_id, data in prev_solution:
                if data and 'x' in data:
                    new_data = dict(data)
                    new_data['x'] = max(0, min(63, int(data['x'] + dx)))
                    new_data['y'] = max(0, min(63, int(data['y'] + dy)))
                    transferred.append((act_id, new_data))
                else:
                    transferred.append((act_id, data))

            g = _fast_deepcopy(game)
            for i, (act_id, data) in enumerate(transferred):
                try:
                    ai = (ActionInput(id=GameAction.from_id(act_id), data=data)
                          if data else ActionInput(id=GameAction.from_id(act_id)))
                    r = g.perform_action(ai, raw=True)
                    if r.levels_completed > level_idx or g._current_level_index > level_idx:
                        sol = transferred[:i + 1]
                        self.solutions[level_idx] = sol
                        logger.info(f"BFS L{level_idx}: TRANSFER (offset dx={dx:.0f},"
                                    f"dy={dy:.0f}, {i+1} actions)")
                        return sol
                except:
                    break

            # Action multiplier transfer
            for multiplier in [2, 3, 4]:
                expanded = []
                for act_id, data in prev_solution:
                    for _ in range(multiplier):
                        if data:
                            new_data = dict(data)
                            new_data['x'] = max(0, min(63, int(data.get('x', 32) + dx)))
                            new_data['y'] = max(0, min(63, int(data.get('y', 32) + dy)))
                            expanded.append((act_id, new_data))
                        else:
                            expanded.append((act_id, data))
                g = _fast_deepcopy(game)
                for i, (act_id, data) in enumerate(expanded):
                    try:
                        ai = (ActionInput(id=GameAction.from_id(act_id), data=data)
                              if data else ActionInput(id=GameAction.from_id(act_id)))
                        r = g.perform_action(ai, raw=True)
                        if r.levels_completed > level_idx or g._current_level_index > level_idx:
                            sol = expanded[:i + 1]
                            self.solutions[level_idx] = sol
                            logger.info(f"BFS L{level_idx}: TRANSFER (x{multiplier}, "
                                        f"{i+1} actions)")
                            return sol
                    except:
                        break
        except Exception as e:
            logger.warning(f"BFS transfer failed: {e}")
        return None

    def _mcts_phase(self, init_game, f0, actions, level_idx, time_budget, hfn):
        """UCT MCTS: tree search with random rollouts for non-monotonic games."""
        C = 1.41

        class Node:
            __slots__ = ['game', 'hist', 'parent', 'children', 'unexplored',
                         'visits', 'value', 'pdiff']
            def __init__(self, game, hist, parent, available_actions):
                self.game = game
                self.hist = hist
                self.parent = parent
                self.children = []
                self.unexplored = list(available_actions)
                random.shuffle(self.unexplored)
                self.visits = 0
                self.value = 0.0
                self.pdiff = 0.0

        root = Node(_fast_deepcopy(init_game), [], None, actions)
        t0 = time.time()
        n_nodes = 1

        while time.time() - t0 < time_budget:
            node = root
            while not node.unexplored and node.children:
                best = max(
                    node.children,
                    key=lambda c: (
                        (c.value / c.visits) + C * math.sqrt(math.log(node.visits) / c.visits)
                        if c.visits > 0 else float('inf')
                    )
                )
                node = best

            if not node.unexplored:
                n = node
                while n:
                    n.visits += 1
                    n = n.parent
                continue

            act_id, data = node.unexplored.pop()
            g2 = _fast_deepcopy(node.game)
            try:
                ai = (ActionInput(id=GameAction.from_id(act_id), data=data)
                      if data else ActionInput(id=GameAction.from_id(act_id)))
                r = g2.perform_action(ai, raw=True)
            except:
                continue
            if not r.frame:
                continue

            f = np.array(r.frame[-1])
            new_hist = node.hist + [(act_id, data)]

            if (r.levels_completed > level_idx or g2._current_level_index > level_idx):
                logger.info(f"BFS L{{level_idx}}: SOLVED (MCTS expand) in {{len(new_hist)}} actions")
                self.solutions[level_idx] = new_hist
                return new_hist

            child = Node(g2, new_hist, node, actions)
            child.pdiff = float(np.sum(f != f0)) / 4096.0
            node.children.append(child)
            n_nodes += 1

            sim_game = _fast_deepcopy(child.game)
            h_val = hfn(f, child.game)
            sim_val = child.pdiff + 0.1 / (1.0 + h_val)
            sim_hist = list(new_hist)

            for _ in range(8):
                act_id2, data2 = random.choice(actions)
                ai2 = (ActionInput(id=GameAction.from_id(act_id2), data=data2)
                       if data2 else ActionInput(id=GameAction.from_id(act_id2)))
                try:
                    r2 = sim_game.perform_action(ai2, raw=True)
                    if not r2.frame:
                        break
                    f2 = np.array(r2.frame[-1])
                    sim_hist.append((act_id2, data2))
                    pd = float(np.sum(f2 != f0)) / 4096.0
                    h2 = hfn(f2, sim_game)
                    sim_val = max(sim_val, pd + 0.1 / (1.0 + h2))
                    if (r2.levels_completed > level_idx or
                            sim_game._current_level_index > level_idx):
                        logger.info(f"BFS L{{level_idx}}: SOLVED (MCTS rollout) in {{len(sim_hist)}} actions")
                        self.solutions[level_idx] = sim_hist
                        return sim_hist
                except:
                    break

            n = child
            while n is not None:
                n.visits += 1
                n.value += sim_val
                n = n.parent

            if n_nodes >= 1500:
                logger.info(f"BFS L{{level_idx}}: MCTS tree capped at {{n_nodes}} nodes")
                break

        return None


def find_game_source_and_class(game_id, arc_env=None):
    import re
    parts = game_id.split('-', 1)
    gid = parts[0]
    guid_suffix = parts[1] if len(parts) > 1 else ''

    # Primary: competition-structured path
    competition_path = (
        f"/kaggle/input/competitions/arc-prize-2026-arc-agi-3"
        f"/environment_files/{gid}/{guid_suffix}/{gid}.py"
    )
    if os.path.exists(competition_path):
        src = competition_path
        m = re.search(r'class\s+(\w+)\s*\(', open(src).read()[:2000])
        cls_name = m.group(1) if m else gid[0].upper() + gid[1:]
        logger.info(f"BFS: found {src} class={cls_name}")
        return src, cls_name

    # Fallback: environment_info or glob
    if arc_env and hasattr(arc_env, 'environment_info'):
        ei = arc_env.environment_info
        if hasattr(ei, 'local_dir') and ei.local_dir:
            from pathlib import Path
            ld = Path(ei.local_dir)
            for cand in [ld / f"{gid}.py", ld / f"{gid.upper()}.py"]:
                if cand.exists():
                    m = re.search(r'class\s+(\w+)\s*\(', cand.read_text()[:2000])
                    cls_name = m.group(1) if m else gid[0].upper() + gid[1:]
                    return str(cand), cls_name

    for pattern in [f"/kaggle/input/**/{gid}.py", f"/tmp/**/{gid}.py",
                    f"/kaggle/working/**/{gid}.py"]:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            src = matches[0]
            m = re.search(r'class\s+(\w+)\s*\(', open(src).read()[:2000])
            cls_name = m.group(1) if m else gid[0].upper() + gid[1:]
            logger.info(f"BFS: found {src} class={cls_name}")
            return src, cls_name

    logger.warning(f"BFS: game source not found for {game_id}")
    return None, gid[0].upper() + gid[1:]


# ==================== CNN MODULES ====================


class ActionModel(nn.Module):
    """SG architecture: 16-ch input, 4 conv layers, action head (5) + 64x64 coord head."""
    def __init__(self, input_channels=16, grid_size=64):
        super().__init__()
        self.grid_size = grid_size
        self.num_action_types = 5

        # Shared conv backbone (all 64x64)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # Action head: maxpool to 16x16 -> flatten -> 512 -> 5
        self.action_pool = nn.MaxPool2d(4, 4)
        self.action_fc = nn.Linear(256 * 16 * 16, 512)
        self.action_head = nn.Linear(512, self.num_action_types)

        # Coord head: spatial 256 -> 128 -> 64 -> 32 -> 1, output 64x64 logits
        self.coord_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.coord_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.coord_conv3 = nn.Conv2d(64, 32, kernel_size=1)
        self.coord_conv4 = nn.Conv2d(32, 1, kernel_size=1)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        conv_features = F.relu(self.conv4(x))

        action_features = self.action_pool(conv_features)
        action_features = action_features.view(action_features.size(0), -1)
        action_features = F.relu(self.action_fc(action_features))
        action_features = self.dropout(action_features)
        action_logits = self.action_head(action_features)

        coord_features = F.relu(self.coord_conv1(conv_features))
        coord_features = F.relu(self.coord_conv2(coord_features))
        coord_features = F.relu(self.coord_conv3(coord_features))
        coord_logits = self.coord_conv4(coord_features)
        coord_logits = coord_logits.view(coord_logits.size(0), -1)

        return torch.cat([action_logits, coord_logits], dim=1)  # (B, 5+4096)




class MyAgent(Agent):
    """BFS + SG-CNN hybrid."""
    MAX_ACTIONS = float('inf')

    def __init__(s, *a, **kw):
        super().__init__(*a, **kw)
        seed = int(hashlib.md5(str(s.game_id).encode()).hexdigest()[:8], 16)
        random.seed(seed)
        np.random.seed(seed % (2 ** 32 - 1))
        torch.manual_seed(seed % (2 ** 32 - 1))

        s.start_time = time.time()
        s.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        s.grid_size = 64
        s.num_coordinates = 64 * 64
        s.num_colours = 16

        s._bfs = None
        s._bfs_tried = False
        s._bfs_solved_last = False
        s._bfs_solution = None
        s._bfs_step = 0
        s.cl = -1

        s.action_model = None
        s.optimizer = None
        s.experience_buffer = deque(maxlen=200000)
        s.experience_hashes = set()
        s.batch_size = 64
        s.train_frequency = 5
        s.prev_frame = None
        s.prev_action_idx = None
        s.current_score = -1

        s.action_list = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
                         GameAction.ACTION4, GameAction.ACTION5]

    def _init_bfs(s):
        src, cls = find_game_source_and_class(s.game_id, s.arc_env if hasattr(s, "arc_env") else None)
        if src:
            s._bfs = BFSSolver(src, cls, scan_timeout=5, bfs_timeout=180)
            try:
                s._bfs.load()
            except Exception:
                pass

    def _try_bfs_solve(s, level_idx):
        if s._bfs is None or s._bfs.game_cls is None:
            return None
        try:
            sol = s._bfs.solve_level(level_idx, max_states=200000)
        except Exception:
            sol = None
        if sol:
            s._bfs_solution = sol
            s._bfs_step = 0
            s._bfs_solved_last = True
        else:
            s._bfs_solved_last = False
        return sol

    def _frame_to_tensor(s, fd):
        frame = np.array(fd.frame, dtype=np.int64)[-1]
        if frame.shape != (s.grid_size, s.grid_size):
            raise RuntimeError("frame shape " + str(frame.shape))
        frame = np.clip(frame, 0, s.num_colours - 1)
        tensor = torch.zeros(s.num_colours, s.grid_size, s.grid_size, dtype=torch.float32)
        tensor.scatter_(0, torch.from_numpy(frame).unsqueeze(0), 1)
        return tensor.to(s.device)

    def _experience_hash(s, frame_np, action_idx):
        return hashlib.md5(frame_np.tobytes() + str(action_idx).encode()).hexdigest()

    def _sample_sg(s, combined_logits, available_actions):
        action_logits = combined_logits[:5].clone()
        coord_logits = combined_logits[5:].clone()
        action6_available = False
        action_mask = torch.full_like(action_logits, float("-inf"))
        if available_actions:
            for a in available_actions:
                av = a.value if hasattr(a, "value") else int(a)
                if 1 <= av <= 5:
                    action_mask[av - 1] = 0.0
                elif av == 6:
                    action6_available = True
            action_logits = action_logits + action_mask
            if not action6_available:
                coord_logits = coord_logits + torch.full_like(coord_logits, float("-inf"))
        action_probs = torch.sigmoid(action_logits)
        coord_probs = torch.sigmoid(coord_logits) / s.num_coordinates
        all_probs = torch.cat([action_probs, coord_probs])
        total = all_probs.sum()
        if not torch.isfinite(total) or total <= 0:
            valid = [i for i in range(5) if action_mask[i] == 0.0]
            return (random.choice(valid) if valid else 0), None
        all_probs = all_probs / total
        idx = int(np.random.choice(len(all_probs), p=all_probs.cpu().numpy()))
        if idx < 5:
            return idx, None
        c = idx - 5
        return 5, (c // s.grid_size, c % s.grid_size)

    def _train_sg(s):
        if len(s.experience_buffer) < s.batch_size:
            return
        idxs = np.random.choice(len(s.experience_buffer), s.batch_size, replace=False)
        batch = [s.experience_buffer[i] for i in idxs]
        states = torch.stack([torch.from_numpy(e["state"]).float().to(s.device) for e in batch])
        action_indices = torch.tensor([e["action_idx"] for e in batch], dtype=torch.long, device=s.device)
        rewards = torch.tensor([e["reward"] for e in batch], dtype=torch.float32, device=s.device)
        s.optimizer.zero_grad()
        logits = s.action_model(states)
        selected = logits.gather(1, action_indices.unsqueeze(1)).squeeze(1)
        main_loss = F.binary_cross_entropy_with_logits(selected, rewards)
        all_probs = torch.sigmoid(logits)
        loss = main_loss - 0.0001 * all_probs[:, :5].mean() - 0.00001 * all_probs[:, 5:].mean()
        loss.backward()
        s.optimizer.step()

    def _reset_sg_for_level(s):
        s.experience_buffer.clear()
        s.experience_hashes.clear()
        s.action_model = ActionModel(input_channels=s.num_colours, grid_size=s.grid_size).to(s.device)
        s.optimizer = optim.Adam(s.action_model.parameters(), lr=0.0001)
        s.prev_frame = None
        s.prev_action_idx = None

    def _lvl(s, f):
        return getattr(f, "score", None) or f.levels_completed

    def is_done(s, frames, lf):
        return (lf.state is GameState.WIN
                or (time.time() - s.start_time) >= 8 * 3600 - 300)

    def choose_action(s, frames, lf):
        try:
            lvl = s._lvl(lf)
            if lvl != s.cl:
                if not s._bfs_tried:
                    s._bfs_tried = True
                    s._init_bfs()
                s._bfs_solution = None
                s._bfs_step = 0
                if s._bfs:
                    s._try_bfs_solve(lvl)
                s._reset_sg_for_level()
                s.cl = lvl
                s.current_score = lvl

            if lf.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
                s.prev_frame = None
                s.prev_action_idx = None
                a = GameAction.RESET
                a.reasoning = "reset"
                return a

            if s._bfs_solution and s._bfs_step < len(s._bfs_solution):
                act_id, data = s._bfs_solution[s._bfs_step]
                s._bfs_step += 1
                sel = GameAction.from_id(act_id)
                if data:
                    sel.set_data(data)
                sel.reasoning = "bfs:" + str(s._bfs_step) + "/" + str(len(s._bfs_solution))
                return sel

            cur_tensor = s._frame_to_tensor(lf)
            cur_np = cur_tensor.cpu().numpy().astype(bool)
            if s.prev_frame is not None and s.prev_action_idx is not None:
                eh = s._experience_hash(s.prev_frame, s.prev_action_idx)
                if eh not in s.experience_hashes:
                    frame_changed = not np.array_equal(s.prev_frame, cur_np)
                    s.experience_buffer.append({
                        "state": s.prev_frame, "action_idx": s.prev_action_idx,
                        "reward": 1.0 if frame_changed else 0.0})
                    s.experience_hashes.add(eh)
            avail = getattr(lf, "available_actions", None) or []
            with torch.no_grad():
                logits = s.action_model(cur_tensor.unsqueeze(0)).squeeze(0)
            aidx, coords = s._sample_sg(logits, avail)
            if aidx < 5:
                sel = s.action_list[aidx]
                sel.reasoning = "sg:a" + str(aidx + 1)
                unified_idx = aidx
            else:
                sel = GameAction.ACTION6
                y, x = coords
                sel.set_data({"x": int(x), "y": int(y)})
                sel.reasoning = "sg:click(" + str(x) + "," + str(y) + ")"
                unified_idx = 5 + (y * s.grid_size + x)
            s.prev_frame = cur_np
            s.prev_action_idx = unified_idx
            if s.action_counter % s.train_frequency == 0:
                s._train_sg()
            return sel
        except Exception as e:
            traceback.print_exc()
            a = random.choice(s.action_list)
            a.reasoning = "err:" + str(e)[:40]
            return a
