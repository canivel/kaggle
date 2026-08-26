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
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, List, Optional, Set, Tuple
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

class CBAM(nn.Module):
    def __init__(s, ch, r=16):
        super().__init__()
        s.fc1 = nn.Linear(ch, max(ch // r, 4))
        s.fc2 = nn.Linear(max(ch // r, 4), ch)
        s.sp = nn.Conv2d(2, 1, 7, padding=3)

    def forward(s, x):
        B, C, H, W = x.shape
        w = torch.sigmoid(s.fc2(F.relu(s.fc1(x.mean(dim=[2, 3])))))
        x = x * w.view(B, C, 1, 1)
        a = torch.sigmoid(s.sp(torch.cat(
            [x.max(1, keepdim=True)[0], x.mean(1, keepdim=True)], 1)))
        return x * a


class ActionEffectAttention(nn.Module):
    def __init__(s, feat_dim=64, mem_dim=32, n_actions=5):
        super().__init__()
        s.mem_dim = mem_dim
        s.diff_enc = nn.Sequential(
            nn.Conv2d(1, 8, 8, stride=8), nn.ReLU(),
            nn.Conv2d(8, 16, 4, stride=4), nn.ReLU(),
            nn.Flatten(), nn.Linear(16 * 2 * 2, mem_dim))
        s.q_proj = nn.Linear(feat_dim, mem_dim)
        s.v_proj = nn.Linear(mem_dim + 1 + n_actions, n_actions)
        s.scale = mem_dim ** 0.5

    def forward(s, cnn_feat, mem_diffs, mem_actions, mem_rewards):
        B, M = mem_actions.shape
        if M == 0:
            return torch.zeros(B, 5, device=cnn_feat.device)
        keys = s.diff_enc(mem_diffs.reshape(B * M, 1, 64, 64)).reshape(B, M, s.mem_dim)
        q = s.q_proj(cnn_feat).unsqueeze(1)
        attn = F.softmax(torch.bmm(q, keys.transpose(1, 2)) / s.scale, dim=-1)
        act_oh = F.one_hot(mem_actions.clamp(0, 4), 5).float()
        vals = torch.cat([keys, mem_rewards.unsqueeze(-1), act_oh], dim=-1)
        ctx = torch.bmm(attn, vals).squeeze(1)
        return s.v_proj(ctx)


class ForgeNet(nn.Module):
    def __init__(s, in_ch=26, g=64):
        super().__init__()
        s.g = g
        s.c1 = nn.Conv2d(in_ch, 32, 3, padding=1)
        s.c2 = nn.Conv2d(32, 64, 3, padding=1)
        s.c3 = nn.Conv2d(64, 128, 3, padding=1)
        s.c4 = nn.Conv2d(128, 256, 3, padding=1)
        s.attn = CBAM(256)
        s.ar = nn.Conv2d(256, 64, 1)
        s.ap = nn.MaxPool2d(4, 4)
        s.af = nn.Linear(64 * 16 * 16, 256)
        s.ah = nn.Linear(256, 5)
        s.dr = nn.Dropout(0.15)
        s.cc1 = nn.Conv2d(256, 128, 3, padding=1)
        s.cc2 = nn.Conv2d(128, 64, 3, padding=1)
        s.cc3 = nn.Conv2d(64, 32, 1)
        s.cc4 = nn.Conv2d(32, 1, 1)
        s.gp = nn.AdaptiveAvgPool2d(1)
        s.gf = nn.Linear(256, 64)
        s.aea = ActionEffectAttention(feat_dim=64, mem_dim=32, n_actions=5)

    def forward(s, x, mem_diffs=None, mem_actions=None, mem_rewards=None):
        x = F.relu(s.c1(x))
        x = F.relu(s.c2(x))
        x = F.relu(s.c3(x))
        f = F.relu(s.c4(x))
        f = s.attn(f)
        af = F.relu(s.ar(f))
        af = s.ap(af).reshape(f.size(0), -1)
        al = s.ah(s.dr(F.relu(s.af(af))))
        cf = F.relu(s.cc1(f))
        cf = F.relu(s.cc2(cf))
        cf = F.relu(s.cc3(cf))
        cl = s.cc4(cf).reshape(f.size(0), -1)
        if mem_diffs is not None and mem_actions is not None:
            gf = s.gf(s.gp(f).reshape(f.size(0), -1))
            al = al + s.aea(gf, mem_diffs, mem_actions, mem_rewards)
        return torch.cat([al, cl], 1)


def fast_objects(frame, bg):
    objs = []
    for c in range(16):
        if c == bg:
            continue
        mask = (frame == c)
        npix = int(np.sum(mask))
        if npix < 4 or npix > 3000:
            continue
        ys, xs = np.where(mask)
        objs.append((c, float(np.mean(xs)), float(np.mean(ys)), npix))
    return objs


# ==================== AGENT ====================

# ===== v40: GraphExplorer + FrameProcessor (from v24b faithful port) =====
INFINITY = np.iinfo(np.int32).max

edge_dtype = np.dtype([
    ("group", "i4"),
    ("result", "i4"),
    ("target", "U32"),
    ("distance", "i4"),
    ("errors", "i4"),
])

@dataclass
class NodeInfo:
    name: Hashable

    total_candidates: int # how many exist
    num_groups: int = 1 # FIXME: is never used
    active_group: int = 0

    group2remaining_candidate_ids: List[Set[int]] = field(default_factory=list)

    edge_data: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=edge_dtype))

    error_threshold: int = 3
    closed: bool = False # flips when last probe done
    distance: float | None = 0 # TODO: how is it initialized?

    def __post_init__(self):

        assert self.name is not None, "Node name must be provided"

        if self.num_groups > 1 and self.group2remaining_candidate_ids is None:
            raise ValueError("group2remaining_candidate_ids must be provided if num_groups > 1")

        if self.num_groups == 1 and self.group2remaining_candidate_ids is None:
            self.group2remaining_candidate_ids = [set(range(self.total_candidates))]

        self.group2remaining_candidate_ids = [set(r_c_ids) for r_c_ids in self.group2remaining_candidate_ids] # ensure it's a list of sets

        self.edge_data = np.zeros(self.total_candidates, dtype=edge_dtype)
            
        for group_id, remaining_candidate_ids in enumerate(self.group2remaining_candidate_ids):
            self.edge_data["group"][list(remaining_candidate_ids)] = group_id

    @property
    def has_open(self) -> bool:
        """Still hiding ≥1 untested edge?"""
        return len(self.tested) < self.total_candidates

    def record_test(self, edge_idx: int, success: int, target_node: Hashable | None = None) -> bool:

        edge_group_id = self.edge_data[edge_idx]["group"]

        assert self.edge_data["result"][edge_idx] == 0 and \
            self.edge_data["target"][edge_idx] == "" and \
            self.edge_data["distance"][edge_idx] == 0, \
            "Edge result must be untested before recording a test"

        if success == -1:
            self.edge_data["errors"][edge_idx] += 1
            if self.edge_data["errors"][edge_idx] >= self.error_threshold:
                self.edge_data["errors"][edge_idx] = 0
                new_group_id = edge_group_id + 1
                if new_group_id > self.num_groups - 1:
                    # count it as failed and move on
                    self.group2remaining_candidate_ids[edge_group_id].discard(edge_idx)
                    self.edge_data["result"][edge_idx] = -1
                    self.edge_data["distance"][edge_idx] = INFINITY
                    return True
                else:
                    self.edge_data["group"][edge_idx] = new_group_id
                    self.group2remaining_candidate_ids[new_group_id].add(edge_idx)
                    self.group2remaining_candidate_ids[edge_group_id].discard(edge_idx)
            return False

        self.group2remaining_candidate_ids[edge_group_id].discard(edge_idx)

        if success == 1:
            self.edge_data["target"][edge_idx] = str(target_node)
            self.edge_data["distance"][edge_idx] = -1 # NOTE: distance is maintained by the GraphExplorer class
            self.edge_data["result"][edge_idx] = 1
        elif success == 0:
            self.edge_data["distance"][edge_idx] = INFINITY
            self.edge_data["result"][edge_idx] = -1

        return True

    def has_open_group(self, group_id: int) -> bool:
        """Return True if this node has at least one untested edge belonging to *group_id* or below."""
        for i in range(group_id+1):
            if len(self.group2remaining_candidate_ids[i]) > 0:
                return True
        return False
    
    def __repr__(self) -> str:
        edge_data_repr = format_struct_table(self.edge_data)

        return f"""NodeInfo:
name={self.name},
total_candidates={self.total_candidates},
num_groups={self.num_groups},
distance={self.distance},
closed={self.closed},
{edge_data_repr}
"""


class GraphExplorer:

    def __init__(
        self,
        start_node: Hashable | None = None, 
        num_candidates: int | None = None, 
        group2remaining_candidate_ids: List[Set[int]] | None = None,
        n_groups: int = 1,
        verbose_level: int = 0,
        ) -> None:

        self._verbose_level = verbose_level
        self._n_groups = max(1, n_groups)

        self.reset()

    def reset(self) -> None:
        self._nodes: Dict[Hashable, NodeInfo] = {}
        self._G: Dict[Hashable, Set[Tuple[int, Hashable]]] = defaultdict(set) # (edge_idx, target_node)
        self._G_rev: Dict[Hashable, Set[Tuple[int, Hashable]]] = defaultdict(set) # (edge_idx, source_node)
        self._frontier: Set[Hashable] = set()
        self._dist: Dict[Hashable, int] = {}
        self._next: Dict[Hashable, Tuple[int, Hashable]] = {} # (edge_idx, target_node)
        self._active_group: int = 0  # current priority group

        self.suspicious_transitions: Dict[Tuple[Hashable, int, Hashable], int] = {} # (source_node, edge_idx, target_node) -> count
        self.suspicious_transitions_threshold: int = 3

        self._empty = True
    
    def initialize(self, start_node: Hashable | None = None, num_candidates: int | None = None, group2remaining_candidate_ids: List[Set[int]] | None = None) -> None:


        if start_node is not None:
            self._add_new_node(start_node, num_candidates, group2remaining_candidate_ids=group2remaining_candidate_ids)

        if self._verbose_level >= 1:
            print(f"\nGraph is initialized with node: {self._nodes[start_node]}")
            self.dump()

    def record_test(
        self,
        node: Hashable,
        edge_idx: Hashable,
        success: bool,
        target_node: Optional[Hashable] = None,
        target_num_candidates: Optional[int] = None,
        group2remaining_candidate_ids: Optional[List[Set[int]]] = None,
        suspicious_transition: bool = False,
    ) -> None:

        if node not in self._nodes:
            raise KeyError(f"unknown node {node!r}") # TODO: alternatively, add it to the graph
        node_info = self._nodes[node]

        if node_info.closed:
            if target_node == self._nodes[node].edge_data["target"][edge_idx]:
                if self._verbose_level >= 1:
                    print(f"Node {node!r} is closed, skipping test {edge_idx!r}")
                return
            else:
                if self._verbose_level >= 1:
                    print(f"Node {node!r} is closed, we perform the test only if the target node is closer to frontier than the original target node. It will allow to fix the broken transition.")
                dist_to_frontier = self._dist.get(target_node, 0) # 0 if it wasn't previously recorded (so it's in the frontier)
                prev_target_node = self._nodes[node].edge_data["target"][edge_idx]
                prev_dist_to_frontier = self._dist.get(prev_target_node, INFINITY)

                if dist_to_frontier < prev_dist_to_frontier:
                    if self._verbose_level >= 1:
                        print(f"Target node {target_node!r} is closer to frontier than the original target node {prev_target_node!r}, we perform the test")
                else:
                    if self._verbose_level >= 1:
                        print(f"Target node {target_node!r} is further from frontier than the original target node {prev_target_node!r}, we skip the test")
                    return

        # store metadata immediately
        if self._verbose_level >= 1:
            print(f"Recording action {edge_idx} from {node} to {target_node} with success {success}")

        if suspicious_transition:
            self.suspicious_transitions[(node, edge_idx, target_node)] = self.suspicious_transitions.get((node, edge_idx, target_node), 0) + 1
            print(f"Suspicious transition detected: {node, edge_idx, target_node}, count: {self.suspicious_transitions[(node, edge_idx, target_node)]}")

            if self.suspicious_transitions[(node, edge_idx, target_node)] < self.suspicious_transitions_threshold:
                print(f"It will be ignored for now, but will be allowed after {self.suspicious_transitions_threshold} attempts")
                return
            else:
                print(f"Transition is recorded as permanent")
        
        node_info.record_test(edge_idx, success, target_node)
        
        # successful hop ⇒ register edge and maybe discover a brand-new node
        if success == 1:
            if target_node is None:
                raise ValueError("target_node required when success=True")

            if target_node not in self._nodes:
                new_node = True
                if target_num_candidates is None:
                    raise ValueError(
                        "target_num_candidates required for a new node"
                    )
                self._add_new_node(target_node, target_num_candidates, group2remaining_candidate_ids=group2remaining_candidate_ids)
            else:
                new_node = False


            self._G[node].add((edge_idx, target_node))
            self._G_rev[target_node].add((edge_idx, node))

            if not self._nodes[node].has_open_group(self.active_group):
                self._close_node(node)

            if self._nodes[target_node].has_open_group(self.active_group):
                # self._tighten_from_new_source(target_node)
                self._rebuild_distances()
            else:
                self._close_node(target_node)
                self._maybe_advance_group(target_node)

        else:
            if not self._nodes[node].has_open_group(self.active_group):
                self._close_node(node)
                self._maybe_advance_group(node)

        if self._verbose_level >= 1:
            if success == 1:
                success_str = "succeeded"
            elif success == -1:
                success_str = "threw an error"
            else:
                success_str = "failed"

            print(f"\n\nNode {node!r} candidate {edge_idx!r} {success_str}:")
            print(f"Source node:\n{self._nodes[node]}")
            if success == 1:
                print(f"{'NEW' if new_node else 'Existing'} target node:\n{self._nodes[target_node]}")
        self.dump()

    def get_distance(self, node: Hashable) -> Optional[int]:
        d = self._dist.get(node)
        return None if d is None or d == float("inf") else d

    def get_next_hop(self, node: Hashable) -> Optional[Hashable]:
        # NOTE: DEPRECATED
        # Return the node itself only if it truly has open edges in the active group
        if node in self._frontier: # and self._nodes[node].has_open_group(self.active_group):
            return node
        nxt = self._next.get(node)
        if nxt is None:
            return None
        # _next may store (edge_idx, next_node); return the node only
        if isinstance(nxt, tuple) and len(nxt) == 2:
            return nxt[1]
        return nxt

    def edge_info(self, node: Hashable, edge_idx: Hashable) -> np.ndarray:
        return self._nodes[node].edge_data[edge_idx]

    def is_finished(self) -> bool:
        return not self._frontier

    @property
    def active_group(self) -> int:
        return self._active_group
    
    @property
    def empty(self) -> bool:
        return self._empty

    def _add_new_node(self, node: Hashable, 
        n_candidates: int, 
        group2remaining_candidate_ids: Optional[List[Set[int]]] = None
        ) -> None:

        if n_candidates < 1:
            raise ValueError("num_candidates must be positive")

        self._nodes[node] = NodeInfo(node, n_candidates, self._n_groups, group2remaining_candidate_ids=group2remaining_candidate_ids)
        self._G[node] = set()
        self._G_rev[node] = set()

        if self._empty:
            self._empty = False

        if self._nodes[node].has_open_group(self.active_group):
            self._frontier.add(node)
        else:
            self._close_node(node)
            self._maybe_advance_group(node)


    def _close_node(self, node: Hashable) -> None:
        node_info = self._nodes[node]
        if node_info.closed:
            return
        node_info.closed = True
        self._frontier.discard(node)
        self._rebuild_distances() # removal from frontier may increase some distances in the graph

    def _tighten_from_new_source(self, src: Hashable) -> None:
        # NOTE: is not used anymore
        dq = deque([src])
        self._dist[src] = 0
        self._nodes[src].distance = 0
        while dq:
            v = dq.popleft()
            v_dist = self._dist.get(v, INFINITY)
            for edge_idx, u in self._G_rev.get(v, ()):  # (edge_idx, source_node)
                initial_u_dist = self._dist.get(u, INFINITY)
                u_edge_data = self._nodes[u].edge_data
                u_edge_data["distance"][edge_idx] = self._nodes[v].distance + 1
                updated_u_dist = u_edge_data["distance"][u_edge_data["group"] <= self.active_group].min()
                self._nodes[u].distance = updated_u_dist
                self._dist[u] = updated_u_dist
                if updated_u_dist > initial_u_dist:
                    dq.append(u)

    def _rebuild_distances(self) -> None:
        """
        Rebuild the distances from the frontier nodes in the graph.
        """
        self._dist.clear()
        self._next.clear()
        dq = deque(self._frontier)
        for node, node_info in self._nodes.items():
            node_info.distance = INFINITY
            self._dist[node] = INFINITY
        for src in self._frontier:
            self._nodes[src].distance = 0
            self._dist[src] = 0
        while dq:
            v = dq.popleft()
            v_dist = self._dist.get(v, INFINITY)
            for edge_idx, u in self._G_rev.get(v, ()):  # (edge_idx, source_node)
                u_info = self._nodes[u]
                u_dist = self._dist.get(u, INFINITY)
                u_info.edge_data["distance"][edge_idx] = v_dist + 1
                if u_dist > u_info.edge_data["distance"][edge_idx]:
                    u_info.distance = u_info.edge_data["distance"][edge_idx]
                    self._dist[u] = u_info.edge_data["distance"][edge_idx]
                    self._next[u] = (edge_idx, v)
                    dq.append(u)

    def _maybe_advance_group(self, current_node: Hashable) -> None:
        """
        If it's not possible to reach any frontier node from the current node,
        given the current active group, advance to the next higher group id and rebuild distances.
        """

        distance = self._nodes[current_node].distance
        while distance == INFINITY and self.active_group < self._n_groups - 1:
            print(f"Node {current_node!r} is not reachable from any frontier node under {self.active_group}, advancing to the next group")

            self._active_group += 1
            self._dist.clear()
            self._next.clear()
            self._frontier.clear()

            for node, node_info in self._nodes.items():
                node_info.active_group = self.active_group
                if node_info.has_open_group(self.active_group):
                    self._frontier.add(node)
                    node_info.closed = False

            self._rebuild_distances()
            distance = self._dist.get(current_node)
        
    def dump(self) -> None:
        if self._verbose_level >= 1:
            print("=== explorer state ===")
            print("frontier :", self._frontier)
            print("N nodes  :", len(self._nodes))
            print("N edged candidates  :", sum(len(node_info.edge_data) for node_info in self._nodes.values()))
            if self._verbose_level >= 2:
                print("Graph    :", self._G)
                print("dist     :", self._dist)
                print("next hop :", self._next)
            print("======================")

    def print_all_nodes(self) -> None:
        for node_info in self._nodes.values():
            print(node_info)

    def choose_edge(self, node: Hashable, return_reasoning: bool = False) -> Hashable:
        # TODO: make it possible to choose completely random edge
        node_info = self._nodes[node]
        if node_info.has_open_group(self.active_group):
            untested_edges = []
            for group_id in range(self.active_group + 1):
                untested_edges.extend(node_info.group2remaining_candidate_ids[group_id])
            if not untested_edges:
                raise ValueError("No untested edges in the current group while the group is open")

            edge_idx = random.choice(untested_edges)
            reasoning = f"Randomly chose untested edge {edge_idx} from group {self.active_group} with {node_info.group2remaining_candidate_ids} group2candidates\n"
        else:
            lowest_dist = node_info.distance
            print(f"Lowest dist: {lowest_dist}")
            # print(f"Node info: {node_info}")
            edges_with_lowest_dist = [edge_idx for edge_idx, edge_data in enumerate(node_info.edge_data) if edge_data["distance"] <= lowest_dist and edge_data["result"] == 1 and edge_data["group"] <= self.active_group]
            edge_idx = random.choice(edges_with_lowest_dist)
            reasoning = f"Chose edge {edge_idx} with lowest dist {lowest_dist}\n"

        reasoning += f"Node info: {node_info}\n"
        

        if return_reasoning:
            return edge_idx, reasoning
        else:
            return edge_idx



def _generate_random_grid(rows: int, cols: int, density: float = 0.7, seed: int | None = None) -> np.ndarray:
    """
    Return a boolean numpy array of shape *(rows, cols)* where **True** denotes
    a traversable cell (graph node) and **False** denotes an empty/wall cell.
    The *density* parameter controls the probability of a cell being present.
    """

    rng = np.random.default_rng(seed)
    grid = rng.random((rows, cols)) < density

    # Safety: ensure at least one node exists so that we have a valid start.
    if not grid.any():
        # Force the central cell to be traversable.
        grid[rows // 2, cols // 2] = True

    return grid


# Direction vectors indexed 0-3  (U, R, D, L)
_DIRS = {
    0: (-1, 0),  # up
    1: (0, 1),   # right
    2: (1, 0),   # down
    3: (0, -1),  # left
}


def _visualize_grid(grid: np.ndarray, explorer: "GraphExplorer", start_node: tuple[int, int]) -> None:
    """
    Pretty-print the current knowledge stored inside *explorer* on top of the
    underlying *grid*.

    Legend:
        "#"  wall / empty cell
        "?"  traversable cell but undiscovered yet
        "o"  discovered & closed node (all edges tested)
        "F"  frontier node (still holds untested candidates)
        "S"  the start node
    """

    rows, cols = grid.shape
    lines: list[str] = []
    for r in range(rows):
        row_chars: list[str] = []
        for c in range(cols):
            cell = (r, c)
            if not grid[r, c]:
                row_chars.append("#")
                continue

            if cell == start_node:
                row_chars.append("S")
            elif cell in explorer._frontier:
                row_chars.append("F")
            elif cell in explorer._nodes:
                row_chars.append("o")
            else:
                row_chars.append("?")
        lines.append(" ".join(row_chars))

    print("\nCurrent explorer view:")
    print("\n".join(lines))
    print()

def _plot_grid(
    grid: np.ndarray,
    explorer: "GraphExplorer",
    start_node: tuple[int, int],
    last_node: tuple[int, int] | None = None,
    last_edge: tuple[tuple[int, int], int] | None = None,  # (node_coords, edge_idx)
    log_text: str | None = None,
    *,
    figsize: tuple[int, int] | None = None,
    frames: list[np.ndarray] | None = None,
    group_colors: dict[int, str] | None = None,
    n_groups: int = 1,
) -> None:
    """
    Render *grid* with matplotlib showing explorer's knowledge so far.

    - Walls - nothing drawn (white)
    - Undiscovered traversable cells - light grey dots
    - Discovered nodes - blue dots, frontier in orange, start in gold
    - Arrows:
        - Success (edge exists)  - green
        - Failed probe           - red
        - Untested candidate     - grey (thin)
    """

    if n_groups > 1 and group_colors is None:
        default_palette = plt.get_cmap("tab10")
        group_colors = {grp: default_palette(grp % 10) for grp in range(n_groups)}

    rows, cols = grid.shape

    if figsize is None:
        figsize = (max(4, cols), max(4, rows))

    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches(*figsize)
    ax = fig.gca()

    ax.set_aspect("equal")
    # Grid lines
    ax.set_xticks(np.arange(-0.5, cols, 1))
    ax.set_yticks(np.arange(-0.5, rows, 1))
    ax.grid(True, which="both", color="lightgrey", linewidth=0.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    for r in range(rows):
        for c in range(cols):
            # rectangle lower-left corner at (c-0.5, r-0.5)
            facecolor = "black"  # default for walls
            if grid[r, c]:
                cell = (r, c)
                if cell == last_node:
                    facecolor = "blue"
                elif cell in explorer._frontier:
                    facecolor = "green"
                elif cell in explorer._nodes:
                    facecolor = "white"
                else:
                    facecolor = "grey"

            rect = Rectangle(
                (c - 0.5, r - 0.5),
                1,
                1,
                facecolor=facecolor,
                edgecolor="lightgrey",
                linewidth=0.5,
                alpha=0.6,
                zorder=0,
            )
            ax.add_patch(rect)

    # Overlay start marker
    ax.plot(start_node[1], start_node[0], marker="*", color="gold", markersize=12, zorder=4)

    # Draw arrows for each explored node
    for (r, c), info in explorer._nodes.items():
        for edge_idx in range(4):
            dr, dc = _DIRS[edge_idx]

            # Convert to plotting vector (remember inverted y later). Use dy = dr to correct flipped arrow issue.
            dx, dy = dc, dr

            # Decide arrow color & style with fixed length
            length_scale = 0.4  # stays inside cell borders
            succ_flag = False  # will stay False for untested or failed edges

            res = info.edge_data["result"][edge_idx] if edge_idx < len(info.edge_data) else 0
            if res != 0:
                succ_flag = (res == 1)

                # Highlight the very last tested edge in black
                if last_edge is not None and last_edge == ((r, c), edge_idx):
                    color = "black"
                    alpha = 1.0
                    lw = 2.5
                else:
                    color = "green" if succ_flag else "red"  # success green, failed red
                    alpha = 0.9
                    lw = 1.8
            else:
                group_id = int(info.edge_data["group"][edge_idx]) if edge_idx < len(info.edge_data) else 0
                color = group_colors.get(group_id, "grey") if group_colors else "grey"
                alpha = 0.8
                lw = 1.2

            arr = ax.arrow(
                c,
                r,
                dx * length_scale,
                dy * length_scale,
                head_width=0.15,
                head_length=0.15,
                fc=color,
                ec=color,
                alpha=alpha,
                linewidth=lw,
                length_includes_head=True,
                zorder=1,
            )

            # Annotate distance to frontier for successful edges
            if succ_flag:
                # Look up target from explorer graph if exists; otherwise skip distance annotation
                target = None
                for e_idx, tgt in explorer._G.get((r, c), set()):
                    if e_idx == edge_idx:
                        target = tgt
                        break
                if target is not None:
                    dist_val = explorer.get_distance(target)
                    dist_val_txt = "∞" if dist_val is None else str(dist_val)

                    text_x = c + dx * length_scale * 0.5
                    text_y = r + dy * length_scale * 0.5
                    ax.text(text_x, text_y, dist_val_txt, color="black", fontsize=8, ha="center", va="center", zorder=4)

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.invert_yaxis()
    plt.tight_layout()

    # Add log text overlay
    if log_text is not None:
        fig.text(0.02, 0.98, log_text, fontsize=9, va='top', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    legend_elements = [
        Patch(facecolor="black", edgecolor="lightgrey", label="Wall"),
        Patch(facecolor="grey", edgecolor="lightgrey", label="Unknown node"),
        Patch(facecolor="white", edgecolor="lightgrey", label="Discovered node"),
        Patch(facecolor="green", edgecolor="lightgrey", label="Frontier node"),
        Patch(facecolor="blue", edgecolor="lightgrey", label="Current node"),
        Patch(facecolor="gold", edgecolor="lightgrey", label="Start node"),
        Line2D([0], [0], color="black", lw=2, label="Last tested edge"),
        Line2D([0], [0], color="green", lw=2, label="Successful edge"),
        Line2D([0], [0], color="red", lw=2, label="Failed edge"),
        Line2D([0], [0], color="grey", lw=2, label="Untested candidate"),
    ]

    # Add candidate group colors to legend
    if n_groups > 1:
        for gid in range(n_groups):
            col = group_colors.get(gid, plt.get_cmap("tab10")(gid % 10)) if group_colors else plt.get_cmap("tab10")(gid % 10)
            legend_elements.append(Line2D([0], [0], color=col, lw=2, label=f"Candidate group {gid}"))

    # Reserve more space on the right for legend
    plt.subplots_adjust(right=0.65)

    # Place legend outside, based on figure coords for consistent layout
    legend_obj = fig.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(0.68, 0.5),
        bbox_transform=fig.transFigure,
        fontsize=7,
        framealpha=0.9,
    )

    # Optionally increase z-order so legend overlays anything else
    legend_obj.set_zorder(10)

    plt.draw()
    plt.pause(0.001)

    # Capture frame for gif if requested
    if frames is not None:
        canvas = fig.canvas
        canvas.draw()
        w, h = canvas.get_width_height()
        if hasattr(canvas, "tostring_rgb"):
            buf = canvas.tostring_rgb()
            channels = 3
        elif hasattr(canvas, "tostring_argb"):
            buf = canvas.tostring_argb()
            channels = 4
        else:
            raise RuntimeError("Canvas does not support RGB extraction")

        # Account for HiDPI / retina scaling: actual buffer may be larger than (w*h*channels)
        total_px = len(buf) // channels
        scale = int(round((total_px / (w * h)) ** 0.5))
        w_scaled, h_scaled = w * scale, h * scale

        img = np.frombuffer(buf, dtype=np.uint8).reshape(h_scaled, w_scaled, channels)
        if channels == 4:
            # ARGB -> RGB
            img = img[:, :, [1, 2, 3]]
        frames.append(img.copy())


def run_grid_demo(
    rows: int = 6,
    cols: int = 6,
    density: float = 0.7,
    seed: int | None = None,
    step_sleep: float | None = None,
    n_groups: int = 1,
    group_colors: dict[int, str] | None = None,
    plot: bool = True,
    save_gif: bool = True,
    gif_path: str = "exploration.gif",
    error_chance: float = 0.3,
) -> None:
    """
    Drive *GraphExplorer* over a random grid-world and visualize every step.

    - *rows*, *cols*         - grid dimensions
    - *density*              - probability that a cell contains a node
    - *seed*                 - RNG seed for reproducibility (``None`` ⇒ random)
    - *step_sleep*           - optional ``time.sleep`` delay after each step
    """

    import time

    grid = _generate_random_grid(rows, cols, density, seed)

    # Pick a random starting node
    node_coords = list(zip(*np.where(grid)))
    start_node = random.choice(node_coords)

    candidate2group = {i: random.randint(0, n_groups-1) for i in range(4)}

    print(f"Starting exploration at {start_node} on a {rows}x{cols} grid (density={density:.2f})\n")

    gx = GraphExplorer(n_groups=n_groups, verbose_level=2)
    print(f"candidate2group: {candidate2group}\n")
    gx.initialize(start_node=start_node, num_candidates=4, group2remaining_candidate_ids=[{i for i, g in candidate2group.items() if g == gid} for gid in range(n_groups)])

    frames: list[np.ndarray] = [] if plot and save_gif else []

    if plot:
        plt.ion()

    step_counter = 0
    _visualize_grid(grid, gx, start_node)
    if plot:
        _plot_grid(grid, gx, start_node, last_node=start_node, last_edge=None, log_text=f"Group NA | Moved to {start_node}", frames=frames if save_gif else None, n_groups=n_groups, group_colors=group_colors)

        gx.dump()

    current_node = start_node
    while not gx.is_finished():
        node_info = gx._nodes[current_node]

        # If current node is exhausted, travel along the shortest path to the frontier.
        if not node_info.has_open_group(gx.active_group):
            next_hop = gx.get_next_hop(current_node)
            if next_hop is None:
                print(f"Node {current_node} is exhausted and no path to frontier. Finishing.")
                break

            # Guard against degenerate self-looping next-hop
            if next_hop == current_node:
                gx._close_node(current_node)
                gx._maybe_advance_group(current_node)
                next_hop = gx.get_next_hop(current_node)
                if next_hop is None or next_hop == current_node:
                    print(f"Node {current_node} is exhausted and stuck. Finishing.")
                    break

            print(f"Node {current_node} exhausted. Traveling to {next_hop} towards nearest frontier.")
            step_counter += 1
            current_node = next_hop

            # If we arrived at a node that is not open (due to group constraints), try advancing group
            if not gx._nodes[current_node].has_open_group(gx.active_group):
                gx._maybe_advance_group(current_node)

            _visualize_grid(grid, gx, start_node)
            if plot:
                _plot_grid(
                    grid, gx, start_node,
                    last_node=current_node,
                    last_edge=None,
                    log_text=f"Group {gx.active_group} | travel",  
                    frames=frames if save_gif else None,
                    n_groups=n_groups, group_colors=group_colors,
                )

                gx.dump()
                gx.print_all_nodes()
            if step_sleep is not None:
                time.sleep(step_sleep)
                continue

        # We are at a node with open edges. Try them until success.
        group_id = gx.active_group
        prioritized_edges = []
        for gid in range(0, group_id + 1):
            prioritized_edges.extend(list(node_info.group2remaining_candidate_ids[gid]))

        moved = False
        for edge_idx in prioritized_edges:
            step_counter += 1

            dr, dc = _DIRS[edge_idx]
            neigh = (current_node[0] + dr, current_node[1] + dc)

            is_success = 0 <= neigh[0] < rows and 0 <= neigh[1] < cols and grid[neigh]

            if error_chance > random.random():
                result_code = -1
            else:
                result_code = 1 if is_success else 0

            # Record test result
            outcome_str = "fail"
            if result_code == 1:
                outcome_str = "success"
                target_group2remaining_candidate_ids = [set() for _ in range(n_groups)]
                for i in range(4):
                    gid = random.randint(0, n_groups - 1)
                    target_group2remaining_candidate_ids[gid].add(i)
                gx.record_test(current_node, edge_idx, 1, neigh, 4, group2remaining_candidate_ids=target_group2remaining_candidate_ids)
            elif result_code == 0:
                gx.record_test(current_node, edge_idx, 0)
            else:  # result_code == -1
                outcome_str = "error"
                gx.record_test(current_node, edge_idx, -1)

            print(f"Step {step_counter}: at {current_node} tested edge {edge_idx} → {outcome_str}")

            edge_group_id = int(node_info.edge_data["group"][edge_idx]) if edge_idx < len(node_info.edge_data) else 0
            cur_dist = gx.get_distance(current_node)
            dist_txt = "∞" if cur_dist is None else str(cur_dist)
            log_line = (
                f"group={gx.active_group} node={current_node} (dist {dist_txt}) | "
                f"edge {edge_idx} (grp {edge_group_id}) → {outcome_str}"
            )
            _visualize_grid(grid, gx, start_node)
            if plot:
                _plot_grid(
                    grid, gx, start_node,
                    last_node=current_node,
                    last_edge=((current_node), edge_idx),
                    log_text=log_line,
                    frames=frames if save_gif else None,
                    n_groups=n_groups, group_colors=group_colors,
                )

                gx.dump()
                gx.print_all_nodes()
            if step_sleep is not None:
                time.sleep(step_sleep)

            # Update agent position based on outcome
            if result_code == 1:
                current_node = neigh
                moved = True
                break
            elif result_code == -1:
                print(f"Probe error at {current_node}! Returning to start node {start_node}.")
                current_node = start_node
                moved = True
                break

        if not moved:
            # All available edges were tried and failed/errored.
            # Next loop iteration will trigger the travel-to-frontier logic.
            pass

    print("Exploration finished – every node is closed and no frontier remains.")
    if plot:
        # Final frame with no current node highlight
        _plot_grid(
            grid,
            gx,
            start_node,
            last_node=None,
            frames=frames if save_gif else None,
            n_groups=n_groups,
            group_colors=group_colors,
        )

        # Keep the final plot open for the user until they close the figure.
        plt.ioff()

        if save_gif and frames:
            print(f"Saving cropped GIF with {len(frames)} frames to {gif_path} …")

            from PIL import Image, ImageChops

            pil_frames = [Image.fromarray(frame) for frame in frames]

            # Compute union bounding box of non-white areas across frames
            bbox_union = None
            white_bg = Image.new("RGB", pil_frames[0].size, (255, 255, 255))
            for im in pil_frames:
                diff = ImageChops.difference(im, white_bg)
                bbox = diff.getbbox()
                if bbox is None:
                    continue
                if bbox_union is None:
                    bbox_union = bbox
                else:
                    l1, t1, r1, b1 = bbox_union
                    l2, t2, r2, b2 = bbox
                    bbox_union = (min(l1, l2), min(t1, t2), max(r1, r2), max(b1, b2))

            # Fallback to full image if bbox detection failed
            if bbox_union is None:
                bbox_union = (0, 0) + pil_frames[0].size

            cropped_frames = [im.crop(bbox_union) for im in pil_frames]

            # Save using Pillow directly
            cropped_frames[0].save(
                gif_path,
                save_all=True,
                append_images=cropped_frames[1:],
                duration=500,
                loop=0,
            )

        plt.show()


if __name__ == "__main__":

    print("\n========== SIMPLE TEST ==========")
    gx = GraphExplorer(verbose_level=2)
        
    gx.initialize("A", 2) # node A has 2 candidates

    gx.record_test("A", 0, -1) # simulate error
    gx.record_test("A", 0, -1) # simulate error
    gx.record_test("A", 0, -1) # simulate error

    # gx.record_test("A", 0, True,  "B", 3)   # throws an error

    gx.record_test("A", 1, True, "B", 3) # now A is closed automatically


    gx.record_test("B", 0, False)
    gx.record_test("B", 1, True,  "C", 1) # discovers C 
    gx.record_test("B", 2, False) # B becomes closed

    gx.record_test("C", 0, True, "D", 4)

    gx.print_all_nodes()


    print("\n========== TEST WITH GROUPS ==========")
    gx = GraphExplorer(n_groups=3, verbose_level=2)
    gx.initialize("A", 4, group2remaining_candidate_ids=[[0, 1], [2], [3]])

    gx.record_test("A", 0, False)
    gx.record_test("A", 1, False)

    gx.record_test("A", 2, True, "B", 3, group2remaining_candidate_ids=[[0], [2], [1]])

    gx.record_test("B", 0, True, "A")

    gx.record_test("A", 2, True, "B")

    gx.record_test("B", 2, False)

    gx.print_all_nodes()


    print("\n========== GRID WORLD DEMO ==========")
    group_colors = {0: "purple", 1: "orange", 2: "grey"}
    run_grid_demo(rows=6, cols=6, density=0.7, seed=12345, step_sleep=None, plot=True, save_gif=True, gif_path="grid_exploration.gif", n_groups=3, group_colors=group_colors, error_chance=0.1)


class FrameProcessor:
    OFFSETS4: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
    OFFSETS8: tuple[tuple[int, int], ...] = ((-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1))

    def __init__(self):
        self.connectivity_rank = 4
        self.status_bar_mode = "rule"
        self.status_bar_distance_threshold = 3
        self.status_bar_ratio_threshold = 5
        self.status_bar_twins_threshold = 3
        self.frame_shape = (64, 64)

        self.status_bar_color = 16
        self.minimal_width = 2
        self.maximal_width = 32
        self.non_salient_color = set([0,1,2,3,4,5])
        self.salient_color = set([6,7,8,9,10,11,12,13,14,15])

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        pass

    def segment_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """
        Segment `frame` into {self.connectivity_rank}-connected components (same color).

        NOTE: the twins identification increases complexity of the algorithm to O(n^2)

        Returns
        -------
        list[dict]
            One dict per component with keys
            - bounding_box : (x1, y1, x2, y2)   # inclusive pixel coords
            - color        : int                # original greyscale value
            - area         : int                # pixel count
            - is_rectangle : bool               # fully fills its bounding box
            - number_of_twins : int             # number of other components considered twins
            - twin_ids     : list[int]          # ids (1-based) of those twins
                NOTE: here we don't check shapes of the twins thoroughly

        """

        h, w = frame.shape
        label_map = np.zeros((h, w), dtype=int) - 1 # -1 = unvisited
        components: list[dict] = []
        cid = -1                                          # component id counter

        offsets = self.OFFSETS4 if self.connectivity_rank == 4 else self.OFFSETS8

        # --- first pass: flood-fill each blob ---------------------------------
        for y in range(h):
            for x in range(w):
                if label_map[y, x] != -1:                      # already labelled
                    continue
                cid += 1
                color = int(frame[y, x])
                q = deque([(y, x)])
                label_map[y, x] = cid

                min_x = max_x = x
                min_y = max_y = y
                area = 0

                while q:                                 # BFS
                    cy, cx = q.popleft()
                    area += 1
                    min_x, max_x = min(min_x, cx), max(max_x, cx)
                    min_y, max_y = min(min_y, cy), max(max_y, cy)

                    for dy, dx in offsets:
                        ny, nx = cy + dy, cx + dx
                        if (
                            0 <= ny < h and 0 <= nx < w
                            and label_map[ny, nx] == -1 # not visited
                            and frame[ny, nx] == color
                        ):
                            label_map[ny, nx] = cid
                            q.append((ny, nx))

                # rectangle test
                rect_area = (max_x - min_x + 1) * (max_y - min_y + 1)
                is_rect = area == rect_area

                components.append(
                    dict(
                        bounding_box=(min_x, min_y, max_x, max_y),
                        color=color,
                        area=area,
                        is_rectangle=is_rect,
                    )
                )

        # --- second pass: identify twins --------------------------------------
        # here: simple rule → same area, same rectangle status, and same color
        for i, comp in enumerate(components):
            twins = [
                j
                for j, other in enumerate(components)
                if i != j # skip self
                and other["area"] == comp["area"]
                and other["is_rectangle"] == comp["is_rectangle"]
                and other["color"] == comp["color"]
            ]
            comp["number_of_twins"] = len(twins)
            comp["twin_ids"] = twins

        return label_map, components

    def identify_status_bars(self, segmented_frame: np.ndarray, frame_segments: list[dict]) -> tuple[list[list[dict]] | None, np.ndarray]:
        """
        Identify the status bars from the frame segments
        Return a list of dictionaries and a frame mask.
        The list of dictionaries is the same as the input list of dictionaries in frame_segments, but with "id" key added.
        The frame mask is a binary mask where the status bars are 1 and the rest are 0.
        """
        if self.status_bar_mode == "crude":
            status_bar_mask = self.identify_status_bars_crude()
            status_bar_segments_list = None
        elif self.status_bar_mode == "rule" or self.status_bar_mode == "move":
            status_bar_segments_list, status_bar_mask = self.identify_status_bars_with_rule(segmented_frame, frame_segments)
            if self.status_bar_mode == "move":
                raise NotImplementedError("'move' mode is not implemented yet")
        else:
            raise ValueError(f"Invalid status bar mode: {self.status_bar_mode}")
        return status_bar_segments_list, status_bar_mask

    def identify_status_bars_crude(self) -> np.ndarray:
        status_bar_mask = np.zeros(self.frame_shape)
        status_bar_mask[:self.status_bar_distance_threshold, :] = 1
        status_bar_mask[-self.status_bar_distance_threshold:, :] = 1
        status_bar_mask[:, :self.status_bar_distance_threshold] = 1
        status_bar_mask[:, -self.status_bar_distance_threshold:] = 1
        return status_bar_mask
       
    def identify_status_bars_with_rule(self, segmented_frame: np.ndarray, frame_segments: list[dict]) -> tuple[list[list[dict]], np.ndarray]:
        """
        Identify the status bars from the frame segments
        Return a list of dictionaries and a frame mask.
        The list of dictionaries is the same as the input list of dictionaries in frame_segments, but with "id" key added.
        The frame mask is a binary mask where the status bars are 1 and the rest are 0.
        """

        # modes:
            # crude: remove all screen edges 
            # rule: rule-based
            # move: rule-based + movement after the first action 


        # the rules are:
            # the status bars are close to the edges of the screen
            # they can be in any orientation
            # the can be duplicated from both sides of the screen
            # there are 2 types of status bars:
                # 1. the line 
                # 2. the dots, for the dots there should be at least 3 twins


        checked_segment_ids = set()
        status_bar_segment_ids_list = [] # list[list[int]]
        for i, segment in enumerate(frame_segments):

            status_bar_segment_ids = [i]

            if i in checked_segment_ids:
                continue
            checked_segment_ids.add(i)
            on_edge_list = self.check_segment_fully_on_edge(segment, edges=['any'])
            if len(on_edge_list) == 0:
                continue
            directions = []
            if 'left' in on_edge_list or 'right' in on_edge_list:
                directions.append('vertical')
            if 'top' in on_edge_list or 'bottom' in on_edge_list:
                directions.append('horizontal')
            if len(directions) == 2:
                direction = 'any'
            else:
                direction = directions[0]
            is_long_ratio = self.check_segment_ratio(segment, direction=direction)  

            if not is_long_ratio:
                twin_ids_on_edge_list = self.segment_twins_on_edge(segment, frame_segments)
                for twin_id in twin_ids_on_edge_list:
                    checked_segment_ids.add(twin_id)
                if len(twin_ids_on_edge_list) + 1 < self.status_bar_twins_threshold:
                    continue
                status_bar_segment_ids.extend(twin_ids_on_edge_list)

            status_bar_segment_ids_list.append(status_bar_segment_ids)

        status_bar_segments_list = []
        status_bar_mask = np.zeros(segmented_frame.shape, dtype=bool)

        for i, status_bar_segment_ids in enumerate(status_bar_segment_ids_list):
            status_bar_segments = []
            for status_bar_segment_id in status_bar_segment_ids:
                status_bar_mask[segmented_frame == status_bar_segment_id] = 1

                status_bar_segments.append(frame_segments[status_bar_segment_id])
            status_bar_segments_list.append(status_bar_segments)

        return status_bar_segments_list, status_bar_mask

    def check_segment_fully_on_edge(self, segment: dict, edges: list[str] | None = None) -> list[str]:
        """
        Check if the segment is fully on the edge of the screen
        """
        x1, y1, x2, y2 = segment["bounding_box"]
        if edges is None:
            edges = ['any']
        for edge in edges:
            assert edge in ['any', 'left', 'right', 'top', 'bottom']

        result = []

        if 'left' in edges or 'any' in edges:
            max_x = max(x1, x2)
            if max_x < self.status_bar_distance_threshold:
                result.append('left')
        if 'right' in edges or 'any' in edges:
            min_x = min(x1, x2)
            if min_x > self.frame_shape[1] - self.status_bar_distance_threshold:
                result.append('right')
        if 'top' in edges or 'any' in edges:
            max_y = max(y1, y2)
            if max_y < self.status_bar_distance_threshold:
                result.append('top')
        if 'bottom' in edges or 'any' in edges:
            min_y = min(y1, y2)
            if min_y > self.frame_shape[0] - self.status_bar_distance_threshold:
                result.append('bottom')
        # NOTE: there can be some mess with the y-axis direction (should it start from the top or the bottom), need to double check
        return result

    def check_segment_ratio(self, segment: dict, direction: str | None = None) -> bool:
        """
        Check if the segment is a status bar
        """
        if direction is None:
            direction = 'any'
        assert direction in ['any', 'horizontal', 'vertical']

        x_length, y_length = segment["bounding_box"][2] - segment["bounding_box"][0] + 1, segment["bounding_box"][3] - segment["bounding_box"][1] + 1
        x_to_y_ratio = x_length / y_length
        if x_to_y_ratio >= self.status_bar_ratio_threshold and direction in ('any', 'horizontal'):
            return True
        if x_to_y_ratio <= 1 / self.status_bar_ratio_threshold and direction in ('any', 'vertical'):
            return True
        return False

    def segment_twins_on_edge(self, segment: dict, frame_segments: list[dict], edges: list[str] | None = None) -> list[int]:
        """
        Check if the segment has twins on the same edge
        """

        if edges is None:
            edges = self.check_segment_fully_on_edge(segment, edges=['any'])
            if len(edges) == 0:
                return []

        twins = []
        for twin_id in segment["twin_ids"]:
            twin = frame_segments[twin_id]
            twin_edges = self.check_segment_fully_on_edge(twin, edges=edges)
            if len(twin_edges) > 0:
                twins.append(twin_id)
        
        return twins
        
    def visualize_components(self, frame: np.ndarray, components: list[dict], *, cmap: str = "nipy_spectral",
                             save_path: str = "components.png", click_points: list[tuple[int, int]] | None = None
    ) -> None:
        """
        Show the frame with every connected component marked and
        print a short description for each one.

        Parameters
        ----------
        frame : np.ndarray
            The original HxW greyscale (label-value) image.
        components : list[dict]
            Output of `segment_frame()`.
        cmap : str, optional
            Matplotlib colour map for the background image.  Default is *nipy_spectral*.
        """
        if frame.ndim != 2:
            raise ValueError("`frame` must be a 2-D array")

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(frame, cmap=cmap, interpolation="nearest")
        ax.set_axis_off()

        # Plot bounding box + id at the centroid of each blob
        for idx, comp in enumerate(components, start=1):
            x1, y1, x2, y2 = comp["bounding_box"]
            w = x2 - x1 + 1
            h = y2 - y1 + 1

            # draw bounding box
            ax.add_patch(
                Rectangle(
                    (x1 - 0.5, y1 - 0.5),
                    w,
                    h,
                    edgecolor="white",
                    facecolor="none",
                    linewidth=1.2,
                )
            )

            # annotate with id number
            cx = x1 + w / 2.0
            cy = y1 + h / 2.0
            ax.text(
                cx,
                cy,
                str(idx),
                color="white",
                fontsize=8,
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.2", facecolor="black", alpha=0.6, lw=0
                ),
            )

        if click_points is not None:
            for x, y in click_points:
                ax.plot(x, y, 'ro')

        plt.tight_layout()
        plt.savefig(save_path)

        # ---------------------------------------------------------------------
        # Console description
        # ---------------------------------------------------------------------
        for idx, comp in enumerate(components, start=1):
            bb = comp["bounding_box"]
            print(
                f"Component {idx}: "
                f"colour={comp['color']:>2}, "
                f"area={comp['area']:>4}, "
                f"bbox=(x1={bb[0]}, y1={bb[1]}, x2={bb[2]}, y2={bb[3]}), "
                f"rect={comp['is_rectangle']}, "
                f"twins={comp['number_of_twins']} "
                f"{'('+','.join(map(str,comp['twin_ids']))+')' if comp['twin_ids'] else ''}"
            )
    
    def hash_frame(self, frame: np.ndarray) -> str:
        """
        Deterministic 128-bit hash for an integer-valued NumPy array whose
        elements are in the range 0 … 15 (4 bits).

        • Compact: packs two elements per byte before hashing  
        • Stable: identical digest across Python versions & interpreter restarts  
        • Shape-aware: (m, n) and (n, m) views do NOT collide  
        • Dependency-free: only stdlib hashlib
        """
        # TODO: maybe just convert a matrix to a number and store it
        frame = np.asarray(frame, dtype=np.uint8, order='C')

        # ---- pack two 4-bit values into each byte ---------------------------
        flat = frame.ravel()
        if flat.size & 1:                       # pad to even length
            flat = np.concatenate([flat, np.zeros(1, dtype=np.uint8)])
        packed = (flat[0::2] << 4) | (flat[1::2] & 0x0F)
        payload = packed.tobytes()

        # ---- hash with Blake2B (128-bit digest) -----------------------------
        shape_tag = frame.shape.__repr__().encode()
        return hashlib.blake2b(payload,
                            digest_size=16,   # 128 bits
                            person=shape_tag  # embeds the shape
                            ).hexdigest()


    def frame_segments_to_action_groups(self, frame_segments: list[dict], n_groups: int) -> list[list[int]]:
        """
        Assign actions to groups
        """
        group_0_segments = set()
        group_1_segments = set()
        group_2_segments = set()
        group_3_segments = set()
        group_4_segments = set()

        for segment_id, segment in enumerate(frame_segments):
            x_width, y_width = segment["bounding_box"][2] - segment["bounding_box"][0] + 1, segment["bounding_box"][3] - segment["bounding_box"][1] + 1
            is_salient = segment["color"] in self.salient_color
            is_medium_width = self.minimal_width <= x_width <= self.maximal_width and self.minimal_width <= y_width <= self.maximal_width
            is_status_bar = segment["color"] == self.status_bar_color

            assert n_groups == 5, "Only 5 groups are supported for now"

            if is_salient and is_medium_width:
                group_0_segments.add(segment_id)
            elif is_medium_width:
                group_1_segments.add(segment_id)
            elif is_salient:
                group_2_segments.add(segment_id)
            elif not is_status_bar:
                group_3_segments.add(segment_id)
            else:
                group_4_segments.add(segment_id)

        groups2segments = [group_0_segments, group_1_segments, group_2_segments, group_3_segments, group_4_segments]
        # groups2segments = groups2segments[::-1] # NOTE: temporary to check the robustness 

        return groups2segments



# FIXME: hash keyerror when level_up
# TODO: check how hash decision-making generally works

# TODO then: add some value propagation with transitions

# TODO: switch strategies on resets, e.g.:
# - random action selection
# - favor new actions


# TODO: for an action that resulted in a game over, save that it creates a transition, but the frame should be `0`. And then maybe treat it as a basic transition?
# Hmm, but the distance should be indified or set to constant?



class MyAgent(Agent):
    MAX_ACTIONS = float('inf')
    _MAX_FRAMES = 10

    def __init__(s, *a, **kw):
        super().__init__(*a, **kw)
        # v39 FIX: baseline used time.time()+builtin hash() (PYTHONHASHSEED-salted)
        # -> every submission a random draw (campaign-long LB-noise root cause).
        # Stable hashlib seed: reproducible, depends only on game identity.
        seed = int(hashlib.md5(str(s.game_id).encode()).hexdigest()[:8], 16)
        random.seed(seed)
        np.random.seed(seed % (2 ** 32 - 1))
        torch.manual_seed(seed % (2 ** 32 - 1))

        s.start_time = time.time()
        # v40: GraphExplorer + FrameProcessor for per-level GE-PRIMARY routing
        try:
            s._v40_fp = FrameProcessor()
            s._v40_ge = GraphExplorer(verbose_level=0, n_groups=5)
        except Exception:
            s._v40_fp = None; s._v40_ge = None
        s._v40_use_ge_level = False
        s._v40_ge_last_level = -1
        s._v40_ge_status_mask = None
        s._v40_ge_last_hash = None
        s._v40_ge_last_action_id = None

        s.device = torch.device(
            'cuda' if torch.cuda.is_available() else
            ('mps' if torch.backends.mps.is_available() else 'cpu'))
        s.G = 64
        s.IN = 26
        s.net = None
        s.opt = None

        # Replay buffer — prioritized (recent + high-reward weighted)
        s.buf = deque(maxlen=50000)
        s.buf_h = set()
        s.bsz = 64
        s.tfreq = 10

        # Per-step state tracking
        s.pt = None
        s.pai = None
        s.pr = None
        s.ph = None
        s.cl = -1
        s.fhist = deque(maxlen=6)
        s.la = 0

        s.al = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
                GameAction.ACTION4, GameAction.ACTION5]
        s._wd = False
        s._bg = 0
        s._wm = None

        # AEA memory buffers
        s._aem_diffs = deque(maxlen=256)
        s._aem_actions = deque(maxlen=256)
        s._aem_rewards = deque(maxlen=256)

        # Exploration state
        s._ckpt_hash = None
        s._unproductive = 0
        s._undo_avail = False
        s._eps = 0.15
        s._eps_min = 0.03
        s._eps_decay = 0.9997
        # FIX: properly initialize _visited_hashes (was causing reward bug in older versions)
        s._visited_hashes = set()

        # Object movement tracking (for dense rewards)
        s._prev_objs = None
        s._obj_moved = 0

        # BFS solver
        s._bfs = None
        s._bfs_solution = None
        s._bfs_step = 0
        s._bfs_tried = False
        s._bfs_solved_last = False  # FIX: track if BFS solved previous level
        s._clti_demos = []

        # Scanned actions for CNN click masking (from op_5)
        s._scanned_actions = None
        s._visit_counts = defaultdict(int)  # for novelty-guided exploration

    def append_frame(s, f):
        s.frames.append(f)
        if len(s.frames) > s._MAX_FRAMES:
            s.frames = s.frames[-s._MAX_FRAMES:]
        if f.guid:
            s.guid = f.guid
        if hasattr(s, "recorder") and not s.is_playback:
            import json
            s.recorder.record(json.loads(f.model_dump_json()))

    def _lvl(s, f):
        return getattr(f, 'score', None) or f.levels_completed

    def _raw(s, fd):
        return np.array(fd.frame, dtype=np.int64)[-1]

    def _init_bfs(s):
        src, cls = find_game_source_and_class(s.game_id,
                                              s.arc_env if hasattr(s, 'arc_env') else None)
        if src:
            s._bfs = BFSSolver(src, cls, scan_timeout=5, bfs_timeout=180)
            if s._bfs.load():
                logger.info(f"BFS: loaded {cls} from {src}")
            else:
                s._bfs = None
                logger.warning("BFS: failed to load game class")
        else:
            logger.warning(f"BFS: game source not found for {s.game_id}")

    def _capture_clti_demos(s, level_idx, sol):
        """Replay BFS solution and record (frame_before, action_idx, reward=2.0) tuples."""
        try:
            g, last_r = s._bfs._init_game_at_level(level_idx)
            if g is None or not last_r or not last_r.frame:
                return []
            demos = []
            for act_id, data in sol:
                frame_before = np.array(last_r.frame[-1], dtype=np.int64)
                if act_id <= 5:
                    pai = act_id - 1
                elif data and 'x' in data and 'y' in data:
                    pai = 5 + int(data['y']) * 64 + int(data['x'])
                else:
                    continue
                ai = (ActionInput(id=GameAction.from_id(act_id), data=data)
                      if data else ActionInput(id=GameAction.from_id(act_id)))
                last_r = g.perform_action(ai, raw=True)
                if not last_r.frame:
                    break
                demos.append({'s': frame_before, 'a': pai, 'r': 2.0})
            return demos
        except Exception as e:
            logger.warning(f"CLTI capture failed: {e}")
            return []

    def _try_bfs_solve(s, level_idx):
        if s._bfs is None:
            return None

        # Adaptive time budget: if BFS solved the previous level, give it more time
        elapsed = time.time() - s.start_time
        total_budget = 6 * 3600 - 600
        remaining = max(60, total_budget - elapsed)
        if level_idx == 0:
            time_for_bfs = min(remaining * 0.35, 1200)
        elif s._bfs_solved_last:
            time_for_bfs = min(remaining * 0.20, 480)
        else:
            time_for_bfs = min(remaining * 0.08, 180)
        time_for_bfs = max(30, time_for_bfs)
        s._bfs.bfs_timeout = int(time_for_bfs)
        logger.info(f"BFS L{level_idx}: budget={time_for_bfs:.0f}s "
                    f"(remaining={remaining:.0f}s)")

        prev_sol = s._bfs.solutions.get(level_idx - 1) if level_idx > 0 else None

        # Build goal heuristic from previous level solutions
        goal_heuristic = None
        if level_idx > 0 and s._bfs.game_cls is not None:
            try:
                g = s._bfs.game_cls()
                g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                last_r = g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                level_heuristics = []
                for pi in range(level_idx):
                    ps = s._bfs.solutions.get(pi)
                    if not ps:
                        break
                    f_init = np.array(last_r.frame[-1])
                    for act_id, data in ps:
                        ai = (ActionInput(id=GameAction.from_id(act_id), data=data)
                              if data else ActionInput(id=GameAction.from_id(act_id)))
                        last_r = g.perform_action(ai, raw=True)
                    f_win = np.array(last_r.frame[-1])
                    hfn = s._bfs._build_goal_heuristic(f_init, f_win)
                    level_heuristics.append((hfn, pi + 1))
                if level_heuristics:
                    total_w = sum(w for _, w in level_heuristics)
                    def goal_heuristic(f, game=None,
                                       _h=level_heuristics, _t=total_w):
                        return sum(hfn(f, game) * w for hfn, w in _h) / _t
            except Exception as e:
                logger.warning(f"BFS L{level_idx}: goal heuristic build failed: {e}")

        sol = s._bfs.solve_level(level_idx, prev_solution=prev_sol,
                                  goal_heuristic=goal_heuristic)
        if sol:
            s._bfs_solution = sol
            s._bfs_step = 0
            s._bfs_solved_last = True
            s._clti_demos = s._capture_clti_demos(level_idx, sol)
            return sol

        # Retry with distance heuristic if flat
        if (level_idx in s._bfs.timed_out_levels
                and s._bfs.game_cls is not None):
            try:
                g_val = s._bfs.game_cls()
                g_val.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                last_r_val = g_val.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                for pi in range(level_idx):
                    ps = s._bfs.solutions.get(pi)
                    if not ps:
                        break
                    for act_id, data in ps:
                        ai = (ActionInput(id=GameAction.from_id(act_id), data=data)
                              if data else ActionInput(id=GameAction.from_id(act_id)))
                        last_r_val = g_val.perform_action(ai, raw=True)
                mover_colors, target_colors = s._bfs._probe_mover_target_colors(g_val)
                if mover_colors and target_colors:
                    def dist_heuristic(f, game=None,
                                       _m=mover_colors, _t=target_colors):
                        centroids = {}
                        for c in range(16):
                            mask = (f == c)
                            n = int(np.sum(mask))
                            if n < 2:
                                continue
                            ys, xs = np.where(mask)
                            centroids[c] = (float(np.mean(xs)), float(np.mean(ys)))
                        targets = [(centroids[tc][0], centroids[tc][1])
                                   for tc in _t if tc in centroids]
                        if not targets:
                            return 0
                        return sum(
                            min(abs(centroids[mc][0] - tx) + abs(centroids[mc][1] - ty)
                                for tx, ty in targets)
                            for mc in _m if mc in centroids)
                    logger.info(f"BFS L{level_idx}: retrying with distance heuristic")
                    sol2 = s._bfs.solve_level(level_idx, prev_solution=prev_sol,
                                              goal_heuristic=dist_heuristic)
                    if sol2:
                        s._bfs_solution = sol2
                        s._bfs_step = 0
                        s._bfs_solved_last = True
                        s._clti_demos = s._capture_clti_demos(level_idx, sol2)
                        return sol2
            except Exception as e:
                logger.warning(f"BFS L{level_idx}: distance heuristic retry failed: {e}")

        s._bfs_solved_last = False
        return None

    def _tensor(s, fd):
        frame = s._raw(fd)
        oh = torch.zeros(16, 64, 64, dtype=torch.float32)
        oh.scatter_(0, torch.from_numpy(frame).unsqueeze(0), 1)
        cnt = np.bincount(frame.flatten(), minlength=16)
        s._bg = int(cnt.argmax())
        mx = max(cnt.max(), 1)
        bg_m = (frame == s._bg).astype(np.float32)
        rar = np.zeros((64, 64), np.float32)
        for c in range(16):
            if cnt[c] > 0:
                rar[frame == c] = 1.0 - cnt[c] / mx
        pad = np.pad(frame, 1, mode='edge')
        edge = ((frame != pad[:-2, 1:-1]) | (frame != pad[2:, 1:-1]) |
                (frame != pad[1:-1, :-2]) | (frame != pad[1:-1, 2:])).astype(np.float32)
        rp = np.linspace(0, 1, 64, dtype=np.float32).reshape(64, 1).repeat(64, 1)
        cp = np.linspace(0, 1, 64, dtype=np.float32).reshape(1, 64).repeat(64, 0)
        aug = torch.from_numpy(np.stack([bg_m, rar, edge, rp, cp]))
        d1 = torch.zeros(3, 64, 64, dtype=torch.float32)
        for i, prev in enumerate(reversed(list(s.fhist))):
            if i >= 3:
                break
            d1[i] = torch.from_numpy((frame != prev).astype(np.float32))
        d2 = torch.zeros(2, 64, 64, dtype=torch.float32)
        h = list(s.fhist)
        if len(h) >= 2:
            d2[0] = torch.from_numpy((h[-1] != h[-2]).astype(np.float32))
        if len(h) >= 4:
            d2[1] = torch.from_numpy((h[-2] != h[-4]).astype(np.float32))
        s.fhist.append(frame.copy())
        return torch.cat([oh, aug, d1, d2], 0).to(s.device)

    def _detect_template(s, frame):
        mask = torch.ones(4096, dtype=torch.float32)
        col_act = np.sum(frame != s._bg, axis=0)
        for c in range(20, 44):
            if (col_act[c] <= 2 and np.sum(col_act[:c] > 0) >= 5
                    and np.sum(col_act[c + 1:] > 0) >= 5):
                for y in range(64):
                    for x in range(c + 1):
                        mask[y * 64 + x] = 0.05
                return mask
        row_act = np.sum(frame != s._bg, axis=1)
        for r in range(20, 44):
            if (row_act[r] <= 2 and np.sum(row_act[:r] > 0) >= 5
                    and np.sum(row_act[r + 1:] > 0) >= 5):
                for y in range(r + 1):
                    for x in range(64):
                        mask[y * 64 + x] = 0.05
                return mask
        return mask

    def _reward(s, prev_raw, curr_raw, prev_h, curr_h):
        mask = np.ones((64, 64), dtype=bool)
        mask[:2] = False
        mask[62:] = False
        diff = (prev_raw != curr_raw) & mask
        changed = np.any(diff)
        r = 0.0
        if curr_h != prev_h:
            if curr_h not in s._visited_hashes:
                r += 1.5
                s._visited_hashes.add(curr_h)
            else:
                r += 0.2
        else:
            r -= 0.1
        if changed:
            r += 0.5
        curr_objs = fast_objects(curr_raw, s._bg)
        if s._prev_objs and curr_objs:
            moved = 0
            for co in curr_objs:
                for po in s._prev_objs:
                    if co[0] == po[0]:
                        dist = abs(co[1] - po[1]) + abs(co[2] - po[2])
                        if 2 < dist < 20:
                            moved += 1
                            break
            if moved > 0:
                r += 0.3 * min(moved, 3)
                s._obj_moved = moved
        s._prev_objs = curr_objs
        return r

    def _sample(s, logits, avail=None, temp=1.0):
        al = logits[:5].clone()
        cl = logits[5:5 + 4096].clone()
        if avail is not None and len(avail) > 0:
            mask_al = torch.full_like(al, float('-inf'))
            a6 = False
            for a in avail:
                aid = a.value if hasattr(a, 'value') else int(a)
                if 1 <= aid <= 5:
                    mask_al[aid - 1] = 0.0
                elif aid == 6:
                    a6 = True
            al = al + mask_al
            if not a6:
                cl = cl + torch.full_like(cl, float('-inf'))
        # Template masking
        if s._wm is not None:
            cl = cl + torch.log(s._wm.to(s.device).clamp(min=0.01))
        # Click masking: only predict positions we know are effective (from op_5)
        if s._scanned_actions is not None:
            click_mask = torch.full((4096,), -5.0, device=s.device)
            for act_id, data in s._scanned_actions:
                if act_id == 6 and data:
                    x, y = data.get('x', 0), data.get('y', 0)
                    if 0 <= x < 64 and 0 <= y < 64:
                        click_mask[y * 64 + x] = 0.0
            cl = cl + click_mask
        ap = torch.sigmoid(al / temp)
        cp = torch.sigmoid(cl / temp) / (s.G * s.G)
        allp = torch.cat([ap, cp])
        sm = allp.sum()
        if sm < 1e-8:
            allp = torch.ones_like(allp) / len(allp)
        else:
            allp = allp / sm
        idx = np.random.choice(len(allp), p=allp.cpu().numpy())
        if idx < 5:
            return idx, None
        ci = idx - 5
        return 5, (ci // s.G, ci % s.G)

    def _sample_novelty_guided(s, frame, avail):
        """Exploration: pick from scanned actions weighted by inverse visit count."""
        if not s._scanned_actions:
            return s._heuristic(frame, avail, s.la)
        scored = []
        for act_id, data in s._scanned_actions:
            if data:
                key = f"{act_id}:{data.get('x', 0)}:{data.get('y', 0)}"
            else:
                key = str(act_id)
            cnt = s._visit_counts[key]
            score = 1.0 / math.sqrt(cnt + 1)
            scored.append((score, act_id, data))
        scored.sort(reverse=True)
        probs = np.array([x[0] for x in scored], dtype=np.float64)
        probs = probs / probs.sum()
        idx = int(np.random.choice(len(scored), p=probs))
        _, act_id, data = scored[idx]
        key = (f"{act_id}:{data.get('x', 0)}:{data.get('y', 0)}"
               if data else str(act_id))
        s._visit_counts[key] += 1
        if act_id < 6:
            return act_id - 1, None
        return 5, (data['y'], data['x'])

    def _heuristic(s, frame, avail, step):
        av = set(int(a.value) if hasattr(a, 'value') else int(a) for a in avail)
        for d in [1, 2, 3, 4]:
            if d in av and step < 4:
                return d - 1, None
        if 6 in av:
            cnt = np.bincount(frame.flatten(), minlength=16)
            targets = []
            for c in range(16):
                if c == s._bg or cnt[c] == 0 or cnt[c] > 2000:
                    continue
                ys, xs = np.where(frame == c)
                if len(ys) >= 2:
                    targets.append((int(np.median(xs)), int(np.median(ys)), len(ys)))
            targets.sort(key=lambda t: t[2])
            pidx = step - 4
            if 0 <= pidx < len(targets):
                return 5, (targets[pidx][1], targets[pidx][0])
        if 5 in av:
            return 4, None
        choices = [a for a in av if 1 <= a <= 5]
        if choices:
            return random.choice(choices) - 1, None
        return 0, None

    def _frame_to_tensor(s, frame):
        oh = torch.zeros(16, 64, 64, dtype=torch.float32)
        oh.scatter_(0, torch.from_numpy(frame).unsqueeze(0), 1)
        cnt = np.bincount(frame.flatten(), minlength=16)
        bg = int(cnt.argmax())
        mx = max(cnt.max(), 1)
        bg_m = (frame == bg).astype(np.float32)
        rar = np.zeros((64, 64), np.float32)
        for c in range(16):
            if cnt[c] > 0:
                rar[frame == c] = 1.0 - cnt[c] / mx
        pad = np.pad(frame, 1, mode='edge')
        edge = ((frame != pad[:-2, 1:-1]) | (frame != pad[2:, 1:-1]) |
                (frame != pad[1:-1, :-2]) | (frame != pad[1:-1, 2:])).astype(np.float32)
        rp = np.linspace(0, 1, 64, dtype=np.float32).reshape(64, 1).repeat(64, 1)
        cp = np.linspace(0, 1, 64, dtype=np.float32).reshape(1, 64).repeat(64, 0)
        aug = torch.from_numpy(np.stack([bg_m, rar, edge, rp, cp]))
        zeros = torch.zeros(5, 64, 64, dtype=torch.float32)
        return torch.cat([oh, aug, zeros], 0)

    def _train(s):
        if len(s.buf) < s.bsz:
            return
        # Prioritized replay: weight recent + high-reward transitions more
        weights = np.array([abs(e['r']) + 0.1 for e in s.buf])
        n = len(weights)
        weights[max(0, n - 100):] *= 2.0
        weights /= weights.sum()
        indices = np.random.choice(n, s.bsz, replace=False, p=weights)
        batch = [s.buf[i] for i in indices]
        states = torch.stack(
            [s._frame_to_tensor(e['s']).to(s.device) for e in batch])
        acts = torch.tensor([e['a'] for e in batch],
                            dtype=torch.long, device=s.device)
        rews = torch.tensor([e['r'] for e in batch],
                            dtype=torch.float32, device=s.device)
        rews = torch.sigmoid(rews)
        s.opt.zero_grad()
        logits = s.net(states)
        acts_c = acts.clamp(0, logits.size(1) - 1)
        sel = logits.gather(1, acts_c.unsqueeze(1)).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(sel, rews)
        p = torch.sigmoid(logits)
        loss = loss - 0.0001 * p[:, :5].mean() - 0.00001 * p[:, 5:].mean()
        loss.backward()
        s.opt.step()

    def _get_aem_tensors(s):
        if len(s._aem_diffs) < 2:
            return None, None, None
        M = len(s._aem_diffs)
        diffs = torch.zeros(1, M, 1, 64, 64, device=s.device)
        acts = torch.zeros(1, M, dtype=torch.long, device=s.device)
        rews = torch.zeros(1, M, device=s.device)
        for i, (d, a, r) in enumerate(
                zip(s._aem_diffs, s._aem_actions, s._aem_rewards)):
            diffs[0, i, 0] = torch.from_numpy(d.astype(np.float32))
            acts[0, i] = min(a, 4)
            rews[0, i] = r
        return diffs, acts, rews

    def is_done(s, frames, lf):
        try:
            return (lf.state is GameState.WIN
                    or (time.time() - s.start_time) >= 6 * 3600 - 300)
        except:
            return True

    def _v40_ge_pick(s, lf, lvl):
        """v40: GE-PRIMARY policy on live env for BFS-failed levels.
        Same shape as v35's _ge_pick, ported onto v39's full-engine base.
        Returns a GameAction or None (None -> fall through to CNN)."""
        if s._v40_ge is None or s._v40_fp is None:
            return None
        if lvl != s._v40_ge_last_level:
            s._v40_ge.reset()
            s._v40_ge_status_mask = None
            s._v40_ge_last_hash = None
            s._v40_ge_last_action_id = None
            s._v40_ge_last_level = lvl
        frame_np = np.array(lf.frame, dtype=np.uint8)
        if frame_np.size == 0:
            return None
        num_frames = frame_np.shape[0]
        frame_np = frame_np[-1].copy()
        level_up = (getattr(s, '_v40_ge_status_mask', None) is None)
        if level_up:
            seg, segs = s._v40_fp.segment_frame(frame_np)
            try:
                _, mask = s._v40_fp.identify_status_bars(seg, segs)
            except Exception:
                mask = None
            s._v40_ge_status_mask = mask if mask is not None else np.zeros_like(frame_np, dtype=bool)
            s._v40_ge_last_hash = None
            s._v40_ge_last_action_id = None
        if s._v40_ge_status_mask is not None:
            frame_np[s._v40_ge_status_mask] = 16
        segmented_frame, frame_segments = s._v40_fp.segment_frame(frame_np)
        avail_raw = list(getattr(lf, 'available_actions', []) or [])
        avail = [a.value if hasattr(a, 'value') else int(a) for a in avail_raw]
        SIMPLE = {1: GameAction.ACTION1, 2: GameAction.ACTION2,
                  3: GameAction.ACTION3, 4: GameAction.ACTION4,
                  5: GameAction.ACTION5}
        num_click_actions = 0; num_actions = 0; arrow_actions = []
        if 6 in avail:
            num_click_actions = len(frame_segments); num_actions = num_click_actions
            action_groups = s._v40_fp.frame_segments_to_action_groups(frame_segments, n_groups=5)
        else:
            action_groups = [set() for _ in range(5)]
        for aid in avail:
            if aid in SIMPLE:
                arrow_actions.append(SIMPLE[aid])
                action_groups[0].add(num_actions); num_actions += 1
        if num_actions == 0:
            return None
        frame_np[frame_np == 16] = 0
        hashed_frame = s._v40_fp.hash_frame(frame_np)
        if level_up:
            s._v40_ge.initialize(start_node=hashed_frame, num_candidates=num_actions,
                                 group2remaining_candidate_ids=action_groups)
        if (not level_up) and s._v40_ge_last_hash is not None and s._v40_ge_last_action_id is not None:
            transition = hashed_frame != s._v40_ge_last_hash
            try:
                s._v40_ge.record_test(s._v40_ge_last_hash, s._v40_ge_last_action_id,
                                      int(transition), hashed_frame,
                                      target_num_candidates=num_actions,
                                      group2remaining_candidate_ids=action_groups,
                                      suspicious_transition=False)
            except Exception:
                s._v40_ge.reset()
                s._v40_ge.initialize(start_node=hashed_frame, num_candidates=num_actions,
                                     group2remaining_candidate_ids=action_groups)
        if hashed_frame not in s._v40_ge._nodes:
            s._v40_ge.reset()
            s._v40_ge.initialize(start_node=hashed_frame, num_candidates=num_actions,
                                 group2remaining_candidate_ids=action_groups)
        try:
            action_id = s._v40_ge.choose_edge(hashed_frame, return_reasoning=False)
            action_id = int(action_id) if not isinstance(action_id, tuple) else int(action_id[0])
        except Exception:
            return None
        if action_id < num_click_actions:
            seg = frame_segments[action_id]
            seg_mask = (segmented_frame == action_id)
            pts = np.argwhere(seg_mask)
            if len(pts) == 0:
                bbox = seg.get('bbox') or seg.get('bounding_box')
                if bbox:
                    ymin, xmin, ymax, xmax = bbox
                    y, x = (ymin + ymax) // 2, (xmin + xmax) // 2
                else:
                    y, x = 32, 32
            else:
                pt = pts[random.randint(0, len(pts) - 1)]
                y, x = int(pt[0]), int(pt[1])
            action = GameAction.ACTION6
            action.set_data({"x": int(x), "y": int(y)})
        else:
            action = arrow_actions[action_id - num_click_actions]
        s._v40_ge_last_hash = hashed_frame
        s._v40_ge_last_action_id = action_id
        return action

    def choose_action(s, frames, lf):
        try:
            lvl = s._lvl(lf)

            # ===== LEVEL CHANGE =====
            if lvl != s.cl:
                if not s._bfs_tried:
                    s._bfs_tried = True
                    s._init_bfs()

                # Save CLTI demos from previous level BEFORE running BFS for new level
                clti_to_inject = s._clti_demos
                s._clti_demos = []

                s._bfs_solution = None
                s._bfs_step = 0
                if s._bfs:
                    s._try_bfs_solve(lvl)
                # v40: BFS-fails-level -> route ENTIRE level through GraphExplorer
                # (v35 per-level routing applied to v39's full engine). Tomorrow's
                # v41 will switch this to GE-in-clone with minimal-path replay;
                # today's v40 establishes the routing infrastructure.
                s._v40_use_ge_level = (s._bfs_solution is None and s._v40_ge is not None)
                s._v40_ge_last_level = -1  # force per-level GE re-init

                # Scan actions for CNN click masking (from op_5)
                s._scanned_actions = None
                s._visit_counts = defaultdict(int)
                if s._bfs is not None and s._bfs.game_cls is not None:
                    try:
                        g_scan = s._bfs.game_cls()
                        g_scan.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                        g_scan.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                        raw_init = s._raw(lf)
                        bg = int(np.bincount(raw_init.flatten(), minlength=16).argmax())
                        s._scanned_actions = s._bfs._scan_actions(g_scan, raw_init, bg)
                        logger.info(f"CNN: scanned {len(s._scanned_actions)} effective actions")
                    except Exception as e:
                        logger.warning(f"CNN action scan failed: {e}")

                # Init CNN
                s.buf.clear()
                s.buf_h.clear()
                # CLTI: inject previous level's BFS expert demos
                for demo in clti_to_inject:
                    key = hashlib.md5(demo['s'].tobytes() + str(demo['a']).encode()).hexdigest()[:16]
                    if key not in s.buf_h:
                        s.buf.append(demo)
                        s.buf_h.add(key)
                if clti_to_inject:
                    logger.info(f"CLTI: injected {len(clti_to_inject)} expert demos for L{lvl}")
                s.net = ForgeNet(s.IN, s.G).to(s.device)
                for wp in ['/kaggle/input/forge-pretrained-weights/pretrained_weights.pt',
                           'pretrained_weights.pt']:
                    try:
                        if os.path.exists(wp):
                            state = torch.load(wp, map_location=s.device, weights_only=True)
                            ms = s.net.state_dict()
                            for k in list(state.keys()):
                                if k in ms and state[k].shape == ms[k].shape:
                                    ms[k] = state[k]
                            s.net.load_state_dict(ms)
                            break
                    except:
                        pass
                s.opt = optim.Adam(s.net.parameters(), lr=0.0003)
                s.pt = None
                s.pai = None
                s.pr = None
                s.ph = None
                s.cl = lvl
                s.fhist.clear()
                s.la = 0
                s._wd = False
                s._wm = None
                # FIX: only reset epsilon if BFS failed (don't waste good exploration)
                if not s._bfs_solved_last:
                    s._eps = 0.15
                s._aem_diffs.clear()
                s._aem_actions.clear()
                s._aem_rewards.clear()
                s._prev_objs = None
                s._obj_moved = 0
                s._ckpt_hash = None
                s._unproductive = 0
                s._visited_hashes = set()

            # ===== RESET =====
            if lf.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
                s.pt = None
                s.pai = None
                s.pr = None
                s.ph = None
                a = GameAction.RESET
                a.reasoning = "reset"
                return a

            # ===== BFS SOLUTION EXECUTION =====
            if s._bfs_solution and s._bfs_step < len(s._bfs_solution):
                act_id, data = s._bfs_solution[s._bfs_step]
                s._bfs_step += 1
                sel = GameAction.from_id(act_id)
                if data:
                    sel.set_data(data)
                sel.reasoning = f"bfs:{s._bfs_step}/{len(s._bfs_solution)}"
                raw = s._raw(lf)
                s.fhist.append(raw.copy())
                s.pr = raw.copy()
                s.la += 1
                return sel

            # ===== v40: GRAPH-EXPLORER PRIMARY (BFS-failed levels) =====
            if getattr(s, '_v40_use_ge_level', False):
                try:
                    ge_action = s._v40_ge_pick(lf, lvl)
                except Exception as _ge_e:
                    logger.warning(f"v40 GE-primary failed: {_ge_e}")
                    ge_action = None
                if ge_action is not None:
                    raw_ge = s._raw(lf)
                    s.fhist.append(raw_ge.copy())
                    s.pr = raw_ge.copy()
                    s.la += 1
                    return ge_action
                # GE returned None -> fall through to CNN for this tick

            # ===== CNN FALLBACK =====
            tensor = s._tensor(lf)
            raw = s._raw(lf)
            ch = hashlib.md5(raw.tobytes()).hexdigest()[:16]
            avail = getattr(lf, 'available_actions', None) or []
            s._undo_avail = any(
                (a.value if hasattr(a, 'value') else int(a)) == 7 for a in avail)

            if s.pt is not None and s.pai is not None:
                mask = np.ones((64, 64), dtype=bool)
                mask[:2] = False
                mask[62:] = False
                diff_map = (s.pr != raw) & mask
                changed = np.any(diff_map)
                eh = hashlib.md5(
                    s.pr.tobytes()[:1000] + str(s.pai).encode()).hexdigest()[:16]
                if eh not in s.buf_h:
                    r = s._reward(s.pr, raw, s.ph, ch)
                    s.buf.append({'s': s.pr.copy(), 'a': s.pai, 'r': r})
                    s.buf_h.add(eh)
                    if changed:
                        s._aem_diffs.append(diff_map)
                        s._aem_actions.append(min(s.pai, 4))
                        s._aem_rewards.append(r)
                if changed:
                    s._ckpt_hash = ch
                    s._unproductive = 0
                else:
                    s._unproductive += 1

            if s._wm is None:
                s._wm = s._detect_template(raw)

            if s._undo_avail and s._unproductive >= 30 and s._ckpt_hash:
                s._unproductive = 0
                a = GameAction.ACTION7
                a.reasoning = "undo"
                s.pt = tensor
                s.pai = 6
                s.pr = raw.copy()
                s.ph = ch
                s.la += 1
                return a

            if not s._wd:
                if s.la < 10:
                    # Novelty-guided exploration (from op_5)
                    aidx, coords = s._sample_novelty_guided(raw, avail)
                else:
                    s._wd = True
                    for _ in range(min(5, len(s.buf) // s.bsz)):
                        s._train()

            if s._wd:
                if random.random() < s._eps:
                    aidx, coords = s._sample_novelty_guided(raw, avail)
                else:
                    with torch.no_grad():
                        mem = s._get_aem_tensors()
                        if mem[0] is not None:
                            logits = s.net(tensor.unsqueeze(0), *mem).squeeze(0)
                        else:
                            logits = s.net(tensor.unsqueeze(0)).squeeze(0)
                    aidx, coords = s._sample(logits, avail, temp=0.5)
                s._eps = max(s._eps_min, s._eps * s._eps_decay)
            elif s.la >= 10:
                s._wd = True
                aidx, coords = 0, None

            if aidx < 5:
                sel = s.al[aidx]
                sel.reasoning = f"cnn:a{aidx + 1}"
            else:
                sel = GameAction.ACTION6
                y, x = coords
                sel.set_data({"x": int(x), "y": int(y)})
                sel.reasoning = f"cnn:c({x},{y})"

            s.pt = tensor
            s.pai = aidx if aidx < 5 else (5 + coords[0] * s.G + coords[1])
            s.pr = raw.copy()
            s.ph = ch
            s.la += 1
            if s.action_counter % s.tfreq == 0 and s._wd:
                s._train()
            return sel

        except Exception as e:
            traceback.print_exc()
            a = random.choice(s.al)
            a.reasoning = f"err:{str(e)[:40]}"
            return a