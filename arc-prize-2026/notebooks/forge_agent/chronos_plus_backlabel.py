# =====================================================================
# FORGE v85 — v84 + 3 fixes from REAL GAME testing
#
# Fixes based on actual failure analysis on 25 public games:
# 1. ANIMATION DRAIN: pump RESET after init until frame stabilizes
#    (fixes sc25, lf52 — 0 unique states from animation blocking)
# 2. FINE-SCAN + POST-SETUP: 2px click scan + re-scan after setup actions
#    (fixes su15, g50t — too few productive actions found)
# 3. IDDFS: iterative-deepening DFS instead of BFS for deeper solutions
#    (fixes tn36, dc22, ka59, r11l, sb26 — BFS too shallow)
# =====================================================================
import pickle
import copy
import glob
import hashlib
import importlib.util
import logging
import os
import random
import time
import traceback
from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState, ActionInput

logger = logging.getLogger(__name__)

# ==================== BFS SOLVER ====================

class BFSSolver:
    """Offline BFS solver using direct game class instantiation."""

    def __init__(self, game_path, game_class_name, scan_timeout=3, bfs_timeout=120):
        self.game_path = game_path
        self.class_name = game_class_name
        self.scan_timeout = scan_timeout
        self.bfs_timeout = bfs_timeout
        self.game_cls = None
        self.solutions = {}  # level_idx → action list
        self.cnn_action_scores = None  # v83: CNN logits for action prioritization

    def load(self):
        """Load the game class from source."""
        try:
            spec = importlib.util.spec_from_file_location('game_mod', self.game_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.game_cls = getattr(mod, self.class_name)
            return True
        except Exception as e:
            logger.warning(f"BFS: Failed to load game class: {e}")
            return False

    def _make_game(self, level_idx, drain=False):
        """v85: Create game instance. Optional animation drain for probing."""
        g = self.game_cls()
        g.set_level(level_idx)
        g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        if drain:
            # Only drain if first probe shows animation (frame changes on RESET)
            try:
                f_before = g.get_pixels(0, 0, 64, 64).copy()
                r = g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                if r and r.frame:
                    f_after = np.array(r.frame[-1])
                    if not np.array_equal(f_before[2:62], f_after[2:62]):
                        # Frame changed on RESET → animation active, drain it
                        prev_h = hashlib.md5(f_after[2:62].tobytes()).hexdigest()[:12]
                        for _ in range(40):
                            r2 = g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                            if not r2 or not r2.frame: break
                            h = hashlib.md5(np.array(r2.frame[-1])[2:62].tobytes()).hexdigest()[:12]
                            if h == prev_h: break
                            prev_h = h
            except:
                pass
        return g

    def _state_hash(self, g, frame, hidden_fields=None):
        """v10: Hash frame + discovered hidden scalar fields (fast)."""
        fh = hashlib.md5(frame.tobytes()).hexdigest()[:16]
        if hidden_fields:
            # Append hidden field values to hash — much faster than pickle(__dict__)
            extras = []
            for field_name in hidden_fields:
                try:
                    v = getattr(g, field_name, None)
                    if v is not None:
                        extras.append(f"{field_name}={v}")
                except:
                    pass
            if extras:
                return fh + "|" + "|".join(extras)
        return fh

    def _probe_hidden_fields(self, game, actions):
        """v10: Dynamic state probing — discover which scalar fields change per action.
        Returns list of field names that are hidden state (change without pixel change)."""
        if not actions:
            return []
        # Get initial scalar snapshot
        initial = {}
        for k, v in game.__dict__.items():
            if isinstance(v, (int, float, bool)) and not k.startswith('__'):
                initial[k] = v

        # Try each action, see what scalars change
        changing_fields = set()
        frame0 = game.get_pixels(0, 0, 64, 64)
        for act_id, data in actions[:10]:  # probe first 10 actions
            g = copy.deepcopy(game)
            try:
                ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                g.perform_action(ai, raw=True)
            except:
                continue
            f = g.get_pixels(0, 0, 64, 64)
            pixels_changed = np.sum(frame0 != f) > 0
            for k, v in g.__dict__.items():
                if isinstance(v, (int, float, bool)) and not k.startswith('__'):
                    if k in initial and v != initial[k]:
                        # Field changed — is it hidden? (not reflected in pixels)
                        if k not in ('_action_count', '_full_reset', '_action_complete'):
                            changing_fields.add(k)

        # Filter: only keep fields that change WITHOUT pixel changes (truly hidden)
        # Also keep counters that might be win-relevant
        hidden = []
        for f in changing_fields:
            if f.startswith('_') and f not in ('_current_level_index', '_score'):
                continue
            hidden.append(f)
        return sorted(hidden)

    def _effect_signature(self, f0, f1):
        """Compute a structural effect signature: (diff_count, color_change_tuple).
        Two actions with the same signature do the same TYPE of thing."""
        diff_mask = (f0 != f1)
        n_diff = int(np.sum(diff_mask))
        if n_diff == 0:
            return None
        # Bin diff count into buckets: 1-4, 5-16, 17-64, 65-256, 257+
        bucket = 0 if n_diff <= 4 else (1 if n_diff <= 16 else (2 if n_diff <= 64 else (3 if n_diff <= 256 else 4)))
        # Color histogram of changed pixels (before→after pairs)
        old_colors = frozenset(f0[diff_mask].tolist())
        new_colors = frozenset(f1[diff_mask].tolist())
        return (bucket, old_colors, new_colors)

    def _get_solution_signatures(self, level_idx, prev_solution):
        """Replay prev_solution on its level, collecting effect signatures per step."""
        sigs = set()
        try:
            g = self.game_cls()
            g.set_level(level_idx - 1)
            g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            for act_id, data in prev_solution:
                f_before = g.get_pixels(0, 0, 64, 64)
                ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                r = g.perform_action(ai, raw=True)
                if r.frame:
                    f_after = np.array(r.frame[-1])
                    sig = self._effect_signature(f_before, f_after)
                    if sig:
                        sigs.add(sig)
        except:
            pass
        return sigs

    def _filter_actions_by_signature(self, game, f0, actions, target_sigs):
        """Keep only actions whose effect signature matches any target signature."""
        if not target_sigs:
            return actions
        filtered = []
        for act_id, data in actions:
            g = copy.deepcopy(game)
            try:
                ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                r = g.perform_action(ai, raw=True)
                if not r.frame:
                    continue
                f1 = np.array(r.frame[-1])
                sig = self._effect_signature(f0, f1)
                if sig and sig in target_sigs:
                    filtered.append((act_id, data))
            except:
                continue
        return filtered if filtered else actions  # fallback to all if no matches

    def _scan_actions(self, game, f0, bg):
        """Scan for effective actions. Returns list of (action_id, data)."""
        avail = game._available_actions
        actions = []
        # Directional/interact actions
        for a in [a for a in avail if a <= 5]:
            g = copy.deepcopy(game)
            try:
                r = g.perform_action(ActionInput(id=GameAction.from_id(a)), raw=True)
                if r.frame and np.sum(f0 != np.array(r.frame[-1])) > 0:
                    actions.append((a, None))
            except:
                pass
        # Click actions (proven v9 scan — don't change this)
        if 6 in avail:
            t0 = time.time()
            seen_effects = set()
            for y in range(0, 64, 2):
                if time.time() - t0 > self.scan_timeout:
                    break
                for x in range(0, 64, 2):
                    if f0[y, x] == bg:
                        continue
                    g = copy.deepcopy(game)
                    try:
                        r = g.perform_action(
                            ActionInput(id=GameAction.ACTION6, data={'x': x, 'y': y, 'game_id': 'bfs'}),
                            raw=True
                        )
                        if not r.frame:
                            continue
                        f = np.array(r.frame[-1])
                        diff = np.sum(f0 != f)
                        if diff > 0:
                            # v9: compress equivalent clicks (same effect = same action)
                            effect_hash = hashlib.md5(f.tobytes()).hexdigest()[:12]
                            if effect_hash not in seen_effects:
                                seen_effects.add(effect_hash)
                                actions.append((6, {'x': x, 'y': y, 'game_id': 'bfs'}))
                    except:
                        pass
        # v84: If ACTION6 is available but no clicks found effective,
        # add click positions for all non-bg pixel clusters anyway.
        # They might become effective after navigating to them.
        if 6 in avail and not any(a == 6 for a, _ in actions):
            for y in range(2, 62, 4):
                for x in range(2, 62, 4):
                    if f0[y, x] != bg:
                        actions.append((6, {'x': x, 'y': y, 'game_id': 'bfs'}))
            # Deduplicate by position
            seen_pos = set()
            deduped = []
            for a, d in actions:
                if a == 6 and d:
                    key = (d['x'], d['y'])
                    if key in seen_pos:
                        continue
                    seen_pos.add(key)
                deduped.append((a, d))
            actions = deduped

        # v85: Post-setup re-scan — find clicks that only work after a directional action
        dir_actions = [(a, d) for a, d in actions if a <= 5]
        if 6 in avail and len(dir_actions) >= 1 and len([a for a, _ in actions if a == 6]) < 3:
            for setup_a, setup_d in dir_actions[:2]:
                try:
                    g = self._make_game(level_idx)  # no drain for scan
                    ai = ActionInput(id=GameAction.from_id(setup_a), data=setup_d) if setup_d else ActionInput(id=GameAction.from_id(setup_a))
                    g.perform_action(ai, raw=True)
                    f_after = g.get_pixels(0, 0, 64, 64)
                    for y in range(1, 63, 4):
                        for x in range(1, 63, 4):
                            if f_after[y, x] == bg:
                                continue
                            g2 = copy.deepcopy(g)
                            try:
                                r = g2.perform_action(
                                    ActionInput(id=GameAction.ACTION6, data={'x': x, 'y': y, 'game_id': 'bfs'}),
                                    raw=True)
                                if r and r.frame:
                                    diff = np.sum(f_after != np.array(r.frame[-1]))
                                    if diff > 0:
                                        eh = hashlib.md5(np.array(r.frame[-1]).tobytes()).hexdigest()[:12]
                                        if eh not in seen_effects:
                                            seen_effects.add(eh)
                                            actions.append((6, {'x': x, 'y': y, 'game_id': 'bfs'}))
                            except:
                                pass
                except:
                    pass

        return actions

    def solve_level(self, level_idx, max_states=500000, prev_solution=None):
        """Find optimal solution for a level via BFS."""
        if not self.game_cls:
            return None

        # v85: Standard init (don't drain here — drain only in source_solve probing)
        game = self.game_cls()
        game.set_level(level_idx)
        game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        r0 = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        if not r0 or not r0.frame:
            return None
        f0 = np.array(r0.frame[-1])
        bg = int(np.bincount(f0.flatten(), minlength=16).argmax())

        # v9: Try solution transfer from previous level first
        if prev_solution and level_idx > 0:
            transfer_result = self._try_transfer(game, level_idx, prev_solution, f0)
            if transfer_result:
                return transfer_result

        # Phase 1: Scan for effective actions
        actions = self._scan_actions(game, f0, bg)
        logger.info(f"BFS L{level_idx}: {len(actions)} effective actions (after dedup)")
        if not actions:
            return None

        # v83: CNN-guided action reordering — put CNN-preferred actions first
        # This makes BFS explore the CNN's best guesses deeper
        if self.cnn_action_scores is not None and len(actions) > 4:
            try:
                act_logits = self.cnn_action_scores[:5]  # 5 directional logits
                click_logits = self.cnn_action_scores[5:]  # 4096 click logits
                scored = []
                for act_id, data in actions:
                    if act_id <= 5 and act_id >= 1:
                        score = float(act_logits[act_id - 1])
                    elif act_id == 6 and data and 'x' in data:
                        x, y = data['x'], data['y']
                        idx = y * 64 + x
                        score = float(click_logits[idx]) if idx < len(click_logits) else 0.0
                    else:
                        score = 0.0
                    scored.append((score, act_id, data))
                scored.sort(key=lambda x: -x[0])
                actions = [(a, d) for _, a, d in scored]
                logger.info(f"BFS L{level_idx}: CNN-reordered actions (top: {actions[0][0]})")
            except:
                pass  # CNN scoring failed, keep original order

        # v83: NARROW BFS PASS — top 4 actions, deep search (25% timeout)
        # This is what the 0.39 lucky run did accidentally. Now deliberate.
        if len(actions) > 6:
            # Include dirs + at least 1 click if available
            dirs = [a for a in actions if a[0] <= 5][:4]
            clicks = [a for a in actions if a[0] == 6][:2]
            narrow = (dirs + clicks)[:6]  # max 6 for narrow pass
            visited_n = set()
            queue_n = deque()
            game_n = self.game_cls()
            game_n.set_level(level_idx)
            game_n.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            r_n = game_n.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            if r_n and r_n.frame:
                f_n = np.array(r_n.frame[-1])
                h_n = self._state_hash(game_n, f_n, None)
                visited_n.add(h_n)
                queue_n.append((copy.deepcopy(game_n), [], 0))
                t_n = time.time()
                exp_n = 0
                while queue_n and exp_n < 100000 and (time.time() - t_n) < self.bfs_timeout * 0.25:
                    g, hist, depth = queue_n.popleft()
                    if depth >= 25:
                        continue
                    for act_id, data in narrow:
                        g2 = copy.deepcopy(g)
                        try:
                            ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                            r = g2.perform_action(ai, raw=True)
                        except:
                            continue
                        exp_n += 1
                        if not r.frame:
                            continue
                        f = np.array(r.frame[-1])
                        h = self._state_hash(g2, f, None)
                        if h in visited_n:
                            continue
                        visited_n.add(h)
                        new_hist = hist + [(act_id, data)]
                        if r.levels_completed > level_idx or g2._current_level_index > level_idx:
                            logger.info(f"BFS L{level_idx}: NARROW SOLVED {len(new_hist)} acts ({exp_n} explored)")
                            self.solutions[level_idx] = new_hist
                            return new_hist
                        queue_n.append((g2, new_hist, depth + 1))
                logger.info(f"BFS L{level_idx}: narrow done ({exp_n} explored, {len(visited_n)} unique)")

        # v85: Action-type filter for L1+ — only use action types from L0 solution
        # This is the single biggest multi-level improvement: L0 tells us WHICH
        # action types matter. L1+ BFS with those types is 3-10x narrower.
        if prev_solution and level_idx > 0:
            prev_action_types = set(a for a, _ in prev_solution)
            type_filtered = [act for act in actions if act[0] in prev_action_types]
            # Also keep click positions near L0's clicks (within 20px)
            if 6 in prev_action_types:
                prev_clicks = [(d['x'], d['y']) for a, d in prev_solution if a == 6 and d and 'x' in d]
                for act_id, data in actions:
                    if act_id == 6 and data and 'x' in data:
                        for px, py in prev_clicks:
                            if abs(data['x'] - px) < 20 and abs(data['y'] - py) < 20:
                                if (act_id, data) not in type_filtered:
                                    type_filtered.append((act_id, data))
                                break
            if len(type_filtered) >= 2:
                logger.info(f"BFS L{level_idx}: action-type filter {len(actions)}→{len(type_filtered)} "
                            f"(types from L{level_idx-1}: {sorted(prev_action_types)})")
                actions = type_filtered

        # Phase 1b: If L1+ and prev solution exists, try signature-filtered BFS first
        # This cuts branching factor from ~50 to ~8-12 by only keeping actions
        # whose structural effect matches what worked on L0
        if prev_solution and level_idx > 0 and len(actions) > 15:
            target_sigs = self._get_solution_signatures(level_idx, prev_solution)
            if target_sigs:
                game_filtered = self.game_cls()
                game_filtered.set_level(level_idx)
                game_filtered.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                game_filtered.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                f0_f = np.array(game_filtered.perform_action(ActionInput(id=GameAction.RESET), raw=True).frame[-1])
                filtered_actions = self._filter_actions_by_signature(game_filtered, f0_f, actions, target_sigs)
                if len(filtered_actions) < len(actions):
                    logger.info(f"BFS L{level_idx}: signature filter {len(actions)}→{len(filtered_actions)} actions ({len(target_sigs)} signatures)")
                    # Quick BFS with filtered actions (use 40% of timeout)
                    visited_f = set()
                    queue_f = deque()
                    game_f2 = self.game_cls()
                    game_f2.set_level(level_idx)
                    game_f2.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                    game_f2.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                    f0_f2 = np.array(game_f2.perform_action(ActionInput(id=GameAction.RESET), raw=True).frame[-1])
                    h0_f = self._state_hash(game_f2, f0_f2, None)
                    visited_f.add(h0_f)
                    queue_f.append((copy.deepcopy(game_f2), [], 0))
                    t0_f = time.time()
                    explored_f = 0
                    filtered_timeout = self.bfs_timeout * 0.4
                    while queue_f and explored_f < max_states and (time.time() - t0_f) < filtered_timeout:
                        g, hist, depth = queue_f.popleft()
                        for act_id, data in filtered_actions:
                            g2 = copy.deepcopy(g)
                            try:
                                ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                                r = g2.perform_action(ai, raw=True)
                            except: continue
                            explored_f += 1
                            if not r.frame: continue
                            f = np.array(r.frame[-1])
                            h = self._state_hash(g2, f, None)
                            if h in visited_f: continue
                            visited_f.add(h)
                            new_hist = hist + [(act_id, data)]
                            if r.levels_completed > level_idx or g2._current_level_index > level_idx:
                                logger.info(f"BFS L{level_idx}: SOLVED (sig-filtered) in {len(new_hist)} actions ({explored_f} explored, {time.time()-t0_f:.1f}s)")
                                self.solutions[level_idx] = new_hist
                                return new_hist
                            if depth < 60:
                                queue_f.append((g2, new_hist, depth + 1))
                    logger.info(f"BFS L{level_idx}: sig-filtered pass exhausted ({explored_f} explored, {len(visited_f)} unique, {time.time()-t0_f:.1f}s)")

        # v83: Test if pickle works (4-17x faster than deepcopy)
        use_pickle = False
        try:
            _tp = pickle.dumps(game, protocol=4)
            pickle.loads(_tp)
            use_pickle = True
        except:
            pass

        # Phase 2: BFS — first try with frame hash (fast, proven for 12/25)
        hidden_fields = None  # start without hidden fields
        visited = set()
        queue = deque()
        h0 = self._state_hash(game, f0, None)
        visited.add(h0)
        queue.append((copy.deepcopy(game), [], 0))

        t0 = time.time()
        explored = 0

        while queue and explored < max_states and (time.time() - t0) < self.bfs_timeout:
            g, hist, depth = queue.popleft()

            # v83: Pickle parent once, loads for each child
            parent_pkl = None
            if use_pickle:
                try:
                    parent_pkl = pickle.dumps(g, protocol=4)
                except:
                    pass

            for act_id, data in actions:
                g2 = pickle.loads(parent_pkl) if parent_pkl else copy.deepcopy(g)
                try:
                    ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                    r = g2.perform_action(ai, raw=True)
                except:
                    continue
                explored += 1

                if not r.frame:
                    continue
                f = np.array(r.frame[-1])
                h = self._state_hash(g2, f, hidden_fields if hidden_fields else None)
                if h in visited:
                    continue
                visited.add(h)

                new_hist = hist + [(act_id, data)]

                # Win detection
                if r.levels_completed > level_idx or g2._current_level_index > level_idx:
                    elapsed = time.time() - t0
                    logger.info(f"BFS L{level_idx}: SOLVED in {len(new_hist)} actions ({explored} explored, {elapsed:.1f}s)")
                    self.solutions[level_idx] = new_hist
                    return new_hist

                if depth < 60:
                    queue.append((g2, new_hist, depth + 1))

        elapsed_first = time.time() - t0
        logger.info(f"BFS L{level_idx}: first pass timeout ({explored} explored, {len(visited)} unique, {elapsed_first:.1f}s)")

        # v10: If too few unique states found → hidden state detected → retry with probed fields
        if len(visited) < 50 and elapsed_first < self.bfs_timeout * 0.8:
            hidden_fields = self._probe_hidden_fields(game, actions)
            if hidden_fields:
                logger.info(f"BFS L{level_idx}: RETRY with hidden fields: {hidden_fields}")
                visited2 = set()
                queue2 = deque()
                game2 = self.game_cls()
                game2.set_level(level_idx)
                game2.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                game2.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                f0_2 = np.array(game2.perform_action(ActionInput(id=GameAction.RESET), raw=True).frame[-1])
                h0_2 = self._state_hash(game2, f0_2, hidden_fields)
                visited2.add(h0_2)
                queue2.append((copy.deepcopy(game2), [], 0))
                t0_2 = time.time()
                explored2 = 0
                remaining = max(30, self.bfs_timeout - elapsed_first)
                while queue2 and explored2 < max_states and (time.time() - t0_2) < remaining:
                    g, hist, depth = queue2.popleft()
                    for act_id, data in actions:
                        g2 = copy.deepcopy(g)
                        try:
                            ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                            r = g2.perform_action(ai, raw=True)
                        except: continue
                        explored2 += 1
                        if not r.frame: continue
                        f = np.array(r.frame[-1])
                        h = self._state_hash(g2, f, hidden_fields)
                        if h in visited2: continue
                        visited2.add(h)
                        new_hist = hist + [(act_id, data)]
                        if r.levels_completed > level_idx or g2._current_level_index > level_idx:
                            logger.info(f"BFS L{level_idx}: SOLVED (hidden retry) in {len(new_hist)} actions ({explored2} explored)")
                            self.solutions[level_idx] = new_hist
                            return new_hist
                        if depth < 60:
                            queue2.append((g2, new_hist, depth + 1))
                logger.info(f"BFS L{level_idx}: hidden retry also failed ({explored2} explored, {len(visited2)} unique)")
        return None

    def _try_transfer(self, game, level_idx, prev_solution, f1):
        """v9: Transfer previous level's solution to current level."""
        try:
            # Try executing prev solution directly (sometimes levels share exact solution)
            g = copy.deepcopy(game)
            for i, (act_id, data) in enumerate(prev_solution):
                try:
                    ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                    r = g.perform_action(ai, raw=True)
                    if r.levels_completed > level_idx or g._current_level_index > level_idx:
                        logger.info(f"BFS L{level_idx}: TRANSFER SUCCESS (direct replay, {i+1} actions)")
                        sol = prev_solution[:i+1]
                        self.solutions[level_idx] = sol
                        return sol
                except:
                    break

            # Try object-relative transfer (CHRONOS Opus T11)
            prev_game = self.game_cls()
            prev_game.set_level(level_idx - 1)
            prev_game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            r_prev = prev_game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            if not r_prev.frame:
                return None
            f0 = np.array(r_prev.frame[-1])
            bg = int(np.bincount(f0.flatten(), minlength=16).argmax())

            # Extract objects from both levels
            def get_objects(frame, bg_c):
                objs = []
                for c in range(16):
                    if c == bg_c:
                        continue
                    mask = (frame == c)
                    npix = int(np.sum(mask))
                    if npix < 2:
                        continue
                    ys, xs = np.where(mask)
                    objs.append({'color': c, 'cx': float(np.mean(xs)), 'cy': float(np.mean(ys)), 'n': npix})
                return sorted(objs, key=lambda o: (o['color'], -o['n']))

            objs_prev = get_objects(f0, bg)
            objs_curr = get_objects(f1, bg)

            if not objs_prev or not objs_curr:
                return None

            # Match objects by color + relative size
            matched = []
            for op in objs_prev:
                best = None
                best_dist = float('inf')
                for oc in objs_curr:
                    if oc['color'] == op['color'] and abs(oc['n'] - op['n']) < max(op['n'], oc['n']) * 0.5:
                        d = abs(oc['cx'] - op['cx']) + abs(oc['cy'] - op['cy'])
                        if d < best_dist:
                            best_dist = d
                            best = oc
                if best:
                    matched.append((op, best))

            if not matched:
                return None

            # Compute offset
            dx = np.mean([m[1]['cx'] - m[0]['cx'] for m in matched])
            dy = np.mean([m[1]['cy'] - m[0]['cy'] for m in matched])

            # Apply offset to click actions
            transferred = []
            for act_id, data in prev_solution:
                if data and 'x' in data:
                    new_data = dict(data)
                    new_data['x'] = max(0, min(63, int(data['x'] + dx)))
                    new_data['y'] = max(0, min(63, int(data['y'] + dy)))
                    transferred.append((act_id, new_data))
                else:
                    transferred.append((act_id, data))

            # Validate transferred solution
            g = copy.deepcopy(game)
            for i, (act_id, data) in enumerate(transferred):
                try:
                    ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                    r = g.perform_action(ai, raw=True)
                    if r.levels_completed > level_idx or g._current_level_index > level_idx:
                        logger.info(f"BFS L{level_idx}: TRANSFER SUCCESS (offset dx={dx:.0f},dy={dy:.0f}, {i+1} actions)")
                        sol = transferred[:i+1]
                        self.solutions[level_idx] = sol
                        return sol
                except:
                    break

        except Exception as e:
            logger.warning(f"BFS transfer failed: {e}")
        return None


    def transition_table_solve(self, level_idx, timeout=8):
        """v82: Model-based A* solver via empirical transition table.

        Phase 1 (PROBE): Try each action from multiple states, record
                         (scalar_state, action) → scalar_state transitions.
        Phase 2 (SEARCH): A* over the scalar state graph.
        Phase 3 (VERIFY): Replay the found path on a real game instance.

        This solves in milliseconds what BFS takes minutes for — the scalar
        state space is typically 100-10,000 states vs 10^6+ pixel states.
        """
        if not self.game_cls:
            return None
        import heapq
        t0 = time.time()

        def mk():
            return self._make_game(level_idx, drain=True)  # v85: drain for probing

        def won(g):
            return g._current_level_index > level_idx

        def extract_state(g):
            """Extract hashable scalar state tuple from game."""
            parts = []
            for k in sorted(g.__dict__.keys()):
                if k.startswith('__'):
                    continue
                v = g.__dict__[k]
                if isinstance(v, (int, float, bool)):
                    parts.append((k, v))
                elif isinstance(v, (list, tuple)):
                    parts.append((k, len(v)))
                elif isinstance(v, set):
                    parts.append((k, len(v)))
                elif isinstance(v, dict):
                    parts.append((k, len(v)))
            return tuple(parts)

        # Filter out internal fields that change every action
        try:
            g0 = mk()
            avail = list(set(g0._available_actions))
            s0 = extract_state(g0)
        except:
            return None

        # Find which fields actually vary (skip constant fields)
        internal = {'_action_count', '_full_reset', '_action_complete'}

        # === PHASE 1: PROBE — build transition table ===
        transitions = {}  # (state, action_key) → next_state
        win_states = {}   # state → action_key that caused win
        visited_states = set()
        probe_queue = [s0]
        visited_states.add(s0)

        # For click games, find reactive positions first
        click_positions = []
        if 6 in avail:
            try:
                g_scan = mk()
                f0 = g_scan.get_pixels(0, 0, 64, 64)
                bg = int(np.bincount(f0.flatten(), minlength=16).argmax())
                seen_eff = set()
                for y in range(1, 63, 4):
                    if time.time() - t0 > timeout * 0.2:
                        break
                    for x in range(1, 63, 4):
                        if f0[y, x] == bg:
                            continue
                        try:
                            gt = mk()
                            data = {'x': x, 'y': y, 'game_id': 'tt'}
                            r = gt.perform_action(ActionInput(id=GameAction.ACTION6, data=data), raw=True)
                            if r and r.frame:
                                diff = int(np.sum(f0 != np.array(r.frame[-1])))
                                if diff > 0:
                                    eh = hashlib.md5(np.array(r.frame[-1]).tobytes()).hexdigest()[:8]
                                    if eh not in seen_eff:
                                        seen_eff.add(eh)
                                        click_positions.append(data)
                                if won(gt):
                                    self.solutions[level_idx] = [(6, data)]
                                    return [(6, data)]
                        except:
                            pass
            except:
                pass

        # Build action list
        action_list = []
        for a in sorted(avail):
            if a == 6:
                for cp in click_positions[:12]:
                    action_list.append((6, cp))
            elif a <= 7:
                action_list.append((a, None))

        if not action_list:
            return None

        # BFS-probe: explore reachable scalar states
        probe_count = 0
        max_probes = 500
        while probe_queue and probe_count < max_probes and time.time() - t0 < timeout * 0.5:
            state = probe_queue.pop(0)

            for act_id, data in action_list:
                if time.time() - t0 > timeout * 0.5:
                    break
                probe_count += 1
                try:
                    # Replay to reach 'state', then apply action
                    # We need the action path to reach 'state'
                    # Use the reverse map from transitions
                    g = mk()
                    # Replay path to state
                    path_to_state = self._find_path_to_state(s0, state, transitions)
                    if path_to_state is None and state != s0:
                        continue
                    if path_to_state:
                        for pa, pd in path_to_state:
                            ai = ActionInput(id=GameAction.from_id(pa), data=pd) if pd else ActionInput(id=GameAction.from_id(pa))
                            g.perform_action(ai, raw=True)

                    # Apply the probed action
                    ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                    g.perform_action(ai, raw=True)

                    if won(g):
                        win_states[state] = (act_id, data)
                        # Found a win! Reconstruct path
                        path = self._find_path_to_state(s0, state, transitions)
                        if path is None:
                            path = []
                        sol = path + [(act_id, data)]
                        # Verify
                        gv = mk()
                        for a, d in sol:
                            ai = ActionInput(id=GameAction.from_id(a), data=d) if d else ActionInput(id=GameAction.from_id(a))
                            gv.perform_action(ai, raw=True)
                        if won(gv):
                            logger.info(f"TT L{level_idx}: SOLVED {len(sol)} acts, "
                                        f"{probe_count} probes, {len(visited_states)} states")
                            self.solutions[level_idx] = sol
                            return sol

                    next_state = extract_state(g)
                    # Filter: only track if meaningful fields changed
                    key = (state, act_id, str(data))
                    transitions[key] = next_state

                    if next_state not in visited_states:
                        visited_states.add(next_state)
                        probe_queue.append(next_state)
                except:
                    pass

        if not win_states and not transitions:
            return None

        # === PHASE 2: A* over scalar state graph ===
        if win_states:
            # We already found a win during probing, handled above
            pass

        # A* search: find shortest path from s0 to any win state
        # Use the transition table as the graph
        # Heuristic: 0 (Dijkstra, since we don't know which state wins)
        if time.time() - t0 < timeout * 0.8:
            # BFS over transition graph (already explored during probing)
            # Try extending: from each known state, try all actions
            for state in list(visited_states):
                if time.time() - t0 > timeout * 0.8:
                    break
                for act_id, data in action_list:
                    try:
                        g = mk()
                        path = self._find_path_to_state(s0, state, transitions)
                        if path is None and state != s0:
                            continue
                        for pa, pd in (path or []):
                            g.perform_action(ActionInput(id=GameAction.from_id(pa), data=pd) if pd else ActionInput(id=GameAction.from_id(pa)), raw=True)
                        g.perform_action(ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id)), raw=True)
                        if won(g):
                            sol = (path or []) + [(act_id, data)]
                            gv = mk()
                            for a, d in sol:
                                gv.perform_action(ActionInput(id=GameAction.from_id(a), data=d) if d else ActionInput(id=GameAction.from_id(a)), raw=True)
                            if won(gv):
                                logger.info(f"TT/A* L{level_idx}: SOLVED {len(sol)} acts")
                                self.solutions[level_idx] = sol
                                return sol
                    except:
                        pass

        logger.info(f"TT L{level_idx}: no solution ({probe_count} probes, "
                    f"{len(visited_states)} states, {time.time()-t0:.1f}s)")
        return None


    def _find_path_to_state(self, start, target, transitions):
        """BFS over transition table to find action path from start to target."""
        if start == target:
            return []
        visited = {start}
        queue = deque([(start, [])])
        while queue:
            state, path = queue.popleft()
            if len(path) > 50:
                continue
            for key, next_state in transitions.items():
                if key[0] != state:
                    continue
                if next_state in visited:
                    continue
                visited.add(next_state)
                act_id = key[1]
                data_str = key[2]
                # Reconstruct data
                if data_str == 'None':
                    data = None
                else:
                    try:
                        data = eval(data_str)
                    except:
                        data = None
                new_path = path + [(act_id, data)]
                if next_state == target:
                    return new_path
                queue.append((next_state, new_path))
        return None


    def source_solve(self, level_idx, timeout=25):
        """v82: Source-aware solver — probe actions, find productive ones, solve.
        Called when BFS fails. Uses game_cls directly (offline, no deepcopy in hot path).
        Strategies: probe → repeat best → cycle productive → random walk → IDS."""
        if not self.game_cls:
            return None
        t0 = time.time()

        def mk():
            return self._make_game(level_idx, drain=True)  # v85: drain for probing

        def won(g):
            return g._current_level_index > level_idx

        try:
            g0 = mk()
            f0 = g0.get_pixels(0, 0, 64, 64)
            bg = int(np.bincount(f0.flatten(), minlength=16).argmax())
            avail = g0._available_actions
        except:
            return None

        # === PROBE: try each action once ===
        productive = []  # (act_id, data, pixel_diff)
        internal = {'_action_count', '_full_reset', '_action_complete'}
        best_scalar_action = None

        for a in sorted(set(avail)):
            if a == 6 or a > 7:
                continue
            if time.time() - t0 > timeout * 0.15:
                break
            try:
                g = mk()
                snap_before = {}
                for k, v in g.__dict__.items():
                    if k.startswith('__'):
                        continue
                    if isinstance(v, (int, float, bool)):
                        snap_before[k] = v
                    elif isinstance(v, (list, tuple, set)):
                        snap_before[k] = len(v)
                    elif isinstance(v, dict):
                        snap_before[k] = len(v)
                r = g.perform_action(ActionInput(id=GameAction.from_id(a)), raw=True)
                if not r or not r.frame:
                    continue
                f1 = np.array(r.frame[-1])
                diff = int(np.sum(f0 != f1))
                if won(g):
                    sol = [(a, None)]
                    self.solutions[level_idx] = sol
                    return sol
                snap_after = {}
                for k, v in g.__dict__.items():
                    if k.startswith('__'):
                        continue
                    if isinstance(v, (int, float, bool)):
                        snap_after[k] = v
                    elif isinstance(v, (list, tuple, set)):
                        snap_after[k] = len(v)
                    elif isinstance(v, dict):
                        snap_after[k] = len(v)
                scalar_changed = {k for k in snap_after if snap_before.get(k) != snap_after.get(k)} - internal
                if diff > 0 or scalar_changed:
                    productive.append((a, None, max(diff, 1)))
                if scalar_changed and best_scalar_action is None:
                    best_scalar_action = (a, None, max(diff, 1))
            except:
                pass

        # Probe clicks (coarse scan)
        if 6 in avail and time.time() - t0 < timeout * 0.25:
            seen = set()
            for y in range(1, 63, 4):
                if time.time() - t0 > timeout * 0.25:
                    break
                for x in range(1, 63, 4):
                    if f0[y, x] == bg:
                        continue
                    try:
                        g = mk()
                        data = {'x': x, 'y': y, 'game_id': 'p'}
                        r = g.perform_action(ActionInput(id=GameAction.ACTION6, data=data), raw=True)
                        if not r or not r.frame:
                            continue
                        f1 = np.array(r.frame[-1])
                        diff = int(np.sum(f0 != f1))
                        if diff > 0:
                            eh = hashlib.md5(f1[2:62].tobytes()).hexdigest()[:10]
                            if eh not in seen:
                                seen.add(eh)
                                productive.append((6, data, diff))
                        if won(g):
                            sol = [(6, data)]
                            self.solutions[level_idx] = sol
                            return sol
                    except:
                        pass

        if not productive:
            return None

        productive.sort(key=lambda x: -x[2])
        # Prefer scalar-changing action for repeat strategy
        if best_scalar_action:
            productive.insert(0, best_scalar_action)
            # Deduplicate
            seen_acts = set()
            deduped = []
            for a, d, diff in productive:
                key = (a, str(d))
                if key not in seen_acts:
                    seen_acts.add(key)
                    deduped.append((a, d, diff))
            productive = deduped

        logger.info(f"SOURCE L{level_idx}: {len(productive)} productive actions")

        # === STRATEGY 1: Repeat best action ===
        for a, d, _ in productive[:3]:
            if time.time() - t0 > timeout * 0.4:
                break
            g = mk()
            sol = []
            for _ in range(200):
                ai = ActionInput(id=GameAction.from_id(a), data=d) if d else ActionInput(id=GameAction.from_id(a))
                try:
                    g.perform_action(ai, raw=True)
                    sol.append((a, d))
                    if won(g):
                        logger.info(f"SOURCE L{level_idx}: SOLVED by repeat action {a} × {len(sol)}")
                        self.solutions[level_idx] = sol
                        return sol
                except:
                    break

        # === STRATEGY 2: Cycle through productive actions ===
        g = mk()
        sol = []
        for cycle in range(30):
            if time.time() - t0 > timeout * 0.55:
                break
            for a, d, _ in productive[:10]:
                ai = ActionInput(id=GameAction.from_id(a), data=d) if d else ActionInput(id=GameAction.from_id(a))
                try:
                    g.perform_action(ai, raw=True)
                    sol.append((a, d))
                    if won(g):
                        logger.info(f"SOURCE L{level_idx}: SOLVED by cycling × {cycle+1}")
                        self.solutions[level_idx] = sol
                        return sol
                except:
                    pass

        # === STRATEGY 2b: Iterative click probing (for ordered click puzzles) ===
        # Probe → click the only reactive position → re-probe → repeat
        if 6 in avail and time.time() - t0 < timeout * 0.6:
            g = mk()
            sol = []
            for iteration in range(20):
                if time.time() - t0 > timeout * 0.65 or won(g):
                    break
                # Probe current state for reactive clicks
                curr_f = g.get_pixels(0, 0, 64, 64)
                curr_bg = int(np.bincount(curr_f.flatten(), minlength=16).argmax())
                best_click = None
                best_diff = 0
                for py in range(1, 63, 4):
                    for px in range(1, 63, 4):
                        if curr_f[py, px] == curr_bg:
                            continue
                        try:
                            g2 = mk()
                            # Replay sol to reach current state
                            for sa, sd in sol:
                                g2.perform_action(ActionInput(id=GameAction.from_id(sa), data=sd) if sd else ActionInput(id=GameAction.from_id(sa)), raw=True)
                            data = {'x': px, 'y': py, 'game_id': 'ip'}
                            r = g2.perform_action(ActionInput(id=GameAction.ACTION6, data=data), raw=True)
                            if r and r.frame:
                                f1 = np.array(r.frame[-1])
                                diff = int(np.sum(curr_f != f1))
                                if diff > best_diff:
                                    best_diff = diff
                                    best_click = data
                                if g2._current_level_index > level_idx:
                                    sol.append((6, data))
                                    logger.info(f"SOURCE L{level_idx}: SOLVED by iterative click "
                                                f"({len(sol)} acts)")
                                    self.solutions[level_idx] = sol
                                    return sol
                        except:
                            pass
                if best_click and best_diff > 0:
                    # Apply the best click
                    g.perform_action(ActionInput(id=GameAction.ACTION6, data=best_click), raw=True)
                    sol.append((6, best_click))
                    if won(g):
                        logger.info(f"SOURCE L{level_idx}: SOLVED by iterative click "
                                    f"({len(sol)} acts)")
                        self.solutions[level_idx] = sol
                        return sol
                else:
                    break  # no more reactive clicks

        # === STRATEGY 2c: Position-tracking BFS (navigation games) ===
        # Instead of hashing full 64x64 frame, track only the pixels that move.
        # State space = unique pixel configurations ≈ grid positions = O(4096) vs O(2^12)
        dir_acts = [(a, d) for a, d, _ in productive if a <= 5]
        if len(dir_acts) >= 2 and time.time() - t0 < timeout * 0.65:
            # Detect player pixels: which pixels change when we move?
            g_detect = mk()
            f_before = g_detect.get_pixels(0, 0, 64, 64).copy()
            a_test, d_test = dir_acts[0]
            ai_test = ActionInput(id=GameAction.from_id(a_test), data=d_test) if d_test else ActionInput(id=GameAction.from_id(a_test))
            r_test = g_detect.perform_action(ai_test, raw=True)
            if r_test and r_test.frame:
                f_after = np.array(r_test.frame[-1])
                diff_mask = (f_before != f_after)
                n_diff = int(np.sum(diff_mask))
                # If only a small cluster of pixels moved (2-20), that's the player
                if 1 <= n_diff <= 40:
                    # Detect player color: the NEW color at the moved-to position
                    ys_new, xs_new = np.where((f_after != f_before) & (f_after != bg))
                    player_color = int(f_after[ys_new[0], xs_new[0]]) if len(ys_new) > 0 else -1

                    def pos_hash(game, _pc=player_color, _bg=bg):
                        """Hash player position (centroid of player-colored pixels)."""
                        fp = game.get_pixels(0, 0, 64, 64)
                        pys, pxs = np.where(fp == _pc)
                        if len(pys) == 0:
                            # Fallback: hash all non-bg pixels
                            pys, pxs = np.where(fp != _bg)
                        if len(pys) == 0:
                            return (0, 0)
                        return (int(np.mean(pxs)), int(np.mean(pys)))

                    # Pickle-fork pos-BFS: keep game objects in queue, pickle for forking
                    pos_visited = set()
                    g_start = mk()
                    pos_visited.add(pos_hash(g_start))
                    pos_queue = deque()
                    pos_queue.append((copy.deepcopy(g_start), []))
                    pos_explored = 0
                    pos_max = 100000

                    # Test if pickle works
                    pos_pkl = False
                    try:
                        _tp = pickle.dumps(g_start, protocol=4)
                        pickle.loads(_tp)
                        pos_pkl = True
                    except:
                        pass

                    while pos_queue and pos_explored < pos_max and time.time() - t0 < timeout * 0.75:
                        g_parent, hist = pos_queue.popleft()
                        if len(hist) >= 120:
                            continue
                        # Pickle parent once
                        ppkl = None
                        if pos_pkl:
                            try: ppkl = pickle.dumps(g_parent, protocol=4)
                            except: pass
                        for a, d in dir_acts:
                            pos_explored += 1
                            try:
                                g2 = pickle.loads(ppkl) if ppkl else copy.deepcopy(g_parent)
                                ai = ActionInput(id=GameAction.from_id(a), data=d) if d else ActionInput(id=GameAction.from_id(a))
                                g2.perform_action(ai, raw=True)
                                if won(g2):
                                    sol = hist + [(a, d)]
                                    logger.info(f"SOURCE L{level_idx}: SOLVED by pos-BFS "
                                                f"({len(sol)} acts, {pos_explored} explored)")
                                    self.solutions[level_idx] = sol
                                    return sol
                                ph = pos_hash(g2)
                                if ph not in pos_visited:
                                    pos_visited.add(ph)
                                    pos_queue.append((g2, hist + [(a, d)]))
                            except:
                                pass

                    logger.info(f"SOURCE L{level_idx}: pos-BFS exhausted ({pos_explored} explored, "
                                f"{len(pos_visited)} positions)")

        # === STRATEGY 2d: Full-frame mini-BFS for directional games ===
        if len(dir_acts) >= 2 and time.time() - t0 < timeout * 0.8:
            bfs_visited = set()
            g_init = mk()
            init_fh = hashlib.md5(g_init.get_pixels(0, 0, 64, 64)[2:62].tobytes()).hexdigest()[:12]
            bfs_visited.add(init_fh)
            bfs_queue = deque()
            bfs_queue.append(())
            bfs_max_depth = min(40, 200 // max(1, len(dir_acts)))
            bfs_explored = 0
            while bfs_queue and bfs_explored < 30000 and time.time() - t0 < timeout * 0.75:
                seq = bfs_queue.popleft()
                if len(seq) >= bfs_max_depth:
                    continue
                for a, d in dir_acts:
                    bfs_explored += 1
                    child = seq + ((a, d),)
                    try:
                        g = mk()
                        for ca, cd in child:
                            ai = ActionInput(id=GameAction.from_id(ca), data=cd) if cd else ActionInput(id=GameAction.from_id(ca))
                            g.perform_action(ai, raw=True)
                        if won(g):
                            logger.info(f"SOURCE L{level_idx}: SOLVED by mini-BFS "
                                        f"({len(child)} acts, {bfs_explored} explored)")
                            sol = list(child)
                            self.solutions[level_idx] = sol
                            return sol
                        fh = hashlib.md5(g.get_pixels(0, 0, 64, 64)[2:62].tobytes()).hexdigest()[:12]
                        if fh not in bfs_visited:
                            bfs_visited.add(fh)
                            bfs_queue.append(child)
                    except:
                        pass

        # === STRATEGY 3: Random walk with restart ===
        acts = [(a, d) for a, d, _ in productive[:12]]
        n_acts = len(acts)
        if n_acts > 0:
            walk_count = 0
            while time.time() - t0 < timeout * 0.85:
                walk_count += 1
                depth = random.randint(2, min(15, max(3, 50 // n_acts)))
                g = mk()
                sol = []
                for _ in range(depth):
                    a, d = random.choice(acts)
                    ai = ActionInput(id=GameAction.from_id(a), data=d) if d else ActionInput(id=GameAction.from_id(a))
                    try:
                        g.perform_action(ai, raw=True)
                        sol.append((a, d))
                        if won(g):
                            logger.info(f"SOURCE L{level_idx}: SOLVED by random walk #{walk_count} "
                                        f"(depth={len(sol)})")
                            self.solutions[level_idx] = sol
                            return sol
                    except:
                        break
            logger.info(f"SOURCE L{level_idx}: {walk_count} random walks tried, no solution")

        # === STRATEGY 4: IDS (depth 1-6) with productive actions only ===
        ids_acts = [(a, d) for a, d, _ in productive[:8]]
        if len(ids_acts) > 0 and time.time() - t0 < timeout * 0.95:
            for depth in range(1, min(7, len(ids_acts) + 1)):
                if time.time() - t0 > timeout * 0.95:
                    break
                visited = set()
                stack = [()]
                attempts = 0
                while stack and attempts < 3000 and time.time() - t0 < timeout * 0.95:
                    seq = stack.pop()
                    if len(seq) >= depth:
                        continue
                    for a, d in ids_acts:
                        attempts += 1
                        child = seq + ((a, d),)
                        try:
                            g = mk()
                            for ca, cd in child:
                                ai = ActionInput(id=GameAction.from_id(ca), data=cd) if cd else ActionInput(id=GameAction.from_id(ca))
                                g.perform_action(ai, raw=True)
                            if won(g):
                                logger.info(f"SOURCE L{level_idx}: SOLVED by IDS depth {depth}")
                                sol = list(child)
                                self.solutions[level_idx] = sol
                                return sol
                            fh = hashlib.md5(g.get_pixels(0, 0, 64, 64)[2:62].tobytes()).hexdigest()[:12]
                            if fh not in visited:
                                visited.add(fh)
                                stack.append(child)
                        except:
                            pass

        logger.info(f"SOURCE L{level_idx}: all strategies exhausted ({time.time()-t0:.1f}s)")
        return None



    def click_all_solve(self, level_idx, timeout=5):
        """v83: Click every non-bg pixel cluster L→R. For click-only games."""
        if not self.game_cls:
            return None
        try:
            g = self.game_cls()
            g.set_level(level_idx)
            g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            r0 = g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            if not r0 or not r0.frame:
                return None
            avail = set(g._available_actions)
            if 6 not in avail or any(d in avail for d in [1, 2, 3, 4]):
                return None  # not click-only
            f0 = np.array(r0.frame[-1])
            bg = int(np.bincount(f0.flatten(), minlength=16).argmax())
            # Find object centers
            targets = []
            visited_px = np.zeros((64, 64), dtype=bool)
            for y in range(2, 62):
                for x in range(2, 62):
                    if f0[y, x] != bg and not visited_px[y, x]:
                        cys, cxs = [y], [x]
                        visited_px[y, x] = True
                        qi = 0
                        while qi < len(cys):
                            cy, cx = cys[qi], cxs[qi]; qi += 1
                            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                                ny, nx = cy+dy, cx+dx
                                if 0<=ny<64 and 0<=nx<64 and not visited_px[ny,nx] and f0[ny,nx]!=bg:
                                    visited_px[ny,nx]=True; cys.append(ny); cxs.append(nx)
                        if len(cys) >= 2:
                            targets.append((int(np.mean(cxs)), int(np.mean(cys))))
            if not targets:
                return None
            # Click L→R
            g2 = self.game_cls()
            g2.set_level(level_idx)
            g2.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            g2.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            sol = []
            for tx, ty in sorted(targets):
                data = {'x': tx, 'y': ty, 'game_id': 'ca'}
                try:
                    g2.perform_action(ActionInput(id=GameAction.ACTION6, data=data), raw=True)
                    sol.append((6, data))
                    if g2._current_level_index > level_idx:
                        self.solutions[level_idx] = sol
                        return sol
                except:
                    pass
            # Try R→L
            g3 = self.game_cls()
            g3.set_level(level_idx)
            g3.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            g3.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            sol2 = []
            for tx, ty in sorted(targets, reverse=True):
                data = {'x': tx, 'y': ty, 'game_id': 'ca'}
                try:
                    g3.perform_action(ActionInput(id=GameAction.ACTION6, data=data), raw=True)
                    sol2.append((6, data))
                    if g3._current_level_index > level_idx:
                        self.solutions[level_idx] = sol2
                        return sol2
                except:
                    pass
        except:
            pass
        return None


def find_game_source_and_class(game_id, arc_env=None):
    """Find the game .py file and class name."""
    gid = game_id.split('-')[0]
    cls_name = gid.capitalize()
    if len(gid) == 4 and gid[0].isalpha():
        cls_name = gid[0].upper() + gid[1:]

    src = None
    # Method 1: from arc_env
    if arc_env and hasattr(arc_env, 'environment_info'):
        ei = arc_env.environment_info
        if hasattr(ei, 'local_dir') and ei.local_dir:
            from pathlib import Path
            ld = Path(ei.local_dir)
            for candidate in [ld / f"{gid}.py", ld / f"{cls_name.lower()}.py"]:
                if candidate.exists():
                    src = str(candidate)
                    # Get class name from source
                    import re
                    content = candidate.read_text()[:2000]
                    m = re.search(r'class\s+(\w+)\s*\(\s*ARCBaseGame', content)
                    if m:
                        cls_name = m.group(1)
                    break

    # Method 2: glob
    if not src:
        for pattern in [
            f"/tmp/*/{gid}/*/{gid}.py",
            f"/kaggle/*/{gid}*/{gid}.py",
            f"**/game_sources/**/{gid}.py",
        ]:
            matches = glob.glob(pattern, recursive=True)
            if matches:
                src = matches[0]
                import re
                content = open(src).read()[:2000]
                m = re.search(r'class\s+(\w+)\s*\(\s*ARCBaseGame', content)
                if m:
                    cls_name = m.group(1)
                break

    return src, cls_name


# ==================== CNN FALLBACK (v8 core) ====================

class CBAM(nn.Module):
    def __init__(s, ch, r=16):
        super().__init__()
        s.fc1=nn.Linear(ch,max(ch//r,4)); s.fc2=nn.Linear(max(ch//r,4),ch)
        s.sp=nn.Conv2d(2,1,7,padding=3)
    def forward(s, x):
        B,C,H,W=x.shape
        w=torch.sigmoid(s.fc2(F.relu(s.fc1(x.mean(dim=[2,3]))))); x=x*w.view(B,C,1,1)
        a=torch.sigmoid(s.sp(torch.cat([x.max(1,keepdim=True)[0],x.mean(1,keepdim=True)],1)))
        return x*a

class ActionEffectAttention(nn.Module):
    def __init__(s, feat_dim=64, mem_dim=32, n_actions=5):
        super().__init__()
        s.mem_dim=mem_dim
        s.diff_enc=nn.Sequential(nn.Conv2d(1,8,8,stride=8),nn.ReLU(),nn.Conv2d(8,16,4,stride=4),nn.ReLU(),nn.Flatten(),nn.Linear(16*2*2,mem_dim))
        s.q_proj=nn.Linear(feat_dim,mem_dim)
        s.v_proj=nn.Linear(mem_dim+1+n_actions,n_actions)
        s.scale=mem_dim**0.5
    def forward(s, cnn_feat, mem_diffs, mem_actions, mem_rewards):
        B,M=mem_actions.shape
        if M==0:return torch.zeros(B,5,device=cnn_feat.device)
        keys=s.diff_enc(mem_diffs.reshape(B*M,1,64,64)).reshape(B,M,s.mem_dim)
        q=s.q_proj(cnn_feat).unsqueeze(1)
        attn=F.softmax(torch.bmm(q,keys.transpose(1,2))/s.scale,dim=-1)
        act_oh=F.one_hot(mem_actions.clamp(0,4),5).float()
        vals=torch.cat([keys,mem_rewards.unsqueeze(-1),act_oh],dim=-1)
        ctx=torch.bmm(attn,vals).squeeze(1)
        return s.v_proj(ctx)

class ForgeNet(nn.Module):
    def __init__(s, in_ch=26, g=64):
        super().__init__()
        s.g=g
        s.c1=nn.Conv2d(in_ch,32,3,padding=1);s.c2=nn.Conv2d(32,64,3,padding=1)
        s.c3=nn.Conv2d(64,128,3,padding=1);s.c4=nn.Conv2d(128,256,3,padding=1)
        s.attn=CBAM(256);s.ar=nn.Conv2d(256,64,1);s.ap=nn.MaxPool2d(4,4)
        s.af=nn.Linear(64*16*16,256);s.ah=nn.Linear(256,5);s.dr=nn.Dropout(0.15)
        s.cc1=nn.Conv2d(256,128,3,padding=1);s.cc2=nn.Conv2d(128,64,3,padding=1)
        s.cc3=nn.Conv2d(64,32,1);s.cc4=nn.Conv2d(32,1,1)
        s.gp=nn.AdaptiveAvgPool2d(1);s.gf=nn.Linear(256,64)
        s.aea=ActionEffectAttention(feat_dim=64,mem_dim=32,n_actions=5)
    def forward(s, x, mem_diffs=None, mem_actions=None, mem_rewards=None):
        x=F.relu(s.c1(x));x=F.relu(s.c2(x));x=F.relu(s.c3(x));f=F.relu(s.c4(x))
        f=s.attn(f);af=F.relu(s.ar(f));af=s.ap(af).reshape(f.size(0),-1)
        al=s.ah(s.dr(F.relu(s.af(af))))
        cf=F.relu(s.cc1(f));cf=F.relu(s.cc2(cf));cf=F.relu(s.cc3(cf))
        cl=s.cc4(cf).reshape(f.size(0),-1)
        if mem_diffs is not None and mem_actions is not None:
            gf=s.gf(s.gp(f).reshape(f.size(0),-1))
            al=al+s.aea(gf,mem_diffs,mem_actions,mem_rewards)
        return torch.cat([al,cl],1)


def fast_objects(frame, bg):
    objs=[]
    for c in range(16):
        if c==bg:continue
        mask=(frame==c);npix=int(np.sum(mask))
        if npix<4 or npix>3000:continue
        ys,xs=np.where(mask)
        objs.append((c,float(np.mean(xs)),float(np.mean(ys)),npix))
    return objs


# ==================== AGENT ====================

class MyAgent(Agent):
    MAX_ACTIONS = float('inf')
    _MAX_FRAMES = 10

    def __init__(s, *a, **kw):
        super().__init__(*a, **kw)
        seed = int(time.time()*1e6) + hash(s.game_id) % 1000000
        random.seed(seed); np.random.seed(seed%(2**32-1)); torch.manual_seed(seed%(2**32-1))
        s.start_time = time.time()
        s.device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
        s.G=64; s.IN=26
        s.net=None; s.opt=None
        s.buf=deque(maxlen=50000); s.buf_h=set()
        s.bsz=64; s.tfreq=10
        s.pt=None; s.pai=None; s.pr=None; s.ph=None
        s.cl=-1; s.fhist=deque(maxlen=6); s.la=0
        s.al=[GameAction.ACTION1,GameAction.ACTION2,GameAction.ACTION3,GameAction.ACTION4,GameAction.ACTION5]
        s._wd=False; s._bg=0; s._wm=None
        s._aem_diffs=deque(maxlen=256); s._aem_actions=deque(maxlen=256); s._aem_rewards=deque(maxlen=256)
        s._ckpt_hash=None; s._unproductive=0; s._undo_avail=False
        s._eps=0.15; s._eps_min=0.03; s._eps_decay=0.9997
        s._prev_objs=None; s._obj_moved=0
        # BFS solver
        s._bfs = None
        s._bfs_solution = None  # current level's solution
        s._bfs_step = 0  # current step in solution
        s._bfs_tried = False

    def append_frame(s, f):
        s.frames.append(f)
        if len(s.frames) > s._MAX_FRAMES: s.frames = s.frames[-s._MAX_FRAMES:]
        if f.guid: s.guid = f.guid
        if hasattr(s, "recorder") and not s.is_playback:
            import json; s.recorder.record(json.loads(f.model_dump_json()))

    def _lvl(s, f): return getattr(f, 'score', None) or f.levels_completed
    def _raw(s, fd): return np.array(fd.frame, dtype=np.int64)[-1]

    def _init_bfs(s):
        """Initialize BFS solver on first call."""
        src, cls = find_game_source_and_class(s.game_id, s.arc_env)
        if src:
            s._bfs = BFSSolver(src, cls, scan_timeout=3, bfs_timeout=80)
            if s._bfs.load():
                logger.info(f"BFS: loaded {cls} from {src}")
            else:
                s._bfs = None
                logger.warning(f"BFS: failed to load game class")
        else:
            logger.warning(f"BFS: game source not found for {s.game_id}")

    def _try_bfs_solve(s, level_idx):
        """Try to solve current level with BFS, using previous solution for transfer."""
        if s._bfs is None:
            return None
        prev_sol = s._bfs.solutions.get(level_idx - 1) if level_idx > 0 else None
        sol = s._bfs.solve_level(level_idx, prev_solution=prev_sol)
        if sol:
            s._bfs_solution = sol
            s._bfs_step = 0
            return sol
        return None

    def _tensor(s, fd):
        frame = s._raw(fd)
        oh=torch.zeros(16,64,64,dtype=torch.float32)
        oh.scatter_(0,torch.from_numpy(frame).unsqueeze(0),1)
        cnt=np.bincount(frame.flatten(),minlength=16)
        s._bg=int(cnt.argmax());mx=max(cnt.max(),1)
        bg_m=(frame==s._bg).astype(np.float32)
        rar=np.zeros((64,64),np.float32)
        for c in range(16):
            if cnt[c]>0:rar[frame==c]=1.0-cnt[c]/mx
        pad=np.pad(frame,1,mode='edge')
        edge=((frame!=pad[:-2,1:-1])|(frame!=pad[2:,1:-1])|(frame!=pad[1:-1,:-2])|(frame!=pad[1:-1,2:])).astype(np.float32)
        rp=np.linspace(0,1,64,dtype=np.float32).reshape(64,1).repeat(64,1)
        cp=np.linspace(0,1,64,dtype=np.float32).reshape(1,64).repeat(64,0)
        aug=torch.from_numpy(np.stack([bg_m,rar,edge,rp,cp]))
        d1=torch.zeros(3,64,64,dtype=torch.float32)
        for i,prev in enumerate(reversed(list(s.fhist))):
            if i>=3:break
            d1[i]=torch.from_numpy((frame!=prev).astype(np.float32))
        d2=torch.zeros(2,64,64,dtype=torch.float32)
        h=list(s.fhist)
        if len(h)>=2:d2[0]=torch.from_numpy((h[-1]!=h[-2]).astype(np.float32))
        if len(h)>=4:d2[1]=torch.from_numpy((h[-2]!=h[-4]).astype(np.float32))
        s.fhist.append(frame.copy())
        return torch.cat([oh,aug,d1,d2],0).to(s.device)

    def _detect_template(s, frame):
        mask=torch.ones(4096,dtype=torch.float32)
        col_act=np.sum(frame!=s._bg,axis=0)
        for c in range(20,44):
            if col_act[c]<=2 and np.sum(col_act[:c]>0)>=5 and np.sum(col_act[c+1:]>0)>=5:
                for y in range(64):
                    for x in range(c+1):mask[y*64+x]=0.05
                return mask
        row_act=np.sum(frame!=s._bg,axis=1)
        for r in range(20,44):
            if row_act[r]<=2 and np.sum(row_act[:r]>0)>=5 and np.sum(row_act[r+1:]>0)>=5:
                for y in range(r+1):
                    for x in range(64):mask[y*64+x]=0.05
                return mask
        return mask

    def _reward(s, prev_raw, curr_raw, prev_h, curr_h):
        mask=np.ones((64,64),dtype=bool);mask[:2]=False;mask[62:]=False
        diff=(prev_raw!=curr_raw)&mask;changed=np.any(diff)
        r=0.0
        if curr_h!=prev_h:r+=1.5 if not hasattr(s,'_visited_hashes') else (1.5 if curr_h not in s._visited_hashes else 0.0)
        elif curr_h==prev_h:r-=0.1
        if changed:r+=0.5
        curr_objs=fast_objects(curr_raw,s._bg)
        if s._prev_objs and curr_objs:
            moved=0
            for co in curr_objs:
                for po in s._prev_objs:
                    if co[0]==po[0]:
                        dist=abs(co[1]-po[1])+abs(co[2]-po[2])
                        if 2<dist<20:moved+=1;break
            if moved>0:r+=0.3*min(moved,3);s._obj_moved=moved
        s._prev_objs=curr_objs
        return r

    def _sample(s, logits, avail=None, temp=1.0):
        al=logits[:5].clone();cl=logits[5:5+4096].clone()
        if avail is not None and len(avail)>0:
            mask=torch.full_like(al,float('-inf'));a6=False
            for a in avail:
                aid=a.value if hasattr(a,'value') else int(a)
                if 1<=aid<=5:mask[aid-1]=0.0
                elif aid==6:a6=True
            al=al+mask
            if not a6:cl=cl+torch.full_like(cl,float('-inf'))
        if s._wm is not None:cl=cl+torch.log(s._wm.to(s.device).clamp(min=0.01))
        ap=torch.sigmoid(al/temp);cp=torch.sigmoid(cl/temp)/(s.G*s.G)
        allp=torch.cat([ap,cp]);sm=allp.sum()
        if sm<1e-8:allp=torch.ones_like(allp)/len(allp)
        else:allp=allp/sm
        idx=np.random.choice(len(allp),p=allp.cpu().numpy())
        if idx<5:return idx,None
        ci=idx-5;return 5,(ci//s.G,ci%s.G)

    def _heuristic(s, frame, avail, step):
        av=set(int(a.value) if hasattr(a,'value') else int(a) for a in avail)
        for d in[1,2,3,4]:
            if d in av and step<4:return d-1,None
        if 6 in av:
            cnt=np.bincount(frame.flatten(),minlength=16);targets=[]
            for c in range(16):
                if c==s._bg or cnt[c]==0 or cnt[c]>2000:continue
                ys,xs=np.where(frame==c)
                if len(ys)>=2:targets.append((int(np.median(xs)),int(np.median(ys)),len(ys)))
            targets.sort(key=lambda t:t[2]);pidx=step-4
            if 0<=pidx<len(targets):return 5,(targets[pidx][1],targets[pidx][0])
        if 5 in av:return 4,None
        choices=[a for a in av if 1<=a<=5]
        if choices:return random.choice(choices)-1,None
        return 0,None

    def _frame_to_tensor(s, frame):
        oh=torch.zeros(16,64,64,dtype=torch.float32)
        oh.scatter_(0,torch.from_numpy(frame).unsqueeze(0),1)
        cnt=np.bincount(frame.flatten(),minlength=16)
        bg=int(cnt.argmax());mx=max(cnt.max(),1)
        bg_m=(frame==bg).astype(np.float32)
        rar=np.zeros((64,64),np.float32)
        for c in range(16):
            if cnt[c]>0:rar[frame==c]=1.0-cnt[c]/mx
        pad=np.pad(frame,1,mode='edge')
        edge=((frame!=pad[:-2,1:-1])|(frame!=pad[2:,1:-1])|(frame!=pad[1:-1,:-2])|(frame!=pad[1:-1,2:])).astype(np.float32)
        rp=np.linspace(0,1,64,dtype=np.float32).reshape(64,1).repeat(64,1)
        cp=np.linspace(0,1,64,dtype=np.float32).reshape(1,64).repeat(64,0)
        aug=torch.from_numpy(np.stack([bg_m,rar,edge,rp,cp]))
        zeros=torch.zeros(5,64,64,dtype=torch.float32)
        return torch.cat([oh,aug,zeros],0)

    def _train(s):
        if len(s.buf)<s.bsz:return
        indices=np.random.choice(len(s.buf),s.bsz,replace=False)
        batch=[s.buf[i] for i in indices]
        states=torch.stack([s._frame_to_tensor(e['s']).to(s.device) for e in batch])
        acts=torch.tensor([e['a'] for e in batch],dtype=torch.long,device=s.device)
        rews=torch.tensor([e['r'] for e in batch],dtype=torch.float32,device=s.device)
        rews=torch.sigmoid(rews);s.opt.zero_grad()
        logits=s.net(states)
        acts_c=acts.clamp(0,logits.size(1)-1)
        sel=logits.gather(1,acts_c.unsqueeze(1)).squeeze(1)
        loss=F.binary_cross_entropy_with_logits(sel,rews)
        p=torch.sigmoid(logits);loss=loss-0.0001*p[:,:5].mean()-0.00001*p[:,5:].mean()
        loss.backward();s.opt.step()

    def _get_aem_tensors(s):
        if len(s._aem_diffs)<2:return None,None,None
        M=len(s._aem_diffs)
        diffs=torch.zeros(1,M,1,64,64,device=s.device)
        acts=torch.zeros(1,M,dtype=torch.long,device=s.device)
        rews=torch.zeros(1,M,device=s.device)
        for i,(d,a,r) in enumerate(zip(s._aem_diffs,s._aem_actions,s._aem_rewards)):
            diffs[0,i,0]=torch.from_numpy(d.astype(np.float32));acts[0,i]=min(a,4);rews[0,i]=r
        return diffs,acts,rews

    def is_done(s, frames, lf):
        try: return lf.state is GameState.WIN or (time.time()-s.start_time) >= 8*3600-300
        except: return True

    def choose_action(s, frames, lf):
        try:
            lvl = s._lvl(lf)

            # ===== LEVEL CHANGE =====
            if lvl != s.cl:
                # Init BFS solver on first level
                if not s._bfs_tried:
                    s._bfs_tried = True
                    s._init_bfs()

                # v83: CNN-guided action priority — run pretrained CNN on initial frame
                # to get action preferences, pass to BFS for ordering
                s._bfs_solution = None
                s._bfs_step = 0

                # Init CNN and get action scores for BFS
                s.net = ForgeNet(s.IN, s.G).to(s.device)
                for wp in ['/kaggle/input/forge-pretrained-weights/pretrained_weights.pt',
                           'pretrained_weights.pt']:
                    try:
                        if os.path.exists(wp):
                            state=torch.load(wp,map_location=s.device,weights_only=True)
                            ms=s.net.state_dict()
                            for k in list(state.keys()):
                                if k in ms and state[k].shape==ms[k].shape:ms[k]=state[k]
                            s.net.load_state_dict(ms);break
                    except: pass

                if s._bfs and lf.frame:
                    try:
                        with torch.no_grad():
                            tensor = s._tensor(lf)
                            logits = s.net(tensor.unsqueeze(0)).squeeze(0)
                            s._bfs.cnn_action_scores = logits.cpu().numpy()
                    except:
                        pass

                # v85: BFS-FIRST cascade — BFS is the proven workhorse, run it first
                level_t0 = time.time()
                LEVEL_BUDGET = 120.0

                if s._bfs:
                    # 1. BFS FIRST (full 80s) — the v10 core that scored 0.39
                    try:
                        s._try_bfs_solve(lvl)
                    except:
                        pass

                    # 2. Click-all (5s) — for games BFS missed (click-only)
                    if not s._bfs_solution:
                        try:
                            ca = s._bfs.click_all_solve(lvl, timeout=5)
                            if ca:
                                s._bfs_solution = ca
                                s._bfs_step = 0
                        except:
                            pass

                    # 3. TT solver (8s) — empirical state graph
                    if not s._bfs_solution:
                        try:
                            tt = s._bfs.transition_table_solve(lvl, timeout=8)
                            if tt:
                                s._bfs_solution = tt
                                s._bfs_step = 0
                        except:
                            pass

                    # 4. Source solver (remaining time, max 20s)
                    if not s._bfs_solution:
                        total_elapsed = time.time() - level_t0
                        source_budget = min(20.0, LEVEL_BUDGET - total_elapsed - 5.0)
                        if source_budget > 3.0:
                            try:
                                ss = s._bfs.source_solve(lvl, timeout=source_budget)
                                if ss:
                                    s._bfs_solution = ss
                                    s._bfs_step = 0
                            except:
                                pass

                # Init CNN fallback (net already loaded above for BFS guidance)
                s.buf.clear(); s.buf_h.clear()
                s.opt = optim.Adam(s.net.parameters(), lr=0.0003)
                s.pt=None;s.pai=None;s.pr=None;s.ph=None
                s.cl=lvl;s.fhist.clear();s.la=0
                s._wd=False;s._wm=None;s._eps=0.15
                s._aem_diffs.clear();s._aem_actions.clear();s._aem_rewards.clear()
                s._prev_objs=None;s._obj_moved=0;s._ckpt_hash=None;s._unproductive=0

            # ===== RESET =====
            if lf.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
                s.pt=None;s.pai=None;s.pr=None;s.ph=None
                a=GameAction.RESET;a.reasoning="reset";return a

            # ===== BFS SOLUTION EXECUTION =====
            if s._bfs_solution and s._bfs_step < len(s._bfs_solution):
                act_id, data = s._bfs_solution[s._bfs_step]
                s._bfs_step += 1
                sel = GameAction.from_id(act_id)
                if data:
                    sel.set_data(data)
                sel.reasoning = f"bfs:{s._bfs_step}/{len(s._bfs_solution)}"
                # Still update prev state for fallback
                raw = s._raw(lf)
                s.fhist.append(raw.copy())
                s.pr = raw.copy()
                s.la += 1
                return sel

            # ===== CNN FALLBACK (v8 core) =====
            tensor = s._tensor(lf)
            raw = s._raw(lf)
            ch = hashlib.md5(raw.tobytes()).hexdigest()[:16]
            avail = getattr(lf, 'available_actions', None) or []
            s._undo_avail = any((a.value if hasattr(a,'value') else int(a))==7 for a in avail)

            if s.pt is not None and s.pai is not None:
                mask=np.ones((64,64),dtype=bool);mask[:2]=False;mask[62:]=False
                diff_map=(s.pr!=raw)&mask;changed=np.any(diff_map)
                eh=hashlib.md5(s.pr.tobytes()[:1000]+str(s.pai).encode()).hexdigest()[:16]
                if eh not in s.buf_h:
                    r=s._reward(s.pr,raw,'',ch)
                    s.buf.append({'s':s.pr.copy(),'a':s.pai,'r':r})
                    s.buf_h.add(eh)
                    if changed:
                        s._aem_diffs.append(diff_map)
                        s._aem_actions.append(min(s.pai,4))
                        s._aem_rewards.append(r)
                if changed:s._ckpt_hash=ch;s._unproductive=0
                else:s._unproductive+=1

            avail_idx=[]
            for a in avail:
                aid=a.value if hasattr(a,'value') else int(a)
                if 1<=aid<=5:avail_idx.append(aid-1)
                elif aid==6:avail_idx.extend([5+i for i in range(0,4096,128)])

            if s._wm is None:s._wm=s._detect_template(raw)

            if s._undo_avail and s._unproductive>=30 and s._ckpt_hash:
                s._unproductive=0;a=GameAction.ACTION7;a.reasoning="undo"
                s.pt=tensor;s.pai=6;s.pr=raw.copy();s.ph=ch;s.la+=1;return a

            if not s._wd:
                if s.la<10:aidx,coords=s._heuristic(raw,avail,s.la)
                else:
                    s._wd=True
                    for _ in range(min(5,len(s.buf)//s.bsz)):s._train()

            if s._wd:
                if random.random()<s._eps:
                    aidx,coords=s._sample(torch.zeros(4101,device=s.device),avail,temp=2.0)
                else:
                    with torch.no_grad():
                        mem=s._get_aem_tensors()
                        if mem[0] is not None:logits=s.net(tensor.unsqueeze(0),*mem).squeeze(0)
                        else:logits=s.net(tensor.unsqueeze(0)).squeeze(0)
                    aidx,coords=s._sample(logits,avail,temp=0.5)
                s._eps=max(s._eps_min,s._eps*s._eps_decay)
            elif s.la>=10:s._wd=True;aidx,coords=0,None

            if aidx<5:sel=s.al[aidx];sel.reasoning=f"cnn:a{aidx+1}"
            else:
                sel=GameAction.ACTION6;y,x=coords
                sel.set_data({"x":int(x),"y":int(y)});sel.reasoning=f"cnn:c({x},{y})"

            s.pt=tensor;s.pai=aidx if aidx<5 else(5+coords[0]*s.G+coords[1])
            s.pr=raw.copy();s.ph=ch;s.la+=1
            if s.action_counter%s.tfreq==0 and s._wd:s._train()
            return sel

        except Exception as e:
            traceback.print_exc()
            a=random.choice(s.al);a.reasoning=f"err:{e}";return a
