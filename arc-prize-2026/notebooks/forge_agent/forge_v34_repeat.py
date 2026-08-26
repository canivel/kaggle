# =====================================================================
# FORGE v34 — RepeatAction solver for click-heavy games
#
# BASE: forge_v31 (proven 0.26 on leaderboard)
#
# v34 NEW: RepeatAction solver between BFS and MCTS
#
#   BFS max depth=50 never finds solutions requiring 400+ identical clicks.
#   Games like lp85(418), s5i5(595), su15(542) need this.
#   This solver tries each effective action up to 1500x, detects stalls,
#   handles reset-cycles. Runs in <30s per game.
#
#   PIPELINE:
#     1. BFS (fast wins, ~30% games)
#     2. RepeatAction (click-heavy games, 60s budget)
#     3. MCTS+WM fallback (everything else)
#
# CHECKPOINT: /tmp/forge_v34_ckpt.pt
# WM CHECKPOINT: /tmp/forge_v34_wm.pt
# =====================================================================
import copy
import glob
import hashlib
import heapq
import importlib.util
import logging
import math
import os
import random
import time
import traceback
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState, ActionInput

logger = logging.getLogger(__name__)

_CKPT_PATH = '/tmp/forge_v34_ckpt.pt'
_WM_CKPT_PATH = '/tmp/forge_v34_wm.pt'

# ==================== BFS SOLVER (v22, unchanged) ====================

class BFSSolver:
    """Offline BFS solver using direct game class instantiation."""

    def __init__(self, game_path, game_class_name, scan_timeout=5, bfs_timeout=180):
        self.game_path = game_path
        self.class_name = game_class_name
        self.scan_timeout = scan_timeout
        self.bfs_timeout = bfs_timeout
        self.game_cls = None
        self.solutions = {}  # level_idx -> action list
        self._warmup_prefix = []

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

    def _state_hash(self, g, frame, hidden_fields=None):
        fh = hashlib.md5(frame.tobytes()).hexdigest()[:16]
        if hidden_fields:
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

    def _extract_win_field(self):
        try:
            source = open(self.game_path).read()
            lines = source.split('\n')
            for i, line in enumerate(lines):
                if 'self.next_level()' in line:
                    for j in range(i-1, max(0, i-8), -1):
                        s = lines[j].strip()
                        if s.startswith('if ') or s.startswith('elif '):
                            import re
                            m = re.search(r'self\.(\w+)', s)
                            if m:
                                return m.group(1)
                    break
        except:
            pass
        return None

    def _probe_hidden_fields(self, game, actions):
        if not actions:
            return []
        win_field = self._extract_win_field()
        initial = {}
        for k, v in game.__dict__.items():
            if isinstance(v, (int, float, bool)) and not k.startswith('__'):
                initial[k] = v
        changing_fields = set()
        if win_field and win_field in initial:
            changing_fields.add(win_field)
        frame0 = game.get_pixels(0, 0, 64, 64)
        for act_id, data in actions[:10]:
            g = copy.deepcopy(game)
            try:
                ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                g.perform_action(ai, raw=True)
            except:
                continue
            f = g.get_pixels(0, 0, 64, 64)
            for k, v in g.__dict__.items():
                if isinstance(v, (int, float, bool)) and not k.startswith('__'):
                    if k in initial and v != initial[k]:
                        if k not in ('_action_count', '_full_reset', '_action_complete'):
                            changing_fields.add(k)
        hidden = []
        for f in changing_fields:
            if f.startswith('_') and f not in ('_current_level_index', '_score'):
                continue
            hidden.append(f)
        return sorted(hidden)

    def _scan_actions(self, game, f0, bg):
        """Scan effective actions WITHOUT dedup (v15: critical for cd82/sp80 L1)."""
        avail = game._available_actions
        actions = []
        for a in [a for a in avail if a <= 5]:
            g = copy.deepcopy(game)
            try:
                r = g.perform_action(ActionInput(id=GameAction.from_id(a)), raw=True)
                if r.frame and np.sum(f0 != np.array(r.frame[-1])) > 0:
                    actions.append((a, None))
            except:
                pass
        if 6 in avail:
            t0 = time.time()
            hit_positions = []
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
                            raw=True)
                        if not r.frame:
                            continue
                        f = np.array(r.frame[-1])
                        if np.sum(f0 != f) > 0:
                            actions.append((6, {'x': x, 'y': y, 'game_id': 'bfs'}))
                            hit_positions.append((x, y))
                    except:
                        pass
            # Neighbor probe for odd-coord sprites
            for hx, hy in hit_positions:
                if time.time() - t0 > self.scan_timeout + 2:
                    break
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = hx+dx, hy+dy
                    if 0 <= nx < 64 and 0 <= ny < 64 and f0[ny, nx] != bg:
                        g = copy.deepcopy(game)
                        try:
                            r = g.perform_action(
                                ActionInput(id=GameAction.ACTION6, data={'x': nx, 'y': ny, 'game_id': 'bfs'}),
                                raw=True)
                            if r.frame and np.sum(f0 != np.array(r.frame[-1])) > 0:
                                actions.append((6, {'x': nx, 'y': ny, 'game_id': 'bfs'}))
                        except: pass
        return actions

    def solve_level(self, level_idx, max_states=200000, prev_solution=None):
        if not self.game_cls:
            return None
        self._warmup_prefix = []

        game = self.game_cls()
        game.set_level(level_idx)
        game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        r0 = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        if not r0.frame:
            return None
        f0 = np.array(r0.frame[-1])
        bg = int(np.bincount(f0.flatten(), minlength=16).argmax())

        if prev_solution and level_idx > 0:
            transfer_result = self._try_transfer(game, level_idx, prev_solution, f0)
            if transfer_result:
                return transfer_result

        actions = self._scan_actions(game, f0, bg)

        # Warm-up unlock
        if not actions:
            logger.info(f"BFS L{level_idx}: 0 actions found, trying warm-up unlock")
            avail = game._available_actions
            for warmup_id in [a for a in avail if a <= 4]:
                g_warmup = copy.deepcopy(game)
                try:
                    g_warmup.perform_action(ActionInput(id=GameAction.from_id(warmup_id)), raw=True)
                    f_after = np.array(g_warmup.get_pixels(0, 0, 64, 64))
                    warmup_actions = self._scan_actions(g_warmup, f_after, bg)
                    if warmup_actions:
                        logger.info(f"BFS L{level_idx}: UNLOCKED with ACTION{warmup_id}!")
                        game = g_warmup
                        f0 = f_after
                        actions = warmup_actions
                        self._warmup_prefix = [(warmup_id, None)]
                        break
                except:
                    pass

        logger.info(f"BFS L{level_idx}: {len(actions)} effective actions")
        if not actions:
            return None

        # Adaptive depth
        bfs_max_depth = 30
        if len(actions) <= 4:
            bfs_max_depth = 50
        elif len(actions) <= 8:
            bfs_max_depth = 40

        visited = set()
        h0 = self._state_hash(game, f0, None)
        visited.add(h0)
        t0 = time.time()
        explored = 0

        # Standard BFS (proven for 12/25 games)
        queue = deque()
        queue.append((copy.deepcopy(game), [], 0))
        while queue and explored < max_states and (time.time() - t0) < self.bfs_timeout:
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
                h = self._state_hash(g2, f, None)
                if h in visited: continue
                visited.add(h)
                new_hist = hist + [(act_id, data)]
                if r.levels_completed > level_idx or g2._current_level_index > level_idx:
                    logger.info(f"BFS L{level_idx}: SOLVED in {len(new_hist)} actions ({explored} explored, {time.time()-t0:.1f}s)")
                    sol = self._warmup_prefix + new_hist
                    self.solutions[level_idx] = sol
                    return sol
                if depth < bfs_max_depth:
                    queue.append((g2, new_hist, depth + 1))

        elapsed_first = time.time() - t0
        logger.info(f"BFS L{level_idx}: first pass timeout ({explored} explored, {elapsed_first:.1f}s)")

        # ACMD trigger finder
        if len(visited) < 100 and elapsed_first < self.bfs_timeout * 0.8:
            hidden_fields = self._probe_hidden_fields(game, actions)
            if hidden_fields:
                logger.info(f"BFS L{level_idx}: ACMD with fields: {hidden_fields}")
                game2 = self.game_cls()
                game2.set_level(level_idx)
                game2.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                game2.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                r2 = game2.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                f0_2 = np.array(r2.frame[-1])
                init_state = {f: getattr(game2, f, None) for f in hidden_fields}
                visited2 = set()
                h0_2 = self._state_hash(game2, f0_2, hidden_fields)
                visited2.add(h0_2)
                heap2 = [(0, 0, 0, copy.deepcopy(game2), [])]
                fifo2 = 1
                t0_2 = time.time()
                remaining = max(60, self.bfs_timeout - elapsed_first)
                explored2 = 0
                while heap2 and explored2 < max_states and (time.time() - t0_2) < remaining:
                    neg_delta, depth, _, g, hist = heapq.heappop(heap2)
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
                            logger.info(f"BFS L{level_idx}: SOLVED (ACMD) in {len(new_hist)} actions")
                            sol = self._warmup_prefix + new_hist
                            self.solutions[level_idx] = sol
                            return sol
                        trigger_delta = sum(abs(getattr(g2, tf, 0) - init_state.get(tf, 0))
                                           if isinstance(getattr(g2, tf, 0), (int, float)) else
                                           int(getattr(g2, tf, None) != init_state.get(tf))
                                           for tf in hidden_fields)
                        pixels_changed = np.sum(f0_2 != f) > 0
                        if not pixels_changed and trigger_delta == 0:
                            continue
                        fifo2 += 1
                        if depth < 40:
                            heapq.heappush(heap2, (-trigger_delta, depth+1, fifo2, g2, new_hist))
                logger.info(f"BFS L{level_idx}: ACMD finished ({explored2} explored, {time.time()-t0_2:.1f}s)")

        # IDDFS for low-branching games
        elapsed_total = time.time() - t0
        remaining_time = max(30, self.bfs_timeout - elapsed_total)
        if len(actions) <= 6 and remaining_time > 30:
            logger.info(f"BFS L{level_idx}: IDDFS (branching={len(actions)}, {remaining_time:.0f}s)")
            game3 = self.game_cls()
            game3.set_level(level_idx)
            game3.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            game3.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            t0_3 = time.time()
            iddfs_solved = False
            for max_depth in range(10, 60):
                if time.time() - t0_3 > remaining_time * 0.6:
                    break
                stack = [(copy.deepcopy(game3), [], set())]
                while stack and (time.time() - t0_3) < remaining_time * 0.6:
                    g, hist, path_hashes = stack.pop()
                    if len(hist) >= max_depth:
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
                        if h in path_hashes: continue
                        new_hist = hist + [(act_id, data)]
                        if r.levels_completed > level_idx or g2._current_level_index > level_idx:
                            logger.info(f"BFS L{level_idx}: SOLVED (IDDFS depth={max_depth}) in {len(new_hist)} actions")
                            sol = self._warmup_prefix + new_hist
                            self.solutions[level_idx] = sol
                            return sol
                        stack.append((g2, new_hist, path_hashes | {h}))
            if iddfs_solved:
                return None  # handled above

            # NEW v24: Beam search after IDDFS for medium-branching games
            beam_remaining = max(15, remaining_time - (time.time() - t0_3))
            if beam_remaining > 15 and len(actions) <= 12:
                logger.info(f"BFS L{level_idx}: Beam search (width=20, {beam_remaining:.0f}s)")
                game4 = self.game_cls()
                game4.set_level(level_idx)
                game4.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                r4 = game4.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                f4 = np.array(r4.frame[-1])
                beam = [(copy.deepcopy(game4), [], 0)]  # (game, hist, score)
                beam_visited = {hashlib.md5(f4.tobytes()).hexdigest()[:16]}
                t0_b = time.time()
                for bdepth in range(60):
                    if time.time() - t0_b > beam_remaining or not beam:
                        break
                    next_beam = []
                    for bg_game, bg_hist, _ in beam:
                        for act_id, data in actions:
                            g2 = copy.deepcopy(bg_game)
                            try:
                                ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                                r = g2.perform_action(ai, raw=True)
                            except: continue
                            if not r.frame: continue
                            f = np.array(r.frame[-1])
                            fh = hashlib.md5(f.tobytes()).hexdigest()[:16]
                            if fh in beam_visited: continue
                            beam_visited.add(fh)
                            new_hist = bg_hist + [(act_id, data)]
                            if r.levels_completed > level_idx or g2._current_level_index > level_idx:
                                logger.info(f"BFS L{level_idx}: SOLVED (Beam depth={bdepth}) in {len(new_hist)} actions")
                                sol = self._warmup_prefix + new_hist
                                self.solutions[level_idx] = sol
                                return sol
                            score = int(np.sum(f4 != f))  # change from start = progress
                            next_beam.append((g2, new_hist, score))
                    next_beam.sort(key=lambda x: -x[2])
                    beam = next_beam[:20]
                logger.info(f"BFS L{level_idx}: Beam search exhausted")

        return None

    def _try_transfer(self, game, level_idx, prev_solution, f1):
        try:
            g = copy.deepcopy(game)
            for i, (act_id, data) in enumerate(prev_solution):
                try:
                    ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                    r = g.perform_action(ai, raw=True)
                    if r.levels_completed > level_idx or g._current_level_index > level_idx:
                        logger.info(f"BFS L{level_idx}: TRANSFER SUCCESS (direct, {i+1} actions)")
                        sol = prev_solution[:i+1]
                        self.solutions[level_idx] = sol
                        return sol
                except:
                    break

            prev_game = self.game_cls()
            prev_game.set_level(level_idx - 1)
            prev_game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            r_prev = prev_game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            if not r_prev.frame:
                return None
            f0 = np.array(r_prev.frame[-1])
            bg = int(np.bincount(f0.flatten(), minlength=16).argmax())

            def get_objects(frame, bg_c):
                objs = []
                for c in range(16):
                    if c == bg_c: continue
                    mask = (frame == c); npix = int(np.sum(mask))
                    if npix < 2: continue
                    ys, xs = np.where(mask)
                    objs.append({'color': c, 'cx': float(np.mean(xs)), 'cy': float(np.mean(ys)), 'n': npix})
                return sorted(objs, key=lambda o: (o['color'], -o['n']))

            objs_prev = get_objects(f0, bg)
            objs_curr = get_objects(f1, bg)
            if not objs_prev or not objs_curr:
                return None

            matched = []
            for op in objs_prev:
                best = None; best_dist = float('inf')
                for oc in objs_curr:
                    if oc['color'] == op['color'] and abs(oc['n'] - op['n']) < max(op['n'], oc['n']) * 0.5:
                        d = abs(oc['cx'] - op['cx']) + abs(oc['cy'] - op['cy'])
                        if d < best_dist: best_dist = d; best = oc
                if best: matched.append((op, best))
            if not matched: return None

            dx = np.mean([m[1]['cx'] - m[0]['cx'] for m in matched])
            dy = np.mean([m[1]['cy'] - m[0]['cy'] for m in matched])

            transferred = []
            for act_id, data in prev_solution:
                if data and 'x' in data:
                    new_data = dict(data)
                    new_data['x'] = max(0, min(63, int(data['x'] + dx)))
                    new_data['y'] = max(0, min(63, int(data['y'] + dy)))
                    transferred.append((act_id, new_data))
                else:
                    transferred.append((act_id, data))

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
                except: break

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
                g = copy.deepcopy(game)
                for i, (act_id, data) in enumerate(expanded):
                    try:
                        ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                        r = g.perform_action(ai, raw=True)
                        if r.levels_completed > level_idx or g._current_level_index > level_idx:
                            logger.info(f"BFS L{level_idx}: TRANSFER SUCCESS (multiplier={multiplier}, {i+1} actions)")
                            sol = expanded[:i+1]
                            self.solutions[level_idx] = sol
                            return sol
                    except: break
        except Exception as e:
            logger.warning(f"BFS transfer failed: {e}")
        return None


# ==================== ACTION SCANNER (for MCTS — unique-effect dedup) ====================

class ActionScanner:
    """Scan unique-effect actions with frame-hash dedup for MCTS.

    Unlike BFS scan (no dedup), MCTS needs a small action set to be tractable.
    Dedup reduces 5000 click positions -> 5-20 unique frame effects.
    """

    def __init__(self, timeout=4.0):
        self.timeout = timeout

    def _hidden_state(self, g) -> str:
        """Compact hash of game's internal numeric fields for change detection."""
        try:
            parts = []
            for k, v in sorted(g.__dict__.items()):
                if isinstance(v, (int, float, bool)) and not k.startswith('__'):
                    parts.append(f"{k}={v}")
            return "|".join(parts[:20])
        except:
            return ""

    def scan(self, game, f0, bg):
        """Return (act_id, data) list with unique effects.

        v30: Also keep clicks that change INTERNAL state (hidden fields),
        not just visible frame. Fixes selection-based games like tn36 where
        clicking selects a piece (invisible) then clicking destination moves it.
        Uses a grid of positions to ensure broad coverage.
        """
        avail = game._available_actions
        actions = []
        seen_effects: Dict[str, Tuple] = {}
        initial_hidden = self._hidden_state(game)

        # Non-click actions
        for a in avail:
            if a > 5:
                continue
            g = copy.deepcopy(game)
            try:
                r = g.perform_action(ActionInput(id=GameAction.from_id(a)), raw=True)
                if r.frame:
                    f = np.array(r.frame[-1])
                    hidden_changed = self._hidden_state(g) != initial_hidden
                    if np.sum(f0 != f) > 0 or hidden_changed:
                        ek = hashlib.md5(f.tobytes()).hexdigest()[:16]
                        if ek not in seen_effects:
                            seen_effects[ek] = (a, None)
                            actions.append((a, None))
            except:
                pass

        if 6 not in avail:
            return actions

        t0 = time.time()

        # Foreground pixels — likely interactive objects
        ys, xs = np.where(f0 != bg)
        fg_candidates = list(zip(ys.tolist(), xs.tolist()))
        if len(fg_candidates) > 300:
            step = max(1, len(fg_candidates) // 300)
            fg_candidates = fg_candidates[::step]

        # Background sample
        by, bx = np.where(f0 == bg)
        bg_candidates = []
        if len(by) > 0:
            bidx = np.linspace(0, len(by)-1, min(30, len(by)), dtype=int)
            bg_candidates = [(int(by[i]), int(bx[i])) for i in bidx]

        # Grid sample — covers the whole screen evenly (catches selection games)
        # 8x8 = 64 evenly-spaced points regardless of pixel content
        grid_candidates = [
            (int(y), int(x))
            for y in np.linspace(0, 63, 8, dtype=int)
            for x in np.linspace(0, 63, 8, dtype=int)
        ]

        seen_pos: set = set()
        all_candidates = fg_candidates + bg_candidates + grid_candidates

        for y, x in all_candidates:
            pos_key = (y // 2, x // 2)  # coarse dedup of near-identical positions
            if pos_key in seen_pos:
                continue
            if time.time() - t0 > self.timeout:
                break
            g = copy.deepcopy(game)
            try:
                r = g.perform_action(
                    ActionInput(id=GameAction.ACTION6, data={'x': x, 'y': y, 'game_id': 'mcts'}),
                    raw=True)
                if not r.frame:
                    continue
                f = np.array(r.frame[-1])
                hidden_changed = self._hidden_state(g) != initial_hidden
                if np.sum(f0 != f) > 0 or hidden_changed:
                    ek = hashlib.md5(f.tobytes()).hexdigest()[:16]
                    hk = f"{ek}|{self._hidden_state(g)[:32]}"
                    if hk not in seen_effects:
                        seen_effects[hk] = (6, {'x': x, 'y': y, 'game_id': 'mcts'})
                        actions.append((6, {'x': x, 'y': y, 'game_id': 'mcts'}))
                    seen_pos.add(pos_key)
            except:
                pass

        return actions


# ==================== WORLD MODEL ====================

class WorldModel(nn.Module):
    """Online-learned transition model: f(frame, action) -> next_frame.

    Trained from every real interaction. Generalizes to any game by learning
    the game's pixel-level physics. Used for:
      1. Curiosity UCB bonus (prediction error = novel state = explore more)
      2. Fast WM rollouts (no deepcopy, 10x faster than real game simulation)
    """
    G = 64
    N_COLORS = 16
    N_ACTIONS = 7  # actions 1-6 + reset
    HIDDEN = 256

    def __init__(self):
        super().__init__()
        # Compact encoder: frame -> latent
        self.encoder = nn.Sequential(
            nn.Conv2d(self.N_COLORS, 32, 4, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),             # 16x16
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),             # 8x8
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, self.HIDDEN),
            nn.ReLU(),
        )
        # Action embedding
        self.action_embed = nn.Embedding(self.N_ACTIONS + 1, 32)
        # Decoder: latent + action -> next frame logits
        self.decoder = nn.Sequential(
            nn.Linear(self.HIDDEN + 32, self.HIDDEN),
            nn.ReLU(),
            nn.Linear(self.HIDDEN, self.HIDDEN),
            nn.ReLU(),
            nn.Linear(self.HIDDEN, self.N_COLORS * self.G * self.G),
        )

    def forward(self, frame_oh: torch.Tensor, action_idx: torch.Tensor) -> torch.Tensor:
        """frame_oh: (B, N_COLORS, G, G), action_idx: (B,) -> (B, N_COLORS, G, G)"""
        z = self.encoder(frame_oh)
        a = self.action_embed(action_idx.clamp(0, self.N_ACTIONS))
        pred = self.decoder(torch.cat([z, a], dim=1))
        return pred.view(-1, self.N_COLORS, self.G, self.G)

    @staticmethod
    def frame_to_oh(frame_np: np.ndarray, device) -> torch.Tensor:
        oh = torch.zeros(16, 64, 64, dtype=torch.float32)
        oh.scatter_(0, torch.from_numpy(frame_np.astype(np.int64)).unsqueeze(0), 1)
        return oh.to(device)

    def predict(self, frame_np: np.ndarray, action_idx: int, device) -> np.ndarray:
        """Predict next frame (numpy int64 array)."""
        with torch.no_grad():
            oh = self.frame_to_oh(frame_np, device).unsqueeze(0)
            act = torch.tensor([action_idx], device=device)
            pred = self.forward(oh, act).squeeze(0)  # (N_COLORS, G, G)
        return pred.argmax(0).cpu().numpy().astype(np.int64)

    def curiosity(self, frame_np: np.ndarray, action_idx: int,
                  next_frame_np: np.ndarray, device) -> float:
        """WM prediction error = curiosity signal. High error = novel transition."""
        with torch.no_grad():
            oh = self.frame_to_oh(frame_np, device).unsqueeze(0)
            act = torch.tensor([action_idx], device=device)
            pred = self.forward(oh, act).squeeze(0)          # (N_COLORS, G, G)
            target = torch.from_numpy(next_frame_np.astype(np.int64)).to(device)  # (G,G)
            loss = F.cross_entropy(
                pred.view(self.N_COLORS, -1).t(),
                target.view(-1)
            )
        return float(loss)


class WorldModelTrainer:
    """Online trainer for WorldModel. Maintains a replay buffer of transitions."""

    BUFFER_SIZE = 5000
    BATCH_SIZE = 32
    LR = 5e-4
    TRAIN_EVERY = 10   # real interactions between gradient steps
    WARMUP = 30        # min transitions before switching to WM rollouts

    def __init__(self, device):
        self.wm = WorldModel().to(device)
        self.opt = optim.Adam(self.wm.parameters(), lr=self.LR)
        self.device = device
        self.buffer: deque = deque(maxlen=self.BUFFER_SIZE)
        self.trained_steps = 0
        self.interaction_count = 0

    def add(self, frame: np.ndarray, action_idx: int, next_frame: np.ndarray):
        """Record a real transition for WM training."""
        self.buffer.append((frame.copy(), action_idx, next_frame.copy()))
        self.interaction_count += 1
        if self.interaction_count % self.TRAIN_EVERY == 0 and len(self.buffer) >= self.BATCH_SIZE:
            self._train_step()

    def _train_step(self):
        indices = np.random.choice(len(self.buffer), self.BATCH_SIZE, replace=False)
        batch = [self.buffer[i] for i in indices]
        frames = torch.stack([WorldModel.frame_to_oh(b[0], self.device) for b in batch])
        acts = torch.tensor([b[1] for b in batch], device=self.device)
        targets = torch.stack([
            torch.from_numpy(b[2].astype(np.int64)).to(self.device) for b in batch
        ])  # (B, G, G)
        pred = self.wm(frames, acts)  # (B, N_COLORS, G, G)
        # Reshape for cross-entropy: (B*G*G, N_COLORS) vs (B*G*G,)
        B, C, G, _ = pred.shape
        loss = F.cross_entropy(
            pred.permute(0, 2, 3, 1).reshape(-1, C),
            targets.reshape(-1)
        )
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0)
        self.opt.step()
        self.trained_steps += 1

    @property
    def ready_for_rollout(self) -> bool:
        return len(self.buffer) >= self.WARMUP

    def wm_rollout(self, frame_np: np.ndarray, actions: List[Tuple],
                   max_depth: int = 20, win_threshold: float = 0.4) -> float:
        """Simulate a rollout using the world model. No game deepcopy needed.

        Returns: estimated value (0-1). High predicted frame change = potential win.
        """
        current = frame_np.copy()
        prev = frame_np.copy()
        total_change = 0.0
        for step in range(max_depth):
            act_id, _ = random.choice(actions)
            act_idx = min(act_id - 1, WorldModel.N_ACTIONS - 1)
            next_frame = self.wm.predict(current, act_idx, self.device)
            change = float(np.sum(current != next_frame)) / (64 * 64)
            total_change += change
            # Dramatic change at end of rollout = potential win signal
            if change > win_threshold and step > 2:
                return min(1.0, change * 2)
            current = next_frame
        # Return normalized total change as progress signal
        return min(0.5, total_change / max_depth)

    def curiosity_for(self, frame_np: np.ndarray, action_idx: int,
                      next_frame_np: np.ndarray) -> float:
        if self.trained_steps < 5:
            return 0.5  # uniform curiosity before WM is trained
        return self.wm.curiosity(frame_np, action_idx, next_frame_np, self.device)

    def save(self, path: str):
        try:
            torch.save({'wm': self.wm.state_dict(), 'steps': self.trained_steps}, path)
        except: pass

    def load(self, path: str):
        try:
            if os.path.exists(path):
                ck = torch.load(path, map_location=self.device, weights_only=True)
                self.wm.load_state_dict(ck['wm'])
                self.trained_steps = ck.get('steps', 0)
                logger.info(f"WM: loaded from {path} ({self.trained_steps} steps)")
        except: pass


# ==================== MCTS SOLVER ====================

_UCB_C = 1.8
_MAX_MCTS_ITER = 100_000  # effectively time-limited; removed 3000 cap that choked MCTS
_MAX_MCTS_DEPTH = 50  # slightly deeper — L1/L2 solutions may need longer paths
_TOTAL_GAMES = 25
_TOTAL_BUDGET = 8 * 3600 - 300  # 8h minus 5min safety margin


class MCTSNode:
    __slots__ = ['parent', 'action', 'children', 'visits', 'value',
                 'untried', 'depth', 'terminal', 'novel_bonus', 'game_state']

    def __init__(self, parent, action, untried, depth=0, novel_bonus=0.0):
        self.parent = parent
        self.action = action
        self.children: List['MCTSNode'] = []
        self.visits = 0
        self.value = 0.0
        self.untried = list(untried)
        self.depth = depth
        self.terminal = False
        self.novel_bonus = novel_bonus
        self.game_state = None  # deepcopy of game at this node

    def ucb(self) -> float:
        if self.visits == 0:
            return float('inf')
        exploit = self.value / self.visits
        explore = _UCB_C * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploit + explore + self.novel_bonus / (1 + self.visits)

    def best_child(self) -> 'MCTSNode':
        return max(self.children, key=lambda n: n.ucb())

    def is_fully_expanded(self) -> bool:
        return len(self.untried) == 0


class MCTSSolver:
    """MCTS with World Model curiosity UCB + WM-guided rollouts."""

    def __init__(self, game_cls, level_idx: int, actions: List[Tuple],
                 hidden_fields: Optional[List[str]] = None,
                 timeout: float = 90.0,
                 prev_solution: Optional[List] = None,
                 productive_actions: Optional[List[Tuple]] = None,
                 target_length: Optional[int] = None,
                 world_model: Optional['WorldModelTrainer'] = None):
        self.game_cls = game_cls
        self.level_idx = level_idx
        self.actions = actions
        self.hidden_fields = hidden_fields or []
        self.timeout = timeout
        self.prev_solution = prev_solution or []
        self.rollout_actions = productive_actions if productive_actions else actions
        self.target_length = target_length
        self.world_model: Optional[WorldModelTrainer] = world_model  # v30: WM for curiosity+rollout
        self.visit_counts: Dict[str, int] = {}
        self.solution: Optional[List] = None
        self.explored_transitions: List[Tuple] = []
        self._action_change_stats: Dict[int, List[float]] = {
            i: [1.0, 1.0] for i in range(len(self.actions))
        }
        # Map action tuple -> index for fast lookup
        self._act_to_idx = {a: i for i, a in enumerate(self.actions)}

    def _beats_target(self, solution) -> bool:
        """v29: check if solution beats BFS fallback target or is ≤2x baseline."""
        if self.target_length is not None and len(solution) <= self.target_length:
            return True
        baseline = getattr(self, '_baseline_actions', None)
        if baseline and len(solution) <= 2 * baseline:
            return True
        return False

    def _make_game(self):
        g = self.game_cls()
        g.set_level(self.level_idx)
        g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        return g

    def _hidden_str(self, g) -> str:
        if not self.hidden_fields:
            return ""
        parts = []
        for f in self.hidden_fields:
            v = getattr(g, f, None)
            if v is not None:
                parts.append(f"{f}={v}")
        return "|".join(parts)

    def _state_key(self, g, frame: np.ndarray) -> str:
        fh = hashlib.md5(frame.tobytes()).hexdigest()[:16]
        hs = self._hidden_str(g)
        return fh + hs if hs else fh

    def _is_win(self, g, r) -> bool:
        return (r.levels_completed > self.level_idx or
                g._current_level_index > self.level_idx)

    def _novelty(self, key: str) -> float:
        cnt = self.visit_counts.get(key, 0)
        return 1.0 / math.sqrt(cnt + 1)

    def _weighted_rollout_action(self) -> Tuple:
        """Pick action weighted by historical frame-change productivity."""
        pool = self.rollout_actions if self.rollout_actions else self.actions
        # Map pool indices back to self.actions for stats lookup
        try:
            weights = []
            for act in pool:
                try:
                    idx = self.actions.index(act)
                    s = self._action_change_stats[idx]
                    weights.append(s[0] / max(s[1], 1))
                except (ValueError, KeyError):
                    weights.append(1.0)
            total = sum(weights)
            if total <= 0:
                return random.choice(pool)
            r = random.random() * total
            cumsum = 0.0
            for act, w in zip(pool, weights):
                cumsum += w
                if cumsum >= r:
                    return act
            return pool[-1]
        except Exception:
            return random.choice(pool)

    def _update_rollout_stats(self, act: Tuple, frame_change: float):
        """Update frame-change stats for an action after observing its effect."""
        try:
            idx = self.actions.index(act)
            self._action_change_stats[idx][0] += frame_change
            self._action_change_stats[idx][1] += 1.0
        except (ValueError, KeyError):
            pass

    def _rollout(self, g, max_depth: int = 10) -> float:
        """v30: Use WM rollout when ready (faster, deeper), otherwise real-game rollout."""
        # Try WM rollout first — no deepcopy, 20-step lookahead, feeds curiosity
        if self.world_model and self.world_model.ready_for_rollout:
            try:
                frame_np = np.array(g.get_pixels(0, 0, 64, 64))
                return self.world_model.wm_rollout(frame_np, self.actions, max_depth=20)
            except:
                pass  # fall through to real rollout

        # Real-game rollout with frame-change weighted sampling
        prev_frame = None
        try:
            f = g.get_pixels(0, 0, 64, 64)
            prev_frame = np.array(f)
        except:
            pass
        for _ in range(max_depth):
            act = self._weighted_rollout_action()
            act_id, data = act
            try:
                ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                r = g.perform_action(ai, raw=True)
                if not r.frame:
                    return 0.0
                if self._is_win(g, r):
                    return 1.0
                if prev_frame is not None:
                    try:
                        new_frame = np.array(r.frame[-1])
                        change = float(np.sum(prev_frame != new_frame)) / (64 * 64)
                        self._update_rollout_stats(act, change)
                        # Feed real transition to WM for training
                        if self.world_model:
                            act_idx = min(act_id - 1, WorldModel.N_ACTIONS - 1)
                            self.world_model.add(prev_frame, act_idx, new_frame)
                        prev_frame = new_frame
                    except:
                        pass
            except:
                return 0.0
        return 0.0

    def _try_transfer(self) -> Optional[List]:
        if not self.prev_solution:
            return None
        g = self._make_game()
        for i, (act_id, data) in enumerate(self.prev_solution):
            try:
                ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                r = g.perform_action(ai, raw=True)
                if r.frame and self._is_win(g, r):
                    logger.info(f"MCTS L{self.level_idx}: transfer win in {i+1} steps")
                    return self.prev_solution[:i+1]
            except:
                break
        return None

    def solve(self) -> Optional[List]:
        if not self.actions:
            return None
        t0 = time.time()

        # Try cross-level solution transfer first (fast)
        xfer = self._try_transfer()
        if xfer:
            self.solution = xfer
            return xfer

        # Initialize root
        g0 = self._make_game()
        try:
            r0 = g0.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            if not r0.frame:
                return None
            f0 = np.array(r0.frame[-1])
        except:
            return None

        root_key = self._state_key(g0, f0)
        self._root_frame = f0  # for transition reward computation
        self.visit_counts[root_key] = 1
        root = MCTSNode(parent=None, action=None, untried=self.actions,
                        depth=0, novel_bonus=self._novelty(root_key))
        root.game_state = copy.deepcopy(g0)

        # path_store: node_id -> action sequence to reach node
        path_store: Dict[int, List] = {id(root): []}

        n_iter = 0
        while time.time() - t0 < self.timeout and n_iter < _MAX_MCTS_ITER:
            n_iter += 1

            # SELECTION: traverse tree to leaf
            node = root
            g = copy.deepcopy(root.game_state)

            while node.is_fully_expanded() and node.children and not node.terminal:
                node = node.best_child()
                act_id, data = node.action
                try:
                    ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                    r = g.perform_action(ai, raw=True)
                    if r.frame:
                        f = np.array(r.frame[-1])
                        if self._is_win(g, r):
                            full_path = path_store.get(id(node.parent), []) + [(act_id, data)]
                            logger.info(f"MCTS L{self.level_idx}: WIN (selection) {len(full_path)} steps, {n_iter} iters, {time.time()-t0:.1f}s")
                            self.solution = full_path
                            if self._beats_target(full_path):
                                logger.info(f"MCTS L{self.level_idx}: target met (selection), stopping early")
                            return full_path
                    else:
                        node.terminal = True
                        break
                except:
                    node.terminal = True
                    break

            if node.terminal or not node.untried:
                # Backprop zero reward for terminal/stuck nodes
                n = node
                while n is not None:
                    n.visits += 1
                    n = n.parent
                continue

            # EXPANSION: try one untried action
            act = node.untried.pop(random.randrange(len(node.untried)))
            act_id, data = act
            g_exp = copy.deepcopy(g)
            reward = 0.0
            child = None
            try:
                ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                r = g_exp.perform_action(ai, raw=True)
                if r.frame:
                    f = np.array(r.frame[-1])
                    new_path = path_store.get(id(node), []) + [(act_id, data)]

                    if self._is_win(g_exp, r):
                        logger.info(f"MCTS L{self.level_idx}: WIN (expand) {len(new_path)} steps, {n_iter} iters, {time.time()-t0:.1f}s")
                        self.solution = new_path
                        if self._beats_target(new_path):
                            logger.info(f"MCTS L{self.level_idx}: target met (expand), stopping early")
                        return new_path

                    sk = self._state_key(g_exp, f)
                    self.visit_counts[sk] = self.visit_counts.get(sk, 0) + 1
                    novel = self._novelty(sk)

                    # v30: curiosity bonus from WM prediction error
                    # High WM error = model hasn't seen this transition = novel = explore more
                    curiosity_bonus = 0.0
                    if self.world_model and self.world_model.trained_steps >= 5:
                        try:
                            parent_frame = np.array(node.game_state.get_pixels(0, 0, 64, 64))
                            act_idx = min(act_id - 1, WorldModel.N_ACTIONS - 1)
                            curiosity_bonus = self.world_model.curiosity_for(parent_frame, act_idx, f)
                            # Also train WM on this real transition
                            self.world_model.add(parent_frame, act_idx, f)
                        except:
                            pass
                    novel = novel + 0.3 * curiosity_bonus  # blend novelty + curiosity

                    # Record transition for CNN buffer seeding
                    rf = getattr(self, '_root_frame', None)
                    frame_reward = 0.5 if (rf is not None and np.sum(rf != f) > 0) else -0.05
                    self.explored_transitions.append((act_id, data, f, frame_reward))
                    if len(self.explored_transitions) > 2000:
                        self.explored_transitions = self.explored_transitions[-1000:]

                    if node.depth < _MAX_MCTS_DEPTH:
                        child = MCTSNode(parent=node, action=(act_id, data),
                                        untried=list(self.actions),
                                        depth=node.depth + 1, novel_bonus=novel)
                        child.game_state = copy.deepcopy(g_exp)
                        node.children.append(child)
                        path_store[id(child)] = new_path

                        # ROLLOUT from child
                        reward = self._rollout(copy.deepcopy(g_exp))
            except:
                reward = 0.0

            # BACKPROPAGATION
            n = child if child else node
            while n is not None:
                n.visits += 1
                n.value += reward
                n = n.parent

        logger.info(f"MCTS L{self.level_idx}: no solution in {n_iter} iters, {time.time()-t0:.1f}s")
        return None


# ==================== GAME SOURCE FINDER ====================

def find_game_source_and_class(game_id, arc_env=None):
    gid = game_id.split('-')[0]
    cls_name = gid.capitalize()
    if len(gid) == 4 and gid[0].isalpha():
        cls_name = gid[0].upper() + gid[1:]

    src = None
    import re

    if arc_env and hasattr(arc_env, 'environment_info'):
        ei = arc_env.environment_info
        if hasattr(ei, 'local_dir') and ei.local_dir:
            from pathlib import Path
            ld = Path(ei.local_dir)
            for candidate in [ld / f"{gid}.py", ld / f"{cls_name.lower()}.py"]:
                if candidate.exists():
                    src = str(candidate)
                    content = candidate.read_text()[:2000]
                    m = re.search(r'class\s+(\w+)\s*\(\s*ARCBaseGame', content)
                    if m: cls_name = m.group(1)
                    break

    if not src:
        for pattern in [
            f"environment_files/{gid}/**/{gid}.py",
            f"/kaggle/working/environment_files/{gid}/**/{gid}.py",
        ]:
            matches = glob.glob(pattern, recursive=True)
            if matches:
                src = matches[0]
                content = open(src).read()[:2000]
                m = re.search(r'class\s+(\w+)\s*\(\s*ARCBaseGame', content)
                if m: cls_name = m.group(1)
                break

    if not src:
        for pattern in [
            f"/tmp/*/{gid}/*/{gid}.py",
            f"/kaggle/*/{gid}*/{gid}.py",
            f"/kaggle/input/**/{gid}*/**/{gid}.py",
        ]:
            matches = glob.glob(pattern, recursive=True)
            if matches:
                src = matches[0]
                content = open(src).read()[:2000]
                m = re.search(r'class\s+(\w+)\s*\(\s*ARCBaseGame', content)
                if m: cls_name = m.group(1)
                break

    return src, cls_name


# ==================== CNN (v22 unchanged) ====================

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
        # BFS
        s._bfs = None
        s._bfs_solution = None
        s._bfs_step = 0
        s._bfs_tried = False
        s._bfs_fallback = None  # v29: BFS path stored pending MCTS comparison
        # MCTS
        s._mcts_solution = None
        s._mcts_step = 0
        s._mcts_solutions: Dict[int, List] = {}  # level -> solution
        s._mcts_seed_transitions: List = []       # transitions for CNN seeding
        s._scanner = ActionScanner(timeout=4.0)
        # v28: game budget tracking
        s._games_completed = 0       # games finished so far (for per-game budget calc)
        s._game_start_time = time.time()  # reset at each new game
        # baseline awareness
        s._game_baseline: Optional[List[int]] = None
        # v30: World Model — learns game physics online, shared across games
        s._wm_trainer: Optional[WorldModelTrainer] = None
        try:
            s._wm_trainer = WorldModelTrainer(s.device)
            s._wm_trainer.load(_WM_CKPT_PATH)
        except Exception as e:
            logger.warning(f"WM: init failed: {e}")
        try:
            if hasattr(s, 'arc_env') and s.arc_env and hasattr(s.arc_env, 'environment_info'):
                ei = s.arc_env.environment_info
                if hasattr(ei, 'baseline_actions'):
                    s._game_baseline = list(ei.baseline_actions)
        except:
            pass

    def append_frame(s, f):
        s.frames.append(f)
        if len(s.frames) > s._MAX_FRAMES: s.frames = s.frames[-s._MAX_FRAMES:]
        if f.guid: s.guid = f.guid
        if hasattr(s, "recorder") and not s.is_playback:
            import json; s.recorder.record(json.loads(f.model_dump_json()))

    def _lvl(s, f): return getattr(f, 'score', None) or f.levels_completed
    def _raw(s, fd): return np.array(fd.frame, dtype=np.int64)[-1]

    def _init_bfs(s):
        src, cls = find_game_source_and_class(s.game_id, s.arc_env)
        if src:
            s._bfs = BFSSolver(src, cls, scan_timeout=5, bfs_timeout=180)
            if s._bfs.load():
                logger.info(f"BFS: loaded {cls} from {src}")
            else:
                s._bfs = None
                logger.warning(f"BFS: failed to load {cls}")
        else:
            logger.warning(f"BFS: source not found for {s.game_id}")

    def _try_bfs_solve(s, level_idx):
        """v29: BFS finds path and stores as fallback. Does NOT set _bfs_solution.
        Caller decides whether to use BFS or wait for MCTS comparison."""
        if s._bfs is None:
            return None

        # BFS budget: up to 40% of per-game budget, max 150s L0 / 90s L1+.
        # This ensures MCTS always gets at least 50% of the game budget.
        game_budget = s._per_game_budget()
        max_bfs = 150 if level_idx == 0 else 90
        time_for_bfs = min(max_bfs, game_budget * 0.40)
        time_for_bfs = max(30.0, time_for_bfs)
        s._bfs.bfs_timeout = time_for_bfs
        logger.info(f"BFS L{level_idx}: budget={time_for_bfs:.0f}s (game_budget={game_budget:.0f}s)")

        prev_sol = s._bfs.solutions.get(level_idx - 1) if level_idx > 0 else None
        sol = s._bfs.solve_level(level_idx, prev_solution=prev_sol)

        if sol:
            # v29: store as fallback, don't commit to _bfs_solution yet — MCTS may beat it
            s._bfs_fallback = sol
            human_baseline = (s._game_baseline[level_idx]
                              if s._game_baseline and level_idx < len(s._game_baseline)
                              else "unknown")
            logger.info(f"BFS L{level_idx}: found path len={len(sol)} "
                        f"(baseline={human_baseline}). Running MCTS to find shorter.")

            # v30: pre-train WM by replaying BFS solution through fresh game instance.
            # This gives WM basic game physics understanding before MCTS starts.
            if s._wm_trainer is not None and s._bfs.game_cls is not None:
                try:
                    g_replay = s._bfs.game_cls()
                    g_replay.set_level(level_idx)
                    g_replay.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                    g_replay.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                    prev_frame = np.array(g_replay.get_pixels(0, 0, 64, 64))
                    for act_id, data in sol:
                        try:
                            if data:
                                ai = ActionInput(id=GameAction.from_id(act_id), data=data)
                            else:
                                ai = ActionInput(id=GameAction.from_id(act_id))
                            r = g_replay.perform_action(ai, raw=True)
                            if r and r.frame:
                                next_frame = np.array(r.frame[-1])
                                wm_idx = min(act_id - 1, WorldModel.N_ACTIONS - 1)
                                s._wm_trainer.add(prev_frame, wm_idx, next_frame)
                                prev_frame = next_frame
                        except Exception:
                            pass
                    logger.info(f"WM: pre-trained from BFS path "
                                f"({len(sol)} transitions, trained_steps={s._wm_trainer.trained_steps})")
                except Exception as e:
                    logger.warning(f"WM pre-train failed: {e}")

            # v31: shorten BFS solution to improve efficiency score
            game_budget = s._per_game_budget()
            shorten_budget = min(20.0, game_budget * 0.05)
            sol = s._shorten_solution(s._bfs.game_cls, level_idx, sol, time_limit=shorten_budget)
            s._bfs_fallback = sol  # update fallback with shortened version

            return sol
        return None

    def _per_game_budget(s) -> float:
        """Estimate how much time we can spend on this game."""
        elapsed_total = time.time() - s.start_time
        remaining_total = _TOTAL_BUDGET - elapsed_total
        games_left = max(1, _TOTAL_GAMES - s._games_completed)
        per_game = remaining_total / games_left
        return max(120.0, per_game)  # at least 2 min per game

    def _shorten_solution(s, game_cls, level_idx, solution, time_limit: float = 20.0) -> list:
        """v31: greedily remove redundant actions from any found solution.
        Scoring is (baseline/actions)^2 — every step removed is a significant gain.
        E.g. shortening from 14 to 9 steps doubles the score (25% -> 60%)."""
        if game_cls is None or len(solution) <= 1:
            return solution
        t0 = time.time()

        def verify(sol):
            try:
                g = game_cls()
                g.set_level(level_idx)
                g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                for act_id, data in sol:
                    ai = (ActionInput(id=GameAction.from_id(act_id), data=data) if data
                          else ActionInput(id=GameAction.from_id(act_id)))
                    r = g.perform_action(ai, raw=True)
                    if r and (r.levels_completed > level_idx or
                              getattr(g, '_current_level_index', level_idx) > level_idx):
                        return True
                    if r and r.state == GameState.GAME_OVER:
                        return False
                return False
            except Exception:
                return False

        shortened = list(solution)
        improved = True
        while improved and time.time() - t0 < time_limit:
            improved = False
            for i in range(len(shortened)):
                if time.time() - t0 >= time_limit:
                    break
                candidate = shortened[:i] + shortened[i + 1:]
                if verify(candidate):
                    shortened = candidate
                    improved = True
                    break  # restart sweep from beginning
        if len(shortened) < len(solution):
            logger.info(f"Solution shortened L{level_idx}: {len(solution)} -> {len(shortened)} steps")
        return shortened

    def _prewarm_wm(s, game_cls, level_idx, time_budget: float = 60.0):
        """v31: random walk to collect WM transitions before MCTS starts.
        When BFS fails, WM has zero training data — this ensures curiosity
        UCB is grounded in real game physics before MCTS begins."""
        if s._wm_trainer is None or game_cls is None:
            return
        if s._wm_trainer.trained_steps >= WorldModelTrainer.WARMUP:
            return  # already warm (e.g. from BFS pre-training)
        t0 = time.time()
        try:
            g = game_cls()
            g.set_level(level_idx)
            g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            prev_frame = np.array(g.get_pixels(0, 0, 64, 64))
            n_collected = 0
            while time.time() - t0 < time_budget:
                # Cycle through movement + click actions
                act_id = random.choice([1, 2, 3, 4, 5])
                try:
                    ai = ActionInput(id=GameAction.from_id(act_id))
                    r = g.perform_action(ai, raw=True)
                    if r and r.state in (GameState.WIN, GameState.GAME_OVER):
                        g = game_cls()
                        g.set_level(level_idx)
                        g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                        g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                        prev_frame = np.array(g.get_pixels(0, 0, 64, 64))
                        continue
                    if r and r.frame:
                        next_frame = np.array(r.frame[-1])
                        wm_idx = min(act_id - 1, WorldModel.N_ACTIONS - 1)
                        s._wm_trainer.add(prev_frame, wm_idx, next_frame)
                        prev_frame = next_frame
                        n_collected += 1
                except Exception:
                    pass
            logger.info(f"WM prewarm L{level_idx}: {n_collected} transitions in "
                        f"{time.time()-t0:.1f}s, trained_steps={s._wm_trainer.trained_steps}")
        except Exception as e:
            logger.warning(f"WM prewarm failed: {e}")

    def _try_repeat_action_solve(s, level_idx, game_cls, actions: list,
                                 time_budget: float = 60.0) -> 'Optional[list]':
        """v34: For click-heavy games (lp85, r11l, s5i5, su15, tn36, ft09...) BFS times out
        because solutions require 400-1500 identical clicks. Detects this by trying each
        effective action up to 1500 times. Handles reset-cycles within the level.
        """
        if game_cls is None or not actions:
            return None
        t0 = time.time()
        MAX_STEPS = 1500

        for act_id, data in actions:
            if time.time() - t0 > time_budget:
                break
            logger.info(f"RepeatAction L{level_idx}: trying action {act_id} up to {MAX_STEPS}x")
            try:
                g = game_cls()
                g.set_level(level_idx)
                g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            except Exception:
                continue

            solution = []
            frame_changes = 0
            prev_frame = None
            try:
                prev_frame = np.array(g.get_pixels(0, 0, 64, 64))
            except Exception:
                pass

            won = False
            for step in range(MAX_STEPS):
                if time.time() - t0 > time_budget:
                    break
                try:
                    ai = (ActionInput(id=GameAction.from_id(act_id), data=data)
                          if data else ActionInput(id=GameAction.from_id(act_id)))
                    r = g.perform_action(ai, raw=True)
                except Exception:
                    break

                solution.append((act_id, data))

                if r and r.frame:
                    curr_frame = np.array(r.frame[-1])
                    if prev_frame is not None and np.any(curr_frame != prev_frame):
                        frame_changes += 1
                    prev_frame = curr_frame

                # Early exit: 20 steps with zero changes → action does nothing
                if step == 20 and frame_changes == 0:
                    logger.info(f"RepeatAction: action {act_id} stalled after 20 steps, skipping")
                    break

                if r and (r.levels_completed > level_idx or
                          getattr(g, '_current_level_index', level_idx) > level_idx):
                    won = True
                    logger.info(f"RepeatAction L{level_idx}: WON in {len(solution)} steps "
                                f"with action {act_id} ({frame_changes} frame changes)")
                    break

                if r and r.state == GameState.GAME_OVER:
                    try:
                        g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                        g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                        prev_frame = np.array(g.get_pixels(0, 0, 64, 64))
                    except Exception:
                        break

            if won:
                return solution

        return None

    def _try_mcts_solve(s, level_idx, game_cls, f0, bg, bfs_fallback=None):
        """v29: MCTS always runs — either to find first path or shorter than BFS fallback.
        Uses full remaining game budget. Sets _bfs_solution or _mcts_solution based on winner."""
        if game_cls is None:
            return None

        # Full budget utilization: use nearly all remaining game time for MCTS.
        # Higher levels worth more RHAE so they get proportionally more time.
        game_budget = s._per_game_budget()
        elapsed_game = time.time() - s._game_start_time
        remaining_game = max(60.0, game_budget - elapsed_game - 30)  # 30s safety for CNN

        # Level weight: L0=1.0, L1=1.5, L2+=2.0 (matches RHAE reward structure)
        level_weight = 1.0 + 0.5 * min(level_idx, 2)
        # For L1+, we can use more since L0 already resolved (either BFS or failed)
        mcts_budget = min(remaining_game, remaining_game * level_weight / 3.0 + 60)
        mcts_budget = min(mcts_budget, game_budget * 0.85)  # cap at 85% of game budget
        mcts_budget = max(60.0, mcts_budget)

        logger.info(f"v29 MCTS L{level_idx}: game_budget={game_budget:.0f}s, "
                    f"elapsed_game={elapsed_game:.0f}s, mcts_budget={mcts_budget:.0f}s"
                    + (f", bfs_fallback={len(bfs_fallback)}" if bfs_fallback else ", no_bfs_fallback"))

        # Build initial game to scan actions
        try:
            g_scan = game_cls()
            g_scan.set_level(level_idx)
            g_scan.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            g_scan.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        except Exception as e:
            logger.warning(f"MCTS L{level_idx}: failed to make game for scan: {e}")
            return None

        # Scan unique-effect actions
        actions = s._scanner.scan(g_scan, f0, bg)
        if not actions:
            logger.info(f"MCTS L{level_idx}: no unique-effect actions found")
            return None

        logger.info(f"MCTS L{level_idx}: {len(actions)} unique-effect actions, budget={mcts_budget:.0f}s")

        # Get hidden fields from BFS if available
        hidden_fields = None
        if s._bfs and s._bfs.game_cls:
            try:
                hidden_fields = s._bfs._probe_hidden_fields(g_scan, actions)
            except:
                pass

        prev_sol = s._mcts_solutions.get(level_idx - 1) if level_idx > 0 else None

        # FIX 2: pass productive actions for informed rollout
        productive = [a for a in actions if a[0] != 6]  # non-click first pass
        if not productive:
            productive = actions

        # v31: prewarm WM with random walk when BFS found nothing
        if bfs_fallback is None and s._wm_trainer is not None:
            prewarm_budget = min(60.0, game_budget * 0.08)
            s._prewarm_wm(game_cls, level_idx, time_budget=prewarm_budget)
            # Recalculate remaining budget after prewarm
            elapsed_game = time.time() - s._game_start_time
            mcts_budget = max(60.0, game_budget - elapsed_game - 30)

        # v29: target_length tells MCTS to stop as soon as it beats BFS path
        bfs_target = len(bfs_fallback) - 1 if bfs_fallback else None

        solver = MCTSSolver(
            game_cls=game_cls,
            level_idx=level_idx,
            actions=actions,
            hidden_fields=hidden_fields,
            timeout=mcts_budget,
            prev_solution=prev_sol,
            productive_actions=productive,
            target_length=bfs_target,
            world_model=getattr(s, '_wm_trainer', None),  # v30: pass WM for curiosity+rollout
        )
        mcts_sol = solver.solve()

        # FIX 3: seed CNN buffer with MCTS-explored transitions
        if solver.explored_transitions and len(solver.explored_transitions) > 10:
            s._mcts_seed_transitions = getattr(s, '_mcts_seed_transitions', [])
            s._mcts_seed_transitions.extend(solver.explored_transitions[-500:])
            logger.info(f"MCTS L{level_idx}: seeding CNN buffer with {len(solver.explored_transitions)} transitions")

        # v31: shorten MCTS solution before comparing
        if mcts_sol:
            shorten_budget = min(20.0, (game_budget - (time.time() - s._game_start_time)) * 0.05)
            mcts_sol = s._shorten_solution(game_cls, level_idx, mcts_sol,
                                            time_limit=max(5.0, shorten_budget))

        # v29: compare MCTS solution vs BFS fallback, use the shorter one
        if mcts_sol and (bfs_fallback is None or len(mcts_sol) < len(bfs_fallback)):
            logger.info(f"L{level_idx}: MCTS={len(mcts_sol)}"
                        + (f" vs BFS={len(bfs_fallback)} -> using MCTS" if bfs_fallback else " -> using MCTS (no BFS)"))
            s._mcts_solutions[level_idx] = mcts_sol
            s._mcts_solution = mcts_sol
            s._mcts_step = 0
            return mcts_sol
        elif bfs_fallback:
            logger.info(f"L{level_idx}: MCTS={len(mcts_sol) if mcts_sol else 'None'}"
                        f" vs BFS={len(bfs_fallback)} -> using BFS")
            s._bfs_solution = bfs_fallback
            s._bfs_step = 0
            return bfs_fallback
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
        # Persistent weights: save after training
        if s.la % 200 == 0:
            try:
                torch.save(s.net.state_dict(), _CKPT_PATH)
            except:
                pass

    def _get_aem_tensors(s):
        if len(s._aem_diffs)<2:return None,None,None
        M=len(s._aem_diffs)
        diffs=torch.zeros(1,M,1,64,64,device=s.device)
        acts=torch.zeros(1,M,dtype=torch.long,device=s.device)
        rews=torch.zeros(1,M,device=s.device)
        for i,(d,a,r) in enumerate(zip(s._aem_diffs,s._aem_actions,s._aem_rewards)):
            diffs[0,i,0]=torch.from_numpy(d.astype(np.float32));acts[0,i]=min(a,4);rews[0,i]=r
        return diffs,acts,rews

    def _load_net(s):
        """Init ForgeNet. Load persistent weights if available (PMLL)."""
        s.net = ForgeNet(s.IN, s.G).to(s.device)
        # Try persistent checkpoint first (trained on prior levels)
        for wp in [_CKPT_PATH,
                   '/kaggle/input/forge-pretrained-weights/pretrained_weights.pt',
                   'pretrained_weights.pt']:
            if os.path.exists(wp):
                try:
                    state = torch.load(wp, map_location=s.device, weights_only=True)
                    ms = s.net.state_dict()
                    loaded = {k: v for k, v in state.items()
                              if k in ms and v.shape == ms[k].shape}
                    ms.update(loaded)
                    s.net.load_state_dict(ms)
                    logger.info(f"CNN: loaded weights from {wp}")
                    break
                except:
                    pass
        s.opt = optim.Adam(s.net.parameters(), lr=0.0003)

    def is_done(s, frames, lf):
        try:
            done = lf.state is GameState.WIN or (time.time()-s.start_time) >= 8*3600-300
            if done and lf.state is GameState.WIN:
                s._games_completed += 1
                s._game_start_time = time.time()
                s._bfs_tried = False
                s._bfs = None
                s._mcts_solutions = {}
                # v30: save WM so next game starts warm
                if s._wm_trainer:
                    s._wm_trainer.save(_WM_CKPT_PATH)
                    logger.info(f"WM: saved ({s._wm_trainer.trained_steps} steps, "
                                f"{len(s._wm_trainer.buffer)} transitions)")
                logger.info(f"v30: game complete ({s._games_completed}/{_TOTAL_GAMES}), "
                            f"elapsed={time.time()-s.start_time:.0f}s total")
            return done
        except: return True

    def choose_action(s, frames, lf):
        try:
            lvl = s._lvl(lf)

            # ===== LEVEL CHANGE =====
            if lvl != s.cl:
                # Init BFS (once per game) and reset game timer
                if not s._bfs_tried:
                    s._bfs_tried = True
                    s._game_start_time = time.time()  # reset game clock for budget tracking
                    s._init_bfs()

                # Reset solutions for this level
                s._mcts_solution = None
                s._mcts_step = 0
                s._bfs_solution = None
                s._bfs_step = 0
                s._bfs_fallback = None

                # v29: Try BFS first — stores as fallback, always tries MCTS after
                bfs_fallback = None
                if s._bfs:
                    bfs_fallback = s._try_bfs_solve(lvl)

                # v34: RepeatAction — only when BFS found nothing (click-heavy games)
                repeat_sol = None
                if bfs_fallback is None:
                    try:
                        raw_init = s._raw(lf)
                        bg_init = int(np.bincount(raw_init.flatten(), minlength=16).argmax())
                        game_cls = s._bfs.game_cls if s._bfs else None
                        if game_cls:
                            g_scan = game_cls()
                            g_scan.set_level(lvl)
                            g_scan.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                            scan_acts = s._scanner.scan(g_scan, raw_init, bg_init)
                            if scan_acts:
                                repeat_sol = s._try_repeat_action_solve(lvl, game_cls, scan_acts, time_budget=60.0)
                    except Exception as e:
                        logger.warning(f"RepeatAction scan failed: {e}")

                # If RepeatAction found solution, store it directly
                if repeat_sol is not None:
                    s._bfs_solution = repeat_sol
                    s._bfs_step = 0
                    logger.info(f"L{lvl}: RepeatAction found solution ({len(repeat_sol)} steps)")

                # v29: Always run MCTS: either to find first path (BFS failed) or shorter path (BFS found one)
                try:
                    raw_init = s._raw(lf)
                    bg_init = int(np.bincount(raw_init.flatten(), minlength=16).argmax())
                    game_cls = s._bfs.game_cls if s._bfs else None
                    if game_cls:
                        mcts_fallback = bfs_fallback or repeat_sol
                        s._try_mcts_solve(lvl, game_cls, raw_init, bg_init, bfs_fallback=mcts_fallback)
                except Exception as e:
                    logger.warning(f"MCTS init failed: {e}")
                    # If MCTS failed and BFS/RepeatAction had a path, use it
                    if (bfs_fallback or repeat_sol) and not s._bfs_solution:
                        s._bfs_solution = bfs_fallback or repeat_sol
                        s._bfs_step = 0
                        logger.info(f"L{lvl}: MCTS failed, falling back to BFS/RepeatAction")

                # Init CNN with persistent weights
                s.buf.clear(); s.buf_h.clear()
                s._load_net()

                # FIX 3: seed CNN buffer from MCTS-explored transitions
                seed_trans = getattr(s, '_mcts_seed_transitions', [])
                if seed_trans:
                    import hashlib as _hlib
                    for act_id, data, frame_arr, reward in seed_trans[-200:]:
                        act_idx = (act_id - 1) if act_id <= 5 else 5
                        eh = _hlib.md5(frame_arr.tobytes()[:500] + str(act_idx).encode()).hexdigest()[:16]
                        if eh not in s.buf_h:
                            s.buf.append({'s': frame_arr.copy(), 'a': act_idx, 'r': reward})
                            s.buf_h.add(eh)
                    logger.info(f"CNN: pre-seeded buffer with {len(s.buf)} MCTS transitions")
                    s._mcts_seed_transitions = []
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
                if data: sel.set_data(data)
                sel.reasoning = f"bfs:{s._bfs_step}/{len(s._bfs_solution)}"
                raw = s._raw(lf); s.fhist.append(raw.copy()); s.pr = raw.copy(); s.la += 1
                return sel

            # ===== MCTS SOLUTION EXECUTION =====
            if s._mcts_solution and s._mcts_step < len(s._mcts_solution):
                act_id, data = s._mcts_solution[s._mcts_step]
                s._mcts_step += 1
                sel = GameAction.from_id(act_id)
                if data: sel.set_data(data)
                sel.reasoning = f"mcts:{s._mcts_step}/{len(s._mcts_solution)}"
                raw = s._raw(lf); s.fhist.append(raw.copy()); s.pr = raw.copy(); s.la += 1
                return sel

            # ===== CNN FALLBACK (v22 core) =====
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

            if aidx<5:
                sel=s.al[aidx];sel.reasoning=f"cnn:a{aidx+1}"
            else:
                sel=GameAction.ACTION6;y,x=coords
                sel.set_data({"x":int(x),"y":int(y),"game_id":s.game_id})
                sel.reasoning=f"cnn:c({x},{y})"

            s.pt=tensor;s.pai=aidx if aidx<5 else(5+coords[0]*s.G+coords[1])
            s.pr=raw.copy();s.ph=ch;s.la+=1
            if s.action_counter%s.tfreq==0 and s._wd:s._train()
            # v30: feed CNN action into WM for online training
            if s._wm_trainer and s.pr is not None:
                try:
                    wm_act_idx = min(aidx, WorldModel.N_ACTIONS - 1)
                    s._wm_trainer.add(s.pr, wm_act_idx, raw)
                except: pass
            return sel

        except Exception as e:
            traceback.print_exc()
            a=random.choice(s.al);a.reasoning=f"err:{str(e)[:40]}";return a
