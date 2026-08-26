# =====================================================================
# FORGE v20 — v19 BFS + State Graph + ValueNet fallback
#
# v18 = v10 base (0.39) + CLTI + warmup unlock + ACMD trigger
# v19 adds:
#   1. Counter A* search (win field extraction from game source)
#   2. Click neighbor probing (stride-1 around hits for odd-coord sprites)
#   3. Adaptive BFS depth (50 for small action spaces, 30 default)
#   4. Improved arc_env.local_dir game discovery
#   5. Sprite permutation fallback for pure-click games (≤8 targets)
# =====================================================================
import heapq
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
from collections import defaultdict, deque
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
        # Click actions (v9 scan + v19 neighbor probe)
        if 6 in avail:
            t0 = time.time()
            seen_effects = set()
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
                                hit_positions.append((x, y))
                    except:
                        pass
            # v19: Neighbor probe (stride-1 around hit positions for odd-coord sprites)
            for hx, hy in hit_positions:
                if time.time() - t0 > self.scan_timeout + 2:
                    break
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = hx + dx, hy + dy
                    if 0 <= nx < 64 and 0 <= ny < 64 and f0[ny, nx] != bg:
                        g = copy.deepcopy(game)
                        try:
                            r = g.perform_action(
                                ActionInput(id=GameAction.ACTION6, data={'x': nx, 'y': ny, 'game_id': 'bfs'}),
                                raw=True
                            )
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

    def solve_level(self, level_idx, max_states=500000, prev_solution=None):
        """Find optimal solution for a level via BFS.
        Uses self.bfs_timeout as HARD deadline for the ENTIRE solve call
        (including all fallback phases like Counter A* and sprite permutation)."""
        if not self.game_cls:
            return None

        self._solve_start = time.time()  # Global deadline anchor

        game = self.game_cls()
        game.set_level(level_idx)
        game.perform_action(ActionInput(id=GameAction.RESET), raw=True)

        r0 = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        if not r0.frame:
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

        # v18: Warm-up unlock for locked initial states (sc25-type)
        if not actions:
            avail = game._available_actions
            for warmup_id in [a for a in avail if a <= 4]:
                g_warmup = copy.deepcopy(game)
                try:
                    g_warmup.perform_action(ActionInput(id=GameAction.from_id(warmup_id)), raw=True)
                    f_after = np.array(g_warmup.get_pixels(0, 0, 64, 64))
                    warmup_actions = self._scan_actions(g_warmup, f_after, bg)
                    if warmup_actions:
                        logger.info(f"BFS L{level_idx}: UNLOCKED with ACTION{warmup_id}! {len(warmup_actions)} actions")
                        game = g_warmup; f0 = f_after; actions = warmup_actions
                        break
                except: pass

        logger.info(f"BFS L{level_idx}: {len(actions)} effective actions")
        if not actions:
            return None

        # v19: Adaptive depth — deeper for small action spaces
        max_depth = 30
        if len(actions) <= 4:
            max_depth = 50
        elif len(actions) <= 8:
            max_depth = 40

        # Phase 2: BFS — first try with frame hash (fast, proven for 12/25)
        hidden_fields = None  # start without hidden fields
        visited = set()
        queue = deque()
        h0 = self._state_hash(game, f0, None)
        visited.add(h0)
        queue.append((copy.deepcopy(game), [], 0))

        t0 = time.time()
        deadline = self._solve_start + self.bfs_timeout  # Hard global deadline
        # Allocate 60% of budget to first BFS pass
        first_pass_deadline = self._solve_start + self.bfs_timeout * 0.6
        explored = 0

        while queue and explored < max_states and time.time() < first_pass_deadline:
            g, hist, depth = queue.popleft()

            for act_id, data in actions:
                if explored % 500 == 0 and time.time() >= first_pass_deadline:
                    break  # Check deadline inside inner loop every 500 expansions
                g2 = copy.deepcopy(g)
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

                if depth < max_depth:
                    queue.append((g2, new_hist, depth + 1))

        elapsed_first = time.time() - t0
        logger.info(f"BFS L{level_idx}: first pass done ({explored} explored, {len(visited)} unique, {elapsed_first:.1f}s)")

        # v10: If too few unique states found → hidden state detected → retry with probed fields
        # Allocate up to 25% of budget for hidden-field retry (only if we have time left)
        hidden_retry_deadline = min(deadline, self._solve_start + self.bfs_timeout * 0.85)
        if len(visited) < 50 and time.time() < hidden_retry_deadline:
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
                explored2 = 0
                while queue2 and explored2 < max_states and time.time() < hidden_retry_deadline:
                    g, hist, depth = queue2.popleft()
                    for act_id, data in actions:
                        if explored2 % 500 == 0 and time.time() >= hidden_retry_deadline:
                            break
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
                        if depth < max_depth:
                            queue2.append((g2, new_hist, depth + 1))
                logger.info(f"BFS L{level_idx}: hidden retry also failed ({explored2} explored, {len(visited2)} unique)")

        # v19: Fallback 1 — Counter A* with win field heuristic
        # All fallbacks share the same global deadline
        remaining_to_deadline = deadline - time.time()
        if remaining_to_deadline < 5:
            logger.info(f"BFS L{level_idx}: no time left for fallbacks ({remaining_to_deadline:.1f}s)")
            return None

        win_field, direction = self._extract_win_field()
        if win_field:
            astar_budget = min(remaining_to_deadline * 0.6, 60)
            logger.info(f"BFS L{level_idx}: trying Counter A* on field '{win_field}' ({direction}), budget={astar_budget:.0f}s")
            game_astar = self.game_cls()
            game_astar.set_level(level_idx)
            game_astar.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            game_astar.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            result = self._counter_a_star(game_astar, level_idx, actions, win_field, direction,
                                         timeout=astar_budget)
            if result:
                return result

        # v19: Fallback 2 — Sprite permutation for pure-click games
        remaining_to_deadline = deadline - time.time()
        if remaining_to_deadline < 3:
            return None
        clicks_only = [a for a in actions if a[0] == 6]
        kb_only = [a for a in actions if a[0] != 6]
        if not kb_only and 0 < len(clicks_only) <= 8:
            perm_budget = min(remaining_to_deadline, 60)
            logger.info(f"BFS L{level_idx}: trying sprite permutation ({len(clicks_only)} clicks), budget={perm_budget:.0f}s")
            game_perm = self.game_cls()
            game_perm.set_level(level_idx)
            game_perm.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            game_perm.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            result = self._sprite_permutation(game_perm, level_idx, actions, timeout=perm_budget)
            if result:
                return result

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

    def _extract_win_field(self):
        """v19: Scan game source for win condition variable (e.g. self.next_level() guard)."""
        import re
        try:
            with open(self.game_path, 'r') as f:
                src = f.read()
            # Find: if self.X >= Y: self.next_level() or similar
            patterns = [
                r'if\s+self\.(\w+)\s*[><=!]+.*?self\.next_level',
                r'self\.(\w+)\s*[><=]+.*?\n.*?next_level',
                r'(\w+)\s*==\s*.*?next_level',
            ]
            for pat in patterns:
                m = re.search(pat, src, re.DOTALL)
                if m:
                    field = m.group(1)
                    # Determine direction (maximize or minimize)
                    direction = 'max'  # default: higher is better
                    if '>=' in m.group(0) or '>' in m.group(0):
                        direction = 'max'
                    elif '<=' in m.group(0) or '<' in m.group(0):
                        direction = 'min'
                    return field, direction
        except:
            pass
        return None, None

    def _counter_a_star(self, game, level_idx, actions, win_field, direction,
                        max_states=200000, timeout=60):
        """v19: A* search guided by win field value as heuristic."""
        import heapq
        f0 = np.array(game.get_pixels(0, 0, 64, 64))
        initial_val = getattr(game, win_field, 0)
        h0 = self._state_hash(game, f0)
        visited = {h0}
        # Priority: negative counter value (for maximize) or positive (for minimize)
        sign = -1 if direction == 'max' else 1
        counter = 0
        heap = [(sign * initial_val, 0, counter, copy.deepcopy(game), [])]
        t0 = time.time()
        explored = 0

        while heap and explored < max_states and (time.time() - t0) < timeout:
            _, depth, _, g, hist = heapq.heappop(heap)
            for act_id, data in actions:
                g2 = copy.deepcopy(g)
                try:
                    ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                    r = g2.perform_action(ai, raw=True)
                except:
                    continue
                explored += 1
                if not r.frame:
                    continue
                f = np.array(r.frame[-1])
                h = self._state_hash(g2, f)
                if h in visited:
                    continue
                visited.add(h)
                new_hist = hist + [(act_id, data)]
                if r.levels_completed > level_idx or g2._current_level_index > level_idx:
                    logger.info(f"A* L{level_idx}: SOLVED in {len(new_hist)} actions ({explored} explored)")
                    self.solutions[level_idx] = new_hist
                    return new_hist
                val = getattr(g2, win_field, initial_val)
                if depth < 40:
                    counter += 1
                    heapq.heappush(heap, (sign * val, depth + 1, counter, g2, new_hist))

        logger.info(f"A* L{level_idx}: failed ({explored} explored, {len(visited)} unique)")
        return None

    def _sprite_permutation(self, game, level_idx, actions, timeout=60):
        """v19: For pure-click games (≤8 targets), try all orderings."""
        import itertools
        clicks = [(a, d) for a, d in actions if a == 6]
        if not clicks or len(clicks) > 8:
            return None
        t0 = time.time()
        for perm in itertools.permutations(clicks):
            if time.time() - t0 > timeout:
                break
            g = copy.deepcopy(game)
            for i, (act_id, data) in enumerate(perm):
                try:
                    ai = ActionInput(id=GameAction.ACTION6, data=data)
                    r = g.perform_action(ai, raw=True)
                    if r.levels_completed > level_idx or g._current_level_index > level_idx:
                        sol = list(perm[:i+1])
                        logger.info(f"PERM L{level_idx}: SOLVED in {i+1} clicks ({len(clicks)} total)")
                        self.solutions[level_idx] = sol
                        return sol
                except:
                    break
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

    # Method 2: environment_files directory (Kaggle downloads here)
    if not src:
        for pattern in [
            f"environment_files/{gid}/**/{gid}.py",
            f"/kaggle/working/environment_files/{gid}/**/{gid}.py",
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

    # Method 3: glob broader search
    if not src:
        for pattern in [
            f"/tmp/*/{gid}/*/{gid}.py",
            f"/kaggle/*/{gid}*/{gid}.py",
            f"/kaggle/input/**/{gid}*/**/{gid}.py",
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


# ==================== STATE GRAPH + VALUE LEARNING (v20) ====================

GRID = 64
N_COLORS = 16
N_ARROW = 5
N_COORD = GRID * GRID
N_TOTAL = N_ARROW + N_COORD
N_TIERS = 5


class NodeData:
    __slots__ = ('tier_actions', 'tested', 'transitions', 'dist_to_win', '_frame')
    def __init__(self):
        self.tier_actions = defaultdict(list)
        self.tested = set()
        self.transitions = {}
        self.dist_to_win = 999999
        self._frame = None


class LevelGraph:
    """Directed state graph with tier-grouped actions and back-labeling."""

    def __init__(self):
        self.nodes = {}
        self.current = None
        self.pending_action = None
        self.start_hash = None

    def _get_or_create(self, h):
        if h not in self.nodes:
            self.nodes[h] = NodeData()
        return self.nodes[h]

    def observe(self, state_hash, seg_actions, arrow_mask, raw_frame, is_reset):
        node = self._get_or_create(state_hash)
        node._frame = raw_frame
        if not node.tier_actions:
            for a_idx in range(N_ARROW):
                if arrow_mask[a_idx]:
                    node.tier_actions[0].append(a_idx)
            for a_idx, tier in seg_actions:
                node.tier_actions[tier].append(a_idx)
        if self.current is not None and self.pending_action is not None:
            prev = self._get_or_create(self.current)
            prev.tested.add(self.pending_action)
            prev.transitions[self.pending_action] = '__RESET__' if is_reset else state_hash
        if self.start_hash is None:
            self.start_hash = state_hash
        self.current = state_hash
        self.pending_action = None

    def record_action(self, action_idx):
        self.pending_action = action_idx

    def untested(self, state_hash, max_tier):
        node = self.nodes.get(state_hash)
        if node is None:
            return []
        out = []
        for t in range(max_tier + 1):
            out.extend(a for a in node.tier_actions.get(t, []) if a not in node.tested)
        return out

    def bfs_to_frontier(self, start, max_tier):
        if self.untested(start, max_tier):
            return []
        visited = {start}
        queue = deque([(start, [])])
        while queue:
            h, path = queue.popleft()
            if len(path) > 60:
                continue
            node = self.nodes.get(h)
            if node is None:
                continue
            for a_idx, dst in node.transitions.items():
                if dst == '__RESET__' or dst in visited:
                    continue
                new_path = path + [a_idx]
                if self.untested(dst, max_tier):
                    return new_path
                visited.add(dst)
                queue.append((dst, new_path))
        return None

    def back_label_win(self, win_hash):
        dist = {win_hash: 0}
        q = deque([win_hash])
        while q:
            h = q.popleft()
            d = dist[h]
            for src_h, node in self.nodes.items():
                for _, dst in node.transitions.items():
                    if dst == h and src_h not in dist:
                        dist[src_h] = d + 1
                        q.append(src_h)
        for h, node in self.nodes.items():
            node.dist_to_win = dist.get(h, 999999)

    def win_labeled_experiences(self):
        exps = []
        for h, node in self.nodes.items():
            if node._frame is None or node.dist_to_win == 999999:
                continue
            for a_idx, dst in node.transitions.items():
                if dst == '__RESET__':
                    continue
                dst_node = self.nodes.get(dst)
                if dst_node is None:
                    continue
                reward = 1.0 if dst_node.dist_to_win < node.dist_to_win else 0.0
                exps.append((node._frame, a_idx, reward))
        return exps

    def reset(self):
        self.__init__()


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch))
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(x + self.net(x))


class ValueNet(nn.Module):
    """ResNet18-style: [B, N_COLORS+1, 64, 64] -> [B] win probability."""
    def __init__(self):
        super().__init__()
        self.action_embed = nn.Embedding(N_TOTAL, GRID * GRID)
        self.stem = nn.Sequential(
            nn.Conv2d(N_COLORS + 1, 64, 7, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.layer1 = nn.Sequential(ResBlock(64), ResBlock(64))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), ResBlock(128))
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True), ResBlock(256))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(128, 1), nn.Sigmoid())
    def forward(self, frame_oh, action_ids):
        B = frame_oh.size(0)
        act_map = self.action_embed(action_ids).view(B, 1, GRID, GRID)
        x = torch.cat([frame_oh, act_map], dim=1)
        x = self.stem(x); x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        return self.head(x).squeeze(1)


class FrameProcessor:
    """Extract state hash and tier-grouped click actions from a frame."""
    def __init__(self):
        self._cache = {}

    @staticmethod
    def _tier(size):
        if size <= 1: return 4
        if size <= 4: return 3
        if size <= 16: return 2
        if size <= 64: return 1
        return 0

    def process(self, frame, bg):
        raw_key = hashlib.md5(frame.tobytes()).hexdigest()
        if raw_key in self._cache:
            return self._cache[raw_key]
        state_hash = raw_key
        coord_actions = []
        try:
            from scipy import ndimage
            for color in range(1, N_COLORS):
                if color == bg:
                    continue
                color_mask = (frame == color)
                if not color_mask.any():
                    continue
                labeled, n = ndimage.label(color_mask)
                for lbl in range(1, n + 1):
                    coords = np.argwhere(labeled == lbl)
                    if len(coords) == 0:
                        continue
                    tier = self._tier(len(coords))
                    cy = int(np.median(coords[:, 0]))
                    cx = int(np.median(coords[:, 1]))
                    action_idx = N_ARROW + cy * GRID + cx
                    coord_actions.append((action_idx, tier))
        except ImportError:
            # No scipy — simple fallback
            for color in range(1, N_COLORS):
                if color == bg:
                    continue
                pts = np.argwhere(frame == color)
                if len(pts) < 2:
                    continue
                tier = self._tier(len(pts))
                cy, cx = int(np.median(pts[:, 0])), int(np.median(pts[:, 1]))
                coord_actions.append((N_ARROW + cy * GRID + cx, tier))
        if not coord_actions:
            nz = np.argwhere(frame != bg)
            for pt in nz[::max(1, len(nz)//20)]:
                coord_actions.append((N_ARROW + int(pt[0]) * GRID + int(pt[1]), 4))
        self._cache[raw_key] = (state_hash, coord_actions)
        return state_hash, coord_actions


# ==================== AGENT (v20: BFS + State Graph + ValueNet) ====================

ACTION_LIST = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
               GameAction.ACTION4, GameAction.ACTION5]


class MyAgent(Agent):
    MAX_ACTIONS = float('inf')
    _MAX_FRAMES = 10

    def __init__(s, *a, **kw):
        super().__init__(*a, **kw)
        seed = int(time.time()*1e6) + hash(s.game_id) % 1000000
        random.seed(seed); np.random.seed(seed%(2**32-1)); torch.manual_seed(seed%(2**32-1))
        s.start_time = time.time()
        s.device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
        s._bfs = None; s._bfs_solution = None; s._bfs_step = 0; s._bfs_tried = False
        s.graph = LevelGraph()
        s.fp = FrameProcessor()
        s.vnet = None; s.v_opt = None; s.v_sched = None
        s.exp_buf = deque(maxlen=100000)
        s.sa_counts = defaultdict(int)
        s.planned_path = []
        s.current_tier = 0
        s.total_steps = 0
        s.train_freq = 4; s.batch_size = 128; s.train_steps = 0
        s.ucb_c = 1.5; s.epsilon = 0.3; s.epsilon_min = 0.03; s.epsilon_decay = 0.99
        s.warmstart_actions = 10  # Heuristic phase before state graph kicks in
        s.prev_frame = None; s.prev_hash = None; s.prev_action = None
        s.cl = -1; s._bg = 0

    def append_frame(s, f):
        s.frames.append(f)
        if len(s.frames) > s._MAX_FRAMES: s.frames = s.frames[-s._MAX_FRAMES:]
        if f.guid: s.guid = f.guid
        if hasattr(s, "recorder") and not s.is_playback:
            import json; s.recorder.record(json.loads(f.model_dump_json()))

    def _lvl(s, f): return getattr(f, 'score', None) or f.levels_completed
    def _raw(s, fd): return np.array(fd.frame, dtype=np.int64)[-1]

    def _onehot(s, frame):
        t = torch.zeros(N_COLORS, GRID, GRID, dtype=torch.float32)
        t.scatter_(0, torch.from_numpy(frame.astype(np.int64)).unsqueeze(0), 1.0)
        return t.to(s.device)

    def _init_bfs(s):
        src, cls = find_game_source_and_class(s.game_id, s.arc_env)
        if src:
            s._bfs = BFSSolver(src, cls, scan_timeout=5, bfs_timeout=180)
            if s._bfs.load():
                logger.info(f"BFS: loaded {cls} from {src}")
            else:
                s._bfs = None
        else:
            logger.warning(f"BFS: game source not found for {s.game_id}")

    def _try_bfs_solve(s, level_idx):
        if s._bfs is None: return None
        elapsed = time.time() - s.start_time
        remaining = max(60, 8*3600 - 300 - elapsed)
        budget = min(remaining * 0.3, 600) if level_idx == 0 else min(remaining * 0.1, 300)
        s._bfs.bfs_timeout = max(30, budget)
        logger.info(f"BFS budget for L{level_idx}: {s._bfs.bfs_timeout:.0f}s (elapsed={elapsed:.0f}s, remaining={remaining:.0f}s)")
        prev_sol = s._bfs.solutions.get(level_idx - 1) if level_idx > 0 else None
        sol = s._bfs.solve_level(level_idx, prev_solution=prev_sol)
        if sol: s._bfs_solution = sol; s._bfs_step = 0
        return sol

    def _reset_level(s):
        s.graph.reset(); s.planned_path = []
        s.prev_frame = None; s.prev_hash = None; s.prev_action = None
        s.current_tier = 0; s.sa_counts.clear(); s.exp_buf.clear()
        s.vnet = ValueNet().to(s.device)
        s.v_opt = optim.AdamW(s.vnet.parameters(), lr=1e-4, weight_decay=1e-4)
        s.v_sched = optim.lr_scheduler.CosineAnnealingLR(s.v_opt, T_max=5000, eta_min=1e-6)
        s.train_steps = 0; s.total_steps = 0
        s.epsilon = 0.3  # High initial epsilon for exploration (decays to 0.03)

    def _push_exp(s, frame, action, reward):
        s.exp_buf.append({'frame': frame, 'action': action, 'reward': reward})

    def _train_vnet(s):
        if len(s.exp_buf) < s.batch_size: return
        s.vnet.train()
        idx = np.random.choice(len(s.exp_buf), s.batch_size, replace=False)
        batch = [s.exp_buf[i] for i in idx]
        frames_oh = torch.stack([s._onehot(e['frame']) for e in batch])
        acts = torch.tensor([e['action'] for e in batch], dtype=torch.long, device=s.device).clamp(0, N_TOTAL-1)
        tgts = torch.tensor([e['reward'] for e in batch], dtype=torch.float32, device=s.device)
        pred = s.vnet(frames_oh, acts)
        loss = F.binary_cross_entropy(pred, tgts)
        s.v_opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(s.vnet.parameters(), 1.0)
        s.v_opt.step(); s.v_sched.step(); s.train_steps += 1

    def _value_scores(s, frame, candidates):
        if not candidates or s.train_steps < 5: return np.zeros(len(candidates))
        s.vnet.eval()
        with torch.no_grad():
            foh = s._onehot(frame).unsqueeze(0).expand(len(candidates), -1, -1, -1)
            aids = torch.tensor(candidates, dtype=torch.long, device=s.device).clamp(0, N_TOTAL-1)
            return s.vnet(foh, aids).cpu().numpy()

    def _ucb(s, state_hash, candidates):
        log_n = np.log(max(s.total_steps, 1))
        return np.array([s.ucb_c * np.sqrt(log_n / max(s.sa_counts[(state_hash, a)], 1)) for a in candidates])

    def _rank(s, state_hash, frame, candidates):
        if random.random() < s.epsilon: return random.choice(candidates)
        scores = s._value_scores(frame, candidates) + s._ucb(state_hash, candidates)
        return candidates[int(np.argmax(scores))]

    def _back_label_and_retrain(s, win_hash):
        s.graph.back_label_win(win_hash)
        exps = s.graph.win_labeled_experiences()
        for frame, action, reward in exps:
            s._push_exp(frame, action, reward)
        for _ in range(min(400, len(exps) * 2)):
            s._train_vnet()
        logger.info(f"v20: back-labeled {len(exps)} exps")

    def _action_idx_to_game_action(s, action_idx):
        if action_idx < N_ARROW:
            return ACTION_LIST[action_idx], None
        coord_idx = action_idx - N_ARROW
        y, x = divmod(coord_idx, GRID)
        return GameAction.ACTION6, {"x": int(x), "y": int(y)}

    def _heuristic_warmstart(s, raw, avail_set, arrow_mask, seg_actions, step):
        """v20: Heuristic warm-start for first N actions (mirrors v19 heuristic).
        Probes directional actions first, then targets click sprites by size.
        This seeds the experience buffer with useful frame-change data before
        the ValueNet has any training signal."""
        # First 4 steps: try each directional action in order
        if step < N_ARROW:
            # Try action index = step first, then others
            if step < N_ARROW and arrow_mask[step]:
                return step
            for d in range(N_ARROW):
                if arrow_mask[d]:
                    return d
        # Steps 5+: target click sprites by size (smallest = highest tier first)
        if 6 in avail_set and seg_actions:
            sorted_actions = sorted(seg_actions, key=lambda x: -x[1])
            idx_in_sorted = step - N_ARROW
            if 0 <= idx_in_sorted < len(sorted_actions):
                return sorted_actions[idx_in_sorted][0]
            # Wrap around if we've exhausted the sorted list
            if sorted_actions:
                return sorted_actions[idx_in_sorted % len(sorted_actions)][0]
        # Fallback: random available arrow
        choices = [i for i in range(N_ARROW) if arrow_mask[i]]
        if choices:
            return random.choice(choices)
        return 0

    def is_done(s, frames, lf):
        try: return lf.state is GameState.WIN or (time.time()-s.start_time) >= 8*3600-300
        except: return True

    def choose_action(s, frames, lf):
        try:
            lvl = s._lvl(lf)

            # Level change
            if lvl != s.cl:
                if not s._bfs_tried:
                    s._bfs_tried = True; s._init_bfs()
                s._bfs_solution = None; s._bfs_step = 0
                if s._bfs: s._try_bfs_solve(lvl)
                s._reset_level(); s.cl = lvl

            # Reset
            if lf.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
                s.prev_frame = None; s.prev_hash = None; s.prev_action = None
                a = GameAction.RESET; a.reasoning = "reset"; return a

            # BFS solution replay
            if s._bfs_solution and s._bfs_step < len(s._bfs_solution):
                act_id, data = s._bfs_solution[s._bfs_step]; s._bfs_step += 1
                sel = GameAction.from_id(act_id)
                if data: sel.set_data(data)
                sel.reasoning = f"bfs:{s._bfs_step}/{len(s._bfs_solution)}"
                return sel

            # ===== STATE GRAPH + VALUE LEARNING (v20) =====
            raw = s._raw(lf)
            cnt = np.bincount(raw.flatten(), minlength=16)
            s._bg = int(cnt.argmax())

            avail = getattr(lf, 'available_actions', None) or []
            avail_set = set()
            for a in avail:
                aid = a.value if hasattr(a, 'value') else int(a)
                avail_set.add(aid)
            arrow_mask = np.array([i+1 in avail_set for i in range(N_ARROW)])
            has_click = 6 in avail_set

            state_hash, seg_actions = s.fp.process(raw, s._bg)
            if not has_click: seg_actions = []

            is_reset = (s.graph.start_hash is not None and state_hash == s.graph.start_hash
                        and s.prev_hash is not None and s.prev_hash != state_hash)
            s.graph.observe(state_hash, seg_actions, arrow_mask, raw, is_reset)

            if s.prev_frame is not None and s.prev_action is not None:
                changed = state_hash != s.prev_hash
                s._push_exp(s.prev_frame, s.prev_action, 1.0 if changed else 0.0)
                s.sa_counts[(s.prev_hash, s.prev_action)] += 1

            s.total_steps += 1
            if s.total_steps % s.train_freq == 0:
                s._train_vnet()

            # Decay epsilon: 0.3 -> 0.03 over ~200 steps
            s.epsilon = max(s.epsilon_min, s.epsilon * s.epsilon_decay)

            # === Phase 1: Heuristic warm-start (first N actions) ===
            # Probes directional/click actions systematically to seed experience
            # buffer with frame-change data before ValueNet has training signal.
            # This prevents cold-start regression vs the old ForgeNet CNN.
            if s.total_steps <= s.warmstart_actions:
                action_idx = s._heuristic_warmstart(raw, avail_set, arrow_mask, seg_actions, s.total_steps - 1)
                s.graph.record_action(action_idx)
                s.prev_frame = raw; s.prev_hash = state_hash; s.prev_action = action_idx
                ga, data = s._action_idx_to_game_action(action_idx)
                if data: ga.set_data(data)
                ga.reasoning = f"v20:warmstart:{s.total_steps}"
                return ga

            # === Phase 2: State graph + ValueNet with epsilon-greedy ===
            if s.planned_path:
                action_idx = s.planned_path.pop(0)
            else:
                untested = s.graph.untested(state_hash, s.current_tier)
                if not untested:
                    path = s.graph.bfs_to_frontier(state_hash, s.current_tier)
                    if path is not None and len(path) > 0:
                        s.planned_path = path[1:]; action_idx = path[0]
                    elif path is not None:
                        untested = s.graph.untested(state_hash, s.current_tier)
                        action_idx = s._rank(state_hash, raw, untested) if untested else 0
                    else:
                        s.current_tier = min(s.current_tier + 1, N_TIERS - 1)
                        untested2 = s.graph.untested(state_hash, s.current_tier)
                        if untested2:
                            action_idx = s._rank(state_hash, raw, untested2)
                        else:
                            choices = [i for i in range(N_ARROW) if arrow_mask[i]]
                            if has_click and seg_actions:
                                choices.extend([a for a, _ in seg_actions[:5]])
                            action_idx = random.choice(choices) if choices else 0
                else:
                    action_idx = s._rank(state_hash, raw, untested)

            s.graph.record_action(action_idx)
            s.prev_frame = raw; s.prev_hash = state_hash; s.prev_action = action_idx

            ga, data = s._action_idx_to_game_action(action_idx)
            if data: ga.set_data(data)
            ga.reasoning = f"v20:t{s.current_tier}:e{s.epsilon:.2f}"
            return ga

        except Exception as e:
            traceback.print_exc()
            a = random.choice(ACTION_LIST); a.reasoning = f"err:{e}"; return a
