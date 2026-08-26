"""Kaggle agent wrapper for our novel solver v2.
Wraps solver_v2.BFSSolver into the ARC-AGI-3-Agents framework Agent class.

Flow:
1. On first call: load game via importlib, run BFS offline
2. If BFS found solution: replay actions one per step
3. If BFS failed: fall back to greedy novelty exploration
4. On level change: run BFS for next level (replaying prev solutions first)
"""

import random
import time
import hashlib
import traceback
import os
import copy
import re
import importlib.util
from typing import Any
from collections import defaultdict, deque

import numpy as np

from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent

# Game action map
GA_MAP = {a.value: a for a in GameAction}
TOTAL_BUDGET = 5 * 3600  # 5 hours (safe for 6hr Kaggle limit)


class BFSSolverLight:
    """Lightweight BFS solver for Kaggle (runs inside Agent thread)."""

    def __init__(self, game_cls, source_path):
        self.game_cls = game_cls
        self.source_path = source_path
        self.solutions = {}

    def _scan_actions(self, game, f0, avail, bg, timeout=5):
        actions = []
        seen = set()
        t0 = time.time()

        # Keyboard (no dedup)
        for a in avail:
            if a in (0, 6) or a not in GA_MAP:
                continue
            g = copy.deepcopy(game)
            try:
                from arcengine.enums import ActionInput
                r = g.perform_action(ActionInput(id=GA_MAP[a]), raw=True)
                if r.frame and np.any(f0 != np.array(r.frame[-1])):
                    actions.append((a, None))
            except:
                pass

        # Click (with dedup)
        if 6 in avail:
            from arcengine.enums import ActionInput
            for y in range(0, 64, 2):
                if time.time() - t0 > timeout:
                    break
                for x in range(0, 64, 2):
                    if f0[y, x] == bg:
                        continue
                    g = copy.deepcopy(game)
                    try:
                        r = g.perform_action(
                            ActionInput(id=GameAction.ACTION6, data={'x': x, 'y': y, 'game_id': ''}),
                            raw=True)
                        if r.frame:
                            f1 = np.array(r.frame[-1])
                            if np.any(f0 != f1):
                                eh = hashlib.md5((f0 ^ f1).tobytes()).hexdigest()[:16]
                                if eh not in seen:
                                    seen.add(eh)
                                    actions.append((6, {'x': x, 'y': y, 'game_id': ''}))
                    except:
                        pass
        return actions

    def _probe_hidden(self, game, actions):
        initial = {k: v for k, v in game.__dict__.items()
                   if not k.startswith('_') and isinstance(v, (int, float, bool))}
        changing = set()
        for act_id, data in actions[:5]:
            g = copy.deepcopy(game)
            try:
                from arcengine.enums import ActionInput
                if data:
                    g.perform_action(ActionInput(id=GameAction.ACTION6, data=data), raw=True)
                else:
                    g.perform_action(ActionInput(id=GA_MAP[act_id]), raw=True)
                for k, v0 in initial.items():
                    if getattr(g, k, None) != v0:
                        changing.add(k)
            except:
                pass
        return list(changing)

    def solve_level(self, level_idx, timeout=120, max_states=200000):
        from arcengine.enums import ActionInput

        game = self.game_cls()
        r0 = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        if not r0.frame:
            return None

        # Replay previous solutions to reach this level
        if level_idx > 0:
            for prev_lvl in range(level_idx):
                if prev_lvl not in self.solutions:
                    return None
                for act_id, data in self.solutions[prev_lvl]:
                    try:
                        if data:
                            r0 = game.perform_action(ActionInput(id=GameAction.ACTION6, data=data), raw=True)
                        else:
                            r0 = game.perform_action(ActionInput(id=GA_MAP[act_id]), raw=True)
                    except:
                        pass

        if not r0.frame:
            return None
        f0 = np.array(r0.frame[-1])
        bg = int(np.bincount(f0.flatten(), minlength=16).argmax())

        # Transfer from prev level
        if level_idx > 0 and (level_idx - 1) in self.solutions:
            prev_sol = self.solutions[level_idx - 1]
            for dx, dy in [(0,0),(1,0),(-1,0),(0,1),(0,-1)]:
                g = copy.deepcopy(game)
                try:
                    for act_id, data in prev_sol:
                        if data:
                            nd = dict(data)
                            nd['x'] = max(0, min(63, data.get('x',32)+dx))
                            nd['y'] = max(0, min(63, data.get('y',32)+dy))
                            r = g.perform_action(ActionInput(id=GameAction.ACTION6, data=nd), raw=True)
                        else:
                            r = g.perform_action(ActionInput(id=GA_MAP[act_id]), raw=True)
                        if r.levels_completed > level_idx:
                            sol = [(a, dict(d) if d else d) for a, d in prev_sol]
                            if dx or dy:
                                sol = [(a, {**d, 'x':max(0,min(63,d['x']+dx)), 'y':max(0,min(63,d['y']+dy))} if d else d) for a, d in sol]
                            self.solutions[level_idx] = sol
                            return sol
                except:
                    break

        actions = self._scan_actions(game, f0, r0.available_actions, bg, timeout=min(5, timeout*0.1))
        if not actions:
            return None

        hidden = self._probe_hidden(game, actions)

        def sh(g, f):
            h = hashlib.md5(f.tobytes()).hexdigest()
            for field in hidden:
                try: h += f"|{field}={getattr(g, field)}"
                except: pass
            return h

        t0 = time.time()
        ih = sh(game, f0)
        parent = {ih: None}
        queue = deque([(copy.deepcopy(game), f0, ih)])
        visited = {ih}

        while queue and time.time()-t0 < timeout and len(visited) < max_states:
            g, f, ch = queue.popleft()
            for i, (act_id, data) in enumerate(actions):
                g2 = copy.deepcopy(g)
                try:
                    from arcengine.enums import ActionInput as AI
                    if data:
                        r2 = g2.perform_action(AI(id=GameAction.ACTION6, data=data), raw=True)
                    else:
                        r2 = g2.perform_action(AI(id=GA_MAP[act_id]), raw=True)
                except:
                    continue
                if not r2.frame:
                    continue
                f2 = np.array(r2.frame[-1])
                if r2.levels_completed > level_idx:
                    path = [(act_id, data)]
                    h = ch
                    while parent[h] is not None:
                        ph, pi = parent[h]
                        path.append(actions[pi])
                        h = ph
                    path.reverse()
                    self.solutions[level_idx] = path
                    return path
                h2 = sh(g2, f2)
                if h2 not in visited:
                    visited.add(h2)
                    parent[h2] = (ch, i)
                    queue.append((g2, f2, h2))

        return None


def find_game_files(game_id):
    """Find game source file on Kaggle or locally."""
    gid = game_id.split('-')[0]
    cls_name = gid[0].upper() + gid[1:]

    search_paths = [
        f'environment_files/{gid}',
        f'/tmp/*/{gid}',
        f'/kaggle/input/*/{gid}*',
    ]

    import glob
    for pattern in search_paths:
        for d in glob.glob(pattern):
            for root, dirs, files in os.walk(d):
                for f in files:
                    if f.endswith('.py') and not f.startswith('__'):
                        return os.path.join(root, f), cls_name
    return None, None


class MyAgent(Agent):
    MAX_ACTIONS = 50000

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1000000) + hash(self.game_id) % 1000000
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        self.start_time = time.time()
        print(f'SolverAgent: game={self.game_id}')

        self._bfs = None
        self._bfs_tried = False
        self._bfs_solution = None
        self._bfs_step = 0
        self._current_level = -1
        self._action_count = 0

        # Fallback state
        self._prev_hash = None
        self._prev_action = None
        self._visited = set()
        self._tried = defaultdict(set)
        self._productive = defaultdict(set)
        self._globally_productive = defaultdict(int)

    def _init_bfs(self):
        """Load game class and create BFS solver."""
        try:
            gid = self.game_id.split('-')[0]
            # Try to find game source via arc_env
            game_path = None
            cls_name = gid[0].upper() + gid[1:]

            # Search common paths
            import glob
            for pattern in [
                f'environment_files/{gid}/**/{gid}.py',
                f'/tmp/**/{gid}/**/{gid}.py',
                f'/kaggle/**/{gid}*/**/{gid}.py',
            ]:
                matches = glob.glob(pattern, recursive=True)
                if matches:
                    game_path = matches[0]
                    # Read source to find actual class name
                    with open(game_path) as f:
                        src = f.read()[:2000]
                    import re
                    m = re.search(r'class\s+(\w+)\s*\(', src)
                    if m:
                        cls_name = m.group(1)
                    break

            if game_path:
                mod_name = f'game_{gid}_{id(self)}'
                spec = importlib.util.spec_from_file_location(mod_name, game_path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                game_cls = getattr(mod, cls_name)
                self._bfs = BFSSolverLight(game_cls, game_path)
                print(f'  BFS solver ready: {cls_name} from {game_path}')
            else:
                print(f'  No game source found for {gid}')
        except Exception as e:
            print(f'  BFS init failed: {e}')
            traceback.print_exc()

    def _try_bfs(self, level_idx):
        if self._bfs is None:
            return None
        elapsed = time.time() - self.start_time
        remaining = max(60, TOTAL_BUDGET - elapsed)
        budget = min(remaining * 0.4, 600)  # 40% of remaining, cap 10min
        budget = max(30, budget)

        print(f'  BFS L{level_idx}: budget={budget:.0f}s')
        sol = self._bfs.solve_level(level_idx, timeout=budget)
        if sol:
            self._bfs_solution = sol
            self._bfs_step = 0
            print(f'  BFS SOLVED L{level_idx} in {len(sol)} actions!')
        return sol

    def _hash_frame(self, fd):
        frame = np.array(fd.frame, dtype=np.int8)
        if frame.ndim == 3: frame = frame[-1]
        return hashlib.md5(frame.tobytes()).hexdigest()

    def is_done(self, frames, latest_frame):
        elapsed = time.time() - self.start_time
        return latest_frame.state is GameState.WIN or elapsed >= TOTAL_BUDGET

    def choose_action(self, frames, latest_frame):
        try:
            return self._choose_impl(frames, latest_frame)
        except Exception as e:
            print(f'  Error: {e}')
            a = random.choice([GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4])
            a.reasoning = f'err:{e}'
            return a

    def _choose_impl(self, frames, latest_frame):
        self._action_count += 1
        level = latest_frame.score if hasattr(latest_frame, 'score') else 0

        # Level change
        if level != self._current_level:
            if not self._bfs_tried:
                self._bfs_tried = True
                self._init_bfs()

            self._bfs_solution = None
            self._bfs_step = 0
            self._try_bfs(level)

            self._visited.clear()
            self._tried.clear()
            self._prev_hash = None
            self._prev_action = None
            self._current_level = level

        # Reset
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            self._prev_hash = None
            self._prev_action = None
            a = GameAction.RESET
            a.reasoning = 'reset'
            return a

        # BFS solution replay
        if self._bfs_solution and self._bfs_step < len(self._bfs_solution):
            act_id, data = self._bfs_solution[self._bfs_step]
            self._bfs_step += 1
            if act_id == 6 and data:
                a = GameAction.ACTION6
                a.set_data(data)
            else:
                a = GA_MAP.get(act_id, GameAction.ACTION1)
            a.reasoning = f'bfs:{self._bfs_step}/{len(self._bfs_solution)}'
            return a

        # Fallback: greedy novelty exploration
        fh = self._hash_frame(latest_frame)
        self._visited.add(fh)

        if self._prev_hash and self._prev_action is not None:
            if fh != self._prev_hash:
                self._productive[self._prev_hash].add(self._prev_action)
                self._globally_productive[self._prev_action] += 1

        all_actions = [a for a in GameAction if a is not GameAction.RESET]
        tried = self._tried[fh]
        untried = [a for a in all_actions if a.value not in tried]

        if untried:
            scored = [(a, self._globally_productive.get(a.value, 0)) for a in untried]
            scored.sort(key=lambda x: -x[1])
            action = scored[0][0]
        else:
            prod_vals = self._productive.get(fh, set())
            prod = [a for a in all_actions if a.value in prod_vals]
            action = random.choice(prod) if prod else random.choice(all_actions)

        if action.is_complex():
            frame = np.array(latest_frame.frame, dtype=np.int8)
            if frame.ndim == 3: frame = frame[-1]
            nz = np.argwhere(frame != 0)
            if len(nz) > 0:
                idx = random.randint(0, len(nz)-1)
                action.set_data({'x': int(nz[idx][1]), 'y': int(nz[idx][0])})
            else:
                action.set_data({'x': random.randint(0,63), 'y': random.randint(0,63)})
            action.reasoning = f'explore:click'
        else:
            action.reasoning = f'explore:{action.value}'

        self._tried[fh].add(action.value)
        self._prev_hash = fh
        self._prev_action = action.value
        return action
