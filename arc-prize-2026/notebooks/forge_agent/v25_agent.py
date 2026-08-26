# =====================================================================
# FORGE v19 — v18 base + 4 targeted bug fixes
#
# Fixes applied on top of v18:
#
# FIX 1: _visited_hashes was never initialized in __init__ — reward
#         signal was broken: always gave +1.5 for ANY hash change,
#         never penalizing loops. Now properly tracks and deduplicates.
#
# FIX 2: CLTI frame extraction used get_pixels() which is inconsistent
#         with _raw() (which reads frame[-1] from perform_action).
#         Now uses perform_action result frames throughout, so injected
#         expert demos have correct state representations.
#
# FIX 3: BFS hidden retry used 3 RESET calls instead of 2, landing
#         in a different initial state than the first pass scan,
#         causing the retry to search from a mismatched baseline.
#
# FIX 4: Epsilon always reset to 0.15 on level change even when BFS
#         already solved the level. Now only resets if BFS failed,
#         preserving learned exploration for CNN fallback.
# =====================================================================
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
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState, ActionInput

logger = logging.getLogger(__name__)

# ==================== v19: CROSS-GAME TYPE MEMORY ====================
# Module-level dict persists across MyAgent instances within a single Kaggle run.
# Keyed by game_type (e.g. 'cd82' from game_id 'cd82-fb555c5d').
# Generalizes: any repeating game type benefits from priors collected on prior instance(s).
# Generic — no per-game-id branches, works for ANY game type pattern.
_CROSS_GAME_MEMORY: Dict[str, dict] = {}

# ==================== BFS SOLVER ====================
def _fast_deepcopy(game):
    """Deepcopy game object, skipping the camera (rendering-only, never mutates)."""
    camera = game._camera
    game._camera = None
    g = copy.deepcopy(game)
    game._camera = camera
    g._camera = camera
    return g

class BFSSolver:
    """Offline BFS solver using direct game class instantiation."""

    def __init__(self, game_path, game_class_name, scan_timeout=3, bfs_timeout=120, cgm=None):
        self.game_path = game_path
        self.class_name = game_class_name
        self.scan_timeout = scan_timeout
        self.bfs_timeout = bfs_timeout
        self.game_cls = None
        self.solutions = {}  # level_idx → action list
        self.cgm = cgm  # v19: cross-game memory dict (shared across BFSSolver instances of same game type)
        self.timed_out_levels = set()

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

    def _save_state(self, game):
        return copy.deepcopy(game.__dict__)

    def _restore_state(self, base_game, state_dict):
        g = copy.deepcopy(base_game)
        g.__dict__.update(copy.deepcopy(state_dict))
        return g

    def _perform_and_drain(self, game, ai, max_drain=5, drain=True):
        try:
            r = game.perform_action(ai, raw=True)
        except Exception as e:
            logger.warning(f"BFS drain: initial perform_action failed: {e}")
            raise
        if not drain or not r.frame:
            return r
    
        prev_frame = np.array(r.frame[-1])
        for _ in range(max_drain):
            try:
                r2 = game.perform_action(ActionInput(id=GameAction.ACTION1), raw=True)
            except:
                break
            if not r2.frame:
                break
            curr_frame = np.array(r2.frame[-1])
            if np.array_equal(curr_frame, prev_frame):
                break
            r = r2
            prev_frame = curr_frame
        return r

    def _analyse_demo(self, frames_and_actions):
        """Analyse a demonstration (sequence of frame, action pairs) to extract:
        - Which colors are player-controlled (move in response to actions)
        - Which colors are passive targets (stationary until win)
        - What the win condition looks like structurally
        
        Returns a demo_model dict with this information.
        """
        if len(frames_and_actions) < 2:
            return None
        
        bg = int(np.bincount(
            frames_and_actions[0][0].flatten(), minlength=16).argmax())
        
        # Action direction vectors
        action_dirs = {1: (0,-1), 2: (0,1), 3: (-1,0), 4: (1,0)}
        
        def get_centroids(frame):
            result = {}
            for c in range(16):
                if c == bg: continue
                mask = (frame == c)
                n = int(np.sum(mask))
                if n < 4: continue
                ys, xs = np.where(mask)
                result[c] = (float(np.mean(xs)), float(np.mean(ys)), n)
            return result
        
        # Track per-color movement correlation with action direction
        # player-controlled colors move in the action direction
        color_action_corr = {}  # color -> list of (expected_dx, actual_dx, expected_dy, actual_dy)
        color_movement = {}     # color -> total movement across all steps
        
        prev_frame, _ = frames_and_actions[0]
        prev_centroids = get_centroids(prev_frame)
        
        for frame, action in frames_and_actions[1:]:
            curr_centroids = get_centroids(frame)
            adx, ady = action_dirs.get(action, (0, 0))
            
            for c in prev_centroids:
                if c not in curr_centroids:
                    continue
                actual_dx = curr_centroids[c][0] - prev_centroids[c][0]
                actual_dy = curr_centroids[c][1] - prev_centroids[c][1]
                movement = abs(actual_dx) + abs(actual_dy)
                
                if c not in color_action_corr:
                    color_action_corr[c] = []
                    color_movement[c] = 0
                color_movement[c] += movement
                
                # Does this color move in the action direction?
                if movement > 1:
                    if adx != 0:
                        corr = np.sign(actual_dx) == np.sign(adx)
                    elif ady != 0:
                        corr = np.sign(actual_dy) == np.sign(ady)
                    else:
                        corr = False
                    color_action_corr[c].append(corr)
            
            prev_frame = frame
            prev_centroids = curr_centroids
        
        # Track pixel count stability per color
        # Player colors maintain consistent pixel counts
        # Target colors that get overlapped show sudden pixel count changes at win step
        color_pixel_counts = {}  # color -> list of pixel counts across frames
        for frame, action in frames_and_actions:
            c_counts = {}
            for c in range(16):
                if c == bg: continue
                n = int(np.sum(frame == c))
                if n >= 4:
                    c_counts[c] = n
            for c, n in c_counts.items():
                if c not in color_pixel_counts:
                    color_pixel_counts[c] = []
                color_pixel_counts[c].append(n)
    
        player_colors = set()
        passive_colors = set()
        for c, corrs in color_action_corr.items():
            total_movement = color_movement.get(c, 0)
            
            # Check pixel count stability
            counts = color_pixel_counts.get(c, [])
            if len(counts) >= 2:
                count_variance = max(counts) - min(counts)
                # High variance in pixel count = color appears/disappears = target being overlapped
                count_stable = count_variance < max(counts) * 0.3
            else:
                count_stable = True
    
            if not corrs:
                if total_movement < 1:
                    passive_colors.add(c)
                continue
            corr_rate = sum(corrs) / len(corrs)
            if corr_rate > 0.5 and total_movement > 5 and count_stable:
                player_colors.add(c)
            elif corr_rate < 0.3 or not count_stable:
                passive_colors.add(c)
        
        # Win frame analysis
        win_frame = frames_and_actions[-1][0]
        init_frame = frames_and_actions[0][0]
        win_centroids = get_centroids(win_frame)
        init_centroids = get_centroids(init_frame)
        
        # What changed at the win step vs second-to-last step?
        pre_win_frame = frames_and_actions[-2][0]
        pre_win_centroids = get_centroids(pre_win_frame)
        
        win_changes = {}  # color -> (pre_win_pos, win_pos)
        for c in pre_win_centroids:
            if c not in win_centroids:
                continue
            dx = abs(win_centroids[c][0] - pre_win_centroids[c][0])
            dy = abs(win_centroids[c][1] - pre_win_centroids[c][1])
            if dx + dy > 2:
                win_changes[c] = (
                    (pre_win_centroids[c][0], pre_win_centroids[c][1]),
                    (win_centroids[c][0], win_centroids[c][1])
                )
        
       # Win conditions: which player colors moved TOWARD passive colors at the win step?
        # Compare pre-win distance vs post-win distance for each (player, passive) pair
        win_conditions = []
        for pc in player_colors:
            if pc not in win_centroids or pc not in pre_win_centroids:
                continue
            for tc in passive_colors:
                if tc not in win_centroids or tc not in pre_win_centroids:
                    continue
                # Distance before and after win step
                pre_dist = (abs(pre_win_centroids[pc][0] - pre_win_centroids[tc][0]) +
                           abs(pre_win_centroids[pc][1] - pre_win_centroids[tc][1]))
                post_dist = (abs(win_centroids[pc][0] - win_centroids[tc][0]) +
                            abs(win_centroids[pc][1] - win_centroids[tc][1]))
                # Player color moved toward passive color at win step
                if post_dist < pre_dist and post_dist < 15:
                    win_conditions.append((pc, tc))
        
        # Pixel-level win signature: what transformation happened?
        changed_mask = init_frame != win_frame
        n_changed = int(np.sum(changed_mask))
        
        return {
            'player_colors': player_colors,
            'passive_colors': passive_colors,
            'win_conditions': win_conditions,  # (player_color, target_color) pairs
            'win_centroids': win_centroids,
            'init_centroids': init_centroids,
            'bg': bg,
            'n_changed': n_changed,
            'win_frame': win_frame,
            'init_frame': init_frame,
        }

    def _build_goal_heuristic(self, f_init, f_prev_win, demo_model=None):
        """Build A* heuristic using game-state introspection.
        
        Scans game object for indicator sprites (any dict->list->sprite
        with is_visible property) and counts unsatisfied conditions.
        Falls back to uniform cost if no indicators found.
        General: works for any game using the indicator pattern.
        """
        def introspection_heuristic(f, game=None):
            if game is None:
                return 0
            try:
                total, satisfied = 0, 0
                for attr_val in game.__dict__.values():
                    if not isinstance(attr_val, dict):
                        continue
                    for v in attr_val.values():
                        if not isinstance(v, list):
                            continue
                        for item in v:
                            if hasattr(item, 'is_visible') and hasattr(item, 'pixels'):
                                total += 1
                                if item.is_visible:
                                    satisfied += 1
                if total == 0:
                    return 0
                return total - satisfied
            except:
                return 0

        # Validate signal exists on a fresh game instance
        if self.game_cls:
            try:
                test = self.game_cls()
                test.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                test.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                h = introspection_heuristic(None, test)
                if h > 0:
                    logger.info(f"BFS heuristic: introspection found {h} indicators")
                    return introspection_heuristic
            except:
                pass

        logger.info(f"BFS heuristic: no indicators found, uniform cost")
        return lambda f, game=None: 0
     
    def _state_hash(self, g, frame, hidden_fields=None, transient_fields=None):
        fh = hashlib.md5(frame.tobytes()).hexdigest()[:16]
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
            eh = hashlib.md5("|".join(sorted(extras)).encode()).hexdigest()[:12]
            return fh + "|" + eh
        return fh

    def _probe_hidden_fields(self, game, actions):
        """Dynamic state probing — discover which scalar fields change per action.
        Returns list of field names that are hidden state (change without pixel change)."""
        if not actions:
            return []
        initial = {}
        for k, v in game.__dict__.items():
            if isinstance(v, (int, float, bool)) and not k.startswith('__'):
                initial[k] = v

        changing_fields = set()
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

    def _detect_transient_fields(self, game, actions):
        """Detect scalar fields that change on every action (e.g. budget counters,
        monotonic clocks). These add no state-distinguishing value to the hash and
        cause state space explosion if included."""
        if not actions:
            return set()
        initial = {k: v for k, v in game.__dict__.items()
                   if isinstance(v, (int, float, bool)) and not k.startswith('__')
                   and k not in ('_action_count', '_full_reset', '_action_complete')}
        # Track how many sampled actions changed each field
        changed_count = {k: 0 for k in initial}
        n_sampled = 0
        for act_id, data in actions[:min(12, len(actions))]:
            g = copy.deepcopy(game)
            try:
                ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                g.perform_action(ai, raw=True)
            except:
                continue
            n_sampled += 1
            for k in initial:
                if getattr(g, k, initial[k]) != initial[k]:
                    changed_count[k] += 1
        # Also sample click actions so click-triggered transients are detected
        if hasattr(game, '_get_valid_actions'):
            try:
                for va in game._get_valid_actions()[:4]:
                    g = copy.deepcopy(game)
                    try:
                        g.perform_action(va, raw=True)
                    except:
                        continue
                    n_sampled += 1
                    for k in initial:
                        if getattr(g, k, initial[k]) != initial[k]:
                            changed_count[k] += 1
            except:
                pass            
        if n_sampled == 0:
            return set()
        # A field is transient if it changed in every sampled action
        # Exclude monotonic counters (always decrease/increase) but keep boolean flags
        # Boolean flags encode meaningful state (e.g. which object is selected)
        transient = set()
        for k, cnt in changed_count.items():
            if cnt != n_sampled:
                continue
            v = initial[k]
            if isinstance(v, bool):
                continue  # boolean flags are meaningful state, never transient
            transient.add(k)
        if transient:
            logger.info(f"BFS: detected transient fields (excluded from hash): {transient}")
        return transient
    
    def _build_goal_heuristic(self, f_init, f_prev_win, demo_model=None):
    
        def count_indicators(game):
            try:
                total, satisfied = 0, 0
                for av in game.__dict__.values():
                    if not isinstance(av, dict): continue
                    for v in av.values():
                        if not isinstance(v, list): continue
                        for item in v:
                            if hasattr(item, 'is_visible') and hasattr(item, 'pixels'):
                                total += 1
                                if item.is_visible: satisfied += 1
                return total, satisfied
            except:
                return 0, 0
    
        # Cache selectable actions at heuristic build time, not per node
        cached_selectable_actions = []
        if self.game_cls:
            try:
                test = self.game_cls()
                test.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                test.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                if 6 in test._available_actions and hasattr(test, '_get_valid_actions'):
                    f0 = np.array(test.perform_action(
                        ActionInput(id=GameAction.ACTION1), raw=True).frame[-1])
                    bg = int(np.bincount(f0.flatten(), minlength=16).argmax())
                    # detect once here, store action inputs only
                    seen = set()
                    for va in test._get_valid_actions():
                        act_id = va.id._value_ if hasattr(va.id, '_value_') else int(va.id)
                        if act_id == 6:
                            cached_selectable_actions.append(va)
            except:
                pass
    
        def introspection_heuristic(f, game=None):
            if game is None:
                return 0
            try:
                total, satisfied = count_indicators(game)
                if total == 0:
                    return 0
                base_cost = total - satisfied
                # Use pre-cached selectable actions — no deepcopy detection per node
                extra_cost = 0
                for va in cached_selectable_actions:
                    gc = copy.deepcopy(game)
                    try:
                        gc.perform_action(va, raw=True)
                        t, s = count_indicators(gc)
                        if t > 0:
                            extra_cost += (t - s)
                    except:
                        pass
                return base_cost + extra_cost
            except:
                return 0
    
        # Validate
        if self.game_cls:
            try:
                test = self.game_cls()
                test.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                test.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                total, _ = count_indicators(test)
                if total > 0:
                    logger.info(f"BFS heuristic: introspection found {total} indicators")
                    return introspection_heuristic
            except:
                pass
    
        logger.info(f"BFS heuristic: no indicators found, uniform cost")
        return lambda f, game=None: 0
        
    def _scan_actions(self, game, f0, bg):
        """Scan for effective actions. Returns list of (action_id, data).

        v19: Records productive actions/clicks into cross-game memory (self.cgm)
        so subsequent instances of the same game type get priors.
        """
        avail = game._available_actions
        actions = []
        # v19: read priors from cross-game memory (if any) — try previously-known
        # productive actions FIRST, before scanning others. Generic prioritization.
        prior_actions = []
        prior_clicks = []
        if self.cgm:
            prior_actions = sorted(self.cgm.get('productive_actions', set()))
            prior_clicks = list(self.cgm.get('productive_clicks', []))[:24]
        # Directional/interact actions
        base_scalars = {k: v for k, v in game.__dict__.items()
                       if isinstance(v, (int, float, bool))
                       and not k.startswith('__')
                       and k not in ('_action_count', '_full_reset', '_action_complete')}
        for a in [a for a in avail if a <= 5]:
            actions.append((a, None))
        # Click actions — use _get_valid_actions() if available (much faster and correct)
        if 6 in avail:
            seen_effects = set()
            # Primary: use game's own valid action list for exact click coords
            if hasattr(game, '_get_valid_actions'):
                try:
                    valid = game._get_valid_actions()
                    for ai_obj in valid:
                        act_id = ai_obj.id._value_ if hasattr(ai_obj.id, '_value_') else int(ai_obj.id)
                        if act_id == 6:
                            g = copy.deepcopy(game)
                            try:
                                r = g.perform_action(ai_obj, raw=True)
                                if r.frame:
                                    f = np.array(r.frame[-1])
                                    diff = np.sum(f0 != f)
                                    if diff > 0:
                                        eh = hashlib.md5(f.tobytes()).hexdigest()[:12]
                                        if eh not in seen_effects:
                                            seen_effects.add(eh)
                                            actions.append((6, ai_obj.data))
                                        # v19: record productive click into cross-game memory
                                        if self.cgm is not None and isinstance(ai_obj.data, dict):
                                            x, y = ai_obj.data.get('x'), ai_obj.data.get('y')
                                            if x is not None and y is not None:
                                                self.cgm['productive_clicks'].append((x, y))
                                                if len(self.cgm['productive_clicks']) > 200:
                                                    self.cgm['productive_clicks'] = self.cgm['productive_clicks'][-100:]
                            except:
                                pass
                except:
                    pass
            # Fallback: pixel scan if _get_valid_actions unavailable
            if not seen_effects:
                t0 = time.time()
                for y in range(0, 64, 2):
                    if time.time() - t0 > self.scan_timeout:
                        break
                    for x in range(0, 64, 2):
                        if f0[y, x] == bg:
                            continue
                        g = copy.deepcopy(game)
                        try:
                            r = g.perform_action(ActionInput(id=GameAction.ACTION6, data={'x': x, 'y': y}), raw=True)
                            if not r.frame:
                                continue
                            f = np.array(r.frame[-1])
                            diff = np.sum(f0 != f)
                            if diff > 0:
                                effect_hash = hashlib.md5(f.tobytes()).hexdigest()[:12]
                                if effect_hash not in seen_effects:
                                    seen_effects.add(effect_hash)
                                    actions.append((6, {'x': x, 'y': y}))
                                # v19: record productive click into cross-game memory
                                if self.cgm is not None:
                                    self.cgm['productive_clicks'].append((x, y))
                                    if len(self.cgm['productive_clicks']) > 200:
                                        self.cgm['productive_clicks'] = self.cgm['productive_clicks'][-100:]
                        except:
                            pass
        # v19: record all productive action_ids found this scan
        if self.cgm is not None:
            for act_id, _ in actions:
                self.cgm['productive_actions'].add(act_id)
        return actions
        
    def _probe_mover_target_colors(self, game):
        """Classify colors as movers vs targets by running 20 random actions."""
        g = copy.deepcopy(game)
        avail = [a for a in game._available_actions if 1 <= a <= 4]
        if not avail:
            return set(), set()
        r0 = g.perform_action(ActionInput(id=GameAction.from_id(avail[0])), raw=True)
        if not r0.frame:
            return set(), set()
        f0 = np.array(r0.frame[-1])
        bg = int(np.bincount(f0.flatten(), minlength=16).argmax())
    
        def get_centroids(frame):
            result = {}
            for c in range(16):
                if c == bg: continue
                mask = (frame == c)
                n = int(np.sum(mask))
                if n < 2: continue
                ys, xs = np.where(mask)
                result[c] = (float(np.mean(xs)), float(np.mean(ys)))
            return result
    
        movement = {}
        prev_c = get_centroids(f0)
        for _ in range(20):
            act = random.choice(avail)
            try:
                r2 = g.perform_action(ActionInput(id=GameAction.from_id(act)), raw=True)
            except:
                break
            if not r2.frame:
                break
            curr_c = get_centroids(np.array(r2.frame[-1]))
            for c in prev_c:
                if c in curr_c:
                    movement[c] = movement.get(c, 0.0) + abs(curr_c[c][0] - prev_c[c][0]) + abs(curr_c[c][1] - prev_c[c][1])
            prev_c = curr_c
    
        mover_colors  = {c for c, m in movement.items() if m > 5}
        target_colors = {c for c, m in movement.items() if m == 0}
        return mover_colors, target_colors
    
    def solve_level(self, level_idx, max_states=500000, prev_solution=None, goal_heuristic=None):
        """Find optimal solution for a level via BFS (Memory Optimised via Action Replay)."""
        if not self.game_cls:
            return None

        game = self.game_cls()
        game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        r0 = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)

        # Advance to target level by replaying previous solutions
        last_r = r0
        for prev_idx in range(level_idx):
            prev_sol = self.solutions.get(prev_idx)
            if not prev_sol:
                return None
            for act_id, data in prev_sol:
                ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                last_r = game.perform_action(ai, raw=True)

        if not last_r.frame:
            return None
        f0 = np.array(last_r.frame[-1])
        bg = int(np.bincount(f0.flatten(), minlength=16).argmax())

        # Try solution transfer from previous level first
        if prev_solution and level_idx > 0:
            transfer_result = self._try_transfer(game, level_idx, prev_solution, f0)
            if transfer_result:
                return transfer_result

        # Phase 1: Scan for effective actions
        actions = self._scan_actions(game, f0, bg)

        # Warm-up unlock for locked initial states (sc25-type)
        if not actions:
            avail = game._available_actions
            # Try all non-reset actions as warmup, including clicks
            warmup_candidates = [a for a in avail if 1 <= a <= 5]
            # Also try click actions from _get_valid_actions if available
            if 6 in avail and hasattr(game, '_get_valid_actions'):
                try:
                    for va in game._get_valid_actions():
                        act_id = va.id._value_ if hasattr(va.id, '_value_') else int(va.id)
                        if act_id == 6:
                            g_warmup = _fast_deepcopy(game)
                            try:
                                g_warmup.perform_action(va, raw=True)
                                f_after = np.array(g_warmup.perform_action(
                                    ActionInput(id=GameAction.ACTION1), raw=True).frame[-1])
                                warmup_actions = self._scan_actions(g_warmup, f_after, bg)
                                if warmup_actions:
                                    logger.info(f"BFS L{level_idx}: UNLOCKED with click! {len(warmup_actions)} actions")
                                    game = g_warmup; f0 = f_after; actions = warmup_actions
                                    break
                            except:
                                pass
                except:
                    pass
            if not actions:
                for warmup_id in [a for a in avail if a <= 4]:
                    g_warmup = _fast_deepcopy(game)
                    try:
                        g_warmup.perform_action(ActionInput(id=GameAction.from_id(warmup_id)), raw=True)
                        f_after = np.array(g_warmup.get_pixels(0, 0, 64, 64))
                        warmup_actions = self._scan_actions(g_warmup, f_after, bg)
                        if warmup_actions:
                            logger.info(f"BFS L{level_idx}: UNLOCKED with ACTION{warmup_id}! {len(warmup_actions)} actions")
                            game = g_warmup; f0 = f_after; actions = warmup_actions
                            break
                    except:
                        pass

        logger.info(f"BFS L{level_idx}: {len(actions)} effective actions")
        if not actions:
            return None

       # ==========================================
        # Phase 2: A* with goal heuristic from prev level
        # ==========================================
        import heapq
        hidden_fields = None
        transient_fields = self._detect_transient_fields(game, actions)
        visited = set()
        h0 = self._state_hash(game, f0, None, transient_fields=transient_fields)
        visited.add(h0)
        base_game = _fast_deepcopy(game)

        hfn = goal_heuristic if goal_heuristic is not None else (lambda f, game=None: 0)
        # If heuristic is flat (no goal_heuristic provided or indicator-based),
        # probe mover/target colors and use distance heuristic instead
        
        _hfn_uses_game = goal_heuristic is not None
        counter = 0
        pq = [(hfn(f0, game) * 10, 0, counter, [], base_game)]
        t0 = time.time()
        explored = 0

        while pq and explored < max_states and (time.time() - t0) < self.bfs_timeout:
            f_score, g_score, _, hist, node_game = heapq.heappop(pq)
            
            for act_id, data in actions:
                g2 = _fast_deepcopy(node_game)
                try:
                    ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                    r = g2.perform_action(ai, raw=True)
                except:
                    continue
                explored += 1

                if not r.frame:
                    continue
                f = np.array(r.frame[-1])
                h = self._state_hash(g2, f, hidden_fields, transient_fields=transient_fields)
                if h in visited:
                    continue
                visited.add(h)

                new_hist = hist + [(act_id, data)]
                new_g = g_score + 1

                if r.levels_completed > level_idx or g2._current_level_index > level_idx:
                    elapsed = time.time() - t0
                    logger.info(f"BFS L{level_idx}: SOLVED (A*) in {len(new_hist)} actions ({explored} explored, {elapsed:.1f}s)")
                    self.solutions[level_idx] = new_hist
                    return new_hist

                h_val = hfn(f, g2 if _hfn_uses_game else None) * 10 
                counter += 1
                heapq.heappush(pq, (new_g + h_val, new_g, counter, new_hist, g2))

        elapsed_first = time.time() - t0
        logger.info(f"BFS L{level_idx}: first pass timeout ({explored} explored, {len(visited)} unique, {elapsed_first:.1f}s)")
        self.timed_out_levels.add(level_idx)
        # Dynamic action rescan BFS — triggers when state space exhausted quickly
        # indicating actions expand as state evolves (e.g. flood fill games)
        exhausted_quickly = len(pq) == 0 and elapsed_first < self.bfs_timeout * 0.5
        if exhausted_quickly:
            logger.info(f"BFS L{level_idx}: queue exhausted early — retrying with dynamic action rescan")
            visited_d = set()
            visited_d.add(self._state_hash(base_game, f0, hidden_fields, transient_fields=transient_fields))
            queue_d = deque()
            queue_d.append(([], 0, base_game))
            t0_d = time.time()
            explored_d = 0
            remaining_d = max(30, self.bfs_timeout - elapsed_first)
            current_actions = list(actions)

            while queue_d and explored_d < max_states * 10 and (time.time() - t0_d) < remaining_d:
                hist_d, depth_d, node_game_d = queue_d.popleft()

                for act_id, data in current_actions:
                    g2_d = _fast_deepcopy(node_game_d)
                    try:
                        ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                        r = g2_d.perform_action(ai, raw=True)
                    except:
                        continue
                    explored_d += 1
                    if not r.frame:
                        continue
                    f2_d = np.array(r.frame[-1])
                    h_d = self._state_hash(g2_d, f2_d, hidden_fields, transient_fields=transient_fields)
                    if h_d in visited_d:
                        continue
                    visited_d.add(h_d)
                    # Rescan from child state to find newly unlocked actions
                    try:
                        new_acts = self._scan_actions(g2_d, f0, bg)
                        added = [a for a in new_acts if a not in current_actions]
                        if added:
                            logger.info(f"BFS L{level_idx}: rescan found {len(added)} new actions at depth {depth_d}")
                            current_actions.extend(added)
                    except:
                        pass
                    new_hist_d = hist_d + [(act_id, data)]
                    if r.levels_completed > level_idx or g2_d._current_level_index > level_idx:
                        logger.info(f"BFS L{level_idx}: SOLVED (dynamic rescan) in {len(new_hist_d)} actions ({explored_d} explored)")
                        self.solutions[level_idx] = new_hist_d
                        return new_hist_d
                    if depth_d < 30:
                        queue_d.append((new_hist_d, depth_d + 1, g2_d))

            logger.info(f"BFS L{level_idx}: dynamic rescan also failed ({explored_d} explored)")

        # Smart early exit — game may be too expensive to BFS
        if explored < 20 and elapsed_first > 10.0:
            logger.info(f"BFS L{level_idx}: early exit (only {explored} explored in {elapsed_first:.1f}s) — handing off to CNN")
            return None

        # If too few unique states found → hidden state detected → retry with probed fields
        if explored > 0 and (len(visited) < 200 or explored / len(visited) > 5) and elapsed_first < self.bfs_timeout * 0.8:
            hidden_fields = self._probe_hidden_fields(game, actions)
            if hidden_fields:
                logger.info(f"BFS L{level_idx}: RETRY with hidden fields: {hidden_fields}")

                # FIX 3: Use exactly 2 RESET calls (not 3) to match the first pass baseline
                game2 = self.game_cls()
                game2.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                last_r2 = game2.perform_action(ActionInput(id=GameAction.RESET), raw=True)

                for prev_idx in range(level_idx):
                    prev_sol = self.solutions.get(prev_idx)
                    if not prev_sol:
                        return None
                    for act_id, data in prev_sol:
                        ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                        last_r2 = game2.perform_action(ai, raw=True)

                if not last_r2.frame:
                    return None
                f0_2 = np.array(last_r2.frame[-1])
                h0_2 = self._state_hash(game2, f0_2, hidden_fields, transient_fields=transient_fields)

                base_game2 = _fast_deepcopy(game2)
                visited2 = set()
                visited2.add(h0_2)
                queue2 = deque()
                queue2.append(([], 0, base_game2))

                t0_2 = time.time()
                explored2 = 0
                remaining = max(30, self.bfs_timeout - elapsed_first)

                while queue2 and explored2 < max_states and (time.time() - t0_2) < remaining:
                    hist, depth, node_game2 = queue2.popleft()

                    for act_id, data in actions:
                        g2 = _fast_deepcopy(node_game2)
                        try:
                            ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                            r = g2.perform_action(ai, raw=True)
                        except:
                            continue
                        explored2 += 1

                        if not r.frame:
                            continue
                        f = np.array(r.frame[-1])
                        h = self._state_hash(g2, f, hidden_fields, transient_fields=transient_fields)
                        if h in visited2:
                            continue
                        visited2.add(h)

                        new_hist = hist + [(act_id, data)]

                        if r.levels_completed > level_idx or g2._current_level_index > level_idx:
                            logger.info(f"BFS L{level_idx}: SOLVED (hidden retry) in {len(new_hist)} actions ({explored2} explored)")
                            self.solutions[level_idx] = new_hist
                            return new_hist

                        if depth < 30:
                            queue2.append((new_hist, depth + 1, g2))

                logger.info(f"BFS L{level_idx}: hidden retry also failed ({explored2} explored, {len(visited2)} unique)")

        return None

    def _try_transfer(self, game, level_idx, prev_solution, f1):
        """Transfer previous level's solution to current level."""
        try:
            # Try executing prev solution directly
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

            # Try object-relative transfer
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
                except:
                    break

        except Exception as e:
            logger.warning(f"BFS transfer failed: {e}")
        return None


def find_game_source_and_class(game_id, arc_env=None):
    """Find the game .py file and class name."""
    import re

    # game_id format: sk48-d8078629
    # file lives at: .../environment_files/sk48/d8078629/sk48.py
    parts = game_id.split('-', 1)
    gid = parts[0]                          # e.g. sk48
    guid_suffix = parts[1] if len(parts) > 1 else ''  # e.g. d8078629

    # Primary: competition path on Kaggle
    competition_path = (
        f"/kaggle/input/competitions/arc-prize-2026-arc-agi-3"
        f"/environment_files/{gid}/{guid_suffix}/{gid}.py"
    )
    if os.path.exists(competition_path):
        src = competition_path
        content = open(src).read()[:2000]
        m = re.search(r'class\s+(\w+)\s*\(', content)
        cls_name = m.group(1) if m else gid[0].upper() + gid[1:]
        logger.info(f"BFS: found game source at {src}, class={cls_name}")
        return src, cls_name

    # Fallback: broad glob search
    for pattern in [
        f"/kaggle/input/**/{gid}.py",
        f"/tmp/**/{gid}.py",
        f"/kaggle/working/**/{gid}.py",
    ]:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            src = matches[0]
            content = open(src).read()[:2000]
            m = re.search(r'class\s+(\w+)\s*\(', content)
            cls_name = m.group(1) if m else gid[0].upper() + gid[1:]
            logger.info(f"BFS: found game source at {src}, class={cls_name}")
            return src, cls_name

    logger.warning(f"BFS: game source not found for {game_id}")
    return None, gid[0].upper() + gid[1:]


# ==================== CNN FALLBACK ====================

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


def fast_objects(frame, bg, exclude_colours=None, static_mask=None):
    if exclude_colours is None:
        exclude_colours = set()
    objs = []
    for c in range(16):
        if c == bg or c in exclude_colours:
            continue
        if static_mask is not None:
            mask = (frame == c) & ~static_mask
        else:
            mask = (frame == c)
        npix = int(np.sum(mask))
        if npix < 4 or npix > 3000:
            continue
        ys, xs = np.where(mask)
        objs.append((c, float(np.mean(xs)), float(np.mean(ys)), npix,
                     int(xs.max()-xs.min()), int(ys.max()-ys.min()),
                     int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))
    return objs


def find_composite_objects(objs, proximity=6):
    if not objs:
        return []
    n = len(objs)
    adjacent = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            oi, oj = objs[i], objs[j]
            x_gap = max(0, max(oi[6], oj[6]) - min(oi[8], oj[8]))
            y_gap = max(0, max(oi[7], oj[7]) - min(oi[9], oj[9]))
            if x_gap <= proximity and y_gap <= proximity:
                adjacent[i].add(j)
                adjacent[j].add(i)
    visited = [False] * n
    groups = []
    for i in range(n):
        if visited[i]:
            continue
        group = []
        stack = [i]
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            group.append(node)
            stack.extend(adjacent[node] - set(g for g in group))
        groups.append([objs[k] for k in group])
    filtered = []
    for group in groups:
        x_min = min(o[6] for o in group)
        y_min = min(o[7] for o in group)
        x_max = max(o[8] for o in group)
        y_max = max(o[9] for o in group)
        area = (x_max - x_min + 1) * (y_max - y_min + 1)
        if area < 64 * 64 * 0.4:
            filtered.append(group)
    return filtered


# ==================== AGENT ====================

# ===== GraphExplorer + FrameProcessor (from v24) =====
INFINITY = np.iinfo(np.int32).max


# NOTE: all data formats here chosen crudely, to be optimized later
edge_dtype = np.dtype([
    ("group", "i4"), # 0-indexed group id
    ("result", "i4"), # 1 if success, -1 if failed, 0 if not tested yet
    ("target", "U32"), # target node hash-name, "" if not tested or failed
    ("distance", "i4"), # distance to the frontier node, 0 means next node is the frontier
    ("errors", "i4"), # number of errors so far
])

def format_struct_table(arr):
    names = ("idx",) + arr.dtype.names
    cols = []
    for name in names:
        if name == "idx":
            cols.append([str(i) for i in range(len(arr))])
        else:
            cols.append([str(r[name]) for r in arr])
    widths = [max(len(n), *(len(v) for v in col)) for n, col in zip(names, cols)]
    header = " | ".join(n.ljust(w) for n, w in zip(names, widths))
    sep = "-+-".join("-"*w for w in widths)
    lines = []
    for i in range(len(arr)):
        line = " | ".join(cols[j][i].ljust(widths[j]) for j in range(len(names)))
        lines.append(line)
    return "\n".join([header, sep, *lines])

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
        # v19 FIX 1: deterministic seed — same game_id always gets same seed.
        # Removes time.time() variance that was causing ±0.10 score swings.
        # Generalizes: seed depends only on game identity, not submission time.
        seed = abs(hash(s.game_id)) % (2**32 - 1)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
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
        # FIX 1: Initialize _visited_hashes so _reward() deduplication works correctly
        s._visited_hashes = set()
        # BFS solver
        s._bfs = None
        s._bfs_solution = None
        s._bfs_step = 0
        s._bfs_tried = False
        # v25 hybrid: graph-explorer state for fallback
        s._ge_fp = FrameProcessor()
        s._ge = GraphExplorer(verbose_level=0, n_groups=5)
        s._ge_status_mask = None
        s._ge_last_hash = None
        s._ge_last_action_id = None
        s._ge_level = -1
        s._ge_action_to_action_groups = None  # cached for safety
        s._ge_failed = False

        # v20: First-step verification gate. After applying BFS step 0 to live env,
        # we'll re-simulate it in sim and compare frames; if mismatch, BFS path is
        # stale (sim/runtime divergence) and we abandon it to fall through to CNN.
        s._bfs_verified = False
        # v19 FIX 2: cross-game type memory — generic priors for repeating game types.
        # game_type = first segment of game_id (e.g. 'cd82' from 'cd82-fb555c5d').
        # Persists across MyAgent instances within the same Kaggle run via module-level dict.
        # On instance 2+ of same game type, agent already knows productive actions/clicks.
        try:
            game_type = s.game_id.split('-', 1)[0]
        except Exception:
            game_type = 'unknown'
        s._game_type = game_type
        s._cgm = _CROSS_GAME_MEMORY.setdefault(game_type, {
            'productive_actions': set(),  # action_ids that produced frame changes
            'productive_clicks': [],       # list of (x,y) positions that activated sprites
            'effect_signatures': set(),    # frozenset of (color_in, color_out) seen
            'win_actions': [],             # action sequences that won levels
            'visit_count': 0,              # how many times we've seen this game type
        })
        s._cgm['visit_count'] += 1
        if s._cgm['visit_count'] > 1:
            logger.info(f"CGM: game_type={game_type} visit #{s._cgm['visit_count']}, "
                        f"priors: {len(s._cgm['productive_actions'])} actions, "
                        f"{len(s._cgm['productive_clicks'])} clicks, "
                        f"{len(s._cgm['win_actions'])} past wins")

        # Object model
        s._frame_buffer = []
        s._static_mask = None
        s._dynamic_mask = None
        s._static_ready = False
        s._structural_colours = set()
        s._target_colours = set()
        s._goal_groups = []
        s._bg = 0

    def append_frame(s, f):
        s.frames.append(f)
        if len(s.frames) > s._MAX_FRAMES: s.frames = s.frames[-s._MAX_FRAMES:]
        if f.guid: s.guid = f.guid
        if hasattr(s, "recorder") and not s.is_playback:
            import json; s.recorder.record(json.loads(f.model_dump_json()))

    def _lvl(s, f): return getattr(f, 'score', None) or f.levels_completed
    def _raw(s, fd): return np.array(fd.frame, dtype=np.int64)[-1]

    def _bfs_predict_frame(s, level_idx, n_steps):
        """v20: Replay first n_steps of current BFS plan in a fresh sim instance.
        Returns the predicted final frame[-1] (np.int64 array) or None on failure.
        Used by the first-step verification gate to detect sim/runtime divergence.
        """
        try:
            if not s._bfs or not s._bfs.game_cls or not s._bfs_solution:
                return None
            g = s._bfs.game_cls()
            g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            # Replay any prior-level solutions to reach the current level
            for pi in range(level_idx):
                prev_sol = s._bfs.solutions.get(pi) or []
                for act_id, data in prev_sol:
                    ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                    g.perform_action(ai, raw=True)
            # Apply the first n_steps of the current BFS plan
            last_r = None
            for act_id, data in s._bfs_solution[:n_steps]:
                ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                last_r = g.perform_action(ai, raw=True)
            if last_r and last_r.frame:
                return np.array(last_r.frame, dtype=np.int64)[-1]
            return None
        except Exception as e:
            logger.warning(f"_bfs_predict_frame error: {e}")
            return None

    def _init_bfs(s):
        """Initialize BFS solver on first call."""
        src, cls = find_game_source_and_class(s.game_id, s.arc_env)
        if src:
            s._bfs = BFSSolver(src, cls, scan_timeout=5, bfs_timeout=180, cgm=s._cgm)
            if s._bfs.load():
                logger.info(f"BFS: loaded {cls} from {src}")
            else:
                s._bfs = None
                logger.warning(f"BFS: failed to load game class")
        else:
            logger.warning(f"BFS: game source not found for {s.game_id}")
            
    def _update_object_model(s, prev_raw, curr_raw, last_action_idx, last_action_data):
        """
        Maintains a provisional static/dynamic classification of objects.
        
        Objects are classified as STATIC (candidate targets) if they have not
        moved across multiple frames. However, if an action causes a previously
        static object to change (move, appear, disappear), it is immediately
        reclassified as DYNAMIC and removed from the target set.
        
        This means targets are always provisional — interaction can reveal
        that a 'static' object is actually responsive.
        """
        if not s._static_ready:
            s._frame_buffer.append(curr_raw.copy())
            if len(s._frame_buffer) >= 4:
                # Build initial static mask from first N frames
                base = s._frame_buffer[0]
                static = np.ones((64, 64), dtype=bool)
                for f in s._frame_buffer[1:]:
                    static &= (f == base)
                s._static_mask = static
                s._dynamic_mask = ~static
                s._static_ready = True
                
                cnt = np.bincount(curr_raw.flatten(), minlength=16)
                s._bg = int(cnt.argmax())
                
                # Identify structural colours (large static regions = play area border)
                cnt_static = np.bincount(curr_raw[s._static_mask].flatten(), minlength=16)
                cnt_static[s._bg] = 0
                structural_col = int(cnt_static.argmax())
                s._structural_colours = {structural_col} if cnt_static[structural_col] > 200 else set()
                
                # Initial target detection: rare static colours are candidate targets
                s._target_colours = set()
                for c in range(16):
                    if c == s._bg or c in s._structural_colours:
                        continue
                    n_static = int(np.sum(s._static_mask & (curr_raw == c)))
                    if 2 <= n_static <= 200:
                        s._target_colours.add(c)
                
                logger.info(f"Object model: bg={s._bg} structural={s._structural_colours} targets={s._target_colours}")

                # Detect goal groups by spatially clustering rare static pixels
                # Works regardless of where goals appear on screen
                from collections import defaultdict
                s._goal_groups = []
                rare_pixels = []
                for c in s._target_colours:
                    ys, xs = np.where(s._static_mask & (curr_raw == c))
                    for y, x in zip(ys, xs):
                        rare_pixels.append((int(x), int(y), c))

                if rare_pixels:
                    cluster_ids = list(range(len(rare_pixels)))

                    def find(i):
                        while cluster_ids[i] != i:
                            cluster_ids[i] = cluster_ids[cluster_ids[i]]
                            i = cluster_ids[i]
                        return i

                    def union(i, j):
                        ri, rj = find(i), find(j)
                        if ri != rj:
                            cluster_ids[ri] = rj

                    for i in range(len(rare_pixels)):
                        for j in range(i+1, len(rare_pixels)):
                            xi, yi, _ = rare_pixels[i]
                            xj, yj, _ = rare_pixels[j]
                            if abs(xi-xj) <= 12 and abs(yi-yj) <= 12:
                                union(i, j)

                    clusters = defaultdict(set)
                    for i, (x, y, c) in enumerate(rare_pixels):
                        clusters[find(i)].add(c)

                    s._goal_groups = [cols for cols in clusters.values()]
                    logger.info(f"Object model: detected {len(s._goal_groups)} goal groups: {s._goal_groups}")
            return

        # Already have a static mask — check if this action disturbed any static object
        diff = (prev_raw != curr_raw)
        if not np.any(diff):
            return

        # Check which previously-static colours changed
        disturbed = set()
        for c in s._target_colours | s._structural_colours:
            prev_static_pixels = s._static_mask & (prev_raw == c)
            if np.any(prev_static_pixels & diff):
                disturbed.add(c)

        if disturbed:
            # Reclassify disturbed colours as dynamic — they are NOT fixed targets
            for c in disturbed:
                s._target_colours.discard(c)
                # Update static mask to mark these pixels as dynamic
                s._static_mask[curr_raw == c] = False
                s._static_mask[prev_raw == c] = False
            s._dynamic_mask = ~s._static_mask
            logger.info(f"Object model: reclassified as dynamic after interaction: {disturbed}")

        # Also update static mask by removing any pixel that changed
        # This handles gradual revelation of dynamic objects
        s._static_mask[diff] = False
        s._dynamic_mask = ~s._static_mask
    def _try_bfs_solve(s, level_idx):
        """Try to solve current level. For L1+, uses A* with a goal
        heuristic derived from the previous level's win frame."""
        if s._bfs is None:
            return None

        prev_sol = s._bfs.solutions.get(level_idx - 1) if level_idx > 0 else None
        goal_heuristic = None

        # In _try_bfs_solve, replace the cumulative heuristic block with:
        if level_idx > 0 and prev_sol is not None:
            try:
                g = s._bfs.game_cls()
                g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                last_r = g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                level_heuristics = []
        
                for pi in range(level_idx):
                    ps = s._bfs.solutions.get(pi)
                    if not ps:
                        break
                    f_level_init = np.array(last_r.frame[-1])
                    for act_id, data in ps:
                        ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                        last_r = g.perform_action(ai, raw=True)
                    f_level_win = np.array(last_r.frame[-1])
                    # Build heuristic once per level, reuse cached selectable actions
                    hfn = s._bfs._build_goal_heuristic(f_level_init, f_level_win)
                    level_heuristics.append((hfn, pi + 1))  # single replay, no re-instantiation
        
                if level_heuristics:
                    total_weight = sum(w for _, w in level_heuristics)
                    def goal_heuristic(f, game=None, _h=level_heuristics, _t=total_weight):
                        return sum(hfn(f, game) * w for hfn, w in _h) / _t

            except Exception as e:
                logger.warning(f"BFS L{level_idx}: goal heuristic failed: {e}")
                # Build demo model from prev level solution
                demo_model = None
                try:
                    g_demo = s._bfs.game_cls()
                    g_demo.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                    g_demo.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                    for pi in range(level_idx - 1):
                        ps = s._bfs.solutions.get(pi)
                        if not ps:
                            raise ValueError(f"missing L{pi}")
                        for act_id, data in ps:
                            ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                            g_demo.perform_action(ai, raw=True)
                    frames_and_actions = [(f_prev_init, None)]
                    for act_id, data in prev_sol:
                        ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                        r = g_demo.perform_action(ai, raw=True)
                        if r.frame:
                            frames_and_actions.append((np.array(r.frame[-1]), act_id))
                    demo_model = s._bfs._analyse_demo(frames_and_actions)
                except Exception as e:
                    logger.warning(f"BFS demo analysis failed: {e}")

                goal_heuristic_raw = s._bfs._build_goal_heuristic(f_prev_init, f_prev_win, demo_model=demo_model)
                
                # Calibrate: evaluate heuristic after one move to get baseline offset
                # L1 starts at L0 win state so raw h=0 there — we need relative change
                try:
                    g_cal = s._bfs.game_cls()
                    g_cal.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                    g_cal.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                    for pi in range(level_idx):
                        ps = s._bfs.solutions.get(pi)
                        if not ps: break
                        for act_id, data in ps:
                            ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                            g_cal.perform_action(ai, raw=True)
                    # Take one step to move away from L0 win state
                    r_cal = g_cal.perform_action(ActionInput(id=GameAction.ACTION1), raw=True)
                    if r_cal.frame:
                        f_after_move = np.array(r_cal.frame[-1])
                        h_after_move = goal_heuristic_raw(f_after_move, g_cal)
                        h_init = goal_heuristic_raw(f_prev_win, None)
                        logger.info(f"BFS L{level_idx}: heuristic calibration h_init={h_init:.2f} h_after_move={h_after_move:.2f}")
                        if h_after_move > h_init:
                            # Heuristic is working — use as-is
                            goal_heuristic = goal_heuristic_raw
                        else:
                            # Heuristic is flat — offset by subtracting init value
                            h_offset = h_init
                            def goal_heuristic(f, game=None, _offset=h_offset, _raw=goal_heuristic_raw):
                                return _raw(f, game) - _offset
                    else:
                        goal_heuristic = goal_heuristic_raw
                except Exception as e:
                    logger.warning(f"BFS heuristic calibration failed: {e}")
                    goal_heuristic = goal_heuristic_raw

        # Validate heuristic is not flat — if it is, replace with distance heuristic
        if goal_heuristic is not None:
            try:
                g_val = s._bfs.game_cls()
                g_val.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                last_r_val = g_val.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                for pi in range(level_idx):
                    ps = s._bfs.solutions.get(pi)
                    if not ps: break
                    for act_id, data in ps:
                        ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                        last_r_val = g_val.perform_action(ai, raw=True)
                if last_r_val.frame:
                    f_val = np.array(last_r_val.frame[-1])
                    h_vals = set()
                    h_vals.add(round(goal_heuristic(f_val, g_val), 4))
                    avail_val = [a for a in g_val._available_actions if 1 <= a <= 4]
                    for act_id in avail_val[:4]:
                        g2_val = copy.deepcopy(g_val)
                        r2_val = g2_val.perform_action(ActionInput(id=GameAction.from_id(act_id)), raw=True)
                        if r2_val.frame:
                            h_vals.add(round(goal_heuristic(np.array(r2_val.frame[-1]), g2_val), 4))
                    if len(h_vals) == 1 and level_idx in s._bfs.timed_out_levels:
                        logger.info(f"BFS L{level_idx}: heuristic is flat (h={list(h_vals)[0]}), switching to distance heuristic")
                        mover_colors, target_colors = s._bfs._probe_mover_target_colors(g_val)
                        if mover_colors and target_colors:
                            def goal_heuristic(f, game=None, _m=mover_colors, _t=target_colors):
                                centroids = {}
                                for c in range(16):
                                    mask = (f == c)
                                    n = int(np.sum(mask))
                                    if n < 2: continue
                                    ys, xs = np.where(mask)
                                    centroids[c] = (float(np.mean(xs)), float(np.mean(ys)))
                                targets = [(centroids[tc][0], centroids[tc][1]) for tc in _t if tc in centroids]
                                if not targets: return 0
                                total = 0
                                for mc in _m:
                                    if mc not in centroids: continue
                                    mx, my = centroids[mc]
                                    total += min(abs(mx - tx) + abs(my - ty) for tx, ty in targets)
                                return total
                            logger.info(f"BFS L{level_idx}: distance heuristic movers={mover_colors} targets={target_colors}")
            except Exception as e:
                logger.warning(f"BFS L{level_idx}: heuristic validation failed: {e}")
        
        sol = s._bfs.solve_level(level_idx, prev_solution=prev_sol, goal_heuristic=goal_heuristic)
        if sol:
            s._bfs_solution = sol
            s._bfs_step = 0
            return sol
        
        # First attempt failed — check if heuristic was flat and retry with distance heuristic
        if level_idx in s._bfs.timed_out_levels:
            try:
                g_val = s._bfs.game_cls()
                g_val.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                last_r_val = g_val.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                for pi in range(level_idx):
                    ps = s._bfs.solutions.get(pi)
                    if not ps: break
                    for act_id, data in ps:
                        ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                        last_r_val = g_val.perform_action(ai, raw=True)
                if last_r_val.frame:
                    f_val = np.array(last_r_val.frame[-1])
                    h_vals = set()
                    h_val_hfn = goal_heuristic if goal_heuristic is not None else (lambda f, game=None: 0)
                    h_vals.add(round(h_val_hfn(f_val, g_val), 4))
                    for act_id in [a for a in g_val._available_actions if 1 <= a <= 4][:4]:
                        g2_val = copy.deepcopy(g_val)
                        r2_val = g2_val.perform_action(ActionInput(id=GameAction.from_id(act_id)), raw=True)
                        if r2_val.frame:
                            h_vals.add(round(h_val_hfn(np.array(r2_val.frame[-1]), g2_val), 4))
                    if len(h_vals) == 1:
                        logger.info(f"BFS L{level_idx}: heuristic was flat — retrying with distance heuristic")
                        mover_colors, target_colors = s._bfs._probe_mover_target_colors(g_val)
                        if mover_colors and target_colors:
                            def dist_heuristic(f, game=None, _m=mover_colors, _t=target_colors):
                                centroids = {}
                                for c in range(16):
                                    mask = (f == c)
                                    n = int(np.sum(mask))
                                    if n < 2: continue
                                    ys, xs = np.where(mask)
                                    centroids[c] = (float(np.mean(xs)), float(np.mean(ys)))
                                targets = [(centroids[tc][0], centroids[tc][1]) for tc in _t if tc in centroids]
                                if not targets: return 0
                                total = 0
                                for mc in _m:
                                    if mc not in centroids: continue
                                    mx, my = centroids[mc]
                                    total += min(abs(mx - tx) + abs(my - ty) for tx, ty in targets)
                                return total
                            logger.info(f"BFS L{level_idx}: distance heuristic movers={mover_colors} targets={target_colors}")
                            sol = s._bfs.solve_level(level_idx, prev_solution=prev_sol, goal_heuristic=dist_heuristic)
                            if sol:
                                s._bfs_solution = sol
                                s._bfs_step = 0
                                return sol
            except Exception as e:
                logger.warning(f"BFS L{level_idx}: distance heuristic retry failed: {e}")
        
        return None
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

    def _reward(s, prev_raw, curr_raw, prev_h, curr_h, last_action_idx=0, last_action_data=None):
        # Update object model with this transition
        s._update_object_model(prev_raw, curr_raw, last_action_idx, last_action_data)

        mask = np.ones((64,64), dtype=bool); mask[:2]=False; mask[62:]=False
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

        smask = s._static_mask if s._static_ready else None
        curr_objs = fast_objects(curr_raw, s._bg, s._structural_colours, smask)
        prev_objs = s._prev_objs or []

        prev_colors = {o[0] for o in prev_objs}
        curr_colors = {o[0] for o in curr_objs}

        # Object movement reward
        if prev_objs and curr_objs:
            moved = 0
            for co in curr_objs:
                for po in prev_objs:
                    if co[0] == po[0]:
                        dist = abs(co[1]-po[1]) + abs(co[2]-po[2])
                        if 2 < dist < 20:
                            moved += 1
                            break
            if moved > 0:
                r += 0.3 * min(moved, 3)
                s._obj_moved = moved

            # Contact reward: dynamic object touching a target
            # Tracks progress per goal group and applies diminishing returns
            # to groups already ahead, forcing balanced multi-goal solving
            if s._static_ready and s._target_colours:
                group_progress = {}
                for dobj in curr_objs:
                    d_col, d_cx, d_cy, d_npix, d_w, d_h, d_x0, d_y0, d_x1, d_y1 = dobj
                    for tc in s._target_colours:
                        if tc == d_col:
                            continue
                        rs_ys, rs_xs = np.where(s._static_mask & (curr_raw == tc))
                        if len(rs_xs) == 0:
                            continue
                        rs_x0, rs_x1 = int(rs_xs.min()), int(rs_xs.max())
                        rs_y0, rs_y1 = int(rs_ys.min()), int(rs_ys.max())
                        x_gap = max(0, max(d_x0, rs_x0) - min(d_x1, rs_x1))
                        y_gap = max(0, max(d_y0, rs_y0) - min(d_y1, rs_y1))
                        contact_score = 0.0
                        if x_gap <= 2 and y_gap <= 2:
                            contact_score = 2.0
                        elif x_gap <= 10 and y_gap <= 10:
                            contact_score = 0.5
                        if contact_score > 0:
                            group_idx = None
                            for gi, grp in enumerate(s._goal_groups):
                                if tc in grp:
                                    group_idx = gi
                                    break
                            if group_idx is not None:
                                group_progress[group_idx] = max(
                                    group_progress.get(group_idx, 0.0),
                                    contact_score)
                            else:
                                r += contact_score

                if group_progress and s._goal_groups:
                    scores = [group_progress.get(i, 0.0) for i in range(len(s._goal_groups))]
                    for gi, score in enumerate(scores):
                        if score > 0:
                            other_scores = [sc for j, sc in enumerate(scores) if j != gi]
                            max_other = max(other_scores) if other_scores else 0.0
                            lag_bonus = 1.0 if score <= max_other else 0.5
                            r += score * lag_bonus
                elif group_progress:
                    for score in group_progress.values():
                        r += score

            # Composite object movement toward targets
            if s._static_ready and s._target_colours:
                prev_composites = find_composite_objects(prev_objs)
                curr_composites = find_composite_objects(curr_objs)
                for cc in curr_composites:
                    cc_cols = {o[0] for o in cc}
                    cc_cx = float(np.mean([o[1] for o in cc]))
                    cc_cy = float(np.mean([o[2] for o in cc]))
                    # Find nearest target
                    best_target_dist = 999.0
                    for tc in s._target_colours:
                        rs_ys, rs_xs = np.where(s._static_mask & (curr_raw == tc))
                        if len(rs_xs) == 0:
                            continue
                        td = abs(float(np.mean(rs_xs)) - cc_cx) + abs(float(np.mean(rs_ys)) - cc_cy)
                        best_target_dist = min(best_target_dist, td)
                    # Compare to previous position of same composite
                    for pc in prev_composites:
                        pc_cols = {o[0] for o in pc}
                        if cc_cols == pc_cols:
                            pc_cx = float(np.mean([o[1] for o in pc]))
                            pc_cy = float(np.mean([o[2] for o in pc]))
                            # Reward moving toward target
                            prev_target_dist = 999.0
                            for tc in s._target_colours:
                                rs_ys, rs_xs = np.where(s._static_mask & (curr_raw == tc))
                                if len(rs_xs) == 0:
                                    continue
                                td = abs(float(np.mean(rs_xs)) - pc_cx) + abs(float(np.mean(rs_ys)) - pc_cy)
                                prev_target_dist = min(prev_target_dist, td)
                            if prev_target_dist - best_target_dist > 1:
                                r += 0.4  # moved closer to a target
                            break

        # Disappeared object reward (pickup / elimination)
        disappeared = prev_colors - curr_colors
        if disappeared:
            r += 2.0 * len(disappeared)

        s._prev_objs = curr_objs
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

    def _ge_pick(s, lf, lvl):
        """Run one tick of the graph-explorer policy and return an action.

        Returns None on any unrecoverable issue so caller falls through.
        """
        # Reset GE on level change
        if lvl != s._ge_level:
            s._ge.reset()
            s._ge_status_mask = None
            s._ge_last_hash = None
            s._ge_last_action_id = None
            s._ge_level = lvl

        frame_np = np.array(lf.frame, dtype=np.uint8)
        if frame_np.size == 0:
            return None
        num_frames = frame_np.shape[0]
        frame_np = frame_np[-1].copy()

        level_up = (s._ge_status_mask is None) or s._ge_failed
        if level_up:
            seg, segs = s._ge_fp.segment_frame(frame_np)
            _, mask = s._ge_fp.identify_status_bars(seg, segs)
            s._ge_status_mask = mask
            s._ge_last_hash = None
            s._ge_last_action_id = None
            s._ge_failed = False

        if s._ge_status_mask is not None:
            frame_np[s._ge_status_mask] = 16
        segmented_frame, frame_segments = s._ge_fp.segment_frame(frame_np)
        avail_raw = list(getattr(lf, 'available_actions', []) or [])
        avail = [a.value if hasattr(a, 'value') else int(a) for a in avail_raw]

        SIMPLE = {1: GameAction.ACTION1, 2: GameAction.ACTION2,
                  3: GameAction.ACTION3, 4: GameAction.ACTION4,
                  5: GameAction.ACTION5}
        num_click_actions = 0
        num_actions = 0
        arrow_actions = []
        if 6 in avail:
            num_click_actions = len(frame_segments)
            num_actions = num_click_actions
            action_groups = s._ge_fp.frame_segments_to_action_groups(frame_segments, n_groups=5)
        else:
            action_groups = [set() for _ in range(5)]
        for aid in avail:
            if aid in SIMPLE:
                arrow_actions.append(SIMPLE[aid])
                action_groups[0].add(num_actions)
                num_actions += 1

        if num_actions == 0:
            return None

        frame_np[frame_np == 16] = 0
        hashed_frame = s._ge_fp.hash_frame(frame_np)

        if level_up:
            s._ge.initialize(start_node=hashed_frame, num_candidates=num_actions,
                             group2remaining_candidate_ids=action_groups)

        # Record transition from previous step
        if (not level_up) and s._ge_last_hash is not None and s._ge_last_action_id is not None:
            transition = hashed_frame != s._ge_last_hash
            try:
                s._ge.record_test(s._ge_last_hash, s._ge_last_action_id,
                                  int(transition), hashed_frame,
                                  target_num_candidates=num_actions,
                                  group2remaining_candidate_ids=action_groups,
                                  suspicious_transition=False)
            except Exception:
                # Stale graph or unknown source node — re-init from current
                s._ge.reset()
                s._ge.initialize(start_node=hashed_frame, num_candidates=num_actions,
                                 group2remaining_candidate_ids=action_groups)

        # If somehow current frame still missing, lazy add via re-init
        if hashed_frame not in s._ge._nodes:
            s._ge.reset()
            s._ge.initialize(start_node=hashed_frame, num_candidates=num_actions,
                             group2remaining_candidate_ids=action_groups)

        try:
            action_id = s._ge.choose_edge(hashed_frame, return_reasoning=False)
            action_id = int(action_id) if not isinstance(action_id, tuple) else int(action_id[0])
        except Exception:
            return None

        if action_id < num_click_actions:
            seg = frame_segments[action_id]
            seg_mask = (segmented_frame == action_id)
            pts = np.argwhere(seg_mask)
            if len(pts) == 0:
                bbox = seg.get("bbox") or seg.get("bounding_box")
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

        s._ge_last_hash = hashed_frame
        s._ge_last_action_id = action_id
        return action

    def choose_action(s, frames, lf):
        try:
            lvl = s._lvl(lf)

            # ===== LEVEL CHANGE =====
            if lvl != s.cl:
                # Init BFS solver on first level
                if not s._bfs_tried:
                    s._bfs_tried = True
                    s._init_bfs()

                # Try BFS for this level
                s._bfs_solution = None
                s._bfs_step = 0
                s._bfs_verified = False  # v20: re-verify per level
                if s._bfs:
                    s._try_bfs_solve(lvl)

                # Init CNN fallback
                s.buf.clear(); s.buf_h.clear()
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
                s.opt = optim.Adam(s.net.parameters(), lr=0.0003)
                s.pt=None;s.pai=None;s.pr=None;s.ph=None
                s.cl=lvl;s.fhist.clear();s.la=0
                s._wd=False;s._wm=None
                s._aem_diffs.clear();s._aem_actions.clear();s._aem_rewards.clear()
                s._prev_objs=None;s._obj_moved=0;s._ckpt_hash=None;s._unproductive=0
                # FIX 1: Reset visited hashes on every level change
                s._visited_hashes = set()
                # Reset object model
                s._frame_buffer = []
                s._static_mask = None
                s._dynamic_mask = None
                s._static_ready = False
                s._structural_colours = set()
                s._target_colours = set()
                s._goal_groups = []
                # FIX 4: Only reset epsilon if BFS didn't solve this level.
                # If BFS solved it, keep current eps so CNN fallback (if needed)
                # benefits from accumulated exploration knowledge.
                if not s._bfs_solution:
                    s._eps = 0.15

                # CLTI — inject BFS demos from previous level into CNN replay buffer
                # FIX 2: Use perform_action frame[-1] consistently with _raw(),
                # instead of get_pixels() which returns a different format.
                if lvl > 0 and s._bfs and s._bfs.solutions.get(lvl - 1):
                    prev_sol = s._bfs.solutions[lvl - 1]
                    try:
                        replay_game = s._bfs.game_cls()
                        replay_game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                        r0 = replay_game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                        if r0.frame:
                            # Start from the post-reset frame, consistent with _raw()
                            prev_frame = np.array(r0.frame[-1], dtype=np.int64)
                            for act_id, data in prev_sol:
                                ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                                result = replay_game.perform_action(ai, raw=True)
                                action_idx = (act_id - 1) if act_id <= 5 else (
                                    5 + data.get('y', 0) * 64 + data.get('x', 0) if data else 0)
                                s.buf.append({'s': prev_frame.copy(), 'a': action_idx, 'r': 2.0})
                                # Advance prev_frame using the action result, not get_pixels()
                                if result.frame:
                                    prev_frame = np.array(result.frame[-1], dtype=np.int64)
                            if len(s.buf) >= s.bsz:
                                for _ in range(min(20, len(s.buf) // s.bsz)):
                                    s._train()
                                logger.info(f"CLTI: injected {len(prev_sol)} expert demos from L{lvl-1}")
                    except Exception as e:
                        logger.warning(f"CLTI failed: {e}")

            # ===== RESET =====
            if lf.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
                s.pt=None;s.pai=None;s.pr=None;s.ph=None
                return GameAction.RESET

            # ===== BFS SOLUTION EXECUTION =====
            if s._bfs_solution and s._bfs_step < len(s._bfs_solution):
                # v20: First-step verification gate. Before consuming step 1 (i.e.
                # right after step 0 was sent on the previous turn), check that the
                # live frame matches what BFS predicted. If they diverge, the BFS
                # plan is for a DIFFERENT game state than the live host, so abandon
                # it and fall through to the CNN/WorldModel path.
                if s._bfs_step == 1 and not s._bfs_verified:
                    raw = s._raw(lf)
                    sim_frame = s._bfs_predict_frame(s.cl, 1)
                    if sim_frame is None or sim_frame.shape != raw.shape or not np.array_equal(raw, sim_frame):
                        logger.warning(f"BFS sync FAIL @ L{s.cl}: live!=sim after step 0; abandoning BFS path (will fall through to CNN)")
                        s._bfs_solution = None
                        s._bfs_step = 0
                        # Fall through — CNN block below picks the action this turn
                    else:
                        s._bfs_verified = True
                        logger.info(f"BFS sync OK @ L{s.cl}: continuing BFS plan ({len(s._bfs_solution)} actions total)")

            if s._bfs_solution and s._bfs_step < len(s._bfs_solution):
                act_id, data = s._bfs_solution[s._bfs_step]
                s._bfs_step += 1
                sel = GameAction.from_id(act_id)
                if data:
                    payload = {k: v for k, v in data.items() if k != 'game_id'}
                    if payload:
                        sel.set_data(payload)
                    s._last_action_data = payload
                else:
                    s._last_action_data = None
                raw = s._raw(lf)
                s.fhist.append(raw.copy())
                s.pr = raw.copy()
                s.la += 1
                return sel

                        # ===== GRAPH-EXPLORER FALLBACK (v25 hybrid) =====
            # Replaces CNN/WorldModel fallback path. Used when BFS has no
            # plan for current level. Graph-explorer maintains state-graph,
            # picks frontier-aware actions across 5 priority tiers.
            try:
                ge_action = s._ge_pick(lf, lvl)
                if ge_action is not None:
                    raw_now = s._raw(lf)
                    s.pr = raw_now.copy()
                    s.la += 1
                    return ge_action
            except Exception as _ge_e:
                logger.warning(f"GE fallback failed: {_ge_e}; using safe default")

            # Safe default: random valid arrow if available, else click center
            avail = list(getattr(lf, 'available_actions', []) or [])
            arrows = [a for a in avail if (a.value if hasattr(a, 'value') else int(a)) in (1, 2, 3, 4, 5)]
            if arrows:
                return random.choice(arrows) if hasattr(random.choice(arrows), 'value') else GameAction.from_id(int(random.choice(arrows)))
            sel = GameAction.ACTION6
            sel.set_data({"x": 32, "y": 32})
            return sel


        except Exception as e:
            traceback.print_exc()
            a=random.choice(s.al);a.reasoning=f"err:{e}";return a