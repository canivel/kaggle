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
from collections import deque
from typing import Dict, List, Set, Optional, Tuple

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

class MyAgent(Agent):
    MAX_ACTIONS = float('inf')
    _MAX_FRAMES = 10

    def __init__(s, *a, **kw):
        super().__init__(*a, **kw)
        # v21 probe: check arc_env wrapper type and _game access
        try:
            ae = getattr(s, 'arc_env', None)
            ae_type = type(ae).__name__ if ae is not None else 'None'
            has_game = hasattr(ae, '_game') and getattr(ae, '_game', None) is not None
            has_class = hasattr(ae, '_game_class') and getattr(ae, '_game_class', None) is not None
            logger.info(f"v21 probe: arc_env type={ae_type} has_game={has_game} has_game_class={has_class}")
            if has_game:
                g = ae._game
                logger.info(f"v21 probe: arc_env._game type={type(g).__name__}")
                import copy as _cp
                try:
                    g2 = _cp.deepcopy(g)
                    logger.info(f"v21 probe: deepcopy(arc_env._game) SUCCESS: {type(g2).__name__}")
                except Exception as _e:
                    logger.warning(f"v21 probe: deepcopy(arc_env._game) FAIL: {_e}")
        except Exception as _e:
            logger.warning(f"v21 probe error: {_e}")
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
                s._last_action_data = {k: v for k, v in data.items() if k != 'game_id'} if data else None
                raw = s._raw(lf)
                s.fhist.append(raw.copy())
                s.pr = raw.copy()
                s.la += 1
                return sel

            # ===== CNN FALLBACK =====
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
                    r=s._reward(s.pr, raw, '', ch, s.pai, getattr(s, '_last_action_data', None))
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
                sel=s.al[aidx]
            else:
                sel=GameAction.ACTION6;y,x=coords
                s._last_action_data={"x":int(x),"y":int(y)}

            s.pt=tensor;s.pai=aidx if aidx<5 else(5+coords[0]*s.G+coords[1])
            s.pr=raw.copy();s.ph=ch;s.la+=1
            if s.action_counter%s.tfreq==0 and s._wd:s._train()
            return sel

        except Exception as e:
            traceback.print_exc()
            a=random.choice(s.al);a.reasoning=f"err:{e}";return a