"""
Local RHAE evaluator for FORGE agents.
Runs the agent against ARC-AGI-3 games locally using the arc-agi SDK.

Usage (must use C:/Python313/python.exe which has arc-agi installed):
    C:/Python313/python.exe local_rhae_eval.py
    C:/Python313/python.exe local_rhae_eval.py --agent forge_v22_bfs --games 10 --budget 60
    C:/Python313/python.exe local_rhae_eval.py --games 25 --budget 120
"""

import argparse
import importlib.util
import logging
import os
import random
import sys
import time
import traceback
import types
from collections import deque
from pathlib import Path

import numpy as np

# Silence noisy loggers
logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(name)s: %(message)s')
for name in ['arc_agi', 'arcengine', 'urllib3', 'httpx', 'httpcore']:
    logging.getLogger(name).setLevel(logging.ERROR)

# Patch StreamHandler to not write to stdout (protects any MCP transport)
import logging as _logging, sys as _sys
_orig_sh_init = _logging.StreamHandler.__init__
def _safe_sh_init(self, stream=None):
    if stream is _sys.stdout:
        stream = _sys.stderr
    _orig_sh_init(self, stream)
_logging.StreamHandler.__init__ = _safe_sh_init

import arc_agi
from arcengine.enums import GameAction as GA, GameState as GS
from arcengine import ActionInput

# ── Mock agent SDK (so forge_vXX.py can be imported without Kaggle runtime) ───

class _FakeAgent:
    MAX_ACTIONS = float('inf')
    _MAX_FRAMES = 10
    game_id: str = ''
    arc_env = None
    action_counter: int = 0
    frames: list = []
    is_playback: bool = False
    guid: str = ''
    recorder = None

_agents_mod = types.ModuleType('agents')
_agent_mod = types.ModuleType('agents.agent')
_agent_mod.Agent = _FakeAgent
_agents_mod.agent = _agent_mod
sys.modules.setdefault('agents', _agents_mod)
sys.modules.setdefault('agents.agent', _agent_mod)
# NOTE: do NOT mock 'arcengine' — the real module has all enums and arc_agi needs it

# ── RHAE ──────────────────────────────────────────────────────────────────────

def compute_rhae(level_actions: dict, baseline: list) -> float:
    if not level_actions:
        return 0.0
    n = len(baseline)
    total_w = n * (n + 1) / 2
    score = 0.0
    for l in range(n):
        if l in level_actions:
            score += (l + 1) * min(1.0, baseline[l] / max(level_actions[l], 1)) ** 2
    return score / total_w


# ── Fake arc_env with local_dir for BFS source discovery ──────────────────────

def make_arc_env(game_id, env_files_dir, baseline_actions=None):
    gid = game_id.split('-')[0]
    local_dir = None
    if env_files_dir:
        p = Path(env_files_dir)
        candidates = list(p.glob(f"{gid}/**/{gid}.py"))
        if candidates:
            local_dir = str(candidates[0].parent)

    class _EnvInfo:
        pass
    ei = _EnvInfo()
    ei.local_dir = local_dir
    ei.game_id = game_id
    ei.baseline_actions = baseline_actions or []

    class _ArcEnv:
        environment_info = ei
    return _ArcEnv()


# ── Main game loop ─────────────────────────────────────────────────────────────

def run_game(AgentCls, env_info, arcade, time_budget, max_actions, env_files_dir):
    """Run one game with the agent, return RHAE + stats."""

    # Instantiate agent with game_id set before __init__
    agent = object.__new__(AgentCls)
    agent.game_id = env_info.game_id
    agent.arc_env = make_arc_env(env_info.game_id, env_files_dir, baseline_actions=env_info.baseline_actions)
    agent.action_counter = 0
    agent.frames = []
    agent.is_playback = False
    agent.guid = ''
    agent.recorder = None
    try:
        AgentCls.__init__(agent)
    except Exception as e:
        return {"rhae": 0.0, "levels": 0, "actions": 0, "error": f"init: {e}"}

    # Override start_time so BFS/MCTS get 80% of our local time_budget.
    # Agent computes: remaining = 8*3600-600 - elapsed
    # BFS L0 gets: min(remaining * 0.3, 600) → we want remaining ≈ time_budget
    # So: fake_elapsed = 8*3600-600 - time_budget
    # But then BFS L0 = min(time_budget*0.3, 600) which is too short for slow games.
    # Instead: make remaining = time_budget * 3 so BFS L0 gets min(budget*0.9, 600)
    fake_remaining = time_budget * 3
    fake_elapsed = max(0, 8 * 3600 - 600 - fake_remaining)
    if hasattr(agent, 'start_time'):
        agent.start_time = time.time() - fake_elapsed

    env = arcade.make(env_info.game_id)
    frame = env.reset()
    if frame is None:
        return {"rhae": 0.0, "levels": 0, "actions": 0, "level_actions": {}, "baseline": env_info.baseline_actions, "elapsed": 0, "errors": 1, "error": "env.reset() returned None"}

    total_actions = 0
    levels_done = 0
    level_actions = {}
    lvl_start = 0
    t0 = time.time()
    errors = 0
    consecutive_errors = 0

    while time.time() - t0 < time_budget and total_actions < max_actions:
        # Win check
        if frame.state is GS.WIN:
            break

        # is_done check
        try:
            if agent.is_done(agent.frames, frame):
                break
        except:
            pass

        # Level transition tracking
        if frame.levels_completed > levels_done:
            level_actions[levels_done] = total_actions - lvl_start
            levels_done = frame.levels_completed
            lvl_start = total_actions

        # append_frame
        try:
            agent.append_frame(frame)
        except:
            if len(agent.frames) > agent._MAX_FRAMES:
                agent.frames = agent.frames[-agent._MAX_FRAMES:]
            else:
                agent.frames.append(frame)

        # choose_action
        try:
            action = agent.choose_action(agent.frames, frame)
            consecutive_errors = 0
        except Exception as e:
            errors += 1
            consecutive_errors += 1
            if consecutive_errors > 10:
                break
            action = GA.RESET

        if action is None:
            action = GA.RESET

        # Extract click data from action (agent calls sel.set_data(data) → action.action_data)
        data = None
        try:
            if hasattr(action, 'action_data') and action.action_data:
                ad = action.action_data
                data = {'x': ad.x, 'y': ad.y}
                if hasattr(ad, 'game_id') and ad.game_id:
                    data['game_id'] = ad.game_id
        except:
            pass

        agent.action_counter += 1
        frame = env.step(action, data=data)
        total_actions += 1

    elapsed = time.time() - t0
    rhae = compute_rhae(level_actions, env_info.baseline_actions)
    return {
        "rhae": rhae,
        "levels": levels_done,
        "actions": total_actions,
        "elapsed": elapsed,
        "level_actions": level_actions,
        "baseline": env_info.baseline_actions,
        "errors": errors,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', default='forge_v24_mcts',
                        help='Agent file stem under notebooks/forge_agent/')
    parser.add_argument('--games', type=int, default=25)
    parser.add_argument('--budget', type=int, default=120,
                        help='Seconds per game')
    parser.add_argument('--max-actions', type=int, default=5000)
    parser.add_argument('--env-dir', default='environment_files')
    args = parser.parse_args()

    # Load agent module
    for candidate in [
        Path(f'notebooks/forge_agent/{args.agent}.py'),
        Path(args.agent + '.py'),
        Path(args.agent),
    ]:
        if candidate.exists():
            agent_path = candidate
            break
    else:
        print(f"ERROR: cannot find agent '{args.agent}'"); sys.exit(1)

    print(f"\n{'='*65}")
    print(f"Agent : {agent_path}")
    spec = importlib.util.spec_from_file_location('forge_agent', str(agent_path))
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"ERROR loading agent: {e}"); traceback.print_exc(); sys.exit(1)
    AgentCls = mod.MyAgent
    print(f"Budget: {args.budget}s/game  |  Max actions: {args.max_actions}")

    # Load games sorted easiest → hardest
    print("Connecting to ARC-AGI-3 API...")
    arcade = arc_agi.Arcade()
    envs = sorted(arcade.get_environments(), key=lambda e: sum(e.baseline_actions))
    envs = envs[:args.games]

    env_dir = args.env_dir if os.path.exists(args.env_dir) else None
    if env_dir:
        print(f"BFS src: {env_dir}")
    else:
        print("BFS src: NOT FOUND (BFS will be disabled)")

    print(f"\nRunning {len(envs)} games...\n")
    print(f"{'#':<3} {'Game':<22} {'Lvls':>5} {'RHAE':>6} {'Actions':>8} {'Time':>6}  Level breakdown")
    print('-' * 75)

    total_rhae = 0.0
    results = []

    for i, env_info in enumerate(envs):
        sys.stdout.write(f"{i+1:<3} {env_info.game_id:<22}  running...")
        sys.stdout.flush()

        result = run_game(AgentCls, env_info, arcade,
                          time_budget=args.budget,
                          max_actions=args.max_actions,
                          env_files_dir=env_dir)
        results.append(result)
        total_rhae += result['rhae']

        # Level breakdown: show agent/human per level
        la = result['level_actions']
        base = result['baseline']
        breakdown = ' '.join(
            f"L{l}:{la[l]}/{base[l] if l < len(base) else '?'}"
            for l in sorted(la.keys())
        ) if la else '–'

        sys.stdout.write(
            f"\r{i+1:<3} {env_info.game_id:<22} {result['levels']:>5} "
            f"{result['rhae']:>6.4f} {result['actions']:>8} "
            f"{result['elapsed']:>5.0f}s  {breakdown}\n"
        )
        sys.stdout.flush()

    # Summary
    print('-' * 75)
    n = len(results)
    games_scored = sum(1 for r in results if r['rhae'] > 0)
    l0_solved   = sum(1 for r in results if r['levels'] >= 1)
    l1_solved   = sum(1 for r in results if r['levels'] >= 2)
    l2_solved   = sum(1 for r in results if r['levels'] >= 3)

    print(f"\n{'Total RHAE (sum):':<28} {total_rhae:.4f}")
    print(f"{'Mean RHAE / game:':<28} {total_rhae/n:.4f}")
    print(f"{'Games RHAE > 0:':<28} {games_scored}/{n}")
    print(f"{'L0 solved (≥1 level):':<28} {l0_solved}/{n}")
    print(f"{'L1 solved (≥2 levels):':<28} {l1_solved}/{n}")
    print(f"{'L2 solved (≥3 levels):':<28} {l2_solved}/{n}")

    # Kaggle LB estimate
    # Competition: RHAE mean over 25 games (or sum/normalizer).
    # Our local run likely underestimates since local timeout < Kaggle 8h budget.
    if n == 25:
        lb_est = total_rhae / 25
    else:
        lb_est = (total_rhae / n) * (n / 25)
    print(f"\n{'Kaggle LB estimate:':<28} ~{lb_est:.4f}  "
          f"({'full 25 games' if n==25 else f'extrapolated from {n}/25'})")
    print(f"  Note: local budget={args.budget}s << Kaggle 8h, BFS/MCTS may score higher on Kaggle\n")


if __name__ == '__main__':
    main()
