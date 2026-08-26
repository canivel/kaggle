"""Deep local validation of forge_v35 MyAgent.

Runs MyAgent with the REAL ARC-AGI-3 agent framework constructor,
same execution path as Kaggle competition rerun.

Usage:
    cd f:/kaggle/arc-prize-2026
    PYTHONIOENCODING=utf-8 uv run --project f:/kaggle python test_forge_v35.py
    PYTHONIOENCODING=utf-8 uv run --project f:/kaggle python test_forge_v35.py --games ft09,lp85,r11l
    PYTHONIOENCODING=utf-8 uv run --project f:/kaggle python test_forge_v35.py --time 120 --all
"""

import argparse
import importlib.util
import json
import logging
import os
import sys
import time
import traceback
import types
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s | %(levelname)-7s | %(message)s')
# Show forge agent logs
logging.getLogger('forge_v35').setLevel(logging.INFO)

# ── Point to real ARC-AGI-3-Agents so imports resolve ─────────────────────
AGENTS_DIR = Path(__file__).parent / 'kaggle-data/ARC-AGI-3-Agents'
sys.path.insert(0, str(AGENTS_DIR))

# Stub agents package to avoid pulling in langgraph/langsmith deps
# We only need agents.agent.Agent — bypass the __init__ that imports templates
import types as _types
_agents_pkg = _types.ModuleType('agents')
_agents_pkg.__path__ = [str(AGENTS_DIR / 'agents')]
_agents_pkg.__package__ = 'agents'
sys.modules['agents'] = _agents_pkg
# Now import agents.agent directly via importlib
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location('agents.agent', str(AGENTS_DIR / 'agents/agent.py'))
_agent_mod = _ilu.module_from_spec(_spec)
_agent_mod.__package__ = 'agents'
sys.modules['agents.agent'] = _agent_mod
_spec.loader.exec_module(_agent_mod)
_agents_pkg.agent = _agent_mod

# ── ARC SDK ────────────────────────────────────────────────────────────────
import arc_agi
from arcengine.enums import GameAction as GA, GameState as GS, ActionInput
from arcengine       import FrameData

# ── Load forge_v35 ────────────────────────────────────────────────────────
AGENT_PATH = Path(__file__).parent / 'notebooks/forge_agent/forge_v35_tips.py'

def load_agent_module():
    spec = importlib.util.spec_from_file_location('forge_v35', str(AGENT_PATH))
    mod  = importlib.util.module_from_spec(spec)
    mod.FrameData   = FrameData
    mod.ActionInput = ActionInput
    mod.GameAction  = GA
    mod.GameState   = GS
    spec.loader.exec_module(mod)
    return mod

# ── RHAE ───────────────────────────────────────────────────────────────────
def compute_rhae(level_actions, baseline_actions):
    if not level_actions:
        return 0.0
    n = len(baseline_actions)
    total_w = n * (n + 1) / 2
    score = 0.0
    for l in range(n):
        w = l + 1
        if l in level_actions:
            h = baseline_actions[l]
            a = level_actions[l]
            score += w * min(1.0, h / max(a, 1)) ** 2
    return score / total_w

# ── Run one game using real Agent constructor ──────────────────────────────
def run_game(env_info, MyAgent, time_budget=120, verbose=True):
    arcade   = arc_agi.Arcade()
    arc_env  = arcade.make(env_info.game_id)

    # Real constructor: card_id, game_id, agent_name, ROOT_URL, record, arc_env
    agent = MyAgent(
        card_id    = 'local-test',
        game_id    = env_info.game_id,
        agent_name = 'myagent',
        ROOT_URL   = 'http://localhost:8001',
        record     = False,
        arc_env    = arc_env,
    )

    frame = arc_env.reset()

    level_actions    = {}
    levels_completed = 0
    current_level_start = 0
    total_actions    = 0
    t0 = time.time()

    while time.time() - t0 < time_budget:
        state = getattr(frame, 'state', GS.NOT_FINISHED)

        if state in (GS.NOT_PLAYED, GS.GAME_OVER):
            frame = arc_env.step(GA.RESET)
            continue

        if state == GS.WIN:
            break

        # Level change detection
        lc = getattr(frame, 'levels_completed', 0)
        if lc > levels_completed:
            actions_this = total_actions - current_level_start
            level_actions[levels_completed] = actions_this
            if verbose:
                h = env_info.baseline_actions[levels_completed] \
                    if levels_completed < len(env_info.baseline_actions) else '?'
                eff = f"{h/max(actions_this,1):.2f}x" if isinstance(h, int) else '?'
                print(f"    L{levels_completed}: {actions_this} acts "
                      f"(human={h}, eff={eff}) t={time.time()-t0:.1f}s")
            levels_completed = lc
            current_level_start = total_actions

        # Agent step — same as ARC-AGI-3-Agents framework
        try:
            agent.append_frame(frame)
            action_input = agent.choose_action(agent.frames, frame)
        except Exception as e:
            print(f"  choose_action error: {e}")
            traceback.print_exc()
            break

        if action_input is None:
            action_input = ActionInput(id=GA.RESET)

        # Execute — mirrors real framework's do_action_request():
        #   data = action.action_data.model_dump(); arc_env.step(action, data=data)
        # choose_action returns GameAction (not ActionInput) in all paths
        try:
            if hasattr(action_input, 'id') and isinstance(action_input.id, GA):
                act_id = action_input.id
                act_data = getattr(action_input, 'data', None)
            else:
                act_id = action_input  # it's already a GameAction
                # Real framework: data = action.action_data.model_dump()
                ad = getattr(act_id, 'action_data', None)
                act_data = ad.model_dump() if (ad is not None and hasattr(ad, 'model_dump')) else None
            frame = arc_env.step(act_id, data=act_data)
        except Exception as e:
            print(f"  env.step error: {e}")
            break

        total_actions += 1
        if total_actions > 8000:
            print("  Hit 8000 action cap")
            break

    elapsed = time.time() - t0
    rhae = compute_rhae(level_actions, env_info.baseline_actions)

    return {
        "game_id":           env_info.game_id,
        "title":             env_info.title or env_info.game_id.split('-')[0],
        "levels_completed":  levels_completed,
        "win_levels":        len(env_info.baseline_actions),
        "level_actions":     level_actions,
        "total_actions":     total_actions,
        "rhae":              round(rhae, 6),
        "elapsed":           round(elapsed, 1),
        "baseline":          env_info.baseline_actions,
    }

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', default='',
        help='Comma-separated game IDs, or integer count')
    parser.add_argument('--time', type=int, default=90,
        help='Seconds per game (default 90)')
    parser.add_argument('--all', action='store_true',
        help='Run all 25 games')
    args = parser.parse_args()

    # ── TEST 1: Import ─────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("TEST 1: Import forge_v35")
    print("="*70)
    try:
        mod     = load_agent_module()
        MyAgent = mod.MyAgent
        print(f"  OK  Import — MyAgent loaded")
        print(f"  OK  _TIPS_LOCK present: {mod._TIPS_LOCK}")
        print(f"  OK  _TIPS_MODEL=None (not loaded yet): {mod._TIPS_MODEL}")
        print(f"  OK  _TIPS_FAILED=False: {mod._TIPS_FAILED}")
    except Exception as e:
        print(f"  FAIL Import: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ── Select games ───────────────────────────────────────────────────────
    arcade   = arc_agi.Arcade()
    all_envs = sorted(arcade.get_environments(), key=lambda e: sum(e.baseline_actions))

    if args.all:
        envs = all_envs
    elif args.games:
        try:
            n    = int(args.games)
            envs = all_envs[:n]
        except ValueError:
            wanted = {g.strip().lower() for g in args.games.split(',')}
            envs   = [e for e in all_envs
                      if e.game_id.split('-')[0].lower() in wanted
                      or (e.title or '').lower() in wanted]
            if not envs:
                print(f"No games matched: {wanted}")
                print(f"Available: {[e.game_id.split('-')[0] for e in all_envs]}")
                sys.exit(1)
    else:
        # Default: 8 games — REPEAT_CLICK first, then a few directional
        repeat = {'ft09','lp85','r11l','s5i5','sb26','su15','tn36'}
        envs   = sorted(all_envs,
                        key=lambda e: (0 if e.game_id.split('-')[0] in repeat else 1,
                                       sum(e.baseline_actions)))[:10]

    print(f"\n  Games: {len(envs)} x {args.time}s budget")
    print(f"  File:  {AGENT_PATH.name}")

    # ── TEST 2: Per-game runs ──────────────────────────────────────────────
    print("\n" + "="*70)
    print("TEST 2: Per-game agent runs")
    print("="*70)

    results = []
    for i, env_info in enumerate(envs):
        gid   = env_info.game_id.split('-')[0].upper()
        base  = sum(env_info.baseline_actions)
        nlevs = len(env_info.baseline_actions)
        print(f"\n[{i+1:2d}/{len(envs)}] {gid:6s}  ({nlevs}L, human={base} acts)")

        # Reload module each game (mirrors Kaggle: fresh process per game)
        try:
            mod     = load_agent_module()
            MyAgent = mod.MyAgent
        except Exception as e:
            print(f"  FAIL module reload: {e}")
            results.append({"game_id": env_info.game_id, "rhae": 0,
                             "levels_completed": 0, "error": str(e)})
            continue

        try:
            r = run_game(env_info, MyAgent, time_budget=args.time)
            results.append(r)
            lc  = r['levels_completed']
            wl  = r['win_levels']
            ok  = "OK  " if lc > 0 else "    "
            print(f"  {ok}L{lc}/{wl}  RHAE={r['rhae']:.4f}  "
                  f"acts={r['total_actions']}  t={r['elapsed']:.1f}s")
        except Exception as e:
            print(f"  FAIL run: {e}")
            traceback.print_exc()
            results.append({"game_id": env_info.game_id, "rhae": 0,
                             "levels_completed": 0, "error": str(e)})

    # ── TEST 3: Deadlock check ─────────────────────────────────────────────
    print("\n" + "="*70)
    print("TEST 3: Threading — 10 concurrent MyAgent inits (deadlock check)")
    print("="*70)
    import threading as _th

    mod2    = load_agent_module()
    MyAgent2 = mod2.MyAgent

    errors  = []
    agents  = []
    arcade2 = arc_agi.Arcade()

    def init_agent(game_id):
        try:
            arc_env = arcade2.make(game_id)
            a = MyAgent2(
                card_id='t', game_id=game_id, agent_name='myagent',
                ROOT_URL='http://localhost:8001', record=False, arc_env=arc_env,
            )
            agents.append(a)
        except Exception as e:
            errors.append(f"{game_id}: {e}")

    gids    = [e.game_id for e in all_envs[:10]]
    threads = [_th.Thread(target=init_agent, args=(g,)) for g in gids]
    t0      = time.time()
    for t in threads: t.start()
    for t in threads: t.join(timeout=30)
    elapsed = time.time() - t0

    alive = sum(1 for t in threads if t.is_alive())
    if alive:
        print(f"  FAIL DEADLOCK — {alive} threads still running after 30s")
    elif errors:
        print(f"  WARN errors: {errors[:3]}")
    else:
        print(f"  OK   10 concurrent inits in {elapsed:.1f}s, {len(agents)} agents created")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    rhaes     = [r.get('rhae', 0) for r in results]
    mean_rhae = np.mean(rhaes) if rhaes else 0
    solved    = sum(1 for r in results if r.get('levels_completed', 0) > 0)

    print(f"  Mean RHAE:      {mean_rhae:.6f}")
    print(f"  Games solved:   {solved}/{len(results)}")
    print(f"  Total levels:   {sum(r.get('levels_completed',0) for r in results)}"
          f" / {sum(r.get('win_levels',0) for r in results)}")
    print()

    for r in sorted(results, key=lambda x: -x.get('rhae', 0)):
        if r.get('rhae', 0) > 0:
            gid  = r['game_id'].split('-')[0].upper()
            la   = r.get('level_actions', {})
            base = r.get('baseline', [])
            lvls = ', '.join(
                f"L{l}:{la[l]}(h={base[l] if l<len(base) else '?'})"
                for l in sorted(la)
            )
            print(f"  {gid:6s} RHAE={r['rhae']:.4f}  {lvls}")

    out = Path('data/test_forge_v35.json')
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        json.dump({'mean_rhae': mean_rhae, 'results': results}, f,
                  indent=2, default=str)
    print(f"\n  Results saved to {out}")
    print("="*70)


if __name__ == '__main__':
    main()
