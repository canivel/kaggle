"""
Local simulator test for FORGE v29 using arc-agi-3-benchmarking framework.

Uses OFFLINE mode (local environment files) to evaluate our agent against
the real game environments with proper RHAE scoring.

Usage:
  # Test specific games (quick check):
  python local_sim_test.py --games ls20,r11l,cn04 --time-limit 120

  # Full eval (all 25 games, ~30 min):
  python local_sim_test.py --full --time-limit 300

  # Single game debug:
  python local_sim_test.py --games r11l --time-limit 60 --verbose
"""

import argparse
import importlib.util
import logging
import math
import os
import sys
import time
from typing import List, Optional

# Add benchmarking repo to path
sys.path.insert(0, 'f:/Projects/arc-agi-3-benchmarking')

from dotenv import load_dotenv
load_dotenv('f:/Projects/arc-agi-3-benchmarking/.env')

from arc_agi import Arcade, OperationMode
from arcengine import FrameData, GameAction, GameState

# Use benchmarking base class
from benchmarking.base import Agent

ENV_DIR = 'f:/kaggle/arc-prize-2026/environment_files'
AGENT_FILE = 'f:/kaggle/arc-prize-2026/notebooks/forge_agent/forge_v29_compete.py'

ALL_GAMES = [
    'ar25', 'bp35', 'cd82', 'cn04', 'dc22', 'ft09', 'g50t', 'ka59',
    'lf52', 'lp85', 'ls20', 'm0r0', 'r11l', 're86', 's5i5', 'sb26',
    'sc25', 'sk48', 'sp80', 'su15', 'tn36', 'tr87', 'tu93', 'vc33', 'wa30'
]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('f:/kaggle/arc-prize-2026/data/local_sim_v29.log', mode='w'),
    ]
)
logger = logging.getLogger('sim')


def load_forge_class(agent_file: str):
    """Dynamically load MyAgent from the forge file."""
    spec = importlib.util.spec_from_file_location('forge_agent', agent_file)
    mod = importlib.util.module_from_spec(spec)

    # Mock the agents.agent import that Kaggle provides
    import unittest.mock as mock
    import sys as _sys

    fake_agent_mod = mock.MagicMock()

    class _FakeAgent:
        def __init__(self, *a, **kw):
            self.game_id = kw.get('game_id', '')
            self.arc_env = kw.get('arc_env', None)
            self.frames = []
            self.action_counter = 0
        def append_frame(self, f): self.frames.append(f)
        def is_done(self, frames, lf): return False
        def choose_action(self, frames, lf): return GameAction.ACTION1

    fake_agent_mod.Agent = _FakeAgent
    _sys.modules['agents'] = fake_agent_mod
    _sys.modules['agents.agent'] = fake_agent_mod

    spec.loader.exec_module(mod)
    return mod.MyAgent


class ForgeSimAgent(Agent):
    """Wraps FORGE MyAgent into the benchmarking Agent base class."""

    MAX_ACTIONS = 50000  # high limit — FORGE manages its own budget via time

    def __init__(self, forge_class, time_limit: float, *args, **kw):
        super().__init__(*args, **kw)
        self._forge = forge_class(
            card_id=self.card_id,
            game_id=self.game_id,
            agent_name='forge_v29',
            ROOT_URL='',
            record=False,
            arc_env=self.arc_env,
            config=None,
        )
        self._forge.arc_env = self.arc_env
        self._time_limit = time_limit
        self._start = time.time()
        # Override the forge agent's per-game budget so BFS/MCTS respect our time limit.
        # The module is loaded dynamically as 'forge_agent' in sys.modules.
        import sys
        _fa = sys.modules.get('forge_agent')
        if _fa:
            _fa._TOTAL_GAMES = 1
            _fa._TOTAL_BUDGET = time_limit
        self._forge.start_time = self._start
        self._forge._game_start_time = self._start
        self._forge._games_completed = 0

    def is_done(self, frames: list, latest_frame: FrameData) -> bool:
        if time.time() - self._start >= self._time_limit:
            logger.info(f'{self.game_id}: time limit reached ({self._time_limit}s)')
            return True
        return self._forge.is_done(frames, latest_frame)

    def choose_action(self, frames: list, latest_frame: FrameData) -> GameAction:
        return self._forge.choose_action(frames, latest_frame)

    def append_frame(self, frame: FrameData) -> None:
        super().append_frame(frame)
        self._forge.append_frame(frame)


def rhae_score(actions_taken: int, baseline: int) -> float:
    """Competition scoring: (baseline/actions)^2, capped at 1.0"""
    if actions_taken <= 0:
        return 0.0
    return min(1.0, (baseline / actions_taken) ** 2)


def run_game(game_id: str, forge_class, time_limit: float, verbose: bool = False) -> dict:
    """Run a single game and return results."""
    logger.info(f'\n{"="*60}')
    logger.info(f'Starting game: {game_id} (limit={time_limit}s)')

    arcade = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=ENV_DIR,
        arc_api_key=os.getenv('ARC_API_KEY', ''),
    )
    env = arcade.make(game_id)
    if env is None:
        logger.error(f'{game_id}: failed to create environment')
        return {'game_id': game_id, 'error': 'env_failed', 'rhae': 0.0}

    ei = env.environment_info
    baseline = list(ei.baseline_actions) if ei.baseline_actions else []
    logger.info(f'{game_id}: baseline_actions={baseline}')

    agent = ForgeSimAgent(
        forge_class=forge_class,
        time_limit=time_limit,
        card_id='local-test',
        game_id=game_id,
        agent_name='forge_v29',
        ROOT_URL='',
        record=False,
        arc_env=env,
        config=None,
    )

    t0 = time.time()
    try:
        agent.main()
    except Exception as e:
        logger.error(f'{game_id}: agent crashed: {e}', exc_info=True)

    elapsed = time.time() - t0
    levels_completed = agent.levels_completed
    total_actions = agent.action_counter

    # Compute per-level RHAE
    # We don't have per-level action breakdown directly, so use total actions
    # against first level baseline as approximation for L0
    level_scores = []
    if baseline:
        for i, b in enumerate(baseline):
            if i < levels_completed:
                # Estimate: actions are roughly even across levels (conservative)
                est_actions = max(1, total_actions // max(1, levels_completed))
                score = rhae_score(est_actions, b)
                level_scores.append(score)
                if verbose:
                    logger.info(f'  L{i}: baseline={b}, est_actions={est_actions}, score={score:.3f}')

    # Simpler: if we won L0, score at least (baseline[0]/total_actions)^2
    if levels_completed > 0 and baseline:
        l0_score = rhae_score(total_actions, baseline[0])
        approx_rhae = sum((i+1) * s for i, s in enumerate(level_scores)) / max(1, sum(range(1, len(level_scores)+1)))
    else:
        l0_score = 0.0
        approx_rhae = 0.0

    result = {
        'game_id': game_id,
        'levels_completed': levels_completed,
        'total_actions': total_actions,
        'baseline': baseline,
        'l0_score_pct': l0_score * 100,
        'approx_rhae': approx_rhae,
        'elapsed': elapsed,
    }

    logger.info(f'{game_id}: DONE | levels={levels_completed} | actions={total_actions} | '
                f'baseline_L0={baseline[0] if baseline else "?"} | '
                f'l0_score={l0_score*100:.1f}% | elapsed={elapsed:.0f}s')

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=str, default=None, help='Comma-separated game IDs')
    parser.add_argument('--full', action='store_true', help='Run all 25 games')
    parser.add_argument('--time-limit', type=float, default=180, help='Seconds per game')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--agent', type=str, default=AGENT_FILE, help='Path to agent .py file')
    args = parser.parse_args()

    games = ALL_GAMES if args.full else (args.games.split(',') if args.games else ALL_GAMES[:6])

    logger.info(f'Loading agent from {args.agent}')
    try:
        forge_class = load_forge_class(args.agent)
        logger.info(f'Loaded: {forge_class}')
    except Exception as e:
        logger.error(f'Failed to load agent: {e}', exc_info=True)
        return

    results = []
    total_t0 = time.time()

    for game_id in games:
        result = run_game(game_id, forge_class, args.time_limit, args.verbose)
        results.append(result)

    total_elapsed = time.time() - total_t0

    # Summary
    print('\n' + '='*70)
    print(f'FORGE v29 LOCAL SIM RESULTS ({len(games)} games, {total_elapsed:.0f}s total)')
    print('='*70)
    print(f'{"Game":<8} {"Levels":>7} {"Actions":>9} {"BaseL0":>8} {"L0 Score":>10} {"Time":>7}')
    print('-'*70)
    total_l0 = 0.0
    for r in results:
        b0 = r['baseline'][0] if r.get('baseline') else 0
        print(f'{r["game_id"]:<8} {r.get("levels_completed",0):>7} '
              f'{r.get("total_actions",0):>9} {b0:>8} '
              f'{r.get("l0_score_pct",0):>9.1f}% '
              f'{r.get("elapsed",0):>6.0f}s')
        total_l0 += r.get('l0_score_pct', 0)

    avg_l0 = total_l0 / len(results) if results else 0
    print('-'*70)
    print(f'Average L0 score: {avg_l0:.1f}%')
    print(f'Games with L0 solved: {sum(1 for r in results if r.get("levels_completed",0)>0)}/{len(results)}')
    print('='*70)


if __name__ == '__main__':
    main()
