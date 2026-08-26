"""Sequential per-game sweep.

Spawns local_eval.py 25 times — once per game — so each game gets full CPU.
Concurrent swarm in main.py shares one CPU across 25 games, making each ~25x slower.

Usage:
    uv run python sweep_per_game.py <agent_path.py> <agent_class> [--per-game-timeout 90]

Outputs: a sweep summary at runs/sweep-<desc>-<ts>/sweep_summary.json
"""

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).parent.resolve()
ENV_DIR = PROJECT / "kaggle-data" / "environment_files"


def list_games() -> list[str]:
    if not ENV_DIR.exists():
        raise SystemExit(f"missing {ENV_DIR}")
    return sorted(p.name for p in ENV_DIR.iterdir() if p.is_dir())


def parse_scorecard(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("agent_src")
    ap.add_argument("agent_class")
    ap.add_argument("--cli-name", default="myagent")
    ap.add_argument("--per-game-timeout", type=int, default=120,
                    help="Subprocess timeout per game (s)")
    ap.add_argument("--desc", default="sweep")
    ap.add_argument("--games", default=None,
                    help="Comma-separated subset; default all")
    args = ap.parse_args()

    games = args.games.split(",") if args.games else list_games()
    print(f"[sweep] {len(games)} games, timeout={args.per_game_timeout}s each")

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    sweep_dir = PROJECT / "runs" / f"sweep-{args.desc}-{ts}"
    sweep_dir.mkdir(parents=True)

    results = []
    t0 = time.time()
    for i, gid in enumerate(games, 1):
        g_t0 = time.time()
        run_desc = f"{args.desc}-{gid}"
        cmd = [
            sys.executable, "local_eval.py",
            args.agent_src, args.agent_class,
            "--game", gid,
            "--desc", run_desc,
            "--cli-name", args.cli_name,
            "--timeout", str(args.per_game_timeout),
        ]
        print(f"\n[sweep] [{i}/{len(games)}] {gid} ...")
        # Run, capture exit code; stream output sparingly.
        log_path = sweep_dir / f"{gid}.log"
        with open(log_path, "wb") as logf:
            proc = subprocess.run(
                cmd,
                cwd=str(PROJECT),
                stdout=logf,
                stderr=subprocess.STDOUT,
                timeout=args.per_game_timeout + 60,
            )
        elapsed = time.time() - g_t0

        # Find the run dir created by local_eval (most recent matching)
        candidates = sorted(
            (PROJECT / "runs").glob(f"runs-*-{run_desc}"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        run_dir = candidates[0] if candidates else None
        sc = parse_scorecard(run_dir / "scorecard.json") if run_dir else None

        # Pull per-game stats from scorecard
        score = None
        levels = None
        nlevels = None
        actions = None
        completed = None
        if sc and sc.get("environments"):
            env = sc["environments"][0]
            score = env.get("score")
            levels = env.get("levels_completed")
            nlevels = env.get("level_count")
            actions = env.get("actions")
            completed = env.get("completed")

        # Also extract any "BFS SOLVED" / "BFS first pass timeout" tallies from log
        bfs_solved = 0
        bfs_failed = 0
        if run_dir and (run_dir / "run.log").exists():
            try:
                runtxt = (run_dir / "run.log").read_text(encoding="utf-8", errors="replace")
                bfs_solved = len(re.findall(r"BFS L\d+: SOLVED", runtxt))
                bfs_failed = len(re.findall(r"BFS L\d+:.*timeout", runtxt))
            except Exception:
                pass

        result = {
            "game": gid,
            "elapsed_s": round(elapsed, 1),
            "subprocess_exit": proc.returncode,
            "score": score,
            "levels_completed": levels,
            "n_levels": nlevels,
            "actions": actions,
            "completed": completed,
            "bfs_solved_levels": bfs_solved,
            "bfs_timeouts": bfs_failed,
            "run_dir": str(run_dir) if run_dir else None,
        }
        results.append(result)
        score_s = f"{score:.4f}" if isinstance(score, (int, float)) else "?"
        lvl_s = f"{levels}/{nlevels}" if levels is not None else "?"
        print(f"[sweep]   -> score={score_s} levels={lvl_s} acts={actions} bfs_solved={bfs_solved} t={elapsed:.0f}s")

    total = time.time() - t0
    overall_score = None
    scores = [r["score"] for r in results if isinstance(r["score"], (int, float))]
    if scores:
        overall_score = sum(scores) / len(scores)

    summary = {
        "agent_src": args.agent_src,
        "agent_class": args.agent_class,
        "n_games": len(games),
        "total_elapsed_s": round(total, 1),
        "overall_score_avg": overall_score,
        "results": results,
    }
    summary_path = sweep_dir / "sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print()
    print(f"[sweep] DONE total={total:.0f}s avg_score={overall_score}")
    print(f"[sweep] summary -> {summary_path}")

    # Print rank table
    print(f"\n{'GAME':<8} {'SCORE':>8} {'LEVELS':>8} {'ACTS':>6} {'BFS_OK':>7} {'BFS_FAIL':>9} {'T(s)':>6}")
    print("-" * 64)
    for r in sorted(results, key=lambda x: -(x["score"] or 0)):
        sc_s = f"{r['score']:.4f}" if isinstance(r["score"], (int, float)) else "?"
        lvl_s = f"{r['levels_completed']}/{r['n_levels']}" if r["levels_completed"] is not None else "?"
        print(f"{r['game']:<8} {sc_s:>8} {lvl_s:>8} {str(r['actions']):>6} {r['bfs_solved_levels']:>7} {r['bfs_timeouts']:>9} {r['elapsed_s']:>6.0f}")


if __name__ == "__main__":
    main()
