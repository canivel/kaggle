"""Analyze a per-game sweep by parsing run.log from each per-game run dir.

Usage:
    uv run python analyze_sweep.py <pattern>
    e.g. uv run python analyze_sweep.py runs/runs-*v19-perg2-*

Reads each run.log and extracts:
- BFS solves per level
- Max levels_completed observed (proxy for "this many levels done")
- Final action count
- Scorecard score (if scorecard.json present)
"""

import argparse
import json
import re
import sys
from pathlib import Path


def parse_log(p: Path) -> dict:
    out = {
        "game": "?",
        "max_levels_completed": 0,
        "final_action_count": 0,
        "bfs_solved": [],          # ["L0:4", "L1:45", ...]
        "bfs_timeouts": 0,
        "bfs_load_failed": False,
        "score": None,
        "completed": None,
        "n_levels": None,
        "fps_avg": 0.0,
    }
    if not p.exists():
        return out
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return out

    # Identify game from action lines: "ft09 - ACTION6: count..."
    m = re.search(r"^(\w+)\s+-\s+ACTION", text, re.MULTILINE)
    if m:
        out["game"] = m.group(1)

    # Per-action lines: extract max levels_completed and last count
    last_count = 0
    max_lvl = 0
    last_fps = 0.0
    for m in re.finditer(r"count\s+(\d+),\s+levels completed\s+(\d+),\s+avg fps\s+([0-9.]+)", text):
        c, l, f = int(m.group(1)), int(m.group(2)), float(m.group(3))
        if c > last_count:
            last_count = c
        if l > max_lvl:
            max_lvl = l
        last_fps = f
    out["final_action_count"] = last_count
    out["max_levels_completed"] = max_lvl
    out["fps_avg"] = round(last_fps, 2)

    # BFS solved messages
    for m in re.finditer(r"BFS L(\d+):\s+SOLVED.*?in\s+(\d+)\s+actions", text):
        out["bfs_solved"].append(f"L{m.group(1)}:{m.group(2)}")

    # BFS timeouts
    out["bfs_timeouts"] = len(re.findall(r"BFS L\d+:.*timeout", text))

    # BFS source not found
    if "BFS: game source not found" in text:
        out["bfs_load_failed"] = True

    return out


def parse_scorecard(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern", nargs="+",
                    help="Run dir glob, e.g. 'runs/runs-*v19-perg2-*'")
    args = ap.parse_args()

    project = Path(__file__).parent.resolve()
    rows = []
    for pat in args.pattern:
        for d in sorted(project.glob(pat.replace("/", "\\")) if "\\" in pat else project.glob(pat)):
            if not d.is_dir():
                continue
            log_p = d / "run.log"
            sc_p = d / "scorecard.json"
            r = parse_log(log_p)
            # Fallback: extract game id from dir name (last token after last '-')
            if r["game"] == "?":
                m = re.search(r"v19-perg2-(\w+)$", d.name)
                if m:
                    r["game"] = m.group(1)
                else:
                    parts = d.name.rsplit("-", 1)
                    if len(parts) > 1:
                        r["game"] = parts[-1]
            sc = parse_scorecard(sc_p)
            if sc:
                envs = sc.get("environments", [])
                if envs:
                    e = envs[0]
                    r["score"] = e.get("score")
                    r["completed"] = e.get("completed")
                    r["n_levels"] = e.get("level_count")
                    r["scorecard_levels_completed"] = e.get("levels_completed")
            r["run_dir"] = str(d.name)
            rows.append(r)

    rows.sort(key=lambda x: (-(x.get("score") or 0), -x["max_levels_completed"], x["game"]))

    print(f"\nAnalyzed {len(rows)} runs\n")
    print(f"{'GAME':<8} {'SCORE':>7} {'LVLS':>5} {'ACTS':>6} {'BFS_OK':>7} {'BFS_X':>6} {'FPS':>5}  SOLVED_LEVELS")
    print("-" * 80)
    for r in rows:
        gid = r["game"][:7]
        sc_s = f"{r['score']:.4f}" if isinstance(r.get("score"), (int, float)) else "  -  "
        lvl_s = str(r["max_levels_completed"])
        if r.get("n_levels"):
            lvl_s = f"{r['max_levels_completed']}/{r['n_levels']}"
        bfs_load_str = "[FAIL]" if r["bfs_load_failed"] else ""
        solves = ",".join(r["bfs_solved"][:6])
        print(f"{gid:<8} {sc_s:>7} {lvl_s:>5} {r['final_action_count']:>6} {len(r['bfs_solved']):>7} {r['bfs_timeouts']:>6} {r['fps_avg']:>5}  {solves} {bfs_load_str}")

    # Summary stats
    n = len(rows)
    n_with_solve = sum(1 for r in rows if r["bfs_solved"])
    n_lvl1plus = sum(1 for r in rows if r["max_levels_completed"] >= 1)
    n_load_failed = sum(1 for r in rows if r["bfs_load_failed"])
    total_levels = sum(r["max_levels_completed"] for r in rows)

    print()
    print(f"Summary: {n} games | {n_with_solve} had BFS solve | {n_lvl1plus} reached L1+ | "
          f"{n_load_failed} BFS load failed | total levels reached: {total_levels}")


if __name__ == "__main__":
    main()
