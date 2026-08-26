"""Collect observations for all 25 public ARC-AGI-3 games in one Python session
to avoid 25 import-time startups.

Skips games whose observations/<game_id>.json already exists.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exec_wm"))

GAMES = [
    "ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59", "lf52", "lp85",
    "ls20", "m0r0", "r11l", "re86", "s5i5", "sb26", "sc25", "sk48", "sp80", "su15",
    "tn36", "tr87", "tu93", "vc33", "wa30",
]

from collect_observations import collect

OUT_DIR = ROOT / "exec_wm" / "observations"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    n_random = 200
    bfs_timeout = 60
    for gid in GAMES:
        out = OUT_DIR / f"{gid}.json"
        if out.exists():
            print(f"{gid}: SKIP (exists)")
            continue
        t0 = time.time()
        try:
            data = collect(gid, n_random=n_random, bfs_timeout=bfs_timeout, seed=1)
            out.write_text(json.dumps(data))
            s = data["summary"]
            print(f"{gid}: tuples={s['n_tuples']} levels={s['n_levels_observed']} "
                  f"changes={s['n_state_changes']} took {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"{gid}: FAIL {e!r}")


if __name__ == "__main__":
    main()
