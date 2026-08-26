"""Run summarize_observations on every collected game."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OBS_DIR = ROOT / "exec_wm" / "observations"

games = sorted(p.stem for p in OBS_DIR.glob("*.json") if not p.stem.endswith(".summary"))
for g in games:
    summary = OBS_DIR / f"{g}.summary.json"
    if summary.exists() and summary.stat().st_size > 1024:
        print(f"{g}: SKIP (summary exists)")
        continue
    r = subprocess.run(
        ["uv", "run", "python", str(ROOT / "exec_wm" / "summarize_observations.py"), "--game", g],
        capture_output=True, text=True,
    )
    line = r.stdout.strip().splitlines()[0] if r.stdout.strip() else ""
    print(f"{g}: {line[:100]}")
