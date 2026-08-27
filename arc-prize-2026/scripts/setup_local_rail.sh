#!/bin/bash
# Re-install the two IN-REPO packages the local screening rail needs.
#
# RUN THIS AFTER EVERY `uv sync`.
#
# Both are vendored in this repo and installed with --no-deps, i.e. OUTSIDE the
# uv lockfile -- so `uv sync` prunes them every time and the local rail breaks
# with ModuleNotFoundError. This script is idempotent; just re-run it.
#
#   taaf    duck_eval/taaf_bundle/...      the competition harness (Benchmark,
#                                          GameAPI, CompetitionArcadeServer)
#   re_arc  runs/harness_diff_0813/...     `arc-agi-3-local`, carries the 192
#                                          environment_files incl. the official
#                                          25 games. It is the git dependency
#                                          taaf declares; installing from the
#                                          vendored snapshot avoids GitHub.
#
# They are deliberately NOT added to pyproject.toml: taaf pins
# requires-python == 3.12.12 and re_arc pulls a git URL, either of which would
# constrain or break the main resolution.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

UV="${UV:-$HOME/.local/bin/uv}"
PY=".venv/bin/python"
[ -x "$PY" ] || { echo "no .venv -- run 'uv sync' first" >&2; exit 1; }

TAAF="duck_eval/taaf_bundle/src/tufa-arc-agi-framework"
REARC="runs/harness_diff_0813/ds/jeroencottaar_taaf-kaggle-source/src/re-arc-3"

for p in "$TAAF" "$REARC"; do
    [ -d "$p" ] || { echo "MISSING vendored package: $p" >&2; exit 1; }
done

echo "[setup] installing vendored taaf + re_arc (--no-deps)"
"$UV" pip install --python "$PY" --no-deps -e "$TAAF"  >/dev/null
"$UV" pip install --python "$PY" --no-deps -e "$REARC" >/dev/null
# taaf renders MP4 diagnostics; without a backend the gate's harness smoke fails
"$UV" pip install --python "$PY" "imageio-ffmpeg>=0.5"  >/dev/null

echo "[setup] verifying"
"$PY" - <<'PY'
import re_arc, taaf.benchmark, taaf.game_api, imageio_ffmpeg
from pathlib import Path
env = Path(re_arc.__file__).resolve().parent / "environment_files"
official = re_arc.list_game_ids(datasets=["train", "eval"], include_tags="official")
print(f"  taaf     OK  {Path(taaf.__file__).parent}")
print(f"  re_arc   OK  {len(list(env.iterdir()))} env files, {len(official)} official games")
print(f"  ffmpeg   OK  {Path(imageio_ffmpeg.get_ffmpeg_exe()).name}")
PY
echo "[setup] local rail ready"
