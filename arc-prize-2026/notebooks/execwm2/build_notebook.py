"""Build arc3-execwm-v2.ipynb. v65 has ExecWMHook INLINED into my_agent.py
so we only need to ship the agent code + attach the sims dataset.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NB = Path(__file__).resolve().parent / "arc3-execwm-v2.ipynb"


def codecell(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.splitlines(keepends=True)}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": [src]}


def writefile(path, content):
    return codecell(f"%%writefile {path}\n{content}")


cells = []
cells.append(codecell(
    "!pip install --no-index --find-links \\\n"
    "    /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels \\\n"
    "    arc-agi python-dotenv\n"
))

# Diagnostic: verify sims dataset mounted
cells.append(codecell(
    "import glob, os\n"
    "print('=== /kaggle/input/ structure ===')\n"
    "for root in glob.glob('/kaggle/input/*'):\n"
    "    print(root)\n"
    "    for f in glob.glob(root + '/*'):\n"
    "        print(' ', f)\n"
    "print('=== sims discovery ===')\n"
    "patterns = ['/kaggle/input/exec-wm-sims/*_sim.py',\n"
    "            '/kaggle/input/datasets/canivel/exec-wm-sims/*_sim.py',\n"
    "            '/kaggle/input/**/canivel/exec-wm-sims/*_sim.py']\n"
    "for p in patterns:\n"
    "    matches = glob.glob(p, recursive=True)\n"
    "    print(f'{p}: {len(matches)} matches')\n"
))

# Write the agent (single file — hook is inlined)
cells.append(writefile("/kaggle/working/my_agent.py", (ROOT / "notebooks/forge_agent/v65_agent.py").read_text(encoding="utf-8")))

cells.append(md("Below cell runs only on competition rerun (not during build)."))

run_src = """import os
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    !curl --fail --retry 999 --retry-all-errors --retry-delay 5 --retry-max-time 600 http://gateway:8001/api/games
    !cp -r /kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents /kaggle/working/ARC-AGI-3-Agents
    !cp /kaggle/working/my_agent.py /kaggle/working/ARC-AGI-3-Agents/agents/templates/my_agent.py
    with open('/kaggle/working/ARC-AGI-3-Agents/agents/__init__.py','w') as f:
        f.write(\"\"\"from typing import TYPE_CHECKING
from .agent import Agent
from .templates.my_agent import MyAgent
AVAILABLE_AGENTS = {'myagent': MyAgent}
\"\"\")
    with open('/kaggle/working/ARC-AGI-3-Agents/.env','w') as f:
        f.write(\"\"\"ARC_API_KEY=arc-agi-3
ARC_BASE_URL=http://gateway:8001/
OPERATION_MODE=online
RECORDINGS_DIR=/kaggle/working/server_recording
\"\"\")
    !cd /kaggle/working/ARC-AGI-3-Agents && MPLBACKEND=agg python main.py --agent myagent
"""
cells.append(codecell(run_src))

cells.append(md("Dummy submission fallback (only fires during build, not rerun)."))
cells.append(codecell(
    "import os\n"
    "if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):\n"
    "    import pandas as pd\n"
    "    submission = pd.DataFrame(data=[['1_0','1',True,1]],columns=['row_id','game_id','end_of_game','score'])\n"
    "    submission.to_parquet('/kaggle/working/submission.parquet',index=False)\n"
))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.10"}},
      "nbformat": 4, "nbformat_minor": 5}
NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {NB} ({len(cells)} cells)")
