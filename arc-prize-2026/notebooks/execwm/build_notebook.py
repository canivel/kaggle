"""Build arc3-execwm.ipynb by inlining v64_agent.py + exec_wm package via %%writefile.

Sims are NOT inlined — they're attached as the canivel/exec-wm-sims Kaggle dataset.
The ExecWMHook discovers them via glob across known mount paths.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NB = Path(__file__).resolve().parent / "arc3-execwm.ipynb"


def codecell(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [src]}


def writefile(path: str, content: str) -> dict:
    return codecell(f"%%writefile {path}\n{content}")


def read(p: str) -> str:
    return (ROOT / p).read_text(encoding="utf-8")


cells = []

# Cell 0: install
cells.append(codecell(
    "!pip install --no-index --find-links \\\n"
    "    /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels \\\n"
    "    arc-agi python-dotenv\n"
))

# Stage the exec_wm package onto /kaggle/working/
cells.append(codecell("!mkdir -p /kaggle/working/exec_wm\n"))
cells.append(writefile("/kaggle/working/exec_wm/__init__.py", "# exec_wm package\n"))
cells.append(writefile("/kaggle/working/exec_wm/agent_hook.py", read("exec_wm/agent_hook.py")))

# Diagnostic + hook smoke test: load + report sim count
cells.append(codecell(
    "import sys, os, glob\n"
    "if '/kaggle/working' not in sys.path:\n"
    "    sys.path.insert(0, '/kaggle/working')\n"
    "print('=== /kaggle/input/ structure ===')\n"
    "for root in glob.glob('/kaggle/input/*'):\n"
    "    print(root)\n"
    "    for f in glob.glob(root + '/*'):\n"
    "        print(' ', f)\n"
    "print('=== loading ExecWMHook ===')\n"
    "from exec_wm.agent_hook import ExecWMHook\n"
    "hook = ExecWMHook(beam_width=4, lookahead=2)\n"
    "print('available:', hook.available)\n"
    "print('n_sims:', len(hook.registry))\n"
    "print('sims:', sorted(hook.registry))\n"
))

# Cell: write the agent
cells.append(writefile("/kaggle/working/my_agent.py", read("notebooks/forge_agent/v64_agent.py")))

# Markdown
cells.append(md("Below cell runs only on competition rerun (not during build)."))

# Competition rerun cell (copies exec_wm to ARC-AGI-3-Agents tree too)
run_src = """import os
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    !curl --fail --retry 999 --retry-all-errors --retry-delay 5 --retry-max-time 600 http://gateway:8001/api/games
    !cp -r /kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents /kaggle/working/ARC-AGI-3-Agents
    !cp /kaggle/working/my_agent.py /kaggle/working/ARC-AGI-3-Agents/agents/templates/my_agent.py
    !mkdir -p /kaggle/working/ARC-AGI-3-Agents/exec_wm
    !cp /kaggle/working/exec_wm/__init__.py /kaggle/working/ARC-AGI-3-Agents/exec_wm/__init__.py
    !cp /kaggle/working/exec_wm/agent_hook.py /kaggle/working/ARC-AGI-3-Agents/exec_wm/agent_hook.py
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

# Dummy submission fallback (so notebook produces a parquet during build)
cells.append(md("Dummy submission fallback (only fires during build, not rerun)."))
cells.append(codecell(
    "import os\n"
    "if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):\n"
    "    import pandas as pd\n"
    "    submission = pd.DataFrame(data=[['1_0','1',True,1]],columns=['row_id','game_id','end_of_game','score'])\n"
    "    submission.to_parquet('/kaggle/working/submission.parquet',index=False)\n"
))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {NB} ({len(cells)} cells)")
