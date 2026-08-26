"""Build arc3-forge62.ipynb by inlining v62_agent.py + jepa_wm package via %%writefile cells."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NB = Path(__file__).resolve().parent / "arc3-jepa-v2.ipynb"


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

# Cells 1..N: write jepa_wm package files under /kaggle/working/
cells.append(codecell("!mkdir -p /kaggle/working/jepa_wm/models /kaggle/working/jepa_wm/inference\n"))
cells.append(writefile("/kaggle/working/jepa_wm/__init__.py", "# jepa_wm package\n"))
cells.append(writefile("/kaggle/working/jepa_wm/models/__init__.py", "# jepa_wm.models package\n"))
cells.append(writefile("/kaggle/working/jepa_wm/inference/__init__.py", "# jepa_wm.inference package\n"))
cells.append(writefile("/kaggle/working/jepa_wm/models/jepa.py", read("jepa_wm/models/jepa.py")))
cells.append(writefile("/kaggle/working/jepa_wm/inference/mcts.py", read("jepa_wm/inference/mcts.py")))
cells.append(writefile("/kaggle/working/jepa_wm/inference/agent_hooks.py", read("jepa_wm/inference/agent_hooks.py")))

# Cell: diagnose /kaggle/input + try loading the JEPAHook with verbose output
cells.append(codecell(
    "import sys, os, glob, traceback\n"
    "print('=== /kaggle/input/ contents ===')\n"
    "for root in glob.glob('/kaggle/input/*'):\n"
    "    print(root)\n"
    "    for f in glob.glob(root + '/*'):\n"
    "        print(' ', f)\n"
    "if '/kaggle/working' not in sys.path:\n"
    "    sys.path.insert(0, '/kaggle/working')\n"
    "print('=== attempting JEPAHook load ===')\n"
    "from jepa_wm.inference.agent_hooks import JEPAHook, _find_weights\n"
    "found = _find_weights()\n"
    "print('found weight path:', found)\n"
    "import glob\n"
    "print('all .pt under /kaggle/input:', glob.glob('/kaggle/input/**/*.pt', recursive=True)[:10])\n"
    "# Verbose load: bypass try/except to see the real error\n"
    "try:\n"
    "    h = JEPAHook(device='cpu', n_simulations=4, max_depth=2)\n"
    "    print('hook available:', h.available)\n"
    "    if not h.available:\n"
    "        import torch\n"
    "        for w in WEIGHT_CANDIDATES:\n"
    "            if os.path.exists(w):\n"
    "                ck = torch.load(w, map_location='cpu', weights_only=False)\n"
    "                print('  raw load ok, keys:', list(ck.keys()))\n"
    "                print('  cfg:', ck.get('cfg'))\n"
    "                break\n"
    "except Exception as e:\n"
    "    print('LOAD FAILED:', repr(e))\n"
    "    traceback.print_exc()\n"
))

# Cell: write the agent
cells.append(writefile("/kaggle/working/my_agent.py", read("notebooks/forge_agent/v63_agent.py")))

# Markdown
cells.append(md("this only runs if you submit to the competition, not when you do tests"))

# Run cell (from forge35)
run_src = """import os
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    !curl --fail --retry 999 --retry-all-errors --retry-delay 5 --retry-max-time 600 http://gateway:8001/api/games
    !cp -r /kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents /kaggle/working/ARC-AGI-3-Agents
    !cp /kaggle/working/my_agent.py /kaggle/working/ARC-AGI-3-Agents/agents/templates/my_agent.py
    !mkdir -p /kaggle/working/ARC-AGI-3-Agents/jepa_wm/models /kaggle/working/ARC-AGI-3-Agents/jepa_wm/inference
    !cp /kaggle/working/jepa_wm/__init__.py /kaggle/working/ARC-AGI-3-Agents/jepa_wm/__init__.py
    !cp /kaggle/working/jepa_wm/models/__init__.py /kaggle/working/ARC-AGI-3-Agents/jepa_wm/models/__init__.py
    !cp /kaggle/working/jepa_wm/inference/__init__.py /kaggle/working/ARC-AGI-3-Agents/jepa_wm/inference/__init__.py
    !cp /kaggle/working/jepa_wm/models/jepa.py /kaggle/working/ARC-AGI-3-Agents/jepa_wm/models/jepa.py
    !cp /kaggle/working/jepa_wm/inference/mcts.py /kaggle/working/ARC-AGI-3-Agents/jepa_wm/inference/mcts.py
    !cp /kaggle/working/jepa_wm/inference/agent_hooks.py /kaggle/working/ARC-AGI-3-Agents/jepa_wm/inference/agent_hooks.py
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

# Dummy submission fallback
cells.append(md("This is a dummy submission fallback, important to keep"))
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
