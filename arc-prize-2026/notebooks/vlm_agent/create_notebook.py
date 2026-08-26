"""Create arc3-vlm-agent.ipynb from forge_v33_localvlm.py"""
import json

with open('f:/kaggle/arc-prize-2026/notebooks/forge_agent/forge_v33_localvlm.py') as f:
    v33_code = f.read()

agent_source = ('%%writefile /kaggle/working/my_agent.py\n' + v33_code).splitlines(keepends=True)

init_py = (
    'from typing import Type\n'
    'from dotenv import load_dotenv\n'
    'from .agent import Agent, Playback\n'
    'from .swarm import Swarm\n'
    'from .templates.random_agent import Random\n'
    'from .templates.my_agent import MyAgent\n'
    'load_dotenv()\n'
    'AVAILABLE_AGENTS: dict[str, Type[Agent]] = {"random": Random, "myagent": MyAgent}\n'
)

env_content = (
    'SCHEME=http\n'
    'HOST=gateway\n'
    'PORT=8001\n'
    'ARC_API_KEY=test-key-123\n'
    'ARC_BASE_URL=http://gateway:8001/\n'
    'OPERATION_MODE=online\n'
    'RECORDINGS_DIR=/kaggle/working/server_recording\n'
)

run_cell = [
    'import os\n',
    "if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):\n",
    '    !curl --fail --retry 999 --retry-all-errors --retry-delay 5 --retry-max-time 600 http://gateway:8001/api/games\n',
    '    !cp -r /kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents /kaggle/working/ARC-AGI-3-Agents\n',
    '    !cp /kaggle/working/my_agent.py /kaggle/working/ARC-AGI-3-Agents/agents/templates/my_agent.py\n',
    '    with open(\'/kaggle/working/ARC-AGI-3-Agents/agents/__init__.py\', \'w\') as f:\n',
    '        f.write(' + repr(init_py) + ')\n',
    '    with open(\'/kaggle/working/ARC-AGI-3-Agents/.env\', \'w\') as f:\n',
    '        f.write(' + repr(env_content) + ')\n',
    '    !cd /kaggle/working/ARC-AGI-3-Agents && MPLBACKEND=agg python main.py --agent myagent\n',
]

dummy_cell = [
    'import os\n',
    "if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):\n",
    '    import pandas as pd\n',
    "    submission = pd.DataFrame(data=[['1_0','1',True,1]], columns=['row_id','game_id','end_of_game','score'])\n",
    "    submission.to_parquet('/kaggle/working/submission.parquet', index=False)\n",
]

nb = {
    'nbformat': 4,
    'nbformat_minor': 4,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.10.0'}
    },
    'cells': [
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                '!pip install --no-index --find-links \\\n',
                '    /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels \\\n',
                '    arc-agi python-dotenv\n',
                '\n',
                '# qwen-vl-utils for vision processing\n',
                '!pip install -q qwen-vl-utils\n',
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': agent_source
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': run_cell
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': dummy_cell
        }
    ]
}

out = 'f:/kaggle/arc-prize-2026/notebooks/vlm_agent/arc3-vlm-agent.ipynb'
with open(out, 'w') as f:
    json.dump(nb, f)

print(f'Created {out}')
print(f'  Agent cell: {len(agent_source)} lines')
