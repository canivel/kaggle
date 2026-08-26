#!/bin/bash
# Local test using the actual Kaggle ARC-AGI-3-Agents framework.
# This mimics what happens on Kaggle during a competition rerun.
#
# Usage: bash local_test_kaggle.sh

set -e
cd "$(dirname "$0")"

echo "=== Setting up local Kaggle test ==="

# Copy the framework
rm -rf /tmp/arc-test
cp -r kaggle-data/ARC-AGI-3-Agents /tmp/arc-test

# Copy our agent
cp notebooks/forge_agent/forge_v19_improved.py /tmp/arc-test/agents/templates/my_agent.py

# Register our agent
cat > /tmp/arc-test/agents/__init__.py << 'PYEOF'
from typing import Type, cast
from dotenv import load_dotenv
from .agent import Agent, Playback
from .swarm import Swarm
from .templates.random_agent import Random
from .templates.my_agent import MyAgent
load_dotenv()
AVAILABLE_AGENTS: dict[str, Type[Agent]] = {"random": Random, "myagent": MyAgent}
PYEOF

# Configure for local offline mode
cat > /tmp/arc-test/.env << 'ENVEOF'
SCHEME=http
HOST=localhost
PORT=8001
ARC_API_KEY=
ARC_BASE_URL=http://localhost:8001/
OPERATION_MODE=online
RECORDINGS_DIR=/tmp/arc-test/recordings
ENVEOF

echo "=== Running agent ==="
cd /tmp/arc-test
MPLBACKEND=agg python main.py --agent myagent
