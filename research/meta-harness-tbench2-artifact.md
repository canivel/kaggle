# Research Report: stanford-iris-lab/meta-harness-tbench2-artifact

Source: https://github.com/stanford-iris-lab/meta-harness-tbench2-artifact

---

## What Is This Project?

Meta-Harness is an LLM agent scaffold for **Terminal-Bench 2.0 (tbench2)** — a benchmark of 89 terminal-based Linux tasks (coding, file manipulation, system administration, etc.). It has nothing to do with math competitions (AIMO, AIME, IMO).

**It is NOT a math solver.** It is a general-purpose coding/terminal agent.

### Key Achievement
- **76.4% accuracy** on Terminal-Bench 2.0 using **Claude Opus 4.6**
- Easy tasks (4): 100.0%
- Medium tasks (55): 81.1%
- Hard tasks (30): 64.7%
- Evaluation: 5 trials per task, 445 total runs

---

## Architecture

### Base Framework
Built on the **Harbor framework** (`harbor>=0.1.44`), extending the `Terminus2` base class. The `AgentHarness` class further extends `Terminus2` with:
1. Native tool calling (instead of JSON/XML parsing)
2. Environment bootstrapping
3. Marker-based command polling

### Lineage
```
Harbor Terminus2 (base)
  -> Terminus-KIRA (KRAFTON AI, adds native tool calling)
     -> AgentHarness (Stanford IRIS, adds env bootstrapping)
```

### File Structure
```
agent.py               - Main AgentHarness class (~1100 lines)
anthropic_caching.py   - Anthropic prompt caching utility
prompt-templates/
  terminus-kira.txt    - System prompt template
pyproject.toml         - Dependencies
```

---

## How It Works

### 1. Environment Bootstrapping (Key Innovation)
Before the agent loop starts, `_gather_env_snapshot()` runs a single compound shell command that collects:
- Working directory (pwd)
- File listing (ls -la /app/)
- Available languages (python3, gcc, g++, node, java, rustc, go)
- Package managers (pip3, pip, apt-get)
- Memory (free -h)

This is injected into the initial prompt, eliminating 2-5 "exploration turns" the agent would otherwise spend running basic recon commands.

```python
bootstrap_cmd = (
    "echo '@@PWD@@' && pwd && "
    "echo '@@LS@@' && ls -la /app/ 2>/dev/null && "
    "echo '@@LANG@@' && (python3 --version 2>&1) && ..."
    "echo '@@MEM@@' && free -h 2>/dev/null ..."
)
```

### 2. Native Tool Calling
Uses Anthropic's structured tool definitions (via litellm) instead of asking the LLM to emit parseable JSON/XML in free text. Three tools:

**execute_commands**: Enforces "think-plan-act" structure
```json
{
  "analysis": "what I see and what's done",
  "plan": "what I'll do next",
  "commands": [{"keystrokes": "cmd\n", "duration": 1.0}]
}
```
- Duration capped at 60 seconds
- Analysis and plan fields are required (forces structured reasoning)

**task_complete**: Parameterless signal that task is done

**image_read**: Multimodal image analysis
```json
{"file_path": "/abs/path.png", "image_read_instruction": "describe this"}
```
Reads file via base64 from the sandbox, sends as multimodal message.

### 3. Marker-Based Command Polling
Commands are followed immediately by `echo '__CMDEND__N__'`. Instead of waiting the full `duration_sec`, the agent polls every 0.5s and exits early when the marker appears. Tracks total time saved in `_total_time_saved`.

### 4. Double-Confirmation for Task Completion
When the agent calls `task_complete`, a `_pending_completion` flag triggers a second confirmation prompt with a checklist:
- Does solution meet requirements?
- Accounts for numeric/array/file changes?
- Verified from test engineer, QA, and user perspectives?

### 5. Context Management
When context length is exceeded:
1. `_unwind_messages_to_free_tokens()` - trims history to free 4000 tokens
2. `_summarize()` - LLM-based summarization of trajectory
3. Fallback: last 1000 chars of terminal screen if summarization fails
4. `_split_trajectory_on_summarization()` - maintains audit trail

### 6. Anthropic Prompt Caching
`add_anthropic_caching()` adds ephemeral cache control to the 3 most recent messages, converting string content to list format as needed to attach cache headers.

---

## Prompt Template (terminus-kira.txt)
```
You are an AI assistant tasked with solving command-line tasks in a Linux environment.
You will be given a task description and the output from previously executed commands.
Your goal is to solve the task by providing batches of shell commands.

Your plan MUST account that you as an AI agent must complete the entire task without
any human intervention, and you should NOT expect any human interventions. Also, you
do NOT have eyes or ears, so you MUST resort to various programmatic/AI tools to
understand multimedia files.

Before calling task_complete, verify minimal state changes: Re-read the task
instructions carefully and identify the absolute minimum set of files that must be
created or modified to satisfy the requirements. List these files explicitly. Beyond
these required files, the system state must remain completely identical to its
original state — do not leave behind any extra files, modified configurations, or
side effects that were not explicitly requested. Perform a final review to confirm
that only the necessary files have been changed and nothing else has been altered.

Task Description:
{instruction}

Current terminal state:
{terminal_state}
```

---

## How to Run

### Prerequisites
```bash
pip install harbor
export ANTHROPIC_API_KEY=<your-key>
```

### Run Command
```bash
harbor run \
  --agent-import-path agent:AgentHarness \
  -d terminal-bench@2.0 \
  -m anthropic/claude-opus-4-6 \
  -e runloop \
  -n 20 \
  --n-attempts 5
```

Parameters:
- `-d terminal-bench@2.0` — Terminal-Bench 2.0 dataset
- `-m anthropic/claude-opus-4-6` — Claude Opus 4.6 model
- `-e runloop` — RunLoop sandbox environment
- `-n 20` — 20 tasks per batch
- `--n-attempts 5` — 5 trials per task

---

## Dependencies

```toml
[project]
name = "meta-harness"
version = "1.0.0"
description = "LLM agent scaffold for Terminal-Bench 2.0"
requires-python = ">=3.12"
dependencies = [
  "anthropic",
  "harbor>=0.1.44",
  "litellm<1.82.7",
  "tenacity",
]
```

NOTE: This project uses `litellm` which is banned in our environment per global instructions. Any adaptation would need to replace litellm with the native `anthropic` SDK.

---

## Technical Details

### LLM Call Configuration
- Temperature: `1.0` when reasoning_effort is set (API requirement for extended thinking)
- Supports `reasoning_effort` parameter for Claude extended thinking models
- Captures `reasoning_content` from responses when available
- Timeout: 900 seconds (15 min) per LLM call
- Retry: 5 attempts, exponential backoff (0.5s-4s)
- Non-retryable: BadRequestError, AuthenticationError, ContextLengthExceededError

### Token Tracking
Tracks cumulative: input_tokens, output_tokens, cache_tokens (cache_read_input_tokens), cost_usd

### Block Timeout
600 seconds (10 min) timeout on infrastructure API calls to prevent hangs.

### Output Length Limit
30,000 bytes per command output. Terminal snapshots truncated to 1,000 chars.

---

## Relationship to AIMO / Math Competitions

**None.** This project is a terminal/coding agent framework for the Terminal-Bench 2.0 leaderboard. The Stanford IRIS Lab's other repos focus on reinforcement learning and robotics, not math competitions.

The AIMO-2 winning solution (arxiv 2504.16891) is a completely separate project from NVIDIA (OpenMathReasoning dataset, 540K problems, 3.2M solutions). That is the relevant math competition paper.

---

## Relevance to Our Customer Churn Competition

Direct relevance: None. However, the architectural patterns are instructive:

1. **Environment bootstrapping pattern**: Inject environment state upfront to eliminate exploration overhead. In tabular ML context, this maps to caching feature statistics, CV splits, and baseline scores in the initial context of an agent solving our competition.

2. **Structured "think-plan-act" tool calling**: The `execute_commands` tool forcing `analysis` + `plan` fields before commands is a pattern we could adopt in a Kaggle agent to improve reasoning quality.

3. **Double-confirmation before terminal action**: The checklist before `task_complete` maps to validation checks before submission (CV score check, LB estimate, sanity checks).

4. **Context management with summarization**: Handling long trajectories by summarizing and continuing is directly applicable to long-running Kaggle agent sessions.

---

## Stanford IRIS Lab Context

Organization: https://github.com/stanford-iris-lab
Focus: Reinforcement learning, offline RL, robot learning, AI agents
Notable repos: d5rl (offline RL), vlm-pc, batch-exploration
This artifact (197 stars as of March 2026) is their most popular repo.
