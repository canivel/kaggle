"""KAOS-orchestrated opus-4-8 semantic supervisor for ARC-AGI-3 submissions.

Deeper than preflight.py. Spawns a KAOS agent (claude-opus-4-8 via the
kaos.yaml router default) that:

  1. Reads the candidate kernel + baseline kernel from /f/kaggle/arc-prize-2026
     via shell_exec calls to `kaggle kernels pull`.
  2. Compares the agent code semantically (not just structurally):
       - module-level side effects (threads, network calls, mutable globals)
       - import-time blocking operations
       - missing __init__.py imports the rerun expects
       - .env keys the rerun cell needs (SCHEME/HOST/PORT/...)
  3. Runs a 5-minute LOCAL eval against the public games to confirm
     the agent doesn't crash on real frames.
  4. Returns a JSON verdict: ALLOW / WARN / BLOCK with rationale.

USE: Called by daily_submit.py when preflight passes with WARNs, or on
first submission of a new kernel slug. Optional but recommended.

NOTE: KAOS's agent VFS is sandboxed — but shell_exec is NOT, so the
agent can use kaggle CLI + python on the real filesystem. We pass the
candidate kernel slug in the task prompt; the agent fetches it via
shell_exec and does its own analysis.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAOS_DIR = ROOT.parent / "kaos"

TASK_TEMPLATE = """You are an ARC-AGI-3 Kaggle submission supervisor running on claude-opus-4-8.

Your job: rigorously validate that the candidate kernel `{kernel}` will not ERROR
on Kaggle's competition rerun. Five consecutive submissions ERRORed because the
rerun cell wrote the wrong `agents/__init__.py` and `.env` files. We just
fixed that root cause and built a deterministic preflight.py. Your job is to
catch SEMANTIC bugs preflight cannot.

Use `shell_exec` to:
  1. Pull the candidate kernel:
       `kaggle kernels pull {kernel} -p /tmp/cand`
  2. Pull the known-working baseline:
       `kaggle kernels pull canivel/arc3-baseline -p /tmp/base`
  3. Extract the my_agent.py cell from both notebooks. Diff them.
  4. Run the deterministic preflight:
       `cd /f/kaggle/arc-prize-2026 && uv run python scripts/preflight.py --kernel {kernel} --json-only`
     Report its verdict.
  5. (Optional) Run a 3-game local eval to confirm no crashes:
       `cd /f/kaggle/arc-prize-2026 && uv run python eval_harness.py --agent <path-to-extracted-agent> --budget 200 --wall-s 30 --bfs-s 5 --games ft09,lp85,sb26 --out /tmp/sup_eval.json`

ANTI-PATTERNS TO FLAG (BLOCK):
  - Module-level `threading.Thread().start()` or network calls in agent imports
  - Long-running operations in agent __init__ (>5s)
  - Mutable module-level state that persists across MyAgent instances
    without proper isolation (could cause cross-game pollution in swarm)
  - Synchronous filesystem writes from pick_action paths
  - Any `print()` to stdout in hot loops (slow + log spam)

ANTI-PATTERNS TO FLAG (WARN):
  - Large module-level constants that bloat per-process memory
  - Imports of heavy packages (torch, transformers) when not needed
  - Hardcoded paths that won't exist in Kaggle's environment

OUTPUT JSON (last line of your reply, parseable):
  {{"verdict": "ALLOW" | "WARN" | "BLOCK",
    "preflight_verdict": "<from preflight.py>",
    "semantic_findings": [{{"severity": "BLOCK"|"WARN"|"INFO", "issue": "..."}}],
    "eval_results": {{"games": N, "errors": K, "rhae": float}}}}

DO NOT fabricate findings. If you cannot run a check, report it honestly.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--name", default="arc-submit-supervisor")
    ap.add_argument("--timeout", type=int, default=1200)
    args = ap.parse_args()

    task = TASK_TEMPLATE.format(kernel=args.kernel)

    # Spawn KAOS agent. KAOS uses kaos.yaml's claude-opus model
    # (set to claude-opus-4-8 in this campaign per memory feedback).
    cmd = ["uv", "run", "kaos", "run",
           "-n", args.name,
           "-m", "claude-opus",
           task]
    proc = subprocess.run(
        cmd, cwd=str(KAOS_DIR),
        capture_output=True, text=True, timeout=args.timeout,
    )
    out = proc.stdout

    # KAOS prints "Spawned agent: <id>". Capture the id.
    m = re.search(r"Spawned agent:\s*(\S+)", out)
    if not m:
        print(json.dumps({"verdict": "BLOCK", "reason": "kaos-run-failed",
                          "stdout_tail": out[-500:], "stderr_tail": proc.stderr[-500:]}))
        sys.exit(1)
    agent_id = m.group(1)

    # Poll status via kaos ls
    import time
    deadline = time.monotonic() + args.timeout
    final = None
    while time.monotonic() < deadline:
        ls = subprocess.run(
            ["uv", "run", "kaos", "ls"],
            cwd=str(KAOS_DIR), capture_output=True, text=True, timeout=60,
        )
        try:
            agents = json.loads(ls.stdout)
        except Exception:
            time.sleep(30)
            continue
        my = [a for a in agents if a.get("agent_id") == agent_id]
        if not my:
            time.sleep(30)
            continue
        status = my[0].get("status")
        if status in ("completed", "failed", "error"):
            final = status
            break
        time.sleep(30)

    if final is None:
        print(json.dumps({"verdict": "BLOCK", "reason": "kaos-supervisor-timeout"}))
        sys.exit(1)

    # Fetch the agent's final reply via kaos logs
    logs = subprocess.run(
        ["uv", "run", "kaos", "logs", agent_id],
        cwd=str(KAOS_DIR), capture_output=True, text=True, timeout=60,
    )
    try:
        log_obj = json.loads(logs.stdout)
        msgs = log_obj.get("conversation", [])
        assistant_msgs = [m for m in msgs if m.get("role") == "assistant"]
        last = assistant_msgs[-1].get("content", "") if assistant_msgs else ""
    except Exception:
        last = ""

    # Parse the final JSON line from the agent's reply
    verdict_json = {"verdict": "WARN", "reason": "no-json-from-supervisor",
                    "raw": last[-500:]}
    for line in reversed(last.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                verdict_json = json.loads(line)
                break
            except Exception:
                continue

    print(json.dumps(verdict_json, indent=2))
    sys.exit(0 if verdict_json.get("verdict") != "BLOCK" else 1)


if __name__ == "__main__":
    main()
