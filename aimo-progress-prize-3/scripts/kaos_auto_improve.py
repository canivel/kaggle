"""Kaos Auto-Improvement Loop for AIMO3.

Uses kaos DB to track all versions, learnings, and improvements.
Proposes new improvements based on accumulated knowledge.
Runs continuously, creating new notebook variants.

Usage:
    uv run python scripts/kaos_auto_improve.py
"""

from __future__ import annotations
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, "f:/kaggle/kaos")
from kaos import Kaos


def load_all_learnings(db: Kaos, agent_id: str) -> dict:
    """Load all state from kaos DB."""
    keys = ['version_status', 'proven_config', 'fatal_mistakes', 'paper_2603_27844',
            'v25_improvements', 'huikang_model', 'answer_extraction_research', 'next_submit']
    state = {}
    for key in keys:
        val = db.get_state(agent_id, key)
        if val is not None:
            state[key] = val
    return state


def load_learnings_files(db: Kaos, agent_id: str) -> list[str]:
    """Load all learning markdown files."""
    files = []
    try:
        entries = db.ls(agent_id, '/learnings')
        for entry in entries:
            if entry['name'].endswith('.md'):
                content = db.read(agent_id, entry['path'])
                files.append(f"=== {entry['name']} ===\n{content.decode()}")
    except:
        pass
    return files


def generate_improvement_report(state: dict, learnings: list[str]) -> str:
    """Generate a report of what we know and what to try next."""
    report = []
    report.append("# AIMO3 Auto-Improvement Report")
    report.append(f"\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")

    # Score history
    report.append("\n## Score History")
    vs = state.get('version_status', {})
    for k, v in sorted(vs.items()):
        report.append(f"  {k}: {v}")

    # Paper findings
    paper = state.get('paper_2603_27844', {})
    if paper:
        report.append("\n## Key Paper Findings (arxiv 2603.27844)")
        kf = paper.get('key_findings', {})
        report.append(f"  Optimal temperature: {kf.get('optimal_temperature')}")
        report.append(f"  Temperature ablation: {kf.get('temperature_ablation')}")
        report.append(f"  Attempts ceiling: {kf.get('attempts_ceiling')}")
        report.append(f"  Per-attempt accuracy: {kf.get('per_attempt_accuracy')}")
        report.append(f"  Ceiling: {kf.get('ceiling_with_perfect_orchestration')}")

        top3 = paper.get('top_3_unimplemented', [])
        if top3:
            report.append("\n## Top 3 Unimplemented Improvements")
            for i, item in enumerate(top3, 1):
                report.append(f"  {i}. {item}")

    # Current improvements
    v25 = state.get('v25_improvements', {})
    if v25:
        report.append("\n## Implemented Improvements (v25+)")
        for k, v in v25.items():
            report.append(f"  {k}: {v}")

    # Learnings
    report.append(f"\n## Detailed Learnings ({len(learnings)} files)")
    for l in learnings:
        report.append(l)

    # Next actions
    report.append("\n## Priority Actions")
    report.append("  1. [HIGHEST] Multi-turn follow-up when no \\boxed{} answer")
    report.append("  2. [HIGH] Answer verification step at T=0.0")
    report.append("  3. [MEDIUM] Test max_model_len=81920 vs 65536")
    report.append("  4. [MEDIUM] Test min_p=0.01 vs 0.02")
    report.append("  5. [LOW] VOI-based early stopping (Bayesian)")

    return "\n".join(report)


def main():
    db = Kaos("aimo3-learnings.db")
    agents = db.list_agents()
    if not agents:
        print("No agents in kaos DB. Run the main pipeline first.")
        db.close()
        return

    agent_id = agents[0]['agent_id']
    print(f"Agent: {agent_id}")

    # Load all state
    state = load_all_learnings(db, agent_id)
    learnings = load_learnings_files(db, agent_id)

    # Generate report
    report = generate_improvement_report(state, learnings)
    # Print safely on Windows
    print(report.encode('ascii', errors='replace').decode('ascii'))

    # Save report
    report_path = Path("docs/auto_improvement_report.md")
    report_path.write_text(report, encoding='utf-8')
    print(f"\nReport saved to {report_path}")

    # Also store in kaos
    db.write(agent_id, "/reports/auto_improvement.md", report.encode())

    # Track iteration
    iteration = db.get_state(agent_id, 'improvement_iteration') or 0
    db.set_state(agent_id, 'improvement_iteration', iteration + 1)
    db.set_state(agent_id, 'last_report_time', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))

    print(f"\nIteration {iteration + 1} complete.")
    print(f"Kaos DB: aimo3-learnings.db")
    print(f"Agent: {agent_id}")
    print(f"State keys: {list(state.keys())}")
    print(f"Learning files: {len(learnings)}")

    db.close()


if __name__ == "__main__":
    main()
