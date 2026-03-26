"""Agent orchestration: defines the order, dependencies, and learning loop.

The research pipeline follows a DAG where each phase feeds the next,
and learnings flow backwards to improve earlier agents.

PHASE ORDER:
  1. RESEARCHER   → Gather intel (top notebooks, discussions, papers)
  2. EDA          → Understand data deeply (distributions, correlations, outliers)
  3. FEATURE ENG  → Create features informed by research + EDA
  4. MODEL TRAIN  → Train diverse models on engineered features
  5. ENSEMBLE     → Blend models for maximum score
  6. SUBMIT       → Generate and submit predictions
  7. LEARN        → Analyze results, update agent knowledge, loop back

LEARNING FLOW (backwards):
  Submit results → update ENSEMBLE knowledge
  Ensemble insights → update MODEL TRAIN knowledge (which models ensemble well)
  Model importance → update FEATURE ENG knowledge (which features matter)
  Feature performance → update EDA knowledge (what patterns to look for)
  All learnings → update RESEARCHER knowledge (what works for this competition)
"""

from __future__ import annotations

import json
import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AgentPhase:
    """A phase in the research pipeline."""
    name: str
    agent: str  # Agent file name (without .md)
    order: int
    depends_on: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    learnings_feed_to: list[str] = field(default_factory=list)


# Define the canonical agent ordering
PIPELINE_PHASES = [
    AgentPhase(
        name="research",
        agent="kaggle-researcher",
        order=1,
        depends_on=[],
        outputs=["research/findings.md", "research/top_approaches.json"],
        learnings_feed_to=["kaggle-eda", "kaggle-feature-engineer", "kaggle-model-trainer"],
    ),
    AgentPhase(
        name="eda",
        agent="kaggle-eda",
        order=2,
        depends_on=["research"],
        outputs=["eda/report.md", "eda/correlations.json", "eda/feature_analysis.json"],
        learnings_feed_to=["kaggle-feature-engineer", "kaggle-model-trainer"],
    ),
    AgentPhase(
        name="feature_engineering",
        agent="kaggle-feature-engineer",
        order=3,
        depends_on=["research", "eda"],
        outputs=["features/pipeline.py", "features/importance.json"],
        learnings_feed_to=["kaggle-model-trainer", "kaggle-ensembler"],
    ),
    AgentPhase(
        name="model_training",
        agent="kaggle-model-trainer",
        order=4,
        depends_on=["feature_engineering"],
        outputs=["experiments/results.tsv", "checkpoints/"],
        learnings_feed_to=["kaggle-ensembler", "kaggle-feature-engineer"],
    ),
    AgentPhase(
        name="ensemble",
        agent="kaggle-ensembler",
        order=5,
        depends_on=["model_training"],
        outputs=["submissions/", "ensemble/weights.json"],
        learnings_feed_to=["kaggle-model-trainer", "kaggle-orchestrator"],
    ),
    AgentPhase(
        name="submit_and_learn",
        agent="kaggle-orchestrator",
        order=6,
        depends_on=["ensemble"],
        outputs=["learnings/iteration_N.json"],
        learnings_feed_to=["kaggle-researcher"],  # Loop back
    ),
]


@dataclass
class Learning:
    """A single learning/insight discovered during the loop."""
    timestamp: str
    source_agent: str
    phase: str
    iteration: int
    category: str  # "feature", "model", "ensemble", "data", "strategy"
    insight: str
    impact: str  # "high", "medium", "low"
    evidence: str  # e.g., "CV improved from 0.843 to 0.856"
    action: str  # What should change based on this learning
    applied_to: list[str] = field(default_factory=list)  # Which agents were updated


class LearningStore:
    """Persistent store for learnings across iterations.

    Learnings are saved to disk and can be loaded to update agent definitions.
    """

    def __init__(self, store_path: str | Path = "learnings"):
        self.path = Path(store_path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.learnings_file = self.path / "all_learnings.json"
        self._learnings: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if self.learnings_file.exists():
            with open(self.learnings_file) as f:
                return json.load(f)
        return []

    def _save(self) -> None:
        with open(self.learnings_file, "w") as f:
            json.dump(self._learnings, f, indent=2)

    def add(self, learning: Learning) -> None:
        """Add a new learning and persist it."""
        entry = {
            "timestamp": learning.timestamp,
            "source_agent": learning.source_agent,
            "phase": learning.phase,
            "iteration": learning.iteration,
            "category": learning.category,
            "insight": learning.insight,
            "impact": learning.impact,
            "evidence": learning.evidence,
            "action": learning.action,
            "applied_to": learning.applied_to,
        }
        self._learnings.append(entry)
        self._save()

        # Also save per-iteration file
        iter_file = self.path / f"iteration_{learning.iteration:03d}.json"
        iter_learnings = [l for l in self._learnings if l["iteration"] == learning.iteration]
        with open(iter_file, "w") as f:
            json.dump(iter_learnings, f, indent=2)

    def get_for_agent(self, agent_name: str) -> list[dict]:
        """Get all learnings relevant to a specific agent."""
        relevant = []
        for l in self._learnings:
            if agent_name in l.get("applied_to", []) or l["source_agent"] == agent_name:
                relevant.append(l)
        return relevant

    def get_by_category(self, category: str) -> list[dict]:
        """Get learnings by category."""
        return [l for l in self._learnings if l["category"] == category]

    def get_high_impact(self) -> list[dict]:
        """Get high-impact learnings."""
        return [l for l in self._learnings if l["impact"] == "high"]

    def generate_agent_context(self, agent_name: str) -> str:
        """Generate a context block to inject into an agent's prompt.

        This is the key mechanism for evolving agents: learnings are
        formatted as additional instructions that get prepended to
        the agent's system prompt.
        """
        relevant = self.get_for_agent(agent_name)
        if not relevant:
            return ""

        lines = ["## Learnings from Previous Iterations", ""]
        for l in relevant[-20:]:  # Last 20 learnings
            lines.append(f"- [{l['impact'].upper()}] {l['insight']}")
            lines.append(f"  Evidence: {l['evidence']}")
            lines.append(f"  Action: {l['action']}")
            lines.append("")

        return "\n".join(lines)

    def summary(self) -> str:
        """Human-readable summary."""
        if not self._learnings:
            return "No learnings recorded yet."

        by_cat = {}
        for l in self._learnings:
            cat = l["category"]
            by_cat[cat] = by_cat.get(cat, 0) + 1

        high = len([l for l in self._learnings if l["impact"] == "high"])
        iters = set(l["iteration"] for l in self._learnings)

        lines = [
            f"Total learnings: {len(self._learnings)}",
            f"Iterations: {len(iters)}",
            f"High-impact: {high}",
            f"By category: {by_cat}",
        ]
        return "\n".join(lines)


def update_agent_file(
    agent_path: Path,
    learnings: list[dict],
) -> None:
    """Update an agent's .md file with new learnings.

    Appends learnings to the agent definition so it evolves over time.
    The learnings section is replaced entirely on each update.
    """
    content = agent_path.read_text(encoding="utf-8")

    # Remove old learnings section if present
    marker_start = "<!-- LEARNINGS START -->"
    marker_end = "<!-- LEARNINGS END -->"
    if marker_start in content:
        before = content[:content.index(marker_start)]
        after = content[content.index(marker_end) + len(marker_end):]
        content = before.rstrip() + "\n\n" + after.lstrip()

    # Build new learnings section
    if learnings:
        learning_lines = [
            "",
            marker_start,
            "## Accumulated Learnings (Auto-Updated)",
            "",
        ]
        for l in learnings[-30:]:  # Keep last 30
            learning_lines.append(f"### [{l['impact'].upper()}] {l['category']}: {l['insight']}")
            learning_lines.append(f"- Evidence: {l['evidence']}")
            learning_lines.append(f"- Action: {l['action']}")
            learning_lines.append(f"- Iteration: {l['iteration']} ({l['timestamp'][:10]})")
            learning_lines.append("")
        learning_lines.append(marker_end)

        content = content.rstrip() + "\n" + "\n".join(learning_lines) + "\n"

    agent_path.write_text(content, encoding="utf-8")


def update_skill_file(
    skill_path: Path,
    learnings: list[dict],
) -> None:
    """Update a competition skill file with learnings.

    Similar to agent update but focused on competition-specific knowledge.
    """
    content = skill_path.read_text(encoding="utf-8")

    marker_start = "<!-- COMPETITION LEARNINGS START -->"
    marker_end = "<!-- COMPETITION LEARNINGS END -->"
    if marker_start in content:
        before = content[:content.index(marker_start)]
        after = content[content.index(marker_end) + len(marker_end):]
        content = before.rstrip() + "\n\n" + after.lstrip()

    if learnings:
        lines = [
            "",
            marker_start,
            "### Discovered Insights (Auto-Updated)",
            "",
        ]

        # Group by category
        by_cat: dict[str, list] = {}
        for l in learnings:
            cat = l["category"]
            if cat not in by_cat:
                by_cat[cat] = []
            by_cat[cat].append(l)

        for cat, cat_learnings in sorted(by_cat.items()):
            lines.append(f"#### {cat.title()}")
            for l in cat_learnings[-10:]:
                lines.append(f"- {l['insight']} (impact: {l['impact']}, evidence: {l['evidence']})")
            lines.append("")

        lines.append(marker_end)
        content = content.rstrip() + "\n" + "\n".join(lines) + "\n"

    skill_path.write_text(content, encoding="utf-8")


def propagate_learnings(
    learning_store: LearningStore,
    agents_dir: Path,
    skills_dir: Path | None = None,
) -> dict[str, int]:
    """Propagate learnings to all relevant agent and skill files.

    Returns dict of {agent_name: n_learnings_applied}.
    """
    updated = {}

    # Update agent files
    for agent_file in agents_dir.glob("kaggle-*.md"):
        agent_name = agent_file.stem
        relevant = learning_store.get_for_agent(agent_name)
        if relevant:
            update_agent_file(agent_file, relevant)
            updated[agent_name] = len(relevant)

    # Update skill files
    if skills_dir:
        for skill_file in skills_dir.glob("**/*.md"):
            all_learnings = learning_store._learnings
            if all_learnings:
                update_skill_file(skill_file, all_learnings)
                updated[skill_file.stem] = len(all_learnings)

    return updated
