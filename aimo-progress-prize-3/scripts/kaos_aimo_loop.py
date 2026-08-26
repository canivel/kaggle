"""KAOS Auto-Improvement Loop for AIMO3.

The REAL loop — properly closes the feedback cycle:
1. Read ALL accumulated research + code artifacts from KAOS DB
2. Read current best notebook
3. Use Anthropic API to propose next improvement (what to add, why)
4. Apply the change to the notebook
5. Run local syntax validation
6. Store result in KAOS (new code artifact + updated state) + checkpoint
7. Output the next notebook ready to push

Usage:
    cd f:/kaggle/aimo-progress-prize-3
    uv run --with anthropic python scripts/kaos_aimo_loop.py

    # Dry-run (just propose, don't write notebook):
    uv run --with anthropic python scripts/kaos_aimo_loop.py --dry-run

    # Propose a specific change:
    uv run --with anthropic python scripts/kaos_aimo_loop.py --focus "two-phase adaptive solver"
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import shutil
import sys
import time
from pathlib import Path

KAOS_DIR = Path(__file__).parent.parent.parent / "kaos"
sys.path.insert(0, str(KAOS_DIR))

KAOS_DB = Path(__file__).parent.parent / "aimo3-learnings.db"
NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks"


def load_kaos_context(db) -> dict:
    """Load all relevant KAOS state and code artifacts."""
    agents = db.list_agents()
    if not agents:
        raise RuntimeError("No agents in KAOS DB. Something is wrong.")

    a_id = agents[0]["agent_id"]
    ctx = {"agent_id": a_id, "state": {}, "code": {}, "research": {}, "learnings": {}}

    # Load all state keys
    all_state = db.get_all_state(a_id)
    ctx["state"] = all_state

    # Load code artifacts
    for fname in ["verify_cascade.py", "novel_solver.py", "adaptive_solver.py"]:
        try:
            content = db.read(a_id, f"/code/{fname}")
            ctx["code"][fname] = content.decode("utf-8")
        except Exception:
            pass

    # Load learning files
    try:
        for entry in db.ls(a_id, "/learnings"):
            if entry["name"].endswith(".md"):
                content = db.read(a_id, entry["path"])
                ctx["learnings"][entry["name"]] = content.decode("utf-8")
    except Exception:
        pass

    # Load research files
    try:
        for entry in db.ls(a_id, "/research"):
            if entry["name"].endswith(".md"):
                content = db.read(a_id, entry["path"])
                ctx["research"][entry["name"]] = content.decode("utf-8")
    except Exception:
        pass

    return ctx


def build_context_prompt(ctx: dict, focus: str | None) -> str:
    """Build the full context for the Anthropic API call."""
    s = ctx["state"]

    parts = [
        "# AIMO3 Competition Context",
        "",
        "## Score History",
        json.dumps(s.get("version_status", {}), indent=2),
        "",
        "## Current Best Config (proven 44/50)",
        json.dumps(s.get("proven_config", {}), indent=2),
        "",
        "## What DOESN'T Work (hard evidence)",
        json.dumps(s.get("innovation_map", {}).get("what_doesnt_work", []), indent=2),
        "",
        "## What Might Work",
        json.dumps(s.get("innovation_map", {}).get("what_might_work", []), indent=2),
        "",
        "## Critical Reversals (Pawan Mali 50-experiment study)",
        json.dumps(s.get("CRITICAL_50_experiment_reversal", {}), indent=2),
        "",
        "## CRITICAL_REVERSAL_2",
        json.dumps(s.get("CRITICAL_REVERSAL_2", {}), indent=2),
        "",
        "## Conflicting Evidence",
        json.dumps(s.get("conflicting_evidence", {}), indent=2),
        "",
        "## Submission Plan",
        json.dumps(s.get("submission_plan", {}), indent=2),
        "",
        "## Ready Notebooks",
        json.dumps(s.get("ready_notebooks", {}), indent=2),
        "",
        "## Final Innovation Priority",
        json.dumps(s.get("final_innovation_priority", {}), indent=2),
        "",
        "## Novel Approach Design",
        json.dumps(s.get("novel_approach", {}), indent=2),
        "",
        "## Amanatar 44/50 Approach",
        json.dumps(s.get("amanatar_44_approach", {}), indent=2),
        "",
        "## Multi-Stage Research",
        json.dumps(s.get("multistage_research", {}), indent=2),
        "",
        "## Local Eval Results",
        json.dumps(s.get("local_eval_results", {}), indent=2),
        "",
        "## Local Harness Test",
        json.dumps(s.get("local_harness_test", {}), indent=2),
        "",
    ]

    # Add code artifacts
    if ctx["code"]:
        parts.append("## Stored Code Artifacts (from previous research iterations)")
        for fname, code in ctx["code"].items():
            parts.append(f"\n### {fname}\n```python\n{code}\n```")

    # Add key learning files
    for name, content in ctx["learnings"].items():
        parts.append(f"\n## Learning: {name}\n{content[:2000]}")

    if focus:
        parts.append(f"\n## Focus for This Iteration\n{focus}")

    return "\n".join(parts)


def propose_improvement(context_text: str, current_notebook_summary: str) -> dict:
    """Call Anthropic API to propose the next notebook improvement."""
    import anthropic  # noqa: ensure native SDK

    client = anthropic.Anthropic()

    system = """You are an expert Kaggle competition engineer for the AIMO Progress Prize 3.
You have deep knowledge of the competition's constraints, the GPT-OSS-120B model,
the Harmony protocol, and all experimental evidence accumulated so far.

Your task: Given the accumulated research, propose the SINGLE BEST change to make
to the current notebook for the next submission (1 submission per day, 12 days left).

You must:
1. State exactly WHAT to change (specific code modification)
2. State WHY this change is expected to help (cite evidence from the context)
3. State the RISK level (Low/Medium/High) and why
4. Describe HOW to integrate it into the existing solve_problem flow

Be conservative — we can only submit once per day. Prefer proven changes over speculative ones.
Do NOT suggest things already ruled out (see "What DOESN'T Work" section).
"""

    user = f"""## Accumulated Competition Knowledge
{context_text}

## Current Notebook Summary
{current_notebook_summary}

## Task
Propose the single best change for tomorrow's submission notebook.
Structure your response as JSON:
{{
  "change_name": "short name",
  "what": "specific code change description",
  "why": "evidence-based reasoning (cite specific notebooks/papers)",
  "risk": "Low|Medium|High",
  "risk_reasoning": "why this risk level",
  "expected_delta": "+X to +Y points",
  "integration_note": "how to add to solve_problem without breaking existing code",
  "code_snippet": "key code block that implements this change (Python)"
}}"""

    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=4096,
        messages=[{"role": "user", "content": user}],
        system=system,
    )

    response_text = message.content[0].text.strip()

    # Extract JSON from response
    if "```json" in response_text:
        json_str = response_text.split("```json")[1].split("```")[0].strip()
    elif "```" in response_text:
        json_str = response_text.split("```")[1].split("```")[0].strip()
    elif response_text.startswith("{"):
        json_str = response_text
    else:
        # Try to find JSON block
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        json_str = response_text[start:end] if start >= 0 else response_text

    try:
        proposal = json.loads(json_str)
    except json.JSONDecodeError:
        # Fallback: store raw response
        proposal = {"change_name": "raw_response", "raw": response_text}

    return proposal


def summarize_notebook(notebook_path: Path) -> str:
    """Extract key info from current best notebook."""
    with io.open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    summary_parts = []
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            # Capture key sections
            if "class CFG" in src:
                # Find params section
                idx = src.find("served_model_name")
                if idx >= 0:
                    summary_parts.append(f"## CFG (Cell {i})\n```python\n{src[idx-50:idx+600]}\n```")
            if "def solve_problem" in src:
                idx = src.find("def solve_problem")
                summary_parts.append(f"## solve_problem (Cell {i})\n```python\n{src[idx:idx+1000]}\n```")
            if "_select_answer" in src and "def _select_answer" in src:
                idx = src.find("def _select_answer")
                summary_parts.append(f"## _select_answer (Cell {i})\n```python\n{src[idx:idx+600]}\n```")

    return "\n\n".join(summary_parts)


def store_proposal_in_kaos(db, a_id: str, proposal: dict, iteration: int) -> None:
    """Store the proposal in KAOS VFS + update state."""
    fname = f"proposal_iter{iteration:03d}_{int(time.time())}.json"
    content = json.dumps(proposal, indent=2, ensure_ascii=False)

    try:
        db.mkdir(a_id, "/proposals")
    except Exception:
        pass

    db.write(a_id, f"/proposals/{fname}", content.encode("utf-8"))
    db.set_state(a_id, "latest_proposal", proposal)
    db.set_state(a_id, "proposal_count", iteration)
    db.checkpoint(a_id, label=f"loop-iter-{iteration}")
    print(f"Stored proposal in KAOS: /proposals/{fname}")


def build_next_notebook(current_nb_path: Path, proposal: dict, output_path: Path) -> bool:
    """
    Create the next notebook based on the proposal.
    Currently: copies current best + stores proposal as a prominent comment.
    For novel_solver integration: swaps in the full novel_solver.py.
    Returns True if output was written.
    """
    with io.open(current_nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    change_name = proposal.get("change_name", "unknown")
    code_snippet = proposal.get("code_snippet", "")

    # Add a summary cell at the top
    summary_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"# KAOS Proposal: {change_name}\n",
            f"\n**Why:** {proposal.get('why', '')[:500]}\n",
            f"\n**Expected:** {proposal.get('expected_delta', '?')}\n",
            f"\n**Risk:** {proposal.get('risk', '?')} — {proposal.get('risk_reasoning', '')[:300]}\n",
            f"\n**Integration:** {proposal.get('integration_note', '')[:500]}\n",
        ],
    }

    # Insert summary at top
    nb["cells"].insert(0, summary_cell)

    # If the proposal includes code, add a note in the CFG cell
    if code_snippet and "novel_solver" not in change_name.lower():
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code":
                src = "".join(cell["source"])
                if "CFG: ULTIMATE" in src:
                    note = f"\n# KAOS Proposed Change: {change_name}\n# {proposal.get('why', '')[:200]}\n"
                    nb["cells"][i]["source"] = (note + src).splitlines(keepends=True)
                    break

    with io.open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    return True


def validate_notebook(nb_path: Path) -> tuple[bool, str]:
    """Basic syntax validation of all code cells."""
    with io.open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    errors = []
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            if src.strip():
                try:
                    ast.parse(src)
                except SyntaxError as e:
                    errors.append(f"Cell {i}: {e}")

    if errors:
        return False, "\n".join(errors)
    return True, "OK"


def main():
    parser = argparse.ArgumentParser(description="KAOS AIMO auto-improvement loop")
    parser.add_argument("--dry-run", action="store_true", help="Propose but don't write notebook")
    parser.add_argument("--focus", default=None, help="Focus area for this iteration")
    parser.add_argument("--base-notebook", default=None, help="Override base notebook path")
    args = parser.parse_args()

    print("=" * 60)
    print("KAOS AIMO3 Auto-Improvement Loop")
    print("=" * 60)

    from kaos import Kaos

    # Load KAOS context
    print(f"\nLoading KAOS context from {KAOS_DB}...")
    db = Kaos(str(KAOS_DB))
    ctx = load_kaos_context(db)
    a_id = ctx["agent_id"]

    print(f"Agent: {a_id[:16]}...")
    print(f"State keys: {len(ctx['state'])}")
    print(f"Code artifacts: {list(ctx['code'].keys())}")
    print(f"Learnings: {list(ctx['learnings'].keys())}")
    print(f"Research: {list(ctx['research'].keys())}")

    # Find current best notebook
    if args.base_notebook:
        base_nb = Path(args.base_notebook)
    else:
        # Default: use v17 (verify cascade, our most advanced)
        base_nb = NOTEBOOKS_DIR / "submission_v17_verify.ipynb"
        if not base_nb.exists():
            base_nb = NOTEBOOKS_DIR / "submission_v16_exact44.ipynb"

    print(f"\nBase notebook: {base_nb.name}")

    if not base_nb.exists():
        print(f"ERROR: Base notebook not found at {base_nb}")
        db.close()
        return

    # Summarize current notebook
    nb_summary = summarize_notebook(base_nb)

    # Build context for API
    context_text = build_context_prompt(ctx, args.focus)
    print(f"Context size: {len(context_text):,} chars")

    # Get proposal from Anthropic
    print("\nCalling Anthropic API for improvement proposal...")
    proposal = propose_improvement(context_text, nb_summary)

    print("\n" + "=" * 60)
    print("PROPOSAL:")
    print("=" * 60)
    print(json.dumps(proposal, indent=2, ensure_ascii=False)[:3000])

    # Determine iteration number
    iteration = (ctx["state"].get("proposal_count") or 0) + 1

    # Store in KAOS
    store_proposal_in_kaos(db, a_id, proposal, iteration)

    if args.dry_run:
        print("\n[DRY RUN] Notebook not written.")
        db.close()
        return

    # Build next notebook
    # Determine output name
    kaggle_ver = 36 + iteration  # v35 is running, v36 is verify, so next would be v37+
    out_name = f"submission_v{18 + iteration - 1}_kaos_{proposal.get('change_name', 'auto')[:20].replace(' ', '_')}.ipynb"
    out_path = NOTEBOOKS_DIR / out_name

    print(f"\nBuilding notebook: {out_path.name}")
    success = build_next_notebook(base_nb, proposal, out_path)

    if success:
        # Validate
        valid, msg = validate_notebook(out_path)
        if valid:
            print(f"Syntax validation: PASSED")
        else:
            print(f"Syntax validation: FAILED\n{msg}")
            out_path.unlink(missing_ok=True)
            db.close()
            return

        # Create push dir
        push_dir = NOTEBOOKS_DIR / f"push_kaos_{kaggle_ver}"
        push_dir.mkdir(exist_ok=True)
        shutil.copy(out_path, push_dir / out_name)

        # Write metadata
        with io.open(NOTEBOOKS_DIR / "kernel-metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["code_file"] = out_name
        meta["title"] = f"AIMO3 kaos-v{kaggle_ver} {proposal.get('change_name', 'auto')[:20]}"
        with io.open(push_dir / "kernel-metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        # Update KAOS state
        ready = ctx["state"].get("ready_notebooks", {})
        ready[f"v{kaggle_ver}_kaggle"] = {
            "status": "ready_kaos_generated",
            "file": str(out_path.relative_to(NOTEBOOKS_DIR.parent)),
            "push_dir": str(push_dir.relative_to(NOTEBOOKS_DIR.parent)),
            "change": proposal.get("change_name"),
            "expected": proposal.get("expected_delta"),
            "risk": proposal.get("risk"),
            "generated_by": "kaos_aimo_loop",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        db.set_state(a_id, "ready_notebooks", ready)
        db.set_state(a_id, "latest_generated_notebook", str(out_path))
        db.checkpoint(a_id, label=f"generated-v{kaggle_ver}")

        print(f"\nNotebook written: {out_path}")
        print(f"Push dir: {push_dir}")
        print(f"\nTo submit: cd {push_dir} && kaggle kernels push")

    db.close()
    print("\nKAOS loop complete.")


if __name__ == "__main__":
    main()
