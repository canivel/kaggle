"""Adversarial 5-PhD panel round, run through KAOS (agent_sdk provider, claude-fable-5).

Usage:
  uv run python scripts/panel_round.py --round 1 --proposal learnings/winning_solution_v1.md
  uv run python scripts/panel_round.py --round 2 --proposal learnings/winning_solution_v2.md \
      --prior-dir learnings/panel/round1

Spawns 5 KAOS agents in parallel (one per reviewer persona), waits for all,
extracts their reviews, writes learnings/panel/round{N}/{reviewer}.md and a
summary JSON. Pass criterion (enforced by the caller): >=4/5 ACCEPT and zero
unresolved FATAL objections.

The panel is intentionally adversarial: reviewers are instructed that a
clean first-round pass indicates review failure.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAOS_DIR = ROOT.parent / "kaos"

REVIEWERS = {
    "rl-planning": (
        "Professor of Reinforcement Learning and Planning (MCTS, model-based RL, "
        "exploration theory; 20 years; famously skeptical of under-specified search claims)"
    ),
    "llm-agents": (
        "Professor of LLM Agents and Scaffolding (tool-use, agentic harnesses, "
        "prompt-based control of foundation models; reviews for NeurIPS/ICLR; "
        "allergic to 'we will prompt it better' hand-waving)"
    ),
    "prog-synthesis": (
        "Professor of Program Synthesis and Neurosymbolic AI (inductive program "
        "synthesis, world models as code, verification; insists on falsifiable "
        "synthesis-quality metrics)"
    ),
    "methodology": (
        "Professor of Empirical ML Methodology and Statistics (experimental design, "
        "multiple-comparisons, noise-band inference; rejects any plan that draws "
        "conclusions from single noisy samples)"
    ),
    "systems": (
        "Professor of ML Systems (GPU inference, vLLM serving, wall-clock budgets, "
        "quota economics; kills plans that don't fit the compute envelope)"
    ),
}

REVIEW_TEMPLATE = """You are {persona}.

You are reviewer #{idx} on a 5-person adversarial review panel evaluating a competition
strategy proposal for ARC-AGI-3 (Kaggle, deadline Nov 2 2026).

GROUND-TRUTH LEADERBOARD STATE (refreshed {lb_date} from the live Kaggle API; the
draw-by-draw submission ledger is at {lb_artifact}; treat THESE numbers as canonical —
any different numbers you may remember from earlier rounds were a stale briefing):
{lb_state}

PANEL RULES — READ CAREFULLY:
- You are UNBIASED and RIGOROUS. You have no stake in the proposal passing.
- Do NOT be agreeable. A proposal that passes round 1 unchanged indicates REVIEW FAILURE.
- Attack the weakest LOAD-BEARING assumptions, not typos.
- Every objection must be specific and actionable (not "needs more detail").
- Distinguish severity: FATAL (invalidates the plan), MAJOR (must fix before execution),
  MINOR (should fix).
- You review ONLY within your expertise; note explicitly what you cannot judge.
- If evidence is missing for a claim, say exactly what experiment/data would supply it.
{prior_block}
THE PROPOSAL (sha256 of the full document: {doc_sha}; full length {doc_len} chars;
if the text below does not end with the literal line "## END OF PROPOSAL ##",
your copy is TRUNCATED — file that as an objection and review only what you see):
=====================================================================
{proposal}
=====================================================================

OUTPUT FORMAT (exactly this structure):
## Summary (2 sentences)
## Objections
For each: [SEVERITY] title — body (2-5 sentences, specific)
(minimum 3 objections in round 1; in later rounds, review your prior objections'
resolution first, then add new ones only if real)
## Questions for the authors (numbered)
## What I cannot judge
## Verdict: ACCEPT | MAJOR-REVISION | REJECT
## Score: N/10
"""

PRIOR_TEMPLATE = """
YOUR PRIOR-ROUND OBJECTIONS (verify each is resolved in the revision; state
RESOLVED / PARTIALLY-RESOLVED / UNRESOLVED for every one before new comments):
=====================================================================
{prior}
=====================================================================
"""


LB_ARTIFACT = ROOT / "runs" / "lb_ground_truth.md"


def load_lb_state() -> tuple[str, str]:
    """(date, state block) from the daily-refreshed ground-truth artifact.

    R19 FATAL fix (rl-planning + systems, 3 rounds of stale hardcoded
    0.43/1.56): the briefing now reads runs/lb_ground_truth.md, which the
    daily loop rewrites from the live Kaggle API before any panel round.
    A missing/stale artifact is surfaced to the reviewers, never papered over.
    """
    if LB_ARTIFACT.exists():
        import datetime as _dt
        mtime = _dt.date.fromtimestamp(LB_ARTIFACT.stat().st_mtime).isoformat()
        return mtime, LB_ARTIFACT.read_text(encoding="utf-8").strip()
    return ("UNKNOWN", "GROUND-TRUTH ARTIFACT MISSING (runs/lb_ground_truth.md) — "
            "file this as an objection; do not trust any leaderboard number in "
            "the proposal until the ledger artifact is attached.")


def spawn_reviewer(name: str, persona: str, idx: int, proposal: str, prior: str | None,
                   out_dir: Path, panel_model: str = "fable-panel"):
    import hashlib
    prior_block = PRIOR_TEMPLATE.format(prior=prior) if prior else ""
    doc_sha = hashlib.sha256(proposal.encode("utf-8")).hexdigest()[:16]
    doc_len = len(proposal)
    # A20 (amendment 2026-07-18b): circulation must be untruncated. The prompt
    # is written to a file and passed as @file (kaos run @path), bypassing the
    # Windows CreateProcess ~32K argv limit that truncated R14's circulation.
    body = proposal + "\n## END OF PROPOSAL ##"
    lb_date, lb_state = load_lb_state()
    prompt = REVIEW_TEMPLATE.format(
        persona=persona, idx=idx, proposal=body, prior_block=prior_block,
        doc_sha=doc_sha, doc_len=doc_len,
        lb_date=lb_date, lb_artifact="runs/lb_ground_truth.md", lb_state=lb_state,
    )
    prompt_file = out_dir / f"_prompt_{name}.md"
    prompt_file.write_text(prompt, encoding="utf-8")
    # R27 (2026-08-23): kaos agents are sandboxed to their CWD (F:/kaggle/kaos) and
    # REFUSE to read an @file outside it -- all 5 reviewers returned "the file is
    # outside the allowed working directory" (~200 chars, verdict UNKNOWN) and the
    # round scored 0/5 ACCEPT for purely mechanical reasons. R16-R26 predate the
    # sandbox and were unaffected, so the panel's historical record is uncontaminated.
    # Fix: stage a copy of the prompt INSIDE the kaos tree and hand the agent that
    # path. out_dir keeps the canonical archived copy.
    staged_dir = KAOS_DIR / "_panel_prompts"
    staged_dir.mkdir(parents=True, exist_ok=True)
    staged = staged_dir / f"{out_dir.name}_{name}.md"
    staged.write_text(prompt, encoding="utf-8")
    agent_name = f"panel-{name}"
    proc = subprocess.Popen(
        ["uv", "run", "kaos", "run", "-n", agent_name, "-m", panel_model,
         f"@{staged.resolve()}"],
        cwd=str(KAOS_DIR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace",  # Windows default cp1252 crashed the
        # reader thread on 0x81 in kaos stdout (R14: 2/5 reviewers lost)
    )
    return proc


def get_agent_id(stdout: str) -> str | None:
    m = re.search(r"Spawned agent:\s*(\S+)", stdout)
    return m.group(1) if m else None


def wait_all(agent_ids: dict, timeout_s: int = 1800) -> dict:
    deadline = time.monotonic() + timeout_s
    statuses = {}
    while time.monotonic() < deadline:
        ls = subprocess.run(["uv", "run", "kaos", "--json", "ls"],
                            cwd=str(KAOS_DIR), capture_output=True, text=True, timeout=120)
        try:
            agents = json.loads(ls.stdout)
        except Exception:
            time.sleep(20)
            continue
        by_id = {a["agent_id"]: a["status"] for a in agents}
        statuses = {name: by_id.get(aid, "missing") for name, aid in agent_ids.items()}
        if all(s in ("completed", "failed", "error") for s in statuses.values()):
            return statuses
        time.sleep(30)
    return statuses


def fetch_review(agent_id: str) -> str:
    """The CCR runner stores the final reply in agent state under key 'result'
    (kaos/ccr/runner.py: afs.set_state(agent_id, 'result', ...)). The
    conversation log only holds system+user for tool-less agents."""
    q = subprocess.run(
        ["uv", "run", "kaos", "query",
         f"SELECT value FROM state WHERE agent_id='{agent_id}' AND key='result'"],
        cwd=str(KAOS_DIR), capture_output=True, timeout=120,
    )
    raw = q.stdout.decode("utf-8", errors="replace")
    try:
        rows = json.loads(raw[raw.find("["):])
        if not rows:
            return ""
        val = rows[0].get("value", "")
        # value is a JSON-encoded string
        try:
            return json.loads(val)
        except Exception:
            return val
    except Exception:
        return ""


def parse_verdict(review: str) -> dict:
    verdict = "UNKNOWN"
    m = re.search(r"Verdict:\s*\**\s*(ACCEPT|MAJOR-REVISION|REJECT)", review, re.I)
    if m:
        verdict = m.group(1).upper()
    score = None
    m = re.search(r"Score:\s*\**\s*(\d+(?:\.\d+)?)\s*/\s*10", review)
    if m:
        score = float(m.group(1))
    n_fatal = len(re.findall(r"\[FATAL\]", review, re.I))
    n_major = len(re.findall(r"\[MAJOR\]", review, re.I))
    return {"verdict": verdict, "score": score, "n_fatal": n_fatal, "n_major": n_major}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--proposal", required=True)
    ap.add_argument("--prior-dir", default=None,
                    help="dir with prior round reviews ({reviewer}.md) for round>=2")
    ap.add_argument("--reviewers", default=None,
                    help="comma-separated subset of reviewer names (routine days: 2-3)")
    ap.add_argument("--panel-model", default="fable-panel",
                    help="KAOS model alias for reviewers (e.g. opus5-panel, fable-panel)")
    args = ap.parse_args()

    if args.reviewers:
        wanted = [r.strip() for r in args.reviewers.split(",") if r.strip()]
        unknown = [r for r in wanted if r not in REVIEWERS]
        if unknown:
            sys.exit(f"unknown reviewers: {unknown}; choose from {list(REVIEWERS)}")
        for name in list(REVIEWERS):
            if name not in wanted:
                del REVIEWERS[name]

    proposal = (ROOT / args.proposal).read_text(encoding="utf-8")
    out_dir = ROOT / "learnings" / "panel" / f"round{args.round}"
    out_dir.mkdir(parents=True, exist_ok=True)

    procs = {}
    for i, (name, persona) in enumerate(REVIEWERS.items(), 1):
        prior = None
        if args.prior_dir:
            pf = ROOT / args.prior_dir / f"{name}.md"
            if pf.exists():
                prior = pf.read_text(encoding="utf-8")
                # Keep only the objections section of the prior review (the
                # verdict/score would anchor; @file delivery removes the old
                # 6000-char argv cap).
                if "## Objections" in prior:
                    tail = prior.split("## Objections", 1)[1]
                    for stop in ("## Questions", "## What I cannot judge"):
                        if stop in tail:
                            tail = tail.split(stop, 1)[0]
                    prior = "## Objections" + tail
        procs[name] = spawn_reviewer(name, persona, i, proposal, prior, out_dir,
                                     panel_model=args.panel_model)
        time.sleep(2)  # stagger spawns

    # `kaos run` blocks until its agent completes, so communicate() waits for
    # the full review. All five run in parallel; we collect serially.
    # Recovery map: persist collected agent IDs as they appear so a killed
    # parent (e.g. headless loop turn cap, 2026-07-17 incident) leaves enough
    # on disk for a later session to fetch results from KAOS state.
    agents_file = out_dir / "_agents.json"
    agent_ids = {}
    for name, proc in procs.items():
        try:
            out, _ = proc.communicate(timeout=2400)
        except subprocess.TimeoutExpired:
            # proc.kill() alone orphans the claude children spawned by
            # `kaos run` (2026-07-21 wedge: 13 stale procs survived days).
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                           capture_output=True)
            print(f"WARN: reviewer {name} timed out after 40min (tree killed)", file=sys.stderr)
            continue
        aid = get_agent_id(out or "")
        if not aid:
            print(f"WARN: no agent id for {name}: {(out or '')[-200:]}", file=sys.stderr)
            continue
        agent_ids[name] = aid
        agents_file.write_text(json.dumps(agent_ids, indent=2), encoding="utf-8")
    print(f"Completed {len(agent_ids)}/5 reviewers: {agent_ids}", file=sys.stderr)

    statuses = wait_all(agent_ids, timeout_s=300)
    print(f"Statuses: {statuses}", file=sys.stderr)

    summary = {"round": args.round, "reviews": {}}
    for name, aid in agent_ids.items():
        review = fetch_review(aid)
        (out_dir / f"{name}.md").write_text(review, encoding="utf-8")
        summary["reviews"][name] = {
            "agent_id": aid,
            "status": statuses.get(name),
            **parse_verdict(review),
            "chars": len(review),
        }

    verdicts = [r["verdict"] for r in summary["reviews"].values()]
    fatals = sum(r["n_fatal"] for r in summary["reviews"].values())
    summary["n_accept"] = verdicts.count("ACCEPT")
    summary["n_fatal_total"] = fatals
    summary["pass"] = summary["n_accept"] >= 4 and fatals == 0
    (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
