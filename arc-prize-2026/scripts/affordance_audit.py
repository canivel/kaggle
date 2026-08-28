"""AFFORDANCE USE AUDIT — is every affordance available where advertised, and used?

WHY THIS EXISTS
    One failure class has hit this campaign FIVE times and is 5-for-5 whenever
    anyone looks:

      P1 notes        affordance in the tool-call SCHEMA only -> 1.3% use vs a 30% bar
      P2 attempt()    delivered and understood                -> 10.73% use vs a 25% bar
      animation()     ADVERTISED in the prompt, NOT INJECTED  -> 29 wasted actions
                                                                 across 13/25 games
      RESET           available, 100%/306 reliable undo,
                      NEVER advertised in prompt text         -> 7/1555 = 0.45% use
      schema-only     (feedback_advertise_where_model_reads)

    Every one was found by hand, late, and usually after a Kaggle slot was
    already spent. The pattern is mechanical, so it should be a check, not a
    discovery.

THE TAXONOMY -- each verdict names a real, previously-paid failure
    ADVERTISED-NOT-AVAILABLE   the prompt names it; the sandbox does not inject
                               it. The model tries and burns a turn on
                               NameError.                      [animation() class]
    AVAILABLE-NOT-ADVERTISED   it exists and works, but the model only ever
                               meets it as a bare token.       [RESET class]
    ADVERTISED-UNUSED          both present, and the model still does not reach
                               for it. A PREFERENCE failure, not a discovery
                               one -- making it louder is the wrong repair.
                                                               [P1 / P2 class]
    OK                         advertised, available, used above the floor.

USAGE
    scripts/affordance_audit.py                      # newest Kaggle pull
    scripts/affordance_audit.py --pull runs/kernel_pulls/private_edge2_v3
    scripts/affordance_audit.py --local              # local traces.db instead
    scripts/affordance_audit.py --json out.json

This reads artifacts only. It costs no GPU, no Kaggle slot, and no model call.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import sys
from pathlib import Path

_SELF = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != _SELF]

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "duck_eval" / "private" / "bundle_20260815" / "src" / "ARC3-Inference"
PULLS = ROOT / "runs" / "kernel_pulls"

# Sandbox globals the harness may inject, and the game actions it may expose.
SANDBOX_GLOBALS = [
    "action", "animation", "current_frame", "previous_frame", "latest_frame",
    "history", "transitions", "last_transition", "last_action",
    "last_action_frame", "last_action_result", "valid_actions",
]
GAME_ACTIONS = ["RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4",
                "ACTION5", "ACTION6"]

# Below this share of actions, an available+advertised affordance is "unused".
# 5% is deliberately lax: P1 died at 1.3% against a 30% bar and P2 at 10.73%
# against 25%, so this flags only the floor, not the campaign's real bars.
UNUSED_FLOOR = 0.05


# ---------------------------------------------------------------------------
# what the harness ADVERTISES
# ---------------------------------------------------------------------------
def advertised_text(animation_retrieval: bool) -> dict[str, str]:
    """The exact strings the model is shown, built from the real bundle."""
    sys.path.insert(0, str(BUNDLE))
    from inference.agent.tool_agent import (  # noqa: PLC0415
        _build_system_prompt, _python_tool_description)
    return {
        "system_prompt": _build_system_prompt(
            tool_output_tokens=1024, animation_retrieval=animation_retrieval),
        "tool_description": _python_tool_description(
            animation_retrieval=animation_retrieval),
    }


def sandbox_injects() -> dict[str, str]:
    """Which globals the sandbox actually installs, and under what condition."""
    src = (BUNDLE / "inference" / "agent" / "python_tool_sandbox.py").read_text()
    out = {}
    for m in re.finditer(r'runtime_globals\["(\w+)"\]\s*=', src):
        name = m.group(1)
        line_start = src.rfind("\n", 0, m.start())
        prev = src[max(0, line_start - 200):line_start]
        cond = "always"
        gm = re.search(r'if\s+(.+?):\s*$', prev.strip().split("\n")[-1] or "")
        if gm:
            cond = gm.group(1).strip()
        out[name] = cond
    return out


# ---------------------------------------------------------------------------
# what the model actually USED
# ---------------------------------------------------------------------------
def use_from_pull(pull: Path) -> tuple[collections.Counter, collections.Counter, int, int]:
    """Game-action counts and sandbox-global mentions from a real Kaggle pull."""
    actions, globals_used = collections.Counter(), collections.Counter()
    n_actions = n_turns = 0
    for f in glob.glob(str(pull / "artifacts" / "*_events.jsonl")):
        for line in open(f, errors="ignore"):
            try:
                d = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            t = d.get("type")
            if t == "action":
                n_actions += 1
                actions[d.get("action_name") or d.get("action_display") or "?"] += 1
            elif t == "analysis":
                n_turns += 1
    # SANDBOX GLOBALS ARE NOT MEASURABLE FROM A KAGGLE PULL.
    # prompts/*.log is a "LATEST MODEL CALL SNAPSHOT" holding only the last few
    # turns, rendered as prose -- there is no structured tool_call code to scan.
    # Reporting 0 uses here would be an instrument artifact, not a finding, so
    # globals_used is left EMPTY and the caller marks these UNMEASURED.
    # Use --local for global-level use: the trace store keeps full tool_calls.
    return actions, globals_used, n_actions, n_turns


def use_from_local() -> tuple[collections.Counter, collections.Counter, int, int]:
    """Same, from the local trace store's stored tool_call arguments."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import trace_store as ts  # noqa: PLC0415
    con = ts.connect()
    actions, globals_used = collections.Counter(), collections.Counter()
    n_turns = 0
    for (sha,) in con.execute(
            "SELECT response_sha FROM call WHERE response_sha IS NOT NULL"):
        body = ts.get_blob(con, sha)
        if not isinstance(body, dict):
            continue
        msg = ((body.get("choices") or [{}])[0]).get("message") or {}
        for tc in (msg.get("tool_calls") or []):
            n_turns += 1
            code = str((tc.get("function") or {}).get("arguments") or "")
            for g in SANDBOX_GLOBALS:
                if re.search(rf"\b{g}\s*[\(\[\.\b]", code):
                    globals_used[g] += 1
            for a in GAME_ACTIONS:
                actions[a] += len(re.findall(rf"\b{a}\b", code))
    return actions, globals_used, sum(actions.values()), n_turns


# ---------------------------------------------------------------------------
# the audit
# ---------------------------------------------------------------------------
def audit(adv: dict[str, str], injects: dict[str, str],
          actions: collections.Counter, globals_used: collections.Counter,
          n_actions: int, n_turns: int, measured_globals: bool = True) -> list[dict]:
    rows = []
    prompt, tooldesc = adv["system_prompt"], adv["tool_description"]

    for g in SANDBOX_GLOBALS:
        in_prompt = f"`{g}`" in prompt or f"{g}(" in prompt
        in_tool = f"`{g}`" in tooldesc or f"{g}(" in tooldesc
        available = g in injects
        cond = injects.get(g, "-")
        uses = globals_used.get(g, 0)
        rate = uses / n_turns if n_turns else 0.0
        rows.append({"name": g, "kind": "sandbox_global", "available": available,
                     "condition": cond, "advertised_prompt": in_prompt,
                     "advertised_tool": in_tool, "uses": uses,
                     "denominator": n_turns, "rate": rate, "measured": measured_globals,
                     "verdict": verdict(available, in_prompt or in_tool, rate,
                                        measured_globals)})

    for a in GAME_ACTIONS:
        in_prompt = a in prompt
        in_tool = a in tooldesc
        uses = actions.get(a, 0)
        rate = uses / n_actions if n_actions else 0.0
        rows.append({"name": a, "kind": "game_action",
                     "available": uses > 0 or a != "RESET",
                     "condition": "valid_actions", "advertised_prompt": in_prompt,
                     "advertised_tool": in_tool, "uses": uses,
                     "denominator": n_actions, "rate": rate, "measured": True,
                     "verdict": verdict(True, in_prompt or in_tool, rate)})
    return rows


def verdict(available: bool, advertised: bool, rate: float,
            measured: bool = True) -> str:
    """Advertisement only matters when USE is low.

    An affordance the model reaches for 28% of the time is discoverable by
    whatever route it is already finding; flagging it because the prompt does
    not name it would be noise. The whole failure class is about affordances
    that are NOT being used.
    """
    if advertised and not available:
        return "ADVERTISED-NOT-AVAILABLE"     # fires regardless of use: it cannot be used
    if not measured:
        return "UNMEASURED"
    if rate >= UNUSED_FLOOR:
        return "OK"                            # used enough; advertisement moot
    if available and not advertised:
        return "AVAILABLE-NOT-ADVERTISED"
    if available and advertised:
        return "ADVERTISED-UNUSED"
    return "OK"


SEVERITY = {"ADVERTISED-NOT-AVAILABLE": 0, "AVAILABLE-NOT-ADVERTISED": 1,
            "ADVERTISED-UNUSED": 2, "UNMEASURED": 3, "OK": 4}

EXPLAIN = {
    "ADVERTISED-NOT-AVAILABLE":
        "the prompt names it but the sandbox does not inject it -- the model "
        "tries it and burns a turn on NameError (the animation() defect: 29 "
        "wasted actions across 13/25 games)",
    "AVAILABLE-NOT-ADVERTISED":
        "it exists and works, but the model only meets it as a bare token in "
        "valid_actions (the RESET case: a 100%/306 reliable undo in a 90.1% "
        "one-way environment, used 0.45% of the time)",
    "UNMEASURED":
        "this source cannot measure sandbox-global use -- a Kaggle pull keeps "
        "only a last-call prose snapshot. Run with --local, where the trace "
        "store holds full tool_call arguments. NOT a finding.",
    "ADVERTISED-UNUSED":
        "advertised AND available, and still not reached for. A PREFERENCE "
        "failure, not a discovery one -- making it louder is the wrong repair "
        "(P1 1.3% vs a 30% bar; P2 10.73% vs 25%)",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", default=None, help="a runs/kernel_pulls/<arm> dir")
    ap.add_argument("--local", action="store_true", help="use the local trace store")
    ap.add_argument("--animation-retrieval", action="store_true",
                    help="build the prompt as if retrieval were enabled")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    adv = advertised_text(a.animation_retrieval)
    injects = sandbox_injects()

    if a.local:
        src = "local trace store"
        actions, gu, na, nt = use_from_local()
    else:
        pull = Path(a.pull) if a.pull else max(
            (p for p in PULLS.iterdir() if (p / "artifacts").is_dir()),
            key=lambda p: p.stat().st_mtime, default=None)
        if not pull:
            print("no pull with artifacts/ found; use --local", file=sys.stderr)
            return 2
        pull = Path(pull).resolve()
        src = str(pull.relative_to(ROOT)) if str(pull).startswith(str(ROOT)) else str(pull)
        actions, gu, na, nt = use_from_pull(Path(pull))

    rows = audit(adv, injects, actions, gu, na, nt, measured_globals=a.local)
    rows.sort(key=lambda r: (SEVERITY[r["verdict"]], -r["rate"]))

    bar = "=" * 92
    print(bar)
    print(f"  AFFORDANCE USE AUDIT   source: {src}")
    print(f"  animation_retrieval={a.animation_retrieval}   "
          f"{na} actions, {nt} analysis turns")
    print(bar)
    print(f"  {'affordance':<20}{'kind':<16}{'avail':<7}{'prompt':<8}{'tool':<6}"
          f"{'uses':>7}{'rate':>8}  verdict")
    for r in rows:
        print(f"  {r['name']:<20}{r['kind']:<16}"
              f"{'yes' if r['available'] else 'NO':<7}"
              f"{'yes' if r['advertised_prompt'] else 'no':<8}"
              f"{'yes' if r['advertised_tool'] else 'no':<6}"
              f"{r['uses']:>7}{100*r['rate']:>7.1f}%  {r['verdict']}")

    findings = [r for r in rows if r["verdict"] != "OK"]
    if findings:
        print(f"\n{bar}\n  FINDINGS\n{bar}")
        for v in sorted({r["verdict"] for r in findings}, key=lambda x: SEVERITY[x]):
            names = [r["name"] for r in findings if r["verdict"] == v]
            print(f"\n  [{v}]  {', '.join(names)}")
            print(f"    {EXPLAIN[v]}")
    else:
        print("\n  No mismatches. Every affordance is available where advertised "
              "and used above the floor.")

    if a.json:
        Path(a.json).write_text(json.dumps(
            {"source": src, "n_actions": na, "n_turns": nt, "rows": rows}, indent=2))
        print(f"\n  wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
