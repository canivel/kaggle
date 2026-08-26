"""Pass 5: counterfactual store replay. Baseline parser vs tolerant-header parser vs +reasoning.

Validation: the BASELINE simulation must reproduce the empty-carry rate actually observed in the
prompts (53.46%). If it does, the simulation is a faithful model of the harness.
Read-only.
"""
from __future__ import annotations
import json, re, collections, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _wm_probe_0814 import _extract_labeled_blocks, LABELS, split_sections  # noqa

RUNS = Path(r"F:\kaggle\arc-prize-2026\runs")
CARRY_HDR = "Working world model carried from earlier turns:"
WIPE_RE = re.compile(r"You have progressed to a new level!|You have completed the run!|The game is over\.")

KEYMAP = {"World model": "world_model", "Goal model": "goal_model", "Action model": "action_model",
          "Recent findings": "recent_findings", "Open questions": "open_questions",
          "Plan": "current_plan", "Cross-level notes": "cross_level_notes",
          "Hypothesis": "world_model", "History check": "recent_findings", "Next test": "current_plan"}
WIPE_KEYS = ["world_model", "goal_model", "action_model", "recent_findings",
             "open_questions", "current_plan"]

def baseline_note(content: str) -> dict:
    if not content.strip(): return {}
    e = _extract_labeled_blocks(content, LABELS)
    out = {}
    for label in ["World model", "Goal model", "Action model", "Recent findings",
                  "Open questions", "Plan", "Cross-level notes"]:
        if e.get(label): out[KEYMAP[label]] = e[label]
    for fallback, primary in [("Hypothesis", "world_model"), ("History check", "recent_findings"),
                              ("Next test", "current_plan")]:
        if primary not in out and e.get(fallback): out[primary] = e[fallback]
    return out

# ---- TOLERANT header: decoration + optional qualifier before/after the slot name ----
SLOT_ALT = r"(?:world\s*model|goal\s*model|action\s*model|recent\s*findings|open\s*questions|plan|cross[\s\-]*level\s*notes|hypothesis|history\s*check|next\s*test)"
QUAL_PRE = r"(?:revised|updated|update|new|current|refined|final|working)\s+"
QUAL_POST = r"(?:\s*\((?:revised|updated|update|new|current|refined|confirmed|so\s*far|cont(?:inued)?)\)|\s+(?:updated?|update|revision|revised|confirmed|so\s*far))"
TOLERANT = re.compile(
    r"^[\s>#*_\u2022\-]*(?:\d+[.)]\s*)?[\*_#\s]*(?:" + QUAL_PRE + r")?(" + SLOT_ALT + r")(?:" + QUAL_POST + r")?[\*_\s]*:\s*(.*)$",
    re.IGNORECASE)
SLOT_CANON = {"worldmodel": "world_model", "goalmodel": "goal_model", "actionmodel": "action_model",
              "recentfindings": "recent_findings", "openquestions": "open_questions",
              "plan": "current_plan", "crosslevelnotes": "cross_level_notes",
              "hypothesis": "world_model", "historycheck": "recent_findings",
              "nexttest": "current_plan"}

def tolerant_note(content: str) -> dict:
    if not content.strip(): return {}
    acc = collections.defaultdict(list)
    cur = None
    for raw in content.splitlines():
        s = raw.strip()
        m = TOLERANT.match(s) if s else None
        if m:
            key = SLOT_CANON[re.sub(r"[^a-z]", "", m.group(1).lower())]
            cur = key
            if m.group(2).strip(): acc[key].append(m.group(2).strip())
            continue
        if cur is not None and s:
            acc[cur].append(s)
    return {k: " ".join("\n".join(v).split()) for k, v in acc.items() if "\n".join(v).strip()}

def replay(files, note_fn, use_reasoning=False, use_prev_on_wipe=False):
    empty = 0; turns = 0; updates = 0; stale_runs = []
    for fp in files:
        try: lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError: continue
        store = {k: "" for k in WIPE_KEYS + ["cross_level_notes"]}
        prev_render = None; run_len = 0
        for ln in lines:
            if '"transcript"' not in ln: continue
            try: ev = json.loads(ln)
            except Exception: continue
            tr = ev.get("transcript")
            if not isinstance(tr, str) or not tr: continue
            secs = split_sections(tr)
            fu = next((b for l, b in secs if l == "USER PROMPT"), None)
            if fu is None: continue
            # --- observe the store as it would be rendered at the TOP of this turn ---
            turns += 1
            render = tuple(sorted((k, v) for k, v in store.items() if v))
            if not render: empty += 1
            if prev_render is not None:
                if render == prev_render: run_len += 1
                else:
                    if run_len: stale_runs.append(run_len + 1)
                    run_len = 0
            prev_render = render
            # --- apply this turn's updates ---
            cur = None; units = []
            for label, body in secs:
                if label == "MODEL RESPONSE META":
                    if cur: units.append(cur)
                    cur = {"a": "", "t": ""}
                elif cur is not None:
                    if label == "ASSISTANT": cur["a"] += "\n" + body
                    elif label == "THINKING": cur["t"] += "\n" + body
            if cur: units.append(cur)
            wrote = False
            for u in units:
                n = note_fn(u["a"].strip())
                if not n and use_reasoning:
                    n = note_fn(u["t"].strip())
                for k, v in n.items():
                    if v: store[k] = v; wrote = True
            if wrote: updates += 1
            # --- wipe if the NEXT turn's prompt reports a transition ---
            if WIPE_RE.search(fu) and turns > 1:
                pass  # transition is reported at the START of the following turn; handled below
        if run_len: stale_runs.append(run_len + 1)
    return {"turns": turns, "empty_carry": empty,
            "empty_pct": round(100*empty/max(1, turns), 2),
            "turns_with_update": updates,
            "update_pct": round(100*updates/max(1, turns), 2),
            "stale_mean": round(statistics.mean(stale_runs), 2) if stale_runs else None,
            "stale_p90": round(statistics.quantiles(stale_runs, n=10)[8], 1) if len(stale_runs) > 10 else None}

def replay_with_wipe(files, note_fn, use_reasoning=False):
    """Same, but applies the level-transition wipe (detected from the FOLLOWING turn's prompt)."""
    empty = 0; turns = 0; updates = 0; stale_runs = []; wipes = 0
    for fp in files:
        try: lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError: continue
        store = {k: "" for k in WIPE_KEYS + ["cross_level_notes"]}
        prev_render = None; run_len = 0
        for ln in lines:
            if '"transcript"' not in ln: continue
            try: ev = json.loads(ln)
            except Exception: continue
            tr = ev.get("transcript")
            if not isinstance(tr, str) or not tr: continue
            secs = split_sections(tr)
            fu = next((b for l, b in secs if l == "USER PROMPT"), None)
            if fu is None: continue
            if WIPE_RE.search(fu):
                for k in WIPE_KEYS: store[k] = ""
                wipes += 1
            turns += 1
            render = tuple(sorted((k, v) for k, v in store.items() if v))
            if not render: empty += 1
            if prev_render is not None:
                if render == prev_render: run_len += 1
                else:
                    if run_len: stale_runs.append(run_len + 1)
                    run_len = 0
            prev_render = render
            cur = None; units = []
            for label, body in secs:
                if label == "MODEL RESPONSE META":
                    if cur: units.append(cur)
                    cur = {"a": "", "t": ""}
                elif cur is not None:
                    if label == "ASSISTANT": cur["a"] += "\n" + body
                    elif label == "THINKING": cur["t"] += "\n" + body
            if cur: units.append(cur)
            wrote = False
            for u in units:
                n = note_fn(u["a"].strip())
                if not n and use_reasoning:
                    n = note_fn(u["t"].strip())
                for k, v in n.items():
                    if v: store[k] = v; wrote = True
            if wrote: updates += 1
        if run_len: stale_runs.append(run_len + 1)
    return {"turns": turns, "empty_carry": empty, "empty_pct": round(100*empty/max(1, turns), 2),
            "turns_with_update": updates, "update_pct": round(100*updates/max(1, turns), 2),
            "wipes": wipes,
            "stale_mean": round(statistics.mean(stale_runs), 2) if stale_runs else None,
            "stale_p90": round(statistics.quantiles(stale_runs, n=10)[8], 1) if len(stale_runs) > 10 else None}

if __name__ == "__main__":
    files = sorted(p for p in RUNS.rglob("*_events.jsonl") if "tool_compaction" not in p.name)
    out = {
        "OBSERVED_in_prompts": {"turns": 37320, "empty_carry": 19950, "empty_pct": 53.46},
        "SIM_baseline_no_wipe": replay(files, baseline_note),
        "SIM_baseline_with_wipe": replay_with_wipe(files, baseline_note),
        "SIM_tolerant_with_wipe": replay_with_wipe(files, tolerant_note),
        "SIM_tolerant_plus_reasoning_with_wipe": replay_with_wipe(files, tolerant_note, use_reasoning=True),
        "SIM_baseline_plus_reasoning_with_wipe": replay_with_wipe(files, baseline_note, use_reasoning=True),
    }
    print(json.dumps(out, indent=2))
