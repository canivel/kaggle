"""Pass 2: prompt-side. What does the carried world model actually look like turn to turn?

Measures staleness, emptiness, wipes at level transitions, and overwrite-vs-accumulate.
Read-only.
"""
from __future__ import annotations
import json, re, collections, statistics, sys
from pathlib import Path

RUNS = Path(r"F:\kaggle\arc-prize-2026\runs")
SECTION_RE = re.compile(r"^\[([A-Z][A-Z0-9 _:.\-]*)\]$")
CARRY_HDR = "Working world model carried from earlier turns:"
SLOTS = ["World model", "Goal model", "Action model", "Recent findings",
         "Open questions", "Plan", "Cross-level notes"]
SLOT_RE = re.compile(r"^- (" + "|".join(SLOTS) + r"): (.*)$")

def split_sections(transcript: str):
    out, cur_label, cur = [], None, []
    for line in transcript.splitlines():
        m = SECTION_RE.match(line.strip()) if line.startswith("[") else None
        if m:
            if cur_label is not None:
                out.append((cur_label, "\n".join(cur).strip()))
            cur_label, cur = m.group(1), []
        elif cur_label is not None:
            cur.append(line)
    if cur_label is not None:
        out.append((cur_label, "\n".join(cur).strip()))
    return out

def parse_carry(prompt: str):
    """-> dict slot->value (only populated slots), or {} if no carry block."""
    if CARRY_HDR not in prompt:
        return None
    body = prompt.split(CARRY_HDR, 1)[1]
    out = {}
    for line in body.splitlines():
        s = line.strip()
        if s.startswith("- Revise any item above"):
            break
        if s == "end of world model.":
            break
        m = SLOT_RE.match(s)
        if m:
            out[m.group(1)] = m.group(2).strip()
        elif out and s and not s.startswith("-"):
            continue
    return out

def main():
    files = sorted(p for p in RUNS.rglob("*_events.jsonl") if "tool_compaction" not in p.name)
    S = collections.Counter()
    stale_runs = []           # consecutive turns with byte-identical carry block
    slot_pop = collections.Counter()
    slot_len = collections.defaultdict(list)
    wipe_events = 0
    turns_after_wipe_empty = 0
    empty_carry_turns = 0
    total_prompt_turns = 0
    changed_turns = 0
    per_slot_change = collections.Counter()
    overlap_samples = collections.defaultdict(list)
    lvl_transition_turns = 0
    lvl_trans_next_empty = 0
    n_runs = 0

    for fp in files:
        try:
            lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        n_runs += 1
        prev = None
        run_len = 0
        pending_wipe = False
        for ln in lines:
            if '"transcript"' not in ln:
                continue
            try:
                ev = json.loads(ln)
            except Exception:
                continue
            tr = ev.get("transcript")
            if not isinstance(tr, str) or not tr:
                continue
            secs = split_sections(tr)
            # the FIRST user prompt of the turn is the analyzer's real turn prompt
            first_user = None
            for label, body in secs:
                if label == "USER PROMPT":
                    first_user = body
                    break
            if first_user is None:
                continue
            total_prompt_turns += 1
            carry = parse_carry(first_user)
            if carry is None:
                # store completely empty -> _summarized_knowledge_lines() returned []
                empty_carry_turns += 1
                carry = {}
            if pending_wipe:
                if not carry:
                    lvl_trans_next_empty += 1
                pending_wipe = False
            if re.search(r"You have progressed to a new level!|You have completed the run!|The game is over\.", first_user):
                lvl_transition_turns += 1
                pending_wipe = True
            for k, v in carry.items():
                slot_pop[k] += 1
                slot_len[k].append(len(v))
            if prev is not None:
                if carry == prev:
                    run_len += 1
                else:
                    if run_len: stale_runs.append(run_len + 1)
                    run_len = 0
                    changed_turns += 1
                    for k in SLOTS:
                        if carry.get(k, "") != prev.get(k, ""):
                            per_slot_change[k] += 1
                            a, b = set(prev.get(k, "").lower().split()), set(carry.get(k, "").lower().split())
                            if a and b:
                                overlap_samples[k].append(len(a & b) / len(a))
            prev = carry
        if run_len: stale_runs.append(run_len + 1)

    def pct(a, b): return f"{100.0*a/b:.2f}%" if b else "n/a"
    print(json.dumps({
        "files": n_runs,
        "turns_with_a_user_prompt": total_prompt_turns,
        "turns_with_EMPTY_carried_world_model": empty_carry_turns,
        "empty_carry_pct": pct(empty_carry_turns, total_prompt_turns),
        "turns_where_carry_CHANGED_vs_prev": changed_turns,
        "carry_changed_pct": pct(changed_turns, total_prompt_turns),
        "level_transition_turns": lvl_transition_turns,
        "turn_after_transition_had_empty_carry": lvl_trans_next_empty,
        "slot_populated_counts": dict(slot_pop),
        "slot_mean_chars": {k: round(statistics.mean(v), 1) for k, v in slot_len.items()},
        "per_slot_change_counts": dict(per_slot_change),
        "mean_prev_token_retention_on_change": {
            k: round(statistics.mean(v), 3) for k, v in overlap_samples.items() if v},
        "stale_run_len_mean": round(statistics.mean(stale_runs), 2) if stale_runs else None,
        "stale_run_len_median": statistics.median(stale_runs) if stale_runs else None,
        "stale_run_len_max": max(stale_runs) if stale_runs else None,
        "stale_run_len_p90": round(statistics.quantiles(stale_runs, n=10)[8], 1) if len(stale_runs) > 10 else None,
        "n_stale_runs": len(stale_runs),
    }, indent=2))

if __name__ == "__main__":
    main()
