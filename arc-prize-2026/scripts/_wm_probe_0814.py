"""LM-free diagnosis of world-model-update loss. Read-only. Local. Zero spend.

Reproduces the harness parser (_extract_labeled_blocks / _extract_scientist_note from
tool_agent.py) exactly, then partitions no-update turns by cause.
"""
from __future__ import annotations
import json, os, re, sys, collections, random
from pathlib import Path

RUNS = Path(r"F:\kaggle\arc-prize-2026\runs")

# ---------------- faithful copy of harness parser ----------------
LABELS = ["World model", "Goal model", "Action model", "Recent findings",
          "Open questions", "Plan", "Cross-level notes", "Hypothesis",
          "History check", "Next test"]

def _extract_labeled_blocks(content: str, labels: list[str]) -> dict[str, str]:
    normalized_labels = {label.lower(): label for label in labels}
    targets = tuple(f"{label.lower()}:" for label in labels)
    extracted: dict[str, list[str]] = {label: [] for label in labels}
    current_label = None
    for raw_line in content.splitlines():
        stripped = raw_line.strip()
        candidate = stripped
        while candidate.startswith(("-", "*")):
            candidate = candidate[1:].lstrip()
        lowered = candidate.lower()
        matched_label = None
        inline_value = ""
        for target in targets:
            if lowered.startswith(target):
                matched_label = normalized_labels[target[:-1]]
                inline_value = candidate[len(target):].strip()
                break
        if matched_label is not None:
            current_label = matched_label
            if inline_value:
                extracted[current_label].append(inline_value)
            continue
        if current_label is not None and stripped:
            extracted[current_label].append(stripped)
    return {k: " ".join("\n".join(v).strip().split()) for k, v in extracted.items()
            if "\n".join(v).strip()}

STORE_KEYS = ["world_model", "goal_model", "action_model", "recent_findings",
              "open_questions", "current_plan", "cross_level_notes"]

def extract_scientist_note(content: str) -> dict[str, str]:
    if not content.strip():
        return {}
    e = _extract_labeled_blocks(content, LABELS)
    r = {"world_model": e.get("World model", ""), "goal_model": e.get("Goal model", ""),
         "action_model": e.get("Action model", ""), "recent_findings": e.get("Recent findings", ""),
         "open_questions": e.get("Open questions", ""), "current_plan": e.get("Plan", ""),
         "cross_level_notes": e.get("Cross-level notes", "")}
    if not r["world_model"]: r["world_model"] = e.get("Hypothesis", "")
    if not r["recent_findings"]: r["recent_findings"] = e.get("History check", "")
    if not r["current_plan"]: r["current_plan"] = e.get("Next test", "")
    return r

def effective_update(content: str) -> dict[str, str]:
    """Returns the non-empty subset that would actually be written to the store."""
    note = extract_scientist_note(content)
    return {k: v for k, v in note.items() if v}

# ---------- LOOSE detector: any label written in ANY plausible form ----------
LABEL_ALT = r"(?:world[\s_-]*model|goal[\s_-]*model|action[\s_-]*model|recent[\s_-]*findings|open[\s_-]*questions|plan|cross[\s_-]*level[\s_-]*notes|hypothesis|history[\s_-]*check|next[\s_-]*test)"
# line-anchored, tolerant of markdown/list/numbering/bold/heading decoration before AND after
LOOSE = re.compile(r"(?im)^[\s>#*_\-\u2022\d.)\[]*\**\s*" + LABEL_ALT + r"\s*\**[\]\)]*\s*[:\-\u2013]")
# even looser: label appears anywhere on its own with a colon
ANYWHERE = re.compile(r"(?i)\b" + LABEL_ALT + r"\s*\**\s*:")

SECTION_RE = re.compile(r"^\[([A-Z][A-Z0-9 _:.\-]*)\]$")

def split_sections(transcript: str):
    """-> list[(label, body)] in order."""
    out = []
    cur_label, cur = None, []
    for line in transcript.splitlines():
        m = SECTION_RE.match(line.strip()) if line.startswith("[") else None
        if m:
            if cur_label is not None:
                out.append((cur_label, "\n".join(cur).strip()))
            cur_label, cur = m.group(1), []
        else:
            if cur_label is not None:
                cur.append(line)
    if cur_label is not None:
        out.append((cur_label, "\n".join(cur).strip()))
    return out

META_KV = re.compile(r"^(\w+):\s*(.*)$")

def parse_meta(body: str) -> dict:
    d = {}
    for line in body.splitlines():
        m = META_KV.match(line.strip())
        if m and m.group(1) in ("finish_reason", "tool_call_count", "content_chars",
                                "reasoning_chars", "tool_call_markup_in_text",
                                "tool_calls_recovered_from_markup"):
            d[m.group(1)] = m.group(2)
    return d

def main(sample_examples=False):
    files = sorted(p for p in RUNS.rglob("*_events.jsonl")
                   if "tool_compaction" not in p.name)
    stats = collections.Counter()
    per_family = collections.defaultdict(collections.Counter)
    classes = collections.Counter()
    examples = collections.defaultdict(list)
    turn_stats = collections.Counter()
    label_hist = collections.Counter()
    rng = random.Random(20260814)

    for fp in files:
        fam = fp.relative_to(RUNS).parts[0]
        stats["files"] += 1
        try:
            lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            stats["files_unreadable"] += 1
            continue
        for ln in lines:
            if '"transcript"' not in ln:
                continue
            try:
                ev = json.loads(ln)
            except Exception:
                stats["bad_json"] += 1
                continue
            tr = ev.get("transcript")
            if not isinstance(tr, str) or not tr:
                continue
            stats["transcript_events"] += 1
            secs = split_sections(tr)
            # group into model-response units: each starts at a MODEL RESPONSE META
            units = []
            cur = None
            pre_prompt = ""   # last USER/SYSTEM PROMPT seen before this response
            last_prompt = ""
            for label, body in secs:
                if label in ("SYSTEM PROMPT", "USER PROMPT"):
                    last_prompt = body
                if label == "MODEL RESPONSE META":
                    if cur: units.append(cur)
                    cur = {"meta": parse_meta(body), "thinking": "", "assistant": "",
                           "toolcalls": [], "toolresults": [], "prompt": last_prompt,
                           "status": []}
                elif cur is not None:
                    if label == "THINKING":
                        cur["thinking"] += ("\n" + body)
                    elif label == "ASSISTANT":
                        cur["assistant"] += ("\n" + body)
                    elif label.startswith("TOOL CALL"):
                        cur["toolcalls"].append(body)
                    elif label.startswith("TOOL RESULT"):
                        cur["toolresults"].append(body)
                    elif label == "ANALYZER STATUS":
                        cur["status"].append(body)
            if cur: units.append(cur)
            if not units:
                stats["events_no_meta"] += 1
                continue

            turn_updated = False
            for u in units:
                stats["responses"] += 1
                per_family[fam]["responses"] += 1
                a = u["assistant"].strip()
                th = u["thinking"].strip()
                if th: stats["resp_with_thinking"] += 1
                if a:  stats["resp_with_assistant"] += 1
                else:  stats["resp_no_visible_content"] += 1

                upd = effective_update(a) if a else {}
                if upd:
                    stats["resp_with_update"] += 1
                    per_family[fam]["resp_with_update"] += 1
                    turn_updated = True
                    for k in upd: label_hist[k] += 1
                    continue

                # ---- classify the NO-UPDATE response ----
                stats["resp_no_update"] += 1
                th_upd = effective_update(th) if th else {}
                cls = None
                if th_upd:
                    cls = "i_channel_parser_drop"   # well-formed, wrong channel
                elif a and LOOSE.search(a):
                    cls = "ii_schema_near_miss_assistant"
                elif th and LOOSE.search(th):
                    cls = "ii_schema_near_miss_thinking"
                elif a and ANYWHERE.search(a):
                    cls = "ii_label_inline_assistant"
                elif not a and not th:
                    cls = "v_empty_response"
                elif not a and th:
                    cls = "iii_no_visible_content_reasoning_only"
                elif a:
                    cls = "iv_prose_only_no_label"
                classes[cls] += 1
                if sample_examples and len(examples[cls]) < 40 and rng.random() < 0.25:
                    examples[cls].append({
                        "file": str(fp.relative_to(RUNS)), "meta": u["meta"],
                        "assistant": a[:900], "thinking_tail": th[-900:],
                        "prompt_asks": bool(re.search(r"(?i)world model", u["prompt"] or "")),
                        "n_toolcalls": len(u["toolcalls"]),
                    })
                # does the prompt for this response even ask?
                if not re.search(r"(?i)world model", u["prompt"] or ""):
                    stats["no_update_prompt_silent"] += 1

            turn_stats["turns"] += 1
            if turn_updated: turn_stats["turns_with_update"] += 1

    out = {"stats": dict(stats), "classes": dict(classes), "turn_stats": dict(turn_stats),
           "label_hist": dict(label_hist),
           "per_family": {k: dict(v) for k, v in per_family.items()}}
    print(json.dumps(out, indent=2))
    if sample_examples:
        Path(r"F:\kaggle\arc-prize-2026\runs\_wm_examples_0814.json").write_text(
            json.dumps(examples, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main(sample_examples="--examples" in sys.argv)
