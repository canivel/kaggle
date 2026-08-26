"""Pass 4: exhaustive near-miss census over ASSISTANT content that produced NO store update.

Answers: how much of the no-update mass is RECOVERABLE (model already paid the tokens)?
Read-only.
"""
from __future__ import annotations
import json, re, collections, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _wm_probe_0814 import effective_update, split_sections, parse_meta  # noqa

RUNS = Path(r"F:\kaggle\arc-prize-2026\runs")

CORE = r"(?:world\s*model|goal\s*model|action\s*model|recent\s*findings|open\s*questions|plan|cross[\s-]*level\s*notes|hypothesis|history\s*check|next\s*test)"
# a line that clearly announces one of the store slots but is not `^<label>:`
NEARMISS = re.compile(r"(?im)^(.{0,60}?" + CORE + r".{0,40}?)\s*:\s*(?=\S|$)")
HEADERISH = re.compile(r"(?im)^[\s>#*_\-\u2022]*\d*[.)]?\s*\**\s*(" + CORE + r")\b[^\n:]{0,40}:")

def norm_form(line: str) -> str:
    s = line.strip()
    s = re.sub(r"\s+", " ", s)
    return s[:70]

def main():
    files = sorted(p for p in RUNS.rglob("*_events.jsonl") if "tool_compaction" not in p.name)
    forms = collections.Counter()
    n_noupd_with_content = 0
    n_recoverable = 0
    recoverable_chars = 0
    noupd_content_chars = 0
    ex = collections.defaultdict(list)
    think_recoverable = 0
    think_forms = collections.Counter()
    n_noupd_nocontent_with_think = 0
    think_chars = 0

    for fp in files:
        try: lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError: continue
        for ln in lines:
            if '"transcript"' not in ln: continue
            try: ev = json.loads(ln)
            except Exception: continue
            tr = ev.get("transcript")
            if not isinstance(tr, str) or not tr: continue
            cur = None; units = []
            for label, body in split_sections(tr):
                if label == "MODEL RESPONSE META":
                    if cur: units.append(cur)
                    cur = {"a": "", "t": ""}
                elif cur is not None:
                    if label == "ASSISTANT": cur["a"] += "\n" + body
                    elif label == "THINKING": cur["t"] += "\n" + body
            if cur: units.append(cur)
            for u in units:
                a, t = u["a"].strip(), u["t"].strip()
                if effective_update(a): continue
                if a:
                    n_noupd_with_content += 1
                    noupd_content_chars += len(a)
                    hits = HEADERISH.findall(a)
                    lines_hit = [l for l in a.splitlines()
                                 if HEADERISH.match(l.strip()) or NEARMISS.match(l.strip())]
                    if lines_hit:
                        n_recoverable += 1
                        recoverable_chars += len(a)
                        f = norm_form(lines_hit[0])
                        forms[f] += 1
                        if len(ex[f]) < 2:
                            ex[f].append(a[:400])
                else:
                    if t:
                        n_noupd_nocontent_with_think += 1
                        think_chars += len(t)
                        tl = [l for l in t.splitlines()
                              if HEADERISH.match(l.strip()) or NEARMISS.match(l.strip())]
                        if tl:
                            think_recoverable += 1
                            think_forms[norm_form(tl[0])] += 1

    print(json.dumps({
        "no_update_responses_WITH_assistant_content": n_noupd_with_content,
        "  of_which_contain_a_slot_header_line (RECOVERABLE)": n_recoverable,
        "  recoverable_pct_of_that_bucket": f"{100*n_recoverable/max(1,n_noupd_with_content):.1f}%",
        "mean_chars_of_no_update_assistant_content": round(noupd_content_chars/max(1,n_noupd_with_content), 1),
        "mean_chars_of_recoverable_ones": round(recoverable_chars/max(1,n_recoverable), 1),
        "no_update_responses_NO_content_but_thinking": n_noupd_nocontent_with_think,
        "  of_which_thinking_has_slot_header (RECOVERABLE, wrong channel)": think_recoverable,
        "  think_recoverable_pct": f"{100*think_recoverable/max(1,n_noupd_nocontent_with_think):.2f}%",
        "mean_thinking_chars_when_no_content": round(think_chars/max(1,n_noupd_nocontent_with_think), 1),
        "top_40_near_miss_header_forms_assistant": forms.most_common(40),
        "top_15_near_miss_header_forms_thinking": think_forms.most_common(15),
    }, indent=2, ensure_ascii=False))
    Path(RUNS / "_wm_nearmiss_examples_0814.json").write_text(
        json.dumps({k: v for k, v in list(ex.items())[:200]}, indent=2, ensure_ascii=False), encoding="utf-8")

if __name__ == "__main__":
    main()
