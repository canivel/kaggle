"""Pass 3: per-run and per-family breakdown; finish_reason tabulation; why the carry is empty."""
from __future__ import annotations
import json, re, collections, statistics
from pathlib import Path

RUNS = Path(r"F:\kaggle\arc-prize-2026\runs")
SECTION_RE = re.compile(r"^\[([A-Z][A-Z0-9 _:.\-]*)\]$")
CARRY_HDR = "Working world model carried from earlier turns:"
META_KV = re.compile(r"^(\w+):\s*(.*)$")

def split_sections(t):
    out, cl, cur = [], None, []
    for line in t.splitlines():
        m = SECTION_RE.match(line.strip()) if line.startswith("[") else None
        if m:
            if cl is not None: out.append((cl, "\n".join(cur).strip()))
            cl, cur = m.group(1), []
        elif cl is not None: cur.append(line)
    if cl is not None: out.append((cl, "\n".join(cur).strip()))
    return out

def main():
    files = sorted(p for p in RUNS.rglob("*_events.jsonl") if "tool_compaction" not in p.name)
    fam_stats = collections.defaultdict(collections.Counter)
    finish = collections.Counter()
    finish_content = collections.Counter()
    by_turn_idx_empty = collections.Counter(); by_turn_idx_n = collections.Counter()
    runs_never_populated = 0; runs_total = 0
    first_nonempty_idx = []
    frac_empty_per_run = []
    tc_count_hist = collections.Counter()

    for fp in files:
        fam = fp.relative_to(RUNS).parts[0]
        try: lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError: continue
        runs_total += 1
        idx = 0; n_empty = 0; first_ne = None
        for ln in lines:
            if '"transcript"' not in ln: continue
            try: ev = json.loads(ln)
            except Exception: continue
            tr = ev.get("transcript")
            if not isinstance(tr, str) or not tr: continue
            secs = split_sections(tr)
            fu = next((b for l, b in secs if l == "USER PROMPT"), None)
            if fu is None: continue
            empty = CARRY_HDR not in fu
            fam_stats[fam]["turns"] += 1
            if empty:
                fam_stats[fam]["empty_carry"] += 1; n_empty += 1
            elif first_ne is None:
                first_ne = idx
            if idx < 40:
                by_turn_idx_n[idx] += 1
                if empty: by_turn_idx_empty[idx] += 1
            idx += 1
            for l, b in secs:
                if l != "MODEL RESPONSE META": continue
                d = {}
                for line in b.splitlines():
                    m = META_KV.match(line.strip())
                    if m: d[m.group(1)] = m.group(2)
                fr = d.get("finish_reason", "?")
                cc = int(d.get("content_chars", "0") or 0)
                tc = d.get("tool_call_count", "?")
                finish[fr] += 1
                finish_content[(fr, "content>0" if cc > 0 else "content==0")] += 1
                tc_count_hist[tc] += 1
        if idx:
            frac_empty_per_run.append(n_empty / idx)
            if first_ne is None: runs_never_populated += 1
            else: first_nonempty_idx.append(first_ne)

    print(json.dumps({
        "runs_total": runs_total,
        "runs_where_store_NEVER_populated": runs_never_populated,
        "runs_never_pct": f"{100*runs_never_populated/max(1,runs_total):.1f}%",
        "first_nonempty_turn_idx": {
            "mean": round(statistics.mean(first_nonempty_idx), 2) if first_nonempty_idx else None,
            "median": statistics.median(first_nonempty_idx) if first_nonempty_idx else None,
            "p90": round(statistics.quantiles(first_nonempty_idx, n=10)[8], 1) if len(first_nonempty_idx) > 10 else None,
        },
        "mean_frac_of_run_with_EMPTY_carry": round(statistics.mean(frac_empty_per_run), 3) if frac_empty_per_run else None,
        "empty_carry_rate_by_turn_index_0_39": [
            f"{i}:{by_turn_idx_empty[i]}/{by_turn_idx_n[i]}={100*by_turn_idx_empty[i]/max(1,by_turn_idx_n[i]):.0f}%"
            for i in range(40)],
        "per_family": {k: {"turns": v["turns"], "empty_carry": v["empty_carry"],
                           "empty_pct": f"{100*v['empty_carry']/max(1,v['turns']):.1f}%"}
                       for k, v in sorted(fam_stats.items())},
        "finish_reason": dict(finish.most_common()),
        "finish_reason_x_content": {f"{a}|{b}": c for (a, b), c in finish_content.most_common()},
        "tool_call_count_hist": dict(tc_count_hist.most_common(8)),
    }, indent=2))

if __name__ == "__main__":
    main()
