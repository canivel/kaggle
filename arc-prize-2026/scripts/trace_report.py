"""Read the trace store — what happened, and what the model actually said.

    scripts/trace_report.py                 # overview: storage, runs, health
    scripts/trace_report.py --calls 20      # recent calls, one line each
    scripts/trace_report.py --health        # finish_reason / truncation check
    scripts/trace_report.py --last          # full reasoning + answer of the last call
    scripts/trace_report.py --call 42       # ... of a specific call id
    scripts/trace_report.py --prompt 42     # what was SENT for that call
    scripts/trace_report.py --games         # per-game outcomes across runs

Everything is plain SQLite, so ad-hoc SQL works too:
    sqlite3 runs/traces.db "SELECT finish_reason, COUNT(*) FROM call GROUP BY 1"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trace_store as ts  # noqa: E402

BAR = "=" * 78


def overview(con) -> None:
    s = ts.stats(con)
    print(BAR)
    print(f"  TRACE STORE  {ts.DB_PATH}")
    print(BAR)
    print(f"  calls {s['calls']}   game_runs {s['game_runs']}   runs {s['runs']}")
    print(f"  payloads: {s['raw_mb']} MB raw -> {s['stored_mb']} MB stored "
          f"({s['compression']} via {s['codec']}, {s['blobs']} unique blobs)")
    print(f"  db file : {s['db_mb']} MB")
    rows = list(con.execute(
        "SELECT MIN(ts), MAX(ts), AVG(elapsed_s), MAX(prompt_tokens) FROM call"))
    if rows and rows[0][0]:
        lo, hi, avg, maxp = rows[0]
        print(f"  window  : {lo[:19]}  ->  {hi[:19]}")
        print(f"  avg call: {avg:.0f}s   largest prompt: {maxp} tokens")


def health(con) -> None:
    print(f"\n{BAR}\n  CALL HEALTH  (harness calls only, prompt > 1000 tokens)\n{BAR}")
    rows = list(con.execute(
        "SELECT finish_reason, COUNT(*), AVG(completion_tokens), AVG(elapsed_s),"
        " SUM(CASE WHEN n_tool_calls>0 THEN 1 ELSE 0 END)"
        " FROM call WHERE prompt_tokens > 1000 GROUP BY finish_reason"
        " ORDER BY COUNT(*) DESC"))
    if not rows:
        print("  no harness calls recorded yet")
        return
    print(f"  {'finish_reason':<14}{'n':>5}{'avg_ctok':>10}{'avg_s':>8}{'with_tool_call':>16}")
    total = sum(r[1] for r in rows)
    for fr, n, ctok, secs, tools in rows:
        flag = "  <-- TRUNCATED" if fr == "length" else ""
        print(f"  {str(fr):<14}{n:>5}{(ctok or 0):>10.0f}{(secs or 0):>8.0f}"
              f"{tools:>16}{flag}")
    trunc = sum(n for fr, n, *_ in rows if fr == "length")
    if trunc:
        print(f"\n  {trunc}/{total} calls hit the token cap mid-response.")
        print("  A truncated call emits no tool_call, so the harness retries the")
        print("  same turn -- it reads as a reasoning failure but is a serving cap.")
        print("  Raise --max-tokens in scripts/serve_local_model.sh.")
    else:
        print(f"\n  No truncation across {total} calls -- the token cap is not binding.")


def calls(con, n: int) -> None:
    print(f"\n{BAR}\n  LAST {n} CALLS\n{BAR}")
    print(f"  {'id':>5} {'time':<9}{'secs':>6}{'ptok':>7}{'ctok':>6}{'think':>7}"
          f"{'tools':>6}  finish")
    for r in con.execute(
            "SELECT id, ts, elapsed_s, prompt_tokens, completion_tokens,"
            " think_chars, n_tool_calls, finish_reason FROM call"
            " ORDER BY id DESC LIMIT ?", (n,)):
        cid, tsx, secs, p, c, th, tools, fr = r
        print(f"  {cid:>5} {tsx[11:19]:<9}{(secs or 0):>6.0f}{(p or 0):>7}"
              f"{(c or 0):>6}{(th or 0):>7}{(tools or 0):>6}  {fr or '-'}")


def show_call(con, cid: int | None, prompt: bool) -> None:
    if cid is None:
        row = con.execute("SELECT id FROM call ORDER BY id DESC LIMIT 1").fetchone()
        if not row:
            print("  no calls recorded")
            return
        cid = row[0]
    row = con.execute(
        "SELECT ts, elapsed_s, prompt_tokens, completion_tokens, finish_reason,"
        " request_sha, response_sha FROM call WHERE id=?", (cid,)).fetchone()
    if not row:
        print(f"  no call {cid}")
        return
    tsx, secs, p, c, fr, rq, rs = row
    print(f"\n{BAR}\n  CALL {cid}   {tsx[:19]}   {secs:.0f}s   "
          f"{p} prompt / {c} completion tokens   finish={fr}\n{BAR}")

    if prompt:
        body = ts.get_blob(con, rq) or {}
        msgs = body.get("messages") or []
        print(f"  REQUEST: {len(msgs)} messages, "
              f"{len(body.get('tools') or [])} tools, max_tokens={body.get('max_tokens')}")
        for m in msgs[-3:]:
            content = m.get("content")
            text = content if isinstance(content, str) else json.dumps(content)
            print(f"\n  --- {m.get('role')} ({len(text or '')} chars) ---")
            print("  " + (text or "")[:1500].replace("\n", "\n  "))
        return

    body = ts.get_blob(con, rs) or {}
    msg = ((body.get("choices") or [{}])[0]).get("message") or {}
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    content = msg.get("content") or ""
    tools = msg.get("tool_calls") or []

    if reasoning:
        print(f"\n  ---- REASONING ({len(reasoning)} chars) ----")
        print("  " + reasoning.strip()[:3000].replace("\n", "\n  "))
    if content:
        print(f"\n  ---- CONTENT ({len(content)} chars) ----")
        print("  " + content.strip()[:2000].replace("\n", "\n  "))
    if tools:
        print(f"\n  ---- TOOL CALLS ({len(tools)}) ----")
        for t in tools:
            fn = (t.get("function") or {})
            print(f"  {fn.get('name')}({str(fn.get('arguments'))[:400]})")
    if not (reasoning or content or tools):
        print("  (empty response -- likely truncated before emitting anything)")


def games(con) -> None:
    print(f"\n{BAR}\n  GAME OUTCOMES\n{BAR}")
    rows = list(con.execute(
        "SELECT r.label, g.game_id, g.levels_completed, g.number_of_levels,"
        " g.actions, g.final_score, g.state FROM game_run g"
        " LEFT JOIN run r ON r.id=g.run_id ORDER BY g.id DESC LIMIT 40"))
    if not rows:
        print("  no completed game runs yet (a run must finish to record these)")
        return
    for label, gid, lc, nl, acts, score, state in rows:
        mark = "OK " if (lc or 0) > 0 else ("!! " if not acts else " . ")
        print(f"  {mark}{str(label):<18}{str(gid):<22}lc={lc}/{nl:<3}"
              f"actions={acts:<5}score={score}  {state}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, metavar="N")
    ap.add_argument("--health", action="store_true")
    ap.add_argument("--last", action="store_true")
    ap.add_argument("--call", type=int, metavar="ID")
    ap.add_argument("--prompt", type=int, nargs="?", const=-1, metavar="ID")
    ap.add_argument("--games", action="store_true")
    a = ap.parse_args()

    con = ts.connect()
    if a.last or a.call is not None:
        show_call(con, a.call, prompt=False)
    elif a.prompt is not None:
        show_call(con, None if a.prompt == -1 else a.prompt, prompt=True)
    elif a.games:
        games(con)
    elif a.calls:
        calls(con, a.calls)
    elif a.health:
        overview(con); health(con)
    else:
        overview(con); health(con); calls(con, 10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
