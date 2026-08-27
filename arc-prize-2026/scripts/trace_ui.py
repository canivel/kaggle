"""Local web UI over the trace store — see what the model actually did.

    .venv/bin/python scripts/trace_ui.py        # http://127.0.0.1:7777
    .venv/bin/python scripts/trace_ui.py --port 8100

Read-only, binds to localhost, no auth by design — it is a lens on a local
SQLite file, not a service.

    /              dashboard: storage, call health, truncation, recent calls
    /calls         filterable call list (finish_reason, run, has tool_call)
    /call/<id>     deep dive: reasoning, content, tool calls, and the prompt
    /games         per-game outcomes

The call view is the point: it decompresses the stored blob and shows the
model's ACTUAL reasoning next to what it was sent, which is what makes a
failure diagnosable rather than merely counted.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

from flask import Flask, request

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trace_store as ts  # noqa: E402

app = Flask(__name__)

CSS = """
:root{
  --bg:#f6f8f8; --panel:#fff; --panel2:#eef2f3; --ink:#101819; --ink2:#3b4a4d;
  --ink3:#6b7c80; --rule:#d8e0e1; --accent:#0c6e77; --accent-soft:#e2f0f1;
  --good:#2c6e52; --good-soft:#e3efe9; --warn:#9a6415; --warn-soft:#f6eddd;
  --crit:#a33630; --crit-soft:#f7e6e4; --code:#101819; --code-ink:#dce7e8;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#0b1112; --panel:#121a1c; --panel2:#182224; --ink:#e6edee; --ink2:#b2c0c2;
  --ink3:#7e8f92; --rule:#233033; --accent:#4fbbc4; --accent-soft:#12292c;
  --good:#6fc49b; --good-soft:#132621; --warn:#d9a65a; --warn-soft:#2a2115;
  --crit:#e38079; --crit-soft:#2b1917; --code:#070c0d; --code-ink:#d3e0e1;
}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:14px/1.6 "IBM Plex Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
code,pre,.mono{font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace}
header{border-bottom:1px solid var(--rule);background:var(--panel);
  padding:.85rem clamp(1rem,3vw,2rem);display:flex;align-items:baseline;
  gap:1.4rem;flex-wrap:wrap;position:sticky;top:0;z-index:5}
header h1{font-size:1rem;margin:0;letter-spacing:-.01em}
header .lane{font-family:"IBM Plex Mono",monospace;font-size:.7rem;
  letter-spacing:.1em;text-transform:uppercase;color:var(--accent);
  background:var(--accent-soft);padding:.15rem .5rem;border-radius:3px}
nav a{color:var(--ink2);text-decoration:none;margin-right:1.1rem;font-size:.9rem}
nav a:hover,nav a.on{color:var(--accent)}
main{max-width:1240px;margin:0 auto;padding:clamp(1rem,3vw,2rem)}
h2{font-size:.78rem;letter-spacing:.09em;text-transform:uppercase;
  color:var(--ink3);margin:2rem 0 .6rem;font-family:"IBM Plex Mono",monospace}
h2:first-child{margin-top:0}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:.7rem}
.card{background:var(--panel);border:1px solid var(--rule);border-radius:7px;padding:.8rem 1rem}
.card .k{font-size:.68rem;letter-spacing:.08em;text-transform:uppercase;color:var(--ink3);
  font-family:"IBM Plex Mono",monospace}
.card .v{font-size:1.5rem;font-variant-numeric:tabular-nums;margin-top:.15rem;letter-spacing:-.02em}
.card .s{font-size:.76rem;color:var(--ink3)}
.wrap{overflow-x:auto;background:var(--panel);border:1px solid var(--rule);border-radius:7px}
table{border-collapse:collapse;width:100%;font-size:.86rem}
th,td{text-align:left;padding:.5rem .75rem;border-bottom:1px solid var(--rule);white-space:nowrap}
th{font-family:"IBM Plex Mono",monospace;font-size:.68rem;letter-spacing:.07em;
  text-transform:uppercase;color:var(--ink3);background:var(--panel2)}
tbody tr:last-child td{border-bottom:none}
tbody tr:hover{background:var(--panel2)}
td.num{font-variant-numeric:tabular-nums;text-align:right}
a.id{color:var(--accent);text-decoration:none;font-family:"IBM Plex Mono",monospace}
.pill{display:inline-block;font-family:"IBM Plex Mono",monospace;font-size:.68rem;
  padding:.12rem .45rem;border-radius:3px;letter-spacing:.04em}
.ok{background:var(--good-soft);color:var(--good)}
.bad{background:var(--crit-soft);color:var(--crit)}
.warnp{background:var(--warn-soft);color:var(--warn)}
.note{border-left:3px solid var(--warn);background:var(--warn-soft);padding:.8rem 1rem;
  border-radius:0 6px 6px 0;margin:.8rem 0;max-width:80ch}
.note.good{border-left-color:var(--good);background:var(--good-soft)}
pre.block{background:var(--code);color:var(--code-ink);padding:1rem;border-radius:7px;
  overflow-x:auto;white-space:pre-wrap;word-break:break-word;font-size:.82rem;
  max-height:34rem;overflow-y:auto;margin:.5rem 0}
details{background:var(--panel);border:1px solid var(--rule);border-radius:7px;
  padding:.7rem .9rem;margin:.6rem 0}
summary{cursor:pointer;font-family:"IBM Plex Mono",monospace;font-size:.8rem;color:var(--ink2)}
.meta{display:flex;gap:1.4rem;flex-wrap:wrap;font-family:"IBM Plex Mono",monospace;
  font-size:.78rem;color:var(--ink3);margin:.4rem 0 1rem}
.meta b{color:var(--ink2);font-weight:500}
form.filters{display:flex;gap:.5rem;flex-wrap:wrap;margin-bottom:.8rem}
select,input[type=submit]{font:inherit;font-size:.84rem;padding:.32rem .5rem;
  border:1px solid var(--rule);border-radius:5px;background:var(--panel);color:var(--ink)}
input[type=submit]{cursor:pointer;background:var(--accent-soft);color:var(--accent);border-color:var(--accent)}
.empty{color:var(--ink3);padding:1.2rem;text-align:center}
"""


def page(title: str, body: str, active: str = "") -> str:
    nav = "".join(
        f'<a href="{h}" class="{"on" if active == k else ""}">{lbl}</a>'
        for k, h, lbl in [("home", "/", "Dashboard"), ("calls", "/calls", "Calls"),
                          ("games", "/games", "Games")])
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)} — Trace</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{CSS}</style></head><body>
<header><h1>ARC trace</h1><span class="lane">MAC-SCREEN · never certifies</span>
<nav>{nav}</nav></header><main>{body}</main></body></html>"""


def _card(k, v, s=""):
    return (f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div>'
            f'<div class="s">{s}</div></div>')


@app.route("/")
def home():
    con = ts.connect()
    s = ts.stats(con)
    row = con.execute("SELECT MIN(ts), MAX(ts), AVG(elapsed_s), MAX(prompt_tokens)"
                      " FROM call").fetchone()
    lo, hi, avg, maxp = row if row else (None, None, 0, 0)

    cards = "".join([
        _card("calls", s["calls"], f"{s['game_runs']} game runs"),
        _card("stored", f"{s['stored_mb']} MB", f"{s['raw_mb']} MB raw · {s['compression']}"),
        _card("unique blobs", s["blobs"], f"dedupe via {s['codec']}"),
        _card("avg call", f"{(avg or 0):.0f}s", f"largest prompt {maxp or 0} tok"),
    ])

    health = list(con.execute(
        "SELECT finish_reason, COUNT(*), AVG(completion_tokens), AVG(elapsed_s),"
        " SUM(CASE WHEN n_tool_calls>0 THEN 1 ELSE 0 END) FROM call"
        " WHERE prompt_tokens>1000 GROUP BY finish_reason ORDER BY COUNT(*) DESC"))
    total = sum(r[1] for r in health) or 1
    trunc = sum(r[1] for r in health if r[0] == "length")
    rows = "".join(
        f'<tr><td><span class="pill {"bad" if fr=="length" else "ok"}">{fr}</span></td>'
        f'<td class="num">{n}</td><td class="num">{(c or 0):.0f}</td>'
        f'<td class="num">{(sec or 0):.0f}s</td><td class="num">{t}</td></tr>'
        for fr, n, c, sec, t in health)

    if trunc:
        pct = 100 * trunc / total
        warn = (f'<div class="note"><b>{trunc} of {total} calls ({pct:.0f}%) hit the token '
                'cap mid-response.</b><br>A truncated call emits no tool_call, so the harness '
                'retries the same turn — it reads as a reasoning failure but is a serving cap. '
                'Raise <code>--max-tokens</code> in <code>serve_local_model.sh</code>. '
                'Note this aggregate spans <em>all</em> runs, so a mixed pre/post-fix window '
                'shows both; filter by time on the Calls page to isolate one run.</div>')
    else:
        warn = (f'<div class="note good">No truncation across {total} harness calls — '
                'the token cap is not binding.</div>')

    recent = list(con.execute(
        "SELECT id, ts, elapsed_s, prompt_tokens, completion_tokens, think_chars,"
        " n_tool_calls, finish_reason FROM call ORDER BY id DESC LIMIT 15"))
    rrows = "".join(
        f'<tr><td><a class="id" href="/call/{i}">#{i}</a></td><td class="mono">{t[11:19]}</td>'
        f'<td class="num">{(e or 0):.0f}s</td><td class="num">{p or 0}</td>'
        f'<td class="num">{c or 0}</td><td class="num">{th or 0}</td>'
        f'<td class="num">{tc or 0}</td>'
        f'<td><span class="pill {"bad" if fr=="length" else "ok" if fr else ""}">{fr or "-"}</span></td></tr>'
        for i, t, e, p, c, th, tc, fr in recent)

    body = f"""<h2>Store</h2><div class="cards">{cards}</div>
<div class="meta"><span>window <b>{(lo or "")[:19]} → {(hi or "")[:19]}</b></span>
<span>db <b>{s['db_mb']} MB</b></span></div>
<h2>Call health — harness calls only</h2>{warn}
<div class="wrap"><table><thead><tr><th>finish_reason</th><th>n</th>
<th>avg completion tok</th><th>avg time</th><th>with tool_call</th></tr></thead>
<tbody>{rows or '<tr><td colspan=5 class=empty>no harness calls yet</td></tr>'}</tbody></table></div>
<h2>Recent calls</h2>
<div class="wrap"><table><thead><tr><th>id</th><th>time</th><th>secs</th><th>prompt</th>
<th>compl</th><th>think</th><th>tools</th><th>finish</th></tr></thead>
<tbody>{rrows or '<tr><td colspan=8 class=empty>nothing recorded yet</td></tr>'}</tbody></table></div>"""
    return page("Dashboard", body, "home")


@app.route("/calls")
def calls():
    con = ts.connect()
    fr = request.args.get("finish", "")
    tools = request.args.get("tools", "")
    where, args = ["1=1"], []
    if fr:
        where.append("finish_reason = ?"); args.append(fr)
    if tools == "yes":
        where.append("n_tool_calls > 0")
    elif tools == "no":
        where.append("COALESCE(n_tool_calls,0) = 0")
    sql = ("SELECT id, ts, elapsed_s, prompt_tokens, completion_tokens, think_chars,"
           " n_tool_calls, finish_reason FROM call WHERE " + " AND ".join(where) +
           " ORDER BY id DESC LIMIT 400")
    rows = list(con.execute(sql, args))
    opts = [r[0] for r in con.execute(
        "SELECT DISTINCT finish_reason FROM call WHERE finish_reason IS NOT NULL")]
    sel = "".join(f'<option {"selected" if o==fr else ""}>{o}</option>' for o in opts)
    trs = "".join(
        f'<tr><td><a class="id" href="/call/{i}">#{i}</a></td><td class="mono">{t[:19]}</td>'
        f'<td class="num">{(e or 0):.0f}s</td><td class="num">{p or 0}</td>'
        f'<td class="num">{c or 0}</td><td class="num">{th or 0}</td>'
        f'<td class="num">{tc or 0}</td>'
        f'<td><span class="pill {"bad" if f2=="length" else "ok" if f2 else ""}">{f2 or "-"}</span></td></tr>'
        for i, t, e, p, c, th, tc, f2 in rows)
    body = f"""<h2>Calls — {len(rows)} shown</h2>
<form class="filters" method="get">
<select name="finish"><option value="">any finish_reason</option>{sel}</select>
<select name="tools"><option value="">tool call: any</option>
<option value="yes" {"selected" if tools=="yes" else ""}>emitted a tool call</option>
<option value="no" {"selected" if tools=="no" else ""}>no tool call</option></select>
<input type="submit" value="Filter"></form>
<div class="wrap"><table><thead><tr><th>id</th><th>timestamp</th><th>secs</th>
<th>prompt</th><th>compl</th><th>think</th><th>tools</th><th>finish</th></tr></thead>
<tbody>{trs or '<tr><td colspan=8 class=empty>no matching calls</td></tr>'}</tbody></table></div>"""
    return page("Calls", body, "calls")


@app.route("/call/<int:cid>")
def call(cid: int):
    con = ts.connect()
    r = con.execute(
        "SELECT ts, elapsed_s, prompt_tokens, completion_tokens, finish_reason,"
        " request_sha, response_sha, think_chars, n_tool_calls, status"
        " FROM call WHERE id=?", (cid,)).fetchone()
    if not r:
        return page("Not found", f"<div class='empty'>no call #{cid}</div>"), 404
    tsx, secs, ptok, ctok, fr, rq, rs, think, ntools, status = r

    resp = ts.get_blob(con, rs) or {}
    msg = ((resp.get("choices") or [{}])[0]).get("message") or {}
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    content = msg.get("content") or ""
    tcalls = msg.get("tool_calls") or []

    parts = [f"""<h2>Call #{cid}</h2>
<div class="meta"><span><b>{tsx[:19]}</b></span><span>took <b>{(secs or 0):.0f}s</b></span>
<span>status <b>{status}</b></span><span>prompt <b>{ptok or 0}</b> tok</span>
<span>completion <b>{ctok or 0}</b> tok</span>
<span>finish <b>{fr}</b></span></div>"""]

    if fr == "length":
        parts.append('<div class="note"><b>Truncated at the token cap.</b> The response was '
                     'cut mid-generation, so no tool_call could be emitted and the harness '
                     'will retry this same turn. This is a serving limit, not a reasoning '
                     'failure.</div>')

    if reasoning:
        parts.append(f'<h2>Reasoning — {len(reasoning)} chars</h2>'
                     f'<pre class="block">{html.escape(reasoning)}</pre>')
    if content:
        parts.append(f'<h2>Content — {len(content)} chars</h2>'
                     f'<pre class="block">{html.escape(content)}</pre>')
    if tcalls:
        rendered = "\n\n".join(
            f"{(t.get('function') or {}).get('name')}\n"
            f"{(t.get('function') or {}).get('arguments')}" for t in tcalls)
        parts.append(f'<h2>Tool calls — {len(tcalls)}</h2>'
                     f'<pre class="block">{html.escape(rendered)}</pre>')
    if not (reasoning or content or tcalls):
        parts.append('<div class="note">Empty response — nothing was emitted before the '
                     'call ended.</div>')

    req = ts.get_blob(con, rq) or {}
    msgs = req.get("messages") or []
    parts.append(f'<h2>What was sent</h2><div class="meta">'
                 f'<span><b>{len(msgs)}</b> messages</span>'
                 f'<span><b>{len(req.get("tools") or [])}</b> tools</span>'
                 f'<span>max_tokens <b>{req.get("max_tokens")}</b></span>'
                 f'<span>temperature <b>{req.get("temperature")}</b></span></div>')
    for i, m in enumerate(msgs):
        c = m.get("content")
        text = c if isinstance(c, str) else json.dumps(c, indent=2)
        text = text or ""
        parts.append(
            f'<details><summary>{i}. {m.get("role")} — {len(text)} chars</summary>'
            f'<pre class="block">{html.escape(text[:20000])}</pre></details>')
    return page(f"Call {cid}", "".join(parts), "calls")


@app.route("/games")
def games():
    con = ts.connect()
    rows = list(con.execute(
        "SELECT r.label, g.game_id, g.levels_completed, g.number_of_levels, g.actions,"
        " g.final_score, g.state, g.solver_note FROM game_run g"
        " LEFT JOIN run r ON r.id=g.run_id ORDER BY g.id DESC LIMIT 200"))
    trs = "".join(
        f'<tr><td>{html.escape(str(lbl))}</td><td class="mono">{html.escape(str(gid))}</td>'
        f'<td class="num">{lc}/{nl}</td><td class="num">{acts}</td>'
        f'<td class="num">{score}</td><td>{html.escape(str(state))}</td>'
        f'<td>{html.escape((note or "")[:110])}</td></tr>'
        for lbl, gid, lc, nl, acts, score, state, note in rows)
    body = f"""<h2>Game outcomes</h2>
<div class="wrap"><table><thead><tr><th>run</th><th>game</th><th>levels</th>
<th>actions</th><th>score</th><th>state</th><th>solver note</th></tr></thead>
<tbody>{trs or '<tr><td colspan=7 class=empty>no completed game runs yet — a run must finish to record these</td></tr>'}</tbody></table></div>"""
    return page("Games", body, "games")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7777)
    a = ap.parse_args()
    print(f"trace UI -> http://127.0.0.1:{a.port}   (db: {ts.DB_PATH})", flush=True)
    app.run(host="127.0.0.1", port=a.port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
