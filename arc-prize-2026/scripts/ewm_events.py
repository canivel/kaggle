"""EWMEVT aggregator — event-shaped canary accounting for the EWM-execute line.

Parses `EWMEVT` lines (schema: duck_eval/ewm_exec/EVENT_SCHEMA.md) from a run
log and produces the Stage-0 gate accounting that totals-shaped counters
(scripts/predict_metric.py) cannot express:

  * per-game plan/step tables + plan-length and abort-step distributions
  * post-abort survival (did a plan_start or fallback follow within N events)
  * deadlock detection (mismatch_abort with no subsequent progress event)
  * A10 canary verdict   : >=1 executed plan on >=5 games
  * GSME activation prong-0: mechanism ran AND produced verified outcomes

Usage:
  uv run python scripts/ewm_events.py <logfile> [--json out.json] [--n-recovery 25]
  uv run python scripts/ewm_events.py --selftest

CPU-only, read-only, stdlib-only (kernel-embeddable).
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from collections import Counter, defaultdict

ANCHOR = "EWMEVT "
PROGRESS_KINDS = {"plan_start", "plan_step", "plan_done", "fallback"}
KINDS = PROGRESS_KINDS | {"mismatch_abort", "trunc"}
INT_KEYS = {"v", "plan", "len", "step", "match", "lvl", "steps", "lvl_done",
            "dropped_after", "sample"}

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #

def parse_line(line: str):
    """Parse one log line -> event dict, or None if not a valid EWMEVT line.

    Tolerant of: prefixes before the EWMEVT anchor (timestamps), unknown keys,
    and mid-token truncation (a trailing token without '=' or with a garbled
    value is dropped; the event survives if kind= and game= survived).
    """
    i = line.find(ANCHOR)
    if i < 0:
        return None
    ev = {}
    for tok in line[i + len(ANCHOR):].split():
        if "=" not in tok:
            continue  # truncated / junk token
        k, _, val = tok.partition("=")
        if not k or not val:
            continue
        if k in INT_KEYS:
            try:
                ev[k] = int(val)
            except ValueError:
                continue  # garbled int (mid-line cut) -> drop token only
        else:
            ev[k] = val
    if ev.get("kind") not in KINDS or "game" not in ev:
        return None
    return ev


def parse_log(fp_or_lines):
    """Parse a path or an iterable of lines -> (events, n_malformed).

    n_malformed counts lines that contain the EWMEVT anchor but did not yield
    a valid event (e.g. cut before kind=/game=).
    """
    if isinstance(fp_or_lines, str):
        with open(fp_or_lines, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    else:
        lines = list(fp_or_lines)
    events, malformed = [], 0
    for ln in lines:
        if ANCHOR not in ln:
            continue
        ev = parse_line(ln)
        if ev is None:
            malformed += 1
        else:
            events.append(ev)
    return events, malformed


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #

def aggregate(events, n_recovery: int = 25):
    """Aggregate parsed events -> per-game table + run-level verdicts.

    Semantics sealed in EVENT_SCHEMA.md §Aggregator semantics.
    """
    by_game = defaultdict(list)
    for ev in events:
        by_game[ev["game"]].append(ev)

    games = {}
    for gid, evs in sorted(by_game.items()):
        plans = [e for e in evs if e["kind"] == "plan_start"]
        steps = [e for e in evs if e["kind"] == "plan_step"]
        dones = [e for e in evs if e["kind"] == "plan_done"]
        aborts = [e for e in evs if e["kind"] == "mismatch_abort"]
        fbacks = [e for e in evs if e["kind"] == "fallback"]

        # plans that executed >=1 step (canary needs executed, not just planned)
        executed_plan_ids = {e.get("plan") for e in steps}
        matches = sum(e.get("match", 0) for e in steps)

        # post-abort survival + deadlock (ordering properties)
        survived = deadlocked = 0
        abort_steps = []
        for idx, e in enumerate(evs):
            if e["kind"] != "mismatch_abort":
                continue
            abort_steps.append(e.get("step", -1))
            window = evs[idx + 1: idx + 1 + n_recovery]
            if any(w["kind"] in ("plan_start", "fallback") for w in window):
                survived += 1
            if not any(w["kind"] in PROGRESS_KINDS for w in evs[idx + 1:]):
                deadlocked += 1

        games[gid] = {
            "plans": len(plans),
            "plans_executed": len(executed_plan_ids),
            "steps": len(steps),
            "matches": matches,
            "step_acc": (matches / len(steps)) if steps else None,
            "plan_lengths": sorted(e.get("len", -1) for e in plans),
            "plans_done": len(dones),
            "aborts": len(aborts),
            "abort_steps": sorted(abort_steps),
            "abort_reasons": dict(Counter(e.get("reason", "?") for e in aborts)),
            "fallbacks": len(fbacks),
            "aborts_survived": survived,
            "deadlocks": deadlocked,
            "truncated": any(e["kind"] == "trunc" for e in evs),
        }

    # ---- run-level verdicts ----
    fired_games = sorted(g for g, s in games.items()
                         if s["plans_executed"] >= 1)
    tot = {k: sum(g[k] for g in games.values())
           for k in ("plans", "steps", "plans_done", "aborts", "fallbacks",
                     "deadlocks", "aborts_survived")}
    canary_pass = len(fired_games) >= 5
    outcomes = tot["plans_done"] + tot["aborts"]
    active = tot["plans"] >= 1 and tot["steps"] >= 1 and outcomes >= 1

    return {
        "n_events": len(events),
        "n_games": len(games),
        "games": games,
        "totals": tot,
        "canary": {
            "fired_games": fired_games,
            "n_fired": len(fired_games),
            "threshold": 5,
            "verdict": "PASS" if canary_pass else "FAIL",
        },
        "activation": {
            "plans": tot["plans"],
            "steps": tot["steps"],
            "outcomes": outcomes,
            "deadlocks": tot["deadlocks"],
            "verdict": "ACTIVE" if active else "INERT",
        },
    }


def verdict_lines(agg):
    c, a = agg["canary"], agg["activation"]
    return [
        f"EWM_CANARY games_fired={c['n_fired']} threshold={c['threshold']} "
        f"verdict={c['verdict']} fired=[{','.join(c['fired_games'])}]",
        f"EWM_ACTIVATION plans={a['plans']} steps={a['steps']} "
        f"outcomes={a['outcomes']} deadlocks={a['deadlocks']} "
        f"verdict={a['verdict']}",
    ]


def render_table(agg):
    rows = ["| game | plans | steps | step_acc | plan lens | aborts | "
            "abort steps | survived | deadlocks | fallbacks |",
            "|---|---:|---:|---:|---|---:|---|---:|---:|---:|"]
    for gid, g in agg["games"].items():
        acc = f"{g['step_acc']:.3f}" if g["step_acc"] is not None else "-"
        rows.append(
            f"| {gid} | {g['plans']} | {g['steps']} | {acc} | "
            f"{_dist(g['plan_lengths'])} | {g['aborts']} | "
            f"{_dist(g['abort_steps'])} | {g['aborts_survived']} | "
            f"{g['deadlocks']} | {g['fallbacks']} |")
    return rows


def _dist(vals, cap=8):
    """Compact multiset display: '0x51,3x2,12x1' = value 0 seen 51 times..."""
    if not vals:
        return "-"
    cnt = Counter(vals)
    items = sorted(cnt.items())[:cap]
    s = ",".join(f"{v}x{n}" for v, n in items)
    return s + (",.." if len(cnt) > cap else "")


# --------------------------------------------------------------------------- #
# selftest — synthetic logs
# --------------------------------------------------------------------------- #

def _selftest():
    n_pass = n_fail = 0

    def check(name, cond):
        nonlocal n_pass, n_fail
        if cond:
            n_pass += 1
        else:
            n_fail += 1
            print(f"  FAIL: {name}")

    # 1. clean plan
    log = [
        "EWMEVT v=1 kind=plan_start game=ls20 plan=0 len=3 sim=ls20_sim:deadbeef gv=ls20-9607627b lvl=1 t=1.0",
        "EWMEVT v=1 kind=plan_step game=ls20 plan=0 step=0 act=A1 pred=aaaaaaaa obs=aaaaaaaa match=1 lvl=1 t=1.5",
        "EWMEVT v=1 kind=plan_step game=ls20 plan=0 step=1 act=A2 pred=bbbbbbbb obs=bbbbbbbb match=1 lvl=1 t=2.0",
        "EWMEVT v=1 kind=plan_step game=ls20 plan=0 step=2 act=A4 pred=cccccccc obs=cccccccc match=1 lvl=1 t=2.5",
        "EWMEVT v=1 kind=plan_done game=ls20 plan=0 len=3 steps=3 lvl_done=1 t=2.6",
    ]
    evs, bad = parse_log(log)
    agg = aggregate(evs)
    g = agg["games"]["ls20"]
    check("clean: 5 events, 0 malformed", len(evs) == 5 and bad == 0)
    check("clean: 1 plan, 3 steps, acc 1.0",
          g["plans"] == 1 and g["steps"] == 3 and g["step_acc"] == 1.0)
    check("clean: 0 aborts, 0 deadlocks, 1 done",
          g["aborts"] == 0 and g["deadlocks"] == 0 and g["plans_done"] == 1)
    check("clean: activation ACTIVE",
          agg["activation"]["verdict"] == "ACTIVE")
    check("clean: canary FAIL (1 game < 5)",
          agg["canary"]["verdict"] == "FAIL" and agg["canary"]["n_fired"] == 1)

    # 2. mid-plan abort with recovery
    log = [
        "EWMEVT v=1 kind=plan_start game=tr87 plan=0 len=5 sim=tr87_sim:cafe0123 gv=tr87-cd924810 lvl=1 t=1.0",
        "EWMEVT v=1 kind=plan_step game=tr87 plan=0 step=0 act=A1 pred=aaaaaaaa obs=aaaaaaaa match=1 lvl=1 t=1.2",
        "EWMEVT v=1 kind=plan_step game=tr87 plan=0 step=1 act=A3 pred=11111111 obs=22222222 match=0 lvl=1 t=1.4",
        "EWMEVT v=1 kind=mismatch_abort game=tr87 plan=0 step=1 len=5 reason=mismatch pred=11111111 obs=22222222 t=1.4",
        "EWMEVT v=1 kind=fallback game=tr87 plan=0 reason=mismatch t=1.5",
        "EWMEVT v=1 kind=plan_start game=tr87 plan=1 len=2 sim=tr87_sim:cafe0123 gv=tr87-cd924810 lvl=1 t=9.0",
        "EWMEVT v=1 kind=plan_step game=tr87 plan=1 step=0 act=A1 pred=dddddddd obs=dddddddd match=1 lvl=1 t=9.2",
        "EWMEVT v=1 kind=plan_done game=tr87 plan=1 len=2 steps=2 lvl_done=0 t=9.4",
    ]
    agg = aggregate(parse_log(log)[0])
    g = agg["games"]["tr87"]
    check("recovery: 1 abort at step 1", g["aborts"] == 1 and g["abort_steps"] == [1])
    check("recovery: survived=1, deadlocks=0",
          g["aborts_survived"] == 1 and g["deadlocks"] == 0)
    check("recovery: 2 plans, 3 steps, acc 2/3",
          g["plans"] == 2 and g["steps"] == 3
          and abs(g["step_acc"] - 2 / 3) < 1e-9)

    # 3. abort then deadlock (nothing after the abort)
    log = [
        "EWMEVT v=1 kind=plan_start game=vc33 plan=0 len=4 sim=vc33_sim:00ff00ff gv=vc33-5430563c lvl=1 t=1.0",
        "EWMEVT v=1 kind=plan_step game=vc33 plan=0 step=0 act=A6:10,60 pred=aaaaaaaa obs=ffffffff match=0 lvl=1 t=1.3",
        "EWMEVT v=1 kind=mismatch_abort game=vc33 plan=0 step=0 len=4 reason=mismatch pred=aaaaaaaa obs=ffffffff t=1.3",
    ]
    agg = aggregate(parse_log(log)[0])
    g = agg["games"]["vc33"]
    check("deadlock: survived=0, deadlocks=1",
          g["aborts_survived"] == 0 and g["deadlocks"] == 1)
    check("deadlock: activation still ACTIVE (outcome exists)",
          agg["activation"]["verdict"] == "ACTIVE"
          and agg["activation"]["deadlocks"] == 1)

    # 4. truncated log (mid-line cut on the final line)
    log = [
        "EWMEVT v=1 kind=plan_start game=sp80 plan=0 len=2 sim=sp80_sim:12341234 gv=sp80-589a99af lvl=1 t=1.0",
        "EWMEVT v=1 kind=plan_step game=sp80 plan=0 step=0 act=A2 pred=aaaaaaaa obs=aaaaaaaa match=1 lvl=1 t=1.1",
        "EWMEVT v=1 kind=plan_step game=sp80 plan=0 step=1 act=A2 pred=bbbbbbbb obs=bbbbbb",  # cut mid-token
        "EWMEVT v=1 kind=pl",  # cut before kind parsed fully -> malformed
    ]
    evs, bad = parse_log(log)
    agg = aggregate(evs)
    g = agg["games"]["sp80"]
    check("trunc: 3 events kept, 1 malformed", len(evs) == 3 and bad == 1)
    check("trunc: cut step kept w/o match (counts as unmatched)",
          g["steps"] == 2 and g["matches"] == 1)

    # 5. zero-event run
    evs, bad = parse_log(["harness boot", "no ewm lines here", ""])
    agg = aggregate(evs)
    check("zero: no events, no games", agg["n_events"] == 0 and agg["n_games"] == 0)
    check("zero: canary FAIL, activation INERT",
          agg["canary"]["verdict"] == "FAIL"
          and agg["activation"]["verdict"] == "INERT")

    # 6. canary PASS across 5 games + prefix junk tolerated
    log = []
    for i, gid in enumerate(["ls20", "tr87", "tu93", "sp80", "lf52"]):
        log.append(f"2026-07-18T12:00:0{i}Z kern| EWMEVT v=1 kind=plan_start game={gid} plan=0 len=1 sim=x:0 gv={gid}-x lvl=1 t=1.0")
        log.append(f"2026-07-18T12:00:0{i}Z kern| EWMEVT v=1 kind=plan_step game={gid} plan=0 step=0 act=A1 pred=aaaaaaaa obs=aaaaaaaa match=1 lvl=1 t=1.1")
        log.append(f"2026-07-18T12:00:0{i}Z kern| EWMEVT v=1 kind=plan_done game={gid} plan=0 len=1 steps=1 lvl_done=0 t=1.2")
    agg = aggregate(parse_log(log)[0])
    check("canary: 5 games fired -> PASS",
          agg["canary"]["verdict"] == "PASS" and agg["canary"]["n_fired"] == 5)
    check("canary: prefix-stamped lines parsed", agg["n_events"] == 15)

    # 7. planned-but-never-executed plans don't fire the canary
    log = ["EWMEVT v=1 kind=plan_start game=ft09 plan=0 len=9 sim=x:0 gv=ft09-x lvl=1 t=1.0",
           "EWMEVT v=1 kind=fallback game=ft09 plan=0 reason=budget t=1.0"]
    agg = aggregate(parse_log(log)[0])
    check("no-exec: plan w/o steps not counted as fired",
          agg["canary"]["n_fired"] == 0
          and agg["games"]["ft09"]["plans_executed"] == 0)

    print(f"selftest: {n_pass} passed, {n_fail} failed")
    return n_fail


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile", nargs="?", help="run log containing EWMEVT lines")
    ap.add_argument("--json", help="write full aggregate to this JSON path")
    ap.add_argument("--n-recovery", type=int, default=25,
                    help="post-abort survival window (events)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(1 if _selftest() else 0)
    if not args.logfile:
        ap.error("logfile required (or --selftest)")

    events, malformed = parse_log(args.logfile)
    agg = aggregate(events, n_recovery=args.n_recovery)
    agg["n_malformed_lines"] = malformed

    print(f"ewm_events: {agg['n_events']} events ({malformed} malformed lines) "
          f"across {agg['n_games']} games")
    for row in render_table(agg):
        print(row)
    for line in verdict_lines(agg):
        print(line)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(agg, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
