"""Stage-0 dry-run for the EWM-execute line — BEFORE any executor build.

The plan-execute-verify executor (OPINE §3.5 contract; opine_world_deepread.md
Stage 1) does not exist yet. This script demonstrates, on REAL recorded traces,
what its EWMEVT event stream (duck_eval/ewm_exec/EVENT_SCHEMA.md) would have
looked like, and runs the sealed aggregator on it:

  * Sources: runs/kernel_pulls/war_eval_v{1,2,3} (recorded Qwen duck-harness
    Kaggle runs, 25 games each) + runs/gpt56_probe/experiment_full (GPT-5.6
    probe streams, 5 games) — per-action settled `board` frames throughout.
  * Sims: the 12 saturated exec_wm sims (>=99.5% state_exact on held-out local
    tuples; exec_wm/scale_summary.md).
  * Replay: the recorded action sequence stands in for the executor's plan
    (plan boundaries = the recorded agent's own action(...) batches). Each
    recorded action is fed to the sim from the recorded pre-action settled
    frame (teacher forcing — mismatches do not cascade); predicted vs recorded
    settled frame is hash-compared; the first mismatch in a plan emits
    mismatch_abort + fallback exactly as the contract would.
  * Double-run refinement (ii): the whole sequence is replayed through TWO
    independently loaded sim module instances in identical call order; any
    prediction disagreement = selfdiff (hidden nondeterminism -> reject).

What this measures honestly: per-game step-level fidelity of the sims against
the REAL Kaggle engine versions (15/25 local engines drift — the mismatch-abort
landing spots below are precisely the fail-closed events that make shipping the
sims safe), and that the event schema + aggregator express the Stage-0 gate
(canary, survival, deadlock) on real data. What it does NOT measure: BFS plan
quality — plans here are the recorded agent's actions, not sim-derived paths.

Usage:  uv run python scripts/ewm_replay_dryrun.py
Output: runs/ewm_dryrun/<source>.log (EWMEVT streams), report.md, raw.json.
CPU-only, read-only w.r.t. sims/traces, no network, no pushes.
"""
from __future__ import annotations

import glob
import hashlib
import importlib.util
import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ewm_events import aggregate, parse_log, render_table, verdict_lines  # noqa: E402

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

# The 12 saturated sims (exec_wm/scale_summary.md, held-out state_exact %).
SAT12 = {
    "ft09": 100.0, "lf52": 100.0, "lp85": 100.0, "ls20": 100.0,
    "sb26": 100.0, "sp80": 100.0, "tn36": 100.0, "tr87": 100.0,
    "tu93": 100.0, "s5i5": 99.5, "su15": 99.5, "vc33": 99.5,
}

SOURCES = {
    "war_eval_v1": ROOT / "runs/kernel_pulls/war_eval_v1/artifacts",
    "war_eval_v2": ROOT / "runs/kernel_pulls/war_eval_v2/artifacts",
    "war_eval_v3": ROOT / "runs/kernel_pulls/war_eval_v3/artifacts",
    "gpt56_full": ROOT / "runs/gpt56_probe/experiment_full/artifacts",
}

MOUSE_RE = re.compile(r"MOUSE\(row=(\d+), col=(\d+)\)")


def hash8(board) -> str:
    return hashlib.blake2b(
        json.dumps(board, separators=(",", ":")).encode(),
        digest_size=4).hexdigest()


def load_sim(game_id: str, tag: str):
    """Fresh, independently-namespaced instance of exec_wm/sims/<gid>_sim.py."""
    p = ROOT / "exec_wm" / "sims" / f"{game_id}_sim.py"
    spec = importlib.util.spec_from_file_location(f"{game_id}_sim_{tag}", str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.simulate, hashlib.blake2b(p.read_bytes(), digest_size=4).hexdigest()


def parse_action(ev):
    """action event -> (aid, x, y) or None for RESET/unknown."""
    name = ev.get("action_name") or ""
    if name == "RESET":
        return None
    m = re.fullmatch(r"ACTION(\d)", name)
    if not m:
        return None
    aid = int(m.group(1))
    x = y = 0
    if aid == 6:
        mm = MOUSE_RE.search(ev.get("action_display") or "")
        if mm:
            y, x = int(mm.group(1)), int(mm.group(2))  # x=col, y=row (engine data{x,y})
    return aid, x, y


def act_str(aid, x, y):
    return f"A{aid}:{y},{x}" if aid == 6 else f"A{aid}"


def group_plans(events):
    """Recorded action(...) batches = plan proxies.

    A new plan starts when analysis_step changes or batch_index does not
    continue the previous one. RESET actions are control events (plan
    boundaries), never plan steps.
    """
    plans, cur, last = [], [], None  # last = (analysis_step, batch_index)
    for ev in events:
        if ev.get("type") != "action":
            continue
        if (ev.get("action_name") or "") == "RESET":
            if cur:
                plans.append(cur)
            cur, last = [], None
            continue
        key = (ev.get("analysis_step"), ev.get("batch_index"))
        if cur and not (key[0] == last[0] and key[1] == (last[1] or 0) + 1):
            plans.append(cur)
            cur = []
        cur.append(ev)
        last = key
    if cur:
        plans.append(cur)
    return plans


def replay_game(gid: str, gv: str, fp: str, out_lines: list):
    """Replay one recorded trace through the saturated sim; emit EWMEVT lines.

    Returns shadow stats (ALL steps verified, not just pre-abort ones)."""
    with open(fp, encoding="utf-8") as f:
        events = [json.loads(ln) for ln in f if ln.strip()]

    sim1, simhash = load_sim(gid, "a")
    sim2, _ = load_sim(gid, "b")

    # board the agent acted on, teacher-forced from the recorded stream
    prev_board = None
    boards = {}  # id(ev) -> pre-action board
    for ev in events:
        if ev.get("type") == "initial":
            prev_board = ev["board"]
        elif ev.get("type") == "action":
            boards[id(ev)] = prev_board
            prev_board = ev["board"]

    shadow = {"steps": 0, "exact": 0, "selfdiff": 0, "sim_error": 0,
              "first_div": None, "done_agree": 0, "done_total": 0,
              "diff_cells": []}  # changed-cell count per mismatching step

    plans = group_plans(events)
    for pi, plan in enumerate(plans):
        emitted_start = False
        aborted = False
        lvl_done = 0
        for si, ev in enumerate(plan):
            parsed = parse_action(ev)
            if parsed is None:
                continue
            aid, x, y = parsed
            pre = boards[id(ev)]
            if pre is None:
                continue
            if not emitted_start:
                out_lines.append(
                    f"EWMEVT v=1 kind=plan_start game={gid} plan={pi} "
                    f"len={len(plan)} sim={gid}_sim:{simhash} gv={gv} "
                    f"lvl={ev.get('level', 0)} t=na")
                emitted_start = True

            obs = ev["board"]
            obs_h = hash8(obs)
            reason = None
            pred_h = "--------"
            match = 0
            try:
                p1 = sim1([list(r) for r in pre], aid, x, y)
                p2 = sim2([list(r) for r in pre], aid, x, y)
                a1 = np.asarray(p1[0], dtype=np.uint8)
                a2 = np.asarray(p2[0], dtype=np.uint8)
                if a1.shape != (64, 64):
                    raise ValueError(f"shape {a1.shape}")
                pred_list = a1.tolist()
                pred_h = hash8(pred_list)
                if not np.array_equal(a1, a2):
                    reason = "selfdiff"
                    shadow["selfdiff"] += 1
                else:
                    match = int(pred_list == obs)
                    if not match:
                        obs_arr = np.asarray(obs, dtype=np.uint8)
                        if obs_arr.shape == a1.shape:
                            shadow["diff_cells"].append(
                                int((a1 != obs_arr).sum()))
                    shadow["done_total"] += 1
                    if bool(p1[2]) == bool(ev.get("level_completed", False)):
                        shadow["done_agree"] += 1
            except Exception:
                reason = "sim_error"
                shadow["sim_error"] += 1

            shadow["steps"] += 1
            if match:
                shadow["exact"] += 1
            elif shadow["first_div"] is None:
                shadow["first_div"] = shadow["steps"] - 1

            if not aborted:
                out_lines.append(
                    f"EWMEVT v=1 kind=plan_step game={gid} plan={pi} step={si} "
                    f"act={act_str(aid, x, y)} pred={pred_h} obs={obs_h} "
                    f"match={match} lvl={ev.get('level', 0)} t=na")
                if ev.get("level_completed"):
                    lvl_done = 1
                if reason or not match:
                    out_lines.append(
                        f"EWMEVT v=1 kind=mismatch_abort game={gid} plan={pi} "
                        f"step={si} len={len(plan)} reason={reason or 'mismatch'} "
                        f"pred={pred_h} obs={obs_h} t=na")
                    out_lines.append(
                        f"EWMEVT v=1 kind=fallback game={gid} plan={pi} "
                        f"reason={reason or 'mismatch'} t=na")
                    aborted = True

        if emitted_start and not aborted:
            n_steps = sum(1 for e in plan if parse_action(e) is not None)
            out_lines.append(
                f"EWMEVT v=1 kind=plan_done game={gid} plan={pi} "
                f"len={len(plan)} steps={n_steps} lvl_done={lvl_done} t=na")

    return shadow


def main():
    out_dir = ROOT / "runs" / "ewm_dryrun"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = {"sources": {}, "sat12": SAT12}

    for src, art in SOURCES.items():
        files = sorted(glob.glob(str(art / "*_events.jsonl")))
        lines: list[str] = []
        shadows = {}
        for fp in files:
            name = Path(fp).name  # e.g. ls20-9607627b_p0_events.jsonl
            gv = name.split("_")[0]
            gid = gv.split("-")[0]
            if gid not in SAT12:
                continue
            shadows[gid] = replay_game(gid, gv, fp, lines)

        log_fp = out_dir / f"{src}.log"
        log_fp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        events, malformed = parse_log(str(log_fp))
        agg = aggregate(events)
        maxlen = max((len(ln) for ln in lines), default=0)
        raw["sources"][src] = {
            "log_bytes": log_fp.stat().st_size,
            "n_lines": len(lines),
            "max_line_len": maxlen,
            "malformed": malformed,
            "shadow": shadows,
            "aggregate": agg,
        }
        print(f"[{src}] {len(lines)} EWMEVT lines, {log_fp.stat().st_size} B, "
              f"max line {maxlen} ch -> {log_fp.name}")
        for vl in verdict_lines(agg):
            print("  " + vl)

    (out_dir / "raw.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
    _write_report(out_dir, raw)
    print(f"wrote {out_dir/'report.md'} and {out_dir/'raw.json'}")


def _write_report(out_dir, raw):
    L = ["# EWM-execute Stage-0 gate dry-run (schema validation on real traces)\n",
         "Producer: `scripts/ewm_replay_dryrun.py`; schema: "
         "`duck_eval/ewm_exec/EVENT_SCHEMA.md`; aggregator: "
         "`scripts/ewm_events.py`. Recorded Kaggle action streams replayed "
         "through the 12 saturated exec_wm sims; plan proxies = the recorded "
         "agent's own action batches. Mismatch rates below are REAL "
         "sim-vs-Kaggle-engine fidelity (incl. engine-version drift), measured "
         "before any executor build.\n"]
    for src, s in raw["sources"].items():
        agg = s["aggregate"]
        L.append(f"## {src}\n")
        L.append(f"Log: {s['n_lines']} EWMEVT lines, {s['log_bytes']:,} bytes, "
                 f"longest line {s['max_line_len']} chars, "
                 f"{s['malformed']} malformed.\n")
        L.extend(render_table(agg))
        L.append("")
        L.append("Shadow stats (ALL recorded steps verified, not just "
                 "pre-abort):\n")
        L.append("| game | held-out state_exact% | steps | exact | shadow acc | "
                 "first divergence (step#) | med/max diff cells on mismatch | "
                 "selfdiff | sim_error | done-flag agree |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for gid, sh in sorted(s["shadow"].items()):
            acc = sh["exact"] / sh["steps"] if sh["steps"] else 0
            da = (f"{sh['done_agree']}/{sh['done_total']}"
                  if sh["done_total"] else "-")
            fd = sh["first_div"] if sh["first_div"] is not None else "-"
            dc = sh.get("diff_cells") or []
            dcs = (f"{sorted(dc)[len(dc)//2]}/{max(dc)}" if dc else "-")
            L.append(f"| {gid} | {raw['sat12'][gid]} | {sh['steps']} | "
                     f"{sh['exact']} | {acc:.3f} | {fd} | {dcs} | "
                     f"{sh['selfdiff']} | {sh['sim_error']} | {da} |")
        L.append("")
        for vl in verdict_lines(agg):
            L.append(f"`{vl}`")
        L.append("")
    (out_dir / "report.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
