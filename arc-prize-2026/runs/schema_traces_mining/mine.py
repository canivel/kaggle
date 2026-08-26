"""Mine Schema harness trajectories (schema-harness/arc-agi-3-schema-traces) vs our problem set."""
import json, os, re, sys, glob, collections

ROOT = r"F:\kaggle\arc-prize-2026\kaggle-data\schema_traces"
OUT = r"F:\kaggle\arc-prize-2026\runs\schema_traces_mining\stats.json"

MISMATCH_RE = re.compile(r"(\d+) mismatch\(es\)")

def mine_traj(d):
    ev = os.path.join(d, "events.jsonl")
    if not os.path.exists(ev):
        return None
    s = {
        "dir": os.path.basename(d), "game": None, "model": None, "max_actions": None,
        "turns": 0, "actions": 0, "resets": 0, "commits": 0,
        "plan_len_hist": collections.Counter(), "actions_from_plan1": 0,
        "actions_from_plan_2_9": 0, "actions_from_plan_10p": 0, "max_plan": 0,
        "mispredicts": 0, "mispredict_step0": 0, "mispredict_later": 0,
        "backtests": 0, "backtest_fail": 0, "backtest_pass": 0,
        "wm_writes": 0, "notes_writes": 0, "run_python": 0, "run_bfs": 0,
        "tools": collections.Counter(),
        "first_action_turn": None, "wall_hours": None,
        "per_level_actions": collections.Counter(), "per_level_mispredicts": collections.Counter(),
        "final": {},
    }
    t0 = t1 = None
    cur_plan_len = 0
    with open(ev, encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = e.get("kind")
            ts = e.get("ts")
            if ts:
                t0 = ts if t0 is None else t0
                t1 = ts
            if k == "run_started":
                s["game"] = e.get("game_id"); s["model"] = e.get("model")
                s["max_actions"] = e.get("max_actions")
            elif k == "turn_started":
                s["turns"] = max(s["turns"], e.get("turn", 0))
            elif k == "turn_committed":
                s["commits"] += 1
                pl = len(e.get("plan") or [])
                cur_plan_len = pl
                s["plan_len_hist"][pl] += 1
                s["max_plan"] = max(s["max_plan"], pl)
            elif k == "action_taken":
                s["actions"] += 1
                if s["first_action_turn"] is None:
                    s["first_action_turn"] = e.get("turn")
                if e.get("action") == 0:
                    s["resets"] += 1
                lv = e.get("level")
                s["per_level_actions"][lv] += 1
                if cur_plan_len == 1:
                    s["actions_from_plan1"] += 1
                elif cur_plan_len >= 10:
                    s["actions_from_plan_10p"] += 1
                else:
                    s["actions_from_plan_2_9"] += 1
            elif k == "model_mispredicted":
                s["mispredicts"] += 1
                si = e.get("step_index")
                if si == 0:
                    s["mispredict_step0"] += 1
                else:
                    s["mispredict_later"] += 1
                # level of the mispredicted step: approximate with last seen level
            elif k == "tool_started":
                nm = e.get("name")
                s["tools"][nm] += 1
                if nm in ("write_file", "edit_file"):
                    p = (e.get("args") or {}).get("path", "")
                    if "world_model" in p:
                        s["wm_writes"] += 1
                    elif "notes" in p:
                        s["notes_writes"] += 1
                elif nm == "run_python":
                    s["run_python"] += 1
                elif nm == "run_bfs":
                    s["run_bfs"] += 1
            elif k == "tool_finished" and e.get("name") == "run_backtest":
                s["backtests"] += 1
                out = e.get("output") or ""
                m = MISMATCH_RE.search(out)
                if m and int(m.group(1)) > 0:
                    s["backtest_fail"] += 1
                else:
                    s["backtest_pass"] += 1
            elif k == "run_finished":
                s["final"] = {kk: e.get(kk) for kk in ("state", "levels", "win_levels", "actions", "transitions", "has_world_model")}
    if t0 and t1:
        s["wall_hours"] = round((t1 - t0) / 3600, 2)
    s["plan_len_hist"] = dict(sorted(s["plan_len_hist"].items()))
    s["tools"] = dict(s["tools"].most_common())
    s["per_level_actions"] = dict(sorted(s["per_level_actions"].items(), key=lambda x: (x[0] is None, x[0])))
    s["per_level_mispredicts"] = dict(s["per_level_mispredicts"])
    return s

def main():
    results = {}
    for coll in ("claude_fable_opus", "gpt_5_6_sol"):
        for d in sorted(glob.glob(os.path.join(ROOT, coll, "*"))):
            if not os.path.isdir(d) or os.path.basename(d).startswith("."):
                continue
            st = mine_traj(d)
            if st:
                results.setdefault(coll, []).append(st)
                print(f"{coll}/{os.path.basename(d)}: turns={st['turns']} actions={st['actions']} "
                      f"mispredicts={st['mispredicts']} backtests={st['backtests']}({st['backtest_fail']} fail) "
                      f"wm_writes={st['wm_writes']} bfs={st['run_bfs']}", flush=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print("wrote", OUT)

if __name__ == "__main__":
    main()
