# Throwaway: A22 compaction v2 seed-1 screen -- canary + M1 + M2 + event stats.
# Writes _m1m2_partial.json for assembly into m1m2m3_screen.json.
import json, glob, os, re, statistics, sys

ROOT = r"F:\kaggle\arc-prize-2026"
V2 = os.path.join(ROOT, "runs", "a22_v2_seed1")
WAR = os.path.join(ROOT, "runs", "kernel_pulls", "war_eval_v1")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from phase1_gate import signflip_p_exact

def load_bench(d):
    b = json.load(open(os.path.join(d, "benchmark.json")))
    out = {}
    for r in b["game_runs"]:
        g = r["game_id"].split("-")[0]
        m = re.search(r"tokens=(\d+)", r.get("solver_note") or "")
        out[g] = dict(
            game_id=r["game_id"],
            lc=r["levels_completed"],
            actions=len(r["history"]),
            gen_tokens=int(m.group(1)) if m else None,
        )
    return out

def turns_from_transcripts(d):
    out = {}
    for f in glob.glob(os.path.join(d, "transcripts", "*_p0.txt")):
        g = os.path.basename(f).split("-")[0]
        n = 0
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("--- analysis_step="):
                    n += 1
        out[g] = n
    return out

v2b, warb = load_bench(V2), load_bench(WAR)
v2t, wart = turns_from_transcripts(V2), turns_from_transcripts(WAR)
games = sorted(v2b)
assert games == sorted(warb) == sorted(v2t) == sorted(wart), "game list mismatch"
assert len(games) == 25

# ---------------- canary ----------------
log = open(os.path.join(V2, "arc3-duck-compaction-eval.log"), encoding="utf-8", errors="replace").read()
banner_active = "compaction v2: ACTIVE" in log
banner_mirror_off = "retained-reasoning mirroring=OFF" in log
stamp = "COMPACTION=1" in log
patch_failed = "PATCH FAILED" in log
log_event_lines = log.count('"data":"COMPACTION ')

sidecars = sorted(glob.glob(os.path.join(V2, "artifacts", "*_compaction_events.jsonl")))
events = []          # all events, tagged with game
per_game_ev = {}     # game -> list of events
for f in sidecars:
    g = os.path.basename(f).split("-")[0]
    evs = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    per_game_ev[g] = evs
    events.extend(evs)
n_events = len(events)
rrm_nonzero = sum(1 for e in events if e["retained_reasoning_msgs"] != 0)
retain_nonzero = sum(1 for e in events if e.get("retain", 0) != 0)
games_with_events = sorted(per_game_ev)

canary = dict(
    banner_active=banner_active,
    banner_mirroring_off=banner_mirror_off,
    compaction1_stamp=stamp,
    patch_failed=patch_failed,
    log_event_lines=log_event_lines,
    sidecar_events=n_events,
    sidecar_games=len(games_with_events),
    retained_reasoning_msgs_nonzero_events=rrm_nonzero,
    retain_flag_nonzero_events=retain_nonzero,
    retain_off_canary_pass=(rrm_nonzero == 0 and retain_nonzero == 0 and banner_mirror_off),
    void=(not banner_active) or patch_failed or (n_events == 0) or (rrm_nonzero > 0),
)

# ---------------- M1 ----------------
per_game = []
for g in games:
    per_game.append(dict(
        game=g,
        a22_lc=v2b[g]["lc"], war_lc=warb[g]["lc"], dlc=v2b[g]["lc"] - warb[g]["lc"],
        a22_actions=v2b[g]["actions"], war_actions=warb[g]["actions"],
        a22_gen_tokens=v2b[g]["gen_tokens"], war_gen_tokens=warb[g]["gen_tokens"],
        a22_turns=v2t[g], war_turns=wart[g],
    ))
deltas = [r["dlc"] for r in per_game]
mean_dlc = sum(deltas) / len(deltas)
sd = statistics.stdev(deltas)
wins = sum(1 for d in deltas if d > 0)
losses = sum(1 for d in deltas if d < 0)
nz = [d for d in deltas if d != 0]
obs = sum(nz)
p_one = signflip_p_exact(nz, abs(obs))[0] if nz else 1.0
p_two = min(1.0, 2 * p_one)
worst = min(per_game, key=lambda r: r["dlc"])
a22_lc_tot = sum(r["a22_lc"] for r in per_game)
war_lc_tot = sum(r["war_lc"] for r in per_game)

M1 = dict(
    mean_dlc=mean_dlc, sd_games=sd, wins=wins, losses=losses, nonzero_k=len(nz),
    signflip_p_exact=p_two,
    worst_game=worst["game"], worst_dlc=worst["dlc"],
    a22_lc_total=a22_lc_tot, war_lc_total=war_lc_tot,
    nonharm_worst_cap=-1.0, nonharm_worst_pass=(worst["dlc"] >= -1.0),
    nonharm_mean_precedent=-0.128, nonharm_mean_pass=(mean_dlc >= -0.128),
)
M1["verdict"] = "PASS" if (M1["nonharm_worst_pass"] and M1["nonharm_mean_pass"]) else "FAIL"

# sanity: replicate v1 p on v1 deltas
v1p = min(1.0, 2 * signflip_p_exact([1, 1, -1, -1, -1, -1, -2, -1], 5)[0])
print("v1 p replication (expect 0.234375):", v1p)

# ---------------- M2 ----------------
a22_tok = sum(r["a22_gen_tokens"] for r in per_game)
war_tok = sum(r["war_gen_tokens"] for r in per_game)
a22_act = sum(r["a22_actions"] for r in per_game)
war_act = sum(r["war_actions"] for r in per_game)
a22_turns = sum(r["a22_turns"] for r in per_game)
war_turns = sum(r["war_turns"] for r in per_game)
M2 = dict(
    a22_gen_tokens=a22_tok, war_gen_tokens=war_tok,
    a22_actions=a22_act, war_actions=war_act,
    a22_lc=a22_lc_tot, war_lc=war_lc_tot,
    a22_turns=a22_turns, war_turns=war_turns,
    a22_tok_per_action=a22_tok / a22_act, war_tok_per_action=war_tok / war_act,
    ratio_tok_per_action=(a22_tok / a22_act) / (war_tok / war_act),
    a22_tok_per_lc=(a22_tok / a22_lc_tot) if a22_lc_tot else None,
    war_tok_per_lc=war_tok / war_lc_tot,
    a22_tok_per_turn=a22_tok / a22_turns, war_tok_per_turn=war_tok / war_turns,
)
M2["ratio_tok_per_lc"] = (M2["a22_tok_per_lc"] / M2["war_tok_per_lc"]) if M2["a22_tok_per_lc"] else None

# --- v2 amendment: budget-relief attribution split ---
last = {g: evs[-1] for g, evs in per_game_ev.items()}
tot_evicted_msgs = sum(l["total_evicted_msgs"] for l in last.values())
tot_evicted_chars = sum(e["evicted_chars"] for e in events)
tot_digest_tokens = sum(e["digest_tokens"] for e in events)
n_digest_zero = sum(1 for e in events if e["digest_tokens"] == 0)
n_digest_nonzero = n_events - n_digest_zero
n_reserve = sum(1 for e in events if e["reserve_applied"])
cls = {k: sum(l[k] for l in last.values()) for k in ("ev_episode", "ev_user", "ev_reasoning", "ev_fallback")}
stuck_tot = sum(l["stuck_suppressed"] for l in last.values())
gated_tot_last = sum(l["gated_facts"] for l in last.values())
M2["v2_attribution"] = dict(
    n_events=n_events,
    total_evicted_msgs=tot_evicted_msgs,
    total_evicted_chars=tot_evicted_chars,
    total_digest_tokens_injected=tot_digest_tokens,
    events_digest_empty=n_digest_zero,
    events_digest_nonempty=n_digest_nonzero,
    pct_events_digest_empty=100.0 * n_digest_zero / n_events,
    reserve_applied_events=n_reserve,
    reserve_applied_share=n_reserve / n_events,
    eviction_class_totals=cls,
    eviction_class_shares={k: v / max(1, sum(cls.values())) for k, v in cls.items()},
    stuck_suppressed_total=stuck_tot,
    stuck_suppressed_games_nonzero=sum(1 for l in last.values() if l["stuck_suppressed"] > 0),
    gated_facts_final_total=gated_tot_last,
)

# per-game attribution + relate to dlc
attr_pg = {}
for g, evs in per_game_ev.items():
    l = evs[-1]
    attr_pg[g] = dict(
        n_events=len(evs),
        evicted_chars=sum(e["evicted_chars"] for e in evs),
        digest_tokens_sum=sum(e["digest_tokens"] for e in evs),
        digest_nonempty_events=sum(1 for e in evs if e["digest_tokens"] > 0),
        reserve_applied_events=sum(1 for e in evs if e["reserve_applied"]),
        ev_episode=l["ev_episode"], ev_user=l["ev_user"],
        ev_reasoning=l["ev_reasoning"], ev_fallback=l["ev_fallback"],
        stuck_suppressed=l["stuck_suppressed"],
    )
# pearson evicted_chars vs dlc
def pearson(x, y):
    n = len(x); mx = sum(x)/n; my = sum(y)/n
    num = sum((a-mx)*(b-my) for a, b in zip(x, y))
    dx = sum((a-mx)**2 for a in x) ** 0.5
    dy = sum((b-my)**2 for b in y) ** 0.5
    return num / (dx*dy) if dx and dy else float("nan")
dlc_by_g = {r["game"]: r["dlc"] for r in per_game}
xs = [attr_pg[g]["evicted_chars"] for g in games]
ys = [dlc_by_g[g] for g in games]
M2["v2_attribution"]["pearson_evictedchars_vs_dlc"] = pearson(xs, ys)
xs2 = [attr_pg[g]["digest_tokens_sum"] for g in games]
M2["v2_attribution"]["pearson_digesttokens_vs_dlc"] = pearson(xs2, ys)

# ---------------- compaction_events summary (v1 schema + v2 fields) ----------------
def dist(vals, pct=True):
    s = sorted(vals)
    n = len(s)
    d = dict(min=s[0], p25=s[n//4], median=s[n//2], p75=s[(3*n)//4], max=s[-1],
             mean=sum(s)/n)
    if pct:
        d["pct_zero"] = 100.0 * sum(1 for v in s if v == 0) / n
    return d

ev_sorted = events  # sidecar order per game; for first/last100 use concatenation order
ce = dict(
    n_events=n_events,
    facts=dist([e["facts"] for e in events]),
    gated_facts=dist([e["gated_facts"] for e in events]),
    refuted=dist([e["refuted"] for e in events]),
    digest_tokens=dist([e["digest_tokens"] for e in events]),
    evicted_msgs=dist([e["evicted_msgs"] for e in events]),
    evicted_chars=dist([e["evicted_chars"] for e in events]),
    retained_reasoning_msgs=dist([e["retained_reasoning_msgs"] for e in events]),
    facts_and_refuted_both_zero=sum(1 for e in events if e["facts"] == 0 and e["refuted"] == 0),
    gated_facts_and_refuted_both_zero=sum(1 for e in events if e["gated_facts"] == 0 and e["refuted"] == 0),
)

out = dict(canary=canary, M1=M1, M2=M2, compaction_events=ce,
           per_game=per_game, attr_per_game=attr_pg)
json.dump(out, open(os.path.join(V2, "_m1m2_partial.json"), "w"), indent=1)
print(json.dumps(dict(canary=canary, M1=M1), indent=1))
print("M2 ratios: per_action %.4f per_lc %s per_turn %.3f" % (
    M2["ratio_tok_per_action"], M2["ratio_tok_per_lc"],
    M2["a22_tok_per_turn"] / M2["war_tok_per_turn"]))
print("attribution:", json.dumps(M2["v2_attribution"], indent=1))
