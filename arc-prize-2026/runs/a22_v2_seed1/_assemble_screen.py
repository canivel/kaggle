# Throwaway: assemble m1m2m3_screen.json for A22 compaction v2 seed 1 (v1 schema + v2 fields).
import json, os, re

ROOT = r"F:\kaggle\arc-prize-2026"
V2 = os.path.join(ROOT, "runs", "a22_v2_seed1")

part = json.load(open(os.path.join(V2, "_m1m2_partial.json")))
m3raw = json.load(open(os.path.join(V2, "_m3_raw.json")))
m3sum = json.load(open(os.path.join(V2, "_m3_summary.json")))

# ---- first100 / last100 in run-chronological (log) order ----
log_events = []
with open(os.path.join(V2, "arc3-duck-compaction-eval.log"), encoding="utf-8", errors="replace") as fh:
    for line in fh:
        line = line.strip().lstrip("[,")
        if not line or '"data":"COMPACTION ' not in line:
            continue
        try:
            rec = json.loads(line.rstrip("]"))
        except json.JSONDecodeError:
            continue
        data = rec.get("data", "")
        if data.startswith("COMPACTION ") and "kind=evict_compact" in data:
            ev = {}
            for m in re.finditer(r"(\w+)=(-?\d+)(?:\s|$)", data):
                ev[m.group(1)] = int(m.group(2))
            if "facts" in ev and "refuted" in ev:
                log_events.append(ev)
first100 = log_events[:100]
last100 = log_events[-100:]
def mean(v): return sum(v) / len(v)

# ---- M3 block (v1 field names; exact sign-flip p instead of v1's MC 400k) ----
games = sorted(m3raw["v2"])
M3 = dict(
    mean_d_reprop_pp=m3sum["reprop"]["mean_delta_pp"],
    p_reprop=m3sum["reprop"]["p"],
    mean_d_refrep035_pp=m3sum["refrep035"]["mean_delta_pp"],
    p_refrep035=m3sum["refrep035"]["p"],
    mean_d_refrep045_pp=m3sum["refrep045"]["mean_delta_pp"],
    p_refrep045=m3sum["refrep045"]["p"],
    a22_worse_games=m3sum["reprop"]["a22_worse"],
    war_worse_games=m3sum["reprop"]["war_worse"],
    mean_d_refrep060_pp=m3sum["refrep060"]["mean_delta_pp"],
    p_refrep060=m3sum["refrep060"]["p"],
    p_method="exact sign-flip (v1 used MC 400k)",
    procedure_note=("v1 forensics script not preserved; procedure re-implemented per "
                    "learnings/sweeps/a22_seed1_screen_2026-08-03.md SS3 and applied "
                    "identically to both arms; calibration on v1-arm transcripts "
                    "reproduces recorded v1 direction (+1.56pp reprop vs recorded "
                    "+2.24pp; +1.35pp refrep035 vs recorded +1.25pp), mean abs "
                    "per-game deviation 0.030-0.037"),
)

# ---- per_game (v1 schema fields) ----
per_game = []
for r in part["per_game"]:
    g = r["game"]
    per_game.append(dict(
        game=g,
        a22_lc=r["a22_lc"], war_lc=r["war_lc"], dlc=r["dlc"],
        a22_actions=r["a22_actions"], war_actions=r["war_actions"],
        a22_gen_tokens=r["a22_gen_tokens"], war_gen_tokens=r["war_gen_tokens"],
        a22_turns=r["a22_turns"], war_turns=r["war_turns"],
        a22_reprop_rate=m3raw["v2"][g]["reprop"],
        war_reprop_rate=m3raw["war"][g]["reprop"],
        a22_refrep035=m3raw["v2"][g]["refrep"]["0.35"],
        war_refrep035=m3raw["war"][g]["refrep"]["0.35"],
    ))

ce = part["compaction_events"]
ce["first100"] = dict(facts=mean([e["facts"] for e in first100]),
                      gated_facts=mean([e["gated_facts"] for e in first100]),
                      refuted=mean([e["refuted"] for e in first100]),
                      digest_tokens=mean([e["digest_tokens"] for e in first100]))
ce["last100"] = dict(facts=mean([e["facts"] for e in last100]),
                     gated_facts=mean([e["gated_facts"] for e in last100]),
                     refuted=mean([e["refuted"] for e in last100]),
                     digest_tokens=mean([e["digest_tokens"] for e in last100]))
ce["log_event_lines"] = len(log_events)

M1 = part["M1"]
sc25 = next(r for r in per_game if r["game"] == "sc25")

out = dict(
    generated_utc="2026-08-06",
    arm="A22 compaction v2 (COMPACTION=1, region-aware eviction, RETAIN=OFF), seed 1",
    a22_run="runs/a22_v2_seed1",
    baseline="runs/kernel_pulls/war_eval_v1 (arc3-duck-war-eval seed 1, ledger-OFF)",
    prereg="learnings/war_room/a22_compaction_v2_prereg_2026-08-04.md",
    canary=part["canary"],
    M1=M1,
    M2=part["M2"],
    M3=M3,
    compaction_events=ce,
    per_game=per_game,
    attr_per_game=part["attr_per_game"],
    sc25_recovered=(sc25["dlc"] >= 0),
    sc25_dlc=sc25["dlc"],
    kill_rules=dict(
        K1=dict(
            verdict="PASS (not void)",
            evidence=("2617 sidecar COMPACTION events (2606 visible in kernel log) across all 25 games; "
                      "banner \"compaction v2: ACTIVE\"; graft applied=True; "
                      "RETAIN-OFF canary: retained_reasoning_msgs=0 in 2617/2617 events, "
                      "retain=0 in 2617/2617, banner shows mirroring=OFF"),
        ),
        K2=dict(verdict="PASS (no PATCH FAILED / vanilla fallback)"),
        K3=dict(
            verdict="FAIL at seed 1 -> v2 MECHANISM PAUSED; A22 lane one FAIL from DEAD",
            detail=("worst-game -2.0 (sc25) breaches -1.0 cap; mean -0.320 breaches -0.128 "
                    "admission precedent; both legs independently. 2W/9L, exact sign-flip "
                    "p=0.0557. Per prereg K-counting note: v2 seed-1 FAIL pauses v2; "
                    "combined with the v1 K3 FAIL on record the whole A22 lane is one "
                    "independent FAIL from DEAD."),
        ),
        K4=dict(
            verdict="NOT ON TRACK TO KILL (seed-1 only; K4 requires BOTH seeds)",
            detail=("seed 1 shows REDUCTION in re-proposal for the first time: "
                    "-4.57pp reprop (p=0.012, v2 worse in only 7/25 games), "
                    "-0.26pp refrep@0.35 (NS), -0.23pp refrep@0.45 (NS), "
                    "-0.05pp refrep@0.60 (NS) -- all point estimates in the "
                    "prereg-required direction, reversing v1's +2.24pp/+1.25pp"),
        ),
    ),
)
path = os.path.join(V2, "m1m2m3_screen.json")
json.dump(out, open(path, "w"), indent=1)
print("wrote", path)
print("first100:", ce["first100"])
print("last100:", ce["last100"])
print("log-parsed events:", len(log_events))
