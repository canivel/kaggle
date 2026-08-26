"""Smoke tests for the five §5.3 P1 instruments. Offline, $0, zero pushes.

  uv run python duck_eval/p1_instr/smoke_test.py
"""
from __future__ import annotations

import io
import json
import statistics as st
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import drop_policy as dp                    # noqa: E402
import latency_prefix as lp                 # noqa: E402
import namespace_reuse as nr                # noqa: E402
import patch_surface as ps                  # noqa: E402
import safe_modules_patch as smp            # noqa: E402
import sandbox_facts as sf                  # noqa: E402
from pull_io import load_calls, load_pull, parse_solver_banner  # noqa: E402

PASS = 0
FAIL = 0
BASELINE = ["war_eval_v1", "war_eval_v2", "war_eval_v3"]


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}" + (f"  [{detail}]" if detail else ""))
    else:
        FAIL += 1
        print(f"  FAIL  {name}" + (f"  [{detail}]" if detail else ""))


# ==========================================================================
def item1_latency_prefix() -> None:
    print("\n[ITEM 1] latency + matched-action-prefix endpoint")
    v1 = load_pull("war_eval_v1")
    check("banner parsed from the run's own log",
          v1.banner.get("concurrency") == 28
          and v1.banner.get("max_actions_per_game") is None
          and v1.banner.get("max_runtime_s_per_game") == 7920.0,
          json.dumps(v1.banner))

    rows = lp.latency_table(v1)
    capped = sum(1 for r in rows if r.hit_runtime_cap)
    tot_a = sum(r.n_actions for r in rows)
    tot_w = sum(r.total_wallclock_s for r in rows)
    check("all 25 games hit the wall-clock guillotine", capped == 25, f"{capped}/25")
    check("total actions reproduces summary.txt (3638)", tot_a == 3638, str(tot_a))
    check("s per scored action ~54.4 (systems review)",
          abs(tot_w / tot_a - 54.44) < 0.05, f"{tot_w / tot_a:.2f}")

    # exactness of the prefix reconstruction
    ok = all(g.lc_at_action_prefix(g.n_actions) == g.levels_completed
             for g in v1.games.values())
    check("lc_at_action_prefix(full) == levels_completed for all games", ok)
    ok0 = all(g.lc_at_action_prefix(0) == 0 for g in v1.games.values())
    check("lc_at_action_prefix(0) == 0 for all games", ok0)
    mono = all(
        all(g.lc_at_action_prefix(k) <= g.lc_at_action_prefix(k + 7)
            for k in range(0, g.n_actions, 7))
        for g in v1.games.values())
    check("lc_at_action_prefix is monotone non-decreasing", mono)

    # reproduce the minutes' §3.3a arithmetic through the instrument
    v3 = load_pull("war_eval_v3")
    r31 = lp.matched_prefix(v3, [v1])
    mean31 = st.mean(r.dlc_full for r in r31)
    check("v3-v1 mean dlc == -0.3600 (minutes §3.3a)", abs(mean31 + 0.36) < 1e-9,
          f"{mean31:.4f}")
    v2 = load_pull("war_eval_v2")
    mean21 = st.mean(r.dlc_full for r in lp.matched_prefix(v2, [v1]))
    check("v2-v1 mean dlc == -0.2800 (minutes §3.3a)", abs(mean21 + 0.28) < 1e-9,
          f"{mean21:.4f}")

    bases = [v1, v2, load_pull("war_eval_v3")]
    gate = lp.k3prime_line(bases)
    check("K3' s_base for the warpack family == 0.189 (recalibration §4)",
          abs(float(gate["s_base"]) - 0.189) < 5e-4, str(gate["s_base"]))
    check("K3' flags the R25-N1 looser-than-fallback defect",
          "r25_n1_flag" in gate, str(gate.get("line")))

    a22 = load_pull("a22_v2_1")
    rows22 = lp.matched_prefix(a22, bases)
    res = lp.verdict(rows22, float(gate["line"]))
    check("a22_v2_1 vs 3-run mean == -0.147 (minutes §3.3a)",
          abs(float(res["mean_dlc_full"]) + 0.147) < 5e-4,
          str(res["mean_dlc_full"]))
    check("a22 total actions == 3994 (systems review)",
          res["total_actions_arm"] == 3994, str(res["total_actions_arm"]))
    print(f"        matched-prefix delta lc = {res['mean_dlc_matched_prefix']}, "
          f"pooled latency ratio = {res['pooled_latency_ratio']}")

    # the endpoint must actually be able to emit INCONCLUSIVE-ON-LATENCY
    synth = [lp.PrefixRow("g", 60, 100.0, 60, 0, 1.0, -1.0, 1.0, 1.0, 0.0,
                          0.40, 90.0, 54.0, 1.667)] * 25
    sv = lp.verdict(synth, -0.190)
    check("synthetic slow-but-not-harmful arm reads INCONCLUSIVE-ON-LATENCY",
          sv["verdict"] == "INCONCLUSIVE-ON-LATENCY"
          and sv["counts_as_K3prime_strike"] is False, json.dumps(sv["verdict"]))
    synth2 = [lp.PrefixRow("g", 100, 100.0, 100, 0, 1.0, -1.0, 0.0, 1.0, -1.0,
                           0.0, 79.2, 79.2, 1.0)] * 25
    sv2 = lp.verdict(synth2, -0.190)
    check("synthetic genuinely-harmful arm reads FAIL and IS a K3' strike",
          sv2["verdict"] == "FAIL" and sv2["counts_as_K3prime_strike"] is True)


# ==========================================================================
def item2_namespace_reuse() -> None:
    print("\n[ITEM 2] namespace_reuse_rate estimator")
    # -- estimator power: it must NOT be a degenerate constant (S1's failure mode)
    reuse_calls = [
        type("C", (), {"code": "def helper(x):\n    return x*2\nseg = 1\n",
                       "result": "", "fault": None, "is_fault": False})(),
    ] + [
        type("C", (), {"code": "print(helper(seg))\n", "result": "",
                       "fault": None, "is_fault": False})()
        for _ in range(9)
    ]
    pos = nr.game_nrr(reuse_calls, "synthetic_positive")
    check("positive control reads 1.0", pos.nrr_epoch == 1.0, str(pos.nrr_epoch))
    neg_calls = [
        type("C", (), {"code": f"seg{i} = {i}\nprint(seg{i})\n", "result": "",
                       "fault": None, "is_fault": False})()
        for i in range(10)
    ]
    neg = nr.game_nrr(neg_calls, "synthetic_negative")
    check("negative control reads 0.0", neg.nrr_epoch == 0.0, str(neg.nrr_epoch))

    # -- infra confound: a timeout must reset the epoch and be reported
    faulted = list(reuse_calls)
    faulted[4] = type("C", (), {
        "code": "print(helper(seg))\n",
        "result": "error:\nTool timed out after 30s\n",
        "fault": "timeout", "is_fault": True})()
    fg = nr.game_nrr(faulted, "synthetic_fault")
    check("timeout is detected as a destruction event",
          fg.destruction_events.get("timeout") == 1, json.dumps(fg.destruction_events))
    check("epoch resets after a timeout (eligible < index>=1)",
          fg.n_eligible < fg.n_index_ge1, f"{fg.n_eligible} < {fg.n_index_ge1}")
    check("infra depression is separated: pre-fault 1.0 > epoch > raw",
          fg.nrr_prefault == 1.0 and fg.nrr_epoch > fg.nrr_raw,
          f"prefault={fg.nrr_prefault} epoch={fg.nrr_epoch} raw={fg.nrr_raw}")
    check("post-fault calls are counted and bound the infra attribution",
          fg.n_post_fault == 5 and fg.max_infra_drop == fg.post_fault_fraction,
          f"post_fault={fg.n_post_fault} max_infra_drop={fg.max_infra_drop}")
    check("re-definition of a destroyed name is counted as recovery, not adoption",
          fg.n_recovery == 0 and fg.n_reuse == 4,
          f"recovery={fg.n_recovery} reuse={fg.n_reuse}")

    # -- baseline measurement on real no-P1 transcripts
    reports = [nr.pull_nrr(p) for p in BASELINE]
    for rep in reports:
        print(f"        {rep['pull']}: nrr_epoch={rep['pooled_nrr_epoch']:.4f} "
              f"calls={rep['n_calls']} intact={rep['epoch_intact_fraction']:.4f} "
              f"faults={rep['destruction_events']} "
              f"persistence_errors={rep['expected_persistence_errors']}")
    pooled = [float(r["pooled_nrr_epoch"]) for r in reports]
    check("baseline nrr measured on 3 same-config pulls",
          len(pooled) == 3 and all(p < 0.02 for p in pooled),
          f"mean={st.mean(pooled):.5f}")
    all_games = [g["nrr_epoch"] for r in reports for g in r["games"]]
    check("all 75 baseline game-readings sit below the 0.15 floor",
          all(x < 0.15 for x in all_games), f"max={max(all_games):.4f}")
    check("real timeouts DO occur on the rail (the §5.4 confound is real, not hypothetical)",
          sum(sum(r["destruction_events"].values()) for r in reports) > 0,  # type: ignore[union-attr]
          json.dumps([r["destruction_events"] for r in reports]))

    # -- independent ground truth: sandbox NameErrors on prior-turn names
    # F(t): calls the static analyser says reference an unresolved global.
    # Ground truth: the sandbox itself raised NameError on that call.
    tp = tn = fp = fn = 0
    reuse_tp = reuse_fp = 0
    for pname in BASELINE:
        pull = load_pull(pname)
        for _gid, calls in load_calls(pull).items():
            prior: set[str] = set()
            for c in calls:
                blk = nr.analyse_block(c.code)
                flagged = bool(blk.parsed and blk.free_refs)
                truth = bool(nr._NAMEERROR_RE.search(c.result))
                tp += flagged and truth
                fp += flagged and not truth
                fn += (not flagged) and truth
                tn += (not flagged) and (not truth)
                if blk.parsed and (blk.free_refs & prior):
                    reuse_tp += any(n in prior
                                    for n in nr._NAMEERROR_RE.findall(c.result))
                    reuse_fp += not any(n in prior
                                        for n in nr._NAMEERROR_RE.findall(c.result))
                if blk.parsed:
                    prior |= blk.module_bound
                if c.is_fault:
                    prior = set()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    check("free-reference detector: zero false positives vs NameError ground truth",
          fp == 0, f"tp={tp} fp={fp} fn={fn} tn={tn} "
                   f"precision={prec:.3f} recall={rec:.3f}")
    check("every REUSE call the estimator flags is corroborated by the sandbox",
          reuse_fp == 0, f"corroborated={reuse_tp} uncorroborated={reuse_fp}")
    print(f"        detector recall vs NameError ground truth = {rec:.3f} "
          f"({fn} residual misses: a name bound only inside a nested scope in "
          f"the same block -- the estimator is CONSERVATIVE, it undercounts "
          f"reuse, which is the dangerous direction for a kill-gate)")

    # -- K4 guard: must refuse to fire when the epoch is not intact
    confounded = dict(reports[0])
    confounded["status"] = "INFRA-CONFOUNDED"
    confounded["epoch_intact_fraction"] = 0.5
    k4 = nr.k4_read(confounded, 0.15)
    check("K4 refuses to fire when the run is INFRA-CONFOUNDED",
          k4["may_fire"] is False and k4["k4"] == "VOID-INFRA-CONFOUNDED")
    band = {"status": "OK", "pooled_nrr_epoch": 0.13,
            "epoch_intact_fraction": 0.95, "max_infra_drop": 0.05}
    check("K4 reads INDETERMINATE inside the infra attribution band",
          nr.k4_read(band, 0.15)["k4"] == "INDETERMINATE")


# ==========================================================================
def item3_patch_surface() -> None:
    print("\n[ITEM 3] declared prompt / tool-schema patch surface")
    man = json.loads(ps.MANIFEST_PATH.read_text(encoding="utf-8"))
    check("manifest exists and is populated", len(man["entries"]) >= 8,
          str(len(man["entries"])))
    ok, issues = ps.audit(man)
    check("manifest audits clean against the frozen fork", ok, "; ".join(issues))
    crit = [e for e in man["entries"] if e["channel"] == "tool_schema"]
    check("both tool-schema contradictions are declared (schema beats prompt at 27B)",
          len(crit) == 2, ", ".join(e["id"] for e in crit))
    check("the six llm-agents strings are all present",
          {(e["file"], e["resolved_line"]) for e in man["entries"]}
          >= {("prompts.py", 80), ("prompts.py", 82), ("prompts.py", 107),
              ("prompts.py", 113), ("tool_agent.py", 230), ("tool_agent.py", 1347)})
    blocking, _info = ps.sweep_unlisted(man)
    check("zero UNDECLARED model-facing contradictions", not blocking,
          "; ".join(blocking))
    check("sweep has no blind spots on its own declared entries",
          all(ps._SWEEP_RE.search(str(e["baseline"])) for e in man["entries"]))
    check("prompt module allowlist agrees with the item-5 SAFE_MODULES patch",
          set(ps.P1_SAFE_MODULES) == set(smp.patched_safe_modules()))

    # end-to-end: patch a copy and byte-audit it
    tmp = Path(__file__).resolve().parent / "_smoke_patched"
    tmp.mkdir(exist_ok=True)
    for fname in ("prompts.py", "tool_agent.py"):
        src = ps._src(fname).read_text(encoding="utf-8")
        (tmp / fname).write_text(ps.apply_patch(src, fname, man), encoding="utf-8")
    (tmp / "python_tool_sandbox.py").write_text(
        ps._src("python_tool_sandbox.py").read_text(encoding="utf-8"), encoding="utf-8")
    ok2, issues2 = ps.verify_patched(tmp, man)
    check("patched tree byte-audits clean", ok2, "; ".join(issues2))
    (tmp / "prompts.py").write_text(
        ps._src("prompts.py").read_text(encoding="utf-8"), encoding="utf-8")
    ok3, issues3 = ps.verify_patched(tmp, man)
    check("byte audit CATCHES an unpatched file", not ok3, f"{len(issues3)} issues")
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()


# ==========================================================================
def item4_drop_policy() -> None:
    print("\n[ITEM 4] §6.1 restated as a drop-policy invariant")
    hashes = dp.policy_source_hashes()
    check("all five policy functions hashed for the freeze clause",
          len(hashes) == 5, ", ".join(sorted(hashes)))
    pull = load_pull("war_eval_v1")
    gid, tpath = next(pull.iter_transcripts())
    system, msgs = dp.reconstruct_messages(tpath)
    check("system prompt recovered from a real transcript", len(system) > 5000,
          f"{len(system)} chars")
    arm_system, applied = dp.p1_system_prompt(system)
    check("all six system-prompt surface strings apply to the rendered prompt",
          len(applied) == 6, str(len(applied)))
    td = dp.token_delta(system)
    check("token delta is a small positive constant", 0 < td["delta_tokens"] < 400,
          json.dumps(td))
    rep = dp.validate_on_pull("war_eval_v1", n_games=5)
    games = rep["games"]
    n_naive_void = sum(g["naive_6_1"]["verdict"] == "VOID" for g in games)
    n_comp_hold = sum(g["compensated_6_1prime"]["verdict"] == "HOLDS" for g in games)
    check("naive §6.1 self-voids on every game (the FATAL, reproduced)",
          n_naive_void == len(games), f"{n_naive_void}/{len(games)}")
    check("§6.1' with the compensating budget holds on every game",
          n_comp_hold == len(games), f"{n_comp_hold}/{len(games)}")
    tot_drops = sum(g["n_drop_events_baseline"] for g in games)
    check("the replay actually exercised the eviction path", tot_drops > 1000,
          f"{tot_drops} drop events")
    # the comparator must be able to fail
    a = [dp.DropEvent(1, 0, ("user",), 5)]
    b = [dp.DropEvent(1, 0, ("assistant",), 5)]
    check("trace comparator detects a divergence",
          dp.compare_traces(a, b)["verdict"] == "VOID")
    check("recorder patch source is syntactically valid",
          compile(dp.RECORDER_PATCH_SOURCE, "<recorder>", "exec") is not None)


# ==========================================================================
def item5_safe_modules() -> None:
    print("\n[ITEM 5] SAFE_MODULES gap + risk class")
    ok, missing = sf.tycho_state_importable()
    check("gap confirmed: Tycho State not importable on baseline",
          (not ok) and missing == ["dataclasses", "enum", "typing"], str(missing))
    ok2, missing2 = sf.tycho_state_importable(frozenset(smp.patched_safe_modules()))
    check("patched allowlist closes the import gap", ok2 and not missing2)
    check("baseline sandbox cannot execute ANY class statement",
          "__build_class__" not in sf.safe_builtins())
    check("modules alone are NOT sufficient",
          smp.tycho_probe(smp.patched_safe_modules())["ok"] is False,
          str(smp.tycho_probe(smp.patched_safe_modules()).get("error")))
    full = smp.tycho_probe(smp.patched_safe_modules(),
                           extra_builtins=smp.ADDED_BUILTINS, register_module=True)
    check("full patch makes the Tycho State dataclass constructible",
          full["ok"] is True, json.dumps(full))

    boot = sf._bootstrap_source()
    patched = smp.patch_all(boot)
    import ast
    ast.parse(patched)
    check("patched bootstrap parses", True, f"{len(boot)} -> {len(patched)} chars")
    check("patched bootstrap carries all four edits",
          '"dataclasses",' in patched and '"__build_class__",' in patched
          and 'sys.modules["__python_tool__"]' in patched
          and "RLIMIT_AS" in patched)
    try:
        smp.patch_bootstrap_text(patched)
        drifted = False
    except ValueError:
        drifted = True
    check("patcher refuses to double-apply / patch drifted source", drifted)

    check("no memory rlimit today (§5.4)", "RLIMIT_AS" not in sf.rlimit_kinds(),
          str(sorted(sf.rlimit_kinds())))
    demo = smp.escape_demo()
    check("direct `import os` is blocked", demo["direct_os_import_blocked"] is True)
    check("but the CURRENT allowlist already reaches the real builtins",
          demo["real_import_reachable"] is True,
          "=> the three added modules do not change the risk class")
    spec = smp.P1_SANDBOX_CANARY_SPEC
    check("canary spec carries the concurrency correction (28, not 16)",
          "28" in json.dumps(spec["live_children"]))
    check("canary spec carries the RLIMIT_CPU re-accounting hazard",
          "SIGXCPU" in json.dumps(spec["rlimit_cpu_reaccounting"]))
    banner = parse_solver_banner(load_pull("war_eval_v1").path)
    check("concurrency is read from the banner, not hard-coded",
          banner["concurrency"] == 28, str(banner["concurrency"]))


def main() -> int:
    print("=" * 74)
    print("P1 instrumentation smoke test  --  offline, $0, zero kernel pushes")
    print("=" * 74)
    item1_latency_prefix()
    item2_namespace_reuse()
    item3_patch_surface()
    item4_drop_policy()
    item5_safe_modules()
    print("\n" + "=" * 74)
    print(f"RESULT: {PASS} PASS / {FAIL} FAIL")
    print("=" * 74)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
