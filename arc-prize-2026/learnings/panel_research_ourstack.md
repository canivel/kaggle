# Internal Stack Audit — ARC-AGI-3 Campaign Forensics (2026-07-06)

Auditor: claude-fable-5. Sources: project memory LB table, fable_external_review_2026-07-06.md, exec_wm/scale_summary.md, runs/.

## 1. Score history — what is actually proven

- **v35 family is the only proven producer.** Distribution across ~15 draws: **{0.10–0.43}**, mean ~0.25, std ~0.10. Samples: 0.43 (v31), 0.31, 0.30, 0.28, 0.25, 0.25, 0.22×3, 0.21, 0.19, 0.14 (v66), 0.10. **v61 = 0.43** (2026-06-19) ties ATB but is statistically a top draw of the same distribution, not a new architecture level.
- **Public LB noise ≥0.21 on identical code** (v31 0.43 vs v32 0.22; v46 0.33 vs 0.24). No single submission can detect improvements < ~0.25. Local eval harness is the only usable A/B signal.
- **Local ceiling proven:** v35 ≈ 24 RHAE / 28 levels / 16 of 25 games. 9 zero-solve games stay L=0 at 5× budget — structural, not budgetary. Field context: leader 1.56, top-100 = 1.04; our 0.43 ≈ rank 275/1626. **v35-family tuning cannot close a 4–5x gap.**

## 2. Dead branches with root causes

- **SG family (5 Kaggle samples, 0.01–0.22, mean ~0.11):** local-Kaggle collapse — v55 local 0.199 RHAE → Kaggle 0.02; v36 pretrained 0.01 (BFS-only data 78.8% positive labels = poisoned). The local-vs-Kaggle divergence mechanism was **never debugged**; branch abandoned on results, not diagnosis.
- **JEPA (v45, v62, v63 — 3 ERRORs → declared dead). CRITICAL RE-OPENING FLAG:** the 2026-06-28 structural-drift root cause (build_notebook.py wrote agents/__init__.py missing `Swarm` import, .env missing gateway keys, no kaggle metadata block) explains **all five** ERRORs including every JEPA run. **JEPA was never actually executed on Kaggle** — every "JEPA ERROR" was a notebook-structure crash before any game started. The 3-strike verdict in feedback_arc_jepa_dead.md is built on contaminated evidence. (v66, post-fix, ran clean — proving the fix, not JEPA's guilt.)
- **5x structural-drift ERRORs (v45/v62/v63/v64/v65):** a month of misattributed failures ("model too big", "slug curse", "throttle it"). Fixed 2026-06-28; preflight.py now mandatory.
- Also dead: AXIOM (0 levels, algorithm-fit), macro-probe/replay/dynamic-budget (all null individually — yet compounded into v61's 0.43 draw).

## 3. Live assets

- **25 opus-generated executable sims** (exec_wm/): 22/24 Class-A (≥50% state-exact), **11 at 100% state-exact** (ft09, lf52, lp85, ls20, sb26, sp80, tn36, tr87, tu93 + s5i5/su15 at 99.5), zero crashes, ~10h opus wall / ~7M tokens invested. v66 shipped them at 0.14 (bottom of band); forensics blames hook design (fires every tick, no budget cap), not sim quality.
- **eval_harness.py**: RHAE-correct local A/B, per-level human baselines, ~40min/25-game sweep.
- **Submission daemon + preflight**: daily_submit.py + queue.py + preflight.py (8 structural checks) + kaos_supervisor.py semantic review. Caveat: queue sat empty 6/29–7/6.
- **JEPA-XXS 2.3M int8 distilled model** + weights dataset (canivel/jepa-wm-weights lineage) — untested on Kaggle in reality (see §2).
- **BFS solver** (engine-clone search, set_data-fixed, load-bearing on 14 games), **GraphExplorer** (Rudakov-style, per-level fallback), **CNN online learner** (random-init, online-only — pretrained variants all failed).
- 100k balanced random-policy trajectory dataset; forensic smoke evals (runs/eval_v64_local.json, eval_v65_local.json: 5 games, 2 solved levels, 0 errors).
- **NOT FOUND:** runs/eval_v65_vs_v35_forensics.json does not exist — the v66 forensics A/B has no completed local artifact yet; conclusions about the hook remain hypothesis-grade.

## 4. Honest capability gaps

- **No LLM-in-the-loop agent.** The Milestone-1 winner (Cottaar/Tufa: Qwen3-27B FP8 via vLLM in-kernel) is orthogonal to everything we have; repro fork queued but not run.
- **No TTT** — the MindsAI-lineage gap we identified in June and never built.
- **No GPU usage in any of our kernels** — CPU-only stack while the winning band runs H100-class inference; our 110-thread-CPU constraint assumption is obsolete under BYOD docker + GPU.
- No LLM code-synthesis at runtime (sims are pre-baked offline); no per-game router shipped; 7-day submission cadence gap just ended.

**Net:** one proven 0.25-mean architecture at its ceiling, a strong but unvalidated-in-anger sim asset, one falsely-convicted branch (JEPA), and a winner's playbook we haven't absorbed.
