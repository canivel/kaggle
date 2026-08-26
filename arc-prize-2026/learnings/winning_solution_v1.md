# WINNING-SOLUTION PROPOSAL v1 — ARC-AGI-3 Campaign

**Author:** Proposal Architect (claude-fable-5) · **Date:** 2026-07-06 · **Targets:** Milestone #2 (Sep 30, top-3 paid), Final Top Score (Nov 2, top-5 paid on ~55 private games)

---

# Thesis

The money is decided on the **private ~55-game set**, where per-game heuristics compress (AERA proved the public 25 games are solvable by trivial strategies; arXiv:2605.25931) and where 100+ duck-fork teams' hardcoded tweaks will regress. Our bet: **adopt the open-sourced Duck harness as our floor within one week** — it is the only >1% offline result and is already everyone else's floor (49+53 teams crossed 1.0 in the last two days alone) — then graft on **three capabilities the fork swarm demonstrably lacks and cannot copy from any public notebook**: (1) a deterministic search substrate — our proven BFS engine-clone solver (load-bearing on 14 games) plus Rudakov salience-tiered graph exploration with a state-dedup archive — exposed as tools inside Duck's Python sandbox; (2) a **runtime Rodionov verify-refactor executable-world-model loop** driven by the in-kernel coder LLM — the highest-verified technique on record (58.12% RHAE post-leakage-audit, arXiv:2605.05138v2), which we have already de-risked at build-quality level (22/24 Class-A sims, 11 at 100% state-exact); and (3) **per-game test-time training** of a small value net — validated by StochasticGoose (12.58%, Preview winner) yet unclaimed in the current meta. Every addition is A/B-gated on our RHAE-correct **local harness**, because the public LB carries ≥0.21 noise on identical code and cannot detect any single improvement we will make. This wins because it is general-by-construction (search + learned world models + online learning, zero game-specific logic), which is precisely what the 55-game private eval selects for.

# Evidence base

What we know for certain, with sources:

1. **Duck harness = 1.21% Kaggle RHAE with Qwen 3.6 27B FP8 on vLLM in-kernel; open-sourced Jun 30** (notebook `jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner`; source bundle inspected). Its architecture is a coding agent: one `python` tool, `action()` batching, segmentation-first object graphs with shape-hash tracking, regex-persisted labeled world model. Its confirmed weaknesses: high variance (author says the cleaned notebook never reproduced 1.21), amnesia at level transitions, **no systematic exploration** (no dedup, no archive), 4-connected single-color segmentation ceiling. (panel_research_winners.md)
2. **Diffusion is near-instant**: 125 teams ≥1.0 within 6 days of open-sourcing; leader 1.56 is itself a small duck delta (model swap, prev-frame tweak). Public top-10 by Sep 30 extrapolates to **1.7–2.0**. (panel_research_lb.md)
3. **Public LB noise ≥0.21 on identical code** (v31 0.43 vs v32 0.22); our local eval_harness.py (~40 min/25-game sweep, per-level human baselines) is the only usable A/B instrument. (panel_research_ourstack.md)
4. **Our v35 family is at its structural ceiling**: mean 0.25, 9 games stuck at L=0 even at 5× budget. Tuning it cannot close a 4–6× gap. (ourstack §1)
5. **Rodionov's model→verify→refactor→plan loop survives its own leakage audit** (5 channels found, contaminated runs discarded, 58.12% stands); Rudakov's LLM-free graph exploration took 3rd on Preview private LB with open code. (panel_research_literature.md §1–2)
6. **Live assets**: 25 opus-built executable sims (22/24 Class-A); BFS solver; GraphExplorer; eval harness; submission daemon + preflight (8 structural checks); a falsely-convicted JEPA branch (all 3 "strikes" were notebook-structure crashes, never actual executions). (ourstack §2–3)
7. **Hard process rules** (paid for in blood): never build kernels from scratch — fork arc3-baseline/duck and run preflight.py; match kernel-metadata.json exactly; runtime-test before every push; per-game generality over public-LB luck. (memory: feedback_arc_kernel_structural_drift, feedback_kaggle_env_match, feedback_test_before_submit, feedback_arc_generalization_first)

# The plan (phased, Jul 7 – Sep 30)

**Cadence invariant:** the daily-submit daemon queue (`scripts/queue.py`, Task `ARCDailySubmit`) is never empty. Default filler: best-known build + one pre-registered variant. The 6/29–7/6 empty-queue lapse does not recur; a weekly queue-depth check is added to preflight.

### Phase 0 — Duck reproduction & baseline lock (Jul 7–13)
**Build:** Fork the Duck notebook verbatim (fresh slug, kernel-metadata.json byte-matched to the original: GPU, docker image, dataset sources). Stand up local duck-eval on a RunPod A40 (Qwen 3.6 27B FP8 fits in 48 GB) wired to eval_harness.py. Run duck locally 3 seeds to establish **RHAE mean ± std baseline**.
**Hypothesis:** we can reproduce ≥1.0 on Kaggle and get a stable local baseline distribution.
**Success criterion:** Kaggle score ≥0.9 within 3 submissions AND local 3-seed baseline with std characterized.
**Slots:** 7 (daily: duck fork ×3, v35 fillers ×4).
**Fallback:** if GPU kernel misbehaves, bisect against the unmodified public notebook (which scores 1.2–1.5 for forkers — env issues are on us, not the code).

### Phase 1 — Tool fusion: search substrate inside the sandbox (Jul 14–27)
**Build:** Expose three tools in Duck's Python sandbox: `bfs_solve()` (our engine-clone solver), `explore()` (GraphExplorer with Rudakov's 5-tier salience click prioritization + directed state graph + shortest-path routing to untested state-actions), and a **state-dedup archive** (hash of segmentation graph → visited; Go-Explore-style return-to-frontier). A lightweight router lets the LLM delegate: "goal understood → bfs; goal unknown → explore()." Persist the archive across level transitions (fixes duck amnesia partially).
**Hypothesis:** deterministic exploration + dedup lifts exactly the sparse-feedback games where duck starves, and BFS instantly closes the 14 games our solver already owns.
**Success criterion (local, paired per-game, 3 seeds):** +≥3 levels aggregate over Phase-0 baseline AND no game regresses by >1 level in ≥2/3 seeds.
**Slots:** 14 (daily: best build + variant).
**Fallback:** ship tools as opt-in with token-budget caps; if the LLM misuses them, hard-gate `explore()` to fire automatically after N no-progress turns (scripted, not LLM-invoked).

### Phase 2 — Runtime executable world models + variance kill (Jul 28–Aug 17)
**Build:** Rodionov loop with the in-kernel Qwen: LLM writes/refactors a Python transition model, `verify()` replays it against `transitions` history (exact-match scoring — same Class-A metric as our offline sims), plans through it (BFS over the model) before acting. Seed the loop with our sim template library (the 25 opus sims become few-shot scaffolds, not hardcoded lookups — no game identifiers, structure only). Add **multi-attempt voting**: within the 2.2 h/game budget, restart-with-carried-notes after stagnation; keep best level reached. Add persistent cross-level world model (duck wipes it; we carry a distilled summary).
**Hypothesis:** even a 27B coder gets Class-A models on a meaningful fraction of games when verification is mechanical and templates constrain scope; planning through a verified model beats raw LLM action-guessing.
**Success criterion (local):** runtime WM reaches ≥50% state-exact on ≥8/25 public games; aggregate +≥4 levels over Phase-1 build; variance (std of RHAE across 3 seeds) reduced ≥30% by voting.
**Slots:** 21.
**Fallback:** if 27B code-gen is too weak, degrade to verify-only (model predicts, mismatch triggers re-exploration — DreamTeam's contradiction signal) and keep voting, which is model-strength-independent.

### Phase 3 — TTT in-kernel (Aug 18–Sep 7)
**Build:** Small CNN value net (random-init, online-only — pretrained variants are proven poison per SG forensics) trained during play on (segmentation-graph features → progress delta); used to rank candidate actions/probes proposed by the LLM and to prioritize the exploration frontier. GPU is co-resident and idle between vLLM batches. Optionally re-test JEPA-XXS (2.3M int8) as feature extractor — it has **never actually run** on Kaggle (all ERRORs were structural), so one clean trial is cheap and pre-registered.
**Hypothesis:** online value learning converts wasted probe actions into directed ones (StochasticGoose precedent) and is the one component no fork team has.
**Success criterion (local):** +≥2 levels aggregate, or ≥15% fewer actions per level solved at equal levels (efficiency feeds RHAE directly).
**Slots:** 21.
**Fallback:** keep the net as tie-breaker only in `explore()`; zero-cost removal flag.

### Phase 4 — Hardening & milestone selection (Sep 8–30)
**Build:** Full ablation matrix on local harness (each Phase 1–3 component in/out, 3 seeds); fix the two best configs; **freeze Sep 22**; submit both daily on alternating days through Sep 29 to sample public-LB noise (we know σ≈0.1+; 6–8 draws materially raise expected max). Preflight + runtime smoke test on every push, no exceptions. Prepare open-source package (milestone eligibility requires it) with a 24 h delay buffer.
**Success criterion:** best Kaggle draw ≥1.6 (above current leader; consistent with our +9-level local delta target compounding on duck's base).
**Slots:** 22.
**Fallback:** if compounding failed, submit best single-phase build; a clean duck-repro + voting alone plausibly lands 1.3–1.5 (forks with smaller deltas already did).

**Oct 1 – Nov 2 (post-milestone, sketch):** absorb the Sep 30 open-source wave within 5 days (proven diffusion speed), re-run the ablation with any newly released ideas, and spend October on private-set generality: adversarial self-testing on held-out local games our components never saw during development.

# Differentiators

Why we don't end up mid-pack in the 1.0–1.3 fork band:

1. **We add search; they add prompts.** No fork on the board has a state-dedup archive, Go-Explore frontier, or BFS-through-verified-model planning. Duck's own author flags exploration as the missing piece; we own working implementations today (BFS solver, GraphExplorer).
2. **We can measure; they gamble.** ≥0.21 public noise means fork teams are hill-climbing on dice rolls. Our local harness gives paired per-game deltas at 3 seeds in hours — every merged component is a verified win, not a lucky draw.
3. **We've already paid the world-model tax.** 10 h of opus and 7M tokens produced a Class-A sim methodology, verification metric, and template library — the exact scaffolding a 27B needs to run the Rodionov loop that scored 58% with frontier models. Forks would start this from zero.
4. **Variance engineering.** Voting/restart-with-notes attacks the documented luck-dependence of the 1.21 result; on the private set (more games) low variance compounds into rank.
5. **Generality bias is enforced, not aspirational**: no game identifiers anywhere (Rodionov's audit is our checklist), components validated on aggregate-minus-regression criteria, final selection biased to the config with the best worst-case game.

# Resource budget

- **Kaggle GPU (30 h/wk):** ~12 h = one full submission run; budget 2 runs/wk for scored submissions + 6 h smoke tests. Daily queue mixes full runs with CPU-cheap fillers to keep cadence without burning quota.
- **Local 3080 (10 GB):** cannot host 27B; used for CNN/TTT prototyping, segmentation/graph unit tests, sim verification, eval-harness CPU sweeps.
- **RunPod A40 (48 GB):** the local duck-eval instance. ~$0.40/h spot; 3-seed 25-game sweep ≈ 8 GPU-h ≈ $3.20. Budget 40 h/wk ≈ $65/wk ≈ **$780 through Sep 30** — trivially worth it as the primary A/B instrument.
- **Opus/frontier tokens:** ~5M/wk for code generation, sim-template refactoring, forensic analysis (KAOS-orchestrated, claude-opus-4-8 per campaign default). Runtime agent tokens are free (in-kernel Qwen).
- **Human/agent wall-clock:** phases sized to ~1 merged, verified component per 2 weeks — matching our historical shipping rate, not aspiration.

# Kill criteria (pre-registered)

- **P0:** duck fork <0.8 on Kaggle after 3 metadata-matched attempts on fresh slugs → stop forking, bisect env for ≤4 days; if still failing by Jul 20, the GPU-kernel path is dead → pivot to CPU track (v35 + Rudakov fusion) targeting ~0.6 and re-plan.
- **P1:** tool fusion fails its criterion in 2 consecutive local ablations → demote tools to scripted-fallback-only (no LLM routing); if even that regresses, revert to pure duck and skip to P2.
- **P2:** runtime WM Class-A on <4/25 games after template seeding and one refactor-prompt iteration → kill runtime synthesis permanently (27B ceiling confirmed); keep verify-only contradiction signal + voting.
- **P3:** TTT net shows no local gain in 2 ablations, or JEPA-XXS ERRORs once on a preflight-clean notebook → both dead, no third chances (feedback_arc_long_bfs_mcts taught us caps; this teaches us strikes only count when the run actually executed).
- **Global:** any component that costs >15% of the 2.2 h/game budget without meeting its criterion is removed regardless of trend-line hope.

# Risks

1. **Public-LB noise buries real progress (likelihood: certain).** Mitigation: local-first gating for all decisions; public submissions only sample the distribution; Phase-4 multi-draw plan explicitly treats the LB as a max-of-N lottery on a distribution we improve locally.
2. **27B can't run the Rodionov loop (high).** His results used GPT-5.5. Mitigation: mechanical verification (no LLM judging), sim-template scaffolds, MDL-style "simplest model that fits" prompts, and a pre-registered degrade path (verify-only) that still pays.
3. **Sep 30 wave resets the floor under us (certain).** Everyone gets the milestone winners' code Oct 1. Mitigation: our differentiators are systems (archive, TTT, harness) that compose with anyone's release; 5-day absorption drill is planned, and we open-source at the deadline, not before.
4. **Kernel/env fragility (proven, 5 ERRORs).** Mitigation: fork-never-build, preflight.py mandatory, byte-matched kernel-metadata, runtime smoke test before push, fresh slugs on any anomaly. These are hard gates in the daemon, not habits.
5. **Integration complexity starves the token budget** — search tools, WM loop, and TTT all compete for the 60 s tool-loop and 32k context. Mitigation: per-component wall/token caps (the forge_v35 hook post-mortem is the precedent), output truncation, and the global 15% kill rule; ablations measure token-per-level so efficiency regressions are visible, not silent.

---
*Sources: panel_research_literature.md, panel_research_winners.md, panel_research_ourstack.md, panel_research_lb.md (all in f:/kaggle/arc-prize-2026/learnings/).*
