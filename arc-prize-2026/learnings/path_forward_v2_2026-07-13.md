# PATH FORWARD v2 — Revised After Panel Round 5 (5× MAJOR-REVISION)

**Author:** claude-fable-5 · **Date:** 2026-07-13 · **Revises:** `path_forward_v1_2026-07-13.md` per `panel/round5/*` (rl-planning FATAL + 5×5/10) · **Evidence:** `failure_analysis_2026-07-13.md`, `headroom_analysis_2026-07-13.md`, `position_analysis_2026-07-13.md`, `panel_research_winners.md`, `panel_research_lb.md`, **new: null10 counterfactual replay (§E, run 2026-07-13, `runs/null10/merged_null_benchmark.json`)**
**Hard constraints (unchanged):** ~$68 RunPod remaining (reserve-only, zero default GPU spend); free Kaggle quota 30 h/wk; ~79 daily submission windows to Sep 30; Milestone-2 Sep 30; Final Nov 2 (~55 private games); no per-game parameters on private-set-facing logic.

---

## Change-log keyed to panel objections

| ID | Objection (reviewer) | Disposition | Where |
|---|---|---|---|
| RL-F1 / LA-M3 / ME-M2 / PS-M3 / SY-M3 | L1's +0.10–0.30 EV contradicts the STUCK finding (FATAL) | **Accepted — replay run, EV retracted.** New null10 replay: only 12/152 clears (7.9%) occur after cumulative action 120; 7/152 (4.6%) in the final 20% of run wall. Marginal-compute-to-progressing-games channel ≈ 0. L1 is demoted to a non-scoring throughput guard; the restart lever (flip-game bimodality, EV re-derived at +0.10 net) is the R1 primary. | §E, §R1 |
| RL-M1 / LA-M4 | L1/L2 interaction semantics unspecified; dead-game thrash | **Accepted.** Merged into one "attempt scheduler" with explicit counter semantics, cap 2 restarts/game, park-after-cap, precedence, and simulated dead-game behavior (bounded at 270 actions vs unbounded grind). | §R1 |
| RL-M2 / LA-M5 / ME-M4 / PS-M2 | R2 ≥3/9 gate treats clustered trials as independent; passable by one-game artifact; inconsistent with window threshold | **Accepted.** Primary gate is now game-level: level-2 clears on **≥2 distinct games** with **≥2/3 seeds each**. Under game-level nulls (0/30 in null10; per-game rule-of-three p<0.1) the two-game requirement gives p<0.01 even with perfect seed clustering. A 2-game crack ≈ +0.35 local → ~+0.19 official > +0.12 window threshold — local and confirmation gates now consistent by construction. | §R2 |
| PS-M4 | Forensics designed and gated on the same 3 games; no out-of-sample test | **Accepted.** r11l (4th 10/10-L1, 0/10-L2 game) is a pre-registered directional holdout: a falsifiable r11l prediction must be published before any GPU spend; an r11l result contradicting the mechanism hypothesis blocks confirmation regardless of the grinder count. | §R2 |
| RL-M3 / LA-M2 / PS-M1 | R2 shortlist is slogans; BFS unspecified; no component fidelity gates | **Accepted.** Pre-registered forensics→intervention decision table; one-page design per shortlist item (state space, executed-vs-simulated, token cost/node) due with the Aug 3 deliverable; free CPU component gates (segmentation fidelity ≥90% on 20 hand-labeled grinder frames; exec-WM next-state accuracy ≥70% on held-out logged transitions) required before GPU. BFS is now specified as stall-scoped *executed* exploration over the duck's existing segmentation graph with a hard action/token budget. | §R2 |
| LA-M1 | Public Milestone-1 winning notebook absent from plan | **Partially rebutted, then accepted in stronger form.** The duck *is* the Milestone-1 winner (Tufa duck harness, 1.21; our substrate is its fork — see `panel_research_winners.md`). What was genuinely missing: an audit of the **1.28–1.56 fork band** above it (leader = Mathurin Ache 1.56, a duck fork with small deltas). Added mandatory R0 fork-delta audit + port-through-gate task. | §R0 |
| RL-M5 / PS-m1 | Nov-2 "compression of overfit rivals" asserted without evidence | **Accepted.** Downgraded from assertion to pre-registered hypothesis with a test: the R0 fork audit counts game-ID-keyed logic/public-set tuning in the top public forks; LB forensics (`panel_research_lb.md`: 82% of the ≥1.0 cohort landed within 2 days of the June-30 open-source; private set ≈55 vs 25 games) is supporting-but-not-sufficient evidence. A pre-registered failure consequence is defined. | §R0, §Risks |
| ME-M1 | "Redraws dead" fails the plan's own both-endpoints rule | **Accepted with recompute.** E[best-of-80] = 1.03 / 1.10 / 1.45 at σ = 0.044 / 0.074 / 0.213; P(≥1.17) = 0.000 / 0.032 / 1.000. The observed 5-draw range (0.20) implies σ̂_range ≈ 0.086, near the point estimate — but the ban is replaced by a **priority rule**: gates always preempt redraws; every unused window defaults to a best-build redraw, so the upper-endpoint lottery is retained at zero opportunity cost. | §Windows |
| ME-M3 / PS-M5 / LA-m1 | Frozen baseline + frozen null10 in a drifting environment | **Accepted.** Rolling 6-draw control (baseline + sentinels + best-build redraws); pre-registered drift rule (sentinel >2σ̂ → freeze in-flight gates, confirm, re-center); per-game drift statistic; null10 version-pinned with per-game refresh trigger on version-suffix bump (game IDs carry version hashes); paired designs restricted to version-matched games. | §Instruments |
| RL-M4 / PS-M6 / SY-M5 | Window gate uninterpretable at σ̂ upper endpoint; false-promote unquantified; R2 promotion authority ambiguous | **Accepted.** Error rates computed and printed: false-promote at Δ=0 is 2.4% (σ=0.074) / 24.5% (σ=0.213); false-kill of true +0.10 is 4.9% / 28.3%. Sign-flip rule pre-registered (extra window → provisional promote → must re-confirm inside the stack gate). df grows ~1/week via sentinels+redraws (df≈15 by Sep). **The free 2-window gate is the sole promotion authority**; "GPU confirmation sweep" language deleted. | §Instruments, §R2 |
| ME-M5 | v1 rehabilitation is unregistered subgroup analysis | **Accepted.** v1=+0.13/+0.42 reclassified as hypothesis; version-stability criterion (version-suffix match) is now pre-registered but its earlier application was post-hoc; explore-min enters R3 with **zero prior credence** and must pass a full gate on its own. The "true v2 cost ≈ −0.2 to −0.3" no longer leans on the p=0.22 non-significance. | §R3 |
| SY-M1 | $15–25 has no provenance; ignores token inflation | **Rebutted on inflation, accepted on provenance.** Runs are wall-clock-capped (7,920 s/game, ~25 games compressed to ~12 h/seed by concurrency; see `panel_research_winners.md` compute profile): token inflation reduces actions-per-wall, not dollars. Provenance: 3 seeds ≈ 36 GPU-h on the A40-class SKU that produced null10 at $0.39–0.79/h → $14–28. Mandatory pre-spend calibration: 1 seed first, measured $, with a pre-registered de-scoping table. | §R2 |
| SY-M2 | No Kaggle quota ledger | **Accepted.** Ledger added: submission of an already-committed kernel version costs 0 GPU-h (the daily daemon has run since May on this basis); each *new build* costs one ~12 h commit → cap ≈ 2 new builds/wk. Weekly ledger ≈ 27 h ≤ 30 h; measured hours verified at R0 exit. | §Windows |
| SY-M4 | 80-window budget contradictory; 45+ windows unaccounted | **Accepted.** Full reconciled ledger: 45 enumerated (gates 28, sentinels 11, R0 1, selection 5) + ~34 default best-build redraws. The redundancy-vs-ban contradiction is dissolved by the priority rule (see ME-M1 row). Sentinels corrected to 11 (1/wk × 11.3 wk). | §Windows |
| ME-m1 | Multiplicity and stacking unquantified | **Accepted.** Expected false promotions over ~6 families: 0.14 (σ point) to 1.5 (upper). Mitigations: stack gate vs **vanilla-duck fork** (own 2-window test), sequential gating against the updated rolling control, provisional-promote re-confirmation. | §R3 |
| ME-m2 | R2 secondary gate has no power at n=3 | **Accepted.** Replaced by a per-game paired sign statistic over ~20 version-matched non-wall games (game as exchangeable unit). | §R2 |
| SY-m1 | Local wall-clock doesn't transfer for RHAE | **Accepted.** All local gates expressed in actions-per-completed-level and tokens/action (hardware-independent), never wall-clock. | §R2 |
| SY-m2 | Weekly sentinel statistically blind at n=1 | **Accepted.** Per-game drift statistic from the sentinel event log (any game shifting >2 game-level sd flags), not aggregate score. | §Instruments |
| RL-m1 | Exploration closed as a family when only one implementation was falsified | **Accepted.** R2 forensics adds a state-coverage metric (distinct segmentation-graph states per 100 actions, null vs stuck) so "wall = exploration failure" is tested, not foreclosed; the family closure applies to *always-on context injection*, not to exploration as a mechanism (stall-scoped BFS is on the shortlist). | §R2 |

---

# Thesis (what actually wins now)

Redraws cannot be the strategy (P(top-100 by luck) ≈ 3% at the σ̂ point estimate, and the cutoff climbs +0.02/day), and always-on context injection is net-harmful as built. The null10 replay now shows the binding constraint precisely: **marginal compute on progressing games is nearly worthless (7.9% of clears after action 120; 4.6% in the final fifth of wall) — but within-game attempt variance is real (16/25 games flip across seeds) and the level-2 wall on three reliable grinders is worth +0.52 local.** So the revised strategy has three legs, in cost order:

1. **Fork-delta audit (free):** we run the Milestone-1 winner's harness; the 1.28–1.56 band above us consists of its forks with small public deltas. Audit, port the game-agnostic ones, gate each.
2. **Attempt scheduler (free):** convert dead compute into fresh attempts on bimodal games — restart, cap, park. Derived EV +0.10 net local; the only lever the replay data actually supports.
3. **One pre-registered shot at the level-2 wall ($≤68):** forensics → decision table → component fidelity gates → a two-distinct-game clear gate with an r11l holdout.

**Re-baselined targets (difficulty ratio 0.55, refit at R1):** Sep-30 top-20 needs local null ≈ 2.9–3.3; no identified stack reaches it — not claimed. Attempt scheduler (+0.10 local → +0.055 official) partially holds rank. Ported fork deltas: the 1.28–1.56 band is existence proof that +0.1–0.35 official sits in cheap game-agnostic deltas; target +0.1–0.25. An R2 two-game crack adds +0.35 local → +0.19 official. Full stack ⇒ draw mean ≈ 1.25–1.35, selection-best ≈ 1.4: **contests Sep-30 top-100 (projected cutoff 1.35–1.5)**. Formal objective unchanged: Sep-30 = instrumented checkpoint targeting top-100; the campaign is built for Nov-2 (~55 private games), where the compression hypothesis (§Risks, now a *tested* hypothesis) does the work. Nothing shipped may key on game identity.

# Evidence base

**Carried forward (unchanged from v1):** context pollution replicated (ar25 p=0.009/0.008 both arms with zero explores; su15 p≈0 both arms; tokens/action null 435 → v2 543 → v1 633); death mode is STUCK (9/126 good runs progressing in final 15% of wall; median good run burns 59% of tokens on one unfinished level); 5 games dead 10/10 consuming 20.2% of tokens; 4 grinders clear L1 10/10 and L2 0/10 (+0.52 local for +1 level on the three w=2 grinders); position rank 187 at 1.02, draws {0.82, 0.89, 0.93, 0.95, 1.02}, mean 0.922, σ̂ 0.074, χ² CI [0.044, 0.213].

**Downgraded (ME-M5):** substrate v1 = +0.13 (+0.42 version-stable) is a **hypothesis**, not a result — the version-stable subset was applied post-hoc. The criterion (game version-suffix match between arm and null10) is now pre-registered for all future paired scoring. v2's systematic cost is bounded in [−0.54, −0.2]; the ft09-lottery decomposition (p=0.22) is illustrative, not evidential.

## §E New: null10 counterfactual replay (2026-07-13; resolves RL-F1)

Computed from `runs/null10/merged_null_benchmark.json` (250 runs; `actions_per_level[i]` = actions spent at level i+1, level i cleared iff i < `levels_completed`):

- **152 true level clears.** Clears after cumulative action 120: **12/152 (7.9%)**; after action 90: 21/152 (13.8%). Clears in the final 20% of a run's wall: **7/152 (4.6%)**. First-level clear: median 25 actions, p90 114, max 194.
- **L1 truncation value corrected downward:** truncating all lc==0 runs at action 120 saves 1.20M tokens = **7.9% of total** (the 20.2% figure counts dead games' *entire* budget, most of which is burned before 120). The freed throughput's marginal value via the "more actions for progressing games" channel is ≈0 by the two bullets above. **The +0.10–0.30 L1 EV is retracted.**
- **Within-game attempt variance is real:** per-game lc across 10 seeds — 5 games 0/10 (dead), 4 games ≥1 in 10/10 (grinders), **16 games flip** (e.g. ft09 [0,0,0,1,2,2,2,2,2,3], tn36 [0×6,1,1,1,2], cn04 [0×5,1×5]). This is the evidence base for restarts: failure on flip games is *not* deterministic per game (contra RL-F1's second clause — that holds only for the dead 5).
- **Restart detector FP measured:** 14/126 good runs (11.1%) have first clear after action 90; 9/126 (7.1%) after 120.
- **Restart EV re-derived** from the headroom stabilization table (V = good-mode value, p = good-mode rate per game): one restart at action 90 with depth discount 0.4 → Σ p·0.4·V·25 = 3.14 game pts = **+0.126 mean**; a second restart adds only +0.02; FP cost (11.1% of good runs restarted pre-clear, partially recovered at p·0.4) ≈ −0.03 to −0.05. **Net attempt-scheduler EV ≈ +0.10 ± 0.05 local.** This replaces both v1 lever EVs.

# Standing instruments (free) — v2

- **null10** (10 seeds × 25 games, per-action events) is the paired-control corpus, **version-pinned**: every game ID carries its version suffix; paired scoring uses version-matched games only (pre-registered criterion). **Refresh trigger:** a version bump observed on any gate-relevant game (grinders, holdout, top-weight flips) invalidates that game's rows; if >5 gate-relevant games bump, in-flight local gates are void until partial re-collection (reserve-gated, grinder games only, ~$5).
- **Window gate v2.** Control = **rolling mean of the 6 most recent control-class draws** (frozen-fork baseline, weekly sentinels, best-build redraws; after a promotion the promoted build becomes the control class). Candidate gets 2 scored windows. SE = σ̂·√(1/2+1/6); promote at Δ ≥ +0.12, kill at Δ < 0, one extra window between. **Printed error rates:** false-promote at Δ=0 = 2.4% (σ̂=0.074) / 24.5% (σ̂ CI upper 0.213); false-kill of true +0.10 = 4.9% / 28.3%. **Sign-flip rule (pre-registered):** if the promote decision differs across the current σ̂ CI endpoints → one extra window; if still flipped → decide at the point estimate, mark **provisional**, and a provisional promote must independently re-confirm inside the stack gate before freeze. σ̂ df grows with every control-class draw (~1/wk sentinel + redraws): df ≈ 8 by first candidate gate, ≈ 15+ by September, shrinking the CI that drives the flip rule.
- **Drift rule (pre-registered):** weekly sentinel scored per-game — any game deviating >2 game-level sd from its control history, or aggregate >2σ̂ → freeze in-flight gate decisions, run one confirmation sentinel; confirmed drift re-centers the rolling control on post-drift draws only and restarts affected gates. (Answers ME-M3/LA-m1/SY-m2.)

# The plan

### R0 — Instruments + fork-delta audit (Jul 14–27; free)

1. Draw #6 completes the σ panel; offline scorer + version-pinned null10 committed; all thresholds in this doc pre-registered in `ITERATION_LOG.md` before any candidate submission.
2. **Fork-delta audit (new, LA-M1/RL-M5):** enumerate public kernels in the 1.28–1.56 band (all Milestone-eligible code is open-sourced); diff each against the vanilla duck. Deliverable by Jul 27: (a) a delta table (model swaps, prev-frame handling, prompt edits, scheduling) classified game-agnostic vs game-keyed; (b) a count of game-ID-keyed logic per fork — the **pre-registered test of the Nov-2 compression hypothesis** (if the top forks are predominantly game-agnostic, Risk #1 is rewritten and contingency windows convert to porting their harness improvements wholesale); (c) the top 2 portable game-agnostic deltas, each gated on 2 windows.
3. **Quota + window ledgers published** (below); measured commit-hours verified against the Kaggle usage page.
**Exit:** thresholds logged; draw #6 scored; audit deliverable filed.

### R1 — Attempt scheduler (Jul 14–Aug 3; free)

One component, full semantics (resolves RL-M1/LA-M4):
- **Trigger:** within a game, if `lc == 0` at **90 actions since episode start** (per-attempt counter, resets on restart) → RESET to fresh episode. FP measured: 11.1% of good runs, partially recovered by the fresh attempt.
- **Cap:** maximum **2 restarts per game** (3 attempts), counted by a **cumulative attempt counter that never resets**.
- **Park:** after the cap, if still `lc == 0`, the game stops consuming analyzer turns (it is parked, not abandoned — a parked game resumes only if all non-parked games are finished). Precedence: park dominates restart.
- **Dead-game behavior (simulated on null10 transcripts, published with the build):** the 5 dead games are bounded at ≤270 actions each then parked — strictly less compute than today's grind-to-wall; no thrash loop is possible because the cap counter is cumulative.
- **Throughput note (non-scoring):** parking frees shared vLLM throughput (games are token-bound, 28-way concurrent on one server) for the restart attempts; no independent EV is claimed for this channel (RL-F1 accepted).
- **EV: +0.10 ± 0.05 local** (§E derivation). Thresholds (90, cap 2) are null10 percentiles, game-agnostic, private-set-safe.
**Gate:** 2-window rule. **Retry budget:** one, with cap 1 (single restart) if the build fails. Mandatory `scripts/preflight.py` + runtime smoke before push.

### R2 — Level-2 wall (Jul 21–Aug 10; free CPU; ≤$68 reserve gated)

**Forensics (free, CPU):** mine null10 transcripts of sb26/su15/lp85 level-2 grinds. Metrics: mechanic verbalization (labels = human-derived from playing the three public games; leakage risk accepted and countered by the holdout + no-game-ID rule), hypothesis churn, action entropy, repeated-plan signatures, and **state coverage** (distinct segmentation-graph states per 100 actions, null vs stuck — RL-m1).

**Pre-registered decision table (LA-M2):**

| Observed signature (30 grinds) | Diagnosis | Intervention class |
|---|---|---|
| Mechanic never stated | Model-capability wall | Stall-scoped systematic exploration (duck+BFS) — *not* prompt restructure |
| Mechanic stated, then lost across turns | Context-management bug | Stall-scoped world-model pinning / verify loop |
| Mechanic stated, plan correct, execution wrong | Grounding failure | exec-WM action-verification loop |
| Low state coverage vs null | Exploration starvation | duck+BFS (coverage-driven) |

**Design specs due Aug 3 (RL-M3/PS-M1), one page each, pre-registered:** state space, what is simulated vs executed, cost in actions and generated tokens per node expanded, predicted tokens/action delta vs the >10% kill criterion. For duck+BFS specifically: node = segmentation-graph state abstraction of the observed frame (the duck's built-in object graph: color, shape-hash, containment, adjacency); edges = **executed** env actions (no forward model exists); it fires only after 90 stalled actions on a level, **replaces** duck analyzer turns rather than adding context, hard cap 40 actions per burst, LLM used only for frontier scoring.

**Component fidelity gates before any GPU spend (PS-M1, free):** (i) segmentation faithfulness ≥90% object-identity/transition consistency on 20 hand-labeled frames from grinder transcripts (the known 4-connected single-color fragmentation is the failure to quantify); (ii) if exec-WM is selected: ≥70% next-state prediction accuracy on held-out logged transitions. Fail → that intervention is struck from the shortlist.

**Reserve unlock:** iff a falsifiable mechanism prediction **and an r11l holdout prediction** (PS-M4) are pre-registered. **Cost provenance (SY-M1):** runs are wall-capped (7,920 s/game, ~25 games/seed ≈ 12 GPU-h at 28-way concurrency); 3 seeds ≈ 36 GPU-h on the null10 A40-class SKU at $0.39–0.79/h = **$14–28**; token inflation changes actions-per-wall, not dollars. **Calibration:** run 1 seed first; if measured 3-seed cost >$35 → 2 seeds × 3 games (gate: ≥2 distinct games at 2/2 seeds); >$50 → 1-seed screen, windows-only thereafter.

**Primary gate (RL-M2/LA-M5/ME-M4/PS-M2):** level-2 clears on **≥2 distinct games** of {sb26, su15, lp85}, with **≥2/3 seeds on each claimed game**. Null: 0/30 level-2 clears in null10; per-game rule-of-three p̂ < 0.1; two independent game-level events → p < 0.01 under full seed clustering. A one-game crack (expected ~+0.09 official < +0.12 window threshold) does **not** pass — local and confirmation gates are now consistent. **Holdout:** the r11l prediction is checked; contradiction blocks confirmation regardless of grinder count. **Secondary (no-collateral, ME-m2/SY-m1):** per-game paired sign statistic vs null10 over the ~20 version-matched non-wall games, expressed in actions-per-completed-level and tokens/action (never wall-clock); ft09 reported separately.

**Promotion authority (SY-M5):** the free 2-window gate, solely. The GPU run is a local screen. Remaining reserve funds at most one retry, only if the first run shows ≥1 game at ≥2/3 seeds (near-miss).

### R3 — Stack and freeze (Aug 11–Sep 30)

Merge window-promoted components sequentially; each later candidate gates against the **updated rolling control** that already contains earlier promotions (ME-m1). **Stack gate:** the final stacked build must beat the **vanilla-duck fork** mean by ≥ +0.12 on its own 2-window test — this is where provisional promotes are re-confirmed or dropped. Expected false promotions across ~6 families: 0.14 (σ̂ point) to 1.5 (CI upper); the stack gate and vanilla floor bound the damage. **Explore-min** (v2-gated explore, `PHASE1_ENABLE_REPL_ARCHIVE=0`, animation summaries ≤5/game) enters with **zero prior credence** (ME-M5): one full gate + one retry; then always-on context injection stays closed for Sep. **Freeze Sep 12.** Sep 13–30: 4–5 selection draws; ship criterion: mean ≥ control mean + 2σ̂·√(1/n+1/6) **at both σ̂ CI endpoints**; floor = never ship below the vanilla-duck fork.

# Submission windows (~79) and quota — reconciled ledgers (SY-M2/SY-M4)

**Window ledger (Jul 14–Sep 30, 11.3 weeks):**

| Class | Count |
|---|---|
| R0 draw #6 | 1 |
| Fork-delta ports (2 families × [2 gate + 1 extra]) | 6 |
| R1 attempt scheduler (2+1 gate, 2 retry) | 5 |
| R2 confirmation (2+1) | 3 |
| Explore-min (2+1) | 3 |
| Stack gate + vanilla-floor check | 3 |
| Second-generation contingency candidates (pre-registered before push, same rules) | 8 |
| Weekly drift sentinels (1/wk × 11) | 11 |
| Final selection draws | 5 |
| **Enumerated** | **45** |
| Default filler: best-build redraws (double as extra control-class draws → σ̂ df growth; retain the ME-M1 upper-endpoint lottery at zero opportunity cost) | ~34 |

**Priority rule (replaces the v1 "ban"):** a window is spent on a redraw only when no pre-registered gate candidate is ready to submit. Redraws of anything other than the current best build remain banned.

**Kaggle quota ledger (30 h/wk):** submitting an already-committed kernel version costs 0 GPU-h (the daily daemon has operated on this basis since May — scoring is organizer-side). New-build commit ≈ 12 h GPU → **cap 2 new builds/wk** (24 h) + smoke tests (2 × 1.5 h) ≈ 27 h ≤ 30 h. Enumerated candidate builds ≈ 12–15 commits over 11 weeks — feasible at half the cap. Measured commit-hours verified at R0 exit; if a commit measures >15 h, the cap drops to 1 new build/wk and contingency candidates are cut first.

# Kill criteria (pre-registered)

- **R0:** fork audit finds the top band predominantly game-agnostic → compression hypothesis fails → Risk #1 rewritten; contingency windows reallocate to wholesale porting; Sep objective restated (top-100 defense is no longer assumed recoverable at Nov).
- **R1:** combined build fails gate → cap-1 retry; that fails → vanilla scheduler, lever dead for Sep.
- **R2:** no falsifiable mechanism + holdout prediction by Aug 3 → **reserve stays unspent**; component fidelity gate fails → intervention struck; local primary <2 distinct games → dead, no reserve retry unless near-miss rule; r11l contradiction → confirmation blocked.
- **Context injection:** always-on REPL archive permanently dead for Sep (replicated p<0.01 both arms). Explore-min: one gate + one retry, zero prior credence.
- **Global:** no game-ID-keyed logic ships, ever. Any component raising tokens/action >10% over null must have passed its own gate. Any A/B touching bimodal games uses ≥5 seeds or ft09-stratified/excluded primaries. All local gate metrics in actions/tokens, never wall-clock.
- **Re-baseline (Sep 1):** difficulty ratio 0.55 refit from R1's per-game official deltas (ME Q7); if stacked local Δ < +0.4, the milestone objective formally becomes top-100 defense and remaining variant windows convert to selection/redundancy draws.

# Risks

1. **Level-2 wall is a model-capability gap (high).** If forensics returns "mechanic never stated" and the BFS component gates fail, ceiling ≈ attempt scheduler + fork ports (+0.05 to +0.3 official). Accepted; the Nov case then rests entirely on Risk 2.
2. **The compression hypothesis may be false.** Now a *tested* hypothesis (R0 audit + LB diffusion forensics: private ≈55 vs 25 games; the 1.0–1.3 band assembled in 5 days of forking), with a pre-registered failure consequence — not an assumption (RL-M5/PS-m1 resolved).
3. **Seed/draw lottery.** ft09 = 26% of score, sd 9.9; σ̂ CI wide (df≈5 → growing weekly). Mitigated by paired version-matched designs, ft09 stratification, both-endpoint reporting, the sign-flip rule, and the retry budget.
4. **Cutoffs may outrun the S-curve.** Re-baseline clause absorbs; targets restated Sep 1 from the live LB.
5. **Difficulty ratio 0.55 fit on one build family.** Window gate is the promotion authority; ratio refit at R1 from per-game official deltas before it is used in any Sep-1 arithmetic.
6. **Env fragility (proven 5×).** Fork-never-build, byte-matched metadata, preflight, runtime smoke — unchanged and mandatory.

---
*Supersedes `path_forward_v1_2026-07-13.md`. Statistical ethos and process rules of `winning_solution_FINAL.md` carry over. New provenance: null10 counterfactual replay (§E) executed 2026-07-13 against `runs/null10/merged_null_benchmark.json`; disposition of every panel objection in `panel/round5/_author_response.md`.*
