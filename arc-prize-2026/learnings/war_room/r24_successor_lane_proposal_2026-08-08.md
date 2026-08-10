# R24 — Successor lane proposal (prepared 2026-08-08, for panel Sunday 2026-08-09)

**Status: PROPOSAL, unsealed.** Nothing here is a pre-registration until R24 rules on §4 and seals §6.
Parent documents (all read in full for this synthesis):
`learnings/war_room/prime_agent_portability_2026-08-08.md`,
`learnings/war_room/tycho_portability_2026-08-08.md`,
`learnings/sweeps/research_2026-08-08.md`,
`learnings/sweeps/research_2026-08-07.md`,
`learnings/daily_brief_2026-08-07.md` §1/§4.

Provenance convention (inherited from the two portability files): **[V]** verified by direct read this
cycle; **[V-ours]** read from our repo; **[V-2nd]** secondary source only; **[SR]** self-reported by a
vendor/author on a non-official venue; **[INF]** inference, stated by no source.

---

## 0. Decision asked of R24

Select the successor lane to the dead A22 compaction lane, and authorise its **first experiment only**.
The proposal is: adopt **(a) state-externalisation / programmatic world model** as the lane, with **Tycho
(arXiv:2607.28287) as the artifact schema inside it rather than as a competing lane**, fold **(b) additive
typed memory** in as a *component* (Prime Agent's `M`, our ledger as carrier) rather than as a lane, and
keep **(c) banking/replay revival** live as a **variance-efficient complement** on a separate clock because
its root cause is now known and its fix is deterministic. Authorise, this week, exactly two things:
**L0** — a free, offline, zero-push re-verification of our 24 existing `exec_wm/` sims under Tycho's
replay-with-threading + abstention protocol — and **P1** — a single seed-1 build-rail screen of a
persistent sandbox namespace, gated on the adoption canary in §6. Everything else in §3 is explicitly
**not** authorised by this document.

---

## 1. State of play

### 1.1 A22 compaction lane — formal death record

Three independent K3 strikes under sealed preregs, all seed-1 build-rail screens paired against
`runs/kernel_pulls/war_eval_v1/` (ledger-OFF `arc3-duck-war-eval` seed 1: 25 games, **22 level-completions,
3,638 scored actions, 1,686 LLM turns, 1,569,582 generated tokens**) [V-ours].

| Strike | Date | Mechanism | mean Δlc | worst game | Verdict |
|---|---|---|---|---|---|
| v1 | 2026-08-03 | region-agnostic eviction + digest injection + reserve | **−0.200** | — | K3 FAIL → PAUSED |
| v2 | 2026-08-06 | region-aware eviction + digest ON | **−0.320** | — | K3 FAIL → one strike from DEAD |
| v2.1 | 2026-08-07 | **pure eviction, digest-OFF** (the LightMem cell, entered by construction) | **−0.360** (gate −0.128) | ar25 **−2**, sc25 **−2** (cap −1.0) | K3 FAIL → **LANE DEAD** |

The v2.1 run is **VALID, not void**: the arm-defining invariant held on 2,780/2,780 events
(`digest_tokens=0` AND `reserve_applied=0`, RETAIN-OFF clean), every canary passed. This is a decisive
negative [V-ours, `runs/kernel_pulls/a22_v2_1/m1m2m3_screen.json`; memo
`learnings/sweeps/a22_v2_1_seed1_screen_2026-08-07.md`; prereg
`learnings/war_room/a22_compaction_v2_1_prereg_2026-08-06.md` §4 which accepted this consequence at seal].

**Mechanism finding (the part that survives into the successor lane):** harm is **eviction itself**, and it
is **monotonic in eviction pressure** — −0.200 → −0.320 → −0.360 as the injection channel was progressively
removed; pearson(`evicted_chars`, Δlc) = **−0.403** at v2.1 (v2: −0.13). 2W/10L, 13 levels vs baseline 22.
stuck-suppress never fired (0 events); sc25 never recovered in any arm. Theory-consistent with
arXiv:2608.01326 (compaction as one-way communication complexity; **selection/eviction is provably the
strictly weaker class than generation**).

**The M3 confound.** v2.1 measured refuted-hypothesis re-proposal at **−6.49pp (p=0.0001) with the injection
channel provably closed**. Therefore v2's **−4.57pp (p=0.012)** "first lane win" was a compaction side-effect
on thinking-text similarity, **not** the refuted-list digest working. This is the single most expensive
lesson of the lane and it drives §5.1.

**Formal record R24 is asked to confirm:** A22 CLOSED. Revival bar = *generation-side* design plus new
mechanism theory, brought as a fresh prereg. No compaction push of any kind absent that.

### 1.2 Position

- Ledger (frozen-fork control, scored draws): **n=25, mean 0.9384, s 0.1569** [given, orchestrator].
- Our LB best **1.33**, below rank #49. Gold/top-13 cutoff **1.56**, fourth consecutive flat day. Top-5
  prize cutoff **1.61**. Leader KOJIMA **1.86** [V-ours, `runs/lb_daily/`, `runs/lb_ground_truth.md`].
- Artifact-family tail: byte-identical public forks have drawn 1.39 (zoli800) and 1.47 (boristown)
  [V-ours, `runs/lb_ground_truth.md`] — i.e. our own artifact has ≥1.47 in it under a different draw, which
  bounds how much of the 1.33→1.56 gap is *mechanism* versus *draw*.

### 1.3 Why a successor lane is needed now

A22 was the only live mechanism lane. With it dead the queue reverts to frozen-fork filler, which is a
*measurement* activity — it sharpens the control ledger and buys nothing on the board. The gap to gold is
**0.23** against a cutoff flat for four days, and a pure-draw strategy needs the artifact tail to deliver
≥1.56 unassisted when the observed family tail tops out at 1.47. **[INF]** A mechanism change is required;
the binding budget is time, not money.

---

## 2. The evidence landscape

### 2.1 Three-team convergence — and the de-rating that must ride with it

Three independent teams reached near-saturation on the **25 public ARC-AGI-3 games** by moving the
load-bearing state **out of the context window into an executable, verifiable artifact**:

| Team | Date | Model | Reported | Artifact available |
|---|---|---|---|---|
| **Schema** (Impossible Research, `schema-harness.github.io`) | ~2026-07-16 — **earliest** | **Opus 4.8** and Fable 5 | ~**99% RHAE**; 95.35% GPT-5.6 Sol; Claude Code baseline 42.83% [SR] | **50 traces + dependency-free scorer** on HF `schema-harness/arc-agi-3-schema-traces` |
| **Tycho** (arXiv:2607.28287, Lehmann/Aioanei/Vahdati, 07-30) | 2026-07-30 | Opus 4.8 (matched), Opus 5 / GPT-5.6 Sol (frontier) | mean **RHAE 88.49** matched-study best-of-4-policies; **100.00** frontier = the **metric ceiling**, 183/183 levels; **−61% actions vs human baselines** [SR] | **Apache-2.0 code**, `github.com/NIMI-research/Tycho` |
| **Prime Agent** (Prime Intellect, MIT) | 2026-08-05 | Opus 5 | **95.5%** (runs 95.0/95.2/95.5, Best@3 99.97%), 183/183 levels [SR] | MIT TypeScript monorepo, no ARC-specific code |

**The de-rating, to be stated at the top of any panel discussion:** all three numbers live on the ARC Prize
**community leaderboard**, which is *self-reported and harness-driven*. The **official** leaderboard —
vendor-model-only, harnesses excluded — reads **Claude Opus 5 at 30.2%**, GPT-5.6 Sol 7.8%, Opus 4.8 1.5%
[V-ours, `runs/lb_ground_truth.md` / arcprize.org 2026-07-24; sweep `research_2026-08-08.md`]. Prime Agent
additionally reports that *its own* harness-vs-harness comparison ran the other way — "we evaluated Opus 5
and GPT-5.6 Sol with Claude Code and Codex respectively, and found **worse** overall performance relative to
the official results, so we yield to their official reported numbers" [V, primeintellect.ai/blog/prime-agent]
— unexplained, and a direct warning about harness-attributed effect sizes.

**Standing rule proposed for the campaign:** never place 88.49 / 95.5 / 99 and our 1.33 in the same
sentence — different venues, different verification, different scoring. Tycho's "100.00" is the *ceiling* of
`min(Σwℓeℓ/Σwℓ, 100·Σwℓcℓ/Σwℓ)`: full completion pins the second term, so it reads "maxed the benchmark",
not "beat it by 61%" [V, arXiv:2607.28287v1].

**What survives the discount is the convergence, not any number** — and the decisive detail is that Schema
reached the regime with **Opus 4.8**, a previous-generation model, which weakens "this only works with a
frontier-latest model" (while saying nothing about 27B). Two 08-06 arXiv items confirm the architecture from
*outside* ARC: **2608.05891 AppDeltaWorld** (next observation as **executable code deltas**, not images) and
**2608.06257 MASS** (**Logic Engine / Rendering Engine** split — independently Tycho's state-vars +
`render()` decomposition). Both training-gated → design-only.

### 2.2 The eviction-after-externalisation reconciliation

Tycho's own frontier config `configs/paper/opus5_orchestrator.yaml` sets `history: tail_evict`,
`grid_keep: 3`, `image_keep: 3`, `reasoning_keep: 3` [V-repo]. It **evicts context hard** — apparently
contradicting our 08-07 finding that eviction is monotonically harmful.

It is not a contradiction, and the resolution is the framing sentence of this whole proposal:

> **Eviction is not harmful per se. Eviction is harmful when it is not preceded by externalisation.**
> A22 v1/v2/v2.1 deleted episodes that were the *only* copy of the information (harm monotonic in
> `evicted_chars`, ρ = −0.403). Tycho deletes frames that are the *second* copy — the first lives in
> `world_model.py` + `notes/` on the workspace filesystem and is re-derivable by replay. Same operation,
> opposite sign; the discriminator is whether a **verified** external artifact exists first. **[INF]**

Corroboration from the third team: Prime Agent *does* compact, but **generation-side** — `generateSummary()`,
an LLM summary at a cut point that never splits a tool result, chained via `previousSummary`, preserving
`keepRecentTokens: 20000` [V, `core/compaction/compaction.ts`]. Not selection. Exactly the class distinction
arXiv:2608.01326 proves and our −0.200/−0.320/−0.360 measured.

**Ordering constraint:** do not re-open *any* eviction or context-shrinking work until an externalised
artifact is live and verified. Tycho's `tail_evict` is **sequenced, not rejected** — a post-hoc token-budget
optimisation available only after the lane's artifact ships and passes.

### 2.3 The metric fact that changed today

The Kaggle metric **is action-efficiency-weighted**: quadratic penalty on actions, a **hard 5× action cap**,
**linear level weighting** so later levels dominate, and **zero credit for unfinished levels** regardless of
efficiency [V-ours, `docs/community_research_apr1.md` L261–L303].

Three consequences:
1. It **discharges the open homework item** in the Prime Agent file (§6 risk #3) in the favourable
   direction: in-sandbox reasoning costs **zero scored actions**, so a persistent namespace is doubly
   attractive. `actions_per_level_completed` becomes a legitimate co-primary rather than a borrowed metric.
2. It makes Tycho's action-efficiency angle (C6 focused `actions(state)` over the 4,096-cell ACTION6 space;
   the −61% story) **directly score-relevant** rather than an RHAE-only artefact.
3. **Completion still dominates.** "Zero credit for unfinished levels" plus linear level weighting means an
   efficiency gain on levels we do not finish is worth exactly nothing. The primary endpoint therefore stays
   **level-completions**, with actions as co-primary — never the reverse. This is also the Goodhart guard
   (§6.4).

---

## 3. Candidate lanes

### (a) State-externalisation / programmatic world model

**Mechanism.** Move the load-bearing game representation out of the message history into (i) a **persistent
Python namespace** that survives across turns and (ii) a **typed, executable, falsifiable world-model
artifact** (Tycho's `State` dataclass + `init_state`/`transition`/`render`/`outcome`, with `UNKNOWN=-1`
abstention and separately-reported coverage). Actions are function calls inside model-written Python;
planning happens against the artifact, not against the environment.

**Evidence for.** Three-team convergence (§2.1), two out-of-domain architectural confirmations (AppDeltaWorld,
MASS), the eviction-after-externalisation reconciliation (§2.2), the metric fact (§2.3), and — decisive for
lift — the structural survey finding that **the duck is two mechanisms short of the RLM shape, not ten**:
`ToolAgent._tools()` (~L1333) already exposes **exactly one** tool `{"name":"python","parameters":{"code":…}}`,
and `step_env` is already passed *down into the sandbox*, so the model already calls `action([...])` inside
its own Python [V-ours, `duck_eval/taaf_bundle/src/ARC3-Inference/inference/agent/tool_agent.py`]. The board
is already externalised (`agent/prompts.py::STRUCTURED_RUNTIME_STATE_ADDENDUM`: "the raw numeric grid is
intentionally not exposed"). Missing: a **persistent namespace** and **durable cross-level memory**.

**Evidence against.** (i) Every saturating result is at frontier scale; **no weak-model ablation exists in any
of the three** — Tycho's floor backend is Opus 4.8, Prime Agent's Opus 5. Our in-kernel actor is
**Qwen3.6-27B-FP8** on local vLLM, 65 K context [V-ours,
`.../inference/framework/kaggle.py`]. (ii) **arXiv:2608.04828 (Skill-Use)** documents that agents frequently
fail to *trigger* procedural affordances at all — a 27B model may never use a namespace it is told about;
this is the most likely null mode. (iii) Tycho's **H2**: the `trigger` policy hit **88.1% accepted transition
match vs orchestrator's 16.2%** and still *lost* on completions — fidelity is necessary, not sufficient, and
is anti-correlated with spend. (iv) Nothing is literally portable: Prime Agent is a TypeScript monorepo
needing npm + an authenticated provider (barred by `enable_internet: false` + `--no-index`); Tycho's runner
would violate fork-never-build, which already cost 5 ERROR submissions (v62–v66).

**Engineering lift.** Unit = one dataset-code push cycle (edit `duck_eval/warpack/_kaggle_dataset/*` →
`datasets version` push → pull-back byte audit → kernel build). **P1 persistent namespace: M (1–2 cycles)** —
the subprocess + JSON line protocol already exists in `python_tool_sandbox.py::run_sandboxed_python` (~L448);
the work is lifecycle (keep the child alive per `_HarnessGameSession`, reap on game end) plus **`RLIMIT_CPU`
re-accounting: a limit that was per-call becomes per-game, and it will kill a long-lived child silently**.
All via warpack-style monkeypatching → zero notebook drift. **L0 sim re-verification: M effort, 0 pushes,
$0** (offline, on `runs/ewm_dryrun` streams + `exec_wm/observations/`). **L1 sim interface migration: L**,
and it is the item whose legitimacy depends on the §5.3 ruling.

**Kill condition.** `namespace_reuse_rate < 0.15` on the P1 screen ⇒ lane NULL on the substrate question,
answered *without needing the score at all*. L0 carrier set still ~4 games after the protocol fix ⇒ exec-wm
closes permanently and the artifact half reduces to schema-only.

---

### (b) Additive typed memory

**Mechanism.** An append-only, typed, trust-gated store outside the window, re-injected as an *additional*
prompt block; nothing is removed from context. Prime Agent's `M`; our existing carrier is the Hypothesis
Ledger (`duck_eval/ledger/ledger_core.py`, per-game, `DIGEST_TOKEN_CAP = 600`) [V-ours].

**Evidence for.** A concrete, known duck defect it repairs: `_summarized_knowledge` (the world-model digest:
`world_model, goal_model, action_model, recent_findings, open_questions, current_plan, cross_level_notes`) is
**wiped on level transition and game over** [V-ours] — under a metric where *later levels dominate linearly*,
we delete the model exactly when it becomes most valuable. Prior local anchor: Reki typed-causal-memory
+0.098. Literature: **2608.04530 FocusMem** (frozen policy + trained memory + **trust gate** — the only shape
compatible with a frozen duck), **2608.05784 Activity Frames** (deterministic typed frames with
evidence-pointers back to raw rows; 86× compression, 98.4% query accuracy vs 66–80% for LLM summaries, at
zero token cost).

**Evidence against.** FocusMem's memory is **trained** → free-Kaggle-GPU-build item at best, not an
inference-time patch. **Generic memory has no notion of being wrong** — externalisation without a falsifier
is exactly the round-18 [MAJOR] filed against our own EWM line ("no world-model fidelity metric —
unfalsifiable by construction"). 2608.06144 FinEvo-Bench is the only longitudinal support, and domain-bound.

**Engineering lift: S (≈1 cycle)** for the un-wipe — stop the wipe in
`_update_summarized_knowledge_from_step_summary`, route survivors into the ledger store, re-inject as an
additional block.

**Minimal first experiment.** Not a lane: run as **arm P3, a separate arm inside lane (a)**, after P1.
Separate because the M3 confound already cost one lane's worth of attribution; bundling P1+P3 in one push
would reproduce that error exactly. **Kill condition:** mean Δlc harm at the §6 thresholds with the un-wipe
as the only change ⇒ component dead, lane proceeds on the artifact half alone.

---

### (c) Banking / replay revival

**Mechanism.** Re-fire the warpack banking path (`duck_eval/warpack/warpack_patch.py`, `WarpackState`,
`trace: list[TraceStep]`, `banked`, `bank_max_replay_actions=1500`) so that a *known* solution prefix is
replayed into a fresh play at zero exploration cost, concentrating the remaining scored actions on
unsolved levels.

**Evidence for.** The **07-15 blocker is root-caused and the fix is deterministic**. The cause is the
**N5 `prune_trace` mechanism**: no-op actions *still advance the hidden phase counter*, so dropping leading
no-ops desyncs the phase and the first replayed action lands on a different frame — step-0
`frame_divergence` aborts on sc25/m0r0 (sc25 is mod-5 aliased; m0r0 is the worst unresolved game, det 0.618)
[V-ours, `learnings/panel/r16_circulation.md` ~L1250]. **The fix is to preserve phase alignment: full
unpruned replay from RESET, zero pruning.** The same audit records that *full unpruned replay survives on
all 25 games*, and gives the prefix-splice-safe set explicitly: **ar25, bp35, ft09, lf52, lp85, ls20, r11l,
sp80, su15, tn36, tu93**; everything else is FULL-REPLAY-ONLY from RESET. Validation already on disk:
**fired + score-invariant on ar25 & s5i5** in `bank_fire_validation.json` [V-ours,
`learnings/daily_brief_2026-07-16.md` ~L132]. Variance argument: **borro1980's map — 2 games carry ~65% of
ledger variance** — so banking the binary clears is the *variance-efficient* target, and with s=0.1569 on
n=25 that is the cheapest available route to a distinguishable draw. Reset accounting is not a hidden tax:
Tycho's `scoring.py` confirms a RESET that *creates a play* is unscored while a mid-level protocol RESET
costs one in-play action [V-repo] — consistent with warpack's existing design (open a new play).
Enabling mechanism from today's sweep: **2608.05628 SkillHEX** — inference-only, converts failure hypotheses
into **executable tests** supplying dense reward **without extra environment attempts** (55.9–57.9% on 87
SkillsBench tasks in 5 iterations). Under a quadratic action penalty with a hard 5× cap, "dense signal at
zero scored actions" is exactly the currency we lack.

**Evidence against.** Field 3 is **near-empty across two consecutive sweeps** — one enabling paper, no field
behind it. Banking is by construction a *public-set* optimisation: it re-plays what we already solved and
contributes nothing on unseen private games, colliding with the standing generalisation rail. And it touches
the scored-action path directly, so a bug here is a scored-draw bug, not a build-rail bug.

**Engineering lift: S–M, 1 cycle.** Prune-disable is a one-flag change; the safe-game table and the
fire-validation harness both already exist.

**Minimal first experiment — free, zero-push.** Re-run `bank_fire_validation.json` **offline** with pruning
disabled (full replay from RESET) across recorded traces for all 25 games; report per game fired / aborted /
abort-step histogram / score-invariance. Zero pushes, zero dollars, and it directly tests the root-cause
claim. SkillHEX enters as a *design note* only: failure hypotheses become executable tests over recorded
traces, never live attempts. **Kill condition:** step-0 `frame_divergence` persisting on the
prefix-splice-safe set with pruning disabled ⇒ the 07-15 root cause is misidentified and (c) closes.

---

## 4. Recommendation and sequencing

**Select lane (a). Fold Tycho into it as the artifact schema. Fold (b) in as a component arm. Keep (c) as a
parallel, variance-efficient complement on its own clock, entered only via its free offline test.**

All three input files independently reach this shape. Prime Agent file: *"the blueprint that should be
adopted is the convergent one, not Prime Agent's headline number."* Tycho file: *"adopt Tycho as the
specification for the successor lane's first artifact type, not as a lane of its own — and spend zero kernel
pushes on it this week."* Sweep: *"(a) programmatic world models / state-externalization > (c) banking-replay
via SkillHEX-style executable failure tests > (b) additive typed memory."* The agreement is not accidental:
Prime Agent supplies the **substrate** (state outside the active context, one Python tool, actions as
function calls); Tycho supplies the **typed falsifiable artifact** to put in it — which is precisely what
answers the round-18 [MAJOR] unfalsifiability charge against our own EWM line. **One lane, two papers.**

### What is sequenced

| # | Step | Cost | Gate to proceed |
|---|---|---|---|
| **S1** | **L0** — offline re-verification of the 24 existing sims under Tycho's protocol: replay from level frame 0 **with state threading**, report **accepted transition match AND coverage** per game, on-trajectory | **$0, 0 pushes**, this week, workstation only | Carrier set must **expand beyond ~4 games**. If not: bank a clean second negative, close exec-wm permanently, retain C1/C2/C3 as schema only |
| **S1b** | **(c) offline bank re-fire** with `prune_trace` disabled across all 25 recorded traces | **$0, 0 pushes**, this week | step-0 `frame_divergence` must clear on the 11-game prefix-splice-safe set |
| **S2** | **P1 persistent-namespace screen**, seed 1, build rail | **1 push cycle**, $0 | §6 gates. `namespace_reuse_rate ≥ 0.15` **and** non-harm |
| **S3** | **Decision point** — R25 or a sealed weekday build-rail disposition | — | P1 NULL ⇒ reallocate to (b)-as-lane or (c). P1 PASS ⇒ S4 |
| **S4** | **P3 durable cross-level memory** (the un-wipe), **separate arm** | 1 push cycle | non-harm; attribution kept clean, no bundling |
| **S5** | Staged **L1+L2+L3+L4** artifact arm (migrated sims, advisory one-action-at-a-time, typed-frame tagger, budgeted planner + consult gate) | ≥1 cycle, own prereg | only if S1 expanded the carrier set **and** §5.3 rules L1 in-bounds |

**Explicitly NOT authorised by this document:** Prime Agent's `/refine` self-editing harness (P7 — makes runs
non-reproducible, and the Factorio reward-hacking anecdote is a live warning on a scored benchmark);
`rlm()` sub-agents (P8 — contend for the same GPU that already serves 16 concurrent games); Tycho's in-kernel
**builder subagent** (C8/L5 — Tycho budgets **3,500 LM calls per game** against our **1,686 turns total,
≈67/game: a 52× gap**, and there is no weak-model ablation anywhere); and any `tail_evict`-style context
policy (C14 — sequenced behind a live, verified artifact, per §2.2).

---

## 5. Auxiliary R24 items

### 5.1 Refuted-list micro-arm — position: **DROP as a standalone arm; do not re-argue yet**

The M3 instrument is confounded, and we know this because v2.1 moved it by **−6.49pp at p=0.0001 with the
injection channel provably closed on 2,780/2,780 events**. A metric that moves that hard when the mechanism
is absent is not measuring the mechanism. Re-arguing the micro-arm would first require a *new* re-proposal
instrument invariant to compaction-induced thinking-text similarity drift — a measurement project, not a
lane, competing for the same push budget as (a). **Recommended disposition:** drop the arm; keep the
*question* on the shelf; entry condition for revival is a validated instrument demonstrated on the existing
v2/v2.1 transcripts (free, workstation-only, any time). Meanwhile fold the refuted-hypothesis question into
lane (a) as a **canary** on the P3 memory arm — does durable cross-level memory reduce re-proposal? — where
it costs nothing extra.

### 5.2 Compaction lane — formal record

R24 to confirm §1.1 as the formal death record: **A22 CLOSED, three K3 strikes, v2.1 VALID-not-void.**
Carried forward: the M3 confound, the borro1980 variance map, the "additive-only memory" datum, and the
eviction-after-externalisation ordering constraint (§2.2). Revival bar: generation-side design + new
mechanism theory + fresh prereg. Note for the record that the field is quiet — **zero** new compaction-theory
work, zero rebuttals to 2608.01326, and the one context-control harness surfaced (Arcgentica: subagents
return compressed textual summaries) is in the *selection/summarisation* class that 2608.01326 proves weaker
and that A22 empirically falsified. Log Arcgentica as a **negative exemplar**.

### 5.3 Governance rulings needed

1. **Does workstation LLM regeneration count against the zero-budget rail?** Flagged in the Tycho file §6.7.
   The rail as written (memory `feedback_arc_zero_budget`, 2026-07-07) bans **cloud GPU/LLM eval spend**;
   local agent-assisted code authoring has been treated as in-bounds by practice but never ruled on. This is
   load-bearing: **L1 is the largest item in the lane** (the June generation pass cost ~10 h / ~7 M tokens for
   24 games) and its legitimacy determines whether the artifact half is affordable at all.
   **Proposed ruling:** in-bounds when it (i) incurs no metered API spend, (ii) produces artifacts committed
   to the repo and byte-audited before they ride in the Kaggle dataset, and (iii) is disclosed in the prereg
   as authoring provenance. Out-of-bounds the moment it requires paid API calls.
2. **Provenance de-rating of community numbers.** Adopt §2.1 as a standing campaign rule: community-leaderboard
   figures are marked **[SR]**, always carry the official-LB counterweight (Opus 5 = 30.2%), and are never
   quoted in the same sentence as a Kaggle score. Corollary the panel should state explicitly: *"public
   ARC-AGI-3 is saturated"* is a claim about **harness design maturity on a self-reported venue**, not about
   our private-set position — do not let it be read as "we have already lost."
3. **Sandbox risk class.** Tycho ships `TYCHO_SANDBOX_RUNTIME=host` (Kaggle cannot nest Docker) and its own
   repo labels host mode "only for trusted local development". Acceptable **only** while executed code is
   ours, generated offline and byte-audited. If C8/L5 ever lands, code becomes model-authored in-kernel and
   this verdict must be re-taken. Ask the panel to record that trigger now.

### 5.4 No-regret actions (do regardless of lane choice)

- **Attach the Schema HF trace set + dependency-free scorer as a Kaggle dataset**
  (`schema-harness/arc-agi-3-schema-traces`, 50 trajectories) — the only artifact in two sweeps our
  offline/zero-budget constraints do not immediately reject: downloadable on the workstation, attachable to
  the kernel, usable to mine transition-model patterns and cross-check any local scorer. Verify licence first.
- **Run L0 and the offline bank re-fire** (S1/S1b) — both free, both genuine falsifiers.
- **Measure the duck's skill/affordance-trigger compliance before building any store** (the 2608.04828
  pre-mortem) — a transcript-forensics job on existing runs that directly de-risks P1's most likely null mode.
- **Add `actions_per_level_completed` to the standing screen metric set** (baseline **3,638/22 = 165.4**).
- **Record AppDeltaWorld delta-coding and the MASS logic/render split** as design inputs; neither is
  buildable (both training-gated).

---

## 6. Pre-registered gates — sketch for the first screen (to be sealed at R24)

Scope: **arm P1, "persistent scratchpad", seed-1, free Kaggle build rail, NEVER submitted.**
Patch surface: `python_tool_sandbox.run_sandboxed_python` only, via `canivel/arc-war-kit`, no notebook
change (cell-12 hook untouched). One sentence added to the system prompt: *variables you define persist
across turns; use them to hold your world model.* Nothing is removed from context.

### 6.1 Arm-defining invariant (the `digest_tokens=0` analogue)

`evicted_chars` and the trimmed-message sequence emitted by `_trim_messages_for_context()` must be
**byte-identical to baseline on every event**. If context handling moves at all, the run is **VOID for §0** —
P1 adds a store outside the window and must not touch the window. Additivity is **proved, not asserted**;
that is what A22 bought us.

### 6.2 Endpoints

- **Primary (non-harm):** paired mean Δ level-completions vs `runs/kernel_pulls/war_eval_v1/` (22 lc), gate
  **mean Δlc ≥ −0.128 AND worst-game Δlc ≥ −1.0** — inherited verbatim from the A22 M1/sentinel A5/A8
  thresholds, so the bar is *unchanged* and cannot be accused of being retuned to fit.
- **Co-primary (newly legitimate, per §2.3):** `actions_per_level_completed`, baseline **165.4**. Reported,
  not gating, at first screen — it cannot gate while completions are the dominant term.
- **Adoption canary (the decisive cheap read):** `namespace_reuse_rate` = fraction of turns that reference a
  name defined in a *previous* turn. **Pre-registered floor 0.15.** Below floor ⇒ the result is
  "27B does not use the substrate", **not** "state-externalisation does not work" — the lane is answered NULL
  immediately, without the score.
- **Safety canaries:** live-child count ≤ concurrency (16); zero orphans at teardown; zero per-game
  `RLIMIT_CPU` kills; automatic fallback to ephemeral-per-call on any child fault, with a fallback count.
  Any fallback count > 0 must be reported and, above a pre-registered fraction of turns, voids the arm.
- **Pinned free parameter:** the tool-output cap (`LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS`, currently 1024) must
  be **pre-registered at its value** before the run. It is the one knob P1 tempts us to move post hoc.

### 6.3 Kill rules (K-series, inherited shape)

- **K1** — 0 persistent-namespace events / banner absent ⇒ mechanism never engaged ⇒ run VOID as evidence
  either way; no further pushes of this arm until root-caused.
- **K2** — `PATCH FAILED` vanilla-fallback banner ⇒ run VOID (counts against neither side).
- **K3** — non-harm FAIL at seed 1 ⇒ arm **PAUSED**; a second independent FAIL at seed 2 ⇒ arm **DEAD**
  (sentinel A5/A8 standard, as applied to A22). K3 clock starts fresh at P1 seed 1.
- **K4** — `namespace_reuse_rate < 0.15` ⇒ lane **NULL on the substrate question**; reallocate per §4 S3.
- **K5** — no scored draw is requested by this screen. Any future scored draw inherits A21/C2 verbatim:
  harm-pause on a draw below the paired trailing-4 rule, no inference from n=1.

### 6.4 Anticipated objections, and where each is answered

| Objection (precedent) | Answer in this design |
|---|---|
| **Unfalsifiable by construction** (round 18, filed against EWM) | ~~Two independent pre-registered falsifiers that do not need the score: `namespace_reuse_rate < 0.15` (K4) and L0's carrier-set-expansion test.~~ **CORRECTED 2026-08-10 (R24 §5.2 i seal, `duck_eval/r24_prep/s1_sealed_spec_2026-08-10.md`): L0 is NOT a falsifier and is withdrawn from this row.** 0 of 25 sims implement abstention, `coverage_strict` is measured at exactly 1.0 (0 sim errors / 0 selfdiffs over 4,996 banked steps, `runs/ewm_dryrun/raw.json`), and the identity-abstention proxy is computed from the observed label, so it is circular. Only K4 remains here, and §3.4 of the R24 minutes further shows K4 can pass validly but cannot fail validly. Tycho's abstention+coverage channel remains the *reason to build* L1; it is not a channel our current sims possess |
| **Goodhart on a single metric** (R11) | Completions stay primary; `actions_per_level_completed` is co-primary and non-gating at first screen; fidelity (`accepted_match`, `coverage`) is a **mechanism canary only**. Tycho's own H2 is the empirical warning we are pre-committing against: `trigger` hit 88.1% accepted match vs orchestrator's 16.2% and **still lost**. We already made this mistake once — **CORRECTED 2026-08-10:** the figure was **22 of 24 games (91.7%) classified Class A**, i.e. a *class share*, not a `state_exact` rate, and it was **never held out** — `exec_wm/validate_sim.py` ran at `split=all` over the same 200 tuples the authoring model studied, scored for the *selected* v1/v2 winner (`exec_wm/scale_summary.md:3-4,54,74`). Per-game in-sample `state_exact` runs 23.0 (r11l) to 100.0 (9 games), mean 81.1. It was near-uninformative on-trajectory (sp80 100.0 → 0.026–0.879; lp85 100.0 → 0.087–0.458; sb26 100.0 → 0.106–0.162) |
| **Confound accounting** (M3) | One change per arm, enforced: P1 and P3 ship as **separate arms in separate pushes** even though they could share a cycle. The arm-defining invariant (§6.1) is the structural guarantee that P1's effect cannot be a context-handling side-effect |
| **Effect size vs draw noise** | A single seed-1 build-rail screen reads **mechanism, not lift** — stated up front. borro1980's map (2 games ≈ 65% of variance) means no single screen can settle variance. Any promotion to a scored draw uses the ledger, not the screen |
| **Cherry-picked thresholds** | Every gate in §6.2/§6.3 is inherited verbatim from the A22 preregs; nothing was retuned after seeing A22's outcome |

### 6.5 Scored-draw arithmetic (for reference only — no draw requested here)

Should any lane-(a) or lane-(c) artifact later earn a scored window, the gate arithmetic against the current
control ledger (**n=25, mean 0.9384, s 0.1569**) is, in the boristown mean-of-4 form and one-sided α=0.05:

  threshold = 0.9384 + t(0.95, df=27) × 0.1569 × √(1/4 + 1/25) = 0.9384 + 1.7033 × 0.08449 = **1.0823**

i.e. a promotable arm must average **≥ 1.0823 over 4 gated draws**. Recompute at the then-current n at seal
time; this line is illustrative arithmetic, not a sealed gate. Note the frozen-fork family tail (1.39, 1.47
on byte-identical public forks) sits well above this, which is exactly why a *paired* build-rail screen —
not a draw — must carry the mechanism decision.

### 6.6 Seeds

Build-rail screens: **seed 1 first** (cheapest decisive read), **seed 2 required for promotion** of any arm
out of PAUSED. **≥3 seeds** required for anything proposed for a scored draw. Budget: max **2 kernel pushes
per day**, free build rail only, **$0 cloud**, dataset version push before kernel push, pull-back byte audit,
runtime banner check, and a runtime smoke test at 100% PASS before any push — all unchanged.

---

## 7. For the minutes

*Compaction is dead and its mechanism is understood (eviction without prior externalisation); three
independent teams converged on externalising state into verified executable artifacts; the duck is two
mechanisms short of that shape; the metric demonstrably rewards it. Therefore: lane (a), Tycho as its
artifact schema, memory as a component, banking as the variance-efficient complement — and this week spend
zero pushes on two free offline falsifiers before spending the one push that matters.*
