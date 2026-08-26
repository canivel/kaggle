# Tycho port re-scope — discharging R25 rl-planning N2 (defanged-test FATAL)

**Date:** 2026-08-10 · **Cost:** $0, zero pushes. Analysis of existing artifacts only.
**Companion:** `learnings/sweeps/rail_regime_gap_2026-08-10.md` (R25 systems FATAL) — the two are
coupled: the port's price is denominated in the currency that memo establishes.

> R25 rl-planning N2 [FATAL]: *"the in-scope port subset EXCLUDES the components carrying Tycho's
> measured lift → lane (a) as scoped does NOT test the architecture hypothesis (defanged test).
> Re-scope to include the lift-bearing components or admit it's not a test of the thesis."*

**Finding: N2 is CORRECT, and the re-scope it asks for is unaffordable by a factor of 3.5× on wall
clock and 52× on LM calls. Recommendation: Option 2 — admit in writing that the port is not a test of
the architecture thesis, restate what it is a test of, and buy back the thesis question with one
narrowly-scoped, single-build authoring-feasibility probe, which is the only affordable fragment of
the lift-bearing component.**

---

## 1. Where Tycho's measured lift actually lives

Everything in this section is from `learnings/war_room/tycho_portability_2026-08-08.md`, which read
the paper (S1/S2), the source tree (S3–S9) and the frontier config (S8) directly.

### 1.1 The paper's three ablation axes, and their deltas

| axis | comparison | measured delta | portability-doc line |
|---|---|---|---|
| **A. world model present vs absent** | `orchestrator` 88.49 vs `no_world_model` 79.07 RHAE | **+9.42 RHAE** | §1a |
| **B. who authors/repairs it, and when** | `orchestrator` 88.49 · `single` 85.36 · `trigger` 83.07 | **+5.42** best-vs-worst-WM | §1a, §2.5 |
| **C. backend scale** | matched Opus 4.8 → 88.49; frontier Opus 5 / GPT-5.6 Sol → 100.00 RHAE, all 183 levels | **+11.51 to the ceiling** | §1a |

**The decisive structural fact: all three WM-bearing arms in axis A require an LLM to write and
repair `world_model.py` in-loop.** `single` = the actor edits it; `orchestrator` = a builder subagent
on actor request; `trigger` = the builder auto-fires on verification failure (§2.5). There is **no arm
anywhere in the paper in which the world model is pre-supplied and only deterministic scaffolding
runs.** Consequently:

- **Axis A's +9.42 is jointly attributable to "a world model exists" AND "an LLM maintains it".** The
  paper cannot separate them and neither can we.
- **Axis B's +5.42 is 100% attributable to the metareasoning/authoring policy** — i.e. to C8 and to
  C9-as-LLM-metareasoning. Nothing else varies between those three rows.
- **Axis C is 100% C12** (frontier backends).

### 1.2 What has *zero* measured lift in the paper

C1 (the 4-function contract), C2 (`UNKNOWN=-1` abstention + coverage), C3 (replay verification),
C4 (advisory one-action commitment + bypass), C5 (typed history), C6 (focused `actions(state)`),
C7 (planner). **None of these is ablated.** They are the substrate that all four policy rows share
(or, for C6, a template instruction present throughout). Their contribution is **unattributed, not
zero** — but it is also **unmeasured**, which is precisely R25's point.

Two of them are additionally warned against by the paper itself:

- **H2 (§2.6):** `trigger` reached **88.1% accepted transition match** against `orchestrator`'s
  **16.2%** and still lost on completions and RHAE. Fidelity — the thing C2+C3 measure — is
  *anti-correlated* with the outcome. Optimising the in-scope components' own metric is measured to be
  the wrong move.
- **C4's bypass** is the *fallback to `no_world_model`*, i.e. it degrades toward the **79.07 floor**.
  It is a safety rail, not a lift source.

---

## 2. In / out table with lift attribution

Scope as currently written: portability §3 (PORT/ADAPT/SKIP) and the proposal's S1–S5 sequence
(`learnings/war_room/r24_successor_lane_proposal_2026-08-08.md` §4), which explicitly lists C8/L5 and
C12 under *"Explicitly NOT authorised by this document"*.

| # | Tycho component | our scope | carries measured lift? | note |
|---|---|---|---|---|
| C1 | `State` dataclass + `init_state`/`transition`/`render`/`outcome` | **IN** (S5/L1) | **no — never ablated** | necessary substrate for every arm; also the diagnosis of our stateless `simulate()` sims |
| C2 | `UNKNOWN=-1` abstention + coverage | **IN** (S1/L0) | **no — and see H2** | and **degenerate on our assets**: 0/25 sims implement abstention, coverage ≡ 1.0 (R24 §3.5) |
| C3 | replay verification threaded from frame 0 | **IN** (S1/L0) | **no — and see H2** | its Tycho consumer is the **builder's counterexample channel** (C8, OUT); with C8 out it feeds only a bypass gate. Only **3** of our sims hold hidden state, so "threading" threads almost nothing (R24 §3.5) |
| C4 | advisory one-action-at-a-time + bypass | **IN** (L2) | **no** | bypass = the 79.07 floor, i.e. the *downside protection*, not the lift |
| C5 | typed decision/animation/terminal frames | **IN** (L3) | **no** | feeds model *induction*, whose consumer is C8 |
| C6 | focused `actions(state)` for the 4096-cell ACTION6 space | **IN** (L3) | **no, but mechanism-independent** | the only in-scope item with a lift story that does not route through C8; the −61%-actions angle. Still unablated |
| C7 | A*/BFS/subgoals planner | **IN, ADAPT** (L4) | **no** | must be wall-time budgeted (portability §6.4) |
| C9 | **active-abstraction metareasoning** — actor decides *when* the model pays | **IN but REDUCED** to a deterministic pre-registered lookup gate | **YES — this is axis B** | our reduction fires on a fixed fidelity table, i.e. it is structurally **`trigger`-shaped (83.07)**, and `orchestrator` (88.49) is defined by being *actor-requested*. **We port the losing policy's decision rule.** |
| C8 | **builder subagent authoring/repairing `world_model.py` in-kernel** | **OUT** (L5, "not a first push"; explicitly not authorised) | **YES — axis A ∧ axis B** | the single component present in every lift-bearing row |
| C10 | container sandbox | ADAPT (host mode) | no | revisit if C8 ever lands |
| C11 | Tycho runner | OUT | no | violates fork-never-build; 5 prior ERRORs |
| C12 | **frontier backends (Opus 5 / GPT-5.6 Sol; floor Opus 4.8)** | **OUT** | **YES — axis C** | `enable_internet:false` + zero budget; ≈$3.0k to reproduce one run |
| C13 | viewer | OUT | no | |
| C14 | `tail_evict` context policy | OUT (sequenced) | no | safe only *after* externalisation |
| — | **incremental repair from replay counterexamples** (§2.6 fallback ladder rung 1) | **OUT by consequence** | **YES (part of axis A/B)** | requires C8; our L1 models are workstation-authored and **frozen at deploy**, so they cannot be repaired in-game at all |

**Score: 3 of 3 lift-bearing axes are OUT of scope (C8, C9-as-metareasoning, C12), plus the repair
ladder. 7 of 7 in-scope components have zero measured lift attribution in the source.** R25's N2 is
upheld in full.

**Aggravating factor R25 did not have:** R24 §3.5 already showed that the two in-scope components that
were supposed to *replace* the missing lift as a falsifier — C2 coverage and C3 threading — are
**degenerate on our actual assets** (coverage ≡ 1.0 on 0/25 abstaining sims; 3/25 sims hold hidden
state). So the current scope is not merely "the lift-free subset"; it is the lift-free subset with its
two headline instruments reading constants.

---

## 3. Option 1 — re-scope to include the lift-bearing components. Priced.

The only lift-bearing component that is even conceivably reachable is **C8** (C12 is barred by
`enable_internet:false` and zero budget; C9's metareasoning is C8 with a different trigger). Price it
in our own measured currency.

**Our per-game budget** (`runs/kernel_pulls/a22_v2_1/m1m2m3_screen.json` `M2`, war-eval seed-1, and
`scripts/rail_regime_gap.py`):

| | value |
|---|---|
| LLM turns per game | **67.4** (1,686 / 25) |
| scored actions per game | **145.5** (3,638 / 25) |
| wall per game | **7,920 s** (hard guillotine, hit by 99.3% of game-runs) |
| **s per LLM turn** | **117.4** |
| s per scored action | **54.4** |
| generated tokens per turn | **931** |

**Tycho's budget** (portability §2.7, §6.2): **3,500 LM calls/game**, **40 tool steps/turn**,
**24,000 answer tokens/call**; observed orchestrator cadence **147 builder calls / 25 games = 5.9
per game**.

### 3.1 The gaps, computed

| gap | ours | Tycho | ratio |
|---|---|---|---|
| LM calls per game | 67.4 | 3,500 | **51.9×** |
| answer tokens per call | 931 | 24,000 | **25.8×** |

### 3.2 What a builder call costs us

Because the 7,920 s guillotine is hard, a builder call does not lengthen the game — it **displaces
scored actions**. At 117.4 s/turn and 54.4 s/action, **one extra LLM turn ≈ 2.16 scored actions**.

| C8 variant | turns/game | wall cost | as % of the 7,920 s game budget | actions displaced (of 145.5) |
|---|---|---|---|---|
| **Tycho as specified** — 5.9 calls × 40 tool steps | 236 | 27,706 s | **349%** — 3.5× the entire budget | n/a, does not fit |
| **one builder call, Tycho-sized** — 1 × 40 steps | 40 | 4,698 s | **59%** | 86.5 (59%) |
| truncated 8× — 5.9 calls × 5 turns | 29.5 | 3,463 s | **44%** | 63.7 (44%) |
| truncated 40× — 5.9 calls × 1 turn | 5.9 | 693 s | **8.8%** | 12.7 (8.8%) |
| the absolute floor — 1 call × 1 turn | 1 | 117 s | **1.5%** | 2.16 |

Then apply **RD-1** from the rail memo (scored-rail concurrency 28 vs build-rail 25, saturated
engine): multiply every wall figure by **1.12–1.25**. The 44%-variant becomes **49–55%** of the
scored-rail game budget.

### 3.3 Verdict on Option 1

**Unaffordable, and the affordable remnant is not the thing that was measured.**

- Tycho as specified needs **3.5× our entire per-game wall budget** for the builder alone, before the
  actor plays a single action. There is no version of "fit it in" here.
- The largest variant that leaves the actor a majority of its actions is **~5 turns/call at Tycho's
  cadence = 44% action displacement**. Against a baseline of 22 level-completions in 3,638 actions
  (war-eval seed-1), that is a self-inflicted −44% action budget which the +9.42-RHAE evidence does
  **not** underwrite, because that evidence was produced with a **52×** larger call allowance where
  builder calls cost the actor nothing it needed.
- The only cheap variant (1 turn/call, 8.8%) is a **40× truncation of the builder episode**. A builder
  that gets one turn and 931 tokens is not the builder that produced 88.49; there is no evidence such
  a thing works, and the paper's own H2 result warns that under-resourced builder policies
  (`trigger`) *lose* despite better fidelity.
- **And the model-scale question is untouched.** The paper's floor backend is Opus 4.8; our in-kernel
  actor is **Qwen3.6-27B-FP8** on local vLLM at 65 K context. **No weak-model ablation exists in Tycho
  or in either sibling blueprint** (portability §6.1; proposal §3(a) "Evidence against" (i)). Writing a
  correct Moore machine with hidden state is program synthesis. Whether a 27B can do it is
  **NOT ESTIMABLE** from any artifact we hold.
- **Build-budget cost, if we did it anyway.** L5 is sized at **≥2 push cycles with its own prereg**
  (portability §5). Under K3′ every arm now also needs **m ≥ 3 same-config baseline runs**, and a
  **warpack-specific null** is separately owed because `null10` understates warpack variance by 4.83×
  (R24 §5.1). That is **≥5–6 builds ≈ 11–13 GPU-h**, i.e. **about half of one week's entire 30 GPU-h
  allowance**, spent on the component with the largest unpriced model-scale risk in the campaign.

---

## 4. Option 2 — admit it, and restate what the port actually tests

### 4.1 The admission, in the words that should go into the prereg

> **Lane (a)'s S5 artifact arm is not a test of the "active abstraction / programmatic world model"
> architecture thesis.** Every component to which Tycho's measured lift can be attributed — the
> in-kernel builder subagent (C8), actor-requested metareasoning (C9), the incremental-repair ladder,
> and the frontier backend (C12) — is out of scope, and no component that *is* in scope was ablated by
> the source. The arm therefore cannot confirm or refute the thesis, and a null result from it is
> **not** evidence against programmatic world models.

### 4.2 What it IS a test of — the honest hypothesis

> **H(scaffolding):** *Given a frozen, offline-authored, per-game programmatic state model, does
> deterministic scaffolding — typed `State` with hidden variables (C1), an abstention channel and
> coverage reporting (C2), replay verification threaded from level frame 0 (C3), advisory
> one-action-at-a-time commitment with silent bypass (C4), typed decision/animation/terminal frames
> (C5), a focused ACTION6 candidate set (C6), and a wall-time-budgeted planner behind a
> pre-registered consult gate (C7+C9-reduced) — change level-completions and actions-per-completion
> for a **Qwen3.6-27B-FP8** actor on the **25 public games**, under a hard 7,920 s/game wall-clock
> guillotine at 25–28 concurrent children?*

That is a real, worthwhile, falsifiable question. It is **the deterministic-scaffolding question**,
not the architecture question. It is also the question our own five-way exec-wm failure diagnosis
(portability §7) actually poses.

### 4.3 Three caveats that must ride with H(scaffolding)

1. **C6 is the only in-scope component with an independent lift story.** It prunes a 4,096-cell click
   space and does not route through model quality. If the arm reads positive, attribution work must
   start with C6, and the arm should be built so C6 can be isolated (env-flag).
2. **Do not optimise fidelity.** H2 measured accepted-match at 88.1% *losing* to 16.2%. Coverage and
   accepted-match are **mechanism canaries only**; primary stays level-completions with
   `actions_per_level_completed` (baseline **165.4** = 3,638/22) as the efficiency read — noting R24
   §3.1's correction that this metric is not a legitimate *co-primary* on a wall-clock-bound rail.
3. **Generalisation rail — this is where H(scaffolding) collides with the rail memo.** Twenty-four
   hand-migrated sims are by construction overfit to the **25 public games** (portability §6.9). The
   scored rail is **110 games** (`learnings/gap_forensics_2026-07-09.md:29`). Even a clean PASS
   transfers to at most **25/110 = 23%** of the scored set, and only if the public games are a subset
   of the official set — which is **NOT ESTIMABLE** from repo data. Under RD-4 of the rail memo, the
   arm must report itself as a public-25 result. **The generalising asset is the schema plus the
   verifier, not the 24 sims** — and the schema's value is exactly what a frozen-model arm cannot
   demonstrate.

---

## 5. Recommendation

**Take Option 2, and buy back the architecture question with one cheap probe rather than a re-scope.**

### 5.1 Sequence

| # | item | cost | why |
|---|---|---|---|
| **R1** | Write §4.1's admission into the S5 prereg verbatim, and replace the arm's stated hypothesis with **H(scaffolding)** (§4.2) plus the three caveats (§4.3). Amend proposal §4's S5 row so it no longer implies an architecture test. | **$0, 0 pushes** | discharges N2 honestly and immediately |
| **R2** | Add to the prereg the explicit non-inference clause: *a null on S5 is not evidence against programmatic world models; it is evidence against deterministic scaffolding over frozen per-game models at 27B on the public 25.* | **$0** | prevents a fifth lane death by mis-attribution — the failure mode R24 §3.1 already flagged |
| **R3** | **C8-feasibility probe (the only affordable fragment of the lift-bearing component).** One build, **zero gameplay**: prompt the in-kernel Qwen3.6-27B-FP8 to author `State`/`init_state`/`transition`/`render`/`outcome` for *k* games from recorded frames, then score the emitted programs **offline** with the C3 replay verifier. Pre-register the bar before running. | **1 build cycle**, well under 2.2 h (pure generation, no 7,920 s guillotine), $0 | this is the **only** experiment that touches an axis where Tycho measured anything, and it is the campaign's largest un-priced risk. It needs no baseline stack under K3′ because it is not a Δlc screen |
| **R4** | Gate S5/L1 on R3. If a 27B cannot author a passing program, then **L1 must be workstation-authored**, which (a) makes the §5.3 zero-budget ruling binding and (b) hard-caps the lane at the public 25 per §4.3(3). Say so before spending L1's effort, not after. | **$0** | L1 is the largest engineering item in the lane and its legitimacy is currently unresolved |
| **R5** | Keep C8/L5-as-gameplay **OUT** permanently at the current call budget, and record the reason as a **number**, not a preference: **349%** of the per-game wall at Tycho's cadence, **51.9×** LM calls, **25.8×** answer tokens, no weak-model ablation. | **$0** | closes the re-open loop |

### 5.2 Why not Option 1

Because the honest reading of §3 is that **the lift-bearing components are unaffordable at our call
budget — and that is the finding.** Re-scoping to include a 40×-truncated builder would not test the
architecture thesis either; it would test a thing nobody has measured, at the cost of 44% of the
actor's action budget, in the one lane that has already burned five LB windows. Option 1 trades a
defanged test for a *differently* defanged test that also costs half a week of GPU allowance.

### 5.3 Sequencing note against R25's own ordering

R25 directs: **(1) estimate ρ (N3) → (2) fix concede-trigger + K3′ + re-scope definitions → (3) THEN
Option-4 time-box; do NOT build until N3 is answered.** R1, R2, R4 and R5 are all step-2 text work
and can land now. **R3 is a build and must queue behind N3.** Note also that R3 is a candidate to
*re-order* S2: if 27B cannot author the artifact, the artifact half of lane (a) collapses to
schema-only regardless of what the persistent-namespace screen says, which makes R3 higher-VOI than
S2 among post-N3 builds.

---

## 6. Source index

| claim | source |
|---|---|
| policy RHAE 88.49 / 85.36 / 83.07 / 79.07; frontier 100.00 / 183 levels; −61% vs human | `learnings/war_room/tycho_portability_2026-08-08.md` §1a |
| four policies = who authors/repairs the model and when | ibid. §2.5, Table 1 |
| H2: `trigger` 88.1% accepted match loses to `orchestrator` 16.2% | ibid. §2.6, §6.6 |
| fallback ladder: repair → abstain → bypass to direct reasoning | ibid. §2.6 |
| budgets: 3,500 LM calls/game, 40 tool steps/turn, 24,000 answer tokens/call | ibid. §2.7 |
| 147 builder calls / 25 games | ibid. §2.5 |
| no weak-model ablation anywhere; floor backend Opus 4.8 | ibid. §2.7, §6.1 |
| C1–C14 PORT/ADAPT/SKIP verdicts | ibid. §3 |
| L0–L5 sizing; L5 ≥2 cycles, separate prereg | ibid. §5 |
| 52× per-game LM-call gap; hand-migrated sims overfit the public 25 | ibid. §6.2, §6.9 |
| S1–S5 sequence; C8/L5 + C12 explicitly not authorised | `learnings/war_room/r24_successor_lane_proposal_2026-08-08.md` §4 |
| actor = Qwen3.6-27B-FP8, 65 K context | ibid. §3(a) |
| coverage degenerate ≡1.0, 0/25 sims abstain, 3/25 hold hidden state | `learnings/war_room/r24_minutes_2026-08-09.md` §3.5 |
| K3′ requires m ≥ 3 same-config baselines; warpack null owed (4.83× variance) | ibid. §5.1 |
| wall clock binds, not actions; `actions_per_level_completed` not a co-primary | ibid. §3.1 |
| 1,686 turns / 3,638 actions / 1,569,582 gen tokens, 25 games | `runs/kernel_pulls/a22_v2_1/m1m2m3_screen.json` `M2` |
| 117.4 s/turn, 54.4 s/action, 145.5 actions/game, 7,920 s guillotine at 99.3% | `scripts/rail_regime_gap.py` → `runs/rail_regime_gap_2026-08-10.json` |
| RD-1 scored-rail latency de-rating 1.12–1.25×; official set = 110 games | `learnings/sweeps/rail_regime_gap_2026-08-10.md` §4, §6.2 |
| baseline 22 lc / 3,638 actions ⇒ 165.4 actions per level completed | `runs/kernel_pulls/war_eval_v1/benchmark.json`; portability §5 |
