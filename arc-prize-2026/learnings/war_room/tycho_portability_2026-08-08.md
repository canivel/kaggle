# Tycho portability assessment — 2026-08-08

**Question:** what of arXiv:2607.28287 (Tycho) is portable into the duck harness under
fork-never-build + zero-cloud-budget + ~9h wall, and does it change the R24 successor-lane call?

**Headline:** Tycho is reachable, its code is public (Apache-2.0), and it is **not a new lane for us —
it is a diagnosis of, and repair kit for, our own shelved exec-wm lane.** Four of the design deltas
between Tycho's world-model contract and our June `exec_wm/` sims map one-to-one onto the exact
failure modes our Stage-0 dry-run and latent-state audit measured. Three of the four repairs are
deterministic code and need no in-kernel LLM.

Provenance key used throughout: **[V]** = verified, read directly from paper/source this session;
**[V-repo]** = read from the Tycho source tree; **[V-ours]** = read from our repo this session;
**[INF]** = my inference, not stated by any source.

---

## 1. Sources actually read

| # | Source | What it gave |
|---|---|---|
| S1 | `https://arxiv.org/abs/2607.28287` — *Tycho: Active Abstraction with Programmatic World Models for ARC-AGI-3*, Jens Lehmann, Andrei Aioanei, Sahar Vahdati, submitted **30 Jul 2026** | title/authors/date/abstract, headline metrics |
| S2 | `https://arxiv.org/html/2607.28287v1` (full HTML, 52pp / 18 figs / 17 tables) | method, Definitions 1 & 3, four policies, verification protocol, RHAE formula, budgets, per-game cost, builder-call counts |
| S3 | `https://github.com/NIMI-research/Tycho` + `raw.githubusercontent.com/.../main/README.md` | licence, runtime requirements, sandbox, provider config |
| S4 | GitHub tree API `NIMI-research/Tycho@main?recursive=1` | complete file/size manifest of the implementation |
| S5 | `tycho/workspace/templates/seed_world_model.py.tmpl` (verbatim) | **the world-model contract** — the single most load-bearing artifact for us |
| S6 | `tycho/harness/scoring.py` | RHAE / reset accounting |
| S7 | `tycho/workspace/sandbox.py` | execution model, container-vs-host fallback, resource limits |
| S8 | `configs/paper/opus5_orchestrator.yaml` (verbatim) | the frontier run's exact budget + **context policy** |
| S9 | `tycho/prompts/partials/wm_orch.j2`, `tycho/workspace/wm_templates.py` | orchestrator↔builder protocol, workspace file set |
| S10 | WebSearch (`Tycho ARC-AGI-3 programmatic world model`) | corroboration; surfaced predecessor arXiv:2605.05138 and a TechTimes 2026-07-31 write-up |

Our own context read this session: `learnings/daily_brief_2026-08-07.md` §1/§4; `learnings/sweeps/research_2026-08-07.md`;
`runs/kernel_pulls/a22_v2_1/m1m2m3_screen.json`; `notebooks/duckfork/tufa-labs-duck-harness-june-30-milestone-winner.ipynb`
(17-cell structure); `notebooks/duckwar/kernel-metadata.json`; `duck_eval/warpack/_kaggle_dataset/warpack_patch.py`;
`exec_wm/collect_observations.py`, `exec_wm/validate_sim.py`, `exec_wm/scale_summary.md`; plus a full
trace of the exec-wm lane history (delegated sub-search over `ITERATION_LOG.md`, `runs/ewm_dryrun/report.md`,
`runs/latent_state_audit/`, `learnings/panel/round18/`).

**Nothing was unreachable.** No secondary-source substitution was needed for any claim below.

### 1a. Metric framing correction (per coordinator)

The "183/183" number is a **completion** count, not RHAE, and the two headline figures come from
different experiments. Precisely, from S1/S2 **[V]**:

- **88.49 mean RHAE** — the *highest of the four orchestration policies* in the matched study
  (orchestrator = actor-requested delegation to a builder subagent), backend **Claude Opus 4.8**,
  25 public games. The other three: single 85.36, trigger 83.07, no-world-model 79.07.
- **100.00 RHAE and all 183 levels** — the *selected* (orchestrator) policy re-run with frontier
  backends **GPT-5.6 Sol** and **Opus 5**. Note 100.00 is the **metric ceiling**: RHAE per game is
  `min(Σwℓeℓ/Σwℓ, 100·Σwℓcℓ/Σwℓ)`, so full completion pins the second term at 100 and the score
  saturates. It is "maxed the benchmark", not "beat it by 61%".
- **−61% actions** — Opus 5 versus the *official human baselines*, not versus any agent.
- **Provenance calibration:** all of the above is **self-reported by the authors on the 25 public
  games**, same footing as the Prime Agent 95.5% figure the sibling assessment is handling. Neither
  is the Kaggle competition leaderboard. Our LB is a different arcade — the milestone duck scored
  1.21, we are at 1.33, gold cutoff 1.56. **Do not put 88.49/100.00 and 1.33 in the same sentence
  anywhere in the R24 doc** — they are not the same measurement. **[INF]**

---

## 2. Tycho method summary **[V]**

### 2.1 Formalisation
Games are **parameterised rendered deterministic Moore machines** (Definition 1): a hidden state
space `S`, an output map emitting `(rendered_grid, available_actions, outcome)`, and transition
emissions `ε_θ` that capture *transient* frames between settled observations.

**Definition 3, typed interaction history**, is the second pillar and is easy to overlook: frames are
tagged by role — **decision** frames vs **transient/animation** vs **terminal** — and
> "A history is faithful when every recorded action is attached to the decision frame from which it
> was chosen, never to a transient or terminal frame."

### 2.2 How the world model is represented and induced **[V-repo, S5]**
An **executable Python program** the agent writes and edits in a workspace, seeded from
`seed_world_model.py.tmpl`. The contract is a game-specific `State` **dataclass** plus:

- `init_state(grid0, level) -> State` — s0 from the level's first frame
- `transition(state, action) -> State` — `action` is `{"action", "row", "col"}` (row/col = the ACTION6 click cell)
- `render(state) -> np.ndarray` — int16 grid, colours 0–15, **may return `UNKNOWN = -1` for cells the
  modeled state cannot legitimately claim yet**
- `outcome(state) -> "ongoing" | "level_complete" | "game_over"`
- optional: `observation_variants(state)` (≤5 alternate grids for display/quantisation ambiguity only),
  `actions(state)` (**focused** candidate set — the template explicitly warns the ACTION6 space is
  4096 cells and demands "object centres, reachable targets, switch cells", not the whole grid),
  `subgoals(state)`, `heuristic(state)`, `planner_key(state)`.

The template's own words on state design:
> "The modeled state is deliberately game-specific. It may store the grid directly, an object list, a
> finite-state controller, a user-interface mode, a selected color or tool, a counter, geometric
> constraints, or a combination of these."
> "A move/life counter rendered to pixels is real state, so hold its value in a field and paint it in render()."

Workspace also seeds `verify.py` and `plan.py` (`python plan.py [auto|astar|bfs|subgoals]` →
`notes/validated_plan.json`), plus prose hypotheses in `notes/world_model.md` **[V-repo, S9]**.

### 2.3 How it is verified against observed frames **[V]**
**Offline replay of recorded history — not extra live play.**
> "Verification replays the most recent attempt for each level from its first frame and compares
> predicted with recorded observations."

Metrics: **accepted transition match** (≥1 claimed cell, and *every* claimed cell matches the next
frame, via canonical render or a bounded variant); **coverage** (fraction of cells the canonical
render actually claims); **terminal-outcome recall** on boundary actions. Abstained cells make no
correctness claim and *reduce coverage* rather than counting as errors; adding a complete-observation
variant does not.

This is the piece our panel demanded and we never had: **a fidelity metric with an honesty channel**
(a model that knows what it doesn't know scores low coverage instead of high-confidence-wrong). **[INF]**

### 2.4 How it drives action selection **[V]**
1. Canonically **replay the model from level start through current observations** — "thereby
   preserving accumulated hidden variables" (state *threading*, not a cold start from the current grid).
2. Forward-search from that threaded state for a sequence reaching `level_complete`.
3. Record the predicted observation hash after each action for **divergence detection**.
4. > "The actor receives the validated route as conditional advice and commits **one action at a time**."
   The actor must "re-observe after acting and re-plan from the resulting state"; the automatic
   planner probe is "separate and **advisory**".

### 2.5 LLM vs deterministic code **[V]**
- **LLM (actor):** decides *whether* a model is worth building/consulting/repairing (this is the
  "active abstraction" metareasoning), writes the hypothesis prose, commits every action. Only the
  actor can act.
- **LLM (builder subagent, orchestrator policy):** writes/repairs `world_model.py`, runs it against
  observations, returns a compact report (dynamics evidence, plan validity, remaining assumptions,
  recommended next actions). Cadence per `wm_orch.j2`: "a few times per level, **not every turn**";
  observed **147 builder calls across 25 games**, 68 recommendations surfaced, 35 exactly followed.
- **Deterministic code:** the workspace `transition/render/outcome` execution, the replay verifier,
  the A*/BFS/subgoal planner, the typed-history tagger, the sandbox, the scorer.

Four policies were compared (Table 1): `no_world_model` / `single` (actor edits) / `orchestrator`
(builder on actor request) / `trigger` (builder auto-fires on verification failure).

### 2.6 Failure and fallback **[V]** — the most important result for us
> H1: "Selective executable-model use improves completion and action efficiency over direct reasoning."
> H2: **"Accurate transition prediction alone does not ensure strong gameplay."**

H2 is demonstrated brutally: `trigger` achieves **88.1% accepted transition match vs orchestrator's
16.2%**, yet orchestrator wins on completions and RHAE, because trigger "makes more builder calls,
spends more inference, and completes fewer levels." Fallbacks, in order: incremental repair using
mismatches as counterexamples → abstention/observation-variants for genuine ambiguity → **bypass**:
the actor reverts to direct reasoning without the executable hypothesis. Failed plans trigger
re-evaluation, never blocking. The paper's own justification:
> "They let the agent simulate alternative action sequences before spending real actions; they make
> exploration purposeful when competing hypotheses predict different observations or plans; and they
> help localize failure when modeled dynamics fit the interaction history but outcome inference or
> planning remains wrong."

### 2.7 Budgets and cost **[V]**
Per game: **3,500 LM calls**, 40 tool steps/turn, 24,000 answer tokens/call, inference ceiling $750
(matched) / $1,500 (frontier, not reached). Measured mean/median API-equivalent cost per game:
no-WM Opus $226/$106; orchestrator Opus $231/$169; GPT-5.6 Sol $179/$114; **Opus 5 $119/$97**.
Wall-clock is not reported. **No weaker/smaller-model ablation exists anywhere in the paper.**

### 2.8 The context-policy finding **[V-repo, S8]** — reconciles Tycho with our A22 death
`configs/paper/opus5_orchestrator.yaml` sets:
```yaml
context:
  prompt_caching: true
  history: tail_evict
  grid_keep: 3
  image_keep: 3
  reasoning_keep: 3
  image_cap: 4
  context_emergency_soft_tokens: 850000
```
**Tycho evicts context aggressively** — keeps only 3 recent grids, 3 images, 3 reasoning blocks. That
looks like a direct contradiction of our 08-07 eviction-harm finding until you see *why* it is safe:
the load-bearing information has already been moved into `world_model.py` + `notes/`, which live in
the **workspace filesystem, outside the context window**, and are re-derivable by replay.

**[INF] This is the cleanest available reconciliation of our A22 result with both blueprints, and it
should be the framing sentence of the R24 lane decision:** eviction is not harmful *per se*; eviction
is harmful **when it is not preceded by externalisation**. A22 v1/v2/v2.1 deleted episodes that were
the only copy of the information (harm monotonic in evicted chars, ρ = −0.403). Tycho deletes frames
that are the *second* copy. Same operation, opposite sign, and the discriminator is whether a verified
external artifact exists first. That also predicts the ordering constraint for us: **do not re-open any
eviction work until an externalised artifact is live and verified.**

---

## 3. Portability table

Rails: **R1** fork-never-build (kernel = fork of `arc3-baseline.ipynb`, code rides in via attached
Kaggle dataset, byte-audited); **R2** zero cloud budget, free Kaggle builds only, ~9h wall;
**R3** no internet in-kernel (`enable_internet: false`), actor LLM is the local vLLM
Qwen3-6-27B-FP8 **[V-ours, `notebooks/duckwar/kernel-metadata.json`]**; **R4** eviction is harmful,
additive/externalised state is the direction.

| # | Tycho component | Verdict | Reason (rail-tied) |
|---|---|---|---|
| C1 | **4-function WM contract** (`State` dataclass + `init_state`/`transition`/`render`/`outcome`) | **PORT** | Pure data-schema change to assets we already own. Our 24 sims use a *stateless* `simulate(grid, action_id, x, y) -> (grid, reward, done)` **[V-ours, `exec_wm/validate_sim.py`]** — the state **is** the grid, so hidden variables are structurally unrepresentable. This is the mechanical root cause of the latent-state audit (parity/mod3/mod4/mod5 aliasing; 10/11 ALIASED-RESOLVABLE games collapsed on holdout). Ships in the dataset, no notebook change. R1/R2 clean. |
| C2 | **`UNKNOWN=-1` abstention + separately-reported coverage** | **PORT** | Deterministic, ~free, and it is *the* answer to panel round18 [MAJOR] "EWM line has no world-model fidelity metric — unfalsifiable by construction." Converts silent wrong predictions (our step-0 aborts) into an honest low-coverage signal a gate can read. R1/R2 clean. |
| C3 | **Replay verification threaded from level frame 0** | **PORT** | Costs **zero scored actions** — it replays *recorded* frames, not the live env (see §6). Directly replaces our IID 70/30 random-tuple split, which Stage-0 already proved measures the wrong quantity (sp80 100.0 held-out → 0.026–0.879 on-trajectory; lp85 100.0 → 0.087–0.458; sb26 100.0 → 0.106–0.162). We can run this **locally, today, offline, at $0** against `runs/ewm_dryrun` streams. |
| C4 | **Advisory one-action-at-a-time commitment + bypass** | **PORT** | Strictly safer than our `ExecWMHook` (beam_width=4, lookahead=2, whole-plan commit) which is where the step-0 abort mass landed (lp85: 126/138 plans abort at step 0). Slots into the existing `_HarnessGameSession._execute_action` seam warpack already patches **[V-ours]**. S-sized. |
| C5 | **Typed interaction history** (decision vs animation vs terminal frames) | **PORT** | Our collector recorded raw engine `(state_t, action, state_t1)` with **no frame typing** **[V-ours, `exec_wm/collect_observations.py`]**, so transitions were fit across animation frames — a strong candidate contributor to the aliasing. Deterministic tagger, no LLM. **[INF]** on the causal attribution. |
| C6 | **Focused `actions(state)` candidate set for ACTION6** | **PORT** | 4096-cell click space; our beam search had no salient-click focusing. Cheap, deterministic, and a direct action-efficiency lever — the −61% angle. |
| C7 | **`plan.py` A*/BFS/subgoals + `heuristic`/`planner_key`** | **ADAPT** | Port the algorithms, not the workspace. Must be hard-budgeted in CPU-ms/turn against the 9h wall (R2) and must respect our standing rule *never cap MCTS budget for long BFS* — so budget by wall-time, not by depth. |
| C8 | **Builder subagent writing/repairing Python in-kernel** | **ADAPT (high risk) / defer** | The heart of "active abstraction", and the only part that makes the model *adaptive per game*. But: (a) our in-kernel model is Qwen3-27B, and the paper has **zero weak-model ablation** — the matched study's floor is Opus 4.8; (b) budget collision, see §6. Not a first push. |
| C9 | **Active-abstraction metareasoning** (actor decides *when* a model pays) | **ADAPT** | The idea ports as a cheap deterministic **gate**, not as LLM metareasoning: consult the model only for games whose offline coverage+accepted-match clears a pre-registered bar; bypass everywhere else. That is C9 reduced to a lookup table, which is exactly what our 24-game fidelity profile gives us. **[INF]** |
| C10 | **Container sandbox** (`--memory 1g --cpus 1 --pids-limit 64`, network-disabled) | **ADAPT** | Kaggle can't nest Docker. The repo ships `TYCHO_SANDBOX_RUNTIME=host` → `[sys.executable, '-B', script, *args]` **[V-repo, S7]**; repo labels host mode "only for trusted local development", which is fine — under our design the code is *ours*, generated offline and byte-audited, not live-authored by an untrusted model. Keep the subprocess timeout + output caps. If C8 ever lands, the code becomes model-authored and this verdict must be revisited. |
| C11 | **Tycho harness/runner** (`run_parallel.py` 70KB, `harness.py` 35KB, `resume.py`, `run_status.py`) | **SKIP** | Replacing the duck's runner violates R1 outright, and structural drift already cost us 5 ERROR submissions (v62–v66). Our integration surface is exactly one cell — the cell-12 customization hook calling `apply(bm)` **[V-ours]**. |
| C12 | **Frontier backends** (Opus 5 / GPT-5.6 Sol via `ANTHROPIC_API_KEY`/`OPENAI_API_KEY`) | **SKIP** | R3 (no internet) and R2 (zero budget). Reproducing one paper run ≈ $119×25 ≈ **$3.0k**. Hard no. |
| C13 | **`tycho-viewer`** (`viz.py` 104KB, frame-by-frame replay UI) | **SKIP** | Nice, but we already have `diagnostics.html` and the warpack sidecar-event pipeline. Zero LB value. |
| C14 | **`history: tail_evict` context policy** | **SKIP for now — sequenced, not rejected** | Per §2.8 this is only safe *after* externalisation is live and verified. Re-proposing eviction before that would re-run A22 with extra steps. Revisit only as a post-hoc token-budget optimisation once C1–C5 have shipped and passed. |

---

## 4. Synergy / conflict with the Prime Agent blueprint

*(The sibling agent owns the Prime Agent assessment; this section only marks the seam.)*

**Complementary, near-orthogonal, and they compose cleanly.** Prime Agent contributes the *substrate*:
harness state `H = (prompt, sub-agents, skills, memory)` held outside the active context, a persistent
IPython kernel as the single tool, sub-agents as function calls, crash recovery via session JSONL +
kernel snapshot. Tycho contributes the *artifact schema and the verification contract* for one
particular, high-value item to put in that store.

Concretely, the slot:

- **Tycho's `world_model.py` is a Prime-Agent memory/skill entry** — but a *typed and falsifiable*
  one. Generic externalised memory has no notion of being wrong; a Moore-machine program has
  `accepted transition match` and `coverage`. Externalisation without a falsifier is how you get an
  unfalsifiable lane, which is precisely the [MAJOR] our own panel filed against EWM. Tycho supplies
  the missing falsifier. **[INF]**
- **Tycho's builder subagent is a Prime-Agent `rlm("sub-task")` call.** Same shape, different payload.
- **Both converge on the same ordering claim** and so does our A22 corpse: externalise first, then the
  context window can be small. Tycho proves the second half empirically (`tail_evict`, `grid_keep: 3`)
  while topping the benchmark.
- **Conflict — the only real one:** both blueprints are validated exclusively at frontier scale
  (Opus 5 / GPT-5.6 Sol). Neither has a small-model ablation. Our binding constraint is a local 27B
  actor, which neither paper speaks to. Anything we port must therefore be biased toward the
  *deterministic* half of each system and away from the "the LLM will figure out when to do this" half.
  That is the same bias that makes C1–C7 PORT and C8 defer.
- **Non-duplication note:** if the successor lane is built as a Prime-Agent-style externalised-state
  harness, Tycho does **not** compete for the lane slot — it becomes the spec for the first artifact
  type that harness stores. One lane, two papers.

---

## 5. Engineering lift and what a single free-kernel seed-1 screen can test

Sizing is in **dataset-code push cycles** (a cycle = build the patch into
`duck_eval/warpack/_kaggle_dataset/`, push the dataset version, push the kernel fork, pull-back
byte-audit, run seed-1 screen). Baseline for all comparisons is the existing **war-eval seed-1**
(ledger-OFF): 25 games, **22 level-completions, 3,638 scored actions, 1,686 LLM turns, 1,569,582
generated tokens** **[V-ours, `runs/kernel_pulls/a22_v2_1/m1m2m3_screen.json`]**.

| Item | Size | Push cycles | Notes |
|---|---|---|---|
| **L0. Local re-verification of the 24 existing sims under Tycho's protocol** (C3+C2): replay from level frame 0 with state threading, report accepted-match **and coverage** per game | **M** | **0** | Runs entirely offline on `runs/ewm_dryrun` streams + `exec_wm/observations/`. $0, no kernel. **This is the gate that decides whether anything else is worth a push.** |
| **L1. Interface migration of the 24 sims** to `State`/`init_state`/`transition`/`render`/`outcome` + abstention (C1+C2) | **L** | 0 (bundled with L2) | The one genuinely large item. Mostly LLM-assisted rewriting on the workstation; the June generation pass cost ~10h / ~7M tokens for 24 games. See §6 budget caveat. Can be **staged**: migrate only the games L0 shows are carriers. |
| **L2. Advisory hook rewrite** (C4): one action at a time, re-observe, silent bypass on divergence, replacing the beam-search whole-plan commit | **S** | 1 (shared) | Existing `_execute_action` seam; warpack pattern, env-flag gated, blanket-guarded. |
| **L3. Typed-frame tagger** (C5) + **focused `actions(state)`** (C6) | **S/M** | 0 (bundled) | Deterministic. C6 is the direct action-efficiency lever. |
| **L4. Budgeted planner** (C7) with wall-time cap and per-game consult gate (C9) | **M** | 0 (bundled) | Must instrument CPU-ms/turn as a canary. |
| **L5. In-kernel builder subagent** (C8) | **L** | ≥2, separate prereg | Do **not** bundle. Needs its own arm and its own kill criteria. |

**Recommended first screen — one kernel push, arm = L1(staged)+L2+L3+L4+C9, everything env-flag gated:**

- **Primary (non-harm, mirrors the A22 prereg shape):** mean Δ level-completions vs war-eval seed-1,
  cap on worst game. Non-harm first; lift second. This lane has burned five LB windows on the
  scratch-build family and a 0.14 ship — the bar is *do no damage*.
- **Mechanism canaries, all pre-registered and all readable from a single run:**
  `wm_consulted_events`, `wm_bypass_events`, `mean_coverage` and `accepted_match` per game
  **on-trajectory in the live kernel** (not held-out), `plan_abort_step_histogram` (the Stage-0 killer
  metric — step-0 mass must fall), `planner_cpu_ms_per_turn`, and **`actions_per_level_completed`
  (baseline 3,638/22 = 165.4)** — the direct −61% analogue and the metric the whole Tycho story is about.
- **Falsifiers, pre-registered:** if step-0 abort mass does not fall materially versus the Stage-0
  dry-run, the migration did not fix the on-trajectory problem and the lane closes for good. If
  `actions_per_level_completed` does not improve while Δlc is flat, the efficiency claim does not
  transfer at 27B.
- **What one seed-1 screen genuinely cannot settle:** whether a *local 27B* can author or repair these
  programs (that is L5, needs its own arm), and anything about variance — borro1980's map says 2 games
  carry ~65% of variance, so a single screen reads mechanism, not lift.

**Sequencing rule:** L0 is free and gates everything. If L0 shows the carrier set is still ~4 games
after the protocol fix, the honest call is to bank that as a second negative result and hand the lane
slot to the sibling's Prime-Agent-style externalisation harness, with C1/C2/C3 retained as the artifact
schema.

---

## 6. Risks and unknowns

1. **Model dependence — the top risk. [V]** No weak-model ablation exists. The matched study's floor
   backend is Opus 4.8; headline results need Opus 5 / GPT-5.6 Sol. Writing a correct Moore machine
   with hidden state is a hard program-synthesis task and there is no published evidence a 27B local
   model can do it. **Mitigation:** ports C1–C7 are deterministic and model-independent; only C8/L5
   is exposed, and it is explicitly deferred to its own arm.
2. **Budget collision on inference. [V + V-ours]** Tycho allows **3,500 LM calls per game**. Our entire
   25-game seed-1 run is **1,686 LLM turns total (~67/game)** — roughly a **52× per-game gap**. Any
   in-kernel builder call is drawn from the same ~67-turn allowance the actor needs to play. This is
   the single hardest constraint and it is why C8 cannot be a first push. The orchestrator policy's
   observed cadence (147 builder calls / 25 games ≈ 5.9 per game) is at least the right order of
   magnitude — that is the number to design against if L5 ever runs. **[INF]**
3. **Environment resets — checked, and it is good news. [V, S2+S6]** Verification is **offline replay
   of recorded frames** and costs **zero scored actions**. Live-scoring rules per `scoring.py`:
   `actions_taken` counts every reset during gameplay *except* the initial setup reset, and a "RESET
   that creates a play is unscored" while a mid-level protocol RESET "costs one in-play action". This
   is consistent with warpack's existing banking design (open a *new* play, unscored). So the method
   does **not** import a hidden reset tax. The `scoring.py` warning is worth adopting verbatim anyway:
   you must feed all *n* levels including unreached ones, or both the weighted-average denominator and
   the completion cap are wrong.
4. **Compute profile in-kernel. [INF]** Executing `transition`/`render` per candidate action per turn
   plus A*/BFS is CPU work competing with the vLLM server on the same box for 9h. Tycho gave its
   sandbox 1 CPU / 1 GB. Unbudgeted planning is a wall-clock DoS risk — instrument `planner_cpu_ms_per_turn`
   as a hard canary with an auto-bypass, and budget by wall-time not by depth.
5. **Sandbox degradation. [V-repo]** We must run `host` mode (no nested Docker). Acceptable only while
   the executed code is ours and byte-audited; if L5 lands, model-authored code executes in-kernel and
   the risk class changes materially.
6. **H2 is a warning aimed at us. [V]** `trigger` had 88.1% accepted match and *still* lost to
   orchestrator's 16.2%. Fidelity is necessary, not sufficient — and it is anti-correlated with
   spend. Our June instinct was to maximise `state_exact` (91.7% = 22/24 games in Class A —
   **CORRECTED 2026-08-10: not held out.** `exec_wm/validate_sim.py` ran at `split=all` over the
   same 200 tuples the sim was authored against, for the selected v1/v2 winner;
   `exec_wm/scale_summary.md:3-4,54,74`) and
   that number turned out to be nearly uninformative on-trajectory. **Do not re-run that mistake by
   optimising accepted-match. The pre-registered primary must stay level-completions / actions-per-
   completion, with fidelity as a mechanism canary only.**
7. **Regeneration cost vs the zero-budget rail. [INF — needs a ruling]** L1 is workstation LLM work
   (the June pass: ~10h, ~7M tokens). The R24 rail as written bans *cloud GPU/LLM eval spend*; local
   agent-assisted code authoring has been treated as in-bounds. **This needs an explicit call at R24
   rather than an assumption**, because L1 is the largest item and its legitimacy determines whether
   the lane is affordable at all.
8. **Unknown: wall-clock.** The paper reports dollars and call counts but **no wall-clock per game**.
   We cannot infer whether the 9h/25-game envelope is even geometrically compatible with the
   orchestrator loop. **[V — absence confirmed in S2.]**
9. **Generalisation caveat (standing rail).** All Tycho numbers are on the **25 public games**. Our
   private-LB exposure is broader. A per-game hand-migrated sim set is by construction overfit to the
   public 25 and contributes **nothing** on unseen games — the honest framing is that C1–C7 buy public
   -set action efficiency, while the *generalising* asset is the schema+verifier, not the 24 sims.

---

## 7. R24 recommendation (one paragraph)

**Adopt Tycho as the specification for the successor lane's first artifact type, not as a lane of its
own — and spend zero kernel pushes on it this week.** The decisive finding of this read is that Tycho
is a point-by-point diagnosis of why our own exec-wm lane failed: our sims were stateless grid→grid
functions (no hidden state — hence the latent-state aliasing), validated on IID random-split tuples
(hence 100% held-out saturation collapsing to 0.026–0.16 on-trajectory), with no abstention channel
(hence confident-wrong step-0 aborts, lp85 126/138), fit across untyped animation frames, and consumed
by a whole-plan beam search instead of advisory one-action-at-a-time commitment. Tycho fixes all five,
and four of the five fixes are deterministic code requiring no LLM in the kernel. It also resolves the
apparent contradiction with our 08-07 eviction-harm result: Tycho evicts context hard
(`history: tail_evict`, `grid_keep: 3`) and is safe doing so *because* the load-bearing state already
lives in a verified external program — eviction is harmful when it is not preceded by externalisation,
which is exactly what A22 did wrong, and which sets the ordering constraint for everything downstream.
So the R24 call: fold Tycho into the state-externalisation lane the sibling is scoping (it is the
typed, falsifiable artifact that lane needs, and it answers the round18 [MAJOR] unfalsifiability
charge), and authorise only **L0** — a free, offline, zero-push re-verification of the 24 existing sims
under Tycho's replay-with-threading protocol reporting coverage and on-trajectory accepted-match. L0
costs nothing, uses assets already on disk, and is a genuine falsifier: if the carrier set is still
~4 games after the protocol fix, we bank a clean second negative and close exec-wm permanently; if it
expands, we have earned the right to one staged L1+L2+L3+L4 arm with non-harm primary and
`actions_per_level_completed` (baseline 165.4) as the efficiency read. Explicitly **do not** authorise
the in-kernel builder subagent (L5) — the 52× per-game LLM-call gap and the total absence of any
weak-model ablation make it the one component whose evidence does not survive contact with a local 27B
actor — and get an explicit ruling on whether workstation LLM regeneration counts against the
zero-budget rail before committing to L1.
