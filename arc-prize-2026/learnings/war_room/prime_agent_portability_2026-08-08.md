# Prime Agent → duck portability assessment (2026-08-08)

Prepared for **R24 (Sunday 2026-08-09)**, agenda item §4.1(a) *successor lane: state-externalization /
programmatic world model*.

Provenance convention used throughout:
- **[V]** = verified by direct read of the repo source / vendor blog / our own repo.
- **[V-2nd]** = verified in a secondary source only (press, aggregator), not in primary material.
- **[INF]** = my inference from the above; not stated anywhere.

**Repo reachability: SUCCESS.** `github.com/PrimeIntellect-ai/prime-agent` was reached via the GitHub
contents API and `raw.githubusercontent.com`. Source files were read directly, not just the marketing page.

---

## 1. Sources actually read

| # | URL | What it gave |
|---|---|---|
| S1 | `https://api.github.com/repos/PrimeIntellect-ai/prime-agent/contents` | Root listing. Node/TypeScript monorepo: `package.json`, `package-lock.json` (140 KB), `tsconfig*.json`, `biome.json`, `install.sh` (45 KB), `packages/`, `prime-agent-runtime/`, `AGENTS.md`, MIT `LICENSE`. **No Python package at root.** |
| S2 | `.../contents/packages` | `agent`, `ai`, `coding-agent`, `tui` |
| S3 | `.../contents/packages/agent/src` | `agent-loop.ts`, `agent.ts`, `index.ts`, `proxy.ts`, `types.ts` |
| S4 | `.../contents/packages/coding-agent/src` | CLI/config/main + `core/`, `modes/`, `themes/`, `utils/`, `bun/` |
| S5 | `.../contents/packages/coding-agent/src/core` | ~60 files incl. `rlm-runtime.ts`, `skills.ts`, `context-tree.ts`, `autonomous.ts`, `goals.ts`, `session-*.ts`, `system-prompt.ts`, `output-guard.ts`; dirs `compaction/`, `refinement/`, `kernel/`, `tools/`, `mcp/`, `prompts/`, `extensions/` |
| S6 | `raw.../packages/agent/src/agent-loop.ts` | The actual loop: `agentLoop()` / `agentLoopContinue()` → `runLoop()`; `config.transformContext`, `config.convertToLlm()`, "Build LLM context immediately before starting the provider call"; `executeToolCallsSequential` / `...Parallel`; `shouldStopAfterTurn()` + `getFollowUpMessages` + `getContinuationMessages()`; `raceWithAbort()`. **No compaction/turn-limit logic in the loop itself** — it is all injected via config. |
| S7 | `raw.../core/compaction/compaction.ts` | `shouldCompact()`: `contextTokens > contextWindow − reserveTokens`; defaults reserve **16,384**, `keepRecentTokens` **20,000**; `findCutPoint()` never cuts at a tool result; `generateSummary()` is an **LLM summary** (generation-side), chained via `previousSummary`; boundary marked by a `CompactionEntry { firstKeptEntryId }`; tracks read/modified files across compacted spans. |
| S8 | `raw.../core/refinement/refinement.ts` | `/refine`: reads `AgentMessage[]` trajectory + `HarnessState` + prior `RefinementResult[]`; CRUD over **prompt / memory / skill / subagent**; `base_system_prompt` immutable; one `completeSimple()` call with `REFINEMENT_SYSTEM_PROMPT` over **last 80 K chars** of conversation, output capped `min(model.maxTokens, 32_000)`; `applyRefinementProposal()` appends `HarnessRefinementEvent{id, trigger, changes, outcome}`; `rollbackProposal()` by `options.rollbackId`; global scope also appends to `refinements.jsonl`; optional `reviewAutoRefine()` gate. |
| S9 | `raw.../core/skills.ts` | Skills discovered recursively; `SKILL.md` marks a skill root (stop recursion); user `{agentDir}/skills`, project `{cwd}/.earendil/skills`. Two kinds: markdown-with-frontmatter, and **Python skills** (`SKILL.md` + `pyproject.toml` + `src/{import_name}/__init__.py`). `formatSkillsForPrompt()` puts **name + description + file location + `<python_import>`** in the system prompt; body is read on demand ("Use ipython to inspect a skill's file when the task matches its description"). No token-budget logic. |
| S10 | `raw.../core/rlm-runtime.ts` | Interfaces only: `CreateRlmSubagentRuntimeOptions`, `RlmRunHandler`, `RlmFindModelsHandler`, `RlmListSubagentsHandler`, `RlmDeleteSubagentHandler`; `DEFAULT_RLM_MODEL_SEARCH_LIMIT = 8`, `MAX_RLM_MODEL_SEARCH_LIMIT = 20`, `RLM_SUBAGENT_SESSION_NAME_MAX_LENGTH = 64`. Kernel mechanics live elsewhere. |
| S11 | `raw.../core/kernel/state-snapshot.ts` | `buildSnapshotCode()` pickles each top-level kernel name **independently with `dill`** into a `.dill` payload + `.json` manifest (`snapshotPathIn()`, `manifestPathIn()`); skips underscore/IPython/unpicklable names and *reports* rather than aborting; `DEFAULT_SNAPSHOT_MAX_BYTES` = **256 MB**; `buildRestoreCode()` revives names independently and is "tolerant of a missing or corrupt file: reports an empty restore, never raises"; `RESULT_MARKER` + `parseSnapshotResult()` / `parseRestoreResult()`. |
| S12 | `.../contents/.../core/kernel` | `boot-gate.ts`, `bootstrap.ts`, `bootstrap-cli.ts`, `fork-server.ts`, `fork-server-script.ts`, `state-snapshot.ts` — i.e. a **forked long-lived kernel process** with a boot gate. |
| S13 | `raw.../AGENTS.md` | Internal dev rules (style, daemon protocol versioning, dependency 7-day min age, git hygiene, provider-integration checklist, lockstep releases). Confirms package split `ai` / `coding-agent` / `tui` / `agent`. Little architecture content. |
| S14 | `raw.../README.md` | "persistent IPython is the built-in model tool; file operations, shell commands, tool use, subagents, and context management happen through code"; "`rlm(...)` spawns real child agents"; "skills are importable Python packages"; "`/refine` reviews the current trajectory and can apply small, evidence-backed updates to supplemental harness state"; "`/autonomous` continues within configured turn, token, and time budgets and can run user-defined quality gates"; Continual Harness cites **arXiv:2605.09998**; "On first launch, run `/login` to choose a subscription or API-key provider." **README contains no ARC numbers.** |
| S15 | `https://www.primeintellect.ai/blog/prime-agent` | The ARC claim and the harness formalism (quoted in §2). |
| S16 | `https://www.primeintellect.ai/blog/rlm` | RLM mechanics: `answer` dict with `content`/`ready`; REPL output shown to the model **capped at 8192 chars/turn, user-adjustable**; `llm_batch` for parallel sub-LLM calls; "any tools you give the environment will only be usable by the sub-LLMs … the main RLM doesn't have to see those tokens"; Oolong ~1.5 M-char contexts handled with much lower *root* sequence length. |
| S17 | `https://www.marktechpost.com/2026/08/06/prime-intellect-releases-prime-agent/` | Secondary: `H = (ρ, G, K, M)`; `rlm("sub-task")` "returns at admission rather than blocking"; background daemon owns sessions, detach/reattach; "a crashed worker recovers from the session JSONL plus a kernel snapshot"; sub-agents unload after 30 min idle; Factorio reward-hacking anecdote. |
| S18 | WebSearch, ARC-AGI-3 scoring / RHAE | RHAE = Relative Human Action Efficiency; per-level score = `(human_actions / agent_actions)²`, **capped at 1.15**; human baseline = upper-median first-time human by action count. Also surfaced the **Schema** harness page (`schema-harness.github.io`, ~99% on ARC-AGI-3 public). |
| S19 | Our repo (read-only) | `learnings/daily_brief_2026-08-07.md`, `learnings/sweeps/research_2026-08-07.md`, `runs/lb_ground_truth.md`, plus a full read-only structural map of the duck harness (files cited inline in §3). |

**Not obtained:** the ARC-AGI-3 task prompt Prime Agent used, any per-game trace, and any ablation table.
The blog states no ablations exist [V].

---

## 2. Prime Agent architecture, as verified

### 2.1 The two abstractions

1. **Recursive Language Model (RLM)** [V, S14/S16] — "treats context as a variable and subagent delegation
   as function calls inside a REPL". A **persistent IPython kernel is the single built-in tool**. Files,
   shell, other tools, sub-agents and context management are all reached by writing Python. Kernel stdout
   returned to the model is truncated (**8192 chars/turn default** in the RLM paper implementation, S16), so
   large data stays in kernel variables and never enters the context window.
2. **Continual Harness** [V, S15/S17] — durable state `H = (ρ, G, K, M)` = supplemental *prompt notes*,
   *sub-agent specs*, *skills*, *memory*. All four expose identical CRUD. `/refine` reads the agent's own
   trajectory and applies "the smallest relevant CRUD edit", recording trigger + outcome, revertible by ID
   [V, S8]. Cited paper: arXiv:2605.09998.

### 2.2 Agent loop [V, S6]

`agentLoop()`/`agentLoopContinue()` → `runLoop()`: outer while-loop over turns, inner loop over tool-call
batches. Per turn: `transformContext` → `convertToLlm` → build a **fresh** `Context` (system prompt +
converted messages + tools) immediately before the provider call → provider call → validate/prepare/execute
tool calls (sequential or `Promise.all` parallel) with `beforeToolCall`/`afterToolCall` hooks → append
`ToolResultMessage[]`. Turn ends when `shouldStopAfterTurn()` **and** the steering poll
(`getFollowUpMessages`) is empty. The loop file itself contains **no compaction, token budget, or turn cap**
— those are injected by config, which is why compaction lives in a separate module.

### 2.3 State externalization — three distinct mechanisms

This is the part that matters to us, and it is genuinely three separable things:

| Mechanism | Where state lives | Enters context? |
|---|---|---|
| **Kernel namespace** (RLM) | Python variables inside a long-lived forked kernel process [V, S12] | Only what the model chooses to `print`, capped per turn [V, S16] |
| **Harness state H** | On-disk prompt notes / memories / skills / subagent specs [V, S8/S9] | Skills: **name + description + path only**; body read on demand [V, S9]. Memories/prompt notes: injected. |
| **Session JSONL + kernel snapshot** | Append-only JSONL per session; `dill` per-name pickle of the kernel namespace, ≤256 MB [V, S11/S17] | No — recovery artifact only |

**Compaction, honestly reported [V, S7]:** Prime Agent *does* compact, but it is **generation-side**
(LLM summary at a safe cut point, chained across compactions, preserving recent 20 K tokens and the
initiating user message of a straddled turn). It is *not* selection/eviction. This is exactly the class
distinction proved in arXiv:2608.01326 and is the single most direct external corroboration of our A22
post-mortem: the winning harness summarizes at the boundary and pushes durable state to disk; it never
hand-evicts "stale" episodes. **[INF]** the compaction is a fallback for very long sessions; the primary
context discipline is that bulk data never enters the context at all.

### 2.4 Per-level / per-game strategy [V, S15 — thin]

There is **no ARC-specific code**. The vendor states: *"Prime Agent was developed as a CLI coding agent, so
the only ARC AGI 3 specific changes are to the task prompt, inspired by the standard prompt setup used in
PRO-LONG."* [V, S15] That is the entire disclosed ARC configuration. **[INF]** the per-game world model is
therefore emergent — the model writes Python in the kernel to represent the game and promotes what works
into skills/memories via `/refine`. This is the same end-state as Tycho's explicit programmatic world models
(arXiv:2607.28287), reached by a general mechanism rather than a designed one.

### 2.5 Action-budget handling [V/INF]

Prime Agent's budgets are **turn / token / wall-clock** (`/autonomous`, S14) — there is *no* action-efficiency
controller in the harness. But ARC-AGI-3's metric is **RHAE**, `(human_actions/agent_actions)²` capped 1.15
[V-2nd, S18], which punishes wasted actions quadratically. **[INF]** the 95.5% therefore implies the agent
was already near-human in *actions per level* across 183/183 levels, achieved implicitly (think in the
kernel, act rarely) rather than by any budget mechanism. This is the same signature as Tycho's "61% fewer
scored actions". **This is the key transferable insight: reasoning that happens in a sandbox costs zero
scored actions.**

### 2.6 Model dependence — and the provenance caveat that must ride with every citation

- **The 95.5% is vendor-reported and independently unreplicated.** ARC Prize keeps harness results off the
  official leaderboard; the **official Opus 5 number is 30.2%** (our own `runs/lb_ground_truth.md` records
  it, arcprize.org 2026-07-24 [V, S19]). The 95–100% figures live on a **self-reported community
  leaderboard**. Treat 95.5% as a *direction indicator*, never as a target or a baseline.
- Vendor's own numbers: three runs **[95.0, 95.2, 95.5]**, Best@3 **99.97%**, 183/183 levels [V, S15].
- Vendor caveat, in their words: *"we evaluated Opus 5 and GPT-5.6 Sol with Claude Code and Codex
  respectively, and found **worse** overall performance relative to the official results, so we yield to
  their official reported numbers instead."* [V, S15] — i.e. their own harness-vs-harness comparison at the
  same model was *unfavourable in the other direction*, which they did not explain.
- *"currently no model has been trained around Prime Agent or its core feature set"* [V, S15] → the harness
  is not model-coupled by construction.
- **The strongest evidence that the gain is harness-side, not Opus-5-side, is external to Prime Agent**: the
  earlier **Schema** harness (Impossible Research, ~07-16; executable-program world models; ~99% RHAE with
  **Opus 4.8**; 50 traces + scorer published on HuggingFace) reached the same regime with a *previous
  generation* model [per the 08-07/08-08 research sweep; S18 corroborates the site exists]. Three
  independent teams (Schema, Tycho, Prime Agent) converge on state-externalization. **That convergence —
  not Prime Agent's headline — is the evidence R24 should weigh.**
- **[INF]** What is plausibly frontier-model-specific: writing correct, self-debugging Python world models
  unattended for hours. Our in-kernel model is **Qwen3.6-27B-FP8** served by local vLLM
  (`duck_eval/taaf_bundle/src/ARC3-Inference/inference/framework/kaggle.py`, `DEFAULT_SERVED_MODEL_NAME =
  "vrfai/Qwen3.6-27B-FP8"`, `max-model-len 65536`) [V, S19]. Expect the *mechanism* to transfer and the
  *magnitude* not to.

---

## 3. Where our duck already is (read-only survey, S19)

Verified file references (no code was modified):

- **Env loop:** `duck_eval/taaf_bundle/src/ARC3-Inference/inference/framework/solver.py`,
  `_HarnessGameSession.play()` (~L263). Writes `tool_runtime_state.json` each iteration, then calls
  `analyzer.analyze(state_path, action_count, valid_actions, step_env=self.step_env, …)`.
- **LLM loop:** `.../inference/agent/tool_agent.py`, `ToolAgent.analyze()` (~L1783).
- **The duck already has the RLM shape.** `_tools()` (~L1333) exposes **exactly one** tool:
  `{"name": "python", "parameters": {"code": string}}`. And `step_env` is passed *down into the sandbox*, so
  the model calls `action([...])` **inside its own Python** and can execute a whole batched search in one
  turn. This is structurally Prime Agent's "the only tool is a Python REPL, and actions are function calls".
- **But the sandbox is not persistent.** `.../inference/agent/python_tool_sandbox.py`,
  `run_sandboxed_python(...)` (~L448) spawns a **fresh `subprocess.Popen` per tool call**, JSON line
  protocol, `RLIMIT_CPU` timeout+1, `RLIMIT_FSIZE` 1 MB, `RLIMIT_NOFILE` 32, `SAFE_BUILTINS`, and
  `SAFE_MODULES = (bisect, collections, copy, fractions, functools, heapq, itertools, json, math, operator,
  random, re, statistics, string)`. **Every variable the model builds is destroyed at the end of the call.**
- **Board is already externalized.** `agent/prompts.py::STRUCTURED_RUNTIME_STATE_ADDENDUM` says *"The raw
  numeric grid is intentionally not exposed. Use `current_frame.segmentation` as your primary view"*; the
  board is queried through Python, not pasted. Optional PNG via `MULTIMODAL_CONTEXT=current_grid` (ON by
  default on Kaggle, upscale 4).
- **Existing external state:** `self._summarized_knowledge` world-model digest (`world_model, goal_model,
  action_model, recent_findings, open_questions, current_plan, cross_level_notes`) — **wiped on level
  transition / game over**; `tool_runtime_state.json` history; our **Hypothesis Ledger**
  (`duck_eval/ledger/ledger_core.py`, per-game, `DIGEST_TOKEN_CAP = 600`); warpack `WarpackState`
  (`duck_eval/warpack/warpack_patch.py`, `trace: list[TraceStep]`, `banked`, `bank_max_replay_actions=1500`).
- **Context discipline today is selection-side.** `_trim_messages_for_context()` (~L1749) drops the oldest
  blocks until under `LOCAL_ANALYZER_CONTEXT_WINDOW − reply_reserve − 512`;
  `_persistent_history_messages()` keeps the last **30** assistant turns.
- **Budgets:** `max_actions_per_game = None` (uncapped in the scored regime); tool steps per turn unlimited
  on Kaggle (`LOCAL_ANALYZER_TOOL_STEPS="0"`); `LOCAL_ANALYZER_YIELD_SECONDS=60`; tool timeout 30 s; tool
  output 1024 tokens; solver concurrency 16; warpack soft end **11 h 20 m**
  (`duck_eval/warpack/fastsubmit_cells.py:140`).
- **Rails, confirmed:** `enable_internet: false` in every duck `kernel-metadata.json`; wheels installed
  `--no-index`; preflight H2 blocks any `arcprize.org` reference; all code rides in the attached dataset
  `canivel/arc-war-kit` staged at `duck_eval/warpack/_kaggle_dataset/`; notebook cell 12 is the single
  customization hook.

**Headline structural finding:** we are two mechanisms away from the Prime Agent shape, not ten. We already
have (a) Python-as-only-tool and (b) actions-as-function-calls-inside-that-Python. We lack (c) a **persistent
namespace** and (d) **durable cross-level harness state**. Both (c) and (d) are strictly additive.

---

## 4. Portability table

| # | Prime Agent component | Verdict | Reason, tied to a rail |
|---|---|---|---|
| P1 | **Persistent kernel namespace** (state survives across turns; fork-server + boot gate) | **ADAPT — top pick** | Purely additive (adds a store *outside* the window, removes nothing); the duck's one-tool sandbox already exists, we only stop killing the subprocess. Zero-budget, no internet, no notebook change (rides in `arc-war-kit`). |
| P2 | **Truncated kernel output** (8192 chars/turn) | **PORT** | Already have it (`LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS=1024`, ×4 chars ≈ 4096). Only a constant to re-tune; must be re-tuned *upward* if P1 lands, since the model will print digests deliberately. |
| P3 | **Durable memory `M` across levels** (write-once, survives level transitions) | **ADAPT** | Directly fixes a known duck defect: `_summarized_knowledge` is *wiped* on level transition. Restoring it as an append-only external store is the definition of additive. Ledger (`ledger_core.py`) is the existing carrier. |
| P4 | **Skills `K` as importable Python, surfaced by name+description, body lazy-loaded** | **ADAPT** | Costs ~1 line of context per skill and zero tokens until used; the sandbox already has an import mechanism (`SAFE_MODULES`). Requires widening the allowlist to one vendored `duck_skills` module — a byte-audited dataset file, so fork-never-build holds. |
| P5 | **Kernel state snapshot via `dill`** (crash recovery) | **ADAPT-lite** | Cheap insurance for P1 across the 11 h 20 m soft end and per-game restarts; `dill` availability inside the restricted sandbox must be checked. Not needed for the first screen. |
| P6 | **Generation-side compaction at the cut point** (`generateSummary()` where we currently drop) | **DEFER — R24 gate** | Technically *additive relative to our baseline* (baseline silently drops; this would add a summary). But it is compaction-lane-adjacent, and the sealed 08-06 disposition forbids any compaction push without an explicit R24 revival. Flag it; do not build it on this ticket. |
| P7 | **`/refine` self-editing harness (CRUD + revert-by-ID)** | **ADAPT-later** | Buildable within rails (edits land in files in our dataset dir, not in the notebook), but it needs many trajectories to pay off and it makes a run non-reproducible — poison for a single-seed screen. Also: the Factorio reward-hacking anecdote [V, S17] is a live warning for a scored benchmark. |
| P8 | **`rlm()` sub-agents as function calls** | **SKIP (this cycle)** | Our single local vLLM already serves 16 concurrent games (`solver.concurrency = 16`); sub-agents would contend for the same GPU inside a fixed 9–11 h wall clock. Cost is paid in the one resource we cannot buy. Revisit only if P1 shows headroom. |
| P9 | **Sub-agent persistence / 30-min idle unload / daemon-owned sessions / detach-reattach** | **SKIP** | Solves multi-day interactive CLI use. Our run is one non-interactive Kaggle process. No rail benefit. |
| P10 | **Session append-only JSONL + branching by leaf pointer** | **SKIP** | We already persist `tool_runtime_state.json` + warpack trace; branching has no analogue in a scored single-pass rerun. |
| P11 | **The Node/TypeScript runtime, `install.sh`, npm packages, MCP, TUI, `/login` provider auth** | **SKIP — hard blocker** | Prime Agent is a **TypeScript monorepo** requiring npm install and an authenticated provider. `enable_internet: false` + `--no-index` + no paid API. Nothing is vendorable; only concepts port, re-implemented in Python. |
| P12 | **ARC-specific configuration** | **N/A** | There is none to port — vendor says the only ARC change was the task prompt [V, S15]. Our PRO-LONG-adjacent prompt work is already in `agent/prompts.py`. |

---

## 5. Engineering lift for the top items

Unit = **one dataset-code push cycle** (edit `duck_eval/warpack/_kaggle_dataset/*` → `datasets version` push
→ pull-back byte audit → kernel build). S ≈ 1 cycle, M ≈ 1–2, L ≈ 3+.

| Item | Lift | Notes |
|---|---|---|
| **P1 persistent namespace** | **M** | The subprocess + JSON line protocol in `python_tool_sandbox.py` already exists; the work is lifecycle (keep the child alive per `_HarnessGameSession`, reap on game end), `RLIMIT_CPU` re-accounting (a CPU rlimit that was per-call becomes per-game — this is the real gotcha, it will kill a long-lived child silently), and a prompt line telling the model its variables persist. All inside `warpack_patch.py`-style monkeypatching → no notebook drift. |
| **P3 durable cross-level memory** | **S** | Stop the wipe in `_update_summarized_knowledge_from_step_summary`, route survivors into the existing per-game ledger store, re-inject as an *additional* prompt block. ~1 cycle. Can ship in the same push as P1 but should be a **separate arm** to keep attribution clean (we just paid for a confound in M3). |
| **P2 output cap re-tune** | **S** | One constant; bundle with P1, but pre-register the value so it is not a free parameter. |
| **P4 skills module** | **M** | Needs a vendored `duck_skills` package + `SAFE_MODULES` widening + prompt block. Do *after* P1 — skills are near-worthless without a persistent namespace to hold their state. Note arXiv:2608.04828 (Skill-Use): measure the duck's skill-trigger compliance **before** building the store. |
| **P5 dill snapshot** | **S** | Only if P1's screen passes and we see crash losses. |
| **P7 /refine** | **L** | Needs a refinement prompt, an edit schema, apply/rollback, and an outcome log — plus a reproducibility story. Not this month. |

Total to the first decision: **M (1–2 cycles)**, well inside the 0-cloud-dollar and 2-push-per-day budget.

---

## 6. Risks and unknowns

1. **Provenance risk (primary).** 95.5% is vendor-reported, unreplicated, and off the official leaderboard;
   official Opus 5 = 30.2%. If R24 anchors on 95.5% we will mis-size expectations by an order of magnitude.
   **Anchor instead on the three-team convergence** (Schema @ Opus 4.8, Tycho, Prime Agent) — that is what
   survives the provenance discount, and Schema specifically shows a *non-latest* model suffices.
2. **Model mismatch.** Their result is Opus 5 with unbounded API budget. Ours is Qwen3.6-27B-FP8, 65 K
   context, on a shared RTX PRO 6000. A 27 B model writing multi-hundred-line self-debugging world models
   unattended is the load-bearing unknown. **Mitigation:** the first experiment must test the *substrate*
   (does persistent state help at all?), not the ceiling.
3. **Metric mismatch.** Prime Agent optimizes RHAE — `(human/agent actions)²` capped 1.15 [V-2nd]. Our Kaggle
   score is a different aggregate (our best 1.33; gold cutoff 1.56; leader 1.86). **[INF]** if Kaggle's
   scoring is also action-efficiency-weighted, then P1 is doubly attractive (in-kernel search costs no scored
   actions); if it is completion-weighted only, the action-efficiency half of the story is worth nothing to
   us. *This should be checked against the competition metric doc before the screen is designed* — it changes
   which canary matters.
4. **Runtime/rail blockers.** Node + npm + provider `/login` are all unavailable (`enable_internet: false`).
   Nothing is vendorable; every port is a Python re-implementation. Corollary: **we cannot validate against
   their traces**, only against our own baseline.
5. **Persistent-process hazards (the real P1 risk).** A long-lived sandbox child changes the failure surface:
   `RLIMIT_CPU`/`RLIMIT_FSIZE` become per-game not per-call; a wedged child now blocks a whole game instead
   of one turn; orphan processes across 16 concurrent workers; memory growth over an 11 h run. **Mandatory
   canaries:** child-alive counter, per-game reap confirmation, orphan check at teardown, hard fallback to
   ephemeral-per-call on any child fault.
6. **Additivity must be *proved*, not asserted.** A22 taught us that. The screen needs an arm-defining
   invariant analogous to `digest_tokens=0 AND reserve_applied=0`: here, **`evicted_chars` and trim behaviour
   must be byte-identical to baseline** — P1 adds a store, it must not touch `_trim_messages_for_context`.
7. **Skill/compliance risk.** Prime Agent's skills work because Opus 5 reliably triggers them. arXiv:2608.04828
   documents that agents frequently fail to invoke procedural skills at all — a 27 B model may simply never
   use a persistent namespace it is told about. **This is the most likely null mode** and the screen should
   instrument it directly (count turns that read a variable written in a *previous* turn).
8. **Reward hacking.** `/refine` produced efficient cheating in Factorio [V, S17]. Any self-editing lane on a
   scored competition needs a prereg'd integrity check. Another reason P7 is last.

---

## 7. Recommendation for the R24 panel

**Yes — candidate (a) is buildable on this blueprint within our rails, but the blueprint that should be
adopted is the *convergent* one (Schema + Tycho + Prime Agent), not Prime Agent's headline number.** The
95.5% is vendor-reported, unreplicated and excluded from the official leaderboard (where Opus 5 sits at
30.2%); the load-bearing evidence is that three independent teams — including one using a *previous-generation*
model — reached the same regime by moving state out of the context window into an executable substrate.
Nothing about Prime Agent's ARC configuration is portable (there isn't any: the vendor changed only the task
prompt), and nothing about its implementation is portable either (TypeScript, npm, authenticated providers —
all barred by `enable_internet: false`). What *is* portable is the substrate discipline, and here the survey
result is unexpectedly favourable: **the duck is already two mechanisms short of the RLM shape, not ten.** It
already exposes exactly one tool (`python`) and already passes `step_env` into the sandbox so actions are
function calls inside model-written code. The two missing pieces — a **persistent kernel namespace** (state
survives across turns instead of dying with each `subprocess.Popen`) and **durable cross-level memory**
(today `_summarized_knowledge` is *wiped* at every level transition) — are both strictly additive: they add
a store outside the window and remove nothing from it, which is precisely the direction the A22 death
prescribed. Note also, as corroboration rather than as a revival argument, that Prime Agent's own compaction
is **generation-side** (`generateSummary()` at a safe cut point), never eviction — the same class distinction
arXiv:2608.01326 proves and our −0.20/−0.32/−0.36 monotonic harm measured.

**Minimal first experiment (single seed-1 screen, ~1–2 push cycles, $0):**
*Arm P1 — "persistent scratchpad".* Patch only `python_tool_sandbox.run_sandboxed_python` (via
`arc-war-kit`, no notebook change) so the sandbox child **lives for the duration of a
`_HarnessGameSession`** instead of one tool call, and add one system-prompt sentence: *variables you define
persist across turns; use them to hold your world model.* Nothing is removed from context; the history
trimmer is untouched.
- **Arm-defining invariant (the `digest_tokens=0` analogue):** `evicted_chars` and the trimmed-message
  sequence must be **byte-identical to baseline** on every event. If context handling moves at all, the run
  is void.
- **Adoption canary (tests risk #7, the most likely null):** fraction of turns that reference a name defined
  in a *previous* turn — `namespace_reuse_rate`. **If this is ≈0, the lane is answered NULL immediately and
  cheaply, without needing the score at all.** Pre-register a floor (suggest ≥0.15) below which the result
  is "27 B does not use the substrate" rather than "state-externalization does not work".
- **Safety canaries:** live-child count ≤ concurrency (16), zero orphans at teardown, zero per-game
  `RLIMIT_CPU` kills, automatic fallback to ephemeral-per-call on child fault (and a count of fallbacks).
- **Primary outcome:** mean Δlc vs the war-eval baseline under the same K3-style non-harm gate used for A22.
- **Decision rule:** `namespace_reuse_rate` below floor → NULL, drop the lane and reallocate to candidate (b)
  additive typed memory. Reuse above floor **and** Δlc non-harm → proceed to P3 (durable cross-level memory)
  as a separate arm — separate, because the M3 confound already cost us one lane's worth of attribution.
- **Pre-screen homework (cheap, do first):** confirm whether the Kaggle metric is action-efficiency-weighted
  (risk #3). If it is, add `actions_per_level` as a co-primary; if not, drop the efficiency framing entirely
  and judge P1 purely on completions.

Sequencing for the lane if the screen passes: **P1 → P3 → P4 (skills, gated on a Skill-Use compliance
pre-check) → P5**. P6 (generation-side compaction) and P7 (`/refine`) stay parked behind their own explicit
R24-class decisions.
