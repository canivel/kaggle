# Harness diff vs the public ≥1.40 band — 2026-08-13

**Order.** The census (`what_the_field_runs_2026-08-13.md`) proved 52 teams span 0.00→1.62 on the
brain we already run, so harness + agent policy are the whole public variance. Pull every public
kernel from a team at ≥1.40, diff it against what we ship, rank the deltas, build.

**Method (read-only; zero pushes, zero submissions, zero spend).**
- `kaggle competitions leaderboard -d` → 2,265-team CSV, snapshot **2026-08-13T01:46:26Z**.
  **60 teams sit at ≥1.40** (the census table only listed the 7 in the top 46; the band is wider).
- Extracted **91 distinct usernames** from those 60 teams → `kaggle kernels list --user <u>` for all 91
  (serial, 3 retries, `PYTHONIOENCODING=utf-8` — the CLI silently emits a header-only CSV and dies on
  a `charmap` error for any user with an emoji/CJK kernel title, which is why an earlier parallel sweep
  showed "no kernels" for boristown/zoli800/romantamrazov. **Operational gotcha worth keeping.**)
- `kaggle kernels pull -m` on every ARC-AGI-3 hit (14 kernels), plus `kaggle datasets download` on the
  four TAAF source bundles the thin notebooks actually load their solver from.
- Local baseline: `notebooks/duckwar/arc3-duck-war.ipynb`, our live head `canivel/arc3-duck-repro`
  (pulled fresh), `duck_eval/taaf_bundle/src/ARC3-Inference/**`.
- Artifacts: `runs/harness_diff_0813/` (kernels + extracted `.py` + `ds/` bundles + LB CSV).

Provenance: **[V]** verified by direct read · **[INF]** inference · **[UNK]** unknown.

---

## 0. HEADLINE — the answer key is blank because it is our own paper

**Every public kernel in the 1.44–1.62 band is the stock Tufa duck harness. Our shipped kernel is
byte-identical to it.** [V]

`canivel/arc3-duck-repro` (our live head, submitted 08-13 00:07Z) vs
`jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner` (Tufa Labs #7, 1.62): **17 cells vs 17
cells, zero code differences.** The only diffs are three markdown cells and one `print()` string where
an em-dash round-tripped to `â€"` through cp1252. `dataset_sources`, `docker_image` sha,
`machine_shape`, `enable_gpu/internet` — all identical. [V]

And the two cleanest peers are *self-declared* stock:

- **暗黑AGI (boristown, #32, 1.47, 29 submissions)** ships `agi-duck-harness-fast-eval` whose two patch
  cells exist only to say the patches are **off**:
  `BASELINE_RUNTIME_STATUS = {"mode": "dataset-baseline", "runtime_monkey_patch": False,
  "action_mapping_changed": False, "reset_behavior_changed": False, "solver_methods_changed": False,
  "tool_result_schema_changed": False, "prompt_changed": False, "dataset_modified": False}` and
  `BASELINE_CUSTOMIZATION_STATUS = {... "local_budget_changed": False, "model_facing_state_changed": False}`.
  Their markdown: *"its score-sensitive animation and prompt changes are neutralized here before the
  benchmark is restored."* Their **only** functional addition is a `wait_vllm_ready()` poll on
  `127.0.0.1:1234/v1/models` before the run. [V]
- **zoli800 (#46, 1.44, 63 submissions)** and **caoyupeng's `1-21-from-great-team-tufa-labs` (#37)**
  are the stock share notebook with the header markdown replaced. `14,493` extracted bytes, identical
  to Tufa's. [V]

So the premise "the public kernels are the answer key" is **falsified**. The three genuine code deltas
in the whole band are Helmut AGI's animation bundle, Tufa's own non-share bundle, and one graft stack;
they are catalogued in §2 and none of them is convergent.

**What replaces the premise** is in §3: the 1.33→1.62 gap is almost entirely **best-of-k order
statistics on a common, right-skewed distribution**, and the one lever the public record does *not*
contain is the one our own 08-12 diagnosis found — **action efficiency**.

---

## 1. INVENTORY — every public ARC-AGI-3 kernel owned by a ≥1.40 team

Complete as of the 08-13T01:46Z LB snapshot. 60 teams / 91 users swept.

| LB | rank | team | kernel | what it actually is | pulled |
|---:|---:|---|---|---|:--:|
| **1.62** | 7 | Tufa Labs | `jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner` | **stock**; bundle `taaf-kaggle-source-share` | ✔ |
| **1.62** | 7 | Tufa Labs | `jeroencottaar/taaf-duck-harness-kaggle` | **552-byte stub, `enable_gpu:false`** — not a scoring artifact; but its bundle `taaf-kaggle-source` is Tufa's *internal* tree (§2) | ✔ |
| **1.61** | 9 | FOYSAL | `foysalemonshanto/ash-s-arc-agi-3-agent-d83ab9` | **2026-05-04, T4, no datasets, 5 cells** — pre-duck starter agent. Not a harness peer. FOYSAL has **no** duck kernel. | ✔ |
| **1.61** | 11 | Helmut AGI | `jakobbrggen/taaf-anim-arc-agi-3-solver` | thin TAAF notebook + bundle `taaf-kaggle-source-anim-20260807-anim` (**animation + hard no-op guard**) | ✔ |
| **1.51** | 22 | The AGI Boys | `romantamrazov/arc-real-agi-solution` | **independent lineage**: Gemma-4-31B-IT, `agents.Agent` swarm template, vision-first | ✔ |
| **1.47** | 32 | 暗黑AGI | `boristown/agi-duck-harness-fast-eval` | **stock, patches explicitly rolled back** + vLLM readiness poll | ✔ |
| **1.47** | 32 | 暗黑AGI | `boristown/taaf-duck-harness-kaggle` | **stock** | ✔ |
| **1.46** | 37 | Tara Labs | `caoyupeng/arc3-duck-v12-1d7d88` | stock + `taaf_grafts` from `thtennant/taaf-kaggle-source-share-fork` (**efficiency / retry_guard / shortcircuit**) + 11h20m submission cap | ✔ |
| **1.46** | 37 | Tara Labs | `caoyupeng/1-21-from-great-team-tufa-labs` | **stock** | ✔ |
| **1.44** | 46 | I forgot the name.. | `zoli800/taaf-duck-harness-kaggle-share-resubmission-573a60` | **stock** | ✔ |

Checked and discarded as non-ARC-AGI-3 (regex false positives on "agi"/"anim"/"arc"):
`johnjanson/hahaha-nondet-agi` (223 votes) and `boristown/top-reproducible-pf-config-branch-conservative`
are both the **ROGII wellbore-geology** competition (identical 57-cell notebook, `rogii-*` datasets);
`boristown/lb0-897-with-full-visual-pipeline-and-animation` is **Biohub cell tracking**;
`baidalinadilzhan`/`hvanphucs112` CompressARC are **ARC-AGI-1**; `ljs2546` is **ARC-AGI-2**. [V]

**Answer to "are there ≥3 usable ≥1.40 kernels": yes — 8 duck-family kernels across 6 teams, plus one
independent-lineage kernel. No widening to ≥1.30 needed.** But 6 of the 8 are stock, so the *effective*
sample of differentiated harnesses at ≥1.40 is **three**: Helmut (1.61), Tara Labs (1.46), AGI Boys (1.51).

**Version history:** the CLI cannot pull a specific kernel version (`kernels pull` always returns head),
so every row is the head as of 08-13. Head `lastRunTime` vs the team's LB motion is the only available
join and it is weak — Tufa's 1.62 came from **104** submissions of which the published notebook is one.
This caveat rides on every row.

---

## 2. SUBSYSTEM DELTA TABLE

Baseline column = what we ship (stock `taaf-kaggle-source-share`, verified byte-identical to our head).

### (a) SYSTEM PROMPT / instructions to the model — where policy lives

Our whole prompt is `inference/agent/prompts.py`, **114 lines**, six addenda.

| # | delta | who | evidence |
|---|---|---|---|
| **a1** | **Efficiency pressure is one unquantified sentence.** Ours, verbatim and in full: `"- Optimize for as few in-game actions as possible while still being reliable.\n"` (prompts.py:17) plus `"- Optimize for the shortest reliable sequence…"` (:87). **The scoring rule is never stated to the model. No action count, no target, no waste feedback, ever.** | — | [V] |
| **a1′** | Tara Labs (1.46) injects a **quantified per-turn budget note**: `"EFFICIENCY BUDGET — your score on each level is (baseline_actions / your_actions)^2, so every wasted action costs you quadratically."` + `"{lvl}: you have used {used} actions; a strong score needs about {target} or fewer. You are {ratio:.1f}x over the target."` + STALL / NET-ZERO WASTE / REVISIT WASTE lines + `"commit to your single best hypothesis and the shortest sequence that tests it — do not exhaustively scan rows/columns or enumerate every option."` | Tara Labs 1.46 (via thtennant fork) | [V] `taaf_grafts/agent_ext.py:219-330` |
| **a2** | **Anti-oscillation instruction.** Ours: absent. AGI Boys (1.51): `"Prefer purposeful new states. A repeated state is not progress."` and `"If Recent transitions show repeated_state=true or state_changed=false, stop following that same direction/coordinate. Try a different axis, activation, or a new click target instead of oscillating."` | AGI Boys 1.51 | [V] |
| **a3** | **State-keyed dead-action list in the prompt.** Ours: absent. AGI Boys: `"Ineffective in this exact state: {ineffective_actions}"` from `failed_state_actions: dict[str, set[str]]`. Helmut implements the same idea **as a hard block**, not a hint (see c/d). | AGI Boys 1.51 + Helmut 1.61 (different mechanism) | [V] |
| **a4** | **Animation semantics** (+1,554 bytes on prompts.py). Helmut adds 7 lines incl. `"animation['board_unchanged'] together with board_changed == False does NOT mean the action did nothing… Treat it as a real outcome and read the transient region."` | Helmut 1.61 only | [V] |
| **a5** | Reflection-memory block (`"Reflection memory (authoritative but revisable): {…}"`, refreshed every 24 steps, persisted per game to disk, ≤1,800 chars). Ours: no cross-turn distilled memory of any kind. | AGI Boys 1.51 only | [V] |

### (b) Tool surface

| | ours (= 1.44/1.46/1.47/1.62 stock) | Helmut 1.61 | AGI Boys 1.51 |
|---|---|---|---|
| tools | exactly one: `python` | `python` + **`animation()`** (diff timeline / `frame=k` / `action_num=n`, costs no action budget) | none — raw JSON action payload, `response_format={"type":"json_object"}` + a repair round-trip |
| python globals | `current_frame`, `previous_frame`, `history`, `transitions`, `last_transition`, `valid_actions`, `last_action_result`, `action(actions)` | + animation record | n/a |
| raw grid | **withheld** — `"The raw numeric grid is intentionally not exposed"`; `.segmentation` is the primary view | same | n/a (image-first) |
| tool output cap | `LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS=1024`, 30 s hard limit | same | `MAX_NEW_TOKENS=512`, `REPAIR_MAX_NEW_TOKENS=160` |

### (c) Context management

| | value | source |
|---|---|---|
| ours | `LOCAL_ANALYZER_CONTEXT_WINDOW=32768` (against `--max-model-len 65536`), history eviction hardcoded `_PERSISTENT_HISTORY_ASSISTANT_TURNS = 30` | [V] |
| **Tufa's own non-share tree** | that constant is **deleted** and replaced by `_LOCAL_ANALYZER_HISTORY_MAX_TURNS = _get_env_int("LOCAL_ANALYZER_HISTORY_MAX_TURNS", 0)` + `_LOCAL_ANALYZER_EVICT_TURNS = _get_env_int("LOCAL_ANALYZER_EVICT_TURNS", 1)`, wired in `setup_commands.json` as `'LOCAL_ANALYZER_HISTORY_MAX_TURNS': '0'`, `'LOCAL_ANALYZER_EVICT_TURNS': '2'` | [V] |
| Helmut / Tara / boristown / zoli800 | identical to ours | [V] |
| AGI Boys | `MAX_HISTORY=12`, `MAX_FRAME_MEMORY=11`, last-4 transitions in prompt, reflection distilled every 24 steps | [V] |

Also in Tufa's non-share tree and **not** in ours: `runtime_state.py` swaps the JSON-file round-trip for
an in-process `_CACHE`, and `_frame_payload()` sends `grid_b64` (base64 of raw bytes) instead of
`{"ascii": …, "grid": [[…]]}`. Both are pure plumbing/throughput, not policy.

### (d) Action policy

| | ours | Helmut 1.61 | Tara Labs 1.46 | AGI Boys 1.51 |
|---|---|---|---|---|
| batch handling | one atomic `step_env({"actions": […]})` per `action()` call | **walks the batch one action at a time** so each can be individually blocked; re-aggregated by `_aggregate_action_batch_result` (`frame_count` = max not sum, rewards summed) | same as ours + `shortcircuit` (stops homogeneous no-op batches) | plan ≤ `MAX_PLAN_ACTIONS=3`, dequeued one/turn, invalidated by `_should_replan_after_transition()` |
| repeat-no-op | nothing | **hard block**: `NoopGuard` keyed `(level, blake2b(board_before), action_sig)`, bounded 512 states/level × 16 actions/state, **exempting animated actions**; returns a synthetic `stop_reason:"known_noop"`, `"blocked before execution, no action budget spent"` | report-only counters (`retry_guard`) | prompt-level `ineffective_actions` |
| retries | none | none | RetryGuard analyzer layer (outermost) | `LLM_REQUEST_TIMEOUT_S=120` + `llm_cooldown_steps` fallback |
| per-game caps | `max_actions_per_game = None`, `max_runtime_s_per_game = 7920.0`, `concurrency = 28` | identical | identical | `MAX_ACTIONS=200`, `GAME_TIME_LIMIT_S=8h`, `FIRST_ACTION_DEADLINE_S=14min` |
| submission wall clock | **`soft_end = None` in a real rerun** — the wall budget is emergent | **also `None`** (`_soft_end_time()` returns `None` when `run_as_submission`) | **capped**: `soft_end = start + timedelta(hours=11, minutes=20)`, *"so the solver drains and the shared scorecard closes before Kaggle's hard kill"* | global 9h − 20min reserve |

`max_runtime_s_per_game = 7920.0` with `concurrency = 28` and 25 games is **identical in all four
bundles** [V, read out of `benchmark_initial.pkl` via `pickletools.dis`] — the whole benchmark is
designed to finish in ≈2.2 h of an 11–12 h allowance. Nobody in the band has touched it.

### (e) Perception

| | ours (= whole duck band) | AGI Boys 1.51 |
|---|---|---|
| primary view | `segmentation` object graph (nodes with `id/color/hash/pixels/boundary/children` + `adjacency_list`); ASCII only for small crops; **raw grid deliberately hidden** | rendered PNG images, 16-colour ARC palette |
| image | 1 image/turn, `MULTIMODAL_CONTEXT=current_grid`, `MULTIMODAL_UPSCALE=4` (64→256 px) | **3-frame chronology** (`ACTION_CONTEXT_FRAMES=3`) at `FRAME_IMAGE_SCALE=8` (64→**512 px**), red STEP labels drawn on, `FRAME_BORDER_IGNORE=3` |
| animation frames | discarded (`taaf/game.py` returns `raw.frame[-1]`) | n/a |

The 512-px upscale is exactly VISTA's setting, which Xuan reported on discussion 734369 *"just does
not work as well" for Qwen* — and our own 08-11 sweep already demoted the VLM lane. Do not port (e).

### (f) vLLM serve config

Identical across **all four** duck bundles [V, `setup_commands.json` diff]:
```
--model <path> --served-model-name vrfai/Qwen3.6-27B-FP8 --tensor-parallel-size 1
--enable-auto-tool-choice --tool-call-parser qwen3_coder --generation-config vllm
--enable-prefix-caching --default-chat-template-kwargs '{"preserve_thinking": true}'
--reasoning-parser qwen3 --max-model-len 65536
LOCAL_ANALYZER_TEMPERATURE=0.6  TOP_P=0.95  TOP_K=20  ENABLE_THINKING=true
CONTEXT_WINDOW=32768  MAX_OUTPUT=0  TOOL_STEPS=0  TOOL_TIMEOUT=30
TOOL_OUTPUT_TOKENS=1024  YIELD_SECONDS=60  MULTIMODAL_CONTEXT=current_grid  MULTIMODAL_UPSCALE=4
```
The **only** delta anywhere is Tufa's non-share tree adding `LOCAL_ANALYZER_HISTORY_MAX_TURNS=0` /
`LOCAL_ANALYZER_EVICT_TURNS=2`. No `--gpu-memory-utilization`, no `--max-num-seqs`, no
`--kv-cache-dtype` anywhere in the band. (AGI Boys, different stack: `--max-num-seqs 20
--gpu-memory-utilization 0.94 --max-model-len 32768 --trust-remote-code`, `repetition_penalty 1.08`,
temperature `0.6` thinking / `0.2` non-thinking, JSON mode on.)

### (g) Structural things we simply do not have

1. **`animation()` retrieval tool + animation metadata** (Helmut). We built our own version and
   **KILLED it** on 08-12 (K-A3, 3.53% token fraction vs a sealed <1% — a bound we later judged
   mis-specified against a generated-token denominator). Brüggen's own public A/B was **NULL
   (+1.4%, p=0.92, +17% tokens/action, 2/181 tool calls informative)**. Not a gap worth reopening.
2. **Hard `NoopGuard`** (Helmut). We have never shipped the hard-block variant; our P1 arm ships the
   *soft* variant with `P1_MAX_DECLINES=1` and is currently **running and unread**.
3. **Quantified efficiency note** (Tara Labs). §4 item 1.
4. **Env-tunable history eviction** (Tufa non-share).
5. **Per-game persisted reflection memory** (AGI Boys).
6. **Submission-mode wall-clock cap** (Tara Labs). We have the code in `duckwar` (`R1b`, 11h20m) but
   our **shipped** head is the frozen fork, which runs `soft_end=None`.

---

## 3. CONVERGENCE ANALYSIS — and the number that reframes the task

### 3.1 Convergence is weak-to-absent

| delta | in ≥2 high scorers? | contradicted? | verdict |
|---|---|---|---|
| "do not oscillate / a repeated state is not progress" | **YES** — AGI Boys (prompt) + Helmut (hard block) + Tara Labs (NET-ZERO/REVISIT lines) — **three independent lineages, three independent implementations** | our own Mechanism C measured **DEAD** (arm 7.8% dead-reissue sits *inside* the 5.3–23.1% block-free control spread; the within-run fall is regression to the mean) | **convergent in intent, negative in our own measurement** |
| quantified action budget in-prompt | 1 (Tara Labs) | nobody contradicts; author of the graft (thtennant) is at 1.28 | **single-source, but uniquely aligned with our own binding constraint** |
| animation awareness | 1 (Helmut) | Brüggen's own A/B NULL; our arm KILLED; Xuan reports context blow-up on Qwen | **contradicted** |
| hard no-op block | 1 (Helmut) | needs the animation arm underneath or it hard-blocks working actions on ft09/sb26 | **coupled to a contradicted arm** |
| submission wall-clock cap | 1 (Tara Labs) | Helmut + Tufa + boristown + zoli800 all run `soft_end=None` at 1.44–1.62 | **not convergent; cheap insurance only** |
| env-tunable history eviction | 1 (Tufa non-share) | the *share* tree with the hardcoded 30 is what scored 1.62/1.47/1.46/1.44 | **no evidence of gain** |
| bigger/more images | 1 (AGI Boys) | ceiling 1.51 < 1.62; Xuan same-model negative; our 08-11 demote | **do not port** |

**There is no convergent set of the usual kind.** The strongest convergent signal is a *belief* — three
teams independently decided the agent wastes actions by re-entering states — implemented three
different ways, and the one way we have measured (naming dead actions in context) did not work.

### 3.2 The 1.33→1.62 gap is order statistics, and the LB proves it

Since the code is the same, the gap must be draws. That is testable, and the LB is the test set. Take
every team with **k ≤ 1 submission** and **score ≥ 0.60** (i.e. one draw of a duck-class harness,
excluding the 0.0–0.2 starter-agent mass):

| population | n | mean | sd | median | max |
|---|---:|---:|---:|---:|---:|
| field, k≤1, s≥0.60 | **85** | **0.923** | **0.159** | 0.90 | 1.32 |
| field, k≤3, s≥0.60 | 189 | 0.954 | 0.175 | 0.94 | 1.64 |
| **our duck ledger** (`runs/ledger.json`) | **29** | **0.9503** | **0.1513** | — | **1.33** |

**Our per-draw distribution is statistically indistinguishable from the field's, and marginally above
its single-draw mean.** [V]

Empirical tail from the same population: `P(draw ≥ 1.4) ≈ 0.016`. Then

| k | P(max ≥ 1.44) | who |
|---:|---:|---|
| 28–29 | **0.37** | boristown 1.47 @ 29 · Tara Labs 1.46 @ 28 · **us: 29 duck draws, max 1.33** |
| 55 | 0.59 | Helmut 1.61 @ 55 |
| 63 | 0.64 | zoli800 1.44 @ 63 |
| 104 | 0.81 | Tufa 1.62 @ 104 |

Every peer's number is a routine outcome of their draw count. **Ours is a ~63%-likely non-event, not a
capability deficit.** Corroboration from the low-k tail of the board: `Nilesh Sarkar 1.47 in 2
submissions`, `common-people 1.45 in 2`, `BambooCopter 1.64 in 3`, `Souhardya 1.49 in 4`,
`Andy liu 1.69 in 7`. Nobody builds a 1.6 harness in two submissions.

**Consequence for the ranking below.** Any item must be justified on **per-draw mean**, not on closing a
gap that is not there. This is the same conclusion R25's ρ̂≈0 pivot reached, arrived at independently
from the public record.

---

## 4. RANKED BUILD PLAN

Ranking = convergence × expected effect on **per-draw mean** × (1/build cost).
**Sequencing constraint:** `canivel/arc3-duck-p1-eval` is RUNNING and UNREAD. Per the 08-12 handoff,
nothing new gets a push slot until P1 is pulled and scored under the sealed prereg + addendum A1.
Items below are queued behind that, not instead of it.

### ★ #1 — QUANTIFIED PER-TURN EFFICIENCY NOTE (port `EfficiencyToolAgent`, report-only)

**Mechanism.** Append to the *user* turn, only when there is something to say: the scoring rule
(`(baseline/actions)^2`), the running action count for this level against a target, the over-target
ratio, and any firing stall detector. All detectors are **pure functions over the frame history the
agent already has** — `detect_net_zero_cycle` (shortest ≥6-action round-trip back to an exact prior
same-level grid, with a divergence requirement so a static board is not flagged),
`detect_stagnation` (≥8 consecutive same-level actions leaving the grid byte-identical),
`count_recent_revisits` (≥4 exact recurrences of the current grid). Zero LLM calls, zero GPU, the hot
`step_env` path is never touched.

**Why it plausibly moves the per-draw mean.** This is the only public implementation that targets the
constraint our own 08-12 diagnosis identified as binding:

> our animation run cleared **17 levels for 1.635**; the *same 17 levels* re-scored at exactly the
> human action baseline = **2.549 local ≈ 1.48–1.58 LB — the gold line, with zero new capability.**
> Six games burn **56%** of achievable score (bp35 **8.33×** human actions, ar25 5.97×, sp80 5.77×,
> m0r0 4.6×, tu93 3.68×, vc33 3.0×), and `level_score=(baseline/actions)²` makes it quadratic.

And the stock prompt is *silent* on it — `prompts.py:17` is the entire treatment, an unquantified
adverb with no feedback loop. Every other lever in this document tries to clear **more levels**; this
one tries to make the levels we **already clear** worth 2–4× more. It moves the mean of every draw, not
the tail of one. It also converges with Retrodict (5.5× token reduction → public SOTA 99.86) and
Brüggen (*"tokens are the real currency"*).

**Evidence class — stated honestly.** Present in **one** ≥1.40 kernel (`caoyupeng/arc3-duck-v12-1d7d88`,
Tara Labs #37 @ 1.46, `install(bm, flags={"efficiency": True, "retry_guard": True, "shortcircuit": True})`).
The graft's *author*, thtennant, is at **1.28**. **This carries no efficacy evidence whatsoever.** It
ranks #1 on mechanism-to-diagnosis fit, not on the public record.

**Build cost: LOW.** ~350 lines of pure Python already written, MIT-adjacent public dataset, four pure
functions that are unit-testable with no LLM and no GPU. Ports into the existing warpack seam
(`duck_eval/warpack/_kaggle_dataset/`) as one more `ToolAgent` subclass. Estimate: one working day
including smoke tests, one kernel build.

**Canaries (pre-register before writing a patch line).**
- **K-E0 delivery:** the note appears on ≥80% of turns where a stall detector fires or the level is
  over target; note length **≤700 chars** (Tara's builder is bounded by construction; our analogous
  Mechanism C measured mean 339 / max 599 against a 900 bound).
- **K-E1 detector sanity:** `net_zero`, `stagnation`, `revisit` each fire on ≥3 distinct games; zero
  fires on a game whose grid never repeats.
- **K-E2 non-harm:** `levels_completed` ≥ the minimum of the three block-free control runs
  (`animation_v1`, `a22_v2_seed1`, `a22_compaction_v1`).
- **K-E3 cost:** bound **statically in characters**, *not* as a token fraction. The note is an
  **input**-token cost and the rail reports generated tokens only — that exact denominator mismatch
  is what killed the animation arm (K-A3) and is what addendum A1 had to rule on for P1. Do not repeat it.

**Primary endpoint (M0).** Median **actions per cleared level**, per game, arm vs the three block-free
control runs. Secondary: RHAE recomputed by `duck_eval/warpack/*_score.py`.

**Kill rule.** (i) K-E2 fails — any trade of levels for efficiency kills it outright; (ii) M0 shows no
reduction against the control **spread** (not against the arm's own first half — Mechanism C's
first→second-half collapse was regression to the mean and reversed on one control; **compute the
control-side statistic first, before reading the arm**); (iii) any detector fires on >40% of turns
(nagging ⇒ ignored).

**Legality:** clean. Detectors are game-agnostic and read only the agent's own frame history.

---

### #2 — SUBMISSION-MODE WALL-CLOCK CAP (`soft_end = start + 11h20m`)

**Mechanism.** Our shipped frozen fork passes `soft_end_time=None` in a real rerun; the wall budget is
emergent and a Kaggle hard kill at 12 h zeroes the whole run. Tara Labs (1.46) caps it:
`soft_end = datetime.fromtimestamp(NOTEBOOK_START_EPOCH) + timedelta(hours=11, minutes=20)`,
commented *"so the solver drains and the shared scorecard closes before Kaggle's hard kill"*.

**Why it plausibly moves the mean.** It cannot raise a good draw; it truncates the **left tail**. Our
ledger's floor draws are `0.65, 0.68, 0.77, 0.78, 0.82, 0.82` — if any of those are drain failures
rather than play failures, this recovers them for free. Given `max_runtime_s_per_game=7920` ×
`concurrency=28`, the nominal run is ~2.2 h, so the cap should be inert in the normal case and only
bind when something hangs. **That inertness is also the reason to expect little.**

**Evidence class.** One of five. Helmut (1.61), Tufa (1.62), boristown (1.47) and zoli800 (1.44) all run
`soft_end=None` and score fine — so this is **not convergent** and the failure mode may not exist.
Our own `duckwar` R1b already contains the identical patch; it has simply never been on the shipped head.

**Build cost: TRIVIAL** (one cell-14 line, code already written and reviewed).
**Canary:** log `soft_end_time` and the actual run duration; **kill** if the cap ever binds in a normal
run (that would mean it is truncating play, not draining).

---

### #3 — ENV-TUNABLE HISTORY EVICTION (`LOCAL_ANALYZER_HISTORY_MAX_TURNS` / `_EVICT_TURNS`)

**Mechanism.** Replace the hardcoded `_PERSISTENT_HISTORY_ASSISTANT_TURNS = 30` with Tufa's own
env-driven pair and expose it as a knob. Tufa's internal default is `MAX_TURNS=0` (unbounded, trim on
overflow) with `EVICT_TURNS=2` (evict two at a time instead of one).

**Why it might move the mean.** With `CONTEXT_WINDOW=32768` against `--max-model-len 65536`, eviction
policy sets how much of the run's own history the agent can see and how often a context-overflow retrim
burns a request. `EVICT_TURNS=2` halves the number of overflow round-trips at the cost of dropping more
history each time. It is the only knob Tufa themselves changed between their public and internal trees.

**Evidence class.** One source, and the *weaker* direction of the evidence: the tree with the
**hardcoded 30** is precisely what produced 1.62, 1.47, 1.46 and 1.44. Tufa's non-share tree carries no
attributable score at all (its kernel is a 552-byte CPU stub).

**Build cost: LOW-MEDIUM** (a module-global patch through the same seam `taaf_grafts` uses for
`context_window`; note the landmine documented there — `LOCAL_ANALYZER_*` are frozen as module globals
at first import during the cell-10 unpickle, so **the env var is too late; you must reassign the
module global in cell 12**).
**Canary:** count context-overflow retrims per game. **Kill** if turns-per-game falls or retrims rise.

---

### Explicitly NOT recommended

- **Animation awareness / `animation()` tool.** Three independent negatives: Brüggen's own A/B (NULL,
  +1.4%, p=0.92, +17% tokens/action, 2/181 informative), our K-A3 kill, Xuan's *"blows up the context
  window for Qwen"*. Our M0 also found ft09 produced **zero** multi-frame responses under a real agent
  against 80.7% under a random-walk probe.
- **Hard `NoopGuard`.** Strictly downstream of the above (Helmut's own code exempts animated actions
  *because* recording them as no-ops "hard-blocked actions that had clearly worked, on exactly the
  games with the most animations"). Also overlaps the running P1 arm; do not run two suppressors.
- **Vision escalation (3 frames @ 512 px).** Ceiling 1.51 < the ASCII/segmentation band's 1.62, plus a
  same-model public negative and our own 08-11 demote.
- **Chasing the max with more identical draws.** §3.2 says this works (P rises from 0.37 to 0.81 going
  29→104 draws) and it is exactly the public-LB luck-chasing `feedback_arc_generalization_first`
  forbids. The private set has more games; the mean is the currency.

---

## 5. LEGALITY / DISQUALIFIER FLAGS — read before copying anything

1. **`caoyupeng`'s duplicate-game gate — DO NOT PORT INTO A SUBMISSION PATH.** It injects
   `taaf.game_api.GameAPI(env_name=first.env_name, arcade_spec=first.arcade_spec,
   external_game_id=f"{first.env_name}-dup")` so `games[0]` "scouts and publishes" and the dup
   "replays" through the transfer family store. It is guarded on `if not TRUE_SUBMISSION:` and exists
   only to make an interactive commit run exercise the transfer path. **Replaying the same environment
   under a second game_id inside a real rerun would be scorecard manipulation.** Keep the guard or
   leave it out entirely.
2. **Baseline lookup — heuristic proxy only, never a per-game table.** `agent_ext._load_baselines_from_metadata`
   rglobs `metadata.json` for `baseline_actions`. That is legal against the *public* environment files
   the competition ships, and it correctly degrades to `None` on the hidden set (*"baselines are hidden
   and env ids are anonymised clones"*). **Port `heuristic_action_target()` (a clamped
   `[40,100]` proxy from valid-action count + board size) and treat the metadata reader as
   offline-diagnostics only.** Hardcoding per-game baselines would be game-specific *and*, per our own
   P1 finding, factually wrong (the latent-state game set is run-dependent: cn04/sc25 in, re86 out).
3. **No game-id conditioning anywhere in the ≥1.40 band.** Checked all 8 duck kernels + 4 bundles: no
   `if game_id ==`, no per-game action tables, no environment allowlists. Helmut's patch is explicitly
   game-agnostic (*"the patch never reads a game id"*). Nothing to inherit here, and nothing to fear.
4. **boristown's `ACTION7` note is inert.** Their cell-9 markdown describes an `ACTION7` round-trip
   patch, and their cell-11 disables it. Do not read the markdown as a shipped feature.
5. **Wheelhouse hygiene.** Every ≥1.40 kernel attaches exactly `driessmit1/arc3-vllm-h100-wheelhouse-v3`
   + one taaf source bundle + `driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot`, on the same pinned
   `gcr.io/kaggle-private-byod/python@sha256:57e612…`. Any port keeps that triple unchanged
   (`feedback_kaggle_env_match`, 5× confirmed).
6. **Two of the three portable items live in an attached dataset, not the notebook.** Per
   `feedback_kaggle_dataset_code_sync`, a `kernels push` does not ship them — `canivel/arc-war-kit`
   needs its own `datasets version` push (with an **absolute** `-p` path on this box) plus a runtime
   banner check.

---

## 6. WHAT THIS DOCUMENT CHANGES

- The census's closing recommendation ("swap the brain") and its framing ("the 1.5+ band knows
  something") are both **weakened by direct read**: there is nothing to catch up to in the public code.
- The 08-12 campaign reframe — **action efficiency is the binding constraint** — survives contact with
  the public record and is now the *only* item in this report with both a mechanism and an
  independent implementation at ≥1.40.
- The honest statement of our position: **we run the same harness as the 1.62, we draw from the same
  distribution as the field's single-draw population, and we have drawn 29 times.** Raising the mean is
  the whole job.
