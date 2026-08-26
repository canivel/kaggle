# PREREG — TURN-CADENCE FAMILY (arms C1 `cadence-effort`, C2 `cadence-cap`), SEALED 2026-08-22, pre-build
**Program:** `perturn_program_2026-08-22.md` §1.0/§2.2/§4 · defect established in `duck_eval/p0/BP35_DIAGNOSTIC_2026-08-22.md` · provenance resolved in `learnings/war_room/reference_config_provenance_2026-08-22.md`.
**Ordered by:** ITERATION_LOG 2026-08-22 — *"MONDAY: slot 1 = deliberation-cap cadence arm (subtraction class; instruments = tokens/acting-turn + acting-turns/game, lc outcome; more-turns-no-lc = decisive family kill); WEEKEND (zero GPU): seal the cadence prereg + price K-token cap vs medium-effort pin … as possibly two arms of one family."*
**Sealed by:** weekend-prep lane, 2026-08-22, **before any build and before the GPU week resets.** No numbers from these arms exist.
**Slots:** C1 = the first slot of the new GPU week. C2 fires only on the branch in §8.

---

## 0. THE QUESTION, AND WHAT ALREADY DECIDED THE DESIGN

**Question.** Our agent spends **4,961** generated tokens per acting turn and gets **18** acting turns and **60** actions per game; a competent configuration of this same harness spends **1,406** and gets **42/130**. Does *bounding inter-action deliberation* — a pure subtraction, budget-positive — convert into levels?

**What the provenance work settled, and which therefore is NOT re-opened here:**

- The reference run's cheapness is **the model** (`vrfai/Qwen3.6-27B-FP8`), not a setting: 18 of our own Qwen3.6 pulls reproduce its 1,406 tokens/turn to ±25% at *identical* analyzer settings, and no Qwen3.6 pull and no Qwen3.8 pull overlap. **Reverting the model is forbidden** — Qwen3.6 has never exceeded lc 22 / mean 2.88 on our rail; Qwen3.8 owns lc 28 / 6.173 and lc 35 / 6.21.
- **`analyzer_timeout=120` (the reference's value) is forbidden.** It is a request timeout, not a token cap; on Qwen3.8 it would abort the majority of turns and its abort path discards the turn's work.
- There *is* one knowable setting that puts the reference's prompt on our weights: **`reasoning_effort='medium'` renders the Qwen3.6 chat template byte-for-byte (1495 chars, 0 diff)**. That is arm **C1**, and it goes first because it attributes most cleanly: one documented setting, no code patch, and a builder + smoke that re-measure the template diff every build.
- The mechanism is **generation length, not generation count** (~1.22 generations per analyzer invocation on *every* config). A per-request cap therefore genuinely bounds the turn. That is arm **C2**.

**The strongest pre-data evidence AGAINST this family, stated here rather than discovered later.** `reasoning_effort='medium'` has been run twice on Qwen3.8: it delivered the mechanism (**2,462–2,565** tokens/turn, −48%; **29–31** acting turns/game, +72%; `Step executed` rate 63.5% vs 43.2%) and scored **lc 21 and lc 17** against the floor's 28. Cheapening turns *lowered* levels. Three reasons that read does not seal the question: (i) both runs used the **June-30 `duck-harness-kaggle`** vehicle, and Arm A measured the harness generation alone at lc 30 vs 28 — the medium pin has **never** been read on the current floor; (ii) those builds existed to isolate *weights* and their turn counts were never an endpoint; (iii) n = 1 per point against pooled seed sd 2.80 — lc 21 vs 28 is −2.5σ, suggestive, not sealed. **If C1 reproduces lc ≈ 17–21 on the current floor, that is a real result and the family loses its cleanest arm.** I am registering that expectation now, at **P(C1 SIGNAL) ≈ 20%**.

---

## 1. VEHICLE (identical for both arms; one variable each)

The **certified field floor**, byte-identical: `notebooks/q38-field-eval/arc3-q38-field-eval.ipynb`, 11 cells / 10 code, code sha `7227f3286cf60b25`, anim-20260807 bundle, `max_runtime_s_per_game = 7920.0`, served `Qwen/Qwen3.8-27B-FP8`, `MULTIMODAL_UPSCALE 4`, `ONLY_RESET_LEVELS=true`. Fresh slugs, `is_private: true`. Dataset/model/docker/machine fields byte-identical to the floor's (`feedback_kaggle_env_match`, `feedback_kaggle_model_attach`: pull-back verify including **model_sources EXACT**).

**Comparator (never a single run):** field floor **lc 28** + Arm A base **lc 30** ⇒ **mean 29.0**, pooled seed sd **2.80** (exp 35). Identical to the comparator sealed for P1 and P2, so the three arms are mutually readable.

---

## 2. ARM C1 — `cadence-effort` (the measured setting; FIRST)

### 2.1 The change — exactly one variable
The bundle's single vLLM launch argument
`--default-chat-template-kwargs '{"preserve_thinking": true}'`
becomes
`--default-chat-template-kwargs '{"preserve_thinking": true, "reasoning_effort": "medium"}'`.

Nothing else moves: wheelhouse, `--tool-call-parser qwen3_coder`, `--reasoning-parser qwen3`, `--enable-prefix-caching`, `--generation-config vllm`, `--max-model-len 65536`, `ANALYZER_CONTEXT_WINDOW 32768`, temp/top_p/top_k, tensor-parallel 1, multimodal, the image sha, machine shape, the competition source, the run cell.

`reasoning_effort` reaches the template as a **server default**; the harness sends only `chat_template_kwargs={"enable_thinking": …}` per request and vLLM 0.19.0 merges `default | request`, so the server default survives and `enable_thinking` is untouched. (Verified in the pinned wheel by the q38 lane: `vllm/entrypoints/openai/engine/serving.py:807`.)

### 2.2 Build recipe — ONE inserted cell, position 4
The floor mounts its bundle read-only from `/kaggle/input`, so the setup command is rewritten by **copying the bundle into the working dir and re-pointing at the copy**, between cell 3 (which resolves `BUNDLE_DIR`) and cell 5 (which executes `setup_commands.json`):

```
# --- [cadence] C1 effort pin -------------------------------------------
import shutil, json as _json
_dst = WORKING_DIR / "cadence-bundle"
if _dst.exists(): shutil.rmtree(_dst)
shutil.copytree(BUNDLE_DIR, _dst)
_sc = _dst / "setup_commands.json"
_cmds = _json.loads(_sc.read_text(encoding="utf-8"))
_ANCHOR = '\'{"preserve_thinking": true}\''
_PIN    = '\'{"preserve_thinking": true, "reasoning_effort": "medium"}\''
assert sum(c.count(_ANCHOR) for c in _cmds) == 1, "C1 FATAL: anchor count != 1 (bundle drift)"
_cmds = [c.replace(_ANCHOR, _PIN) for c in _cmds]
_sc.write_text(_json.dumps(_cmds), encoding="utf-8")
assert (_dst / DATASET_BUNDLE_MARKER).is_file(), "C1 FATAL: marker missing in the copy"
BUNDLE_DIR = _dst
os.environ["TAAF_KAGGLE_BUNDLE_DIR"] = str(_dst)
print("[cadence] effort pin armed: reasoning_effort=medium", flush=True)
```

`_command_env()` reads `BUNDLE_DIR` at call time, so the rebind is picked up by cell 5. **FAIL-LOUD inversion (the q38 rule):** any assert failure RAISES. A silent fallback here would serve `xhigh` and produce a number we would read as a cadence result.

Cell count 11 → **12**, with the insertion **declared** at position 4 so `local_gate` N1b/N6a and `scripts/preflight.py` D4 compare against the base with the declared cell removed — exactly the P1 pattern.

### 2.3 Runtime certification (before any number; any failure ⇒ INFRA DEATH, never NULL)
1. Kernel COMPLETE, `benchmark.json` present, **n_games = 25**.
2. vLLM banner line (the setup script prints the full argv) contains **`"reasoning_effort": "medium"`** — the pin is verified in production, not asserted at build.
3. Served model banner = **`Qwen/Qwen3.8-27B-FP8`**; model_sources EXACT at pull-back.
4. `anim-20260807` bundle markers present; solver banner echoes **`max_runtime_s_per_game=7920.0`**.
5. `[cadence] effort pin armed: reasoning_effort=medium` present.
6. **FORBIDDEN markers, all absent:** `EDGE1`/`EDGE2`, `= 3960.0`, `= 23760.0`, `[notes]` (P1), `attempt(` / `retry_mode` (P2), `LOCAL_ANALYZER_MAX_OUTPUT` (C2), and the graft token set.

> **Standing-rule note (exp 34), resolved without touching another lane.** `q38field_score.py:49`, `q38graft_score.py:53` and `budget_score.py:49` already INFRA-DEATH on the literal `reasoning_effort` in a log ("xhigh default violated"). **They therefore refuse a C1 artifact for free** — the sibling→C1 negative control exists already. Only the C1→sibling direction needs building, and it lives in this lane's own scorer. **No edits to other lanes' scorers are required or permitted.** (C2 is the exception: no sibling scorer looks for `LOCAL_ANALYZER_MAX_OUTPUT`; the cadence lane will *request* that token from the q38-field / graft / budget owners rather than edit their files.)

---

## 3. ARM C2 — `cadence-cap` (the invented cap; SECOND, conditional)

### 3.1 The change
`LOCAL_ANALYZER_MAX_OUTPUT = 768` (currently `0` = no `max_tokens` sent at all), plus **one confound-neutralising companion**: `LOCAL_ANALYZER_CONTEXT_WINDOW = 33024`.

The companion is not a second mechanism. Setting `MAX_OUTPUT=K` sets `_reply_reserve_tokens = K`, which would shrink `context_budget_tokens` from **31,744** to 32,768 − 768 − 512 = **31,488** — a −0.8% context change in the direction of the edge-1 family (widening context measured HARM at −12 lc; narrowing is the opposite sign and small, but it is *not nothing* and it is not the mechanism under test). Setting the window to 31,744 + 768 + 512 = **33,024** makes `context_budget_tokens` land on **exactly 31,744**, byte-identical to the floor and certifiable from the `[ANALYZER STATUS]` line. `VLLM_MAX_MODEL_LEN` is 65,536, so there is ample headroom.

### 3.2 Why K = 768 — derived, not chosen
The harness ships `LOCAL_ANALYZER_YIELD_SECONDS = 60`: the authors' declared unit of deliberation, the budget after which the analyzer must hand control back. `control_yield_reason()` is evaluated only **between** generations, so a single generation longer than 60 s makes the budget structurally unreachable. Our **measured** generation throughput on the certified floor is **12.88 tok/s per game-slot** (median over 25 games, concurrency 28, one RTX PRO 6000).

```
K = 60 s  ×  12.88 tok/s  =  773 tokens   ->   K = 768  (nearest power of two)
```

That is the largest generation that can complete inside the harness's own turn budget on our own rail. It is not a guess and it is not tuned to an outcome. Two cross-checks:

- **It is not below this harness's functional floor.** Qwen3.6 — the configuration that sustains 42 acting turns/game — already runs 36–42% of its generations above 768 tokens and plays perfectly well; K = 768 puts us at that same operating point rather than somewhere no configuration has ever worked.
- **It lands the family's terminal read at the reference's own cadence.** Predicted from the per-generation distributions: 61–66% of generations truncated, 59–70% of generated tokens removed, **tokens/acting-turn ≈ 1,500–2,000** and **acting turns/game ≈ 40–60** at constant clock. C1 lands at ≈2,460/30. So C2 is not a repeat of C1 — it is the escalation to the point where the *only* configuration known to sustain reference cadence lives, with better weights. If it delivers and nulls, the family has been tested at its own best case.

Rejected alternatives, priced, so nobody re-derives them: **K = 4096** removes only 13.5–15.7% of Q3.8 tokens (×1.19 turns ⇒ ≈ +2 lc, far under the MDE) — unreadable. **K = 2048** removes 25–39% (×1.65 ⇒ ≈ +8 lc) — sits exactly on the MDE with no margin *and* lands on C1's operating point, duplicating it. **Lowering `yield_seconds`** does not bite, because the budget cannot interrupt a generation. **`tool_steps`** does not bite, because generations-per-invocation is already ~1.22.

### 3.3 Build recipe — ONE inserted cell, position 6 (P1's slot)
`_run_shell_commands()` ends with `os.environ.update(env)` from the bundle's setup env, which would overwrite anything set earlier — so the cell must run **after cell 5** — and `_LOCAL_ANALYZER_MAX_OUTPUT` is read at **module import**, so it must run **before cell 7** loads the pickle that first imports `inference`.

```
# --- [cadence] C2 deliberation cap -------------------------------------
assert "inference" not in sys.modules, "C2 FATAL: inference already imported"
_CAD = {"LOCAL_ANALYZER_MAX_OUTPUT": "768", "LOCAL_ANALYZER_CONTEXT_WINDOW": "33024"}
os.environ.update(_CAD)
_write_setup_env_updates(_CAD)          # child processes inherit it too
print("[cadence] max_output armed: 768 ctx_window 33024", flush=True)
```

### 3.4 Runtime certification (any failure ⇒ INFRA DEATH)
Items 1, 3, 4 of §2.3, plus:
- `[ANALYZER STATUS]` shows **`max_output_tokens: 768`** (today it reads `server default`) **and `context_budget_tokens: 31744`** (byte-identical to the floor — the confound is proven held, not asserted).
- `[cadence] max_output armed: 768 ctx_window 33024` present.
- **`reasoning_effort` ABSENT** (C1's marker is forbidden here — the two arms must never compound silently).
- Same forbidden-marker set as §2.3 item 6, with C1's pin added and C2's own tokens removed.

---

## 4. INSTRUMENTS — the mechanism is measured, not assumed

The reader is **`duck_eval/cadence/cadence_instrument.py`**, written and validated on 2026-08-22, **before either arm exists**. It re-derives the 08-22 BP35 diagnostic table exactly from the artifacts already on disk (31/31 checks) and is wired into `scripts/local_gate.py` as check **P9**, with a negative control (**S13**) proving it reports failures against a poisoned expectation. `feedback_audit_the_instrument` is discharged for this arm before its data lands.

| instrument | definition | floor value | source |
|---|---|---|---|
| **M1 tokens per acting turn** (primary mechanism) | Σ`generated_tokens` ÷ #history entries with `generated_tokens > 0` | **4,961** | `benchmark.json`, no new logging |
| **M2 acting turns per game** (primary mechanism) | median over 25 games of that turn count | **18** | idem |
| M3 actions per game | median `len(history)` | 60 | idem |
| M4 tail tokens with no action | `solver_note tokens` − Σ attributed | 282,209 | idem |
| M5 truncation rate (**C2 only**) | share of generations with `finish_reason: length` | **0.0%** | transcripts |
| M6 step-executed rate | share of `[ANALYZER STATUS]` blocks ending `Step executed` | 43.2% (edge2 proxy) | transcripts |
| **OUTCOME lc_total** | Σ`levels_completed` over 25 games | **28** (comparator mean 29.0) | scorer |
| co-primary `trim1` | mean per-game score minus the best game | reported, never resolved silently against lc | scorer |

---

## 5. DELIVERY — read BEFORE the effect, and it has its own verdict class

**Seed 1 alone certifies DELIVERY. Seed 1 alone NEVER produces an lc verdict** (n=1 MDE is 11.1; the house rule is that a single seed is never read).

| gate | C1 threshold | C2 threshold |
|---|---|---|
| armed (certification §2.3/§3.4) | pin in the vLLM banner | `max_output_tokens: 768` + `context_budget_tokens: 31744` |
| **D1** M1 tokens/acting-turn | **≤ 3,200** (≥ −35% vs 4,961) | **≤ 2,500** (≥ −50%) |
| **D2** M2 acting turns/game | **≥ 25** (≥ +39% vs 18) | **≥ 33** (≥ +83%) |
| **D3** (C2 only) M5 truncation rate | — | **≥ 40%** of generations (from a measured 0.0%) |

**Failing any of D1–D3 with the arm ARMED ⇒ `DELIVERY FAILURE`.** This is a **distinct verdict class**, not a null and not an infra death:
- it does **not** count against the family's kill criteria,
- it does **not** license any statement about whether cadence buys levels,
- it triggers **re-scope, never re-read** — and it is logged as `DELIVERY FAILURE` in KAOS with the measured M1/M2/M5.

This clause exists because exp 9, exp 36 and P1's mechanism-C all delivered a mechanism and were read as mechanism nulls, and because the P1-suppressor reached **96.3% delivery with no behaviour change** — the campaign has now been burned four times by conflating "the mechanism did not run" with "the mechanism does not work" (`feedback_guard_never_fired`, `feedback_verify_treatment_can_fire`).

The specific delivery hazard for **C2**, named in advance: a truncated generation emits no tool call, the harness appends *"You have not acted yet… emit exactly one `python` tool call directly"* and re-requests inside the same 60 s window. If the model simply restarts its reasoning, the saved tokens are re-spent and **M1 will not fall while M5 is high** — D1-fail with D3-pass is exactly that signature, and it is a DELIVERY FAILURE, not evidence about cadence.

---

## 6. SEALED READ (n = 2 seeds; comparator mean 29.0, pooled sd 2.80, n=2 MDE 7.84)

Identical bands for both arms, so the family is internally comparable and comparable to P1/P2:

- **lc_total mean ≥ 37** ⇒ **SIGNAL** (+8, the n=2 MDE)
- **22 ≤ mean ≤ 36** ⇒ **NULL**
- **mean ≤ 21** ⇒ **HARM** (the setting is reverted permanently)

Co-primary `trim1` reported alongside; **an lc/trim1 disagreement is reported, never silently resolved.** Secondary, pre-registered and non-inferential: dead-phase turn count (perturn §1.2) and terminal-level action ratio (§1.3), plus M1/M2 as continuous covariates.

---

## 7. THE DECISIVE KILL — stated pre-data

**KILL-F (the family kill the order asked for): an arm that DELIVERS more acting turns and moves no levels retires that branch, and C2's version retires the whole family.**

- **C1 delivered (D1+D2 pass) AND lc mean ≤ 36** ⇒ the **uniform-effort branch is dead**. Only C2 survives, and it fires once.
- **C1 HARM (lc ≤ 21) with delivery proven** ⇒ **the entire cadence family is retired immediately.** Cheapening turns actively costs levels; no cap will fix that, and C2 is never built.
- **C2 delivered (M1 ≤ 2,500 and M2 ≥ 33 — i.e. the reference's own cadence, with better weights) AND lc mean ≤ 36** ⇒ **THE CADENCE FAMILY IS RETIRED IN FULL.** Recorded consequence, worth the slot on its own: *turn count is not the binding constraint; the 45–50-turn configurations are not ahead because they take more turns.* That closes the last "more of the same currency" lever after exp 39 closed clock, and leaves per-turn decision quality (perturn §2.4) as the entire remaining program.
- **Two infra deaths on an arm** ⇒ that arm is parked (standing rule).
- **Any DELIVERY FAILURE** ⇒ no kill fires, re-scope.

---

## 8. ORDER, CONTINGENCIES AND COST

| step | when | condition |
|---|---|---|
| **C1 seed 1** | first slot of the new GPU week | unconditional |
| **C1 seed 2** | next available slot | seed 1 CERTIFIED and DELIVERED (a single seed is never read) |
| **C2 seed 1** | after C1's n=2 read | only if C1 is **NULL with delivery proven**; never if C1 is HARM |
| **C2 seed 2** | next available slot | C2 seed 1 certified and delivered |

**Cost:** ~2.3 GPU-h per build. C1 = 4.6 GPU-h; C2 = 4.6 GPU-h if it fires. **0 submission slots** — neither arm is competition-legal-by-default as a head candidate until it certifies, and Arm 0's nightly field-floor redraw is untouched.

**Interaction with the other queued arms:**
- **P1 (`arc3-p1-notes-eval`, persistent namespace)** — seed 2 is already owed on the new GPU week. P1 and C1 are **separate kernels on separate slugs**, so they can be built the same day without confounding; each forbids the other's marker (`[notes]` vs `reasoning_effort`/`[cadence]`). They must **not** be compounded until both have read.
- **P2 (reset-anchored retry)** — separate kernel, separate slug, mutually forbidden markers, no shared state. Same day is fine. See `p2_reset_retry_prereg_2026-08-22.md` §9.
- **Slot arithmetic is the real constraint, not confounding:** 2 pushes/day against the weekly GPU pool. The owed sequence is P1 seed 2 → C1 seed 1 → C1 seed 2 → P2 seed 1 → P2 seed 2, and it does not fit in one day.

---

## 9. WHAT THIS FAMILY CANNOT SETTLE

Whether Qwen3.8's *longer reasoning* is valuable **in general** — only whether bounding it on this harness, at this clock, buys levels. Whether a *smarter* stopping rule (stop when the plan is decided, rather than at a token count) would work; both arms are blunt by design because every "teach the model to notice" addition on this campaign has gone 0-for-6 and RedundancyBench retired that class. Anything about Qwen3.6 as a policy — the model comparison in the provenance document is descriptive and rests on n=1 builds per config.

## 10. ADVERSARIAL NOTE (my own strongest objection)

**"C1 is a re-run of the Q38-medium REFUTE with a different vehicle, and you already know the answer."** Partly fair — I put P(SIGNAL) at 20% and I have written the negative expectation into §0 rather than leaving it for the read. The defence is that the June-30 vehicle differs from the floor by a *measured* lc 30 vs 28 on Arm A, that the medium runs' turn counts were never an endpoint, and that a −2.5σ n=1 result is exactly the kind the campaign has been wrong about before. The real value is asymmetric: a NULL-with-delivery here plus a NULL-with-delivery at C2 **retires an entire family in two reads**, and this campaign's own record (0-for-6 on additions, exp 39 closing the budget family for 2.2 GPU-h) says that certainty about what does not work is the cheapest thing we buy.
