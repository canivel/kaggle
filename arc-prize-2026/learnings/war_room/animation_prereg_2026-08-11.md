# ANIMATION-AWARENESS (sweep 08-11 ADOPT #1) — SEALED PRE-BUILD INTENT

**Status: INTENT — sealed 2026-08-11, BEFORE any `animation_patch.py` line is
written and BEFORE any arm run exists.** The verification measurement in §1 is
*not* an outcome of this arm: it is the pre-existing defect measurement that
motivates it, it was produced by an LM-free offline audit of our own engines
(`duck_eval/warpack/animation_frame_audit.py`), and it is sealed here together
with the intent so the two cannot be reordered after the fact.

Parent documents: `learnings/war_room/intel_sweep_2026-08-11.md` §1 (the sweep
finding, mechanism evidence only), `duck_eval/SCREEN_PROTOCOL.md` (binding;
K3″, P1/P2/P3, §4 seal language, §4.6 power-honesty clause),
`a22_compaction_v2_1_prereg_2026-08-06.md` (house prereg shape).

---

## 0. The single question this arm answers

**Does surfacing per-action intermediate-frame ("animation") information to the
agent, in a compact deterministic summary, change behaviour on the games where
our own audit proves that information is currently invisible — without harming
levels-completed?**

This is a **perception-defect repair**, not a strategy idea. The defect is in
the artifact we ship:
`duck_eval/taaf_bundle/src/tufa-arc-agi-framework/src/taaf/game.py:170`
returns `Frame(data=self.raw.frame[-1])`; `animation_frames` (`raw.frame[:-1]`)
and `all_frames` (`raw.frame`) are defined at lines 172–180 and have **zero
consumers** anywhere under
`duck_eval/taaf_bundle/src/ARC3-Inference/`.

---

## 1. VERIFICATION — the evidence base, measured on OUR engines (pre-seal)

Script `duck_eval/warpack/animation_frame_audit.py`; artifacts
`runs/animation/frame_audit.json` + `runs/animation/frame_audit.md`.
LM-free, offline, `arc_agi.Arcade(OFFLINE, environments_dir=kaggle-data/environment_files)`,
$0, 0 pushes, ~35 s wall.

Two probes per game, **each on its own fresh play**:
- **probe A (recorded)** — the real recorded action history of our own 25-game
  kernel pull `runs/kernel_pulls/a22_v2_1/benchmark.json`, replayed verbatim
  through `env.step` (≤400 actions/game). Realistic action distribution.
- **probe B (seeded)** — 300 seeded actions from the game's initial non-RESET
  `available_actions` (ACTION6 gets seeded x/y in 0..63). Covers action and
  coordinate space the recorded run never touched.

**Definitions (sealed).** For each action, from the RAW `resp.frame` list:
- `MULTI` — `len(resp.frame) > 1`.
- `INVISIBLE` — the settled frame `resp.frame[-1]` is **identical to the
  previous settled frame** AND at least one intermediate frame differs from it.
  This is the quantity that matters: it is exactly the case where the agent is
  shown a board byte-identical to the one before its action while the engine
  did render something. It is our named **state-aliasing / false-no-op** class.
- Per-game verdict: `type1` = ≥1 INVISIBLE; `type2` = MULTI but 0 INVISIBLE
  (pure motion interpolation); `single` = no multi-frame response observed.

**Result — 11,104 actions over 25 games:**

| game | type | actions | MULTI | MULTI% | max frames | settled-unchanged | **INVISIBLE** | INV% actions | INV% of no-ops | INV probe A | INV probe B |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ar25 | single | 531 | 0 | 0.0% | 1 | 85 | **0** | 0.0% | 0.0% | 0/231 | 0/300 |
| bp35 | type2 | 458 | 458 | 100.0% | 47 | 0 | **0** | 0.0% | 0.0% | 0/158 | 0/300 |
| cd82 | **type1** | 522 | 96 | 18.4% | 16 | 120 | **20** | 3.8% | 16.7% | 8/222 | 12/300 |
| cn04 | single | 700 | 0 | 0.0% | 1 | 116 | **0** | 0.0% | 0.0% | 0/400 | 0/300 |
| dc22 | single | 343 | 0 | 0.0% | 1 | 76 | **0** | 0.0% | 0.0% | 0/43 | 0/300 |
| **ft09** | **type1** | 352 | 284 | 80.7% | 5 | 283 | **281** | **79.8%** | **99.3%** | 0/52 | 281/300 |
| g50t | type2 | 346 | 182 | 52.6% | 53 | 84 | **0** | 0.0% | 0.0% | 0/46 | 0/300 |
| ka59 | type2 | 367 | 3 | 0.8% | 7 | 51 | **0** | 0.0% | 0.0% | 0/67 | 0/300 |
| lf52 | type2 | 325 | 325 | 100.0% | 16 | 0 | **0** | 0.0% | 0.0% | 0/25 | 0/300 |
| lp85 | type2 | 309 | 1 | 0.3% | 2 | 294 | **0** | 0.0% | 0.0% | 0/9 | 0/300 |
| ls20 | **type1** | 533 | 33 | 6.2% | 6 | 19 | **19** | 3.6% | **100.0%** | 19/233 | 0/300 |
| m0r0 | single | 354 | 0 | 0.0% | 1 | 94 | **0** | 0.0% | 0.0% | 0/54 | 0/300 |
| r11l | type2 | 330 | 246 | 74.5% | 23 | 0 | **0** | 0.0% | 0.0% | 0/30 | 0/300 |
| re86 | single | 700 | 0 | 0.0% | 1 | 26 | **0** | 0.0% | 0.0% | 0/400 | 0/300 |
| s5i5 | single | 360 | 0 | 0.0% | 1 | 0 | **0** | 0.0% | 0.0% | 0/60 | 0/300 |
| sb26 | type2 | 482 | 173 | 35.9% | **118** | 225 | **0** | 0.0% | 0.0% | 0/182 | 0/300 |
| sc25 | **type1** | 428 | 91 | 21.3% | 22 | 111 | **81** | 18.9% | 73.0% | 24/128 | 57/300 |
| sk48 | type2 | 511 | 231 | 45.2% | 3 | 198 | **0** | 0.0% | 0.0% | 0/211 | 0/300 |
| sp80 | type2 | 635 | 51 | 8.0% | 28 | 0 | **0** | 0.0% | 0.0% | 0/335 | 0/300 |
| su15 | type2 | 376 | 192 | 51.1% | 14 | 173 | **0** | 0.0% | 0.0% | 0/76 | 0/300 |
| tn36 | type2 | 355 | 19 | 5.4% | 7 | 0 | **0** | 0.0% | 0.0% | 0/55 | 0/300 |
| tr87 | single | 395 | 0 | 0.0% | 1 | 0 | **0** | 0.0% | 0.0% | 0/95 | 0/300 |
| tu93 | type2 | 358 | 187 | 52.2% | 15 | 0 | **0** | 0.0% | 0.0% | 0/58 | 0/300 |
| vc33 | type2 | 334 | 1 | 0.3% | 2 | 0 | **0** | 0.0% | 0.0% | 0/34 | 0/300 |
| wa30 | single | 700 | 0 | 0.0% | 1 | 156 | **0** | 0.0% | 0.0% | 0/400 | 0/300 |

**Totals: 17/25 games return multi-frame responses (4 type-1, 13 type-2,
8 single-frame). 2,573/11,104 actions (23.2%) were animated. 401 actions
(3.6% of all actions, 19.0% of all apparent no-ops) carried signal the agent
could not see.**

### 1.1 What our own measurement CONFIRMS, CORRECTS and ADDS vs the sweep

- **CONFIRMED — the defect is real and it is ours.** `all_frames` has zero
  consumers; multi-frame responses are common (17/25 games; the sweep quoted
  13/24 by a different method), and up to **118 frames arrive for one action**
  (sb26) — 100% of which our agent discards except the last.
- **CONFIRMED — `ft09` is the fat-head type-1 game.** 281/352 audited actions
  (79.8%) were INVISIBLE, and **99.3% of everything that looked like a no-op on
  ft09 was in fact an engine-rendered event.** ft09 is the game every public
  per-game table names as the top of the score distribution.
- **CORRECTED — `sb26` is NOT type-1 by the metric that matters.** The sweep
  (quoting the competitor's docstring) classes sb26 with ft09 because
  `frames[0] == frames[-1]` within a response (we measure 109 such responses of
  173 multi-frame ones). But in **every** one of those the settled frame still
  **differed from the previous settled frame**, so the agent was never aliased:
  0/482 INVISIBLE. The competitor's within-response `first==last` test is not
  the same quantity as cross-action board-identity, and only the latter is what
  the agent actually experiences. **We therefore do not inherit their taxonomy;
  we use ours, and we report both columns.**
- **ADDED — three type-1 games the sweep did not name:** `cd82` (20 INVISIBLE,
  16.7% of its no-ops), `sc25` (81 INVISIBLE, 73.0% of its no-ops), `ls20`
  (19 INVISIBLE, **100%** of its no-ops).
- **METHOD NOTE that changes numbers (recorded pre-seal, because it bit us):**
  running probe B on the env probe A left behind measures probe A's end state,
  not the game. Chained, ft09 reports 0/300 INVISIBLE; on a fresh play it
  reports 281/300. Every number above uses a fresh play per probe.
- **HONEST CAVEAT — the INVISIBLE rate is behaviour-dependent.** Probe A
  (our own recorded agent) yields 51/1,400 INVISIBLE on the four type-1 games;
  probe B (uninformed/random clicking) yields 350/1,200 on the same four. The
  defect bites hardest exactly when the agent is *searching* — which is the
  regime a stuck agent is in, and the regime the score is lost in — but this
  arm may not claim the probe-B rate as the rate it will repair.
- **Efficacy remains unmeasured.** Helmut AGI's 1.61 is not attributable to
  this feature (6-game/4-pass local A/B, no ablation published). This prereg
  treats the sweep as **mechanism evidence only**, per its own verdict.

---

## 2. The ONE change (sealed)

**Flag: `ANIMATION_AWARE=1`. Kill switch: `ANIMATION_DISABLE=1`. Master
fallback: any failure anywhere → VANILLA duck, never 0.** Module
`duck_eval/warpack/_kaggle_dataset/animation_patch.py`, house patch pattern
(VERSION marker, env-driven config, blanket-guarded `apply(bm)`, runtime
banner, `bm.label` stamp, no locks, no game-id logic, zero LLM calls).

Three seams, all harness-side, all deterministic:

1. **`inference.framework.solver._HarnessGameSession._execute_action`** —
   after the vanilla call, compute a per-action **animation summary** from
   `self.game.current_state.raw.frame` (the full list the engine returned) and
   the pre-action settled grid, and attach it to the returned payload as
   `payload["animation"]`. **`None`/absent whenever the response is
   single-frame or all frames are identical**, so ordinary actions cost zero
   tokens.
2. **`inference.framework.solver._HarnessGameSession.step_env`** — merge the
   per-action summaries of a batch into the batch's `final_payload` (vanilla
   keeps only the last action's payload, so without this a batch loses every
   animation but the last).
3. **`inference.agent.tool_agent.ToolAgent._compact_action_result`** (carry the
   field through to the model — the vanilla compactor drops unknown keys) and
   **`ToolAgent._summarize_step_sequence` / `_describe_last_outcome`** (one
   sentence, only when `board_changed` is False *and* an animation was
   observed, replacing the vanilla "did not show a confirmed board change;
   treat this as weak evidence" text, which is precisely the aliased reading).

### 2.1 The summary is token-bounded by construction (sealed)

Fixed-schema scalar dict, **no raw frames, ever**:
`{frames, unique_frames, board_unchanged, transient_cells, transient_bbox,
signature}` where `signature ∈ {reject_or_consumed, motion}` and
`transient_bbox` is 4 ints. Hard ceiling ≈ 45 tokens, emitted only on
animated actions. Sealed budget: **`animation_tokens_est / total_tokens` must
stay < 1%** (§5 K-A3). No retrieval tool, no per-frame timeline, no proactive
hint — those are the competitor's stages 2 and 3 and are **explicitly OUT of
this arm**.

### 2.2 Explicitly OUT (recorded now so it cannot be smuggled in later)

- **The hard no-op guard (sweep ADOPT #2) is NOT in this flag and NOT in this
  module.** It is strictly downstream and separately gated; the sweep's own
  finding is that it is *actively harmful* on type-1 games without animation
  awareness, so shipping them behind one flag would make the harm mechanism
  unattributable. A future arm gets its own flag, its own module, its own
  prereg.
- The `animation()` retrieval tool (diff timeline) — OUT.
- The proactive `turns_without_progress ≥ 6` hint — OUT.
- Any per-game special-casing — OUT (game-agnostic only; the patch never reads
  a game id).

---

## 3. Canary (mechanism-engagement gate)

Read from the build log; absent ⇒ the run did not test the arm.

- **K-A0 banner:** `animation vN: ACTIVE …` plus the `ANIMATION_AWARE=1` stamp.
  Absent or `PATCH FAILED` ⇒ ran VANILLA ⇒ **VOID** (not a FAIL).
- **K-A1 events:** ≥1 greppable `ANIMATION ` event line per run, on **≥5
  distinct games**. Per-game jsonl sidecars written next to the runtime state.
- **K-A2 invisible-repair counter:** the run must report a nonzero
  `invisible` count on **≥1** of {ft09, cd82, sc25, ls20}. Zero across all four
  ⇒ the mechanism never engaged where the audit says it must ⇒ **VOID**, and
  the audit method itself goes back under review.
- **K-A3 token bound:** per-run `animation_tokens_est / total_tokens < 1%`.
  Breach ⇒ kill the arm (the summary is supposed to be nearly free).
- **K-A4 exception counter:** `animation_errors == 0`. Any nonzero ⇒ report and
  kill; a perception patch that raises inside the action path is not shippable
  even if it scores.

---

## 4. Metrics

- **M0 (PRIMARY — mechanism).** Fraction of executed actions whose
  intermediate-frame signal was previously invisible and is now delivered:
  `invisible_actions / executed_actions`, reported overall and per game, with
  `multi_frame_actions / executed_actions` alongside. Pre-registered
  expectation from §1: nonzero on ft09/cd82/sc25/ls20, ~0 elsewhere. **This is
  the endpoint this arm is powered to answer.**
- **M1 (SECONDARY — DESCRIPTIVE ONLY, NOT A SCREEN).** Paired mean Δlc vs the
  legal same-config comparator family `duck-harness-kaggle-continuation-v1`
  (duck + (f), NO warpack — P1-legal because this arm is duck + (f) +
  animation). **That family has m = 2 (lc totals 10, 16), so per
  `SCREEN_PROTOCOL.md` §1 P2 this arm is NOT SCREENABLE.** Reported as
  advisory in the same sentence, every time, with the sign-flip p-value as
  descriptive significance only.
- **M2 (descriptive).** tokens/action, tokens/lc, wall-clock/action, vs the
  same family — to show the summary is free.
- **M3 (descriptive).** Rate of repeated identical no-op actions on the four
  type-1 games, which is the behaviour the repaired signal should reduce. This
  is *the* quantity the downstream no-op guard will act on, so measuring it
  here sizes that arm at zero extra cost.

### 4.1 Mandatory seal arithmetic (SCREEN_PROTOCOL §4)

1. **Baseline family:** `duck-harness-kaggle-continuation-v1` — runs
   `runs/kernel_pulls/w0_eval_s1/` (lc **16**) and
   `runs/kernel_pulls/w0_cont_eval/` (lc **10**) — both
   `label = duck-harness-kaggle-continuation-v1`, read from their
   `benchmark.json` at seal time. P1 banner evidence: both carry the
   `continuation v1: … (f) game-over-continuation graft applied … NO
   warpack/ledger` banner; the arm will carry the same banner **plus** the
   animation banner, and must carry **no** `warpack:` / `LEDGER` / `SENTINEL` /
   `COMPACTION ` lines. The warpack band is ILLEGAL as a control here
   (`runs/sealed/r17_thresholds.json → thresholds.control_band`) and is not used.
2. **m = 2 → the arm is NOT SCREENABLE.** Stated plainly, not worked around.
3. **σ̂ = 0.14174, df = 6**, pooled build-rail estimate (families
   `…-warpack-v1`, `duck-harness-kaggle`, `…-continuation-v1`,
   `…-sentinel-v2`; Bartlett p = 0.625).
4. **Advisory K3″ line at m = 2:** C(2) = 2.10 ⇒ −2.10 × 0.14174 =
   **−0.2977 lc/game**. C(2) is listed in the protocol "for advisory arithmetic
   only"; its measured type-I on the null10 corpus is the m-column of
   §2's OC table (m = 1: 2.0%, m = 3: 4.4%; m = 2 is interpolated and was never
   measured — a further reason this is not a gate).
5. **80%-power detection floor at m = 2:**
   `floor = C(2)·σ̂ + 0.8416·σ̂·√(1+1/2) = 0.29765 + 0.14609 = 0.4437 lc/game`
   = **11.09 levels over 25 games**, against a comparator family mean of
   **13.0 levels**. The arm would have to score ≈ 1.9 levels in total before
   M1 could call harm.
6. **POWER-HONESTY CLAUSE, INVOKED (SCREEN_PROTOCOL §4.6).** Power at m = 2
   against any plausible effect is far below 50%. **This run is an exploratory
   mechanism probe, not a screen. No PASS on M1 may be reported as non-harm;
   an M1 result may only be reported as "uninformative in both directions".**
   The arm's real result is M0 + the canaries.

### 4.2 What would make this screenable later (priced now, not bought)

One additional `--w0` (duck + (f)) build takes the comparator family to m = 3
⇒ C(3) = 2.02, line −0.286, 80% floor 0.4174 lc/game = 10.4 levels — still a
weak screen. A genuine −0.20 answer needs k = m = 8 (16 builds, ~35 GPU-h,
> 1 week of the 30 GPU-h/wk rail). **Not proposed here.** If the mechanism
probe reads well on M0/M3, the honest next step is the no-op-guard arm (which
has a directly countable endpoint), not a bigger Δlc screen.

---

## 5. Kill rules

- **K-A0/K-A1/K-A2 fail ⇒ VOID** (patch didn't run / didn't engage). Rebuild,
  do not record a verdict in either direction.
- **K-A3 (token bound) or K-A4 (exceptions) fail ⇒ arm KILLED** regardless of
  M1, and the module is reverted. A perception summary is only worth having if
  it is free and cannot throw.
- **M1 may not kill this arm** (§4.1.6) and may not promote it either.
- **Promotion requires a separate, later decision** on evidence this run cannot
  produce alone. Nothing here authorises a submission, a queue change, or the
  no-op guard.

## 6. Budget + process constraints (binding)

Free Kaggle build rail ONLY. **NEVER submitted**; the submission queue keeps
the frozen fork. Dataset version push (`canivel/arc-war-kit`) BEFORE the kernel
push, with a **byte-audit** of the pushed files and a runtime banner check in
the build log (`feedback_kaggle_dataset_code_sync`). Runtime-tested pre-push
via `duck_eval/warpack/animation_smoke.py` (100% PASS required)
(`feedback_test_before_submit`). Builder regression required: `default`,
`--w0`, `--sentinel`, `--compaction`, `--a17-canary` outputs byte-identical
before and after the builder gains `--animation`. `scripts/preflight.py` must
pass (`feedback_arc_kernel_structural_drift`: the notebook is derived from
`notebooks/duckwar/arc3-duck-war.ipynb`, never written from scratch). Max 2
kernel pushes/day. **$0 cloud spend** (`feedback_arc_zero_budget`).
