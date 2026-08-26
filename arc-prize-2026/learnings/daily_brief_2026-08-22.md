# DAILY BRIEF — 2026-08-22 (Sat)

## 1a. RESULT DEEP-DIVE — the field-floor config REPLICATED
`arc3-q38-field-eval` v1, submitted 00:07:11Z, **COMPLETE at 1.58.** Draw 1 was 1.59. **Spread 0.01.**
- Pre-registered expectation (a redraw of a certified SIGNAL config): **met.** Both draws sit **+3.7σ** above the frozen-filler
  ledger (`runs/ledger.json` re-read: n=37, mean 0.9316, s 0.1771, promotion bar **1.1087**).
- **Interpretation, not the number:** one 1.59 was a draw; **two at ≈1.585 is a config-level shift.** This retires the
  "1.59 was a lucky max-of-draws" reading of 08-21 and gives the campaign its **first replicated public step**.
- **What it does NOT buy:** the public score is a **MAX over submissions**, so draw 2 moves rank by construction zero.
  Board: **Canivel 1.59, #239 of 2465** — **−28 ranks overnight on an unchanged score**, on the ≈30/day drift measured in
  exp_id 29. Gold 2.37 → **2.47**, prize 2.58 → **2.72**.
- **Per-mechanism evidence:** none new — this was a redraw of an existing certified artifact, by design.

## 1b. DISCUSSIONS SWEEP — one new topic, and it is the most relevant post in weeks
Forum enumerated by CLI 2.2.2 (browser route dead: the chrome-devtools profile is locked by a running Chrome).
Max id yesterday 736540; today **736578** is the only new topic. (736540 was swept yesterday → IGNORE, unchanged.)

**736578 — "Public vs. Private Discrepancy"** (Nick Pellegrin, 08-21 15:56Z, 4 votes, 2 comments). → **ADAPT.**
He reports duck+qwen3.8 → local **2.1** / LB **~1.4**; his **own** harness → local **5.0–5.4** / LB **still ~1.4**.
A **2.5× local gain that bought 0.00.** Only reply (Son Pham, +3) is the generic "public set is easier / has been trained on",
which does not explain a *differential* drop between two harnesses — the poster says so himself and is right.
**Why this is not just someone else's problem:** our entire screening rail is sealed lc bands on that same local 25-game
benchmark. If the instrument does not transfer, every arm we have screened is uninterpretable.
**We therefore tested it against our own record — see §2. It is the day's work.**

## 1c. RESEARCH SWEEP
- **"Test-Time Adaptation for LLM Agents via Environment Interaction"** (Chen et al., ICLR 2026; arXiv 2511.04847) — *dynamics
  grounding*: persona-driven exploration probes causal dynamics **before** task execution, giving an in-context world model;
  reported effective at low compute, most in environments with unpredictable dynamics. → **ADAPT, Sunday agenda.** It is the
  explore-then-exploit shape ARC-AGI-3 already demands, and unlike the context-ceiling raise we just killed it adds *structured
  probing*, not *more window*. Caveat before anyone gets excited: its benchmarks are function-calling and web navigation, **not**
  ARC-AGI-3 — transfer is [INF], and this campaign has now logged **three** non-comparable published headlines (MAP's 22/25,
  ARChitect's ARC-AGI-1 win, Sensi's 50–94× efficiency).
- Surfaced titles-only, no disposition claimed: EvoAgentBench (2607.05202), Agentic Test-Time Training (2607.03441),
  TAME agent-memory (2602.03224). BeliefMem (2605.05583) and yesterday's Sensi perceptual-grounding ADAPT still parked → Sunday.
- No new ARC-AGI-3-specific result. cstl (#1) remains untraced.

## 2. THE DAY'S FINDING — our sealed single-seed bands are ~1.8σ, and we already produced one false positive
Full working: `learnings/war_room/local_lb_transfer_2026-08-22.md`. Headlines:
1. **The poster's failure mode exists in our record — as a single-seed artifact, not a transfer failure.** `war_eval_v1`
   scored **22 lc (+3.16σ vs the vanilla null)** and bought **×1.00 on the LB over 5 draws.** But v1/v2/v3 carry the
   **identical config label** `duck-harness-kaggle-warpack-v1` (verified in the artifacts) — three seeds of one config:
   **22 / 15 / 13.** Against **its own** replicates, 22 is **+1.13σ. Nothing.** The LB was right; our read was wrong.
2. **We compared a single seed against the wrong reference distribution.** That is the transferable instrument lesson.
3. **Within-config seed noise, measured for the first time:** vanilla sd **2.15** (n=10), warpack sd **4.73** (n=3),
   **pooled 2.80 lc**. Our ±5 lc single-seed bands are therefore **1.06σ–2.33σ** (pooled **1.79σ**, one-tail FP **3.7%**) —
   materially weaker than the "diff-SD 5.011" the preregs cite.
4. **Re-reading every single-seed call against pooled sd 2.80:** edge-1 **−4.3σ** (HARM stands), Arm 3 **−3.6σ** (stands;
   its vehicle/bundle confounds are separate and still binding), Arm A **+0.7σ** (NULL stands), **edge-2's pending ±5 gate
   = 1.8σ — the weakest call we will have made.**
5. **Large local effects HAVE transferred on our rail:** field-floor local ×1.84 → LB ×1.70 (ratios agree to 8%). But this
   rests on an **[INF]** baseline (null10 is not proven byte-identical to the submitted fork), so it is directional only.
6. **What our record cannot answer, and where the poster may still be right about himself:** every config we have is
   **duck-lineage**. A from-scratch harness has far more freedom to overfit the 25 public games. That cell is untested.

**Filed PRE-OBSERVATION (edge-2's number has not been seen): the gate STANDS UNCHANGED** — moving it after reading the forum
would be moving goalposts, and heading costs nothing under MAX scoring. What changes is what a pass may be *written as*:
> A single-seed SIGNAL of +5..+8 (≤~3σ pooled) is a **DRAW, not a finding**. It may head the queue; it may **not** be recorded
> as a mechanism, entered in the registry as confirmed, or used to justify a follow-on build, until a **second seed**
> reproduces it. ≥ +9 (≥3.2σ) may be written as provisional.

## 3. STATE
- **Slot 1 SPENT:** `arc3-q38-private-eval` **v3 = EDGE-2** (visible-updates capture contract on the certified base), pushed
  ~06:5X, **RUNNING**, terminal ~15:00 EDT. Prereg `edge2_prereg_2026-08-22.md` sealed pre-push (paired ±5 vs base lc 30).
- **Tonight's head, pre-ruled mechanical:** certified SIGNAL (≥+5) ⇒ heads via the snapshot convention; NULL/HARM ⇒ Arm 0
  field-floor redraw (draw 3). No middle cases.
- **Queue:** 1 pending — `arc3-duck-repro-pathsafe` v1. Not empty. Frozen duck-repro RETIRED per Arm 0.
- **Daemon:** clean; 00:07:13Z fired `ok: true`, `queue_remaining: 1`. Heartbeat OK, 2465 rows.
- **KAOS agent spawn FAILED again** on the documented blocker (`kaos run` kills any agent whose first token takes >60s;
  fable-panel is text-only). The adversarial review was done directly against artifacts instead — which is stronger evidence
  than a text-only opinion, but the KAOS-native mandate is **blocked by infrastructure**, not by choice. Third recurrence.

## 4. OPEN QUESTIONS
1. **Does edge-2 replicate?** Whatever tonight's read says, it is n=1. A second seed is now the cheapest high-value build.
2. **Should every future arm ship 2 seeds by default?** At 1.8σ per single seed, the marginal seed buys more than the
   marginal arm. This is a Sunday-panel question with a real cost implication.
3. Does dynamics grounding (2511.04847) survive contact with a 31,744-token window that we just proved gets *worse* when widened?
4. Unchanged: cstl untraced; "forgetting REFUTED or DELIVERY-WITHOUT-USE?"; perceptual grounding (Sensi) vs delivery-without-use.
