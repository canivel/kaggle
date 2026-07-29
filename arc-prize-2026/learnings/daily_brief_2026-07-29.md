# Daily Brief — 2026-07-29 (weekday cycle: no panel; sealed gates govern)

Cadence note: full panel = Sundays only (restructure 2026-07-27). Today's decisions
were all pre-registered — the v6 read executed sealed branch B2 verbatim, no
discretion exercised.

## 1a. Result deep-dive

### A17 v6 full-window bench — S₁ = ΣN₇₂B = 5. VALID READ → branch B2 (seed-2 fired)

Gate-look artifact: `runs/a17_v6_gate_look_2026-07-29.json`. Read validity: 4/4
games present, per-game window 7920.0–7928.6 s, no drift WARN — no §7.2 void.

- **ΣN₇₂B = 5** (ft09 2, sb26 1, lp85 0, vc33 2; lc = 0 everywhere). ρ_action in
  the frozen-N form = 480/5 = **96** vs kill line 3.5 — a ~27× overshoot of the
  sealed threshold, and ~5–6× below even the pre-registered doom projection
  (26–33). The honest prior ("v6 expected to FAIL") was met and exceeded.
- **G1 FAIL**: action-parse success 8/1044 LLM calls (native hermes emissions
  **0/1044**; fenced-recovery 8, precision 8/8; hard parse failures 1036/1044 ≈
  0.992) vs prereg ≥ 0.95. **G2 FAIL**: 5 vs ≥ 100. G3 delivered: all 5 actions
  landed in the first ~90 s; `actions_total` frozen from heartbeat t=720 s to
  window end.
- **Mechanism (measurement, from `learnings/a17_v6_diagnosis_2026-07-29.md`):
  deterministic format-non-compliance livelock, NOT GPU throughput.** vLLM served
  1047 requests, all HTTP 200, `finish_reason=stop` on 1044/1044 (zero
  truncation), 14–93 tok/s, `stall_s=0`, MM cache → 100%. The model narrates a
  plan and ends with the bare word `python` instead of a tool call; the harness
  re-prompts ("You have not acted yet") and the model returns **byte-identical**
  text — a busy repeat loop. The only working parse path all window was the
  fenced-recovery adapter. The v5 boot canary's tool-call PASS used **forced**
  `tool_choice`; the harness's free-form dialogue never elicited a single native
  tool call. This kills two prior hypotheses: the KV-preemption/thrashing story
  (research sweep item, refuted — all requests completed) and "warmup suppressed
  the v5 rate" (N is window-length-invariant once the livelock engages).
- **Why the projection was 5–6× high**: the linear 26–33 extrapolation assumed a
  rate; the true process is "a fixed handful of lucky fenced recoveries early,
  then zero" — the failure is structural, not rate-limited.
- **Branch executed: B2** (no kill at k=1, R22 D4/NC-10). **v7 = seed-2
  confirmation pushed today (kernel push 1/2), RUNNING.** `build_v7_seed2.py`
  PROVES v7 = v6 with only the two seed substrings changed (every other cell
  byte-identical — asserted programmatically); pull-back verified (metadata:
  model_sources = [], `canivel/qwen25-vl-72b-awq` attached; code: seed=2,
  window 7920). Diagnosis verdict "deterministic" ⇒ expected outcome is
  concordance → **B2a tomorrow: 72B route DEAD, no third seed, no fix lane;
  build priority reverts to the boristown readiness-gate A/B.** If discordant
  (≥138): B2b, Sunday panel, no verdict.
- NC bookkeeping: **NC-4 DISCHARGED** (1044 replays analyzed ≥ 200 required, raw
  counts published); NC-9 satisfied pre-observation (commit `4ecf49a`); NC-10 in
  progress via v7; NC-12 GPU parity — canary and frozen-fork family share
  `machine_shape=NvidiaRtxPro6000`; scored reruns expose no logs, so parity is
  metadata-level only (Sunday panel item).
- For any future multimodal bench (post-B2a): the certified artifacts survive
  (weights-dataset route, fenced-recovery adapter, boot asserts). The diagnosis
  suggests the contract failure is prompt/parser-level (forced `tool_choice`
  per request worked at boot) — but per the seal, **no "one more fix" lane
  exists for the 72B screen**; any such lane is a NEW pre-registration for the
  Sunday panel, competing on merit against the boristown A/B.

### Overnight scored draw: 1.03 frozen filler (in-band)

`runs/lb_ground_truth.md` refreshed from live API. Frozen ledger **n=15, mean
0.9727, s ≈ 0.1343**; 1.03 is interior (z ≈ +0.44 vs prior stats), no drift, no
trigger. LB head unchanged: KOJIMA 1.86, gold cutoff 1.49, our best 1.33 (#51
band). Filler continues to hold rank, not climb — as designed.

## 1b. Discussions sweep (`learnings/sweep_discussions_2026-07-29.md`)

Thin. One new thread (#730225 neuro-symbolic MDL self-promo, −5 votes) —
**IGNORE**: its "25/25 SUCCESS" table is 3 execution steps/game declared wins,
the same low-ΣN mirage we just measured ourselves. Scoring-mechanics thread
picked up host reconfirmation: **9 h hard wall for v3 scored runs** (the "<12 h"
figure is Verified-Leaderboard-only) — ADOPT as locked budget fact. Nothing on
serving, stalls, or >1.33 forks. Plan unchanged.

## 1c. Research sweep (`learnings/sweep_research_2026-07-29.md`)

6 findings, deduped. Headline: arXiv:2607.16892 (KV-cache preemption under
output-length uncertainty) *looked* like our stall — **directly refuted by the
v6 diagnosis** (all requests completed; no preemption signature). Kept as ADAPT
for instrumentation hygiene only. AgentTether (2607.06273): transition-scoped
repair memory — right shape for any future banking arm, parked. Explorer-Definer
ARC harness (2607.06764): "generation-bound, not selection-bound" prior —
consistent with our livelock (the model never *generates* a valid call). No
plan change.

## 2. Build-rail state + queue

- Kernel pushes: 1/2 used (v7). Slot 2 held in reserve for a v7 retry only.
- Tonight's queue head: frozen-fork filler `canivel/arc3-duck-repro` v3
  (trusted-fork), already pending — queue non-empty ✓. Nothing passed promotion
  gates today (v7 is measurement-only by construction).
- **Boristown readiness-gate A/B (R22 D2, 5/5, date ~Aug 2): prereg DRAFT
  commissioned today** → `learnings/war_room/boristown_ab_prereg_2026-07-29_DRAFT.md`
  (n≈4 gated draws replacing fillers vs frozen control n=15, one-sided test at
  the 1.47-anchor effect, error rates published per NC-11 discipline; sentinel
  draw #2 queued behind). Fires after seal + trusted-fork preflight +
  ratification (Sunday 08-02 natural point; earlier only on principal's order).

## 3. Open questions (for Sunday panel R23, 2026-08-02)

1. If B2a lands tomorrow: ratify the boristown A/B prereg as the build priority;
   dispose of the "72B with forced tool_choice / grammar-constrained decoding"
   question explicitly (new prereg or permanent close).
2. NC-12 GPU parity: is metadata-level parity (machine_shape match) sufficient,
   given scored reruns expose no logs?
3. Sentinel un-shelve rule ratification (already drafted, queued behind A/B).
4. EWM Stage-1 (due Aug 4): still blocked by latent-state audit; the A17 rail
   resolves ≤ Aug 3 either way — re-price at panel with the B2a/B2b outcome
   known.
