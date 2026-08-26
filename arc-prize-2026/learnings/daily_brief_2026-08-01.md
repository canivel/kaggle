# Daily brief — 2026-08-01 (Saturday; seal day for the boristown A/B)

## 1. Result deep-dive (validated interpretation, not raw numbers)

### Overnight scored draw: 0.65 — OUT-OF-BAND campaign low, verdict ISOLATED TAIL
- **0.65 frozen-fork filler (08-01 00:07Z, API COMPLETE)** — below the historical
  0.82–1.33 band; z = **−2.40** vs the frozen n=15 control. Full analysis:
  `learnings/sweeps/draw_deepdive_2026-08-01.md`.
- **NOT platform-wide:** full LB CSV (2001 teams) shows the head byte-frozen vs
  07-31 (KOJIMA 1.86, Andy liu 1.69, GeniusYY 1.64) and **our banked 1.33 intact**
  — a rescoring would have rewritten it. Isolated draw, not an infra event.
- **Tail-consistent with stationarity:** P(single ≤0.65 | frozen) = 0.8% Gaussian
  / 1.8% t; **P(≥1 of 18 ≤0.65) = 13.7% / 27.6%** — a ~1-in-7 event over the
  record's length, not shift evidence. Independent corroboration: yw8837's public
  duck fork (same family/config) reports an 11-run spread **0.55–1.29, σ≈0.24**.
- **Prereg clause audit: NOTHING triggered.** The <0.80 harm-pause is scoped to
  gated arm-B draws; the control is sealed at n=15 (0.9727/0.1343) and 0.65 never
  enters it; promote threshold 1.0970 stands. **Seal proceeds.** Record ledger
  n=18 (0.9550/0.1500). Watch-rule: a SECOND sub-0.80 control filler soon →
  stationarity re-check before further gated draws.

### Rank/cutoff drift
- Our 1.33: **#51 → #63** (pure churn). **Gold cutoff up again: ≈1.50 → ≈1.54.**
  The boristown anchor (1.47) is now ~0.07 below gold — the A/B is still the
  right validated mechanism test, but gold increasingly requires the post-gate
  lane (compaction arm) regardless of A/B outcome. R23 agenda.

### Gate-eval rail (entry-gates for the A/B)
- **Seed-1 (v2) COMPLETE + read:** all four §6 markers green (`seed=1` banner,
  `GATE armed poll=5s timeout=180s`, `GATE fired vllm_ready_latency_s=0.0` ≤180s,
  RTX PRO 6000 for NC-12). Offline bench mean 1.43 / 25 games.
- **Non-harm screen seed-1: PASS, direction POSITIVE** — paired Δlc **+0.112**
  (sd 0.353, 9W/7L, pos-tail p=0.070, harm-tail p=0.944); worst game cn04 −0.5 vs
  −1.0 cap; secondary ΔRHAE −0.098 (p=0.86, non-gating). Far above the sentinel
  admission precedent (−0.128). `runs/gate_eval_v1/screen_report.md`.
- **Seed-2 (v3) pushed 08:4x today** (yesterday's session ended before its monitor
  fired — 2nd process slip in 2 days, both log entry + push missed; backfilled).
  Pull-back verified byte-exact. Building ~2.2 h; last blocker for entry-gates.
- **Arm B canary `canivel/arc3-duck-gate` v1 PUSHED + byte-verified** (18/18
  cells, metadata field-identical to frozen family) — prereg fire-condition §7.2.
  Pinned single-diff preflight (`--max-diff-cells 1 --pin boris_16_gatebody.txt`,
  sha `37e30181…078b` re-verified byte-exact today) runs on build COMPLETE.

## 2. Discussions sweep (`learnings/sweeps/discussions_2026-08-01.md`)
- **ADAPT — yw8837 "[LB 1.17] Qwen3.6 Duck + 300-Game Diagnostics"** (07-31):
  open-source duck fork, baseline config, public 1.17. Two future single-diff arm
  candidates (P1 repeated-no-effect action guard; P2 analyzer yield 60→90 s) —
  queued strictly AFTER the A/B, never bundled. Their 11-run σ≈0.24 spread
  independently confirms our variance regime (and contextualizes today's 0.65).
  300-game diagnostic dataset flagged for free offline mining.
- Dual-notebook thread bump (CPMP confirm): IGNORE, known mechanic.

## 3. Research sweep (`learnings/sweeps/research_2026-08-01.md`)
- **ADAPT (principle) — Tycho** (arXiv:2607.28287): executable game-specific
  world models + actionable/animation frame separation → RHAE=100 on all 25
  public games — but WITH code-exec tools + multi-model orchestration on the
  PUBLIC set (out-of-bounds for our restricted private track). Post-compaction
  candidate arm; don't over-index on public RHAE saturation. R23 agenda.
- **ADAPT (spec-sharpener) — "Plans Don't Persist"** (2606.22953): re-surfacing
  evicted plans does NOT recover lost plan signal → the compaction arm must pair
  compaction with retained-reasoning + measure plan-signal survival, not just
  re-inject summaries. Amends the compaction-arm draft before prereg.
- IGNOREs: MemHarness (2607.28272, second source keeping EWM+banking shelved),
  Self-GC, Addressable Recall Compaction (design reference), vLLM hermes quiet.

## 4. Today's development (Saturday build-rail — SEAL DAY)
Single lane: discharge remaining §7 fire conditions and SEAL the prereg.
1. ~~Push seed-2~~ DONE (v3, push 1/2). ~~Push arm B canary~~ DONE (v1, push 2/2).
2. On arm B COMPLETE → pinned single-diff preflight → recorded ALLOW (§7.2 path b).
3. On seed-2 COMPLETE → pull → grep 4 markers + GPU string → screen vs null10 →
   entry-gates #1/#2 discharged both-seed.
4. Promote prereg DRAFT → SEALED (status header + evidence appendix: pin sha,
   marker greps, screen numbers, preflight verdicts) → **git commit = the seal**
   (predates first gated push; house style 4ecf49a).
5. Queue untouched tonight (frozen filler): first gated draw fires only after R23
   ratifies tomorrow (§7.3 compliant path).

## 5. Open questions (→ R23 tomorrow)
- Ratify the sealed A/B; first gated draw Sunday-night queue (08-02→08-05 draws).
- Gold cutoff climbed 1.49→1.54 in a week while our band is static — endorse
  compaction arm (with Plans-Don't-Persist amendments) as the immediate post-A/B
  prereg, regardless of A/B outcome?
  - **UPDATE (08-01 build session): the arm is now BUILT, smoked 41/41, and
    pre-registered as A22** (`learnings/war_room/a22_compaction_prereg_2026-08-01.md`;
    audit: stock stack's retained reasoning is silently dead — `reasoning` vs
    `reasoning_content` key mismatch starves preserve_thinking). Eval kernel +
    arc-war-kit dataset version STAGED for 08-02 slot-1, strictly behind any
    A/B slot need. R23 asked to ratify the EVAL plan only (no scored draws).
- 0.65 watch-rule ratification: second sub-0.80 control filler → stationarity
  re-check (MK/CUSUM) before continuing gated draws.
- Sentinel draw #2 still queued strictly behind the A/B (backstop 08-10).
- Process-slip mitigation: 2 consecutive sessions died waiting on build monitors
  (07-30, 07-31) losing end-of-day entries + a push slot — consider moving
  end-of-day log write BEFORE long monitor waits, or a scheduled 17:00 backstop.
