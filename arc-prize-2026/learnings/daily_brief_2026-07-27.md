# Daily Brief — 2026-07-27 (Sunday — full panel + weekly consolidation day)

## 1a. Result deep-dive (validated interpretation)
- **Draw:** 2026-07-27T00:07Z submit = `canivel/arc3-duck-repro` v3 (frozen-fork filler) → **1.02**. Pre-registered expectation (frozen band 0.82–1.33) **met**, comfortably interior.
- **Ledger math (validated vs runs/lb_ground_truth.md + 07-26 brief):** frozen ledger now **n=13, mean ≈ 0.974, s ≈ 0.143**. Today ≈ +0.34 t-units above prior mean — unremarkable. Band unchanged 0.82–1.33.
- **No drift:** frozen tail 0.82 → 1.05 → 0.84 → 1.02 alternates around mean; 07-24 MK/CUSUM no-trend verdict stands.
- **No triggers:** A21/C2 harm-pause is exploration-arm-only (and 1.02 ≥ 0.80); W2 rule and OBJ-H null10 kill-switch are eval-side. Nothing pauses/kills/escalates.
- Ledger file deliberately not edited (precedent: only the live-API-verified refresh path edits it). Full artifact: `learnings/artifacts/result_deepdive_2026-07-27.md`.

## A17 priority pin (single highest-priority build item; C4 deadline Aug 3)
- **v5 boot canary (dataset-weights route, `canivel/qwen25-vl-72b-awq`) is RUNNING** on `canivel/arc3-a17-72b-canary` as of this morning — no PASS/FAIL yet.
- **v6 staged in parallel** (build-only, no push): v5 with sealed window restored (`A17_WINDOW_S=7920`, original soft_end verbatim), banner `mode=throughput-canary-v6-dataset-weights`. Discharges 07-26 prereg gates G1–G4 + delivers ρ_action denominator. Push fires only on v5 PASS; staging report `learnings/artifacts/a17_v6_staged_2026-07-27.md` (agent in flight at brief-writing time).
- Model-mount API route remains DEAD (root-caused 07-26); dataset route is the only route.

## 1b. Discussions sweep (`learnings/artifacts/discussions_sweep_2026-07-27.md`)
- **0 new threads.** One comment on #728278 deleted before readable (re-check tomorrow); one missed 07-25 Scott Le Grand comment (hardware-equivalence opinion) — IGNORE.
- **Zero community intel on the Qwen2.5-VL tool-call format defect** — still ours to solve; staged fenced-recovery adapter unaffected.
- **Leaderboard static:** KOJIMA 1.86 #1; gold cutoff ≈1.49 (top-13); wall ~1.44; boristown 1.47 #16; us 1.33 ~#50+. Two gold-line resubmits overnight, no score change.

## 1c. Research sweep (`learnings/artifacts/research_sweep_2026-07-27.md`)
- **1 ADAPT-low:** Prime Intellect **"Continual Harness"** (Seth Karten; amplified by @arcprize ~Jul 11) — reset-free harness with stored memories + reusable skill bank + self-refined prompt. Third independent convergence (after Schema, OCM/Rodionov) on *executable skill memory + in-play WM refinement*. Design input for post-A17 war-v4 only; no frozen-fork action. Watch: amplification raises competitor-adoption odds.
- **3 IGNORE:** COMAP 2606.02372 (training-time, off-regime), Text-World-Models 2606.09032, general-agent benchmark tail.
- No new in-window arXiv papers; Schema still self-reported/unreplicated (no-chasing posture C3 stands); Qwen2.5-VL + AWQ searches re-surfaced only known items.

## Weekly consolidation (Sunday items)
- **KAOS dream:** 6 memories ingested (190 total scored); digest `Dreams/2026-07-27-122503.md`; **0 skills auto-promoted** — matches "recency digest only" expectation. → panel agenda.
- **Fingerprint report:** 16 incidents, 8 recurring families, **nothing new since 07-08**. Dominant historical families remain `class:ERROR:none` (n=7) and `provenance:scratch-built` (n=5) — both addressed by fork-never-build + preflight; the discipline is holding.

## Open questions for today's panel (Sunday full panel)
1. **A17 v5/v6:** if v5 PASSes today, v6 (~2.5 GPU-h) fires as slot-1 — any panel objection to the sealed-window discharge plan? If v5 FAILs, retry slot is reserved; C4 has 6 days + slack.
2. **Process restructure (principal's 07-27 addendum):** panels now Sundays-only; weekday promotion via sealed arithmetic gates + A22 intent-files. Panel to acknowledge + name any conditions requiring weekday escalation.
3. **Post-A17 war-v4 design:** three-way convergence (Schema / OCM / Continual Harness) on skill-memory + in-play refinement — should war-v4 spec work start now in parallel, or strictly after A17 numbers exist?
4. **Exploration cadence:** A21 draw #1 (0.71) done 07-24; frozen filler since. When does draw #2 fire, and does the restructure change the exploration schedule?
