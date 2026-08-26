# A22 — Compaction v2.1 (pure eviction, digest-OFF) — SEALED PRE-BUILD INTENT

**Status: INTENT — sealed pre-observation 2026-08-06, BEFORE any v2.1 patch
code is written (panel-free weekday build-rail). Parent documents:
`a22_compaction_v2_prereg_2026-08-04.md` (v2 prereg; everything not amended
here is inherited VERBATIM) and `learnings/sweeps/a22_v2_seed1_screen_2026-08-06.md`
(the v2 seed-1 K3 FAIL evidence this sub-arm answers; its §recommendation
names this exact sub-arm as the highest-information push). Lane state at
seal: v1 K3 FAIL + v2 K3 FAIL ⇒ a v2.1 seed-1 K3 FAIL makes the A22 lane
DEAD. That consequence is understood and accepted at seal time: a third FAIL
with the injection channel removed is a decisive negative answer for
compaction-as-tested in this regime, not a wasted push.**

## 0. The single question v2.1 seed-1 answers

Does **pure region-aware eviction** — zero digest injection, zero reserve
subtraction, eviction only under genuine token pressure — pass the M1
non-harm screen (mean Δlc ≥ −0.128 AND worst-game Δlc ≥ −1.0) vs
`runs/kernel_pulls/war_eval_v1/` (ledger-OFF `arc3-duck-war-eval` seed 1)?

Motivation from the v2 M2 attribution split (screen §M2): the digest paid
back ~51% of eviction relief and the reserve shrank the window on 80.9% of
events — the LightMem regime (relief with digest≈0) was never entered. v2.1
enters it by construction.

## 1. The ONE change vs v2 (sealed)

**Digest injection is DISABLED.** No digest message is ever rendered or
injected; no reserve tokens are ever subtracted (`reserve_applied` must be 0
on every event). Everything else is inherited from v2 unchanged:

- Region-aware eviction order and pins (v2 prereg §2.1) — UNCHANGED.
- Capture-into-store at both capture points (extraction, action-effect
  counts, sidecar logging, event schema) — UNCHANGED, measurement only.
  `digest_tokens` stays in the event schema and must be 0 on every event.
- RETAIN OFF (v2 §2.3) — UNCHANGED; `retained_reasoning_msgs=0` every event.
- Suppress-cut-while-stuck K=5 (v2 §2.4) — UNCHANGED (note: with no reserve,
  clause (b) is vacuous; clauses (a)/(c) unchanged).
- One-flag graft `COMPACTION=1`, kill switch `COMPACTION_DISABLE=1`, vanilla
  fallback on any failure, zero LLM calls, NO locks, no game-id logic — all
  UNCHANGED (v2 §2.5).

Implementation: `duck_eval/warpack/_kaggle_dataset/compaction_patch.py`
VERSION → **v2.1**; digest rendering/injection and reserve subtraction
compiled out behind `COMPACTION_DIGEST` (default **0**; `=1` restores v2
behavior — not used in this arm). Anti-self-ingestion strip (v2 §2.2) is
kept at ingest (store hygiene; costless, no injection path).

Explicitly OUT of this arm (recorded for a possible v3, NOT built):
minimum-age / delayed-eviction gate per arXiv:2608.00902 (research sweep
2026-08-06 ADAPT). One change per arm.

## 2. Canary (mechanism-engagement gate, from the build log)

- `compaction v2.1: ACTIVE` banner, showing **digest=OFF** and mirroring=OFF.
  Absent / `PATCH FAILED` ⇒ VANILLA, VOID (K2).
- `COMPACTION=1` stamp line.
- ≥1 `COMPACTION ` event line (K1); per-game sidecars.
- **Digest-OFF canary:** `digest_tokens=0` AND `reserve_applied=0` on EVERY
  event. Any nonzero ⇒ injection channel leaked ⇒ run VOID for §0.
- RETAIN-OFF canary: `retained_reasoning_msgs=0` on every event.

## 3. Metrics

- **M1 (primary): inherited VERBATIM** (paired Δlc vs war-eval seed 1; worst
  ≥ −1.0, mean ≥ −0.128).
- **M2:** inherited ratios (tokens/action, tokens/lc, tokens/turn) + the
  attribution split, which for v2.1 must show digest-injected tokens = 0 and
  relief = pure eviction (this IS the LightMem cell of the design matrix).
- **M3: measured for the record, NOT a kill criterion for this arm** —
  amendment vs v2: the v2.1 mechanism has no injection channel, so no
  re-proposal reduction is expected; the v2 seed-1 M3 win (−4.57pp, p=0.012)
  belongs to the digest and is preserved as evidence for a future
  standalone refuted-list micro-arm (R24 pile). A significant M3 reduction
  appearing WITHOUT injection would instead flag a confound in the M3
  measurement itself — report it, don't celebrate it.

## 4. Kill rules

- K1/K2/K3: inherited VERBATIM. **K3 at v2.1 seed 1: FAIL ⇒ A22 lane DEAD**
  (third independent K3 strike across the lane; sentinel A5/A8 standard).
  Post-death disposition (sealed now): close the lane in the project memory;
  carry forward (i) the M3 refuted-list result and (ii) the borro1980
  variance map as inputs to any future memory-lane design; no compaction
  push of any kind without a Sunday-panel revival decision.
- K4: NOT applicable to v2.1 (see §3 M3 amendment).
- K5: inherited (no scored draw requested here).

## 5. Budget + process constraints (binding, unchanged)

Free Kaggle build rail ONLY; NEVER submitted; dataset version push
(`canivel/arc-war-kit`) BEFORE kernel push + runtime banner check +
byte-audit; runtime-tested pre-push via `duck_eval/warpack/compaction_smoke.py`
(extended with digest-OFF assertions, 100% PASS required); builder
regression (default/--sentinel/--w0/--a17-canary byte-identical) before the
--compaction rebuild. Max 2 kernel pushes/day respected (this is push 1).
The build session does NOT push — the orchestrator verifies and pushes.
