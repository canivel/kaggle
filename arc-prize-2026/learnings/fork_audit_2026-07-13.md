# R0 fork-band audit — day 1 (2026-07-13)

**Scope:** public duck-fork kernels above our control band (0.922). Sources pulled to `runs/fork_audit/`.

## Map of the public fork band

| Actor | Kernel(s) | LB attribution | Delta vs vanilla duck |
|---|---|---|---|
| Teddy Tennant (`thtennant`) | `arc3-duck-v7`(44 votes)…`v13` (daily iteration; v13 = Jul 13) | unknown LB name; iteration cadence + graft maturity suggest 1.3–1.5 band | `taaf_grafts` composite: efficiency + retry_guard + shortcircuit + recovery (v13 dropped banking) |
| Yin Li (`kevin250304`) | `arc3-duck-v9b-recovery-banking` | unknown | same grafts + banking=True |
| `caoyupeng` | `1-21-from-great-team-tufa-labs` | 1.21 vanilla resubmit | none (drift/σ anchor) |
| `ishivapatil`, `junjin2`, `rokaiyasomapti`, … | vanilla resubmits | ~duck band | none |
| Leaders (Tecnod8.AI 1.61, Mathurin/anngle/NoOneAhead 1.56) | **no public kernel found** | — | v3's "verify the 1.56 anchor is public" → **NOT public**; porting target restated against the highest audited fork = the graft band |

## `taaf_grafts` source audit (3545 LOC, `runs/fork_audit/tennant_src/src/taaf-grafts/`)

Shared engineering standard (relevant to our port gate): all flags default OFF; `install(bm, {})` is byte-identity ("the **1.15-floor guarantee**" — implies their vanilla floor ≈ 1.15); every graft individually guarded, any error → stock solver restored; inner `analyze` never wrapped (crashes propagate as stock); solver swapped by field-copy to survive `Benchmark.run` deepcopies; `context_window` patched as module global (env var would be too late — matches our phase-1 kit deployment lesson).

Per graft:
- **recovery.py (810 LOC)** — the interesting one. Forensics-driven (their v8 commit run): the two dominant losses are control-flow failures. R1 REFRESH: on detected death-spiral / post-death stall / lock-in, clear inner chat history in place + write synthesized fresh-start world model incl. a **hypothesis graveyard** (models snapshotted at each death). Zero action cost. R2 PROBE: bounded scripted probe burst on deathless lock-in, ≤ PROBE_MAX_ACTIONS, ≤ once/level, only far past baseline. R3 HANDOFF: distill models into `cross_level_notes` (the one key the vendor level-wipe spares).
  **Contrast with our dead phase-1:** theirs is *event-triggered, zero-action-cost, history-clearing*; ours was *always-on, real-action-spending, history-adding*. Their design agrees with our own pollution findings (ar25/su15) — REFRESH removes context, never injects.
- **banking_solver.py (329)** — WIN-gated per-level replay; card = max over plays. Confirmed dead for duck-class (0 WINs in 250 null runs) — Tennant dropped it in v13, independent corroboration.
- **shortcircuit_solver.py (294)** — no-op overshoot trimmer for homogeneous repeated action batches; "provably monotonic non-decreasing score." Directly targets pooled-action RHAE tax (fewer wasted actions in the eventual first-clear bucket).
- **retry_guard.py (233)** — pass-through analyzer chain rider (their layering substrate).
- **agent_ext.py / efficiency (568)** — budget-aware per-turn note + net-zero waste detection, report-only (their own comment: report-only prompt pressure "demonstrably does not change behaviour" — matches our prompt-is-noise memory).
- **transfer_solver.py (435) / family_store.py (354)** — cross-game mechanic transfer (transfer implies banking). Not yet flagged on in their public kernels' cells.

## Port-bundle candidates (per v4 §R0, Track A bundle, 3 Kaggle-rail seeds)

Ranked by (mechanism soundness under confirmed pooled semantics × independence from our failed levers):
1. **recovery R1 REFRESH** (event-triggered history clear + graveyard) — attacks the STUCK death mode our forensics also identified; zero action cost.
2. **shortcircuit** (no-op batch trimmer) — pure action-efficiency, monotone by construction, directly reduces the first-clear denominator.
3. **recovery R3 HANDOFF** (cross-level notes) — cheap, targets the sk48-class one-level-deeper wall.
4. recovery R2 PROBE — spends real actions (same failure class as our phase-1 explore); lowest priority, gate separately.

Not ported: banking (WIN-gated, dead), efficiency (report-only, proven inert), transfer (depends on banking).

**License check pending** (repo headers say nothing yet — check dataset README/license before shipping derived code).

## Open questions for tomorrow
- LB attribution: watch the LB names' score trajectories vs Tennant kernel versions (his v10 12 votes Jul 11 → who moved that day?).
- Vanilla-drift question (LA-N1): `caoyupeng` 1.21 resubmit + our σ-draws (0.82–1.02) on the same code = the drift measurement; their 1.21 was scored Jul 1 — is the environment harder now? Our 5-draw mean 0.922 vs Tufa's June 1.21/1.22/1.30 suggests **yes, ~−0.3 environment/rerun drift since Jul 1** — affects every EV baseline.
