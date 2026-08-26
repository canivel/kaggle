# Daily brief — 2026-07-31

## 1. Result deep-dive (validated interpretation, not raw numbers)

### Overnight scored draw
- **1.10 frozen-fork filler (07-31 00:07Z, API COMPLETE)** — interior draw, z ≈ +0.95
  vs the frozen n=15 A/B control (0.9727/0.1343). Record ledger now **n=17 (mean
  0.9729, s 0.1332)**; control stays frozen at n=15 per prereg §3. Third-highest
  frozen draw on record (behind 1.33, 1.14) — consistent with the established
  distribution, no drift signal, no trigger. `runs/lb_ground_truth.md` refreshed
  from live API.

### Leaderboard movement (first head change since 07-24)
- **Andy liu NEW at #2 with 1.69** (submitted 07-30 06:42Z), GeniusYY 1.64 at #3;
  KOJIMA 1.86 unchanged at #1. **Gold cutoff moved UP to ≈1.50** (#14–15 at 1.50,
  #16–17 at 1.49; was 1.49). Dense band 1.47–1.61 intact. Implication: the bar the
  boristown A/B is chasing (anchor 1.47) is now BELOW the gold line by ~0.03 — the
  A/B remains the right next step (validated mechanism > luck-chasing), but gold
  requires the post-gate lane too (see §3 compaction).

### Gate-eval build rail (entry-gate #1 in flight)
- **Seed-1 pushed 09:0x local → `canivel/arc3-duck-gate-eval` version 2** (RUNNING,
  ~2.2 GPU-h expected; monitor armed). Pull-back verified: cell-content identical
  to the staged smoke-tested notebook (18/18 cells), metadata round-trip clean
  (3 dataset_sources, no model_sources, RTX Pro 6000, byte-matched docker image),
  `A17-GATE-EVAL seed=1` banner confirmed in served code.
- **Version-mapping correction:** an early seed-1 iteration was pushed as v1 on
  07-30 ~08:42 and never logged (pullback `runs/tmp_pullback_gate_eval_s1/`
  proves it). Two-seed proof mapping is therefore **v2 = seed-1, v3 = seed-2**.
- **Pin reconciliation RESOLVED (blocks removed for tomorrow's seal):** the
  readiness note's `boris_16_wait_body.txt` (sha 9755ac54) and the preflight
  report's `boris_16_gatebody.txt` (sha 37e30181) are BOTH contiguous substrings
  of the true audited artifact — the pulled boristown notebook cell 16 (LF, 602
  bytes). The sealed `boris_16_code.txt` dump is CRLF (Windows write artifact) and
  does NOT byte-match the notebook; gatebody = notebook cell 16 minus the trailing
  bare `wait_vllm_ready()` call, byte-exact. **THE pin for the seal =
  `boris_16_gatebody.txt` sha 37e3018108c7058ce3295dd27d1a3baa3548a311d28f3e99c43be131bd99078b**
  (maximal audited span); wait_body is a strict subset, superseded.

## 2. Discussions sweep (`learnings/sweeps/discussions_2026-07-31.md`)
- 1 new post since 07-30: Kaggle dual-notebook-execution Q&A (disc 731290) —
  **IGNORE** (known mechanic; our build runs are load-bearing evidence artifacts,
  we don't cancel save-version runs). No plan change.

## 3. Research sweep (`learnings/sweeps/research_2026-07-31.md`)
- **ADAPT (headline): OpenAI 07-30 — retained reasoning across turns + context
  compaction tripled GPT-5.6 Sol on ARC-AGI-3 public: 13.3% → 38.3%** (new
  frontier SOTA past Opus 5's 30.2%), harness-only, ~6× fewer output tokens.
  Model-agnostic, zero-cost, fits the 9 h wall, and directly validates our
  latent-state/context-management thesis. **Queued as R23 agenda item + candidate
  next preregistered arm AFTER the boristown A/B** — NOT injected into the active
  gate (single-diff invariant is the whole point of the A/B).
- IGNOREs: arXiv 2607.28573 (inference-compute scaling shifts failure modes —
  corroborates simplicity-wins), 2607.05775 + 2607.05378 (citation-only
  cross-refs for the error model). vLLM hermes tool-parser watch quiet.

## 4. Today's development (build-rail, weekday — no panel)
Single lane: **discharge entry-gates #1 and #2 for the boristown A/B** so the
prereg can seal tomorrow (08-01) and R23 ratifies Sunday (08-02).
1. Seed-1 eval build (v2) RUNNING; on COMPLETE → push seed-2 (v3) [2/2 push
   budget], pull v2 outputs → `runs/kernel_pulls/gate_eval_v1/`.
2. Grep both kernel logs for the 4 markers (seed banner / GATE armed / vLLM
   server ready / GATE fired latency ≤180 s) → entry-gate #1.
3. `uv run python scripts/gate_eval_screen.py gate_eval_v{1,2}` vs `runs/null10`
   → entry-gate #2 (non-harm; sentinel-precedent bar).
4. Queue: frozen-fork filler already queued for tonight (nothing fires until the
   A/B is sealed + ratified).

## 5. Open questions
- Seed-2 build may finish after the daemon window tonight — acceptable: the seal
  (08-01) needs both logs, not tonight's queue.
- NC-12 GPU parity note: attach to entry-gate artifacts for Sunday (grep RTX PRO
  6000 string from both eval logs while grepping the markers).
- Compaction arm (post-gate): needs an intent-file + prereg draft AFTER R23 —
  do not let it leak into the A/B.
- Sentinel draw #2 still queued strictly behind the A/B (backstop 08-10).
