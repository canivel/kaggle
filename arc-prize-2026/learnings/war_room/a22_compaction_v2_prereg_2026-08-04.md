# A22 — Compaction v2 (region-aware eviction) — SEALED PRE-BUILD INTENT

**Status: INTENT — sealed pre-observation 2026-08-04, BEFORE any v2 patch code
is written (panel-free weekday build-rail; daily brief 2026-08-03 §4 is the
design source; seed-1 screen `learnings/sweeps/a22_seed1_screen_2026-08-03.md`
is the failure evidence this redesign answers). v1 prereg
`a22_compaction_prereg_2026-08-01.md` remains the parent document: M1/M2/M3
and K1–K5 are inherited from it VERBATIM (restated in §4/§5 below for
self-containment). The v1 arm is PAUSED under K3; v2 is a NEW mechanism under
the same arm/flag, same build rail, same pairing convention. Nothing here
touches boristown (DORMANT, NC-14) or the submission queue.**

## 0. The single question v2 seed-1 answers

Does **region-aware eviction alone** (RETAIN off, digest empty-allowed)
recover the war-eval seed-1 baseline — i.e., paired **M1 mean Δlc ≥ −0.128
AND worst-game Δlc ≥ −1.0** vs `runs/kernel_pulls/war_eval_v1/` (ledger-OFF
`arc3-duck-war-eval` seed 1)?

Nothing else. If sc25 recovers we cannot attribute digest-poisoning vs
batching (v2 removes both channels); attribution via sub-arms later — the
lane needs to move (brief 2026-08-03 §5).

## 1. v1 root causes this design answers (screen §1b/§5, verbatim summary)

1. **Toxic digest**: HeuristicExtractor promoted hedged, mid-sentence-truncated
   musings to "do NOT re-verify" FACTs; collapsed the refuted list to "+77
   more"; self-ingested through the model's echoed world-model (sc25 F5
   quotes F3; sc25 refuted=0 across all 42 events).
2. **Retained reasoning changed the policy, not just the memory**: +34%
   tokens/turn under fixed wallclock → blind action batching (sc25 49 turns @
   5.9 acts/turn vs 101 @ 1.8 paired).
3. Minor pure loss: the 1000-token digest reserve was subtracted
   unconditionally (window −3.2% even on the 244 content-free events).

Research reframe (sweep 2026-08-03, 5 ADAPTs): the duck harness already IS
the rolling-cut recency baseline; the literature axis is **region-aware
eviction with pinning vs recency** (CWL 2606.11213, MemDecay 2607.10582);
compaction wins are mostly **budget relief** (LightMem 2607.29104); suppress
eviction while stuck (SelfCompact 2606.23525); zero LLM calls in the eviction
path (Zero-Mem 2607.29377 — v1 complied, kept).

## 2. v2 mechanism spec (the 5 design points, sealed)

Patch: `duck_eval/warpack/_kaggle_dataset/compaction_patch.py`, VERSION → v2.
Same one-flag graft (`COMPACTION=1`), same kill switch
(`COMPACTION_DISABLE=1`), same vanilla-fallback-on-any-failure, zero LLM
calls in the eviction path, NO locks anywhere (forge_v35 lesson), no game-id
logic. `ledger_core.py` is NOT modified (shared library with the war-v2
ledger graft); all v2 hygiene is applied inside compaction_patch at
ingest/render time.

### 2.1 Region-aware eviction (PRIMARY; replaces digest-of-evicted-span)

The v1 capture-only wrap of oldest-first dropping is replaced by a
region-aware trim: history is parsed into blocks (a user message; or an
assistant message + its trailing tool results), and eviction under token
pressure selects blocks by class instead of pure recency.

- **Pinned (never evicted except last-resort):** the system prompt
  (structurally outside history; always messages[0]), the **most recent
  scientist-note carrier** (last assistant message whose content carries ≥2
  scientist-note labels: World model / Goal model / Action model / Recent
  findings / Open questions / Plan / Cross-level notes — older carriers are
  overwrite-superseded by design), the **most recent reasoning block** (last
  assistant message with non-empty reasoning/reasoning_content), and the
  trailing `preserve_recent` messages (the current-turn user message).
- **Eviction order (oldest-first within each class):**
  1. **stale action-episode blocks** — assistant-with-tool_calls blocks plus
     their tool results, EXCEPT the most recent episode. Staleness proxy
     (deterministic, no LLM): every executed action's board effect is visible
     in the current frame (the newest user message always carries the full
     current board), so all episodes but the newest are stale; the newest is
     kept for immediate observe-act context. The final user block (current
     frame carrier) is never an eviction candidate.
  2. non-pinned older user (state-dump) blocks — skipping a head user block
     whose removal would expose a pinned block to the harness's
     leading-non-user drop.
  3. non-pinned assistant reasoning/text blocks.
  4. the most recent episode block (non-pinned).
  5. last-resort fallback: oldest block regardless of pin (pins yield ONLY
     here — the request must never be bricked; class logged as `fallback`).
- Both v1 capture points remain capture points (token-budget trim + the
  30-assistant-turn persistence cap): every evicted message is still folded
  into the mechanical store (ledger_core extractor + action-effect counts),
  so nothing is silently lost — including messages removed by the harness's
  leading-non-user invariant.

### 2.2 Digest demoted + hygiene-gated (render-time; extraction untouched)

- The digest is SECONDARY: budget relief is the win (LightMem). It renders
  ONLY: (a) the refuted list, (b) hygiene-gated FACTs, (c) one small
  EVICTED/ACTION-EFFECTS/PROGRESS meta tail. **No ACTIVE and no CONFIRMED
  hypothesis lines** (the v1 ACTIVE "Wait, maybe…" lines were a harm
  channel; the model's own scientist note owns current plans).
- **FACT hygiene gate** (per record, at render): reject if hedge-prefixed —
  leading word in {"actually", "wait", "maybe", "i think"}, case-insensitive
  — or not a complete declarative sentence (must end '.' or '!' after
  stripping trailing quotes/brackets; mid-sentence truncations and questions
  fail).
- **Refuted list NEVER elided**: every refuted record renders as its own
  line; NO "+N more" collapsing, ever. Budget priority is REFUTED first,
  FACT second, meta third; rendered newest-first so if the reserve is ever
  exceeded the oldest lines drop silently (residual truncation, never a
  count line).
- **Header softened + non-quotable**: v1's "do NOT re-verify / do NOT
  re-test" directive becomes "previously observed evidence — treat as prior,
  not proof; re-testing is allowed", plus an internal-memo instruction not
  to quote or restate the digest.
- **Anti-self-ingestion (round-trip break)**: at ingest time, digest-shaped
  lines (FACT F\d+: / REFUTED|ACTIVE|CONFIRMED H\d+ / the digest marker) are
  stripped from evicted assistant text BEFORE extraction — the sc25 F5-quotes-
  F3 loop cannot recur.
- **Empty gate ⇒ inject NOTHING**: a digest is injected only if ≥1 refuted
  record or ≥1 gated FACT survives. No header-only digests (v1's 244
  pure-header events become zero-injection budget relief).
- **Reserve only when earned**: the token reserve (default 1000, env
  `COMPACTION_RESERVE_TOKENS`) is subtracted from the trim budget only when
  the pre-trim store would render a non-empty digest. If records first
  appear during a no-reserve trim, injection is deferred to the next trim
  (records persist in the store). This removes v1's unconditional 3.2%
  window shrink and makes budget-relief attribution clean.

### 2.3 RETAIN decoupled, OFF by default

New flag semantics: `COMPACTION_RETAIN` **defaults to 0** (v1 defaulted to
1). The reasoning→reasoning_content outbound mirroring is installed ONLY when
`COMPACTION_RETAIN=1`. **v2 seed-1 runs with it OFF** — the blind-batching
harm channel is out of the tested mechanism; retain becomes a separate
sub-arm (K4's sanctioned route), not a rider. Event field
`retained_reasoning_msgs` stays and is expected **0 on every event** (the
inverse of the v1 RETAIN canary).

### 2.4 Suppress-cut-while-stuck (deterministic rubric, K = 5)

- **stuck** := the last K executed actions visible in the outbound message
  tail all have `board_changed == false` (parsed mechanically from tool
  payloads; fewer than K observed results ⇒ not stuck). Env override
  `COMPACTION_STUCK_K`; default **K=5**.
- **K choice from v1 events data** (computed 2026-08-04 from
  `runs/a22_compaction_v1/artifacts/*_p0_events.jsonl`, board-diff over all
  25 games, 6,416 actions, 1,526 no-change streaks): streak length median 1,
  p75 2, p90 3, p95 3, p99 6, max 19. K=5 sits between p95 and p99 ⇒ the
  agent is in the stuck state ~1.8% of action-time (K=3: 6.0%, K=4: 2.8%,
  K=6: 1.3%) — deep enough to catch lp85/sb26-class derivation loops, rare
  enough not to starve compaction. This is the "K=5 if nothing better"
  default, now positively supported by the data.
- **While stuck:** (a) the 30-turn persistence cap is DEFERRED outright (the
  only fully discretionary cut — no eviction, no event); (b) no digest
  reserve is subtracted and no digest is injected, so compaction never
  causes an eviction the vanilla harness would not have made; (c)
  budget-forced evictions (physics: the request must fit the context window;
  literal suppression would overflow the request or trigger the harness's
  force-reduce path) still occur, still region-aware (protecting the active
  derivation is exactly the point of the guard), are captured into the
  store, and emit **NO eviction event** — their counts accumulate and flush
  into the next non-stuck event line. `stuck_suppressed` counts every
  suppressed cut/emission opportunity.

### 2.5 Kept from v1 (unchanged discipline)

One-flag graft (`COMPACTION=1` in cell 2; graft cell 12 via
`build_eval_notebook.py --compaction`); kill switch `COMPACTION_DISABLE=1`;
vanilla fallback on ANY failure (never 0); zero LLM calls in the eviction
path; state on the ToolAgent instance (one-thread-per-game, no shared state,
NO locks); greppable `COMPACTION ` stdout events + per-game
`*_compaction_events.jsonl` sidecars — now with NEW fields: cumulative
eviction-class counts (`ev_episode`, `ev_user`, `ev_reasoning`,
`ev_fallback`), `stuck_suppressed`, `reserve_applied` (0/1 this event),
`gated_facts` (post-hygiene count, for the M2 attribution split), `retain`
(0/1). `episodes` increments only on trims that actually evicted.

## 3. Canary (mechanism-engagement gate, counted from the build log)

- `compaction v2: ACTIVE` banner present (dataset-version proof per
  feedback_kaggle_dataset_code_sync). Absent / `PATCH FAILED` ⇒ run is
  VANILLA, VOID for this arm (K2).
- `COMPACTION=1` stamp line present (cell-2 banner).
- **≥1 `COMPACTION ` event line** in the run (K1); per-game sidecars.
- **RETAIN-OFF canary (inverted vs v1):** `retained_reasoning_msgs=0` on
  every event AND banner shows mirroring OFF. Any nonzero ⇒ the sub-arm flag
  leaked ⇒ run VOID for the §0 question.

## 4. Metrics — M1/M2/M3 inherited VERBATIM from the v1 prereg

- **M1 (primary): paired Δlc** vs ledger-OFF `arc3-duck-war-eval` seed N
  (seed N = push N of the identical notebook). Non-harm screen exactly as
  the gate-eval/sentinel precedent: worst-game cap ≥ −1.0, mean admission
  precedent ≥ −0.128.
- **M2 (token efficiency):** generated tokens per executed action and per
  levels_completed vs the paired seed. **NEW (v2 amendment): budget-relief
  attribution split** — using the per-event fields, decompose the relief
  into (a) evicted-chars removed from the live window, (b) digest-tokens
  re-injected (cost), (c) reserve_applied share (window shrink), and relate
  each to the outcome deltas; the LightMem prediction is that (a) with (b)≈0
  carries the win.
- **M3 (plan-signal survival):** refuted-hypothesis re-proposal rate in
  transcripts (same forensics procedure as `transcript_forensics.md`) vs the
  paired seed. Compaction must REDUCE re-proposal of refuted goals.

## 5. Kill rules — K1–K5 inherited VERBATIM (v1 prereg §5)

- **K1:** canary = 0 COMPACTION events in an eval run → mechanism never
  engaged → no further pushes of this arm until root-caused; the run is VOID
  as evidence either way.
- **K2:** vanilla-fallback banner (`PATCH FAILED`) → run VOID (counts
  against neither side).
- **K3:** non-harm screen FAIL on the sentinel-precedent thresholds at seed
  1 → arm PAUSED; a second independent FAIL at seed 2 → arm DEAD (3-strike
  rule not needed; this is the sentinel A5/A8 standard).
- **K4:** M3 shows NO reduction in refuted re-proposal rate across both eval
  seeds → kill the compaction component (retain-only rerun allowed once as a
  sub-arm via COMPACTION_RETAIN semantics inverted, single flag discipline).
- **K5:** any future scored draw (NOT requested here; separate gate + R23+)
  inherits A21/C2 verbatim: harm-pause on draw <0.80, no inference from n=1.

(K-counting note: v1's seed-1 K3 FAIL stands against the v1 mechanism; v2 is
a new mechanism under the same arm — its K3/K4 clocks start at v2 seed 1,
per the brief's pivot decision. A v2 seed-1 K3 FAIL pauses v2; combined with
the v1 record it would put the whole A22 lane one FAIL from DEAD.)

## 6. Budget + process constraints (binding, unchanged)

Free Kaggle build rail ONLY (zero-budget rule); NEVER submitted; dataset
version push (`canivel/arc-war-kit`) BEFORE kernel push + runtime banner
check + byte-audit; runtime-tested pre-push via
`duck_eval/warpack/compaction_smoke.py` (v2-extended, 100% PASS required);
builder regression (default/--sentinel/--w0/--a17-canary byte-identical
outputs) before the --compaction rebuild. This build session does NOT push —
the orchestrator verifies and pushes.
