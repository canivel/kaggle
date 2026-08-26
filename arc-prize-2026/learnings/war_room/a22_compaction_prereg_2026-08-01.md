# A22 — Compaction + retained-reasoning arm (INTENT, pre-registered pre-outcome)

**Status: INTENT — sealed pre-observation 2026-08-01 (before any eval build of
this arm exists). R23 (2026-08-02) asked to ratify the eval plan; scored-draw
entry is a SEPARATE later gate and is NOT requested here. Post-gate lane only:
nothing in this file touches the sealed boristown A/B (its prereg and calendar
are unchanged).**

## 0. External motivation (why this arm, why now)

OpenAI's ARC-AGI-3 harness result (research sweep 2026-07-31, ADAPT headline):
GPT-5.6 Sol 13.3% → 38.3% with output tokens cut to ~1/6, from exactly two
harness-side settings — (1) retain internal reasoning across turns instead of
discarding it each turn, (2) compact evicted context into summaries instead of
rolling cuts. No model change. This maps 1:1 onto our diagnosed MEMORY failure
class (`transcript_forensics.md`): the eviction window drops refuted hypotheses
which re-enter verbatim (su15, lp85) and drops action-effect facts which get
re-probed (sb26 SPACE=timer). Gold cutoff is 1.54 and climbing while our band
is static — the compaction lane is the endorsed post-A/B route (daily brief
2026-08-01 §5) regardless of A/B outcome.

Amendment absorbed at design time (research sweep 2026-08-01, "Plans Don't
Persist", arXiv:2606.22953): re-surfacing evicted plans alone does NOT recover
plan signal → this arm pairs compaction WITH retained reasoning in one flag and
measures plan-signal survival (§4 M3), not just summary re-injection.

## 1. Harness audit findings (2026-08-01, this file is their record)

Audited: `duck_eval/taaf_bundle/src/ARC3-Inference/inference/agent/tool_agent.py`.

- **A. Retained reasoning is silently dead in the stock stack.** The serve
  config arms `--default-chat-template-kwargs '{"preserve_thinking": true}'`
  (configs/inference.json §server; framework/kaggle.py serve cmd), and the
  qwen3 reasoning parser returns `reasoning_content` on responses. ToolAgent
  persists it across turns — but re-keyed as `"reasoning"`
  (`analyze()`: `assistant_message["reasoning"] = reasoning`), and
  `build_chat_payload` sends messages verbatim. Qwen3-family chat templates
  with preserve_thinking read `message.reasoning_content`; the outbound key
  `reasoning` is invisible to the template. Net: preserve_thinking is armed but
  starved — prior-turn thinking never re-enters the prompt. (The per-request
  `chat_template_kwargs: {enable_thinking: ...}` merge does NOT override
  preserve_thinking; the field name is the leak.)
- **B. Eviction is two-layer hard-drop, no summary.**
  (1) `_trim_messages_for_context` → `_drop_oldest_history_block`: token-budget
  trim (context 32768 − reply reserve 512 − safety 512 = 31744-token budget,
  chars/3 estimator) drops the oldest user/assistant/tool blocks;
  (2) `_persistent_history_messages` → `_keep_recent_history_turns`: the
  `_PERSISTENT_HISTORY_ASSISTANT_TURNS = 30` cap silently drops everything
  older between turns. Both discard content irrecoverably — exactly the
  "rolling cuts" OpenAI replaced.
- **C. Partial existing mitigations don't close the gap:** the scientist-note
  `_summarized_knowledge` block survives eviction but is model-authored,
  overwrite-only, and wiped on level/game transitions; the war-v2 ledger graft
  attacks the same class but is a different composition (paused arm). Neither
  compacts the evicted span itself.

## 2. Mechanism (built + smoked 2026-08-01)

`duck_eval/warpack/_kaggle_dataset/compaction_patch.py` (VERSION v1), grafted
by `build_eval_notebook.py --compaction` → `canivel/arc3-duck-compaction-eval`.
ONE flag (`COMPACTION=1`) gates both coupled components:

- **RETAIN:** mirror assistant `reasoning` → `reasoning_content` on every
  outbound history message (strict superset; feeds the already-armed
  preserve_thinking). Sub-kill `COMPACTION_RETAIN=0`.
- **COMPACT:** capture every message evicted at BOTH drop points and fold it
  through a mechanical digester (no LLM calls, no prose): assistant text
  (content + reasoning) → `ledger_core` HeuristicExtractor (FACT /
  refuted-HYPOTHESIS records, reused verbatim from the ledger stack as a pure
  library — this is NOT the ledger graft); tool payloads → per-action-name
  counts with board-changed tallies, level-ups, GAME_OVERs, action span. A
  single marker-prefixed digest message (≤ reserved 1000 tokens, reserve
  subtracted from the trim budget so overflow is impossible) is injected after
  the system prompt on every request, replacing the evicted span. Stale digests
  are stripped and re-rendered fresh (never compound, never self-ingested).

Composition: duck baseline + (f) continuation default (rides per 07-23
amendment) + COMPACTION=1. NO warpack, NO ledger graft, NO sentinel.
Game-agnostic (graft rule). Failure policy: vanilla fallback (never 0);
kill switch `COMPACTION_DISABLE=1`; no locks anywhere (deadlock lesson).

## 3. Canary (mechanism-engagement gate, counted from the build log)

- `compaction v1: ACTIVE ...` banner present (dataset-version proof per
  feedback_kaggle_dataset_code_sync) — absent/`PATCH FAILED` → run is VANILLA,
  VOID for this arm.
- **≥1 `COMPACTION ` event line per run** (greppable stdout; per-game
  `*_compaction_events.jsonl` sidecars). Expectation from phase1 forensics:
  the L2 grinder games each blow the 31744-token budget many times per run.
- RETAIN canary: `retained_reasoning_msgs>0` in event lines.

## 4. Metrics (offline eval rail only; gate vs `runs/null10` per house rule)

- **M1 (primary): paired Δlc** vs ledger-OFF `arc3-duck-war-eval` seed N
  (identical seed convention: seed N = push N of the identical notebook).
  Non-harm screen exactly as the gate-eval/sentinel precedent (paired Δlc,
  worst-game cap −1.0, admission precedent −0.128).
- **M2 (token efficiency, the OpenAI 6× claim):** generated tokens per
  executed action and per levels_completed vs the paired seed (ToolAgent
  usage accumulators / benchmark artifacts; requests.jsonl fallback).
- **M3 (plan-signal survival, Plans-Don't-Persist amendment):** refuted-
  hypothesis re-proposal rate in transcripts (same forensics procedure as
  `transcript_forensics.md`) vs the paired seed. Compaction must REDUCE
  re-proposal of refuted goals; if it doesn't, the arm is summary-theater
  regardless of Δlc noise.

## 5. Kill rules (sealed now, before any build)

- **K1:** canary = 0 COMPACTION events in an eval run → mechanism never
  engaged → no further pushes of this arm until root-caused; the run is VOID
  as evidence either way.
- **K2:** vanilla-fallback banner (`PATCH FAILED`) → run VOID (counts against
  neither side).
- **K3:** non-harm screen FAIL on the sentinel-precedent thresholds at seed 1
  → arm PAUSED; a second independent FAIL at seed 2 → arm DEAD (3-strike rule
  not needed; this is the sentinel A5/A8 standard).
- **K4:** M3 shows NO reduction in refuted re-proposal rate across both eval
  seeds → kill the compaction component (retain-only rerun allowed once as a
  sub-arm via COMPACTION_RETAIN semantics inverted, single flag discipline).
- **K5:** any future scored draw (NOT requested here; separate gate + R23+)
  inherits A21/C2 verbatim: harm-pause on draw <0.80, no inference from n=1.

## 6. Budget + process constraints (binding)

- Free Kaggle build rail ONLY (feedback_arc_zero_budget); NEVER submitted —
  submission queue stays pinned per its own rules. No cloud spend.
- 2 kernel pushes/day: 08-01 slots were 2/2 used (gate-eval seed-2 v3 + arm B
  canary v1) → arc-war-kit dataset version push + compaction-eval kernel push
  are STAGED for 2026-08-02 slot-1, AFTER the R23 look, and must not displace
  any push the sealed A/B needs (A/B has absolute priority on slots).
- Dataset push discipline: `kaggle datasets version` on `canivel/arc-war-kit`
  BEFORE the kernel push + runtime banner check (feedback_kaggle_dataset_code_sync);
  byte-audit the staged dataset dir pre-push.
- Runtime-tested pre-push: `duck_eval/warpack/compaction_smoke.py` 41/41 PASS
  2026-08-01 (flag gate, eviction→digest, budget discipline, event+sidecar,
  digest hygiene, 30-turn-cap capture, reasoning mirroring, builder output).
