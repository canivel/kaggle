# Sentinel Q2 discharge report (2026-07-22)

R16 Q2 ruling: SENTINEL_BUDGET=150 APPROVED-WITH-CONDITIONS; push CLEARED once
conditions 1–3 are discharged (all $0/local); condition 4 seals before
ship/binding look. This report discharges conditions 1–3. **Nothing was pushed**
— push execution belongs to the orchestrator.

Provenance note: the 2026-07-21 session (wedged before reporting) had already
produced the attempt-unit analysis, implemented the v2 re-key, updated the smoke
suite, and re-run baseline canaries (file mtimes 07-21 08:35–08:40). Today's
session verified all of it, added the condition-2 defect-sensitive counters to
the canary, produced `runs/sentinel_canary_v3_b150.json`, wrote the condition-3
context-tax sentence into the build doc, and wrote this report.

## Condition 1 — attempt-unit analysis: **PASS (approximation FAILS → re-key REQUIRED and DONE)**

Tool: `duck_eval/sentinel/attempt_unit_analysis.py` (drives the REAL
`_GameBudget` over the certified-seed traces `runs/kernel_pulls/war_eval_v{1,2,3}`).
Artifact: `runs/sentinel_attempt_unit_b150.json` (07-21 08:35).

llm-agents' deciding statistic (per-attempt unit, B=150):
- (a) budget-attributable GAME_OVERs: 6 total; **1/6 (16.7%) in games with >1
  attempt**; (b) **median actions already consumed when the fatal attempt
  begins: 0** (all: 0,0,0,0,0,151). Taken alone, the death statistic looks
  benign — but that is exactly the blind spot the reviewers predicted: under
  per-attempt keying, multi-attempt grinders rarely register a per-attempt
  "budget death" at all, because no single attempt reaches 150.
- The decisive evidence is the envelope view: **15 of 33 envelope-crossing
  (game,seed) units (total actions ≥ 150) received NO v1 warning by 0.9×B
  cumulative** (13 of them never fired at all), and there are **13
  cross-attempt-waste episodes** (game total > 150, no single attempt > 75) —
  structurally invisible to v1.
- Carrier games (the (a) target population): **ka59/re86/tu93 are multi-attempt
  in EVERY certified seed** — attempt counts ka59 [2,2,3], re86 [3,4,4], tu93
  [7,9,12]. tu93 appears twice in the cross-attempt-waste list (v1: 154 acts /
  max attempt 50 / never fired; v3: 314 acts / max attempt 51 / never fired);
  re86 fired only late (v1: first fire at 138 of 296; v2: 154 of 163).

**Decision: the single-attempt approximation does NOT hold on the carrier
games. Re-key to cumulative remaining-envelope mandated — and implemented.**

### Re-key diff summary (v1 → v2, `duck_eval/sentinel/budget_sentinel_patch.py`)

- `VERSION` "v1" → "v2"; event line `SENTINEL v=2 ...`; banner gains
  `unit=game-envelope`; `bm.label` suffix `-sentinel-v2`. Event/banner FORMAT
  preserved (same anchor, same key=value tokens, same sidecar schema).
- `_GameBudget.reset_attempt()` no longer re-arms thresholds or restarts the
  budget clock — it only advances the attempt ordinal (event metadata). Each
  threshold fires at most ONCE per GAME (hard cap 3 events/game).
- Consumed fraction = **CUMULATIVE game actions / budget** (was
  actions-since-attempt-start / budget). Attempt boundaries still detected
  (level-up, GAME_OVER restart) but only to label `attempt=` in events.
- FACT wording updated to game-envelope semantics ("of the game's action
  budget", "actions remain for ALL remaining levels", "gone for the whole game,
  not just this attempt"). Presence/timing is the value, not phrasing
  (feedback_prompt_is_noise) — no A/B.
- Dataset staging copy `duck_eval/warpack/_kaggle_dataset/budget_sentinel_patch.py`
  verified **byte-identical** to the runtime copy (cmp, today).

### Smoke re-run: **30/30 PASS** (verified today)

`uv run python duck_eval/sentinel/sentinel_smoke.py` → `RESULT: 30 passed, 0
failed` (incl. kill-switch subprocess arm 3/3). The v1 build doc's 29/29 suite
was updated for v2 semantics: M4 now asserts a fresh level attempt does NOT
re-arm (v1 would have fired 3 more), M4b asserts crossings use CUMULATIVE
actions across an attempt boundary, M4c asserts post-boundary events carry the
new attempt ordinal as metadata. Structural checks (S1–S6), banner/label checks
(now `sentinel v2:` / `-sentinel-v2`), FACT one-shot injection, and kill-switch
all pass against the REAL bundled harness classes.

## Condition 2 — defect-sensitive B=150 canary re-run: **PASS**

Tool: `duck_eval/sentinel/compressed_canary.py`, extended today with the R16
counters (cross-attempt-waste episodes + warned-by-0.9B check + multi-attempt-
game presence; exit code now fails on any of them).
Artifact: **`runs/sentinel_canary_v3_b150.json`** (written today).

- A10 canary (≥5 games fire per run): **PASS** — 20/25 games fire on each of
  war_eval_v1/v2/v3 (pooled 60/75 (game,seed) units).
- R15 O5 deterministic predicate ("sentinel fired strictly before every budget
  death"): **PASS** — 54 budget-attributable GAME_OVERs, 0 violations.
- **Cross-attempt-waste counter (the check the R16 reviewers said the bare
  canary could not see): 13 episodes (game total > 150, no attempt > 75);
  13/13 fired; 13/13 warned by 0.9×B=135** — every one first fires at
  cumulative action 75 (the 50% threshold), i.e. the v2 game-envelope keying
  warns mid-envelope in exactly the population v1 was blind to (incl. targets
  sc25 ×2, tu93 ×2, su15 ×1).
- Multi-attempt game presence: **PASS** — 59/75 units are multi-attempt (≥1
  required); every carrier game (ka59/re86/tu93) is multi-attempt in all 3
  seeds.
- Consistency: per-run firing/death/violation numbers are identical to
  yesterday's `runs/sentinel_canary_v2_b150.json` (same v2 logic; v3 adds the
  condition-2 counters only).

## Condition 3 — W1 prong context-tax sentence: **PASS**

Appended to `learnings/war_room/sentinel_build_2026-07-19.md` (ADDENDUM
2026-07-22, "W1 prong — context-tax sentence"). The sentence, verbatim:

> Tokens-per-fire is ≤ ~95 tokens (worst-case FACT measured at 370 chars / 62
> words; ≤124 tokens even at a conservative 3 chars/token bound), and because
> v2 fires each threshold at most once per GAME the worst-case cumulative
> injection is 3 fires × ~95 ≈ 285 tokens (≤372 conservative) ≈ 0.45% (≤0.59%)
> of the 63k per-game context envelope, even if all three FACTs persist in
> history for the rest of the game; the v1 scenario systems #3 asked about — 3
> fires/attempt across a multi-attempt game, worst observed 14 attempts (sp80
> v1) → up to 42 fires ≈ 3.9–5.2k tokens ≈ 6.2–8.3% of envelope — is
> structurally impossible in v2 (no re-arm).

Measurement basis: worst-case FACT rendered from the live `_build_fact`
(game=tu93_p1, 135/150, 90%, 15 remaining) = 370 chars / 62 words; no local
tokenizer installed, so the count is bounded (chars/4 typical ≈ 93, chars/3
conservative ≈ 124) rather than tokenizer-exact. The bound is what the sentence
carries.

Note: `grinder_design_R16_republication.md` §6.3 (the circulated W1 prong text)
was NOT edited — it is a circulated panel document; the R17 sealing text should
cite the build-doc addendum. Flagged for the orchestrator.

## Condition 4 — for the record (NOT blocking today's push)

Scored-regime envelope check (methodology R6, systems #18): before ship/binding
look, the tokens/game grep on a scored-run pull must seal within ±15% of the
63k envelope, else B is re-derived by the frozen formula; token-implied capacity
recomputed from W1/W2 ON-seed transcripts before the A14 binding look
(component (b) changes tokens/action by design). Not discharged here; it is a
sealed pre-ship check, not a pre-push condition.

## Verdict

| condition | status | evidence |
|---|---|---|
| 1. attempt-unit analysis + re-key decision | **PASS — re-key REQUIRED, implemented (v2), smoke 30/30** | `runs/sentinel_attempt_unit_b150.json`; `budget_sentinel_patch.py` VERSION=v2; smoke re-run today |
| 2. defect-sensitive B=150 canary | **PASS — 13/13 waste episodes warned by 0.9B; ≥1 multi-attempt game (59/75)** | `runs/sentinel_canary_v3_b150.json` |
| 3. W1 context-tax sentence | **PASS — appended to build doc** | `sentinel_build_2026-07-19.md` addendum |
| 4. scored-envelope ±15% | noted, pre-ship (not pre-push) | — |

**Push is GO from the build side** (conditions 1–3 discharged; re-key landed
before the push exactly as the R16 ruling requires). Orchestrator reminders:
cell 2 of the W1 eval notebook must export `SENTINEL_BUDGET=150` (open risk 1 —
uncapped regime makes the sentinel inert); dataset copy is byte-identical but
needs its own `datasets version` push + runtime banner check
(feedback_kaggle_dataset_code_sync) — grep the build log for
`sentinel v2: budget sentinel ACTIVE (unit=game-envelope; ...)` and
`SENTINEL v=2` events.
