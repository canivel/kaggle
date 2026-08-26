# PREREG — GRAFT SCORE-CONFIRMATION ARM (`arc3-graft-floor-eval` v5), sealed 2026-08-19 evening BEFORE the push
**Slot:** 2026-08-20 slot 1 (coordinator ruling 08-19: "the SCORE-primary rerun takes 08-20 slot 1, over the pair arm").
**Lane:** graft-floor, REOPENED for exactly one confirmation question; lock re-acquired in `runs/lane_locks.json`.
**Binding context:** `graft_floor_prereg_2026-08-17.md` (the LEVELS seal, executed 08-19, verdict NULL decisive, exp_id 19) + `conversion_trace_2026-08-17.md` §3.2/§9.

---

## 0. WHAT THIS ARM IS, STATED WITHOUT PRETENSE

**This is a post-hoc-motivated confirmation arm. The motivating result has been SEEN.**
Seed 1 (v4, 08-19, exp_id 19): lc_total 18, mean_score **2.303**, total_actions **3257** — vs the lc-MATCHED baseline (gate_eval_v1, also lc 18): score 1.427, actions 4757. Descriptively: **+0.876 score at identical capability from −32% actions** — the shortcircuit quadratic-denominator signature (`min(115,(baseline/actions)²·100)`, vendored taaf/game.py:403) in exactly the direction the 08-17 conversion trace predicted. That result was sealed NON-INFERENTIAL under the levels prereg and **carries no verdict**; this arm exists to test whether it REPLICATES under a score-primary seal.

- **The CONFIRM bar below is set such that seed 1 would have passed it.** That is what a replication bar means; it is stated so no reader mistakes this for a blind test.
- **Treatment is byte-identical to v4** (artifact sha `b0316275f53c6c85`, v19 flags, banking/transfer OFF and asserted-absent, incumbent Qwen3.6). This is a **second seed under a corrected instrument** — the campaign's instrument lesson applied: we killed our own July banking/shortcircuit line by measuring LEVELS when the metric pays SCORE.
- **Same-seed honesty:** the harness banner says seed=1 and the rerun is nominally same-seed. The 08-18 replicate pair (q38 engine v2/v3, same config, same seed) measured 21 vs 17 levels — same-seed reruns on this rail are effectively independent draws (vLLM nondeterminism + timing). A rerun is therefore a legitimate second draw, and this is declared rather than assumed.

## 1. THE ONE QUESTION

Does the graft floor's score advantage at matched capability **replicate**? Mechanism: `shortcircuit` trims two-strike-confirmed no-op actions; every no-op otherwise increments the scored action denominator, squared. Levels are NOT expected to move (the 08-19 NULL is the standing levels result); levels serve only as the matching variable and the harm guard.

## 2. SEALED VERDICTS — evaluated in this order (three-state minimum honored)

Baselines (family `duck-harness-kaggle`, m=3): lc 18 → score 1.427 · lc 19 → 1.939 · lc 21 → 3.420.
**lc-matched bar** = score of the baseline with minimal |lc distance| to the arm's lc_total (**ties resolved to the HIGHER-score baseline** — conservative) **+ 0.5**:

| arm lc_total | matched baseline | CONFIRM bar (mean_score ≥) |
|---|---|---|
| 13–18 | 1.427 (lc 18) | **1.927** |
| 19 | 1.939 (lc 19) | **2.439** |
| 20 | tie → 3.420 (lc 21) | **3.920** |
| ≥ 21 | 3.420 (lc 21) | **3.920** |

| verdict | rule |
|---|---|
| **INFRA DEATH** (not decisive) | any §3 assertion fails · no `benchmark.json` · n_games ≠ 25 |
| **HARM** (decisive) | lc_total ≤ **12** (the standing K3″ levels guard, mean Δlc ≤ −0.286315 — the graft must not LOSE capability) |
| **CONFIRM** | lc_total ≥ 13 AND mean_score ≥ the lc-matched bar above |
| **NULL** | lc_total ≥ 13 AND mean_score below the bar |

- **Statistical honesty:** score dispersion at fixed lc is unknown (n=1 per lc in the baseline family; overall sd 1.033 on n=3). Type-I for CONFIRM is therefore a judgment, not a measurement — estimated ~15–25% (a null draw behaving like a baseline must beat its own lc-match by +0.5). **This arm is a replication SCREEN, not a K3″-grade decisive test**, and a CONFIRM is licensed to justify an A21 exploration DRAW (which risks one queue slot), not a promotion claim.
- **Pre-registered expectation:** CONFIRM at lc 15–20 with mean_score 1.9–2.7 and total_actions ≤ 3800. P(CONFIRM) ≈ 55–65% if the mechanism is real; the descriptive prior says it is, which is exactly why the bar must be pre-committed now.
- **Descriptive secondaries (no verdict):** total_actions vs 3257/4757/4033; actions-per-level; games_won (predict 0 — the banking reachability fact, n=495).

## 3. MANDATORY GATE ASSERTIONS — unchanged from the levels seal
Assertions 1–7 of `graft_floor_prereg_2026-08-17.md` §5 apply verbatim (banner + API_VERSION=1; [goalkeep]/[hudmask] armed; efficiency/retry_guard/shortcircuit in FEATURES; banking/transfer ABSENT; no stock fallback; bundle re-diffs to `df447f61…`/`7705481…` on a fresh download at push; served engine `vrfai/Qwen3.6-27B-FP8`). Any failure ⇒ INFRA DEATH, never HARM/NULL/CONFIRM.
**Scorer:** `duck_eval/graft/graft_confirm_score.py`, sealed and selftested BEFORE the push, decoding CLI-2.2.3 JSON logs natively (the 08-19 instrument fix is inherited, with its regression fixtures).

## 4. ON A CONFIRM — pre-committed consequence
Per coordinator ruling: a CONFIRM certified by ~noon 08-20 puts this kernel at the queue head for the 08-20 scored window as an **A21 exploration draw** (message cites: A21 exploration budget; entry = certified CONFIRM under this prereg; the 10-team external evidence from the conversion trace; this file). NULL or HARM or INFRA DEATH ⇒ filler fires, no swap, and the pair arm (its prereg unwritten) is next in the queue for coordinator consideration.

## 5. WHAT THIS ARM CANNOT SETTLE
- Whether the score effect transfers from the offline eval rail to the scored competition rerun (different environment — the 08-19 mount-layout divergence proves the two rails differ in at least one way we did not choose).
- Which flag carries the effect (floor-adoption, not factorial; shortcircuit is the predicted carrier, goalkeep/hudmask ride along).
- Anything about banking/transfer (still unreachable, still untested in either direction).
- The capture-side memory defect (exp 17) — orthogonal, queued behind this.

## 6. ARTIFACTS
Same notebook `notebooks/graft-floor-eval/` (byte-identical, sha `b0316275f53c6c85`) — the push is a DELIBERATE identical-code re-run under `GRAFT_ALLOW_DUPLICATE=1` + `GRAFT_ALLOW_V2=1`, recorded here in advance. Push script `duck_eval/graft/graft_push.sh` re-dated to 2026-08-20 under the coordinator's authorization. Results → `runs/kernel_pulls/graft_confirm_v1/`. Scorer selftest + fresh ledger read at push time are mandatory pre-push gates.

---

## DATED ADDENDUM — 2026-08-20 pre-push (original sealed text above left readable)

**The share-fork bundle was republished between seal and push** (v21 wave): 89→90 files. Full per-file re-audit against the 08-17 audited copy: **ADDED `clickmap.py`; MODIFIED `composite.py` (+15 lines, every one gated on `flags.get("clickmap")`) and `goalkeep.py` (+52 lines that no-op when clickmap is unarmed — the author's own invariant, verbatim: "Absent -> the digest is byte-identical to goalkeep+hudmask"); 0 other files touched; 0 stock files touched.** For this arm's flag set (clickmap NOT armed) the delta is inert by construction. Actions taken per the bundle-check's own protocol: audited shas updated TOGETHER in `graft_bundle_check.py` (bundle `7a61fafc99545550a2bd914fd2b94ee2`, grafts `1c991e0ba557fee663aef614822fe01c`, 90/17 files); **`clickmap` added to the scorer's FLAGS_FORBIDDEN and `[clickmap] armed` to ARMED_LINES_FORBIDDEN** (wrong-arm markers, negative-controlled — both fire); artifact bytes UNTOUCHED (sha `b0316275f53c6c85` — the second-seed identity is the point; its cell-2 banner still prints the 08-17 sha `df447f61…` as `audited_bundle_sha`, now a documented-stale label, superseded by this addendum). **Residual honesty: seed 1 ran the 89-file bundle, seed 2 runs the 90-file bundle — "byte-identical treatment" is now "byte-identical artifact + functionally-inert bundle delta, audited line-by-line." If the verdict lands within noise of a boundary, this caveat must be cited.**
